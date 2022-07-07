# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 13:47:56 2020

@author: genie
"""

import os
import pathlib
from typing import Any, Dict, List, NamedTuple, Union

import numpy as np
import pandas as pd
import pydantic

import normits_demand as nd
from normits_demand import cost
from normits_demand.cost import utils as cost_utils
from normits_demand.concurrency import multiprocessing as mp
from normits_demand.utils import file_ops, math_utils
from normits_demand.utils import utils as nup
from normits_demand import logging as nd_log
from normits_demand import constants

LOG = nd_log.get_logger(__name__)
_TINY_INFILL = 1 * 10 ** -8


class DisaggregationSettings(pydantic.BaseModel):  # pylint: disable=no-member
    """Optional settings for adjusting `disaggregate_segments`."""

    aggregate_surplus_segments: bool = True
    export_original: bool = True
    export_furness: bool = False
    time_period: str = "24hr"
    intrazonal_cost_infill: float = 0.5
    maximum_furness_loops: int = 1999
    pa_furness_convergence: float = 0.1
    bandshare_convergence: float = 0.975
    max_bandshare_loops: int = 200
    multiprocessing_threads: int = -1


class DisaggregationSummaryResults(NamedTuple):
    """High level results from the `_segment_build_worker` function."""

    initial_bs_convergence: float
    final_bs_convergence: float
    target_productions: float
    target_attractions: float
    estimated_productions: float
    estimated_attractions: float
    initial_pa_difference: float
    final_pa_difference: float
    convergence_fail: bool


class DisaggregationResults(NamedTuple):
    """All results returned by the `_segment_build_worker` function."""

    furness_matrix: np.ndarray
    final_matrix: np.ndarray
    segmentation: dict[str, Any]
    cost_tld: cost.CostDistribution
    final_matrix_tld: cost.CostDistribution
    target_tld: cost.CostDistribution
    summary: DisaggregationSummaryResults


def read_pa_seg(pa_df, exc=("trips", "attractions", "productions"), in_exc=("zone",)):
    """ """
    cols = list(pa_df)
    cols = [x for x in cols if x not in exc]

    for ie in in_exc:
        drops = []
        for col in cols:
            if ie in col:
                drops.append(col)
    for d in drops:
        cols.remove(d)

    seg = pa_df.reindex(cols, axis=1).drop_duplicates().reset_index(drop=True)

    return seg


def _build_enhanced_pa(tld_mat, df, df_value, col_limit, ia_name=None, unq_zone_list=None):
    """
    Private
    """
    if ia_name is None:
        ia_name = list(df)[0]
    if unq_zone_list is None:
        unq_zone_list = nup.get_zone_range(tld_mat[ia_name])

    out_list = []
    for _, itd in tld_mat.iterrows():
        te = df.copy()
        calib_params = {c: itd[c] for c in col_limit if not _isnull(itd[c])}

        te, total = nup.filter_pa_vector(
            te, ia_name, calib_params, round_val=3, value_var=df_value,
        )
        te = nup.df_to_np(
            te, v_heading=ia_name, values=df_value, unq_internal_zones=unq_zone_list
        )

        if total == 0:
            raise Warning("Filter returned 0 trips, failing")
        calib_params.update({"trips": te})
        out_list.append(calib_params)

    return out_list


def _control_a_to_seed_matrix(sd, attr_list):
    """
    Function to get target attraction vectors by splitting seed matrix
    cellwise on attractions

    Parameters
    ----------
    sd:
        Seed matrix, numpy format - as imported from TMS & compiled
    attr_list:
        List of attractions for new segments.
    Returns:
    ----------
    new_attr:
        Updated vector
    """
    cell_sum = []
    for a in attr_list:
        cell_sum.append(a["trips"])
    cell_sum = np.where(sum(cell_sum) == 0, 0.000001, sum(cell_sum))

    cell_splits = []
    for a in attr_list:
        cell_splits.append(a["trips"] / cell_sum)

    mat_attr = sd.sum(axis=0)

    new_attr = attr_list.copy()

    for ait, dat in enumerate(new_attr, 0):
        new_attr[ait]["trips"] = mat_attr * cell_splits[ait]

    audit_sum = []
    for a in new_attr:
        audit_sum.append(a["trips"])
    audit_sum = sum(audit_sum)
    audit = audit_sum.round() == mat_attr.round()

    return new_attr, audit


def _is_nan(value: Any) -> bool:
    if not isinstance(value, (int, float)):
        return False
    return np.isnan(value)


def _control_a_to_enhanced_p(prod_list, attr_list):
    """
    Function to control a best estimate list of attraction vectors to a similar
    enhanced list of production vectors, as production vectors are more reliable.
    Looks up a fitting attraction vector for the productions, adds segments
    from productions to attractions, balances attractions to productions.

    Takes
    ------
    prod_list:
        List like of a dictionary containing production vectors (calib params
        plus zonal prods)
    attr_list:
        As above, but attractions. Should have less detailed segmentation.
    Returns
    -----
    new_attr:
        Further segmented, balanced attractions
    audit:
        Boolean - did it work
    """
    # Define empty list for output
    new_attr = []
    for prod in prod_list:
        # Build empty dict for calib params only
        p_types = {}
        # Get sum of productions
        p_tot = prod["trips"].sum()

        # Iterate over prod items and build calib params dict
        for p_item, p_dat in prod.items():
            if p_item != "trips":
                p_types.update({p_item: p_dat})

        # Iterate over attr items
        for attr in attr_list:
            work_dict = attr.copy()
            # Find one that matches
            for item, dat in work_dict.items():
                record_match = True
                # If they don't match kick the loop
                if item in p_types.keys():
                    # Continue if both values are NaN
                    if _is_nan(dat) and _is_nan(p_types[item]):
                        continue
                    if dat != p_types[item]:
                        record_match = False
                        break

            # If the record matches, build out calib params and balance
            if record_match:
                # Update dict with anything not there
                for p_item, p_dat in p_types.items():
                    if p_item not in work_dict.keys():
                        work_dict.update({p_item: p_dat})
                # Balance - make copy of original
                new_frame = work_dict["trips"].copy()
                # Get demand as factor
                new_frame = new_frame / sum(new_frame)
                # Multiply factor by production demand
                new_frame = new_frame * p_tot
                # Put new df in dict
                work_dict.update({"trips": new_frame})
                new_attr.append(work_dict)
                # Don't do it again (will square)
                break

    audit = []
    # Audit balance
    for ri in list(range(len(prod_list))):

        a = prod_list[ri]["trips"].sum().round(0)
        b = new_attr[ri]["trips"].sum().round(0)

        if a == b:
            audit.append(True)
        else:
            audit.append(False)

    # Lists should have popped in order and so x_prod, should equal x_attr

    return (new_attr, audit)


def _set_dataframe_dtypes(df: pd.DataFrame, dtypes: Dict[str, str]) -> pd.DataFrame:
    """Set DataFrame columns to specific types if the column exists."""
    for col, dt in dtypes.items():
        if col in df.columns:
            df.loc[:, col] = df[col].astype(dt)
    return df


def disaggregate_segments(
    import_folder: pathlib.Path,
    target_tld_folder: pathlib.Path,
    tld_units: nd.CostUnits,
    model: nd.AssignmentModel,
    base_productions: nd.DVector,
    base_attractions: nd.DVector,
    export_folder: pathlib.Path,
    cost_folder: pathlib.Path,
    trip_origin: nd.TripOrigin = nd.TripOrigin.HB,
    settings: DisaggregationSettings = DisaggregationSettings(),
) -> None: # TODO Update docstring
    """
    Parameters
    ----------
    trip_origin:
        from 'hb', 'nhb', 'both'

    aggregate_surplus_segments = True:
        If there are segments on the left hand side that aren't in the
        enhanced segmentation, aggregated them. Will
    """
    # Find all matrices and extract segmentation info
    LOG.info("Finding base matrices in %s", import_folder)
    required_columns = ["matrix_type", "trip_origin", "uc", "yr", "m", "ca"]
    seg_dtypes = dict.fromkeys(("p", "yr", "m", "ca", "soc", "ns"), "Int64")
    base_mat_seg = pd.DataFrame(
        nup.parse_matrix_folder(
            import_folder,
            extension_filter=constants.VALID_MAT_FTYPES,
            required_data=required_columns,
        )
    )
    base_mat_seg = _set_dataframe_dtypes(base_mat_seg, seg_dtypes)
    base_mat_seg.loc[:, "base_seg"] = base_mat_seg["path"].apply(lambda p: p.name)
    base_mat_seg = base_mat_seg.loc[
        base_mat_seg["matrix_type"] == "pa", ["base_seg", *required_columns]
    ].drop(columns="matrix_type")
    duplicates = base_mat_seg.duplicated().sum()
    if duplicates > 0:
        raise ValueError(f"{duplicates} matrices with the same segmentation found")
    base_mat_seg.to_csv("DO_NOT_COMMIT/base_mat_seg.csv", index=False)

    # Find all TLDs and extract segmentation info
    LOG.info("Finding TLDs in %s", target_tld_folder)
    required_columns = ["trip_origin", "m", "uc", "ca"]
    tld_seg = pd.DataFrame(
        nup.parse_matrix_folder(
            target_tld_folder,
            extension_filter=constants.VALID_MAT_FTYPES,
            required_data=required_columns,
        )
    )
    tld_seg = _set_dataframe_dtypes(tld_seg, seg_dtypes)
    tld_seg.loc[:, "enh_tld"] = tld_seg["path"].apply(lambda p: p.name)
    tld_seg = tld_seg.loc[:, ["enh_tld", *required_columns, "soc", "ns"]]
    duplicates = tld_seg.duplicated().sum()
    if duplicates > 0:
        raise ValueError(f"{duplicates} TLDs with the same segmentation found")

    unique_zones = base_productions.zoning_system.unique_zones
    ia_name = base_productions.zone_col
    base_productions = base_productions.to_df()
    base_attractions = base_attractions.to_df()

    # Get trip end segmentations
    exclude_columns = ["trips", "attractions", "productions", "val"]
    productions_seg = read_pa_seg(base_productions, exc=exclude_columns, in_exc=["zone"])
    attractions_seg = read_pa_seg(base_attractions, exc=exclude_columns, in_exc=["zone"])

    ## Build iteration parameters

    # Get cols in both the base and the tld
    merge_cols = [x for x in list(tld_seg) if x in list(base_mat_seg)]

    # Get cols in base that will have to be aggregated
    agg_cols = [x for x in list(base_mat_seg) if x not in list(tld_seg)]
    agg_cols.remove("base_seg")
    if model == nd.AssignmentModel.NORMS and "ca" in agg_cols:
        agg_cols.remove("ca")

    # Get cols in tld that form the added segments
    add_cols = [x for x in list(tld_seg) if x not in list(base_mat_seg)]
    add_cols.remove("enh_tld")

    # Define costs to lookup for later
    # If time period is involved in the merge, it'll be tp

    # Build a set of every tld split and every aggregation
    it_set = base_mat_seg.merge(tld_seg, how="inner", on=merge_cols)

    # Drop surplus segments on import descriptors - if you want
    if settings.aggregate_surplus_segments:
        it_set = it_set.drop(agg_cols, axis=1)
        base_mat_seg = base_mat_seg.drop(agg_cols, axis=1)

    # Get cols in productions that are also in it set
    p_cols = [x for x in list(it_set) if x in list(productions_seg)]
    # get cols in attractions that are also in it set
    a_cols = [x for x in list(it_set) if x in list(attractions_seg)]

    # Apply any hard constraints here eg, origin, mode
    if trip_origin != "both":
        it_set = it_set[it_set["trip_origin"] == trip_origin.value].reset_index(drop=True)

    # Build a set of every unique combo of base / target split - main iterator
    unq_splits = (
        it_set.copy().reindex(merge_cols, axis=1).drop_duplicates().reset_index(drop=True)
    )

    # Main disaggregation loop

    unchanging_kwargs = {
        "unq_splits": unq_splits,
        "it_set": it_set,
        "base_mat_seg": base_mat_seg,
        "import_folder": import_folder,
        "add_cols": add_cols,
        "target_tld_folder": target_tld_folder,
        "tld_units": tld_units,
        "base_p": base_productions,
        "p_cols": p_cols,
        "base_a": base_attractions,
        "a_cols": a_cols,
        "unq_zone_list": unique_zones,
        "cost_folder": cost_folder,
        "ia_name": ia_name,
        "export_folder": export_folder,
        "trip_origin": trip_origin,
        "settings": settings,
    }

    kwargs_list = list()
    for i in unq_splits.index:
        kwargs = unchanging_kwargs.copy()
        kwargs["agg_split_index"] = i
        kwargs_list.append(kwargs)

    # Call using multiple threads
    LOG.debug("Running segment disaggregator on %s threads", settings.multiprocessing_threads)
    mp.multiprocess(
        _segment_build_worker,
        kwargs=kwargs_list,
        process_count=settings.multiprocessing_threads,
    )


def _isnull(value: Any) -> bool:
    """Checks if value is NaN, None or pandas NA."""
    if value is None or value is pd.NA:
        return True
    if isinstance(value, (float, int)):
        return np.isnan(value)
    return False


def _segment_build_worker(
    agg_split_index: int,
    unq_splits: pd.Series,
    it_set: Union[pd.Series, pd.DataFrame],
    base_mat_seg: pd.DataFrame,
    import_folder: pathlib.Path,
    add_cols: List,
    target_tld_folder: pathlib.Path,
    tld_units: nd.CostUnits,
    base_p: nd.DVector,
    p_cols: List,
    base_a: nd.DVector,
    a_cols: List,
    unq_zone_list: np.ndarray,
    cost_folder: pathlib.Path,
    ia_name: str,
    export_folder: pathlib.Path,
    trip_origin: nd.TripOrigin,
    settings: DisaggregationSettings,
):
    """
    Worker for running segment disaggregator in parallel.
    """

    ie_params = it_set.copy()
    import_params = base_mat_seg.copy()

    # Get required distributions for import
    agg_split = unq_splits.loc[agg_split_index]

    for name, desc in agg_split.iteritems():
        ie_params = ie_params[ie_params[name] == desc]
        import_params = import_params[import_params[name] == desc]
    ie_params = ie_params.reset_index(drop=True)
    import_params = import_params.reset_index(drop=True)

    # Don't include any null values in calib params
    calib_params = {k: v for k, v in agg_split.iteritems() if not _isnull(v)}

    # Mat list
    mat_list = list()

    # Import and aggregate
    for _, ipr in import_params.iterrows():
        print("Importing " + ipr["base_seg"])
        sd = file_ops.read_df(import_folder / ipr["base_seg"])

        if "zone" in list(sd)[0] or "Unnamed" in list(sd)[0]:
            sd = sd.drop(list(sd)[0], axis=1)
        sd = sd.values
        mat_list.append(sd)

    # Compile matrices in segment into single matrix
    sd: pd.DataFrame = sum(mat_list)  # I can't believe this works

    # Get required tld for splitting
    tld_mat = (
        ie_params.copy().drop("base_seg", axis=1).drop_duplicates().reset_index(drop=True)
    )

    tld_list: List[cost.CostDistribution] = []
    for _, itd in tld_mat.iterrows():
        tld_dict = {}
        for col in add_cols:
            if not _isnull(itd[col]):
                tld_dict.update({col: itd[col]})
        tld_dict.update(
            {"enh_tld": pd.read_csv(os.path.join(target_tld_folder, itd["enh_tld"]))}
        )
        path = target_tld_folder / itd["enh_tld"]
        tld_list.append(cost.CostDistribution.from_csv(path, tld_units))

    # Get best possible production subset - control to matrix total
    prod_list = _build_enhanced_pa(tld_mat, base_p, "val", p_cols, unq_zone_list=unq_zone_list)

    # Get best possible attraction subset
    attr_list = _build_enhanced_pa(tld_mat, base_a, "val", a_cols, unq_zone_list=unq_zone_list)

    # control to sum of target share of attraction vector, cell-wise.
    attr_list, control_aud = _control_a_to_seed_matrix(sd, attr_list)
    # Control a to p, exactly this time
    attr_list, bal_aud = _control_a_to_enhanced_p(prod_list, attr_list)

    # TODO: Make this a parameter for func
    match_tol = 0.95

    # Check audit vectors
    if sum(control_aud) < len(control_aud) * match_tol:
        # TODO: Error format
        raise Warning("PA Vectors not balanced")
    if sum(bal_aud) < len(bal_aud) * match_tol:
        # TODO: Error format
        raise Warning("PA Vectors not balanced")

    # Get distance/cost
    # Costs should be same for each segment, so get here
    cost_cp = calib_params.copy()
    if "ca" in calib_params and settings.time_period == "tp":
        cost_cp.pop("ca")

    print(cost_cp)

    costs = get_costs(cost_folder, cost_cp, unq_zone_list)

    # Pass to dissagg function
    results = _dissag_seg(
        prod_list,
        attr_list,
        tld_list,
        sd,
        costs,
        furness_loops=settings.maximum_furness_loops,
        min_pa_diff=settings.pa_furness_convergence,
        bs_con_crit=settings.bandshare_convergence,
        max_bs_loops=settings.max_bandshare_loops,
    )

    reports_folder = export_folder / "Reports"
    reports_folder.mkdir(exist_ok=True)

    # Read, convert, build path and write out
    for res in results:
        if settings.export_original:
            mat = pd.DataFrame(
                res.final_matrix, index=unq_zone_list, columns=unq_zone_list
            ).reset_index()

            mat = mat.rename(columns={"index": ia_name})

            # Path export
            es_path = export_folder / (trip_origin.value + "_enhpa")
            if settings.time_period in add_cols:
                es_path = nup.build_path(es_path, res.segmentation, tp=settings.time_period)
            else:
                es_path = nup.build_path(es_path, res.segmentation)

            mat.to_csv(es_path, index=False)

        if settings.export_furness:
            furness_mat = pd.DataFrame(
                res.furness_matrix, index=unq_zone_list, columns=unq_zone_list
            ).reset_index()

            furness_mat = furness_mat.rename(columns={"index": ia_name})

            # Path export
            es_path = export_folder / (trip_origin.value + "_enhpafn")
            if settings.time_period in add_cols:
                es_path = nup.build_path(es_path, res.segmentation, tp=settings.time_period)
            else:
                es_path = nup.build_path(es_path, res.segmentation)

            furness_mat.to_csv(es_path, index=False)

        # Output reports
        report_path = reports_folder / (trip_origin.value + "_disagg_report")
        if settings.time_period in add_cols:
            report_path = nup.build_path(
                report_path, res.segmentation, tp=settings.time_period
            )
        else:
            report_path = nup.build_path(report_path, res.segmentation)
        report_path = report_path.with_suffix(".xlsx")

        with pd.ExcelWriter(report_path) as excel:
            summary = pd.DataFrame.from_dict(
                res.summary._asdict(), columns=["Value"], orient="index"
            )
            summary.index = summary.index.str.replace("_", " ").str.title()
            summary.to_excel(excel, sheet_name="Summary")

            tlds = []
            for nm, tld in (
                ("Final Matrix", res.final_matrix_tld),
                ("Observed", res.target_tld),
                ("Cost", res.cost_tld),
            ):
                index_cols = [tld.min_bounds_col, tld.max_bounds_col, tld.mid_bounds_col]
                df = tld.to_df().set_index(index_cols)
                df.columns = [f"{nm} {c.title()}" for c in df.columns]
                df.index.names = [i.title() for i in df.index.names]
                tlds.append(df)
            pd.concat(tlds, axis=1).to_excel(excel, sheet_name="Cost Distribution")

        path = reports_folder / (trip_origin.value + f"_tld")
        if settings.time_period in add_cols:
            path = nup.build_path(path, res.segmentation, tp=settings.time_period)
        else:
            path = nup.build_path(path, res.segmentation)
        path = path.with_suffix(".pdf")

        tlds = []
        for nm, tld in (
            ("Final", res.final_matrix_tld),
            ("Target", res.target_tld),
            ("Cost", res.cost_tld),
        ):
            tlds.append(cost_utils.PlotData(tld.band_means, tld.band_shares, f"{nm}"))
        cost_utils.plot_cost_distributions(tlds, path.stem, path=path)

    return results


def _calculate_bandshare_convergence(
    matrix: cost.CostDistribution, target: cost.CostDistribution
) -> float:
    bs_con = 1 - (
        np.sum((matrix.band_shares - target.band_shares) ** 2)
        / np.sum((target.band_shares - np.mean(target.band_shares)) ** 2)
    )
    return max(bs_con, 0)


def _dissag_seg(
    prod_list: List[Dict[str, Any]],
    attr_list: List[Dict[str, Any]],
    tld_list: List[cost.CostDistribution],
    sd: pd.DataFrame,
    costs: pd.DataFrame,
    furness_loops=1500,
    min_pa_diff=0.1,
    bs_con_crit=0.975,
    max_bs_loops: int = 300,
) -> List[DisaggregationResults]:  # TODO Update docstring
    """
    prod_list:
        List of production vector dictionaries
    attr_list:
        List of attraction vector dictionaries
    tld_list:
        List of tld dictionaries
    sd:
        Base matrix
    """

    # build prod cube
    # build attr cube
    # build tld cube

    out_mats = []

    seg_cube = np.ndarray((len(sd), len(sd), len(prod_list)))
    factor_cube = np.ndarray((len(sd), len(sd), len(prod_list)))
    out_cube = np.ndarray((len(sd), len(sd), len(prod_list)))

    # seg_x = 0,prod_list[0]
    seg_audit = list()
    for i, (prod, attr, target_tld) in enumerate(zip(prod_list, attr_list, tld_list)):
        target_p: np.ndarray = prod["trips"]
        target_a: np.ndarray = attr["trips"]

        # Build audit dict
        audit_dict = {"segmentation": {k: v for k, v in prod.items() if k != "trips"}}

        # Initialise the output mat
        new_mat = sd / sd.sum()
        new_mat = new_mat * target_p.sum()

        # Update audit_dict
        audit_dict["target_p"] = target_p.sum()
        audit_dict["target_a"] = target_a.sum()

        # Calculate matrix tlb
        matrix_tld = cost.CostDistribution.from_trips(
            new_mat,
            costs.values,
            target_tld.min_bounds,
            target_tld.max_bounds,
            target_tld.cost_units,
        )
        cost_tld = cost.CostDistribution.from_trips(
            np.ones_like(costs.values),
            costs.values,
            target_tld.min_bounds,
            target_tld.max_bounds,
            target_tld.cost_units,
        )
        # Calculate the band share convergence
        bs_con = _calculate_bandshare_convergence(matrix_tld, target_tld)

        # TLB balance/furness
        tlb_loop = 1
        conv_fail = False
        audit_dict["initial_bs_convergence"] = bs_con
        audit_dict["cost_tld"] = cost_tld
        while bs_con < bs_con_crit:

            # Calculating K factors for cells based on which distance band they're in
            k_factors = np.ones_like(costs.values)
            for min_, max_, obs, mat in zip(
                target_tld.min_bounds,
                target_tld.max_bounds,
                target_tld.band_trips,
                matrix_tld.band_trips,
            ):
                if mat <= 0:
                    continue  # Leave K factor as 1 if there aren't any matrix trips

                k_factors = np.where(
                    (costs >= min_) & (costs < max_),
                    min(max(obs / mat, 0.001), 10),
                    k_factors,
                )

            # Run furness
            new_mat = k_factors * new_mat

            # TODO Replace with furness function
            # Furness
            for fur_loop in range(furness_loops):

                fur_loop += 1

                # Adjust attractions
                mat_d = np.sum(new_mat, axis=0)
                mat_d[mat_d == 0] = 1
                new_mat = new_mat * target_a / mat_d

                # Adjust productions
                mat_o = np.sum(new_mat, axis=1)
                mat_o[mat_o == 0] = 1
                new_mat = (new_mat.T * target_p / mat_o).T

                # Get pa diff
                mat_o = np.sum(new_mat, axis=1)
                mat_d = np.sum(new_mat, axis=0)
                pa_diff = math_utils.get_pa_diff(mat_o, target_p, mat_d, target_a)  # .max()

                if pa_diff < min_pa_diff or np.isnan(np.sum(new_mat)):
                    print(f"Furness took {fur_loop} loops")
                    break

            prior_bs_con = bs_con

            # Calculate new TLD and bandshare convergence
            matrix_tld = cost.CostDistribution.from_trips(
                new_mat,
                costs.values,
                target_tld.min_bounds,
                target_tld.max_bounds,
                target_tld.cost_units,
            )
            bs_con = _calculate_bandshare_convergence(matrix_tld, target_tld)

            print("Loop " + str(tlb_loop), "Band share convergence: " + str(bs_con))

            # If tiny improvement, exit loop
            if (np.absolute(bs_con - prior_bs_con) < 0.001) and (bs_con != 0):
                break

            tlb_loop += 1

            if tlb_loop >= max_bs_loops:
                conv_fail = True

            # Add to audit dict
            audit_dict.update(
                {
                    "estimated_p": mat_o.sum(),
                    "estimated_a": mat_d.sum(),
                    "pa_diff": pa_diff,
                    "bs_con": bs_con,
                    "conv_fail": conv_fail,
                }
            )

            if conv_fail:
                break

        # Append dict
        seg_audit.append(audit_dict)

        # Push back to the cube
        seg_cube[:, :, i] = new_mat

    # ORDER OF PREFERENCE
    # Balance to p/a sd, tld, cells @ sd if possible, P/A slice

    # TODO(BT): Replace with LAD aggregation method
    # Replaces all zeros with tiny value - prevents zero splits
    seg_cube = np.where(seg_cube == 0, _TINY_INFILL, seg_cube)

    # Calculate the total of all the matrices
    cube_sd = seg_cube.sum(axis=2)

    # calculate the splitting factors
    target_tld: cost.CostDistribution
    for i, (prod, target_tld) in enumerate(zip(prod_list, tld_list)):
        # Get share of cell values from original matrix
        factor_cube[:, :, i] = seg_cube[:, :, i] / cube_sd

        # Multiply original matrix by share to get cell balanced out matrix
        out_cube[:, :, i] = sd * factor_cube[:, :, i]
        print(out_cube[:, :, i].sum())

        # Get trip length by band
        matrix_tld = cost.CostDistribution.from_trips(
            out_cube[:, :, i],
            costs.values,
            target_tld.min_bounds,
            target_tld.max_bounds,
            target_tld.cost_units,
        )
        bs_con = _calculate_bandshare_convergence(matrix_tld, target_tld)

        seg_audit[i].update(
            {"target_tld": target_tld, "final_tld": matrix_tld, "final_bs_con": bs_con}
        )

        out_dict = {"furness_mat": seg_cube[:, :, i], "mat": out_cube[:, :, i]}

        # Add lavels back on to export list
        out_dict.update({k: v for k, v in prod.items() if k != "trips"})
        out_mats.append(out_dict)

    out_sd = out_cube.sum(axis=2)

    final_pa_diff = math_utils.get_pa_diff(
        out_sd.sum(axis=1), sd.sum(axis=1), out_sd.sum(axis=0), sd.sum(axis=0)
    )

    outputs = []
    for mats, res in zip(out_mats, seg_audit):
        outputs.append(
            DisaggregationResults(
                furness_matrix=mats["furness_mat"],
                final_matrix=mats["mat"],
                segmentation=res["segmentation"],
                cost_tld=res["cost_tld"],
                final_matrix_tld=res["final_tld"],
                target_tld=res["target_tld"],
                summary=DisaggregationSummaryResults(
                    initial_bs_convergence=res["initial_bs_convergence"],
                    final_bs_convergence=res["final_bs_con"],
                    target_productions=res["target_p"],
                    target_attractions=res["target_a"],
                    estimated_productions=res["estimated_p"],
                    estimated_attractions=res["estimated_a"],
                    initial_pa_difference=res["pa_diff"],
                    final_pa_difference=final_pa_diff,
                    convergence_fail=res["conv_fail"],
                ),
            )
        )

    return outputs


def get_costs(
    cost_folder: pathlib.Path, calib_params: Dict[str, Any], zones: np.ndarray
) -> pd.DataFrame:
    purpose_lookup = {
        "commute": [1],
        "business": [2, 12],
        "other": [3, 4, 5, 6, 7, 8, 13, 14, 15, 16, 18],
    }

    seg_columns = ["trip_origin", "m", "p", "ca"]
    cost_files = pd.DataFrame(
        nup.parse_matrix_folder(
            cost_folder,
            extension_filter=constants.VALID_MAT_FTYPES,
            required_data=seg_columns,
        )
    )
    cost_files = cost_files.set_index(seg_columns)

    purposes = purpose_lookup[calib_params["uc"]]
    cost_files = cost_files.loc[
        pd.IndexSlice[
            calib_params["trip_origin"], calib_params["m"], purposes, calib_params["ca"]
        ],
        "path",
    ]

    if len(cost_files) == 0:
        raise ValueError(f"cannot find any cost files for {calib_params}")

    costs: List[pd.DataFrame] = []
    for file in cost_files:
        mat = file_ops.read_df(file, index_col=0)
        mat.columns = pd.to_numeric(mat.columns, errors="ignore", downcast="unsigned")
        mat.index = pd.to_numeric(mat.index, errors="ignore", downcast="unsigned")

        if np.not_equal(mat.columns, zones).any():
            raise ValueError("Wrong zones found in cost file columns: {file}")
        if np.not_equal(mat.index, zones).any():
            raise ValueError("Wrong zones found in cost file index: {file}")

        costs.append(mat)

    if len(costs) == 1:
        return costs[0]

    return pd.DataFrame(np.mean(costs, axis=0), index=costs[0].index, columns=costs[0].columns)
