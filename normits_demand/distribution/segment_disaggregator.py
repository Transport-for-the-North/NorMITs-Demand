# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 13:47:56 2020

@author: genie
"""

import csv
import enum
import pathlib
from typing import Any, Dict, List, NamedTuple, Optional

import numpy as np
import pandas as pd
import pydantic

import normits_demand as nd
from normits_demand import cost
from normits_demand.cost import utils as cost_utils
from normits_demand.concurrency import multiprocessing as mp
from normits_demand.distribution import furness
from normits_demand.utils import file_ops, math_utils
from normits_demand.utils import utils as nup  # TODO Remove dependancy on this
from normits_demand import logging as nd_log
from normits_demand import constants
from normits_demand.core import enumerations as nd_enum

LOG = nd_log.get_logger(__name__)
_TINY_INFILL = 1 * 10 ** -8


class TripEndConstraint(nd_enum.IsValidEnumWithAutoNameLower):
    """Constraint to use for target trip ends within the matrix disaggregation."""

    DOUBLE = enum.auto()
    SINGLE = enum.auto()
    NONE = enum.auto()


class DisaggregationSettings(pydantic.BaseModel):  # pylint: disable=no-member
    """Optional settings for adjusting `disaggregate_segments`."""

    aggregate_surplus_segments: bool = True
    export_original: bool = True
    export_furness: bool = False
    intrazonal_cost_infill: float = 0.5
    maximum_furness_loops: int = 1999
    pa_furness_convergence: float = 0.1
    bandshare_convergence: float = 0.975
    max_bandshare_loops: int = 200
    multiprocessing_threads: int = -1
    pa_match_tolerance: float = 0.95
    minimum_bandshare_change: float = 1e-8
    trip_end_constraint: TripEndConstraint = TripEndConstraint.DOUBLE


class DisaggregationSummaryResults(NamedTuple):
    """High level results from the `_segment_build_worker` function."""

    tld_convergence_loop: int
    tld_loop_convergence_fail: bool
    bandshare_convergence: float
    minimum_k_factor: float
    maximum_k_factor: float
    target_productions: float
    target_attractions: float
    estimated_productions: float
    estimated_attractions: float
    target_trip_end_pa_difference: float
    furness_loops: int
    furness_converged: bool
    input_matrix_total: float
    segmented_matrix_total: float


class DisaggregationCombinedResults(NamedTuple):
    """Results comparing the combined segmented matrices to the input matrix."""

    input_matrix_total: float
    combined_segment_total: float
    input_matrix_productions: float
    input_matrix_attractions: float
    combined_segment_productions: float
    combined_segment_attractions: float
    matrix_pa_difference: float


class DisaggregationResults(NamedTuple):
    """All results returned by the `_segment_build_worker` function."""

    furness_matrix: np.ndarray
    final_matrix: np.ndarray
    segmentation: dict[str, Any]
    cost_tld: cost.CostDistribution
    input_matrix_tld: cost.CostDistribution
    final_matrix_tld: cost.CostDistribution
    target_tld: cost.CostDistribution
    summary: DisaggregationSummaryResults
    combined: DisaggregationCombinedResults


class DisaggregationOutputSegment(nd_enum.IsValidEnumWithAutoNameLower):
    """Adding segmentation to disaggregate the matrix too."""

    G = enum.auto()
    SOC = enum.auto()
    NS = enum.auto()


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


def _build_enhanced_pa(
    calib_params: dict[str, Any],
    trip_ends: pd.DataFrame,
    value_col: str,
    ia_name: str = None,
    unq_zone_list: np.ndarray = None,
) -> np.ndarray:
    """
    Private
    """
    if ia_name is None:
        ia_name = list(trip_ends.columns)[0]
    if unq_zone_list is None:
        unq_zone_list = nup.get_zone_range(trip_ends[ia_name])

    trip_ends_filtered, total = nup.filter_pa_vector(
        trip_ends,
        ia_name,
        calib_params,
        round_val=None,
        value_var=value_col,
    )
    trip_ends_filtered = nup.df_to_np(
        trip_ends_filtered,
        v_heading=ia_name,
        values=value_col,
        unq_internal_zones=unq_zone_list,
    )

    if total == 0:
        raise Warning("Filter returned 0 trips, failing")

    return trip_ends_filtered


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
    cell_sum = np.where(sum(attr_list) == 0, 0.000001, sum(attr_list))

    mat_attr = sd.sum(axis=0)

    new_attr = attr_list.copy()

    for ait, a in enumerate(attr_list):
        new_attr[ait] = mat_attr * (a / cell_sum)

    audit_sum = sum(new_attr)
    audit = audit_sum.round() == mat_attr.round()

    return new_attr, audit


def _is_nan(value: Any) -> bool:
    if not isinstance(value, (int, float)):
        return False
    return np.isnan(value)


def _control_a_to_enhanced_p(
    productions: list[np.ndarray],
    attractions: list[np.ndarray],
):
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
    # TODO Update docstring
    if len(productions) != len(attractions):
        raise ValueError(
            "productions and attractions lists are not the same lengths, "
            f"{len(productions)} and {len(attractions)} respectively"
        )

    new_attractions: list[np.ndarray] = []
    audit: list[bool] = []
    for prod, attr in zip(productions, attractions):
        prod_tot = np.sum(prod)

        # Get attractions demand as factor
        new_attr = attr / np.sum(attr)
        # Multiply factor by production demand
        new_attr = new_attr * prod_tot

        new_attractions.append(new_attr)

        # Check trip end totals
        audit.append(prod_tot.round(0) == np.sum(new_attr).round(0))

    return new_attractions, audit


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
    disaggregation_segment: DisaggregationOutputSegment,
    trip_origin: nd.TripOrigin = nd.TripOrigin.HB,
    settings: DisaggregationSettings = DisaggregationSettings(),
) -> None:  # TODO Update docstring
    """
    Parameters
    ----------
    trip_origin:
        from 'hb', 'nhb', 'both'

    aggregate_surplus_segments = True:
        If there are segments on the left hand side that aren't in the
        enhanced segmentation, aggregated them. Will
    """
    required_columns = ["matrix_type", "trip_origin", "yr", "m"]
    if model == nd.AssignmentModel.NORMS:
        required_columns.append("ca")
    seg_dtypes = dict.fromkeys(("p", "yr", "m", "ca", "soc", "ns"), "Int64")

    # Find all matrices and extract segmentation info
    LOG.info("Finding base matrices in %s", import_folder)
    base_mat_seg = pd.DataFrame(
        file_ops.parse_folder_files(
            import_folder,
            extension_filter=constants.VALID_MAT_FTYPES,
            required_data=required_columns,
        )
    )
    base_mat_seg = _set_dataframe_dtypes(base_mat_seg, seg_dtypes)
    base_mat_seg.loc[:, "name"] = base_mat_seg["path"].apply(lambda p: p.name)
    base_mat_seg = base_mat_seg.loc[
        (base_mat_seg["matrix_type"] == "pa")
        & (base_mat_seg["trip_origin"] == trip_origin.value)
    ].drop(columns="matrix_type")
    duplicates = base_mat_seg.duplicated().sum()
    if duplicates > 0:
        raise ValueError(f"{duplicates} matrices with the same segmentation found")
    exclude_columns = ["path", "name", "yr"]
    segment_columns = [c for c in base_mat_seg.columns if c not in exclude_columns]
    base_mat_seg.rename(columns={"path": "matrix_path", "name": "matrix_name"}, inplace=True)
    LOG.info("Found matrices at segmentation: %s", " ".join(segment_columns))

    # Find all TLDs and extract segmentation info
    LOG.info("Finding TLDs in %s", target_tld_folder)
    tld_seg = pd.DataFrame(
        file_ops.parse_folder_files(
            target_tld_folder,
            extension_filter=constants.VALID_MAT_FTYPES,
            required_data=segment_columns + [disaggregation_segment.value],
        )
    )
    tld_seg = _set_dataframe_dtypes(tld_seg, seg_dtypes)
    tld_seg.loc[:, "name"] = tld_seg["path"].apply(lambda p: p.name)
    tld_seg = tld_seg.loc[
        tld_seg["trip_origin"] == trip_origin.value,
        ["path", "name", *segment_columns, disaggregation_segment.value],
    ]
    tld_seg.rename(columns={"path": "tld_path", "name": "tld_name"}, inplace=True)
    duplicates = tld_seg.duplicated().sum()
    if duplicates > 0:
        raise ValueError(f"{duplicates} TLDs with the same segmentation found")
    output_segments = base_mat_seg.merge(
        tld_seg, on=segment_columns, how="left", validate="1:m"
    )
    segment_columns += [disaggregation_segment.value]
    out_path = export_folder / "output_segmentations.csv"
    output_segments.set_index(segment_columns, inplace=True)
    output_segments.to_csv(out_path)
    LOG.info("Segmentations found written to: %s", out_path)

    missing = output_segments.isna().any(axis=1)
    if missing.sum() > 0:
        LOG.error(
            "%s segments with missing data dropped:\n%s",
            missing.sum(),
            output_segments.loc[missing],
        )
        output_segments = output_segments.loc[~missing]
    output_segments = output_segments.reset_index()

    # Check trip ends are provided at the correct segmentation,
    # and aggregate any unecessary columns
    trip_end_seg_cols = [c for c in segment_columns if c != "trip_origin"]
    unique_zones = base_productions.zoning_system.unique_zones
    zone_col_name = base_productions.zone_col
    dvecs = {"productions": base_productions, "attractions": base_attractions}

    trip_ends = []
    for nm, dvec in dvecs.items():
        df = dvec.to_df().rename(columns={"val": nm})
        missing = [c for c in trip_end_seg_cols if c not in df.columns]
        if missing:
            raise KeyError(f"columns missing from {nm} trip ends: {', '.join(missing)}")

        trip_ends.append(df.groupby([zone_col_name] + trip_end_seg_cols).agg({nm: "sum"}))

    trip_ends = pd.concat(trip_ends, axis=1, verify_integrity=True).reset_index()
    # TODO Check that TLDs and trip ends contain the same segment values for all columns

    # Main disaggregation loop
    kwargs: list[dict[str, Any]] = []
    for matrix_path in output_segments["matrix_path"].unique():
        mat_segments = output_segments.loc[output_segments["matrix_path"] == matrix_path]
        matrix_segmentation = {
            c: mat_segments.iloc[0].at[c]
            for c in segment_columns
            if c != disaggregation_segment.value
        }

        tld_paths: list[pathlib.Path] = []
        segment_params: list[dict[str, Any]] = []
        for tld_data in mat_segments.itertuples(index=False):
            tld_paths.append(tld_data.tld_path)
            segment_params.append({c: getattr(tld_data, c) for c in segment_columns})

        kwargs.append(
            {
                "matrix_path": matrix_path,
                "matrix_segmentation": matrix_segmentation,
                "tld_paths": tld_paths,
                "segmentation_parameters": segment_params,
                "tld_units": tld_units,
                "trip_ends": trip_ends,
                "unique_zones": unique_zones,
                "zone_col_name": zone_col_name,
                "cost_folder": cost_folder,
                "export_folder": export_folder,
                "trip_origin": trip_origin,
                "settings": settings,
            }
        )

    # Call using multiple threads
    LOG.debug("Running segment disaggregator on %s threads", settings.multiprocessing_threads)
    mp.multiprocess(
        _segment_build_worker,
        kwargs=kwargs,
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
    matrix_path: pathlib.Path,
    matrix_segmentation: dict[str, Any],
    tld_paths: list[pathlib.Path],
    segmentation_parameters: list[dict[str, Any]],
    tld_units: nd.CostUnits,
    trip_ends: pd.DataFrame,
    unique_zones: np.ndarray,
    zone_col_name: str,
    cost_folder: pathlib.Path,
    export_folder: pathlib.Path,
    trip_origin: nd.TripOrigin,
    settings: DisaggregationSettings,
) -> List[DisaggregationResults]:
    # TODO Add docstring
    if len(tld_paths) != len(segmentation_parameters):
        raise ValueError(
            f"{len(tld_paths)} found but {len(segmentation_parameters)} segmentations"
        )

    # Read & check base matrix
    base_matrix = file_ops.read_df(matrix_path, index_col=0)
    base_matrix.index = pd.to_numeric(base_matrix.index, downcast="unsigned", errors="ignore")
    base_matrix.columns = pd.to_numeric(
        base_matrix.columns, downcast="unsigned", errors="ignore"
    )
    if not base_matrix.index.equals(base_matrix.columns):
        raise ValueError("base matrix columns and index aren't equal")
    if len(unique_zones) != len(base_matrix.index):
        raise ValueError(
            "base matrix has a different number of zones to expected zones, "
            f"{len(base_matrix.index)} and {len(unique_zones)} respectively"
        )
    if not np.equal(np.sort(unique_zones), np.sort(base_matrix.index.values)).all():
        raise ValueError("base matrix zones not equal to expected zones")

    # Filter productions and attractions for each output segment
    productions_list: list[np.ndarray] = []
    attractions_list: list[np.ndarray] = []
    for seg in segmentation_parameters:
        productions_list.append(
            _build_enhanced_pa(seg, trip_ends, "productions", zone_col_name, unique_zones)
        )
        attractions_list.append(
            _build_enhanced_pa(seg, trip_ends, "attractions", zone_col_name, unique_zones)
        )

    # control to sum of target share of attraction vector, cell-wise.
    attractions_list, control_aud = _control_a_to_seed_matrix(base_matrix, attractions_list)
    # Control a to p, exactly this time
    attractions_list, bal_aud = _control_a_to_enhanced_p(productions_list, attractions_list)

    # Check audit vectors
    if (sum(control_aud) < len(control_aud) * settings.pa_match_tolerance) or (
        sum(bal_aud) < len(bal_aud) * settings.pa_match_tolerance
    ):
        raise ValueError(
            f"PA Vectors not balanced, within tolerance {settings.pa_match_tolerance}"
        )

    # Setup generator which will read TLDs
    tld_gen = [cost.CostDistribution.from_csv(p, tld_units) for p in tld_paths]

    costs = get_costs(cost_folder, matrix_segmentation, unique_zones)

    reports_folder = export_folder / "Reports"
    reports_folder.mkdir(exist_ok=True)

    # Pass to dissagg function
    results = _dissag_seg(
        segmentation_parameters,
        productions_list,
        attractions_list,
        tld_gen,
        base_matrix,
        costs,
        reports_folder,
        furness_loops=settings.maximum_furness_loops,
        min_pa_diff=settings.pa_furness_convergence,
        bs_con_crit=settings.bandshare_convergence,
        max_bs_loops=settings.max_bandshare_loops,
        min_bs_change=settings.minimum_bandshare_change,
        trip_end_constraint=settings.trip_end_constraint,
    )

    for res in results:
        reports_folder = _segment_disaggregator_outputs(
            res,
            export_folder,
            unique_zones,
            zone_col_name,
            trip_origin,
            export_furness=settings.export_furness,
            export_original=settings.export_original,
        )

    # Graph comparing segmented matrix TLDs to original
    tlds = [
        cost_utils.PlotData(
            results[0].input_matrix_tld.band_means,
            results[0].input_matrix_tld.band_shares,
            "Original Matrix",
        )
    ]

    for res in results:
        label = trip_origin.value.upper() + " ".join(
            f"{k}{v}" for k, v in res.segmentation.items() if k != "trip_origin"
        )
        tlds.append(
            cost_utils.PlotData(
                res.final_matrix_tld.band_means, res.final_matrix_tld.band_shares, label
            )
        )

    path = build_path(
        reports_folder / (trip_origin.value + "_tld"),
        {k: v for k, v in matrix_segmentation.items() if k != "trip_origin"},
        suffix=".png",
    )
    cost_utils.plot_cost_distributions(tlds, path.stem, path=path)
    print(f"Created: {path.name}")

    return results


def _segment_disaggregator_outputs(
    results: DisaggregationResults,
    export_folder: pathlib.Path,
    unique_zones: np.ndarray,
    zones_name: str,
    trip_origin: nd.TripOrigin,
    export_furness: bool,
    export_original: bool,
) -> pathlib.Path:
    """Write matrices, summary spreadsheets and TLD graphs to `export_folder`."""
    if export_original:
        mat = pd.DataFrame(
            results.final_matrix, index=unique_zones, columns=unique_zones
        ).reset_index()
        mat = mat.rename(columns={"index": zones_name})

        OUTPUT_NAME = "_pa"
        path = build_path(
            export_folder / (trip_origin.value + OUTPUT_NAME),
            {k: v for k, v in results.segmentation.items() if k != "trip_origin"},
            suffix=".csv.bz2",
        )
        mat.to_csv(path, index=False)

    if export_furness:
        furness_mat = pd.DataFrame(
            results.furness_matrix, index=unique_zones, columns=unique_zones
        ).reset_index()
        furness_mat = furness_mat.rename(columns={"index": zones_name})

        path = build_path(
            export_folder / (trip_origin.value + OUTPUT_NAME + "_fn"),
            {k: v for k, v in results.segmentation.items() if k != "trip_origin"},
            suffix=".csv.bz2",
        )
        furness_mat.to_csv(path, index=False)

    # Output Excel based reports
    reports_folder = export_folder / "Reports"
    path = build_path(
        reports_folder / (trip_origin.value + "_disagg_report"),
        {k: v for k, v in results.segmentation.items() if k != "trip_origin"},
        suffix=".xlsx",
    )

    with pd.ExcelWriter(path) as excel:
        summary = pd.DataFrame.from_dict(
            results.summary._asdict(), columns=["Value"], orient="index"
        )
        summary.index = summary.index.str.replace("_", " ").str.title()
        summary.to_excel(excel, sheet_name="Summary")

        combined = pd.DataFrame.from_dict(
            results.combined._asdict(), columns=["Value"], orient="index"
        )
        combined.index = combined.index.str.replace("_", " ").str.title()
        combined.to_excel(excel, sheet_name="Combined")

        tlds = []
        for nm, tld in (
            ("Input Matrix", results.input_matrix_tld),
            ("Segmented Matrix", results.final_matrix_tld),
            ("Observed", results.target_tld),
            ("Cost", results.cost_tld),
        ):
            index_cols = [tld.min_bounds_col, tld.max_bounds_col, tld.mid_bounds_col]
            df = tld.to_df().set_index(index_cols)
            df.columns = [f"{nm} {c.title()}" for c in df.columns]
            df.index.names = [i.title() for i in df.index.names]
            tlds.append(df)
        pd.concat(tlds, axis=1).to_excel(excel, sheet_name="Cost Distribution")

    # Output TLD graphs
    path = build_path(
        reports_folder / (trip_origin.value + "_tld"),
        {k: v for k, v in results.segmentation.items() if k != "trip_origin"},
        suffix=".png",
    )

    tlds = []
    for nm, tld in (
        (f"Segmented {results.summary.bandshare_convergence:.3f}", results.final_matrix_tld),
        ("Observed", results.target_tld),
    ):
        tlds.append(cost_utils.PlotData(tld.band_means, tld.band_shares, f"{nm}"))

    cost_utils.plot_cost_distributions(tlds, path.stem, path=path)

    return reports_folder


def _calculate_bandshare_convergence(
    matrix: cost.CostDistribution, target: cost.CostDistribution
) -> float:
    bs_con = 1 - (
        np.sum((matrix.band_shares - target.band_shares) ** 2)
        / np.sum((target.band_shares - np.mean(target.band_shares)) ** 2)
    )
    return max(bs_con, 0)


def _dissag_seg(
    segmentation_list: list[dict[str, Any]],
    prod_list: List[pd.DataFrame],
    attr_list: List[pd.DataFrame],
    tld_list: list[cost.CostDistribution],
    base_matrix: pd.DataFrame,
    costs: pd.DataFrame,
    report_folder: pathlib.Path,
    furness_loops: int,
    min_pa_diff: float,
    bs_con_crit: float,
    max_bs_loops: int,
    min_bs_change: float,
    trip_end_constraint: TripEndConstraint,
) -> List[DisaggregationResults]:  # TODO Update docstring
    out_mats = []

    base_matrix: np.ndarray = np.array(base_matrix)
    costs: np.ndarray = np.array(costs)

    seg_cube = np.ndarray((len(base_matrix), len(base_matrix), len(prod_list)))
    factor_cube = np.ndarray((len(base_matrix), len(base_matrix), len(prod_list)))
    out_cube = np.ndarray((len(base_matrix), len(base_matrix), len(prod_list)))

    # seg_x = 0,prod_list[0]
    summary_results: list[DisaggregationSummaryResults] = []
    seg_audit = []
    for i, (target_p, target_a, target_tld, segmentation) in enumerate(
        zip(prod_list, attr_list, tld_list, segmentation_list)
    ):
        # Build audit dict
        audit_dict = {"segmentation": segmentation}

        # Convert to arrays because all the matrix calculations
        # are done as arrays instead of DataFrames
        target_p = np.array(target_p)
        target_a = np.array(target_a)

        # Initialise the output mat
        new_mat = np.divide(
            base_matrix.T,
            base_matrix.sum(axis=1),
            out=np.zeros_like(base_matrix),
            where=base_matrix.sum(axis=1) > 0,
        )
        new_mat: np.ndarray = (new_mat * target_p).T

        target_te_totals = (target_p.sum(), target_a.sum())
        input_matrix_total = np.sum(base_matrix)

        # Calculate matrix tlb
        audit_dict["input_matrix_tld"] = cost.CostDistribution.from_trips(
            base_matrix,
            costs,
            target_tld.min_bounds,
            target_tld.max_bounds,
            target_tld.cost_units,
        )
        audit_dict["cost_tld"] = cost.CostDistribution.from_trips(
            np.ones_like(costs),
            costs,
            target_tld.min_bounds,
            target_tld.max_bounds,
            target_tld.cost_units,
        )
        matrix_tld = cost.CostDistribution.from_trips(
            new_mat,
            costs,
            target_tld.min_bounds,
            target_tld.max_bounds,
            target_tld.cost_units,
        )
        # Calculate the band share convergence
        bs_con = _calculate_bandshare_convergence(matrix_tld, target_tld)

        report_path = build_path(
            report_folder / (segmentation["trip_origin"] + "_tld_log"),
            {k: v for k, v in segmentation.items() if k != "trip_origin"},
            suffix=".csv",
        )
        print(f"Writing TLD loop log to {report_path.name}")
        with open(report_path, "wt", newline="") as file:
            report_csv = csv.writer(file)
            report_csv.writerow(DisaggregationSummaryResults._fields)

            tlb_loop = 0
            conv_fail = False

            disag_summary = DisaggregationSummaryResults(
                tld_convergence_loop=tlb_loop,
                tld_loop_convergence_fail=conv_fail,
                bandshare_convergence=bs_con,
                minimum_k_factor=None,
                maximum_k_factor=None,
                target_productions=target_te_totals[0],
                target_attractions=target_te_totals[1],
                estimated_productions=None,
                estimated_attractions=None,
                target_trip_end_pa_difference=None,
                furness_loops=None,
                furness_converged=None,
                input_matrix_total=input_matrix_total,
                segmented_matrix_total=np.sum(new_mat),
            )
            report_csv.writerow(disag_summary)

            # TLB balance/furness
            while bs_con < bs_con_crit:

                # Calculating K factors for cells based on which distance band they're in
                k_factors = np.ones_like(costs)
                for min_, max_, obs, mat in zip(
                    target_tld.min_bounds,
                    target_tld.max_bounds,
                    target_tld.band_shares,
                    matrix_tld.band_shares,
                ):
                    if mat <= 0:
                        continue  # Leave K factor as 1 if there aren't any matrix trips

                    k_factors = np.where(
                        (costs >= min_) & (costs < max_),
                        np.clip(obs / mat, 0.001, 10),
                        k_factors,
                    )

                new_mat = k_factors * new_mat

                if trip_end_constraint == TripEndConstraint.DOUBLE:
                    new_mat, fur_loop, pa_diff = furness.doubly_constrained_furness(
                        new_mat, target_p, target_a, tol=min_pa_diff, max_iters=furness_loops
                    )

                elif trip_end_constraint == TripEndConstraint.SINGLE:
                    # Constrain matrix to productions
                    new_mat = np.divide(
                        base_matrix.T,
                        base_matrix.sum(axis=1),
                        out=np.zeros_like(base_matrix),
                        where=base_matrix.sum(axis=1) > 0,
                    )
                    new_mat = (new_mat * target_p).T

                    fur_loop = 0.5
                    pa_diff = math_utils.get_pa_diff(
                        new_mat.sum(axis=1), target_p, new_mat.sum(axis=0), target_a
                    )

                elif trip_end_constraint == TripEndConstraint.NONE:
                    fur_loop = 0
                    pa_diff = math_utils.get_pa_diff(
                        new_mat.sum(axis=1), target_p, new_mat.sum(axis=0), target_a
                    )

                else:
                    raise ValueError(
                        "invalid value for trip_end_constraint, "
                        f"expected {TripEndConstraint} got {trip_end_constraint}"
                    )

                prior_bs_con = bs_con

                # Calculate new TLD and bandshare convergence
                matrix_tld = cost.CostDistribution.from_trips(
                    new_mat,
                    costs,
                    target_tld.min_bounds,
                    target_tld.max_bounds,
                    target_tld.cost_units,
                )
                bs_con = _calculate_bandshare_convergence(matrix_tld, target_tld)

                tlb_loop += 1

                # Log loop results
                disag_summary = DisaggregationSummaryResults(
                    tld_convergence_loop=tlb_loop,
                    tld_loop_convergence_fail=conv_fail,
                    bandshare_convergence=bs_con,
                    minimum_k_factor=np.min(k_factors),
                    maximum_k_factor=np.max(k_factors),
                    target_productions=target_te_totals[0],
                    target_attractions=target_te_totals[1],
                    estimated_productions=np.sum(new_mat),
                    estimated_attractions=np.sum(new_mat),
                    target_trip_end_pa_difference=pa_diff,
                    furness_loops=fur_loop,
                    furness_converged=pa_diff < min_pa_diff,
                    input_matrix_total=input_matrix_total,
                    segmented_matrix_total=np.sum(new_mat),
                )
                report_csv.writerow(disag_summary)

                # If tiny improvement, exit loop
                if (np.absolute(bs_con - prior_bs_con) < min_bs_change) and (bs_con != 0):
                    break

                if tlb_loop >= max_bs_loops:
                    conv_fail = True
                    break

        # Append segmentations
        summary_results.append(disag_summary)
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
    for i, (disag_summary, target_tld, target_p, target_a) in enumerate(
        zip(summary_results, tld_list, prod_list, attr_list)
    ):
        # Get share of cell values from original matrix
        factor_cube[:, :, i] = seg_cube[:, :, i] / cube_sd

        # Multiply original matrix by share to get cell balanced out matrix
        out_cube[:, :, i] = base_matrix * factor_cube[:, :, i]

        # Get trip length by band
        matrix_tld = cost.CostDistribution.from_trips(
            out_cube[:, :, i],
            costs,
            target_tld.min_bounds,
            target_tld.max_bounds,
            target_tld.cost_units,
        )
        bs_con = _calculate_bandshare_convergence(matrix_tld, target_tld)

        pa_diff = math_utils.get_pa_diff(
            out_cube[:, :, i].sum(axis=1), target_p, out_cube[:, :, i].sum(axis=0), target_a
        )

        # Recalculate some summary stats
        overwrite = (
            "bandshare_convergence",
            "segmented_matrix_total",
            "target_trip_end_pa_difference",
        )
        prev_summary = {k: v for k, v in disag_summary._asdict().items() if k not in overwrite}
        summary_results[i] = DisaggregationSummaryResults(
            bandshare_convergence=bs_con,
            segmented_matrix_total=np.sum(out_cube[:, :, i]),
            target_trip_end_pa_difference=pa_diff,
            **prev_summary,
        )

        seg_audit[i].update({"target_tld": target_tld, "final_tld": matrix_tld})

        out_mats.append({"furness_mat": seg_cube[:, :, i], "mat": out_cube[:, :, i]})

    # Produce combined segment matrix comparison results
    combined_mat = out_cube.sum(axis=2)
    input_matrix_productions = base_matrix.sum(axis=1)
    combined_matrix_productions = combined_mat.sum(axis=1)
    input_matrix_attractions = base_matrix.sum(axis=0)
    combined_matrix_attractions = combined_mat.sum(axis=0)

    combined_results = DisaggregationCombinedResults(
        input_matrix_total=input_matrix_total,
        combined_segment_total=np.sum(combined_mat),
        input_matrix_productions=np.sum(input_matrix_productions),
        input_matrix_attractions=np.sum(input_matrix_attractions),
        combined_segment_productions=np.sum(combined_matrix_productions),
        combined_segment_attractions=np.sum(combined_matrix_attractions),
        matrix_pa_difference=math_utils.get_pa_diff(
            combined_matrix_productions,
            input_matrix_productions,
            combined_matrix_attractions,
            input_matrix_attractions,
        ),
    )

    outputs = []
    for mats, res, summary in zip(out_mats, seg_audit, summary_results):
        outputs.append(
            DisaggregationResults(
                furness_matrix=mats["furness_mat"],
                final_matrix=mats["mat"],
                segmentation=res["segmentation"],
                cost_tld=res["cost_tld"],
                input_matrix_tld=res["input_matrix_tld"],
                final_matrix_tld=res["final_tld"],
                target_tld=res["target_tld"],
                summary=summary,
                combined=combined_results,
            )
        )

    return outputs


def get_costs(
    cost_folder: pathlib.Path, calib_params: dict[str, Any], zones: np.ndarray
) -> pd.DataFrame:
    """Find cost file for `calib_params`, raises error if not found.

    Parameters
    ----------
    cost_folder : pathlib.Path
        Folder containing cost matrices.
    calib_params : dict[str, Any]
        Matrix segmentation parameters to find cost for.
    zones : np.ndarray
        Zones to check the cost matrices contain.

    Returns
    -------
    pd.DataFrame
        Cost matrix.

    Raises
    ------
    KeyError
        If any number other than 1 cost file can be found with
        parameters `calib_params`.
    ValueError
        If the cost matrices doesn't have the correct `zones`.
    """
    seg_columns = list(calib_params.keys())
    cost_files = pd.DataFrame(
        file_ops.parse_folder_files(
            cost_folder,
            extension_filter=constants.VALID_MAT_FTYPES,
            required_data=seg_columns,
        )
    )
    if cost_files.empty:
        raise KeyError(f"cannot find any cost files for {calib_params}")
    cost_files = cost_files.set_index(seg_columns)

    try:
        cost_path = cost_files.loc[
            pd.IndexSlice[tuple(calib_params.values())],
            "path",
        ]
    except KeyError as e:
        raise KeyError(f"cannot find any cost files for {calib_params}") from e

    if isinstance(cost_path, (pd.Series, pd.DataFrame)):
        if len(cost_path) == 1:
            cost_path = cost_path.iloc[0]
        else:
            raise KeyError(f"{len(cost_path)} cost files found for {calib_params}")

    cost_mat = file_ops.read_df(cost_path, index_col=0)
    cost_mat.columns = pd.to_numeric(cost_mat.columns, errors="ignore", downcast="unsigned")
    cost_mat.index = pd.to_numeric(cost_mat.index, errors="ignore", downcast="unsigned")

    if np.not_equal(cost_mat.columns, zones).any():
        raise ValueError("Wrong zones found in cost file columns: {file}")
    if np.not_equal(cost_mat.index, zones).any():
        raise ValueError("Wrong zones found in cost file index: {file}")

    return cost_mat


def build_path(
    base_path: pathlib.Path,
    segmentation_params: dict[str, Any],
    tp: Optional[str] = None,
    suffix: Optional[str] = None,
) -> pathlib.Path:
    """Build file name from segmentation parameters.

    Parameters
    ----------
    base_path : pathlib.Path
        Initial file path to add segmentation parameters to.
    segmentation_params : dict[str, Any]
        Segmentation parameter names and values.
    tp : Optional[str], optional
        Time period to add to file name, by default None
    suffix : Optional[str], optional
        Suffix to append to path, by default None

    Returns
    -------
    pathlib.Path
        Path containing segmentation information.
    """
    path = pathlib.Path(base_path)

    suffixes = "".join(path.suffixes)
    name = path.name.removesuffix(suffixes)

    for index, cp in segmentation_params.items():
        if index == "tlb":
            continue  # Ignore trip length bands

        if not _isnull(cp):
            # Don't include UC before user classes
            if index.lower() == "uc":
                name += f"_{cp}"
            else:
                name += f"_{index}{cp}"

    if tp:
        name += "_tp" + str(tp)

    path = path.parent / name
    if suffix:
        path = path.with_suffix(suffix)

    return path
