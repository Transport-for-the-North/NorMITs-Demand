# -*- coding: utf-8 -*-
"""
Created on: Mon Sept 21 09:06:46 2020
Updated on:

Original author: Ben Taylor
Last update made by:
Other updates made by:

File purpose:
Matrix processing functionality belongs here. This will be any processes
that do not belong specifically to pa_to_od.py, or od_to_pa.py.
"""
# Builtins
import os
import pickle
import pathlib

from typing import Any
from typing import List
from typing import Dict
from typing import Tuple
from typing import Iterable

from functools import reduce
from itertools import product
from collections import defaultdict

# Third Party
import numpy as np
import pandas as pd
from tqdm import tqdm

# Local imports
from normits_demand import efs_constants as consts
from normits_demand.utils import general as du

from normits_demand.matrices import pa_to_od as pa2od
from normits_demand.distribution import furness
from normits_demand.concurrency import multiprocessing

from normits_demand.matrices.tms_matrix_processing import *


def _aggregate(import_dir: str,
               in_fnames: List[str],
               export_path: str,
               round_dp: int = consts.DEFAULT_ROUNDING,
               ) -> str:
    """
    Loads the given files, aggregates together and saves in given location

    Returns
    -------
    Path of the exported matrix
    """
    # Load in files and aggregate
    aggregated_mat = None

    for fname in in_fnames:
        mat = pd.read_csv(os.path.join(import_dir, fname), index_col=0)

        # Build a matching df of 0s if not done yet
        if aggregated_mat is None:
            aggregated_mat = pd.DataFrame(0,
                                          index=mat.index,
                                          columns=mat.columns
                                          )
        aggregated_mat += mat

    # Write new matrix out
    aggregated_mat.round(decimals=round_dp).to_csv(export_path)
    print("Aggregated matrix written: %s" % os.path.basename(export_path))


def _recursive_aggregate(candidates: List[str],
                         segmentations: List[List[int]],
                         segmentation_strs: List[List[str]],
                         import_dir: str,
                         export_path: str,
                         round_dp: int = consts.DEFAULT_ROUNDING,
                         ) -> None:
    """
    The internal function of aggregate_matrices(). Recursively steps through
    the segmentations given, aggregating as it goes.


    Parameters
    ----------
    candidates:
        All remaining candidate matrices for this segmentation. This it narrowed
        down through the recursive calls.

    segmentations:
        A list of all segmentations (and their splits) that still need to be
        considered. Directly relates to segmentation_strs. Narrowed
        down through the recursive calls.

    segmentation_strs:
        A list of all segmentations (and their split names) that still need to
        be considered. Directly relates to segmentations. Narrowed
        down through the recursive calls.

    import_dir:
        Directory where the candidate matrices can be found.

    export_path:
        Directory to output the aggregated matrices.

    round_dp:
        The number of decimal places to round the output values to.
        Uses consts.DEFAULT_ROUNDING by default.
    """

    # ## EXIT CONDITION ## #
    if len(segmentations) == 1:
        # Un-nest
        segmentations = segmentations[0]
        segmentation_strs = segmentation_strs[0]

        if du.is_none_like(segmentations):
            # Aggregate remaining candidates
            _aggregate(
                import_dir=import_dir,
                in_fnames=candidates,
                export_path=export_path + '.csv',
                round_dp=round_dp,
            )
        else:
            # Loop through and aggregate
            for segment, seg_str in zip(segmentations, segmentation_strs):
                _aggregate(
                    import_dir=import_dir,
                    in_fnames=[x for x in candidates.copy() if seg_str in x],
                    export_path=export_path + seg_str + '.csv',
                    round_dp=round_dp,
                )
        # Exit condition done, leave recursion
        return

    # ## RECURSIVELY LOOP ## #
    seg, other_seg = segmentations[0], segmentations[1:]
    strs, other_strs = segmentation_strs[0], segmentation_strs[1:]

    if du.is_none_like(seg):
        # Don't need to segment here, next loop
        _recursive_aggregate(
            candidates=candidates,
            segmentations=other_seg,
            segmentation_strs=other_strs,
            import_dir=import_dir,
            export_path=export_path,
            round_dp=round_dp,
        )
    else:
        # Narrow down search, loop again
        for segment, seg_str in zip(seg, strs):
            _recursive_aggregate(
                candidates=[x for x in candidates.copy() if seg_str in x],
                segmentations=other_seg,
                segmentation_strs=other_strs,
                import_dir=import_dir,
                export_path=export_path + seg_str,
                round_dp=round_dp,
            )


def aggregate_matrices(import_dir: str,
                       export_dir: str,
                       trip_origin: str,
                       matrix_format: str,
                       years_needed: List[int],
                       p_needed: List[int],
                       m_needed: List[int],
                       soc_needed: List[int] = None,
                       ns_needed: List[int] = None,
                       ca_needed: List[int] = None,
                       tp_needed: List[int] = None,
                       round_dp: int = consts.DEFAULT_ROUNDING,
                       ) -> List[str]:
    """
    Aggregates the matrices in import_dir up to the given level and writes
    the new matrices out to export_dir

    Parameters
    ----------
    import_dir:
        Where to find the starting matrices.

    export_dir:
        Where to output the aggregated matrices.

    trip_origin:
        Where did the trips originate. Usually 'nhb' or 'hb'.

    matrix_format:
        The format of the matrices to convert. Usually 'pa' or 'od'.

    years_needed:
        Which years to aggregate for.

    p_needed:
        Which purposes to aggregate for.

    m_needed:
        Which mode to aggregate for.

    soc_needed:
        If None, skill levels will be aggregated. If set, chosen skill
        levels will be retained.

    ns_needed:
        If None, income levels will be aggregated. If set, chosen income
        levels will be retained.

    ca_needed:
        If None, car availability levels will be aggregated. If set, chosen
        car availability levels will be retained.

    tp_needed:
        If None, time periods will be aggregated. If set, chosen time periods
        will be retained.

    round_dp:
        The number of decimal places to round the output values to.
        Uses consts.DEFAULT_ROUNDING by default.

    Returns
    -------
    List of all aggregated matrix paths (optional)
    """
    # Init
    if((ns_needed is not None and soc_needed is None)
       or (soc_needed is not None and ns_needed is None)):
        raise ValueError("Both keep_soc and keep_ns need to be set at the same "
                         "time. Cannot set one and not the other.")

    # Build strings if needed
    soc_strs = list() if soc_needed is None else ['_soc' + str(x) for x in soc_needed]
    ns_strs = list() if ns_needed is None else ['_ns' + str(x) for x in ns_needed]
    ca_strs = list() if ca_needed is None else ['_ca' + str(x) for x in ca_needed]
    tp_strs = list() if tp_needed is None else ['_tp' + str(x) for x in tp_needed]

    # Load in all the candidate matrices and narrow down
    mat_format_str = '_' + matrix_format + '_'
    all_matrices = du.list_files(import_dir)
    all_matrices = [x for x in all_matrices if mat_format_str in x]
    all_matrices = [x for x in all_matrices if du.starts_with(x, trip_origin)]

    # for year, purpose, mode, time_period
    print("Writing files to: %s" % export_dir)
    mat_export_paths = []
    for year, m, p in product(years_needed, m_needed, p_needed):
        # Init
        if p in consts.SOC_P:
            segment_needed = soc_needed
            segment_str = soc_strs
        elif p in consts.NS_P:
            segment_needed = ns_needed
            segment_str = ns_strs
        elif p in consts.ALL_NHB_P:
            segment_needed = None
            segment_str = list()
        else:
            raise ValueError("Purpose '%s' is neither a soc, ns or nhb "
                             "segmentation somehow?" % str(p))

        # ## NARROW DOWN TO RELEVANT MATRICES ## #
        # Create segmentation strings
        compile_mats = all_matrices.copy()
        p_str = '_p' + str(p) + '_'
        m_str = '_m' + str(m) + '_'
        year_str = '_yr' + str(year) + '_'

        # Narrow down to matrices in this category
        compile_mats = [x for x in compile_mats if p_str in x]
        compile_mats = [x for x in compile_mats if m_str in x]
        compile_mats = [x for x in compile_mats if year_str in x]

        # Recursively narrow down further if needed
        base_fname = du.get_dist_name(
            trip_origin=trip_origin,
            matrix_format=matrix_format,
            year=str(year),
            purpose=str(p),
            mode=str(m)
        )
        out_path = os.path.join(export_dir, base_fname)

        _recursive_aggregate(
            candidates=compile_mats,
            segmentations=[segment_needed, ca_needed, tp_needed],
            segmentation_strs=[segment_str, ca_strs, tp_strs],
            import_dir=import_dir,
            export_path=out_path,
            round_dp=round_dp,
        )


def get_tour_proportion_seed_values(m: int,
                                    p: int,
                                    tp_split_path: str = None,
                                    phi_lookup_folder: str = None,
                                    phi_type: str = 'fhp_tp',
                                    aggregate_to_wday: bool = True,
                                    infill: float = 0.001
                                    ) -> np.ndarray:
    """
    TODO: write get_seed_values doc

    Parameters
    ----------
    m
    p
    tp_split_path
    phi_lookup_folder
    phi_type
    aggregate_to_wday
    infill

    Returns
    -------

    """
    # Init
    # TODO: Hardcoding this is bad!
    if phi_lookup_folder is None:
        phi_lookup_folder = 'Y:/NorMITs Demand/import/phi_factors'

    tp_split_path = r"Y:\NorMITs Demand\import\tfn_segment_production_params\hb_ave_time_split.csv"

    # Get appropriate phis and filter to purpose
    phi_factors = pa2od.get_time_period_splits(
        m,
        phi_type,
        aggregate_to_wday=aggregate_to_wday,
        lookup_folder=phi_lookup_folder
    )
    phi_factors = pa2od.simplify_time_period_splits(phi_factors)
    phi_factors = phi_factors[phi_factors['purpose_from_home'] == p]

    # Get the time period splits
    tp_splits = du.get_mean_tp_splits(
        tp_split_path=tp_split_path,
        p=p,
        aggregate_to_weekday=aggregate_to_wday,
        tp_as='int'
    )

    # Create the seed values
    valid_tps = phi_factors['time_from_home'].unique()
    seed_values = np.zeros((len(valid_tps), len(valid_tps)))

    for fh_idx, fh_tp in enumerate(valid_tps):
        # Extract the from home phi values
        fh_phi = phi_factors[phi_factors['time_from_home'] == fh_tp].copy()

        # Extract the from home time split
        time_split = tp_splits[fh_tp].values

        # Multiply phi factors by time splits
        for th_idx, th_tp in enumerate(valid_tps):
            # Extract the to home phi value
            th_phi = fh_phi[fh_phi['time_to_home'] == th_tp].copy()
            th_phi = th_phi['direction_factor'].values

            seed_values[fh_idx, th_idx] = th_phi * time_split

    # Check for really bad cases
    total_seed = seed_values.sum()
    if total_seed > 1.1 or total_seed < 0.9:
        raise ValueError("Something has gone wrong while generating tour "
                         "proportion seed values. The total seed value should "
                         "be 1, but we got %.2f." % total_seed)

    # infill as needed
    seed_values = np.where(seed_values <= 0, infill, seed_values)

    return seed_values


def get_vdm_tour_proportion_seed_values(m: int,
                                        uc: str,
                                        tp_split_path: str = None,
                                        phi_lookup_folder: str = None,
                                        phi_type: str = 'fhp_tp',
                                        aggregate_to_wday: bool = True,
                                        infill: float = 0.001
                                        ) -> np.ndarray:
    # TODO: Write get_vdm_tour_proportion_seed_values() docs
    # Init
    uc = du.validate_user_class(uc)

    # Get seed values
    seed_values_list = list()
    for p in consts.HB_USER_CLASS_PURPOSES[uc]:
        seed_values_list.append(get_tour_proportion_seed_values(
            m=m,
            p=p,
            tp_split_path=tp_split_path,
            phi_lookup_folder=phi_lookup_folder,
            phi_type=phi_type,
            aggregate_to_wday=aggregate_to_wday,
            infill=infill
        ))

    # No need to average
    n_purposes = len(seed_values_list)
    if n_purposes == 1:
        return seed_values_list[0]

    # Get the average of the seed values
    seed_values = reduce(lambda x, y: x + y, seed_values_list)
    seed_values = seed_values / n_purposes

    # Normalise array to sum=1
    seed_values /= seed_values.sum()

    # Check for really bad cases
    total_seed = seed_values.sum()
    if total_seed > 1.1 or total_seed < 0.9:
        raise ValueError("Something has gone wrong while generating tour "
                         "proportion seed values. The total seed value should "
                         "be 1, but we got %.2f." % total_seed)

    # infill as needed
    return np.where(seed_values <= 0, infill, seed_values)


def furness_tour_proportions(orig_vals,
                             dest_vals,
                             fh_mats,
                             th_mats,
                             seed_values,
                             tour_prop_name,
                             zone_translate_dir,
                             model_name,
                             tp_needed,
                             tour_prop_tol,
                             furness_tol,
                             furness_max_iters,
                             generate_tour_props=True
                             ):
    # TODO: Write furness_tour_proportions() docs()
    # ## INIT LOOP ## #
    # Create empty matrices for PA outputs
    pa_out_mats = dict()
    for tp in tp_needed:
        pa_out_mats[tp] = pd.DataFrame(0.0,
                                       index=orig_vals,
                                       columns=dest_vals)

    # Load the zone aggregation dictionaries for this model
    model2lad = du.get_zone_translation(
        import_dir=zone_translate_dir,
        from_zone=model_name,
        to_zone='lad'
    )
    model2tfn = du.get_zone_translation(
        import_dir=zone_translate_dir,
        from_zone=model_name,
        to_zone='tfn_sectors'
    )

    # Define the default value for the nested defaultdict
    def empty_tour_prop():
        return np.zeros((len(tp_needed), len(tp_needed)))

    # Use function to initialise defaultdicts
    lad_tour_props = defaultdict(lambda: defaultdict(empty_tour_prop))
    tfn_tour_props = defaultdict(lambda: defaultdict(empty_tour_prop))

    # TODO: optimise with numpy.
    # To preserve index/columns use i/j then:
    # cell_val = array[df.index[i], df.columns[j]]

    # Value Init
    zero_count = 0
    report = defaultdict(list)
    tour_proportions = defaultdict(dict)

    total = len(orig_vals) * len(dest_vals)
    desc = "Generating tour proportions for %s..." % tour_prop_name
    for orig, dest in tqdm(product(orig_vals, dest_vals), total=total, desc=desc):
        # Build the from_home vector
        fh_target = list()
        for tp in tp_needed:
            fh_target.append(fh_mats[tp].loc[orig, dest])
        fh_target = np.array(fh_target)
        pre_bal_fh = fh_target.copy()

        # Build the to_home vector
        th_target = list()
        for tp in tp_needed:
            th_target.append(th_mats[tp].loc[orig, dest])
        th_target = np.array(th_target)
        pre_bal_th = th_target.copy()

        # ## BALANCE FROM_HOME AND TO_HOME ## #
        # First use tp4 to bring both vector sums to average
        fh_th_avg = (fh_target.sum() + th_target.sum()) / 2
        fh_target[-1] = fh_th_avg - np.sum(fh_target[:-1])
        th_target[-1] = fh_th_avg - np.sum(th_target[:-1])

        # Correct for the resulting negative value
        seed_val = seed_values[-1][-1]
        if fh_target[-1] < 0:
            th_target[-1] -= (1 + seed_val) * fh_target[-1]
            fh_target[-1] *= -seed_val
        elif th_target[-1] < 0:
            fh_target[-1] -= (1 + seed_val) * th_target[-1]
            th_target[-1] *= -seed_val

        # ## STORE NEW PA VALS ## #
        for i, tp in enumerate(tp_needed):
            pa_out_mats[tp].loc[orig, dest] = fh_target[i]

        # ## Check for unbalanced tour proportions ## #
        if th_target.sum() != 0:
            temp_fh_target = fh_target.copy() / fh_target.sum()
            temp_th_target = th_target.copy() / th_target.sum()
        else:
            temp_fh_target = np.array([0] * len(tp_needed))
            temp_th_target = np.array([0] * len(tp_needed))

        # If tp4 is greater than the tolerance, this usually means the original
        # to_home and from_home targets were not balanced
        if(temp_th_target[-1] > tour_prop_tol
           or temp_fh_target[-1] > tour_prop_tol):
            report['tour_prop_fname'].append(tour_prop_name)
            report['OD_pair'].append((orig, dest))
            report['fh_before'].append(pre_bal_fh)
            report['fh_after'].append(fh_target)
            report['th_before'].append(pre_bal_th)
            report['th_after'].append(th_target)

        # ## FURNESS ## #
        if not generate_tour_props:
            # Return all zeroes if we don't need to furness
            furnessed_mat = np.zeros((len(tp_needed), len(tp_needed)))
        elif fh_target.sum() == 0 or th_target.sum() == 0:
            # Skip furness, create matrix of 0s instead
            zero_count += 1
            furnessed_mat = np.zeros((len(tp_needed), len(tp_needed)))

        else:
            furnessed_mat = furness.doubly_constrained_furness(
                seed_vals=seed_values,
                row_targets=fh_target,
                col_targets=th_target,
                tol=furness_tol,
                max_iters=furness_max_iters
            )

        # Store the tour proportions
        # furnessed_mat = furnessed_mat.astype('float64')
        tour_proportions[orig][dest] = furnessed_mat

        # TODO: Manually assign the missing aggregation zones
        # NOTE: Here we are assigning any zone we can't aggregate to
        # -1. These are usually point zones etc. and won't cause a problem
        # when we use these aggregated tour proportions later.
        # Making a note here in case it becomes a problem later

        # Calculate the lad aggregated tour proportions
        lad_orig = model2lad.get(orig, -1)
        lad_dest = model2lad.get(dest, -1)
        lad_tour_props[lad_orig][lad_dest] += furnessed_mat

        # Calculate the tfn aggregated tour proportions
        tfn_orig = model2tfn.get(orig, -1)
        tfn_dest = model2tfn.get(dest, -1)
        tfn_tour_props[tfn_orig][tfn_dest] += furnessed_mat

    return (
        tour_proportions,
        lad_tour_props,
        tfn_tour_props,
        pa_out_mats,
        report,
        zero_count
    )


def _tms_seg_tour_props_internal(od_import: str,
                                 tour_proportions_export: str,
                                 pa_export: str,
                                 model_name: str,
                                 trip_origin: str,
                                 year: int,
                                 p: int,
                                 m: int,
                                 seg: int,
                                 ca: int,
                                 tp_needed: List[int],
                                 tour_prop_tol: float,
                                 furness_tol: float,
                                 furness_max_iters: int,
                                 phi_lookup_folder: str,
                                 phi_type: str,
                                 aggregate_to_wday: bool,
                                 zone_translate_dir: str,
                                 generate_tour_props: bool,
                                 ) -> Tuple[str, int, float]:
    # TODO: Write _tms_seg_tour_props_internal() docs
    out_fname = du.get_dist_name(
        trip_origin=trip_origin,
        matrix_format='tour_proportions',
        year=str(year),
        purpose=str(p),
        mode=str(m),
        segment=str(seg),
        car_availability=str(ca),
        suffix='.pkl'
    )

    # Load the from_home matrices
    fh_mats = dict()
    for tp in tp_needed:
        dist_name = du.get_dist_name(
            trip_origin=trip_origin,
            matrix_format='od_from',
            year=str(year),
            purpose=str(p),
            mode=str(m),
            segment=str(seg),
            car_availability=str(ca),
            tp=str(tp),
            csv=True
        )
        fh_mats[tp] = pd.read_csv(os.path.join(od_import, dist_name),
                                  index_col=0)
        fh_mats[tp].columns = fh_mats[tp].columns.astype(int)
        fh_mats[tp].index = fh_mats[tp].index.astype(int)

    # Load the to_home matrices
    th_mats = dict()
    for tp in tp_needed:
        dist_name = du.get_dist_name(
            trip_origin=trip_origin,
            matrix_format='od_to',
            year=str(year),
            purpose=str(p),
            mode=str(m),
            segment=str(seg),
            car_availability=str(ca),
            tp=str(tp),
            csv=True
        )
        th_mats[tp] = pd.read_csv(os.path.join(od_import, dist_name),
                                  index_col=0).T
        th_mats[tp].columns = th_mats[tp].columns.astype(int)
        th_mats[tp].index = th_mats[tp].index.astype(int)

    # Make sure all matrices have the same OD pairs
    n_rows, n_cols = fh_mats[list(fh_mats.keys())[0]].shape
    for mat_dict in [fh_mats, th_mats]:
        for _, mat in mat_dict.items():
            if mat.shape != (n_rows, n_cols):
                raise ValueError(
                    "At least one of the loaded matrices does not match the "
                    "others. Expected a matrix of shape (%d, %d), got %s."
                    % (n_rows, n_cols, str(mat.shape))
                )

    # Get a list of the zone names for iterating
    orig_vals = list(fh_mats[list(fh_mats.keys())[0]].index.values)
    dest_vals = list(fh_mats[list(fh_mats.keys())[0]])

    # make sure they are integers
    orig_vals = [int(x) for x in orig_vals]
    dest_vals = [int(x) for x in dest_vals]

    # Get the seed values for this purpose
    seed_values = get_tour_proportion_seed_values(
        m=m,
        p=p,
        phi_lookup_folder=phi_lookup_folder,
        phi_type=phi_type,
        aggregate_to_wday=aggregate_to_wday,
        infill=0.001,
    )

    # ## CALL INNER FUNCTION ## #
    furness_return_vals = furness_tour_proportions(
        orig_vals=orig_vals,
        dest_vals=dest_vals,
        fh_mats=fh_mats,
        th_mats=th_mats,
        seed_values=seed_values,
        tour_prop_name=out_fname,
        zone_translate_dir=zone_translate_dir,
        model_name=model_name,
        tp_needed=tp_needed,
        tour_prop_tol=tour_prop_tol,
        furness_tol=furness_tol,
        furness_max_iters=furness_max_iters,
        generate_tour_props=generate_tour_props
    )

    # Split out the return values
    model_tour_props, lad_tour_props, tfn_tour_props = furness_return_vals[:3]
    pa_out_mats, report, zero_count = furness_return_vals[3:6]

    # Normalise all of the tour proportion matrices to 1
    for agg_tour_props in [model_tour_props, lad_tour_props, tfn_tour_props]:
        for key1, inner_dict in agg_tour_props.items():
            for key2, mat in inner_dict.items():
                # Avoid warning if 0
                if mat.sum() == 0:
                    continue
                agg_tour_props[key1][key2] = mat / mat.sum()

    # ## WRITE TO DISK ## #
    # Can just be normal dicts now - keeps pickle happy
    model_tour_props = du.defaultdict_to_regular(model_tour_props)
    lad_tour_props = du.defaultdict_to_regular(lad_tour_props)
    tfn_tour_props = du.defaultdict_to_regular(tfn_tour_props)

    if generate_tour_props:
        # Save the tour proportions for this segment (model_zone level)
        print('Writing outputs to disk for %s' % out_fname)
        out_path = os.path.join(tour_proportions_export, out_fname)
        with open(out_path, 'wb') as f:
            pickle.dump(model_tour_props, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Write the LAD tour proportions
        lad_out_fname = out_fname.replace('tour_proportions', 'lad_tour_proportions')
        out_path = os.path.join(tour_proportions_export, lad_out_fname)
        with open(out_path, 'wb') as f:
            pickle.dump(lad_tour_props, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Write the TfN Sector tour proportions
        tfn_out_fname = out_fname.replace('tour_proportions', 'tfn_tour_proportions')
        out_path = os.path.join(tour_proportions_export, tfn_out_fname)
        with open(out_path, 'wb') as f:
            pickle.dump(tfn_tour_props, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Write the error report to disk
        error_fname = out_fname.replace('tour_proportions', 'error_report')
        error_fname = error_fname.replace('.pkl', '.csv')
        out_path = os.path.join(tour_proportions_export, error_fname)
        pd.DataFrame(report).to_csv(out_path, index=False)

    # Write balanced PA to disk
    for tp, mat in pa_out_mats.items():
        dist_name = du.get_dist_name(
            trip_origin=trip_origin,
            matrix_format='pa',
            year=str(year),
            purpose=str(p),
            mode=str(m),
            segment=str(seg),
            car_availability=str(ca),
            tp=str(tp),
            csv=True
        )
        mat.to_csv(os.path.join(pa_export, dist_name))

    zero_percentage = (zero_count / float(n_rows * n_cols)) * 100
    return out_fname, zero_count, zero_percentage


def _tms_seg_tour_props(od_import: str,
                        tour_proportions_export: str,
                        zone_translate_dir: str,
                        pa_export: str,
                        model_name: str,
                        year: int = consts.BASE_YEAR,
                        p_needed: List[int] = consts.ALL_HB_P,
                        m_needed: List[int] = consts.MODES_NEEDED,
                        soc_needed: List[int] = None,
                        ns_needed: List[int] = None,
                        ca_needed: List[int] = None,
                        tp_needed: List[int] = consts.TIME_PERIODS,
                        tour_prop_tol: float = 0.5,
                        furness_tol: float = 1e-9,
                        furness_max_iters: int = 5000,
                        phi_lookup_folder: str = None,
                        phi_type: str = 'fhp',
                        aggregate_to_wday: bool = True,
                        generate_tour_props: bool = True,
                        process_count: int = -2
                        ) -> None:
    """
    TODO: Write _tms_seg_tour_props() docs
    """
    # Init
    soc_needed = [None] if soc_needed is None else soc_needed
    ns_needed = [None] if ns_needed is None else ns_needed
    ca_needed = [None] if ca_needed is None else ca_needed

    # Make sure all purposes are home based
    for p in p_needed:
        if p not in consts.ALL_HB_P:
            raise ValueError("Got purpose '%s' which is not a home based "
                             "purpose. generate_tour_proportions() cannot "
                             "handle nhb purposes." % str(p))
    trip_origin = 'hb'

    loop_generator = du.segmentation_loop_generator(
        p_list=p_needed,
        m_list=m_needed,
        soc_list=soc_needed,
        ns_list=ns_needed,
        ca_list=ca_needed
    )

    # ## MULTIPROCESS ## #
    unchanging_kwargs = {
        'od_import': od_import,
        'tour_proportions_export': tour_proportions_export,
        'zone_translate_dir': zone_translate_dir,
        'pa_export': pa_export,
        'model_name': model_name,
        'year': year,
        'trip_origin': trip_origin,
        'tp_needed': tp_needed,
        'tour_prop_tol': tour_prop_tol,
        'furness_tol': furness_tol,
        'furness_max_iters': furness_max_iters,
        'phi_lookup_folder': phi_lookup_folder,
        'phi_type': phi_type,
        'aggregate_to_wday': aggregate_to_wday,
        'generate_tour_props': generate_tour_props,
    }

    # Build a list of the changing arguments
    kwargs_list = list()
    for p, m, seg, ca in loop_generator:
        kwargs = unchanging_kwargs.copy()
        kwargs.update({
            'p': p,
            'm': m,
            'seg': seg,
            'ca': ca
        })
        kwargs_list.append(kwargs)

    # Multiprocess and write final matrices to disk
    zero_counts = multiprocessing.multiprocess(
        _tms_seg_tour_props_internal,
        kwargs=kwargs_list,
        process_count=process_count
    )

    # Output a log of the zero counts found
    header = ['tour_file', 'zero_count', 'percentage']
    out_name = "yr%d_tour_proportions_log.csv" % year
    out_path = os.path.join(tour_proportions_export, out_name)
    du.write_csv(header, zero_counts, out_path)

    # ## COPY OVER NHB MATRICES ## #
    if pa_export is not None:
        nhb_mats = [x for x in du.list_files(od_import) if
                    du.starts_with(x, 'nhb')]
        for fname in nhb_mats:
            pa_name = fname.replace('od', 'pa')
            du.copy_and_rename(
                src=os.path.join(od_import, fname),
                dst=os.path.join(pa_export, pa_name)
            )


def _vdm_seg_tour_props_internal(od_import: str,
                                 tour_proportions_export: str,
                                 pa_export: str,
                                 model_name: str,
                                 trip_origin: str,
                                 year: int,
                                 uc: str,
                                 m: int,
                                 ca: int,
                                 tp_needed: List[int],
                                 tour_prop_tol: float,
                                 furness_tol: float,
                                 furness_max_iters: int,
                                 phi_lookup_folder: str,
                                 phi_type: str,
                                 aggregate_to_wday: bool,
                                 zone_translate_dir: str,
                                 generate_tour_props: bool,
                                 ) -> Tuple[str, int, float]:
    """
    The internals of generate_tour_proportions().
    Used to implement multiprocessing.

    Returns
    -------
    tour_proportions_fname:
        THe name of the tour proportions output file

    zero_count:
        How many tour proportion matrices are full of 0s

    zero_percentage:
        The percentage of all tour proportion matrices that are full of 0s

    """
    # Figure out the output filename
    out_fname = du.get_vdm_dist_name(
        trip_origin=trip_origin,
        matrix_format='tour_proportions',
        year=str(year),
        user_class=str(uc),
        mode=str(m),
        ca=ca,
        suffix='.pkl'
    )

    # Load the from_home matrices
    fh_mats = dict()
    for tp in tp_needed:
        dist_name = du.get_vdm_dist_name(
            trip_origin=trip_origin,
            matrix_format='od_from',
            year=str(year),
            user_class=str(uc),
            mode=str(m),
            ca=ca,
            tp=str(tp),
            csv=True
        )
        fh_mats[tp] = pd.read_csv(os.path.join(od_import, dist_name),
                                  index_col=0)
        fh_mats[tp].columns = fh_mats[tp].columns.astype(int)
        fh_mats[tp].index = fh_mats[tp].index.astype(int)

    # Load the to_home matrices
    th_mats = dict()
    for tp in tp_needed:
        dist_name = du.get_vdm_dist_name(
            trip_origin=trip_origin,
            matrix_format='od_to',
            year=str(year),
            user_class=str(uc),
            mode=str(m),
            ca=ca,
            tp=str(tp),
            csv=True
        )
        th_mats[tp] = pd.read_csv(os.path.join(od_import, dist_name),
                                  index_col=0).T
        th_mats[tp].columns = th_mats[tp].columns.astype(int)
        th_mats[tp].index = th_mats[tp].index.astype(int)

    # Make sure all matrices have the same OD pairs
    n_rows, n_cols = fh_mats[list(fh_mats.keys())[0]].shape
    for mat_dict in [fh_mats, th_mats]:
        for _, mat in mat_dict.items():
            if mat.shape != (n_rows, n_cols):
                raise ValueError(
                    "At least one of the loaded matrices does not match the "
                    "others. Expected a matrix of shape (%d, %d), got %s."
                    % (n_rows, n_cols, str(mat.shape))
                )

    # Get a list of the zone names for iterating
    orig_vals = list(fh_mats[list(fh_mats.keys())[0]].index.values)
    dest_vals = list(fh_mats[list(fh_mats.keys())[0]])

    # make sure they are integers
    orig_vals = [int(x) for x in orig_vals]
    dest_vals = [int(x) for x in dest_vals]

    # Get the seed values
    seed_values = get_vdm_tour_proportion_seed_values(
        m=m,
        uc=uc,
        phi_lookup_folder=phi_lookup_folder,
        phi_type=phi_type,
        aggregate_to_wday=aggregate_to_wday,
        infill=0.001,
    )

    # ## CALL INNER FUNCTION ## #
    furness_return_vals = furness_tour_proportions(
        orig_vals=orig_vals,
        dest_vals=dest_vals,
        fh_mats=fh_mats,
        th_mats=th_mats,
        seed_values=seed_values,
        tour_prop_name=out_fname,
        zone_translate_dir=zone_translate_dir,
        model_name=model_name,
        tp_needed=tp_needed,
        tour_prop_tol=tour_prop_tol,
        furness_tol=furness_tol,
        furness_max_iters=furness_max_iters,
        generate_tour_props=generate_tour_props
    )

    # Split out the return values
    model_tour_props, lad_tour_props, tfn_tour_props = furness_return_vals[:3]
    pa_out_mats, report, zero_count = furness_return_vals[3:6]

    # Normalise all of the tour proportion matrices to 1
    for agg_tour_props in [model_tour_props, lad_tour_props, tfn_tour_props]:
        for key1, inner_dict in agg_tour_props.items():
            for key2, mat in inner_dict.items():
                # Avoid warning if 0
                if mat.sum() == 0:
                    continue
                agg_tour_props[key1][key2] = mat / mat.sum()

    # ## WRITE TO DISK ## #
    # Can just be normal dicts now - keeps pickle happy
    model_tour_props = du.defaultdict_to_regular(model_tour_props)
    lad_tour_props = du.defaultdict_to_regular(lad_tour_props)
    tfn_tour_props = du.defaultdict_to_regular(tfn_tour_props)

    if generate_tour_props:
        # Save the tour proportions for this segment (model_zone level)
        print('Writing outputs to disk for %s' % out_fname)
        out_path = os.path.join(tour_proportions_export, out_fname)
        with open(out_path, 'wb') as f:
            pickle.dump(model_tour_props, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Write the LAD tour proportions
        lad_out_fname = out_fname.replace('tour_proportions',
                                          'lad_tour_proportions')
        out_path = os.path.join(tour_proportions_export, lad_out_fname)
        with open(out_path, 'wb') as f:
            pickle.dump(lad_tour_props, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Write the TfN Sector tour proportions
        tfn_out_fname = out_fname.replace('tour_proportions',
                                          'tfn_tour_proportions')
        out_path = os.path.join(tour_proportions_export, tfn_out_fname)
        with open(out_path, 'wb') as f:
            pickle.dump(tfn_tour_props, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Write the error report to disk
        error_fname = out_fname.replace('tour_proportions', 'error_report')
        error_fname = error_fname.replace('.pkl', '.csv')
        out_path = os.path.join(tour_proportions_export, error_fname)
        pd.DataFrame(report).to_csv(out_path, index=False)

    # Write balanced PA to disk
    for tp, mat in pa_out_mats.items():
        dist_name = du.get_vdm_dist_name(
            trip_origin=trip_origin,
            matrix_format='pa',
            year=str(year),
            user_class=str(uc),
            mode=str(m),
            ca=ca,
            tp=str(tp),
            csv=True
        )
        mat.to_csv(os.path.join(pa_export, dist_name))

    zero_percentage = (zero_count / float(n_rows * n_cols)) * 100
    return out_fname, zero_count, zero_percentage


def _vdm_seg_tour_props(od_import: str,
                        tour_proportions_export: str,
                        zone_translate_dir: str,
                        pa_export: str,
                        model_name: str,
                        year: int = consts.BASE_YEAR,
                        to_needed: List[str] = consts.VDM_TRIP_ORIGINS,
                        uc_needed: List[str] = consts.USER_CLASSES,
                        m_needed: List[int] = consts.MODES_NEEDED,
                        ca_needed: List[int] = None,
                        tp_needed: List[int] = consts.TIME_PERIODS,
                        tour_prop_tol: float = 0.5,
                        furness_tol: float = 1e-9,
                        furness_max_iters: int = 5000,
                        phi_lookup_folder: str = None,
                        phi_type: str = 'fhp',
                        aggregate_to_wday: bool = True,
                        generate_tour_props: bool = True,
                        process_count: int = -2
                        ) -> None:
    """
    TODO: Write _vdm_seg_tour_props() docs
    """
    # Init
    ca_needed = [None] if ca_needed is None else ca_needed

    # ## MULTIPROCESS ## #
    unchanging_kwargs = {
        'od_import': od_import,
        'tour_proportions_export': tour_proportions_export,
        'zone_translate_dir': zone_translate_dir,
        'pa_export': pa_export,
        'model_name': model_name,
        'year': year,
        'trip_origin': 'hb',
        'tp_needed': tp_needed,
        'tour_prop_tol': tour_prop_tol,
        'furness_tol': furness_tol,
        'furness_max_iters': furness_max_iters,
        'phi_lookup_folder': phi_lookup_folder,
        'phi_type': phi_type,
        'aggregate_to_wday': aggregate_to_wday,
        'generate_tour_props': generate_tour_props
    }

    # Build a list of the changing arguments
    kwargs_list = list()
    for uc, m, ca in product(uc_needed, m_needed, ca_needed):
        kwargs = unchanging_kwargs.copy()
        kwargs.update({
            'uc': uc,
            'm': m,
            'ca': ca
        })
        kwargs_list.append(kwargs)

    # Multiprocess and write final matrices to disk
    zero_counts = multiprocessing.multiprocess(
        _vdm_seg_tour_props_internal,
        kwargs=kwargs_list,
        process_count=process_count
    )

    # Output a log of the zero counts found
    header = ['tour_file', 'zero_count', 'percentage']
    out_name = "yr%d_vdm_tour_proportions_log.csv" % year
    out_path = os.path.join(tour_proportions_export, out_name)
    du.write_csv(header, zero_counts, out_path)

    if 'nhb' not in to_needed:
        return

    # ## COPY OVER NHB MATRICES ## #
    if pa_export is not None:
        nhb_mats = [x for x in du.list_files(od_import) if du.starts_with(x, 'nhb')]
        for fname in nhb_mats:
            pa_name = fname.replace('od', 'pa')
            du.copy_and_rename(
                src=os.path.join(od_import, fname),
                dst=os.path.join(pa_export, pa_name)
            )


def generate_tour_proportions(od_import: str,
                              tour_proportions_export: str,
                              zone_translate_dir: str,
                              pa_export: str,
                              model_name: str,
                              seg_level: str,
                              seg_params: Dict[str, Any],
                              year: int = consts.BASE_YEAR,
                              tour_prop_tol: float = 0.5,
                              furness_tol: float = 1e-9,
                              furness_max_iters: int = 5000,
                              phi_lookup_folder: str = None,
                              phi_type: str = 'fhp',
                              aggregate_to_wday: bool = True,
                              generate_tour_props: bool = True,
                              process_count: int = -2
                              ) -> None:
    """
    Generates the 4x4 matrix of tour proportions for every OD pair for all
    given segmentations.

    Uses the "od_from" and "od_to" matrices in od_import to generate target
    from-home and to-home trip proportions. They are then furnessed for each
    OD pair to produce the tour factors. Tour factors are then written to disk
    as a .pkl at tour_proportions_export, in the format
    tour_proportions[O][D] = np.array. np.array is the tour proportions for
    that OD pair. .pkl files are named depending on their segmentation.
    Optionally outputs converted PA matrices.

    Parameters
    ----------
    od_import:
        Where to find the od_from and od_to matrices.

    tour_proportions_export:
        Where to write the tour proportions as a .pkl

    zone_translate_dir:
        Where to find the zone translation files from the model zoning system
        to the aggregated LAD nad TfN zoning systems.

    pa_export:
        Where to export the converted pa_matrices. If left as None,
        no pa matrices are written.

    seg_level:
        The level of segmentation of the matrices in od_import. This should
        be one of the values in efs_constants.SEG_LEVELS.

    year:
        Which year to generate the tour proportions for. Usually the base year

    p_needed:
        Which purposes to use. Usually only home based.

    m_needed:
        Which mode to use.

    soc_needed:
        Which skill levels to segment by.

    ns_needed:
        Which income levels to segment by.

    ca_needed:
        Which car_availabilities to segment by.

    tp_needed:
        Which time periods to use. Usually 1-4.

    tour_prop_tol:
        Used for reporting unbalanced from_home and to_home targets when
        generating tour proportions. If the difference between the total of
        the two vectors is greater than this value, it will be reported in
        an output report.

    furness_tol:
        What tolerance to use during the furness.
        See furness_process.doubly_constrained_furness() for more information.

    furness_max_iters:
        Max number of iterations for the furness.
        See furness_process.doubly_constrained_furness() for more information.

    phi_lookup_folder:
        The directory to find the phi lookups - these are used to generate the
        seed values for each purpose.

    phi_type:
        The type of phi lookups to use. This argument is passed directly to
        pa2od.get_time_period_splits().

    aggregate_to_wday:
        Whether to aggregate the loaded phi lookups to weekday or not. This
        argument is passed directly to pa2od.get_time_period_splits().

    process_count:
        How many processes to use during multiprocessing. Usually set to
        number_of_cpus - 1.

    Returns
    -------
    None

    """
    # TODO: Update generate_tour_proportions() docs
    # Init
    seg_level = du.validate_seg_level(seg_level)

    # Call the correct mid-level function to deal with the segmentation
    if seg_level == 'tms':
        segmentation_fn = _tms_seg_tour_props
    elif seg_level == 'vdm':
        segmentation_fn = _vdm_seg_tour_props
    else:
        raise NotImplementedError(
            "'%s' is a valid segmentation level, however, we do not have a "
            "mid-level function to deal with it at the moment."
            % seg_level
        )

    segmentation_fn(
        od_import=od_import,
        tour_proportions_export=tour_proportions_export,
        zone_translate_dir=zone_translate_dir,
        pa_export=pa_export,
        model_name=model_name,
        year=year,
        tour_prop_tol=tour_prop_tol,
        furness_tol=furness_tol,
        furness_max_iters=furness_max_iters,
        phi_lookup_folder=phi_lookup_folder,
        phi_type=phi_type,
        aggregate_to_wday=aggregate_to_wday,
        generate_tour_props=generate_tour_props,
        process_count=process_count,
        **seg_params
    )


def build_compile_params(import_dir: str,
                         export_dir: str,
                         matrix_format: str,
                         years_needed: Iterable[int],
                         m_needed: List[int] = consts.MODES_NEEDED,
                         ca_needed: Iterable[int] = None,
                         tp_needed: Iterable[int] = None,
                         split_hb_nhb: bool = False,
                         split_od_from_to: bool = False,
                         output_headers: List[str] = None,
                         output_format: str = 'wide',
                         output_fname: str = None
                         ) -> str:
    """
    Create a compile_params file to be used with compile_od().
    In the future this should also work with compile_pa().

    Parameters
    ----------
    import_dir:
        Directory containing all of the matrices to be compiled.

    export_dir:
        Directory to output the compile parameters.

    matrix_format:
        Format of the input matrices. Usually 'pa' or 'od'.

    years_needed:
        Which years compile parameters should be generated for.

    m_needed:
        Which mode compile parameters should be generated for.

    ca_needed:
        Which car availabilities compile parameters should be generated for.
        If None, car availabilities are not used

    tp_needed:
        Which time periods compile parameters should be generated for.

    split_hb_nhb:
        Whether the home based and non-home based matrices should be compiled
        together or not. If True, separate hb and nhb compiled matrices are
        created.

    split_od_from_to:
        Whether the od_from and od_to matrices should be compiled together or
        not. If True, separate od_from and od_to compiled matrices are created.

    output_headers:
        Optional. Use if custom output headers are needed. by default the
        following headers are used:
        ['distribution_name', 'compilation', 'format']

    output_format:
        What format the compiled matrices should be output as. Usually either
        'wide' or 'long'.

    output_fname:
        The name to give to the output file. If left as None,
        du.get_compile_params_name(matrix_format, year) is used to generate
        the output name.

    Returns
    -------
    compile_params_path:
        The path to the compile parameters produced
    """
    # Error checking
    if len(m_needed) > 1:
        raise ValueError("Matrix compilation can only handle one mode at a "
                         "time. Received %d modes" % len(m_needed))
    mode = m_needed[0]

    if split_od_from_to and matrix_format != 'od':
        raise ValueError("Can only split od_from and od_to matrices if the "
                         "matrix format is 'od'.")

    # Init
    ca_needed = [None] if ca_needed is None else ca_needed
    tp_needed = [None] if tp_needed is None else tp_needed
    to_needed = [None] if not split_hb_nhb else ['hb', 'nhb']
    od_from_to = [None] if not split_od_from_to else ['od_from', 'od_to']
    all_od_matrices = du.list_files(import_dir)
    out_lines = list()

    if output_headers is None:
        output_headers = ['distribution_name', 'compilation', 'format']

    for year in years_needed:
        for user_class, purposes in consts.USER_CLASS_PURPOSES.items():
            for ca, tp, to, od_ft in product(ca_needed, tp_needed, to_needed, od_from_to):
                # Init
                compile_mats = all_od_matrices.copy()

                # include _ before and after to avoid clashes
                ps = ['_p' + str(x) + '_' for x in purposes]
                mode_str = '_m' + str(mode) + '_'
                year_str = '_yr' + str(year) + '_'

                # Narrow down to matrices for this compilation
                compile_mats = [x for x in compile_mats if year_str in x]
                compile_mats = [x for x in compile_mats if du.is_in_string(ps, x)]
                compile_mats = [x for x in compile_mats if mode_str in x]

                # Narrow down further if we're using ca
                if ca is not None:
                    ca_str = '_ca' + str(ca) + '_'
                    compile_mats = [x for x in compile_mats if ca_str in x]

                # Narrow down again if we're using tp
                if tp is not None:
                    tp_str = '_tp' + str(tp)
                    compile_mats = [x for x in compile_mats if tp_str in x]

                # Narrow down again if we're using hb/nhb separation
                if to is not None:
                    compile_mats = [x for x in compile_mats if du.starts_with(x, to)]

                # Narrow down again if we're using od_from/od_to separation
                # From/To splits are a bit more complicated :(
                if od_ft is not None:
                    # Don't split for nhb trips
                    if to == 'nhb':
                        matrix_format = 'od'

                        # Avoid repeats by skipping od_from
                        if od_ft == 'od_from':
                            continue

                    else:
                        # If we get here, it is safe to filter for hb trips
                        od_ft_str = '_' + str(od_ft) + '_'
                        compile_mats = [x for x in compile_mats if od_ft_str in x]
                        matrix_format = str(od_ft)

                # Build the final output name
                compiled_mat_name = du.get_compiled_matrix_name(
                    matrix_format,
                    user_class,
                    str(year),
                    trip_origin=to,
                    mode=str(mode),
                    ca=ca,
                    tp=str(tp),
                    csv=True
                )

                # Add lines to output
                for mat_name in compile_mats:
                    line_parts = (mat_name, compiled_mat_name, output_format)
                    out_lines.append(line_parts)

        # Write outputs for this year
        if output_fname is None:
            output_fname = du.get_compile_params_name(matrix_format, str(year))
        out_path = os.path.join(export_dir, output_fname)
        du.write_csv(output_headers, out_lines, out_path)

        return out_path


def build_24hr_vdm_mats(import_dir: str,
                        export_dir: str,
                        matrix_format: str,
                        to_needed: str,
                        years_needed: List[str],
                        uc_needed: List[str] = consts.USER_CLASSES,
                        m_needed: List[int] = consts.MODES_NEEDED,
                        ca_needed: List[int] = None,
                        tp_needed: List[int] = consts.TIME_PERIODS,
                        split_factors_path: str = None
                        ) -> None:
    # TODO: Write build_24hr_vdm_mats() docs
    # Init
    ca_needed = [None] if ca_needed is None else ca_needed

    # Go through all segmentations, for all years
    for year in years_needed:
        loop_generator = du.vdm_segment_loop_generator(
            to_list=to_needed,
            uc_list=uc_needed,
            m_list=m_needed,
            ca_list=ca_needed
        )

        for to, uc, m, ca in loop_generator:
            # Figure out output name to tell user
            output_dist_name = du.get_vdm_dist_name(
                trip_origin=to,
                matrix_format=matrix_format,
                year=str(year),
                user_class=str(uc),
                mode=str(m),
                ca=ca,
                csv=True
            )
            print("Generating output matrix %s..." % output_dist_name)

            # Read in all time period matrices
            tp_mats = list()
            for tp in tp_needed:
                dist_name = du.get_vdm_dist_name(
                    trip_origin=to,
                    matrix_format=matrix_format,
                    year=str(year),
                    user_class=str(uc),
                    mode=str(m),
                    ca=ca,
                    tp=str(tp),
                    csv=True
                )
                dist_path = os.path.join(import_dir, dist_name)
                tp_mats.append(pd.read_csv(dist_path, index_col=0))

            # Check all the input matrices have the same columns and index
            col_ref = tp_mats[0].columns
            idx_ref = tp_mats[0].index
            for i, mat in enumerate(tp_mats):
                if len(mat.columns.difference(col_ref)) > 0:
                    raise ValueError(
                        "tp matrix %s columns do not match the "
                        "others." % str(tp_needed[i]))

                if len(mat.index.difference(idx_ref)) > 0:
                    raise ValueError(
                        "tp matrix %s index does not match the "
                        "others." % str(tp_needed[i]))

            # Combine all matrices together
            full_mat = reduce(lambda x, y: x.add(y, fill_value=0), tp_mats)

            # Output to file
            full_mat.to_csv(os.path.join(export_dir, output_dist_name))

            # Only Calculate the splitting factors if we need to
            if split_factors_path is None:
                continue

            # TODO: Move into output_converter
            # ## SPLITTING FACTORS ## #
            # Init
            splitting_factors = defaultdict(list)

            # Make sure rows and columns are ints
            full_mat.columns = full_mat.columns.astype(int)
            full_mat.index = full_mat.index.astype(int)

            orig_vals = [int(x) for x in tp_mats[0].index.values]
            dest_vals = [int(x) for x in list(tp_mats[0])]
            desc = 'Generating splitting factors'
            for tp, tp_mat in enumerate(tqdm(tp_mats, desc=desc), 1):
                # Make sure rows and columns are ints
                tp_mat.columns = tp_mat.columns.astype(int)
                tp_mat.index = tp_mat.index.astype(int)

                for orig, dest in product(orig_vals, dest_vals):
                    if full_mat.loc[orig, dest] == 0:
                        tp_split = 0.0
                    else:
                        tp_split = tp_mat.loc[orig, dest] / full_mat.loc[orig, dest]

                    splitting_factors['Origin'].append(orig)
                    splitting_factors['Destination'].append(dest)
                    splitting_factors['TimePeriod'].append(tp)
                    splitting_factors['Factor'].append(tp_split)

            # Write to disk
            out_name = output_dist_name.replace('od', 'split_factors')
            out_path = os.path.join(split_factors_path, out_name)
            pd.DataFrame(splitting_factors).to_csv(out_path, index=False)


def build_24hr_mats(import_dir: str,
                    export_dir: str,
                    matrix_format: str,
                    years_needed: List[str],
                    p_needed: List[int] = consts.ALL_HB_P,
                    m_needed: List[int] = consts.MODES_NEEDED,
                    soc_needed: List[int] = None,
                    ns_needed: List[int] = None,
                    ca_needed: List[int] = None,
                    tp_needed: List[int] = consts.TIME_PERIODS
                    ) -> None:
    """
    Compiles time period split matrices int import_dir into 24hr Matrices,
    saving them back to export dir

    Parameters
    ----------
    import_dir:
        Directory to find the time period split matrices.

    export_dir:
        Directory to store the created 24hr matrices.

    matrix_format:
        Format of the matrices to convert. Usually either 'pa' or 'od'.

    years_needed:
        Which years of matrices in import_dir to convert.

    p_needed:
        Which purposes of matrices in import_dir to convert.

    m_needed:
        Which modes of matrices in import_dir to convert.

    soc_needed:
        Which skill levels of matrices in import_dir to convert. If left as
        None, this segmentation is ignored.

    ns_needed:
        Which income levels of matrices in import_dir to convert. If left as
        None, this segmentation is ignored.

    ca_needed:
        Which car availabilities matrices in import_dir to convert. If left as
        None, this segmentation is ignored.

    tp_needed:
        Which time period matrices in import_dir to combine to get to
        24hr matrices.

    Returns
    -------
    None
    """
    # Init
    soc_needed = [None] if soc_needed is None else soc_needed
    ns_needed = [None] if ns_needed is None else ns_needed
    ca_needed = [None] if ca_needed is None else ca_needed

    for year in years_needed:
        loop_generator = du.segmentation_loop_generator(
            p_list=p_needed,
            m_list=m_needed,
            soc_list=soc_needed,
            ns_list=ns_needed,
            ca_list=ca_needed
        )

        for p, m, seg, ca in loop_generator:
            # Figure out trip origin
            if p in consts.ALL_HB_P:
                trip_origin = 'hb'
            elif p in consts.ALL_NHB_P:
                trip_origin = 'nhb'
            else:
                raise ValueError("'%s' is not a valid purpose. Don't know if it "
                                 "is home based or non-home based.")

            # Figure out output name to tell user
            output_dist_name = du.get_dist_name(
                trip_origin=trip_origin,
                matrix_format=matrix_format,
                year=str(year),
                purpose=str(p),
                mode=str(m),
                segment=str(seg),
                car_availability=str(ca),
                csv=True
            )
            print("Generating output matrix %s..." % output_dist_name)

            # Read in all time period matrices
            tp_mats = list()
            for tp in tp_needed:
                dist_name = du.get_dist_name(
                    trip_origin=trip_origin,
                    matrix_format=matrix_format,
                    year=str(year),
                    purpose=str(p),
                    mode=str(m),
                    segment=str(seg),
                    car_availability=str(ca),
                    tp=str(tp),
                    csv=True
                )
                dist_path = os.path.join(import_dir, dist_name)
                tp_mats.append(pd.read_csv(dist_path, index_col=0))

            # Check all the input matrices have the same columns and index
            col_ref = tp_mats[0].columns
            idx_ref = tp_mats[0].index
            for i, mat in enumerate(tp_mats):
                if len(mat.columns.difference(col_ref)) > 0:
                    raise ValueError("tp matrix %s columns do not match the "
                                     "others." % str(tp_needed[i]))

                if len(mat.index.difference(idx_ref)) > 0:
                    raise ValueError("tp matrix %s index does not match the "
                                     "others." % str(tp_needed[i]))

            # Combine all matrices together
            full_mat = reduce(lambda x, y: x.add(y, fill_value=0), tp_mats)

            # Output to file
            full_mat.to_csv(os.path.join(export_dir, output_dist_name))


def copy_nhb_matrices(import_dir: str,
                      export_dir: str,
                      replace_pa_with_od: bool = False,
                      replace_od_with_pa: bool = False,
                      ) -> None:
    """
    Copies NHB matrices from import dir to export dir.

    Optionally replaces the pa in the matrix names with od, or the od in
    matrix names in pa.

    Parameters
    ----------
    import_dir:
        Path to the directory where the nhb matrices to copy exist.
    
    export_dir:
        Path to the directory to copy the nhb matrices to.
    
    replace_pa_with_od:
        Whether to replace the '_pa_' in the matrix names with '_od_'.

    replace_od_with_pa:
        Whether to replace the '_od_' in the matrix names with '_pa_'.

    Returns
    -------
    None
    """
    # Find the .csv nhb mats
    mats = du.list_files(import_dir)
    mats = [x for x in mats if '.csv' in x]
    nhb_mats = [x for x in mats if du.starts_with(x, 'nhb')]

    # Copy them over without a rename
    for mat_fname in nhb_mats:
        # Deal with the simple case
        if not replace_pa_with_od and not replace_od_with_pa:
            out_mat_fname = mat_fname

        # Try to rename if needed
        elif replace_pa_with_od:
            if '_pa_' not in mat_fname:
                raise ValueError(
                    "Cannot find '_pa_' in '%s' to replace." % mat_fname
                )
            out_mat_fname = mat_fname.replace('_pa_', '_od_')

        elif replace_od_with_pa:
            if '_od_' not in mat_fname:
                raise ValueError(
                    "Cannot find '_od_' in '%s' to replace." % mat_fname
                )
            out_mat_fname = mat_fname.replace('_od_', '_pa_')
        else:
            raise ValueError(
                "This shouldn't happen! Somehow replace_od_with_pa and "
                "replace_pa_with_od are set and not set?!"
            )

        # Copy over and rename
        du.copy_and_rename(
            src=os.path.join(import_dir, mat_fname),
            dst=os.path.join(export_dir, out_mat_fname),
        )
        

def compile_matrices(mat_import: str,
                     mat_export: str,
                     compile_params_path: str,
                     round_dp: int = consts.DEFAULT_ROUNDING,
                     build_factor_pickle: bool = False,
                     factor_pickle_path: str = None,
                     factors_fname: str = 'od_compilation_factors.pickle'
                     ) -> None:
    """
    Compiles the matrices in mat_import, writes to mat_export

    Parameters
    ----------
    mat_import:
        Path to the directory containing the matrices to compile

    mat_export:
        Path to the directory to output the compiled matrices

    compile_params_path:
        Path to the compile params, as produced by build_compile_params()

    round_dp:
            The number of decimal places to round the output values to.
            Uses consts.DEFAULT_ROUNDING by default.

    build_factor_pickle:
        If True, a dictionary of factors that can be used to decompile the
        compiled matrices will be created. This will be in the format of:
        factors[compiled_matrix][import_matrix] = np.array(factors)

    factor_pickle_path:
        Where to export the decompile factors. This should be a path to a
        directory, not including the filename. If left as None, mat_export
        will be used in place.

    factors_fname:
        The filename to give to the exported decompile factors when writing to
        disk

    Returns
    -------
    None
    """
    # TODO: Add in some Audit checks and return the report
    if not os.path.isdir(mat_import):
        raise IOError("Matrix import path '%s' does not exist." % mat_import)

    if not os.path.isdir(mat_export):
        raise IOError("Matrix export path '%s' does not exist." % mat_export)

    # Init
    compile_params = pd.read_csv(compile_params_path)
    compiled_names = compile_params['compilation'].unique()
    factor_pickle_path = mat_export if factor_pickle_path is None else factor_pickle_path

    # Need to get the size of the output matrices
    check_mat_name = compile_params.loc[0, 'distribution_name']
    check_mat = pd.read_csv(os.path.join(mat_import, check_mat_name), index_col=0)
    n_rows = len(check_mat.index)
    n_cols = len(check_mat.columns)

    # Define the default value for the nested defaultdict
    def empty_factors():
        return np.zeros(n_rows, n_cols)

    # Use function to initialise defaultdict
    decompile_factors = defaultdict(lambda: defaultdict(empty_factors))

    desc = 'Compiling Matrices'
    for comp_name in tqdm(compiled_names, desc=desc):
        # ## COMPILE THE MATRICES ## #
        # Get the input matrices
        mask = (compile_params['compilation'] == comp_name)
        subset = compile_params[mask].copy()
        input_mat_names = subset['distribution_name'].unique()

        # Read in all the matrices
        in_mats = list()
        for mat_name in input_mat_names:
            in_path = os.path.join(mat_import, mat_name)
            in_mats.append(pd.read_csv(in_path, index_col=0))

        # Combine all matrices together
        full_mat = reduce(lambda x, y: x.add(y, fill_value=0), in_mats)

        # Output to file
        output_path = os.path.join(mat_export, comp_name)
        full_mat.round(decimals=round_dp).to_csv(output_path)

        # Go to the next iteration if we don't need the factors
        if not build_factor_pickle:
            continue

        # ## CALCULATE THE DECOMPILE FACTORS ## #
        for part_mat, mat_name in zip(in_mats, input_mat_names):
            # Avoid divide by zero
            part_mat = np.where(part_mat == 0, 0.0001, part_mat)
            decompile_factors[comp_name][mat_name] = part_mat / full_mat

    # Write factors to disk if we made them
    if build_factor_pickle:
        print('Writing decompile factors to disk - might take a while...')
        decompile_factors = du.defaultdict_to_regular(decompile_factors)

        out_path = os.path.join(factor_pickle_path, factors_fname)
        with open(out_path, 'wb') as f:
            pickle.dump(decompile_factors, f, protocol=pickle.HIGHEST_PROTOCOL)


def matrices_to_vector(mat_import_dir: pathlib.Path,
                       years_needed: List[str],
                       ) -> pd.DataFrame:
    # TODO: Write matrices_to_vector() docs
    # Init





    # def load_seed_dists(mat_folder: str,
    #                     segments_needed: List[str],
    #                     base_year: str,
    #                     zone_column: str,
    #                     trip_type: str = "productions",
    #                     ) -> pd.DataFrame:
    # 
    #     # Define the columns needed
    #     group_cols = [zone_column] + segments_needed
    #     required_cols = group_cols + [base_year]
    # 
    #     # Get the list of available files in the seed dist folder
    #     files = du.parse_mat_output(
    #         os.listdir(mat_folder),
    #         mat_type="pa"
    #     )
    #     # Filter to just HB matrices
    #     hb_files = files.loc[files["trip_origin"] == "hb"]
    #     # Define dataframe to store the observed trip ends
    #     all_obs = pd.DataFrame()
    # 
    #     iterator = tqdm(
    #         hb_files.to_dict(orient="records"),
    #         desc=f"Loading Base Observed {trip_type}"
    #     )
    #     # Loop through each matrix in the path and add to the overall dataframe
    #     for row in iterator:
    #         file_name = row.pop("file")
    #         file_path = os.path.join(mat_folder, file_name)
    # 
    #         obs = pd.read_csv(file_path, index_col=0)
    #         # Sum along columns for productions and rows for attractions
    #         if trip_type == "productions":
    #             obs = obs.sum(axis=1)
    #         elif trip_type == "attractions":
    #             obs = obs.sum(axis=0)
    #         else:
    #             raise ValueError("Invalid Trip Type supplied")
    # 
    #         # Set column names
    #         obs = obs.reset_index()
    #         obs.columns = [zone_column, base_year]
    #         obs[zone_column] = obs[zone_column].astype("int")
    # 
    #         # Extract segments from the file names
    #         for segment in segments_needed:
    #             obs[segment] = row[segment]
    # 
    #         # Add to the overall dataframe
    #         if all_obs.empty:
    #             all_obs = obs
    #         else:
    #             all_obs = all_obs.append(obs)
    # 
    #     # Change data types for all integer columns
    #     for col in ["p", "ca"]:
    #         if col in all_obs.columns:
    #             all_obs[col] = all_obs[col].astype("int")
    # 
    #     # Finally group and sum the dataframe
    #     all_obs = all_obs.groupby(
    #         group_cols,
    #         as_index=False
    #     )[base_year].sum()
    # 
    #     return all_obs[required_cols]
