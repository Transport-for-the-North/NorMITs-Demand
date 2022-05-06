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
import operator
import itertools

from typing import Any
from typing import List
from typing import Dict
from typing import Tuple
from typing import Union
from typing import Iterable

import functools
from itertools import product
from collections import defaultdict

# Third Party
import numpy as np
import pandas as pd
from tqdm import tqdm

# Local imports
import normits_demand as nd
from normits_demand import constants as consts
from normits_demand import efs_constants as efs_consts
from normits_demand.utils import general as du
from normits_demand.utils import file_ops
from normits_demand.utils import compress
from normits_demand.utils import pandas_utils as pd_utils

from normits_demand.matrices import pa_to_od as pa2od
from normits_demand.matrices import utils as mat_utils
from normits_demand.matrices import compilation as mat_comp
from normits_demand.distribution import furness
from normits_demand.concurrency import multiprocessing
from normits_demand.validation import checks

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

    if in_fnames == list():
        raise nd.NormitsDemandError(
            "Couldn't find any matrices to aggregate up to create %s!"
            % os.path.basename(export_path)
        )

    for fname in in_fnames:
        mat = file_ops.read_df(os.path.join(import_dir, fname), index_col=0)

        # Build a matching df of 0s if not done yet
        if aggregated_mat is None:
            aggregated_mat = pd.DataFrame(0,
                                          index=mat.index,
                                          columns=mat.columns
                                          )
        aggregated_mat += mat

    # Write new matrix out
    file_ops.write_df(aggregated_mat.round(decimals=round_dp), export_path)
    print("Aggregated matrix written: %s" % os.path.basename(export_path))


def _recursive_aggregate(candidates: List[str],
                         segmentations: List[List[int]],
                         segmentation_strs: List[List[str]],
                         import_dir: str,
                         export_path: str,
                         compress_out: bool = False,
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

    compress_out:
        Whether to compress the write to disk or not.

    round_dp:
        The number of decimal places to round the output values to.
        Uses efs_consts.DEFAULT_ROUNDING by default.
    """

    # ## EXIT CONDITION ## #
    if len(segmentations) == 1:
        # Un-nest
        segmentations = segmentations[0]
        segmentation_strs = segmentation_strs[0]

        # Determine the ftype
        ftype = '.csv'
        if compress_out:
            ftype = consts.COMPRESSION_SUFFIX

        if du.is_none_like(segmentations):
            # Aggregate remaining candidates
            _aggregate(
                import_dir=import_dir,
                in_fnames=candidates,
                export_path=export_path + ftype,
                round_dp=round_dp,
            )
        else:
            # Loop through and aggregate
            for segment, seg_str in zip(segmentations, segmentation_strs):
                _aggregate(
                    import_dir=import_dir,
                    in_fnames=[x for x in candidates.copy() if seg_str in x],
                    export_path=export_path + seg_str + ftype,
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
            compress_out=compress_out,
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
                compress_out=compress_out,
                round_dp=round_dp,
            )


def _aggregate_matrices_internal(year,
                                 p,
                                 m,
                                 all_matrices,
                                 import_dir,
                                 export_dir,
                                 trip_origin,
                                 matrix_format,
                                 segment_needed,
                                 ca_needed,
                                 tp_needed,
                                 segment_str,
                                 ca_strs,
                                 tp_strs,
                                 compress_out,
                                 round_dp,
                                 ):
    """
    The internal function of aggregate_matrices(). Used for multiprocessing.
    """
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
        compress_out=compress_out,
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
                       compress_out: bool = False,
                       round_dp: int = consts.DEFAULT_ROUNDING,
                       process_count: int = consts.PROCESS_COUNT,
                       ):
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

    compress_out:
        Whether to compress the output to disk or not.

    round_dp:
        The number of decimal places to round the output values to.
        Uses efs_consts.DEFAULT_ROUNDING by default.

    process_count:
        The number of processes to use when multiprocessing. See
        concurrency.multiprocess() to see what the vales mean.
        Set to 0 to not use multiprocessing.

    Returns
    -------
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
    # ## MULTIPROCESS ## #
    unchanging_kwargs = {
        'all_matrices': all_matrices,
        'import_dir': import_dir,
        'export_dir': export_dir,
        'trip_origin': trip_origin,
        'matrix_format': matrix_format,
        'ca_needed': ca_needed,
        'tp_needed': tp_needed,
        'ca_strs': ca_strs,
        'tp_strs': tp_strs,
        'compress_out': compress_out,
        'round_dp': round_dp,
    }

    # Build the kwarg list
    kwarg_list = list()
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

        kwargs = unchanging_kwargs.copy()
        kwargs.update({
            'year': year,
            'p': p,
            'm': m,
            'segment_needed': segment_needed,
            'segment_str': segment_str,
        })
        kwarg_list.append(kwargs)

    # Run
    multiprocessing.multiprocess(
        fn=_aggregate_matrices_internal,
        kwargs=kwarg_list,
        process_count=process_count,
        # process_count=0,

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
        phi_lookup_folder = 'I:/NorMITs Demand/import/phi_factors'

    tp_split_path = r"I:\NorMITs Demand\import\tfn_segment_production_params\hb_ave_time_split.csv"

    # Get appropriate phis and filter to purpose
    phi_factors = pa2od.get_time_period_splits(
        m,
        phi_type,
        aggregate_to_wday=aggregate_to_wday,
        lookup_folder=phi_lookup_folder
    )
    phi_factors = pa2od.simplify_phi_factors(phi_factors)
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
    seed_values = functools.reduce(lambda x, y: x + y, seed_values_list)
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
    desc = "Generating tour props for %s..." % tour_prop_name
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
            furnessed_mat, *_ = furness.doubly_constrained_furness(
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
        matrix_format='tms_tour_proportions',
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
                        year: int = efs_consts.BASE_YEAR,
                        p_needed: List[int] = consts.ALL_HB_P,
                        m_needed: List[int] = efs_consts.MODES_NEEDED,
                        soc_needed: List[int] = None,
                        ns_needed: List[int] = None,
                        ca_needed: List[int] = None,
                        tp_needed: List[int] = efs_consts.TIME_PERIODS,
                        tour_prop_tol: float = 0.5,
                        furness_tol: float = 1e-9,
                        furness_max_iters: int = 5000,
                        phi_lookup_folder: str = None,
                        phi_type: str = 'fhp',
                        aggregate_to_wday: bool = True,
                        generate_tour_props: bool = True,
                        process_count: int = consts.PROCESS_COUNT,
                        ) -> None:
    """
    TODO: Write _tms_seg_tour_props() docs
    """
    # Init
    soc_needed = [None] if soc_needed is None else soc_needed
    ns_needed = [None] if ns_needed is None else ns_needed
    ca_needed = [None] if ca_needed is None else ca_needed

    # Split into HB and NHB purposes
    hb_p_needed = list()
    nhb_p_needed = list()
    for p in p_needed:
        if p in consts.ALL_HB_P:
            hb_p_needed.append(p)
        elif p in consts.ALL_NHB_P:
            nhb_p_needed.append(p)
        else:
            raise ValueError(
                "Got purpose '%s' which is not a valid purpose."
                % str(p)
            )

    # Build our loop generator
    trip_origin = 'hb'
    loop_generator = du.segmentation_loop_generator(
        p_list=hb_p_needed,
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
    out_name = "yr%d_tms_tour_proportions_log.csv" % year
    out_path = os.path.join(tour_proportions_export, out_name)
    du.write_csv(header, zero_counts, out_path)

    # ## COPY OVER NHB MATRICES ## #
    if pa_export is not None:
        mat_list = du.list_files(od_import)
        nhb_mats = [x for x in mat_list if du.starts_with(x, 'nhb')]

        # Filter down to nhb purposes
        ps = ['_p%s_' % x for x in nhb_p_needed]
        nhb_mats = [x for x in nhb_mats if du.is_in_string(ps, x)]

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
        matrix_format='vdm_tour_proportions',
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
                        year: int = efs_consts.BASE_YEAR,
                        to_needed: List[str] = efs_consts.VDM_TRIP_ORIGINS,
                        uc_needed: List[str] = efs_consts.USER_CLASSES,
                        m_needed: List[int] = efs_consts.MODES_NEEDED,
                        ca_needed: List[int] = None,
                        tp_needed: List[int] = efs_consts.TIME_PERIODS,
                        tour_prop_tol: float = 0.5,
                        furness_tol: float = 1e-9,
                        furness_max_iters: int = 5000,
                        phi_lookup_folder: str = None,
                        phi_type: str = 'fhp',
                        aggregate_to_wday: bool = True,
                        generate_tour_props: bool = True,
                        process_count: int = consts.PROCESS_COUNT,
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
    out_name = "yr%s_vdm_tour_proportions_log.csv" % year
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
                              year: int = efs_consts.BASE_YEAR,
                              tour_prop_tol: float = 0.5,
                              furness_tol: float = 1e-9,
                              furness_max_iters: int = 5000,
                              phi_lookup_folder: str = None,
                              phi_type: str = 'fhp',
                              aggregate_to_wday: bool = True,
                              generate_tour_props: bool = True,
                              process_count: int = consts.PROCESS_COUNT,
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


def build_norms_vdm_compile_params(import_dir: str,
                                   export_dir: str,
                                   matrix_format: str,
                                   segmentation_aggregation: nd.SegmentAggregationDict,
                                   years_needed: Iterable[int],
                                   m_needed: List[int],
                                   tp_needed: Iterable[int] = None,
                                   output_headers: List[str] = None,
                                   output_format: str = 'wide',
                                   params_suffix: str = None,
                                   csv_out: bool = True,
                                   compress_out: bool = False,
                                   ) -> List[str]:
    # TODO: Write build_norms_vdm_compile_params() docs
    # Error checking
    if len(m_needed) > 1:
        raise ValueError("Matrix compilation can only handle one mode at a "
                         "time. Received %d modes" % len(m_needed))
    mode = m_needed[0]

    # Init
    tp_needed = [None] if tp_needed is None else tp_needed
    all_matrices = du.list_files(import_dir)

    if output_headers is None:
        output_headers = ['distribution_name', 'compilation', 'format']

    # Generate a different compilation dict for each year
    out_paths = list()
    for year in years_needed:
        out_lines = list()

        # Build the iterator
        iterator = itertools.product(
            segmentation_aggregation.items(),
            tp_needed
        )

        # Find all the mats to aggregate into vdm_mat
        for (vdm_mat_name, seg_dict), tp in iterator:
            # Init
            compile_mats = all_matrices.copy()
            purposes = consts.USER_CLASS_PURPOSES[seg_dict['uc']]

            # ## FILTER DOWN TO USER CLASS ## #
            # include _ before and after to avoid clashes
            ps = ['_p' + str(x) + '_' for x in purposes]
            mode_str = '_m' + str(mode) + '_'
            year_str = '_yr' + str(year) + '_'

            # Narrow down to matrices for this compilation
            compile_mats = [x for x in compile_mats if year_str in x]
            compile_mats = [x for x in compile_mats if du.is_in_string(ps, x)]
            compile_mats = [x for x in compile_mats if mode_str in x]

            # Filter by time period if needed
            if tp is not None:
                tp_str = '_tp' + str(tp)
                compile_mats = [x for x in compile_mats if tp_str in x]

            # ## FILTER DOWN TO SUB USER CLASS ## #
            # We're keeping the mats which contain any item in the list
            filtered = list()
            for to in seg_dict['to']:
                filtered += [x for x in compile_mats if du.starts_with(x, to)]
            compile_mats = filtered.copy()

            filtered = list()
            for ca in seg_dict['ca']:
                ca_str = '_ca%s' % str(ca)
                filtered += [x for x in compile_mats if ca_str in x]
            compile_mats = filtered.copy()

            # ## BUILD THE COMPILATION PARAMS ## #
            # Build the final output name
            compiled_mat_name = du.get_compiled_matrix_name(
                matrix_format,
                seg_dict['uc'],
                str(year),
                trip_origin=None,
                mode=str(mode),
                ca=None,
                tp=str(tp),
                suffix='_%s' % vdm_mat_name,
                csv=csv_out,
                compress=compress_out,
            )

            # Add lines to output
            for mat_name in compile_mats:
                line_parts = (mat_name, compiled_mat_name, output_format)
                out_lines.append(line_parts)

        # Write outputs for this year
        output_fname = du.get_compile_params_name(
            matrix_format=matrix_format,
            year=str(year),
            suffix=params_suffix,
        )
        out_path = os.path.join(export_dir, output_fname)
        du.write_csv(output_headers, out_lines, out_path)
        out_paths.append(out_path)

    return out_paths


def build_norms_compile_params(import_dir: str,
                               export_dir: str,
                               matrix_format: str,
                               years_needed: Iterable[int],
                               m_needed: List[int] = efs_consts.MODES_NEEDED,
                               tp_needed: Iterable[int] = None,
                               output_headers: List[str] = None,
                               output_format: str = 'wide',
                               output_fname: str = None
                               ) -> List[str]:
    # TODO: Write build_norms_compile_params() docs
    # Error checking
    if len(m_needed) > 1:
        raise ValueError("Matrix compilation can only handle one mode at a "
                         "time. Received %d modes" % len(m_needed))
    mode = m_needed[0]

    # Init
    tp_needed = [None] if tp_needed is None else tp_needed
    all_od_matrices = du.list_files(import_dir)
    out_paths = list()

    if output_headers is None:
        output_headers = ['distribution_name', 'compilation', 'format']

    for year in years_needed:
        out_lines = list()

        # Build the iterator
        iterator = itertools.product(
            consts.USER_CLASS_PURPOSES.items(),
            tp_needed
        )

        for (user_class, purposes), tp in iterator:
            for sub_uc, seg_dict in efs_consts.NORMS_SUB_USER_CLASS_SEG.items():
                # Init
                compile_mats = all_od_matrices.copy()

                # ## FILTER DOWN TO USER CLASS ## #
                # include _ before and after to avoid clashes
                ps = ['_p' + str(x) + '_' for x in purposes]
                mode_str = '_m' + str(mode) + '_'
                year_str = '_yr' + str(year) + '_'

                # Narrow down to matrices for this compilation
                compile_mats = [x for x in compile_mats if year_str in x]
                compile_mats = [x for x in compile_mats if du.is_in_string(ps, x)]
                compile_mats = [x for x in compile_mats if mode_str in x]

                # Filter by time period if needed
                if tp is not None:
                    tp_str = '_tp' + str(tp)
                    compile_mats = [x for x in compile_mats if tp_str in x]

                # ## FILTER DOWN TO SUB USER CLASS ## #
                # We're keeping the mats which contain any item in the list
                filtered = list()
                for to in seg_dict['to']:
                    filtered += [x for x in compile_mats if du.starts_with(x, to)]
                compile_mats = filtered.copy()

                filtered = list()
                for ca in seg_dict['ca']:
                    ca_str = '_ca%s' % str(ca)
                    filtered += [x for x in compile_mats if ca_str in x]
                compile_mats = filtered.copy()

                filtered = list()
                # Split into trip origins, we can only do this for hb mats
                hb_mats = [x for x in compile_mats if du.starts_with(x, 'hb')]
                nhb_mats = [x for x in compile_mats if du.starts_with(x, 'nhb')]
                for od_ft in seg_dict['od_ft']:
                    # Filter to just the from/to we need
                    od_ft_str = "_%s" % od_ft
                    filtered += [x for x in hb_mats if od_ft_str in x]
                # Stick everything back together
                compile_mats = filtered.copy() + nhb_mats

                # ## BUILD THE COMPILATION PARAMS ## #
                # Build the final output name
                compiled_mat_name = du.get_compiled_matrix_name(
                    matrix_format,
                    user_class,
                    str(year),
                    trip_origin=None,
                    mode=str(mode),
                    ca=None,
                    tp=str(tp),
                    suffix='_%s' % sub_uc,
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
        out_paths.append(out_path)

    return out_paths


def build_compile_params(import_dir: str,
                         export_dir: str,
                         matrix_format: str,
                         years_needed: Iterable[int],
                         m_needed: List[int] = efs_consts.MODES_NEEDED,
                         ca_needed: Iterable[int] = None,
                         tp_needed: Iterable[int] = None,
                         split_hb_nhb: bool = False,
                         split_od_from_to: bool = False,
                         output_headers: List[str] = None,
                         output_format: str = 'wide',
                         output_fname: str = None
                         ) -> List[str]:

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
    out_paths = list()

    if output_headers is None:
        output_headers = ['distribution_name', 'compilation', 'format']

    for year in years_needed:
        out_lines = list()
        for user_class, purposes in consts.USER_CLASS_PURPOSES.items():
            for ca, tp, to, od_ft in product(ca_needed, tp_needed, to_needed, od_from_to):
                # Init
                compile_mats = all_od_matrices.copy()

                # include _ or . before and after to avoid clashes
                ps = ['_p' + str(x) + '_' for x in purposes]
                mode_strs = ['_m' + str(mode) + x for x in ['_', '.']]
                year_strs = ['_yr' + str(year) + x for x in ['_', '.']]

                # Narrow down to matrices for this compilation
                compile_mats = [x for x in compile_mats if du.is_in_string(year_strs, x)]
                compile_mats = [x for x in compile_mats if du.is_in_string(ps, x)]
                compile_mats = [x for x in compile_mats if du.is_in_string(mode_strs, x)]

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
            fname = du.get_compile_params_name(matrix_format, str(year))
        else:
            fname = output_fname
        out_path = os.path.join(export_dir, fname)
        du.write_csv(output_headers, out_lines, out_path)
        out_paths.append(out_path)

    return out_paths


def build_24hr_vdm_mats(import_dir: str,
                        export_dir: str,
                        matrix_format: str,
                        to_needed: str,
                        years_needed: List[str],
                        uc_needed: List[str] = efs_consts.USER_CLASSES,
                        m_needed: List[int] = efs_consts.MODES_NEEDED,
                        ca_needed: List[int] = None,
                        tp_needed: List[int] = efs_consts.TIME_PERIODS,
                        split_factors_path: str = None,
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
            full_mat = functools.reduce(lambda x, y: x.add(y, fill_value=0), tp_mats)

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
                    year_needed: List[str],
                    p_needed: List[int],
                    m_needed: List[int],
                    soc_needed: List[int] = None,
                    ns_needed: List[int] = None,
                    ca_needed: List[int] = None,
                    tp_needed: List[int] = efs_consts.TIME_PERIODS,
                    splitting_factors_export: nd.PathLike = None,
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

    year_needed:
        Which year of matrices in import_dir to convert.

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

    splitting_factors_export:
        A path to a file to write the time period splitting factors to get
        back to the tp matrices produced here. If left as None, no spltting
        factors are generated.

    Returns
    -------
    None
    """
    # Init
    soc_needed = [None] if soc_needed is None else soc_needed
    ns_needed = [None] if ns_needed is None else ns_needed
    ca_needed = [None] if ca_needed is None else ca_needed

    # Need to get the size of the output matrices
    check_to = 'hb' if p_needed[0] in consts.ALL_HB_P else 'nhb'
    check_mat_name = du.get_dist_name(
        trip_origin=check_to,
        matrix_format=matrix_format,
        year=str(year_needed),
        purpose=str(p_needed[0]),
        mode=str(m_needed[0]),
        segment=str(soc_needed[0]),
        car_availability=str(ca_needed[0]),
        tp=str(tp_needed[0]),
        csv=True
    )
    check_mat = file_ops.read_df(os.path.join(import_dir, check_mat_name), index_col=0)
    n_rows = len(check_mat.index)
    n_cols = len(check_mat.columns)

    # Define the default value for the nested defaultdict
    def empty_factors():
        return np.zeros(n_rows, n_cols)

    # Use function to initialise defaultdict
    decompile_factors = defaultdict(lambda: defaultdict(empty_factors))

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
            year=str(year_needed),
            purpose=str(p),
            mode=str(m),
            segment=str(seg),
            car_availability=str(ca),
            csv=True
        )
        print("Generating output matrix %s..." % output_dist_name)

        # Read in all time period matrices
        tp_mats = list()
        tp_mat_names = list()
        for tp in tp_needed:
            dist_name = du.get_dist_name(
                trip_origin=trip_origin,
                matrix_format=matrix_format,
                year=str(year_needed),
                purpose=str(p),
                mode=str(m),
                segment=str(seg),
                car_availability=str(ca),
                tp=str(tp),
                csv=True
            )
            dist_path = os.path.join(import_dir, dist_name)
            tp_mats.append(pd.read_csv(dist_path, index_col=0))
            tp_mat_names.append(dist_name)

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
        full_mat = functools.reduce(lambda x, y: x.add(y, fill_value=0), tp_mats)

        # Output to file
        full_mat.to_csv(os.path.join(export_dir, output_dist_name))

        if not splitting_factors_export:
            continue

        # ## CALCULATE THE DECOMPILE FACTORS ## #
        for part_mat, mat_name in zip(tp_mats, tp_mat_names):
            # Avoid divide by zero
            full_mat = np.where(full_mat == 0, 0.0001, full_mat)
            decompile_factors[output_dist_name][mat_name] = part_mat / full_mat

    # Write factors to disk if we made them
    if splitting_factors_export:
        print('Writing tp splitting factors to disk - might take a while...')
        decompile_factors = du.defaultdict_to_regular(decompile_factors)
        return compress.write_out(decompile_factors, splitting_factors_export)


def match_distribution_params(candidate_dist_params: Dict[Any, Dict[str, int]],
                              in_dist_params: Dict[str, int],
                              match_seg_params: List[str],
                              ) -> List[Any]:
    """
    Tries to match the distribution parameters of dict_a and dict_b.

    Parameters
    ----------
    candidate_dist_params:
        A dictionary of candidates to match to dist_params. The keys should 
        be the values to return if the value matches dist_params.
    
    in_dist_params:
        A dictionary of distribution parameters to try and match to.
        THe keys are segmentation names, the values thier values.
    
    match_seg_params:
        The segmentation parameters (keys of in_dist_params) that need to match
        in the candidate_dist_params to be called a match

    Returns
    -------
    matching_keys:
        Returns a list of the keys in candidate_dist_params where the 
        distribution parameters match to dist_params on match_seg_params.

    """
    # Init
    candidate_factor_keys = list()

    # Check each candidate
    for factor_key, dist_params in candidate_dist_params.items():
        matching_keys = True

        # Check the keys that need to match
        for seg_key in match_seg_params:
            # If no match, we can exit early
            if dist_params[seg_key] != in_dist_params[seg_key]:
                matching_keys = False
                break
        if matching_keys:
            candidate_factor_keys.append(factor_key)
            
    return candidate_factor_keys


def _nhb_tp_split_via_factors_internal(import_dir,
                                       export_dir,
                                       mat_24hr_fname,
                                       tp_dict,
                                       trip_origin,
                                       export_matrix_format,
                                       compress_out,
                                       ):
    print("Splitting %s..." % mat_24hr_fname)
    # Read in the matrix
    mat_path = os.path.join(import_dir, mat_24hr_fname)
    mat_24hr = file_ops.read_df(mat_path, index_col=0, find_similar=True)

    # Apply the splitting factors and write out
    for tp, factors in tp_dict.items():
        tp_mat = pd.DataFrame(
            data=mat_24hr.values * factors.values,
            columns=mat_24hr.columns,
            index=mat_24hr.index,
        )

        # Figure out the output_fname
        in_params = du.fname_to_calib_params(mat_24hr_fname)
        in_params['tp'] = tp
        tp_mat_fname = du.calib_params_to_dist_name(
            trip_origin=trip_origin,
            matrix_format=export_matrix_format,
            calib_params=in_params,
            csv=(not compress_out),
            compressed=compress_out,
        )
        out_path = os.path.join(export_dir, tp_mat_fname)
        file_ops.write_df(tp_mat, out_path)


def nhb_tp_split_via_factors(import_dir: nd.PathLike,
                             export_dir: nd.PathLike,
                             import_matrix_format: str,
                             export_matrix_format: str,
                             tour_proportions_dir: nd.PathLike,
                             model_name: str,
                             years_needed: List[str],
                             p_needed: List[int],
                             m_needed: List[int],
                             soc_needed: List[int] = None,
                             ns_needed: List[int] = None,
                             ca_needed: List[int] = None,
                             process_count: int = consts.PROCESS_COUNT,
                             compress_out: bool = False,
                             verbose: bool = True,
                             ) -> None:
    # TODO(BT): Write nhb_tp_split_via_factors() docs
    # Init
    trip_origin = 'nhb'

    # Make sure we only have NHB purposes
    for p in p_needed:
        if p not in consts.ALL_NHB_P:
            raise ValueError(
                "Can only run nhb_tp_split_via_factors() on non-home based "
                "purposes only. Got purpose %s" % p
            )

    # Read in the splitting factors
    du.print_w_toggle("Reading in the splitting factors...", verbose=verbose)
    fname = consts.POSTME_TP_SPLIT_FACTORS_FNAME
    factor_path = os.path.join(tour_proportions_dir, fname)
    splitting_factors = file_ops.read_pickle(factor_path)

    # Figure out the level of segmentation we are working at
    check_key = list(splitting_factors.keys())[0]
    split_factor_seg_keys = list(du.fname_to_calib_params(check_key).keys())

    # Break the splitting factors down into distribution params
    du.print_w_toggle("Checking the splitting factors...", verbose=verbose)
    factor_params = dict()
    for key in splitting_factors.keys():
        dist_params = du.fname_to_calib_params(key)

        # Make sure it's the same segmentation
        if list(dist_params.keys()) != split_factor_seg_keys:
            raise nd.NormitsDemandError(
                "The segmentation of all the read in splitting factors does "
                "not match. Expected '%s'\nGot '%s' from %s"
                % (split_factor_seg_keys, list(dist_params.keys()), key)
            )

        factor_params[key] = dist_params

    # Remove the year key, as we are year independent
    split_factor_seg_keys = du.list_safe_remove(split_factor_seg_keys, ['yr'])

    # Split year by year
    for year in years_needed:
        matrix_to_split_factors = dict()

        # ## BUILD DICTIONARY OF MATRICES TO TP SPLITTING FACTORS ## #
        # Build the loop generator
        loop_generator = du.cp_segmentation_loop_generator(
            p_list=p_needed,
            m_list=m_needed,
            soc_list=soc_needed,
            ns_list=ns_needed,
            ca_list=ca_needed
        )

        # Find the splitting factors that are best for each matrix
        for in_dist_params in loop_generator:
            in_dist_params['yr'] = year

            # Figure out input matrix name
            input_dist_name = du.calib_params_to_dist_name(
                trip_origin=trip_origin,
                matrix_format=import_matrix_format,
                calib_params=in_dist_params,
                csv=True,
            )

            # Find the best match from the splitting factors
            candidate_factor_keys = match_distribution_params(
                candidate_dist_params=factor_params,
                in_dist_params=in_dist_params,
                match_seg_params=split_factor_seg_keys,
            )

            # If we have more than 1 candidate, we have a problem
            if len(candidate_factor_keys) > 1:
                raise nd.NormitsDemandError(
                    "More than one candidate splitting factor found when "
                    "trying to split %s. Found candidate splitting factors "
                    "from: %s"
                    % (input_dist_name, candidate_factor_keys)
                )
            factor_key = candidate_factor_keys[0]

            # Build a dictionary of time periods to splitting factors
            tp_dict = dict()
            for tp_mat_name, tp_factors in splitting_factors[factor_key].items():
                tp = du.fname_to_calib_params(tp_mat_name)['tp']
                tp_dict[tp] = tp_factors

            # Finally, assign splitting factors to matrix
            matrix_to_split_factors[input_dist_name] = tp_dict

        # ## APPLY SPLITTING FACTORS, WRITE TO DISK ## #
        unchanging_kwargs = {
            'import_dir': import_dir,
            'export_dir': export_dir,
            'trip_origin': trip_origin,
            'export_matrix_format': export_matrix_format,
            'compress_out': compress_out,
        }

        kwarg_list = list()
        for mat_24hr_fname, tp_dict in matrix_to_split_factors.items():
            kwargs = unchanging_kwargs.copy()
            kwargs['mat_24hr_fname'] = mat_24hr_fname
            kwargs['tp_dict'] = tp_dict
            kwarg_list.append(kwargs)

        multiprocessing.multiprocess(
            fn=_nhb_tp_split_via_factors_internal,
            kwargs=kwarg_list,
            process_count=process_count,
        )


def copy_nhb_matrices(import_dir: str,
                      export_dir: str,
                      replace_pa_with_od: bool = False,
                      replace_od_with_pa: bool = False,
                      pa_matrix_desc: str = 'pa',
                      od_matrix_desc: str = 'od',
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
        Whether to replace pa_matrix_desc in the matrix names with od_matrix_desc.

    replace_od_with_pa:
        Whether to replace od_matrix_desc in the matrix names with pa_matrix_desc.

    pa_matrix_desc:
        The name used to describe the pa matrices. Usually just 'pa', but
        will sometimes be 'synthetic_pa' when dealing with TMS synthetic
        matrices.

    od_matrix_desc:
        The name used to describe the od matrices. Usually just 'od', but
        will sometimes be 'synthetic_od' when dealing with TMS synthetic
        matrices.

    Returns
    -------
    None
    """
    # Find the .csv nhb mats
    mats = du.list_files(import_dir)
    nhb_mats = [x for x in mats if du.starts_with(x, 'nhb')]

    pa_nm = '_%s_' % pa_matrix_desc
    od_nm = '_%s_' % od_matrix_desc

    # Copy them over without a rename
    for mat_fname in nhb_mats:
        # Deal with the simple case
        if not replace_pa_with_od and not replace_od_with_pa:
            out_mat_fname = mat_fname

        # Try to rename if needed
        elif replace_pa_with_od:
            if pa_nm not in mat_fname:
                raise ValueError(
                    "Cannot find '%s' in '%s' to replace."
                    % (pa_nm, mat_fname)
                )
            out_mat_fname = mat_fname.replace(pa_nm, od_nm)

        elif replace_od_with_pa:
            if od_nm not in mat_fname:
                raise ValueError(
                    "Cannot find '%s' in '%s' to replace."
                    % (od_nm, mat_fname)
                )
            out_mat_fname = mat_fname.replace(od_nm, pa_nm)
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


def _compile_matrices_internal(mat_import,
                               mat_export,
                               compile_params,
                               comp_name,
                               round_dp,
                               factor_pickle_path,
                               avoid_zero_splits,
                               ) -> None:
    """
    The internal function of compile_matrices
    """
    # ## COMPILE THE MATRICES ## #
    # Get the input matrices
    mask = (compile_params['compilation'] == comp_name)
    subset = compile_params[mask].copy()
    input_mat_names = subset['distribution_name'].unique()

    # Read in all the matrices
    in_mats = list()
    for mat_name in input_mat_names:
        in_path = os.path.join(mat_import, mat_name)
        df = file_ops.read_df(in_path, index_col=0)
        df.columns = df.columns.astype(df.index.dtype)
        in_mats.append(df)

    # Combine all matrices together
    full_mat = functools.reduce(operator.add, in_mats)

    # Output to file
    output_path = os.path.join(mat_export, comp_name)
    full_mat = full_mat.round(decimals=round_dp)
    file_ops.write_df(full_mat, output_path)

    # Go to the next iteration if we don't need the factors
    if factor_pickle_path is None:
        return None

    # ## CALCULATE THE DECOMPILE FACTORS ## #
    # Infill all zeroes with a small number - ensures no 0 splits
    if avoid_zero_splits:
        in_mats = [x.where(x != 0, 1e-8) for x in in_mats]
        full_mat = functools.reduce(operator.add, in_mats)

    decompile_factors = dict()
    for part_mat, mat_name in zip(in_mats, input_mat_names):
        # Avoid divide by zero
        full_mat = np.where(full_mat == 0, 0.0001, full_mat)
        # decompile_factors[comp_name][mat_name] = part_mat / full_mat
        decompile_factors[mat_name] = part_mat / full_mat

    return decompile_factors


def compile_matrices(mat_import: str,
                     mat_export: str,
                     compile_params_path: str,
                     factor_pickle_path: str = None,
                     round_dp: int = consts.DEFAULT_ROUNDING,
                     factors_fname: str = 'od_compilation_factors.pickle',
                     avoid_zero_splits: bool = False,
                     process_count: int = consts.PROCESS_COUNT,
                     ) -> nd.PathLike:
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
            Uses efs_consts.DEFAULT_ROUNDING by default.

    factor_pickle_path:
        Where to export the decompile factors. This should be a path to a
        directory, not including the filename. If left as None, mat_export
        will be used in place.

    factors_fname:
        The filename to give to the exported decompile factors when writing to
        disk

    avoid_zero_splits:
        If set to True, then no zero splits will appear in the splitting
        factors. Where there would have been zero splits, this will be
        replaced with even splits across inputs.

    Returns
    -------
    None
    """
    # TODO: Add in some Audit checks and return the report
    if not os.path.isdir(mat_import):
        raise IOError("Matrix import path '%s' does not exist." % mat_import)

    if not os.path.isdir(mat_export):
        raise IOError("Matrix export path '%s' does not exist." % mat_export)

    if factor_pickle_path is not None:
        if pathlib.Path(factor_pickle_path).suffix == '':
            print(
                "WARNING: No filename was given for the pickle factors. "
                "Defaulting to od_compilation_factors.pickle, but this is "
                "deprecated and will be removed in future!"
            )
            factor_pickle_path = os.path.join(factor_pickle_path, factors_fname)

    # Init
    compile_params = pd.read_csv(compile_params_path)
    compiled_names = compile_params['compilation'].unique()

    # Need to get the size of the output matrices
    check_mat_name = compile_params.loc[0, 'distribution_name']
    check_mat = file_ops.read_df(os.path.join(mat_import, check_mat_name), index_col=0)
    n_rows = len(check_mat.index)
    n_cols = len(check_mat.columns)

    # Define the default value for the nested defaultdict
    def empty_factors():
        return np.zeros(n_rows, n_cols)

    # Use function to initialise defaultdict
    decompile_factors = defaultdict(lambda: defaultdict(empty_factors))

    # ## MP Matrix compilation ## #
    unchanging_kwargs = {
        'mat_import': mat_import,
        'mat_export': mat_export,
        'compile_params': compile_params,
        'round_dp': round_dp,
        'factor_pickle_path': factor_pickle_path,
        'avoid_zero_splits': avoid_zero_splits,
    }

    pbar_kwargs = {
        'desc': 'Compiling Matrices',
        'unit': 'matrices',
        'colour': 'cyan',
    }

    kwarg_list = list()
    for comp_name in compiled_names:
        kwargs = unchanging_kwargs.copy()
        kwargs['comp_name'] = comp_name
        kwarg_list.append(kwargs)

    # Compile all the matrices and get the decompile factors back
    factors = multiprocessing.multiprocess(
        fn=_compile_matrices_internal,
        kwargs=kwarg_list,
        process_count=process_count,
        in_order=True,
        pbar_kwargs=pbar_kwargs,
    )

    # Assign the return values
    decompile_factors = {c: f for c, f in zip(compiled_names, factors)}

    # Write factors to disk if we made them
    if factor_pickle_path is not None:
        print('Writing decompile factors to disk - might take a while...')
        decompile_factors = du.defaultdict_to_regular(decompile_factors)
        return compress.write_out(decompile_factors, factor_pickle_path)


def load_matrix_from_disk(mat_import_dir: pathlib.Path,
                          segment_dict: Dict[str, Any],
                          completed_segments: List[Dict[str, Any]] = None,
                          ) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Returns a full OD or PA matrix loaded from disk

    Parameters
    ----------
    mat_import_dir:
        The path to a directory containing the pa or od matrices.

    segment_dict:
        A segment dictionary where the keys are segmentation types, and the
        values are segmentation values. du.fname_to_calib_params() returns
        a version of a segment_dict

    completed_segments:
        A list of segment_dicts without 'matrix_format' keys. This is used
        to check whether a segment has already been loaded in. Useful for
        making sure we don't load in od_to and od_from as two separate
        od matrices.
        NOTE: This argument is not used or needed if only loading PA or
        OD matrices

    Returns
    -------
    matrix:
        The loaded matrix in a pandas dataframe.

    completed_segments:
        An updated version of the completed_segments passed in. If left as
        None, then this will return an empty list intead. This return can be
        safely ignored if only loading PA or OD matrices
    """
    # Init
    trip_origin = checks.validate_trip_origin(segment_dict['trip_origin'])
    matrix_format = checks.validate_matrix_format(segment_dict['matrix_format'])

    completed_segments = list() if completed_segments is None else completed_segments

    # Simple case - we have full PA or OD matrix. Read in and return
    if segment_dict['matrix_format'] in ['pa', 'od']:
        mat_fname = du.calib_params_to_dist_name(
            trip_origin=trip_origin,
            matrix_format=matrix_format,
            calib_params=segment_dict,
            csv=True,
        )
        mat_path = os.path.join(mat_import_dir, mat_fname)
        return pd.read_csv(mat_path, index_col=0), completed_segments

    # Lets make sure we only carry on with the right formats
    if segment_dict['matrix_format'] not in ['od_from', 'od_to']:
        raise ValueError(
            "%s Seems to be a valid matrix format, but I don't know how to "
            "deal with it." % str(segment_dict['matrix_format'])
        )

    # Can only get here if we have od_to or od_from

    # Need to remove the matrix format so od_to and od_from will match
    temp_seg_dict = segment_dict.copy()
    del temp_seg_dict['matrix_format']

    # Don't carry on if we've already done this segment
    if temp_seg_dict in completed_segments:
        return pd.DataFrame(), completed_segments

    # Mark seg_dict as complete
    completed_segments.append(temp_seg_dict)

    # Load in both matrices and stick together
    od_to_fname = du.calib_params_to_dist_name(
        trip_origin=trip_origin,
        matrix_format='od_to',
        calib_params=segment_dict,
        csv=True,
    )
    od_to_path = os.path.join(mat_import_dir, od_to_fname)
    od_to = pd.read_csv(od_to_path, index_col=0)

    od_from_fname = du.calib_params_to_dist_name(
        trip_origin=trip_origin,
        matrix_format='od_to',
        calib_params=segment_dict,
        csv=True,
    )
    od_from_path = os.path.join(mat_import_dir, od_from_fname)
    od_from = pd.read_csv(od_from_path, index_col=0)

    od_mat = od_to + od_from

    return od_mat, completed_segments


def matrices_to_vector(mat_import_dir: pathlib.Path,
                       years_needed: List[str],
                       model_zone_col: str,
                       internal_zones: List[int] = None,
                       external_zones: List[int] = None,
                       verbose: bool = True,
                       ) -> Tuple[pd.DataFrame, pd.DataFrame,
                                  pd.DataFrame, pd.DataFrame,
                                  pd.DataFrame, pd.DataFrame,
                                  pd.DataFrame, pd.DataFrame]:
    """
    Returns either P/A or O/D vectors based on the matrices in mat_import_dir

    8 return values in total. Internal matrices being the first 4, external
    matrices being the second 4.
    If neither internal or external matrices is set, the first 4 returns are
    the full matrices, and the second 4 will be empty.
    If only the internal zones is set the first 4 returns will be empty, and
    the second 4 will be the external matrices.

    Parameters
    ----------
    mat_import_dir:
        The path to a directory containing the pa or od matrices. Matrix
        information will be inferred from the matrix names. This function
        assumes all .csv files in the directory are matrices of the same
        format (either pa or od).

    years_needed:
        A list of years to look for and build vectors for.

    model_zone_col:
        The name to give to the zone columns if it can't be inferred from the
        matrices.

    internal_zones:
        A list of internal zones. If set then only these zones are used
        to extract the internal demand from the matrices.

    external_zones:
        A list of internal zones. If set then only these zones are used
        to extract the internal demand from the matrices.

    verbose:
        Whether to write progress info to the terminal or not.

    Returns
    -------
    internal_hb_p_or_o:
        A segmentation vector of home based productions or origins
        (depending on the format of the matrices in mat_import_dir) for all
        the years asked for in years_needed.

    internal_nhb_p_or_o:
        A segmentation vector of non home based productions or origins
        (depending on the format of the matrices in mat_import_dir) for all
        the years asked for in years_needed.

    internal_hb_a_or_d:
        A segmentation vector of home based attractions or destinations
        (depending on the format of the matrices in mat_import_dir) for all
        the years asked for in years_needed.

    internal_nhb_a_or_d:
        A segmentation vector of non home based attractions or destinations
        (depending on the format of the matrices in mat_import_dir) for all
        the years asked for in years_needed.
    """
    # BACKLOG: matrices_to_vector() needs checks adding for edge cases
    #  labels: EFS, error checks
    # Init
    du.print_w_toggle("Generating vectors from %s..." % mat_import_dir, verbose=verbose)
    mat_names = [x for x in du.list_files(mat_import_dir) if file_ops.is_csv(x)]

    # Try to figure out what all the zones are
    check_path = os.path.join(mat_import_dir, mat_names[0])
    check_mat = file_ops.read_df(check_path, index_col=0)
    all_zones = list(check_mat.index)    # Assume square

    # If no internal or external zones are given, use internal for everything
    if internal_zones is None and external_zones is None:
        internal_zones = all_zones

    # Split the matrices into years
    year_to_segment_dicts = defaultdict(list)
    for fname in mat_names:
        # Parse the matrix fname into a segment dict
        segment_dict = du.fname_to_calib_params(
            fname=fname,
            get_trip_origin=True,
            get_matrix_format=True,
        )

        mat_year = str(segment_dict['yr'])
        year_to_segment_dicts[mat_year].append(segment_dict)

    # Make sure all the years we need exist
    for year in years_needed:
        if year not in year_to_segment_dicts.keys():
            raise du.NormitsDemandError(
                "Cannot find matrices for all the years asked for. "
                "Specifically there does not seem to be any matrices for"
                "year %s." % str(year)
            )

    # Build the progress bar
    total = sum([len(year_to_segment_dicts[year]) for year in years_needed])
    desc = 'Building vectors'
    p_bar = tqdm(total=total, desc=desc, disable=(not verbose))

    # Build a vector for each year
    int_yearly_p_or_o = list()
    int_yearly_a_or_d = list()
    ext_yearly_p_or_o = list()
    ext_yearly_a_or_d = list()
    segment_col_names = list()
    for year in years_needed:

        # This shouldn't change across matrices
        temp_seg_dict = year_to_segment_dicts[year][0]
        matrix_format = checks.validate_matrix_format(temp_seg_dict['matrix_format'])

        # Figure out which column names we should use
        if matrix_format in consts.PA_MATRIX_FORMATS:
            p_or_o_val_name = 'productions'
            a_or_d_val_name = 'attractions'
        elif matrix_format in consts.OD_MATRIX_FORMATS:
            p_or_o_val_name = 'origin'
            a_or_d_val_name = 'destination'
        else:
            raise ValueError(
                "%s seems to be a valid matrix format, but I don't know "
                "how to handle it!"
                % str(matrix_format)
            )

        # Build an efficient_df for each matrix
        segment_col_names = set()
        int_year_eff_dfs = list()
        ext_year_eff_dfs = list()
        completed_segments = list()
        df_lists = [int_year_eff_dfs, ext_year_eff_dfs]
        for segment_dict in year_to_segment_dicts[year]:

            # ## LOAD THE MATRIX IN ## #
            matrix, completed_segments = load_matrix_from_disk(
                mat_import_dir=mat_import_dir,
                segment_dict=segment_dict,
                completed_segments=completed_segments,
            )

            # If empty, its od_from or od_to and we've already done it
            if matrix.empty:
                p_bar.update(1)
                continue

            # Extract just the internal/external data
            zones = [internal_zones, external_zones]
            join_fns = [operator.and_, operator.or_]
            for zone_nums, year_eff_dfs, join_fn in zip(zones, df_lists, join_fns):
                # If we don't have any zone numbers, we can skip
                if zone_nums is None:
                    continue

                # Extract just the zones we need
                zone_mask = mat_utils.get_wide_mask(
                    df=matrix,
                    zones=zone_nums,
                    join_fn=join_fn,
                )
                matrix = matrix.where(zone_mask, 0)

                # ## GET THE ROW AND COLUMN TOTALS AND NAMES ## #
                # Convert into p/a or o/d
                p_or_o = matrix.sum(axis=1)
                a_or_d = matrix.sum(axis=0)

                # Sort out the column naming
                zone_col_name = matrix.index.name
                if zone_col_name is None:
                    zone_col_name = model_zone_col
                p_or_o.index.name = zone_col_name
                a_or_d.index.name = zone_col_name

                p_or_o = p_or_o.reset_index()
                a_or_d = a_or_d.reset_index()

                p_or_o[zone_col_name] = p_or_o[zone_col_name].astype(int)
                a_or_d[zone_col_name] = a_or_d[zone_col_name].astype(int)

                p_or_o = p_or_o.rename(columns={0: p_or_o_val_name})
                a_or_d = a_or_d.rename(columns={0: a_or_d_val_name})

                # ## COMPILE INTO AN EFFICIENT DF ## #
                # Remove the info we no longer need
                eff_df = segment_dict.copy()
                del eff_df['yr']
                del eff_df['trip_origin']
                del eff_df['matrix_format']

                # Keep track of the column names we're keeping
                vector_columns = [zone_col_name] + list(eff_df.keys())
                segment_col_names = set(list(segment_col_names) + vector_columns)

                # Add the dataframe and we're done!
                # After this the df value is a df of either cols:
                # zone_col, productions, attractions
                # zone_col, origin, destination
                eff_df['df'] = pd.merge(
                    p_or_o,
                    a_or_d,
                    on=zone_col_name
                )
                year_eff_dfs.append(eff_df)

            # Update the progress bar
            p_bar.update(1)

        # Figure out the final column names
        segment_col_names = list(segment_col_names)
        value_cols = [p_or_o_val_name, a_or_d_val_name]
        final_col_names = segment_col_names + value_cols

        # Compile into a vector for this year
        po_vecs = [int_yearly_p_or_o, ext_yearly_p_or_o]
        ad_vecs = [int_yearly_a_or_d, ext_yearly_a_or_d]
        for year_eff_dfs, yearly_p_or_o, yearly_a_or_d in zip(df_lists, po_vecs, ad_vecs):
            # If no vectors were created, just make an empty df
            if year_eff_dfs == list():
                year_pa = pd.DataFrame()
                continue

            year_pa = du.compile_efficient_df(year_eff_dfs, col_names=final_col_names)
            year_pa = du.sort_vector_cols(year_pa)
            year_pa = year_pa.sort_values(by=segment_col_names)

            # Tidy up soc/ns columns
            for col_name in ['soc', 'ns']:
                if col_name in list(year_pa):
                    # Need to make sure soc/ns are int and not float
                    year_pa[col_name] = year_pa[col_name].fillna(-1).astype(int)
                    year_pa[col_name] = year_pa[col_name].replace(-1, 'none').astype(str)

            # ## STORE DATAFRAMES FOR CONCAT LATER ## #
            # Remove the other column from each
            p_or_o_vec = year_pa.drop(columns=[a_or_d_val_name])
            a_or_d_vec = year_pa.drop(columns=[p_or_o_val_name])

            # Rename for the year we've just done and store for later
            yearly_p_or_o.append(p_or_o_vec.rename(columns={p_or_o_val_name: year}))
            yearly_a_or_d.append(a_or_d_vec.rename(columns={a_or_d_val_name: year}))

    # At this point we have a list of vectors for different years

    # ## BUILD THE LIST OF RETURN VALUES ## #
    yearly_vectors = [
        int_yearly_p_or_o,
        int_yearly_a_or_d,
        ext_yearly_p_or_o,
        ext_yearly_a_or_d
    ]
    return_values = list()
    for yearly_vec in yearly_vectors:
        # If no matrices, return an empty df
        if yearly_vec == list():
            return_values += [pd.DataFrame, pd.DataFrame]
            continue

        # Merge the years together
        vector = du.merge_df_list(yearly_vec, on=segment_col_names)

        # Split out the HB and NHB, add to return
        hb_mask = vector['p'].isin(consts.ALL_HB_P)
        return_values.append(vector[hb_mask].copy())
        return_values.append(vector[~hb_mask].copy())

    return return_values


def maybe_convert_matrices_to_vector(mat_import_dir: pathlib.Path,
                                     years_needed: List[str],
                                     cache_path: pathlib.Path,
                                     matrix_format: str,
                                     model_zone_col: str,
                                     internal_zones: List[int] = None,
                                     external_zones: List[int] = None,
                                     overwrite_cache: bool = False,
                                     verbose: bool = True,
                                     ) -> pd.DataFrame:
    """
    A cache wrapper around matrices_to_vector().

    Checks if the asked for matrices already exist at the path given in
    cache_path. If they exist, they're loaded. Otherwise matrices_to_vector()
    is ran, and the output saved to disk at cache_path before returning
    the vectors produced.

    Parameters
    ----------
    mat_import_dir:
        The path to a directory containing the pa or od matrices. Matrix
        information will be inferred from the matrix names. This function
        assumes all .csv files in the directory are matrices of the same
        format (either pa or od).

    years_needed:
        A list of years to look for and build vectors for.

    cache_path:
        The path to a directory where the cached vectors should be saved/
        loaded from.

    matrix_format:
        The format of the matrices being produced. Should be one of the
        valid values from efs_consts.MATRIX_FORMATS

    model_zone_col:
        The name to give to the zone columns if it can't be inferred from the
        matrices.

    internal_zones:
        A list of internal zones. If set then only these zones are used
        to extract the internal demand from the matrices.

    external_zones:
        A list of internal zones. If set then only these zones are used
        to extract the internal demand from the matrices.

    overwrite_cache:
        If True, the vectors are remade and overwrite any cache that may
        already exist.

    verbose:
        Whether to write progress info to the terminal or not.

    Returns
    -------
    hb_p_or_o:
        A segmentation vector of home based productions or origins
        (depending on the format of the matrices in mat_import_dir) for all
        the years asked for in years_needed.

    nhb_p_or_o:
        A segmentation vector of non home based productions or origins
        (depending on the format of the matrices in mat_import_dir) for all
        the years asked for in years_needed.

    hb_a_or_d:
        A segmentation vector of home based attractions or destinations
        (depending on the format of the matrices in mat_import_dir) for all
        the years asked for in years_needed.

    nhb_a_or_d:
        A segmentation vector of non home based attractions or destinations
        (depending on the format of the matrices in mat_import_dir) for all
        the years asked for in years_needed.
    """
    # Init
    matrix_format = checks.validate_matrix_format(matrix_format)

    # Figure out the file paths we should be using
    if matrix_format == 'pa':
        hb_p_or_o_fname = consts.PRODS_FNAME % ('cache', 'hb')
        nhb_p_or_o_fname = consts.PRODS_FNAME % ('cache', 'nhb')
        hb_a_or_o_fname = consts.ATTRS_FNAME % ('cache', 'hb')
        nhb_a_or_o_fname = consts.ATTRS_FNAME % ('cache', 'nhb')
    elif matrix_format == 'od':
        hb_p_or_o_fname = consts.ORIGS_FNAME % ('cache', 'hb')
        nhb_p_or_o_fname = consts.ORIGS_FNAME % ('cache', 'nhb')
        hb_a_or_o_fname = consts.DESTS_FNAME % ('cache', 'hb')
        nhb_a_or_o_fname = consts.DESTS_FNAME % ('cache', 'nhb')
    else:
        raise ValueError(
            "%s seems to be a valid matrix format, but I don't know what to "
            "do with it!" % matrix_format
        )

    # ## BUILD THE PATHS THAT THE CACHE WOULD BE IN ## #
    cache_fnames = [
        hb_p_or_o_fname,
        nhb_p_or_o_fname,
        hb_a_or_o_fname,
        nhb_a_or_o_fname,
    ]
    if internal_zones is not None or external_zones is not None:
        # Build different lists for internal and externals
        cache_paths = list()
        if internal_zones is not None:
            # Make sure the dir exists
            int_dir = os.path.join(cache_path, 'internal')
            du.create_folder(int_dir, verbose=False)

            # Add all files to the cache paths
            cache_paths += [os.path.join(int_dir, f) for f in cache_fnames]

        if external_zones is not None:
            # Make sure the dir exists
            ext_dir = os.path.join(cache_path, 'internal')
            du.create_folder(ext_dir, verbose=False)

            # Add all files to the cache paths
            cache_paths += [os.path.join(ext_dir, f) for f in cache_fnames]

    else:
        # No subsets, just look at the top level
        cache_paths = [os.path.join(cache_path, f) for f in cache_fnames]

    # Read from disk if files already exist
    if all([file_ops.file_exists(f) for f in cache_paths]) and not overwrite_cache:
        dtypes = {'soc': str, 'ns': str}
        return [pd.read_csv(f, dtype=dtypes) for f in cache_paths]

    # ## CREATE AND CACHE IF FILES DON'T EXIST YET ## #

    # Make the files - returned in same order as filenames above
    vectors = matrices_to_vector(
        mat_import_dir=mat_import_dir,
        years_needed=years_needed,
        model_zone_col=model_zone_col,
        internal_zones=internal_zones,
        external_zones=external_zones,
        verbose=verbose
    )

    # Save to disk, and return copies
    vectors = [v for v in vectors if not v.empty]
    for vector, path in zip(vectors, cache_paths):
        vector.to_csv(path, index=False)

    return vectors


def compile_norms_to_vdm_internal(mat_import: nd.PathLike,
                                  mat_export: nd.PathLike,
                                  params_export: nd.PathLike,
                                  years_needed: List[str],
                                  m_needed: List[int],
                                  matrix_format: str,
                                  avoid_zero_splits: bool = False,
                                  ) -> List[str]:
    """
    Generates the compile params and compiles norms internal matrices.

    Parameters
    ----------
    mat_import:
        path to the directory containing the matrices to compile

    mat_export:
        path to the directory where the compiled matrices should be written

    params_export:
        path to the directory where the compile params and splitting factors
        should be written

    years_needed:
        A list of years to compile matrices for. Each year is dealt with
        individually. I.e. you cannot compile matrices across multiple years.

    m_needed:
        A list of the modes to compile matrices for. Each mode is dealt with
        individually. I.e. you cannot compile matrices across multiple modes.

    matrix_format:
        The format of of the matrices to compile. Needs to be one of
        efs_consts.MATRIX_FORMATS

    avoid_zero_splits:
        If set to True, then no zero splits will appear in the splitting
        factors. Where there would have been zero splits, this will be
        replaced with even splits across inputs.

    Returns
    -------
    splitting_factor_paths:
        Returns a list of paths to all the generated splitting factors
    """
    # Init
    fname_suffix = 'internal'

    # Build compile params
    params_paths = build_norms_vdm_compile_params(
        import_dir=mat_import,
        export_dir=params_export,
        matrix_format=matrix_format,
        segmentation_aggregation=consts.NORMS_VDM_SEG_INTERNAL,
        years_needed=years_needed,
        m_needed=m_needed,
        params_suffix=fname_suffix,
    )

    # Compile, return split factors
    sf_paths = list()
    for compile_params_path, year in zip(params_paths, years_needed):
        factors_fname = du.get_split_factors_fname(
            matrix_format=matrix_format,
            year=str(year),
            suffix=fname_suffix,
        )
        split_factors_path = os.path.join(params_export, factors_fname)

        # Store for return
        sf_paths.append(split_factors_path)

        compile_matrices(
            mat_import=mat_import,
            mat_export=mat_export,
            compile_params_path=compile_params_path,
            factor_pickle_path=split_factors_path,
            avoid_zero_splits=avoid_zero_splits,
        )

    return sf_paths


def compile_norms_to_vdm_external(mat_import: nd.PathLike,
                                  mat_export: nd.PathLike,
                                  params_export: nd.PathLike,
                                  years_needed: List[str],
                                  m_needed: List[int],
                                  matrix_format: str,
                                  avoid_zero_splits: bool = False,
                                  ) -> List[str]:
    """
    Generates the compile params and compiles norms external matrices.

    Parameters
    ----------
    mat_import:
        path to the directory containing the matrices to compile

    mat_export:
        path to the directory where the compiled matrices should be written

    params_export:
        path to the directory where the compile params and splitting factors
        should be written

    years_needed:
        A list of years to compile matrices for. Each year is dealt with
        individually. I.e. you cannot compile matrices across multiple years.

    m_needed:
        A list of the modes to compile matrices for. Each mode is dealt with
        individually. I.e. you cannot compile matrices across multiple modes.

    matrix_format:
        The format of of the matrices to compile. Needs to be one of
        efs_consts.MATRIX_FORMATS

    avoid_zero_splits:
        If set to True, then no zero splits will appear in the splitting
        factors. Where there would have been zero splits, this will be
        replaced with even splits across inputs.

    Returns
    -------
    splitting_factor_paths:
        Returns a list of paths to all the generated splitting factors
    """
    # Init
    fname_suffix = 'external'

    # Build compile params
    params_paths = build_norms_vdm_compile_params(
        import_dir=mat_import,
        export_dir=params_export,
        matrix_format=matrix_format,
        segmentation_aggregation=consts.NORMS_VDM_SEG_EXTERNAL,
        years_needed=years_needed,
        m_needed=m_needed,
        params_suffix=fname_suffix,
    )

    # Compile, return split factors
    sf_paths = list()
    for compile_params_path, year in zip(params_paths, years_needed):
        factors_fname = du.get_split_factors_fname(
            matrix_format=matrix_format,
            year=str(year),
            suffix=fname_suffix,
        )
        split_factors_path = os.path.join(params_export, factors_fname)

        # Store for return
        sf_paths.append(split_factors_path)

        compile_matrices(
            mat_import=mat_import,
            mat_export=mat_export,
            compile_params_path=compile_params_path,
            factor_pickle_path=split_factors_path,
            avoid_zero_splits=avoid_zero_splits,
        )

    return sf_paths


def _split_int_ext(mat_import,
                   seg_vals,
                   internal_export,
                   external_export,
                   internal_zones,
                   external_zones,
                   csv_out,
                   compress_out,
                   ):
    """
    Internal loop function for split_internal_external()
    """
    # Build the input file path
    fname = du.calib_params_to_dist_name(
        trip_origin=seg_vals['trip_origin'],
        matrix_format=seg_vals['matrix_format'],
        calib_params=seg_vals,
        csv=True,
    )
    path = os.path.join(mat_import, fname)
    full_mat = file_ops.read_df(path, index_col=0, find_similar=True)

    # Build an iterator to go through internal and external
    iterator = zip(
        ['int', 'ext'],
        [internal_export, external_export],
        [internal_zones, external_zones],
        [operator.and_, operator.or_],
    )

    # Extract and write to disk
    for name, out_dir, zones, join_fn in iterator:
        # Skip over the internal or external if we're not writing out
        if out_dir is None:
            continue

        # Get the mask and extract the data
        mask = pd_utils.get_wide_mask(full_mat, zones, join_fn=join_fn)
        sub_mat = full_mat.where(mask, 0)

        fname = du.calib_params_to_dist_name(
            trip_origin=seg_vals['trip_origin'],
            matrix_format=seg_vals['matrix_format'],
            calib_params=seg_vals,
            suffix='_%s' % name,
            csv=csv_out,
            compressed=compress_out,
        )
        out_path = os.path.join(out_dir, fname)
        file_ops.write_df(sub_mat, out_path)


def split_internal_external(mat_import: nd.PathLike,
                            year: Union[int, str],
                            matrix_format: str,
                            internal_zones: List[int] = None,
                            external_zones: List[int] = None,
                            internal_export: nd.PathLike = None,
                            external_export: nd.PathLike = None,
                            csv_out: bool = False,
                            compress_out: bool = True,
                            ) -> None:
    # TODO(BT): Write split_internal_external() docs
    # Init
    if not isinstance(year, int):
        year = int(year)
    ftypes = ['.csv', consts.COMPRESSION_SUFFIX]
    mat_paths = file_ops.list_files(mat_import, ftypes=ftypes)

    # Validate input values
    base_msg = (
        "Both  %s_zones and %s_export need to be either set or not set. "
        "If only one is set, both are ignored."
    )
    msg = base_msg % ('internal', 'internal')
    checks.all_values_set([internal_zones, internal_export], msg, warn=True)

    msg = base_msg % ('external', 'external')
    checks.all_values_set([external_zones, external_export], msg, warn=True)

    internal_export = None if internal_zones is None else internal_export
    external_export = None if external_zones is None else external_export

    # Filter down to just the year we want
    mat_seg_vals = list()
    for path in mat_paths:
        # Parse the filename
        seg_vals = du.fname_to_calib_params(
            path,
            get_trip_origin=True,
            get_matrix_format=False
        )
        seg_vals['matrix_format'] = matrix_format

        # Skip over any file which is not the wanted year
        if seg_vals['yr'] != year:
            continue

        mat_seg_vals.append(seg_vals)

    # ## MULTIPROCESS THE SPLITTING ##
    unchanging_kwargs = {
        'mat_import': mat_import,
        'internal_export': internal_export,
        'external_export': external_export,
        'internal_zones': internal_zones,
        'external_zones': external_zones,
        'csv_out': csv_out,
        'compress_out': compress_out,
    }

    # Build a list of the kwargs
    kwarg_list = list()
    for seg_vals in mat_seg_vals:
        kwargs = unchanging_kwargs.copy()
        kwargs['seg_vals'] = seg_vals
        kwarg_list.append(kwargs)

    # Call
    multiprocessing.multiprocess(
        fn=_split_int_ext,
        kwargs=kwarg_list,
        process_count=consts.PROCESS_COUNT,
    )


def compile_norms_to_vdm(mat_import: nd.PathLike,
                         mat_export: nd.PathLike,
                         params_export: nd.PathLike,
                         year: str,
                         m_needed: List[int],
                         matrix_format: str,
                         internal_zones: List[int],
                         external_zones: List[int],
                         from_to_split_factors: nd.FactorsDict = None,
                         avoid_zero_splits: bool = False,
                         ) -> str:
    # TODO(BT) Write compile_norms_to_vdm() docs
    # Init
    # matrix_format = checks.validate_matrix_format(matrix_format)

    # Build temporary paths
    int_dir = os.path.join(mat_export, 'internal')
    ext_dir = os.path.join(mat_export, 'external')

    for path in [int_dir, ext_dir]:
        file_ops.create_folder(path)

    # Temporary output if we need to split from/to
    compiled_dir = mat_export
    if from_to_split_factors is not None:
        compiled_dir = os.path.join(mat_export, 'compiled_non_split')
        file_ops.create_folder(compiled_dir)

    # Split internal and external
    print("Splitting into internal and external matrices...")
    split_internal_external(
        mat_import=mat_import,
        matrix_format=matrix_format,
        internal_export=int_dir,
        external_export=ext_dir,
        year=year,
        internal_zones=internal_zones,
        external_zones=external_zones,
    )

    # Compile and get the splitting factors for internal mats
    print("Generating internal splitting factors...")
    int_split_factors = compile_norms_to_vdm_internal(
        mat_import=int_dir,
        mat_export=compiled_dir,
        params_export=params_export,
        years_needed=[year],
        m_needed=m_needed,
        matrix_format=matrix_format,
        avoid_zero_splits=avoid_zero_splits,
    )

    print("Generating external splitting factors...")
    ext_split_factors = compile_norms_to_vdm_external(
        mat_import=ext_dir,
        mat_export=compiled_dir,
        params_export=params_export,
        years_needed=[year],
        m_needed=m_needed,
        matrix_format=matrix_format,
        avoid_zero_splits=avoid_zero_splits,
    )

    # We know we're only doing a single year here
    int_split_factors = int_split_factors[0]
    ext_split_factors = ext_split_factors[0]

    # If we don't have the post_me path, exit now. Can't do any more
    if from_to_split_factors is None:
        return int_split_factors, ext_split_factors

    # ## CONVERT TO THE NORMS POST-ME FORMAT ## #
    print("Converting matrices into NoRMS format...")
    mat_comp.convert_efs_to_norms_matrices(
        mat_import=compiled_dir,
        mat_export=mat_export,
        year=year,
        from_to_split_factors=from_to_split_factors
    )

    return int_split_factors, ext_split_factors


def _recombine_internal_external_internal(in_paths,
                                          output_path,
                                          output_suffix,
                                          ) -> None:
    # Read in the matrices and compile
    partial_mats = [file_ops.read_df(x, index_col=0, find_similar=True) for x in in_paths]
    full_mat = functools.reduce(lambda x, y: x.values + y.values, partial_mats)

    # Store back in a df
    full_mat = pd.DataFrame(
        full_mat,
        index=partial_mats[0].index,
        columns=partial_mats[0].columns,
    )

    # Sort out the output suffix
    base_path = file_ops.remove_suffixes(pathlib.Path(output_path))
    output_path = base_path.with_suffix(output_suffix)

    # Write the complete matrix to disk
    file_ops.write_df(full_mat, output_path)


def combine_partial_matrices(import_dirs: List[nd.PathLike],
                             export_dir: List[nd.PathLike],
                             segmentation: nd.SegmentationLevel,
                             import_suffixes: List[str] = None,
                             csv_out: bool = False,
                             process_count: int = consts.PROCESS_COUNT,
                             pbar_kwargs: Dict[str, Any] = None,
                             **file_kwargs,
                             ) -> None:
    """Combines the matrices in import_dirs and writes out to export_dir

    Parameters
    ----------
    import_dirs:
        A list of the directories to read files from to combine

    export_dir:
        The directory to output the combined matrices to

    segmentation:
        The segmentation to use to generate the filenames

    import_suffixes:
        A list of the suffixes for each directory in import_dirs. Should be
        a parallel list to import_dirs. Any directories without a suffix
        should be set to None.

    csv_out:
        Whether to write the combined matrices out as csvs. If False, files
        will be written out in compressed format.

    process_count:
        The number of processes to use when combining matrices.

    pbar_kwargs:
        A dictionary of keyword arguments to pass into tqdm.tqdm to make a 
        progress bar. If left as None, not progress bar will be shown.
    
    file_kwargs:
        Any additional arguments to pass to segmentation.generate_file_name().
    """
    # Init
    if import_suffixes is None:
        import_suffixes = [None] * len(import_dirs)

    # Check paths exist
    if not os.path.exists(export_dir):
        raise FileExistsError(
            "No directory exists for exporting files. Looking here:\n%s"
            % export_dir
        )

    for in_dir in import_dirs:
        if not os.path.exists(in_dir):
            raise FileExistsError(
                "One of the import directories does not exist. Looking here:\n%s"
                % in_dir
            )

    # ## BUILD DICTIONARY OF MATRICES TO COMBINE ## #
    combine_dict = dict()
    for segment_params in segmentation:
        # Get the output path
        out_fname = segmentation.generate_file_name(
            segment_params=segment_params,
            compressed=True,
            **file_kwargs,
        )
        out_path = os.path.join(export_dir, out_fname)

        # Generate the input path
        in_paths = list()
        for in_dir, suffix in zip(import_dirs, import_suffixes):
            fname = segmentation.generate_file_name(
                segment_params=segment_params,
                suffix=suffix,
                **file_kwargs,
            )
            in_paths.append(os.path.join(in_dir, fname))

        combine_dict[out_path] = in_paths

    if csv_out:
        output_suffix = ".csv"
    else:
        output_suffix = ".csv.bz2"

    # ## COMPILE THE MATRICES ## #
    kwarg_list = list()
    for output_path, in_paths in combine_dict.items():
        kwarg_list.append({
            'output_path': output_path,
            'in_paths': in_paths,
            'output_suffix': output_suffix,
        })

    multiprocessing.multiprocess(
        fn=_recombine_internal_external_internal,
        kwargs=kwarg_list,
        process_count=process_count,
        # process_count=0,
        pbar_kwargs=pbar_kwargs,
    )


def recombine_internal_external(internal_import: nd.PathLike,
                                external_import: nd.PathLike,
                                full_export: nd.PathLike,
                                years: List[int],
                                force_csv_out: bool = False,
                                force_compress_out: bool = False,
                                process_count: int = consts.PROCESS_COUNT,
                                pbar_kwargs: Dict[str, Any] = None,
                                ) -> None:
    """
    Combines the internal and external split matrices and write out to full_export

    Will warn the user if all matrices from both folders are not used

    Parameters
    ----------
    internal_import:
        Path to the directory containing the segmented internal matrices

    external_import:
        Path to the directory containing the segmented external matrices

    full_export:
        Path to the directory to write out the combined matrices.

    Returns
    -------
    None

    """
    # Init
    all_internal_fnames = file_ops.list_files(internal_import)
    all_external_fnames = file_ops.list_files(external_import)

    # Filter to just the wanted years
    internal_fnames = list()
    external_fnames = list()
    for year in years:
        yr_str = '_yr%s_' % year
        internal_fnames += [x for x in all_internal_fnames if yr_str in x]
        external_fnames += [x for x in all_external_fnames if yr_str in x]

    all_internal_fnames = internal_fnames
    all_external_fnames = external_fnames

    # ## BUILD DICTIONARY OF MATRICES TO COMBINE ## #
    comp_dict = dict()
    used_external_fnames = list()
    for int_fname in all_internal_fnames:
        # Determine the related filenames
        full_fname = file_ops.remove_internal_suffix(int_fname)
        ext_fname = file_ops.add_external_suffix(full_fname)

        # Check the external file actually exists
        try:
            file_ops.find_filename(
                path=os.path.join(external_import, ext_fname),
                alt_types=['.csv', consts.COMPRESSION_SUFFIX]
            )
        except FileNotFoundError as e:
            print(e)
            raise FileNotFoundError(
                "No external file exists to match the internal file.\n"
                "Internal file location: %s\n"
                "Expected external file location: %s"
                % (os.path.join(internal_import, int_fname),
                   os.path.join(external_import, ext_fname))
            )

        # Make a note of the external files we've used
        used_external_fnames.append(str(ext_fname))

        # Add an entry to the dictionary
        output_path = os.path.join(full_export, full_fname)
        comp_dict[output_path] = [
            os.path.join(internal_import, int_fname),
            os.path.join(external_import, ext_fname),
        ]

    # Make sure we've used all the external matrices
    for ext_fname in all_external_fnames:
        if not file_ops.filename_in_list(ext_fname, used_external_fnames, ignore_ftype=True):
            int_fname = ext_fname.replace(consts.EXTERNAL_SUFFIX, consts.INTERNAL_SUFFIX)
            raise FileNotFoundError(
                "No internal file exists to match the external file.\n"
                "External file location: %s\n"
                "Expected internal file location: %s"
                % (os.path.join(external_import, ext_fname),
                   os.path.join(internal_import, int_fname))
            )

    # ## COMPILE THE MATRICES ## #
    kwarg_list = list()
    for output_path, in_paths in comp_dict.items():
        kwarg_list.append({
            'output_path': output_path,
            'in_paths': in_paths,
            'force_csv_out': force_csv_out,
            'force_compress_out': force_compress_out,
        })
        
    multiprocessing.multiprocess(
        fn=_recombine_internal_external_internal,
        kwargs=kwarg_list,
        process_count=process_count,
        pbar_kwargs=pbar_kwargs,
    )

