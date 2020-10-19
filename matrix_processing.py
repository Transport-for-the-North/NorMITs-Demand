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
import os

import numpy as np
import pandas as pd
import pickle

from typing import List
from typing import Tuple
from typing import Iterable

from functools import reduce
from itertools import product
from collections import defaultdict

import pa_to_od as pa2od
import furness_process as fp

import efs_constants as consts
import demand_utilities.utils as du
import demand_utilities.concurrency as conc


def _aggregate(import_dir: str,
               in_fnames: List[str],
               export_path: str
               ) -> None:
    """
    Loads the given files, aggregates together and saves in given location
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
    aggregated_mat.to_csv(export_path)
    print("Aggregated matrix written: %s" % os.path.basename(export_path))


def _recursive_aggregate(candidates: List[str],
                         segmentations: List[List[int]],
                         segmentation_strs: List[List[str]],
                         import_dir: str,
                         export_path: str
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

    Returns
    -------
    None
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
                export_path=export_path + '.csv'
            )
        else:
            # Loop through and aggregate
            for segment, seg_str in zip(segmentations, segmentation_strs):
                _aggregate(
                    import_dir=import_dir,
                    in_fnames=[x for x in candidates.copy() if seg_str in x],
                    export_path=export_path + seg_str + '.csv'
                )
        # Exit condition done, leave recursion
        return

    # ## RECURSIVELY LOOP ## #
    seg, other_seg = segmentations[0], segmentations[1:]
    strs, other_strs = segmentation_strs[0], segmentation_strs[1:]

    if du.is_none_like(seg):
        # Don't need to segment here, next loop
        _recursive_aggregate(candidates=candidates,
                             segmentations=other_seg,
                             segmentation_strs=other_strs,
                             import_dir=import_dir,
                             export_path=export_path)
    else:
        # Narrow down search, loop again
        for segment, seg_str in zip(seg, strs):
            _recursive_aggregate(
                candidates=[x for x in candidates.copy() if seg_str in x],
                segmentations=other_seg,
                segmentation_strs=other_strs,
                import_dir=import_dir,
                export_path=export_path + seg_str)


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
                       tp_needed: List[int] = None
                       ) -> None:
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

    Returns
    -------
    None
    """
    # Init
    if(ns_needed is not None and soc_needed is None
       or soc_needed is not None and ns_needed is None):
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
            export_path=out_path
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


def _generate_tour_proportions_internal(od_import: str,
                                        tour_proportions_export: str,
                                        pa_export: str,
                                        trip_origin: str,
                                        year: int,
                                        p: int,
                                        m: int,
                                        seg: int,
                                        ca: int,
                                        tp_needed: List[int],
                                        furness_tol: float,
                                        furness_max_iters: int,
                                        phi_lookup_folder: str,
                                        phi_type: str,
                                        aggregate_to_wday: bool,
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
    print("Generating tour proportions for %s..." % out_fname)

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

        # Optionally output converted PA matrices
        if pa_export is not None:
            pa_name = dist_name.replace('od_from', 'pa')
            du.copy_and_rename(
                src=os.path.join(od_import, dist_name),
                dst=os.path.join(pa_export, pa_name)
            )

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

    # Get the seed values for this purpose
    seed_values = get_tour_proportion_seed_values(
        m=m,
        p=p,
        phi_lookup_folder=phi_lookup_folder,
        phi_type=phi_type,
        aggregate_to_wday=aggregate_to_wday,
        infill=0.001,
    )

    # ## FURNESS TOUR PROPORTIONS ## #
    # Init
    zero_count = 0
    tour_proportions = defaultdict(dict)
    for orig in range(n_rows):
        for dest in range(n_cols):
            # Build the from_home vector
            fh_target = list()
            for tp in tp_needed:
                fh_target.append(fh_mats[tp].values[orig, dest])
            fh_target = np.array(fh_target)

            # Build the to_home vector
            th_target = list()
            for tp in tp_needed:
                th_target.append(th_mats[tp].values[orig, dest])
            th_target = np.array(th_target)

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

            # Only furness if targets are not 0
            if fh_target.sum() == 0 or th_target.sum() == 0:
                # Skip furness, create matrix of 0s instead
                zero_count += 1
                furnessed_mat = np.zeros((len(tp_needed), len(tp_needed)))

            else:
                # Convert the numbers to fractional factors
                fh_target /= fh_target.sum()
                th_target /= th_target.sum()

                # ## FURNESS ## #
                n_tp = len(tp_needed)
                seed_vals = np.broadcast_to(seed_val, (n_tp, n_tp))

                furnessed_mat = fp.doubly_constrained_furness(
                    seed_vals=seed_vals,
                    row_targets=fh_target,
                    col_targets=th_target,
                    tol=furness_tol,
                    max_iters=furness_max_iters
                )

            # Store the tour proportions
            furnessed_mat = furnessed_mat.astype('float32')
            tour_proportions[orig][dest] = furnessed_mat

    # Save the tour proportions for this segment (model_zone level)
    print('Writing tour proportions for %s' % out_fname)
    out_path = os.path.join(tour_proportions_export, out_fname)
    with open(out_path, 'wb') as f:
        pickle.dump(tour_proportions, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Aggregate tour proportions to LA and TfN Sector level
    la_tour_props, tfn_tour_props = aggregate_tour_proportions(
        tour_proportions,
        model=du.get_model_name(m),
    )


    zero_percentage = (zero_count / float(n_rows * n_cols)) * 100
    return out_fname, zero_count, zero_percentage


def generate_tour_proportions(od_import: str,
                              tour_proportions_export: str,
                              pa_export: str = None,
                              year: int = consts.BASE_YEAR,
                              p_needed: List[int] = consts.ALL_HB_P,
                              m_needed: List[int] = consts.MODES_NEEDED,
                              soc_needed: List[int] = None,
                              ns_needed: List[int] = None,
                              ca_needed: List[int] = None,
                              tp_needed: List[int] = consts.TIME_PERIODS,
                              furness_tol: float = 1e-9,
                              furness_max_iters: int = 5000,
                              process_count: int = os.cpu_count() - 1,
                              phi_lookup_folder: str = None,
                              phi_type: str = 'fhp',
                              aggregate_to_wday: bool = True,
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

    pa_export:
        Where to export the converted pa_matrices. If left as None,
        no pa matrices are written.

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

    furness_tol:
        What tolerance to use during the furness.
        See furness_process.doubly_constrained_furness() for more information.

    furness_max_iters:
        Max number of iterations for the furness.
        See furness_process.doubly_constrained_furness() for more information.

    process_count:
        How many processes to use during multiprocessing. Usually set to
        number_of_cpus - 1.

    Returns
    -------
    None

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

    # TODO: Can all this faff be put in a function
    #  It'll make the code a little easier to read
    #  Maybe integrate into multiprocessing.py?

    # ## MULTIPROCESS EACH SEGMENT ## #
    unchanging_kwargs = {
        'od_import': od_import,
        'tour_proportions_export': tour_proportions_export,
        'pa_export': pa_export,
        'trip_origin': trip_origin,
        'year': year,
        'tp_needed': tp_needed,
        'furness_tol': furness_tol,
        'furness_max_iters': furness_max_iters,
        'phi_lookup_folder': phi_lookup_folder,
        'phi_type': phi_type,
        'aggregate_to_wday': aggregate_to_wday

    }
    
    process_count = 0

    # use as many as possible if negative
    if process_count < 0:
        process_count = os.cpu_count() - 1

    if process_count == 0:
        # Loop as normal
        zero_counts = list()
        for p, m, seg, ca in loop_generator:
            kwargs = unchanging_kwargs.copy()
            kwargs.update({
                'p': p,
                'm': m,
                'seg': seg,
                'ca': ca,
            })
            zero_counts.append(_generate_tour_proportions_internal(**kwargs))
    else:
        # Build all the arguments, and call in ProcessPool
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

        zero_counts = conc.process_pool_wrapper(
            _generate_tour_proportions_internal,
            kwargs=kwargs_list,
            process_count=process_count,
            in_order=True
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


def build_compile_params(import_dir: str,
                         export_dir: str,
                         matrix_format: str,
                         years_needed: Iterable[int],
                         m_needed: List[int] = consts.MODES_NEEDED,
                         ca_needed: Iterable[int] = None,
                         tp_needed: Iterable[int] = None,
                         split_hb_nhb: bool = False,
                         output_headers: List[str] = None,
                         output_format: str = 'wide'
                         ) -> None:
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
        together or not. If False, separate hb and nhb compiled matrices are
        created.

    output_headers:
        Optional. Use if custom output headers are needed. by default the
        following headers are used:
        ['distribution_name', 'compilation', 'format']

    output_format:
        What format the compiled matrices should be output as. Usually either
        'wide' or 'long'.

    Returns
    -------
    None
    """
    # Error checking
    if len(m_needed) > 1:
        raise ValueError("Matrix compilation can only handle one mode at a "
                         "time. Received %d modes" % len(m_needed))
    mode = m_needed[0]

    # Init
    ca_needed = [None] if ca_needed is None else ca_needed
    tp_needed = [None] if tp_needed is None else tp_needed
    to_needed = [None] if not split_hb_nhb else ['hb', 'nhb']
    all_od_matrices = du.list_files(import_dir)
    out_lines = list()

    if output_headers is None:
        output_headers = ['distribution_name', 'compilation', 'format']

    for year in years_needed:
        for user_class, purposes in consts.USER_CLASS_PURPOSES.items():
            for ca, tp, to in product(ca_needed, tp_needed, to_needed):
                # Init
                compile_mats = all_od_matrices.copy()
                # include _ before and after to avoid clashes
                ps = ['_p' + str(x) + '_' for x in purposes]
                mode_str = '_m' + str(mode) + '_'
                year_str = '_yr' + str(year) + '_'
                tp_str = '_tp' + str(tp)

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
        out_fname = du.get_compile_params_name(matrix_format, str(year))
        out_path = os.path.join(export_dir, out_fname)
        du.write_csv(output_headers, out_lines, out_path)


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

            # Combine all matrices together
            full_mat = reduce(lambda x, y: x.add(y, fill_value=0), tp_mats)

            # Output to file
            full_mat.to_csv(os.path.join(export_dir, output_dist_name))
