import os

import numpy as np
import pandas as pd
import pickle

from typing import List
from itertools import product

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
    # Write Doc

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
    # Write Doc

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
    all_matrices = [x for x in all_matrices if x[:len(trip_origin)] == trip_origin]

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


def _generate_tour_proportions_internal(orig,
                                        dest_vals,
                                        tp_needed,
                                        fh_mats,
                                        th_mats,
                                        furness_tol,
                                        furness_max_iters
                                        ):
    """
    The internals of generate_tour_proportions().
    Used to implement multiprocessing.

    Returns
    -------
    tour_proportions:
        dict(). the tour_proportion values for all destinations in this orig

    """
    print("Working on tour proportions for Origin: %s" % str(orig))
    tour_proportions = dict()
    for dest in dest_vals:
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
        seed_val = 1  # ASSUME 1 for now

        # First use tp4 to bring both vector sums to average
        fh_th_avg = (fh_target.sum() + th_target.sum()) / 2
        fh_target[-1] = fh_th_avg - np.sum(fh_target[:-1])
        th_target[-1] = fh_th_avg - np.sum(th_target[:-1])

        # Correct for the resulting negative value
        if fh_target[-1] < 0:
            th_target[-1] -= (1 + seed_val) * fh_target[-1]
            fh_target[-1] *= -seed_val
        elif th_target[-1] < 0:
            fh_target[-1] -= (1 + seed_val) * th_target[-1]
            th_target[-1] *= -seed_val

        # Convert the numbers to fractional factors
        fh_target /= fh_target.sum()
        th_target /= th_target.sum()

        # ## FURNESS ## #
        n_tp = len(tp_needed)
        seed_vals = np.broadcast_to(seed_val, (n_tp, n_tp))

        furnessed_mat = doubly_constrained_furness(
            seed_vals=seed_vals,
            row_targets=fh_target,
            col_targets=th_target,
            tol=furness_tol,
            max_iters=furness_max_iters
        )

        # Store the tour proportions
        furnessed_mat = furnessed_mat.astype('float16')
        tour_proportions[dest] = furnessed_mat

    return tour_proportions


def generate_tour_proportions(od_import: str,
                              tour_proportions_export: str,
                              year: int = consts.BASE_YEAR,
                              p_needed: List[int] = consts.ALL_HB_P,
                              m_needed: List[int] = consts.MODES_NEEDED,
                              soc_needed: List[int] = None,
                              ns_needed: List[int] = None,
                              ca_needed: List[int] = None,
                              tp_needed: List[int] = consts.TIME_PERIODS,
                              furness_tol: float = 1e-9,
                              furness_max_iters: int = 5000,
                              process_count: int = os.cpu_count() - 1
                              ) -> None:
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

    for p, m, seg, ca in loop_generator:
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
                    raise ValueError("At least one of the loaded matrices "
                                     "does not match the others. Expected a "
                                     "matrix of shape (%d, %d), got %s."
                                     % (n_rows, n_cols, str(mat.shape)))

        # ## FURNESS TOUR PROPORTIONS ## #
        # Init
        tour_proportions = dict()

        # TODO: Can all this faff be put in a function
        #  It'll make the code a little easier to read
        # Setup for multiprocessing
        unchanging_kwargs = {
            'dest_vals': list(range(n_cols)),
            'tp_needed': tp_needed,
            'fh_mats': fh_mats,
            'th_mats': th_mats,
            'furness_tol': furness_tol,
            'furness_max_iters': furness_max_iters
        }

        if process_count == 0:
            # Do as for loop
            for orig in range(n_rows):
                kwargs = unchanging_kwargs.copy()
                kwargs['orig'] = orig
                tour_proportions[orig] = _generate_tour_proportions_internal(**kwargs)
                break

        else:
            # Use multiprocessing
            kwargs_list = list()
            orig_zones = list(range(n_rows))
            for orig in orig_zones:
                kwargs = unchanging_kwargs.copy()
                kwargs['orig'] = orig
                kwargs_list.append(kwargs)

            mp_results = conc.process_pool_wrapper(
                _generate_tour_proportions_internal,
                kwargs=kwargs_list,
                process_count=process_count,
                in_order=True
            )

            # decode results
            for orig, tour_prop in zip(orig_zones, mp_results):
                tour_proportions[orig] = tour_prop

        # Save the tour proportions for this segment
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

        print('Writing tour proportions for %s' % out_fname)
        out_path = os.path.join(tour_proportions_export, out_fname)
        with open(out_path, 'wb') as f:
            pickle.dump(tour_proportions, f, protocol=pickle.HIGHEST_PROTOCOL)


# TODO: Move to furness_process.py
def doubly_constrained_furness(seed_vals: np.array,
                               row_targets: np.array,
                               col_targets: np.array,
                               tol: float = 1e-9,
                               max_iters: int = 5000
                               ) -> np.array:
    # Error check
    if seed_vals.shape != (len(row_targets), len(col_targets)):
        raise ValueError(
            "The shape of the seed values given does not match the row "
            "and col targets. Seed_vals are shape %s. Expected shape (%d, %d)."
            % (str(seed_vals.shape), len(row_targets), len(col_targets))
        )

    # Init
    furnessed_mat = seed_vals.copy()

    for i in range(max_iters):
        # ## ROW CONSTRAIN ## #
        # Calculate difference factor
        row_ach = np.sum(furnessed_mat, axis=1)
        row_ach = np.where(row_ach == 0, 1, row_ach)
        diff_factor = row_targets / row_ach

        # adjust rows
        furnessed_mat = (furnessed_mat.T * diff_factor).T

        # ## COL CONSTRAIN ## #
        # Calculate difference factor
        col_ach = np.sum(furnessed_mat, axis=0)
        col_ach = np.where(col_ach == 0, 1, col_ach)
        diff_factor = col_targets / col_ach

        # adjust cols
        furnessed_mat = furnessed_mat * diff_factor

        # Calculate the diff - leave early if met
        row_diff = (row_targets - np.sum(furnessed_mat, axis=1)) ** 2
        col_diff = (col_targets - np.sum(furnessed_mat, axis=0)) ** 2
        cur_diff = np.sum(row_diff + col_diff) ** 0.5
        if cur_diff < tol:
            break

    return furnessed_mat
