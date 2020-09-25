import os

import numpy as np
import pandas as pd

from typing import List
from itertools import product

import efs_constants as consts
import demand_utilities.utils as du


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
