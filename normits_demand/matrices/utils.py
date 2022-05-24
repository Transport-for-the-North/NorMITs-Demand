# -*- coding: utf-8 -*-
"""
Created on: Tues March 2 12:21:12 2021
Updated on:

Original author: Ben Taylor
Last update made by:
Other updates made by:

File purpose:
Utility functions specific to matrices
"""
# builtins
import os
import pathlib
import operator
import functools

from typing import List
from typing import Dict
from typing import Callable

# Third Party
import numpy as np
import pandas as pd

# Local imports
from normits_demand import constants as nd_constants
from normits_demand import core as nd_core
from normits_demand.utils import file_ops
from normits_demand.concurrency import multiprocessing


def check_fh_th_factors(factor_dict: Dict[int, np.ndarray],
                        tp_needed: List[int],
                        n_row_col: int,
                        ) -> None:
    """Validates the given factor_dict

    Checks the the factor_dict has the correct keys, as defined by tp_needed,
    and the np_array values are all the correct shape - (n_row_col, n_row_col)

    Parameters
    ----------
    factor_dict:
        A dictionary of from home or to home splitting factors to check.

    tp_needed:
        The time periods to be expected.

    n_row_col:
        Assumes square PA/OD matrices. The number of zones in the matrices.

    Returns
    -------
    None

    Raises
    ------
    ValueError:
        If all the expected keys do not exist, or the values are not the
        expected shape.
    """
    # Check all the expected keys are there
    current_keys = set(factor_dict.keys())
    expected_keys = set(tp_needed)
    if current_keys != expected_keys:
        raise ValueError(
            "Not all expected time periods are in the given factor_dict."
            "Expected: %s\n"
            "Got: %s\n"
            % (expected_keys, current_keys)
        )

    # Make sure all values are the expected shape
    expected_shape = (n_row_col, n_row_col)
    for k, v in factor_dict.items():
        if v.shape != expected_shape:
            raise ValueError(
                "One of the values in factor_dict is no the expected shape."
                "Expected: %s\n"
                "Got: %s\n"
                % (expected_shape, v.shape)
            )

    # If here, all checks have passed
    return


def _combine_matrices(
    input_paths: List[os.PathLike],
    export_path: os.PathLike,
    combine_fn: Callable = operator.add,
) -> None:
    """Combines the input path matrices and writes back out

    The index/columns of all matrices is ignored, except for the first.
    The first matrix in `input_paths` defines the columns/index for the
    output matrix.

    Parameters
    ----------
    input_paths:
        A list of paths to the matrices to combine

    export_path:
        A path to the location to write the output matrix

    combine_fn:
        The function to use when combining the matrices.

    Returns
    -------
    None
    """
    # Get an output template
    first_mat = file_ops.read_df(input_paths[0], find_similar=True, index_col=0)
    output_index = first_mat[0].index
    output_cols = first_mat[0].columns

    # Read in matrices
    mats = [file_ops.read_df(x, find_similar=True, index_col=0) for x in input_paths[1:]]
    mats = [first_mat] + mats
    
    # Combine matrices and write out
    out_df = pd.DataFrame(
        data=functools.reduce(combine_fn, [x.values for x in mats]),
        columns=output_cols,
        index=output_index,
    )
    file_ops.write_df(out_df, export_path)


def combine_od_to_from_matrices(
    import_dir: pathlib.Path,
    export_dir: pathlib.Path,
    segmentation: nd_core.SegmentationLevel,
    od_fname_template: str,
    od_from_fname_template: str,
    od_to_fname_template: str,
    process_count: int = nd_constants.PROCESS_COUNT,
) -> None:
    """Combines the OD-to and OD-from matrices and writes back out

    Parameters
    ----------
    import_dir:
        The directory containing the matrices to combine.

    export_dir:
        The directory to write out the combined matrices

    segmentation:
        The segmentation level of the matrices to combine. This is used
        alongside `od_fname_template`, `od_from_fname_template`, and
        `od_to_fname_template` to generate the filenames.

    od_fname_template:
        A template filename to generate the output od matrices filenames. Must
        be able to format with
        `od_fname_template.format(segment_params=segment_params)`

    od_from_fname_template:
        A template filename to generate the input od-from matrices filenames.
        Must be able to format with
        `od_from_fname_template.format(segment_params=segment_params)`

    od_to_fname_template:
        A template filename to generate the input od-to matrices filenames.
        Must be able to format with
        `od_to_fname_template.format(segment_params=segment_params)`

    process_count:
        The number of processes to use when combining the matrices.
        See `normits_demand.concurrency.multiprocessing.multiprocess()`
        for more information.

    Returns
    -------
    None
    """
    # Build the arguments to multiprocess
    kwarg_list = list()
    for segment_params in segmentation:
        # Generate input file names
        od_from_fname = segmentation.generate_file_name_from_template(
            template=od_from_fname_template, segment_params=segment_params
        )
        od_to_fname = segmentation.generate_file_name_from_template(
            template=od_to_fname_template, segment_params=segment_params
        )
        input_paths = [
            import_dir / od_from_fname,
            import_dir / od_to_fname,
        ]

        # Generate output filename
        export_path = export_dir / segmentation.generate_file_name_from_template(
            template=od_fname_template, segment_params=segment_params
        )

        print(input_paths)
        print(export_path)
        print("\n\n")

        kwarg_list.append({
            "input_paths": input_paths,
            "export_path": export_path,
        })

    # Multiprocess
    multiprocessing.multiprocess(
        fn=_combine_matrices,
        kwargs=kwarg_list,
        process_count=process_count
    )