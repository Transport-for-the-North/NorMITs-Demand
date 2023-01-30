# -*- coding: utf-8 -*-
"""
"""
from __future__ import annotations

# Built-Ins
import pathlib
import logging

from typing import Any
from typing import Mapping
from typing import Literal
from typing import Iterable

# Third Party
import numpy as np
import pandas as pd

from caf.toolkit import concurrency


# Local Imports
# pylint: disable=import-error,wrong-import-position

# pylint: enable=import-error,wrong-import-position

# # # CONSTANTS # # #
LOG = logging.getLogger(__name__)

# # # CLASSES # # #

# # # FUNCTIONS # # #
# ## Private ## #
def _grow_matrix_pd():
    pass


def _grow_matrix_np():
    pass


# ## Public ## #
def grow_matrix(
    matrix_in_path: pathlib.Path,
    matrix_out_path: pathlib.Path,
    growth_factor: int | float,
    growth_zones: list[Any],
    matrix_type: Literal["pandas", "numpy"],
) -> None:
    pass


def grow_matrices(
    matrix_in_dir: pathlib.Path,
    matrix_out_dir: pathlib.Path,
    matrix_io_names: Mapping[str, str] | Iterable[str],
    growth_factor: int | float,
    growth_zones: list[Any] | None = None,
    matrix_type: Literal["pandas", "numpy"] | None = None,
    process_count: int = -2,
) -> None:
    """Apply a flat growth factor to a set of matrices across set zones

    Parameters
    ----------
    matrix_in_dir:
        Path to a directory to read matrices in from.

    matrix_out_dir:
        Path to a directory to write the grown matrices to.

    matrix_io_names:
        A mapping of `{file_in_name: file_out_name}`, where `file_in_name`
        is the filename in `matrix_in_dir` to read in, grow, and write out
        as `file_out_name` in `matrix_out_dir.
        If an Iterable is given, the same name is assumed to be used for both
        the input and output.
        If None is given, all filenames in matrix_in_dir are grown.

    growth_factor:
        The growth factor to apply to all the input matrices to generate
        the output matrices.

    growth_zones:
        The zones to apply the growth factor to. If left as None, then growth
        is applied to all zone pairs.

    matrix_type:
        Determines which growth method to use based on the type of matrix being
        loaded in. If left as None, the first matrix is read in to determine
        the type. A matrix is determined to be in numpy format only if a square
        matrix is found (there is the same number of rows and columns).

    process_count:
        The number of processes to use when multiprocessing the growth
        application. See `caf.toolkit.concurrency.multiprocess` for full
        documentation.

    Returns
    -------
    None
    """
    # Convert to standard format
    if not isinstance(matrix_io_names, Mapping):
        matrix_io_names = {x: x for x in matrix_io_names}

    # Determine matrix type
    if matrix_type is None:
        first_i_matrix = matrix_in_dir / next(iter(matrix_io_names))
        mat = pd.read_csv(first_i_matrix)
        if isinstance(mat, pd.DataFrame):
            matrix_type = "pandas"
        elif isinstance(mat, np.ndarray):
            matrix_type = "numpy"
        else:
            raise TypeError(
                "Tried to determine the type of the input matrices but got an "
                f"unexpected type! Read in the matrix at {first_i_matrix}\n"
                f"Got a type of {type(mat)}, which is unsupported. Only pd.Dataframe, "
                "np.ndarray are supported."
            )

    # Call as a multiprocess
    # kwarg_list = list()
    # for in_name, out_name in matrix_io_names.items():
    #     kwarg_list.append({
    #         "matrix_in_path": matrix_in_dir / in_name,
    #         "matrix_out_path": matrix_out_dir / in_name,
    #         "growth_factor": growth_factor,
    #         "growth_zones": growth_zones,
    #         "matrix_type": matrix_type,
    #     })
    #
    # concurrency.multiprocess(
    #     fn=grow_matrix,
    #     kwarg_list=kwarg_list,
    #     process_count=process_count,
    # )


    # Testing!
    for in_name, out_name in matrix_io_names.items():
        grow_matrix(
            matrix_in_path=matrix_in_dir / in_name,
            matrix_out_path=matrix_out_dir / in_name,
            growth_factor=growth_factor,
            growth_zones=growth_zones,
            matrix_type=matrix_type,
        )


