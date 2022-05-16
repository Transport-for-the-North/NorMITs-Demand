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
from typing import List
from typing import Dict

# Third Party
import numpy as np
import pandas as pd

# Local imports
import normits_demand as nd
from normits_demand.utils import math_utils

def check_fh_th_factors(factor_dict: Dict[int, np.array],
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


def split_matrix_by_time_periods(
    mat_24: pd.DataFrame,
    tp_factor_dict: Dict[int, np.ndarray],
) -> Dict[int, pd.DataFrame]:
    """Convert 24hr matrix into tp split matrix

    Parameters
    ----------
    mat_24:
        A pandas DataFrame containing the 24hr Demand. The index and the
        columns will be retained and copied into the produced od_from and
        od_to matrices. This means they should likely be the model zoning.

    tp_factor_dict:
        A Dictionary of {tp: split_factor} values. Where `tp` is an integer time
        period, and `split_factor` is a numpy array of shape `mat_24.shape`
        factors to generate the time period split matrices.

    Returns
    -------
    tp_split_matrices:
        A dictionary in the same format as `tp_factor_dict`, but the values
        will be pandas DataFrames of each matrix at a time period.
    """
    # Split the matrix
    tp_mats = dict.fromkeys(tp_factor_dict.keys())
    for tp, factor_mat in tp_factor_dict.items():
        tp_mats[tp] = mat_24 * factor_mat

    # Validate return matrix totals
    tp_total = np.sum([x.values.sum() for x in tp_mats.values()])

    # OD total should be double the input PA
    if not math_utils.is_almost_equal(tp_total, mat_24.values.sum()):
        raise nd.NormitsDemandError(
            "Tp split matrix total isn't similar enough to mat_24 total."
            "Are the given splitting factors correct?\n"
            f"24hr Matrix total: {float(mat_24.values.sum()):.2f}\n"
            f"tp split matrices: {float(tp_total):.2f}\n"
        )

    return tp_mats
