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


# Local imports
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
