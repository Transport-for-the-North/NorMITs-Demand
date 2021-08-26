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
import operator

from typing import Any
from typing import List
from typing import Dict
from typing import Callable

# Third Party
import pandas as pd
import numpy as np

# Local imports


def get_wide_mask(df: pd.DataFrame,
                  zones: List[Any] = None,
                  col_zones: List[Any] = None,
                  index_zones: List[Any] = None,
                  join_fn: Callable = operator.and_
                  ) -> np.ndarray:
    """
    Generates a mask for a wide matrix. Returned mask will be same shape as df

    The zones the set the mask for can be set individually with col_zones and
    index_zones, or to the same value with zones.


    Parameters
    ----------
    df:
        The dataframe to generate the mask for

    zones:
        The zones to match to in both the columns and index. If this value
        is set it will overwrite anything passed into col_zones and
        index_zones.

    col_zones:
        The zones to match to in the columns. This value is ignored if
        zones is set.

    index_zones:
        The zones to match to in the index. This value is ignored if
        zones is set.

    join_fn:
        The function to call on the column and index masks to join them.
        By default, a bitwise and is used. See pythons builtin operator
        library for more options.

    Returns
    -------
    mask:
        A mask of true and false values. Will be the same shape as df.
    """
    # Validate input args
    if zones is None:
        if col_zones is None or index_zones is None:
            raise ValueError(
                "If zones is not set, both col_zones and row_zones need "
                "to be set."
            )
    else:
        col_zones = zones
        index_zones = zones

    # Try and cast to the correct types for rows/cols
    try:
        # Assume columns are strings if they are an object
        col_dtype = df.columns.dtype
        col_dtype = str if col_dtype == object else col_dtype
        col_zones = np.array(col_zones, col_dtype)
    except ValueError:
        raise ValueError(
            "Cannot cast the col_zones to the required dtype to match the "
            "dtype of the given df columns. Tried to cast to: %s"
            % str(df.columns.dtype)
        )

    try:
        index_zones = np.array(index_zones, df.index.dtype)
    except ValueError:
        raise ValueError(
            "Cannot cast the index_zones to the required dtype to match the "
            "dtype of the given df index. Tried to cast to: %s"
            % str(df.index.dtype)
        )

    # Create square masks for the rows and cols
    col_mask = np.broadcast_to(df.columns.isin(col_zones), df.shape)
    index_mask = np.broadcast_to(df.index.isin(index_zones), df.shape).T

    # Combine together to get the full mask
    return join_fn(col_mask, index_mask)


def get_internal_mask(df: pd.DataFrame,
                      zones: List[Any] = None,
                      ) -> np.ndarray:
    """
    Generates a mask for a wide matrix. Returned mask will be same shape as df

    Parameters
    ----------
    df:
        The dataframe to generate the mask for

    zones:
        A list of zone numbers that make up the internal zones

    Returns
    -------
    mask:
        A mask of true and false values. Will be the same shape as df.
    """
    return get_wide_mask(df=df, zones=zones, join_fn=operator.and_)


def get_external_mask(df: pd.DataFrame,
                      zones: List[Any] = None,
                      ) -> np.ndarray:
    """
    Generates a mask for a wide matrix. Returned mask will be same shape as df

    Parameters
    ----------
    df:
        The dataframe to generate the mask for

    zones:
        A list of zone numbers that make up the external zones

    Returns
    -------
    mask:
        A mask of true and false values. Will be the same shape as df.
    """
    return get_wide_mask(df=df, zones=zones, join_fn=operator.or_)


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
