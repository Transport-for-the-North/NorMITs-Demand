# -*- coding: utf-8 -*-
"""
Created on: 23/09/2021
Updated on:

Original author: Ben Taylor
Last update made by:
Other updates made by:

File purpose:
Collections of math based utils for normits demand
"""
# Built-Ins
import math
import warnings

from typing import Any
from typing import Dict
from typing import Union
from typing import Tuple

# Third Party
import numpy as np
import pandas as pd

# Local Imports


def check_numeric(check_dict: Dict[str, Any]) -> None:
    """Checks if check_dict values are floats or ints.

    Parameters
    ----------
    check_dict:
        A dictionary of argument names and argument values to check.
        The names are used for the error if the value isn't a numeric.

    Raises
    ------
    ValueError
        If any of the parameters aren't floats or ints,
        includes the parameter name in the message.
    """
    for name, val in check_dict.items():
        if not isinstance(val, (int, float)):
            raise ValueError(
                "%s should be a scalar number (float or int) not %s" % (name, type(val))
            )


def numpy_cast(x: Any, dtype: np.dtype):
    """
    Casts scalar x to dtype using numpy

    Parameters
    ----------
    x:
        The scalar value to cast

    dtype:
        The numpy data type to cast x to

    Returns
    -------
    x_cast:
        x, but cast to dtype
    """
    return np.array(x).astype(dtype).item()


def is_almost_equal(
    x1: Union[int, float],
    x2: Union[int, float],
    rel_tol: float = 0.0001,
    abs_tol: float = 0.0,
) -> bool:
    """Checks if x1 is similar to x2

    Whether or not two values are considered close is determined
    according to given absolute and relative tolerances.

    Parameters
    -----------
    x1:
        The first value to check if close to x2

    x2:
        The second value to check if close to x1

    rel_tol:
        the relative tolerance – it is the maximum allowed difference
        between the sum of x1 and x2,
        relative to the larger absolute value of x1 or
        x2. By default, this is set to 0.0001,
        meaning the values must be within 0.01% of each other.

    abs_tol:
        is the minimum absolute tolerance – useful for comparisons near
        zero. abs_tol must be at least zero.

    Returns
    -------
    is_close:
         Return True if self.sum() and other.sum() are close to each
         other, False otherwise.
    """
    return math.isclose(
        x1,
        x2,
        rel_tol=rel_tol,
        abs_tol=abs_tol,
    )


def vector_mean_squared_error(
    vector1: np.array,
    vector2: np.array,
) -> float:
    """Calculate the mean squared error between 2 vectors

    Parameters
    ----------
    vector1:
        A np.array of the target values to compare x2 to.

    vector2:
        A np.array of the target values to compare x1 to.

    Returns
    -------
    mean_squared_error:
        The mean squared error between vector1 and vector2
    """
    return ((vector1 - vector2) ** 2).mean()


def curve_convergence(
    target: np.array,
    achieved: np.array,
) -> float:
    """Calculate the convergence between two curves.

    Similar to r-squared, but weighted by the target values.

    Parameters
    ----------
    target:
        A np.array listing y values on the curve we are aiming for

    achieved:
        A np.array listing y values on the curve we have achieved

    Returns
    -------
    convergence:
        A float value between 0 and 1. Values closer to one indicate a better
        convergence.

    Raises
    ------
    ValueError:
        If target and achieved are not the same shape
    """
    if target.shape != achieved.shape:
        raise ValueError(
            "Shape of target and achieved do not match.\n"
            "\tTarget: %s\n"
            "\tAchieved: %s" % (target.shape, achieved.shape)
        )

    # Always return 0 if we achieved NaN
    if np.isnan(achieved).sum() > 0:
        return 0

    # If NaN in our target, raise a warning too
    if np.isnan(target).sum() > 0:
        warnings.warn(
            "Found NaN in the target while calculating curve_convergence. "
            "A NaN value in target will mean 0 is always returned."
        )
        return 0

    # Calculate convergence
    convergence = np.sum((achieved - target) ** 2) / np.sum(
        (target - np.sum(target) / len(target)) ** 2
    )

    # Limit between 0 and 1
    return max(1 - convergence, 0)


def nan_report(mat: np.ndarray) -> pd.DataFrame:
    """Create a report of where np.nan values are in mat

    Parameters
    ----------
    mat:
        The numpy array to generate the report for

    Returns
    -------
    report:
        A pandas DataFrame reporting where the np.nan values are.
        Will have a column named "axis_{i}" for each axis i in mat.
    """
    mat_idxs = np.isnan(mat).nonzero()
    idx_cols = {f"axis_{i}": x for i, x in enumerate(mat_idxs)}
    return pd.DataFrame(idx_cols)


def overflow_msg(
    x1: np.ndarray,
    x2: np.ndarray,
    x1_name: str = None,
    x2_name: str = None,
    **kwargs,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Handles overflow error messaging from a numpy divide

    Parameters
    ----------
    x1:
        The array, that when divided by x2, produces overflow errors

    x2:
        The array, that when x1 is divided by it, produces overflow errors

    x1_name:
        The name to give to the x1 column in the output error table. If
        left as None, will default to 'x1'.

    x2_name:
        The name to give to the x2 column in the output error table. If
        left as None, will default to 'x2'.

    Returns
    -------
    overflow_report:
        A pandas dataframe indicating which values in which arrays, at
        which indexes caused the overflow error.
    """
    # Init
    x1_name = "x1" if x1_name is None else x1_name
    x2_name = "x2" if x2_name is None else x2_name

    # Complete the calculation to find the culprit
    with np.errstate(over="ignore"):
        x3 = np.divide(x1, x2, **kwargs)

    # Find all infs
    inf_idxs = np.isinf(x3).nonzero()

    # Format error as a table
    idx_cols = {"axis_%s" % i: x for i, x in enumerate(inf_idxs)}
    other_cols = {x1_name: x1[inf_idxs], x2_name: x2[inf_idxs]}

    all_cols = idx_cols.copy()
    all_cols.update(other_cols)

    return x3, pd.DataFrame(all_cols)


def np_divide_with_overflow_error(*args, **kwargs) -> pd.DataFrame:
    """Call `np.divide` with overflow error raising turned on

    Use in conjunction with 'math_utils.overflow_msg' to get a nice
    print out of the error

    Parameters
    ----------
    *args:
        passed into `np.divide`

    **kwargs:
        passed into `np.divide`

    Returns
    -------
    divide_result:
        The result of: `np.divide(*args, **kwargs)`

    Raises
    ------
    FloatingPointError:
        If an overflow error occurs during the `np.divide` call

    See Also
    --------
    `np.divide`
    """
    # Set up numpy overflow errors
    with np.errstate(over="raise"):
        return np.divide(*args, **kwargs)


def clip_small_non_zero(a: np.ndarray, min_val: float) -> np.ndarray:
    """Clips all small, non-zero values in a up to min_val

    Any 0 values will be left as is, and only the values less than min_val,
    and greater than 0 will be changed to min_val.

    Parameters
    ----------
    a:
        The array to clip

    min_val:
        The minimum non-zero value to allow in a.

    Returns
    -------
    clipped_a:
        a, with all non-zero values clipped to min_val.
    """
    return np.where((min_val > a) & (a > 0), min_val, a)


def get_pa_diff(new_p, p_target, new_a, a_target):
    pa_diff = (
        (sum((new_p - p_target) ** 2) + sum((new_a - a_target) ** 2)) / len(p_target)
    ) ** 0.5
    return pa_diff
