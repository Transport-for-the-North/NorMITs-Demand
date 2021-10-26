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

from typing import Union

# Third Party
import numpy as np

# Local Imports


def is_almost_equal(x1: Union[int, float],
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
        between the sum of pure_attractions and fully_segmented_attractions,
        relative to the larger absolute value of pure_attractions or
        fully_segmented_attractions. By default, this is set to 0.0001,
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


def vector_mean_squared_error(vector1: np.array,
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


def curve_convergence(target: np.array,
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
            "\tAchieved: %s"
            % (target.shape, achieved.shape)
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
    convergence = (
        np.sum((achieved - target) ** 2)
        / np.sum((target - np.sum(target) / len(target)) ** 2)
    )

    # Limit between 0 and 1
    return max(1 - convergence, 0)


def get_pa_diff(new_p,
                p_target,
                new_a,
                a_target):
    pa_diff = (((sum((new_p - p_target) ** 2) + sum((new_a - a_target) ** 2))/len(p_target)) ** .5)
    return pa_diff
