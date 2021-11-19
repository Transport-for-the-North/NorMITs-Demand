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
import warnings

# Third Party
import numpy as np

# Local Imports


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
