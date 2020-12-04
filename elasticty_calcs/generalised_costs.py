# -*- coding: utf-8 -*-
"""
    Module containing the functionality for performing the elasticty calculations to
    adjust the demand.
"""

##### IMPORTS #####
# Standard imports
from typing import Dict, List

# Third party imports
import numpy as np


##### CONSTANTS #####
RAIL_GC_FACTORS = {"walk": 1.75, "wait": 2, "interchange_penalty": 5}


##### FUNCTIONS #####
def _average_matrices(
    matrices: Dict[str, np.array], expected: List[str], weights: np.array = None
) -> Dict[str, float]:
    """Calculate the weighted average of expected matrices.

    Parameters
    ----------
    matrices : Dict[str, np.array]
        Matrices to calculate averages.
    expected : List[str]
        List of names of matrices to expect, will raise KeyError if any
        values in this list aren't present as keys in `matrices`.
    weights : np.array, optional
        Array for calculating the weighted average of the input matrices,
        by default None. If None then the mean of the input matrices will
        be used instead.

    Returns
    -------
    Dict[str, float]
        Average value for each of the given matrices.

    Raises
    ------
    KeyError
        If any of the expected matrices aren't provided.
    """
    averages = {}
    missing = []
    for nm in expected:
        try:
            averages[nm] = np.average(matrices[nm], weights=weights)
        except KeyError:
            missing.append(nm)
    if missing:
        raise KeyError(f"The following matrices are missing: {missing!s}")
    return averages


def gen_cost_car_mins(
    matrices: Dict[str, np.array],
    vc: float,
    vt: float,
    weights: np.array = None,
) -> float:
    """Calculate the generalised cost for cars in minutes.

    Parameters
    ----------
    matrices : Dict[str, np.array]
        The matrices expected are as follows:
        - time: time matrix in seconds;
        - dist: distance matrix in metres; and
        - toll: toll matrix in pence.
    vc : float
        The vehicle operating cost, in pence per kilometre.
    vt : float
        The value of time * occupancy value, in pence per minute.
    weights : np.array, optional
        Array for calculating the weighted average of the input matrices,
        by default None. If None then the mean of the input matrices will
        be used instead.

    Returns
    -------
    float
        The generalised cost in minutes.

    Raises
    ------
    KeyError
        If any of the expected matrices aren't provided.
    """
    averages = _average_matrices(matrices, ["time", "dist", "toll"], weights=weights)

    return (
        (averages["time"] / 60)
        + ((vc / vt) * (averages["dist"] / 1000))
        + (averages["toll"] / vt)
    )


def gen_cost_rail_mins(
    matrices: Dict[str, np.array],
    vt: float,
    factors: Dict[str, float] = None,
    num_interchanges: int = 0,
) -> float:
    """Calculate the generalised cost for rail in minutes.

    Parameters
    ----------
    matrices : Dict[str, np.array]
        The matrices expected are as follows:
        - walk: the walk time, in minutes;
        - wait: the wait time, in minutes;
        - ride: the in-vehicle time, in minutes; and
        - fare: the fare, in pence.
    vt : float
        Value of time in pence per minute.
    factors : Dict[str, float], optional
        The weighting factors for walk and wait matrices and the interchange
        penalty in minutes, by default None. The default values given in
        `RAIL_GC_FACTORS` will be used for any that are missing.
    num_interchanges : int, optional
        The number of interchanges made, by default 0.

    Returns
    -------
    float
        The generalised cost in minutes.

    Raises
    ------
    KeyError
        If any of the expected matrices aren't provided.
    """
    averages = _average_matrices(matrices, ["walk", "wait", "ride", "fare"])

    # Multply averages by given (or default) weighting factors
    factors = RAIL_GC_FACTORS if factors is None else factors
    for nm in ["walk", "wait"]:
        fac = RAIL_GC_FACTORS[nm] if nm not in factors.keys() else factors[nm]
        averages[nm] = averages[nm] * fac

    nm = "interchange_penalty"
    inter_factor = RAIL_GC_FACTORS[nm] if nm not in factors.keys() else factors[nm]
    return (
        averages["walk"]
        + averages["wait"]
        + averages["ride"]
        + (averages["fare"] / vt)
        + (num_interchanges * inter_factor)
    )
