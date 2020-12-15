# -*- coding: utf-8 -*-
"""
    Module containing the functionality for performing the elasticty calculations to
    adjust the demand.
"""

##### IMPORTS #####
# Standard imports
from typing import Dict, List
from pathlib import Path

# Third party imports
import numpy as np
import pandas as pd

# Local imports
from demand_utilities.utils import zone_translation_df
from zone_translator import translate_matrix
from .utils import COMMON_ZONE_SYSTEM


##### CONSTANTS #####
RAIL_GC_FACTORS = {"walk": 1.75, "wait": 2, "interchange_penalty": 5}
COST_LOOKUP = {
    "rail": {
        "origin": "from_model_zone_id",
        "destination": "to_model_zone_id",
        "walk": "AE_cost",
        "wait": "Wait_Actual_cost",
        "ride": "IVT_cost",
        "fare": "fare_cost",
        "num_int": "Interchange_cost",
    },
    "car": {
        "origin": "from_model_zone_id",
        "destination": "to_model_zone_id",
        "time": "time",
        "dist": "distance",
        "toll": "toll",
    },
}


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


def _check_matrices(
    matrices: Dict[str, np.array], expected: List[str]
) -> Dict[str, float]:
    """Check if all expected matrices are given and are the same shape.

    Parameters
    ----------
    matrices : Dict[str, np.array]
        Matrices to check.
    expected : List[str]
        List of names of matrices to expect, will raise KeyError if any
        values in this list aren't present as keys in `matrices`.

    Returns
    -------
    Dict[str, float]
        Copies of the original matrices given.

    Raises
    ------
    KeyError
        If any of the expected matrices aren't provided.
    ValueError
        If all the matrices aren't the same shape.
    """
    mats = {}
    missing = []
    shapes = []
    for nm in expected:
        try:
            mats[nm] = matrices[nm].copy()
            shapes.append(matrices[nm].shape)
        except KeyError:
            missing.append(nm)
    if missing:
        raise KeyError(f"The following matrices are missing: {missing!s}")
    if not all(s == shapes[0] for s in shapes):
        msg = ", ".join(f"{nm} = {shapes[i]}" for i, nm in enumerate(mats))
        raise ValueError(f"Matrices are not all the same shape: {msg}")
    return mats


def gen_cost_car_mins(
    matrices: Dict[str, np.array],
    vc: float,
    vt: float,
) -> np.array:
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
    np.array
        The generalised cost in minutes.

    Raises
    ------
    KeyError
        If any of the expected matrices aren't provided.
    """
    matrices = _check_matrices(matrices, ["time", "dist", "toll"])

    return (
        (matrices["time"] / 60)
        + ((vc / vt) * (matrices["dist"] / 1000))
        + (matrices["toll"] / vt)
    )


def gen_cost_rail_mins(
    matrices: Dict[str, np.array],
    vt: float,
    factors: Dict[str, float] = None,
) -> np.array:
    """Calculate the generalised cost for rail in minutes.

    Parameters
    ----------
    matrices : Dict[str, np.array]
        The matrices expected are as follows:
        - walk: the walk time, in minutes;
        - wait: the wait time, in minutes;
        - ride: the in-vehicle time, in minutes;
        - fare: the fare, in pence; and
        - num_int: the number of interchanges.
    vt : float
        Value of time in pence per minute.
    factors : Dict[str, float], optional
        The weighting factors for walk and wait matrices and the interchange
        penalty in minutes, by default None. The default values given in
        `RAIL_GC_FACTORS` will be used for any that are missing.

    Returns
    -------
    np.array
        The generalised cost in minutes.

    Raises
    ------
    KeyError
        If any of the expected matrices aren't provided.
    """
    matrices = _check_matrices(matrices, ["walk", "wait", "ride", "fare", "num_int"])

    # Multply matrices by given (or default) weighting factors
    factors = RAIL_GC_FACTORS if factors is None else factors
    for nm in ["walk", "wait"]:
        fac = RAIL_GC_FACTORS[nm] if nm not in factors.keys() else factors[nm]
        matrices[nm] = matrices[nm] * fac

    nm = "interchange_penalty"
    inter_factor = RAIL_GC_FACTORS[nm] if nm not in factors.keys() else factors[nm]
    return (
        matrices["walk"]
        + matrices["wait"]
        + matrices["ride"]
        + (matrices["fare"] / vt)
        + (matrices["num_int"] * inter_factor)
    )


def gen_cost_elasticity_mins(
    elasticity: float,
    gen_cost: np.array,
    cost: np.array,
    demand: np.array,
    cost_factor: float = None,
) -> float:
    """Calculate the weighted average generalised cost elasticity.

    Parameters
    ----------
    elasticity : float
        Implied elasticity value.
    gen_cost : np.array
        Generalised cost in minutes.
    cost : np.array
        Cost in either minutes or monetary units.
    demand : np.array
        Demand for calculating the weighted average.
    cost_factor : float, optional
        Factor to convert cost into minutes, if None (default)
        uses a factor of 1.0.

    Returns
    -------
    float
        The generalised cost elasticity in minutes.
    """
    averages = _average_matrices(
        {"gc": gen_cost, "cost": cost}, ["gc", "cost"], weights=demand
    )
    if cost_factor is None:
        cost_factor = 1.0
    return elasticity * (averages["gc"] / (averages["cost"] * cost_factor))


def get_costs(
    cost_file: Path, mode: str, zone_system: str, zone_translation_folder: Path
) -> pd.DataFrame:
    """Reads the given cost file, expected columns are in `COST_LOOKUP`.

    Parameters
    ----------
    cost_file : Path
        Path to the CSV file containing cost data.
    mode : str
        The mode of the costs, either rail or car.
    zone_system : str
        The zone system of the costs, the cost will be translated
        to the `COMMON_ZONE_SYSTEM` if required.
    zone_translation_folder : Path
        Path to the folder containing the zone translation lookups.

    Returns
    -------
    pd.DataFrame
        Costs using the expected columns in `COST_LOOKUP`.

    Raises
    ------
    ValueError
        If any expected columns are missing.
    """
    try:
        costs = pd.read_csv(cost_file, usecols=COST_LOOKUP[mode.lower()].values())
    except ValueError as e:
        loc = str(e).find("columns expected")
        e_str = str(e)[loc:] if loc != -1 else str(e)
        raise ValueError(f"Columns missing from {mode} cost, {e_str}") from e

    # Convert zone system if required
    if zone_system != COMMON_ZONE_SYSTEM:
        lookup = zone_translation_df(
            zone_translation_folder, zone_system, COMMON_ZONE_SYSTEM
        )
        costs = translate_matrix(
            costs,
            lookup,
            [f"{zone_system}_zone_id", f"{COMMON_ZONE_SYSTEM}_zone_id"],
            square_format=False,
            zone_cols=[COST_LOOKUP[mode][i] for i in ("origin", "destination")],
        )

    return costs
