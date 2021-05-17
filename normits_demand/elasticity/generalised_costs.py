# -*- coding: utf-8 -*-
"""
    Module containing the functionality for performing the elasticty calculations to
    adjust the demand.
"""

##### IMPORTS #####
# Standard imports
import itertools

from collections import defaultdict

from pathlib import Path

from typing import Dict
from typing import List
from typing import Union



# Third party imports
import numpy as np
import pandas as pd

# Local imports
from normits_demand.utils import general as du
from normits_demand.models import efs_zone_translator as zt
from normits_demand.elasticity import constants as ec


##### FUNCTIONS #####
def _average_matrices(
    matrices: Dict[str, np.array],
    expected: List[str],
    weights: np.array = None,
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
            # If matrix is scalar (for other modes) don't need to calculate average
            if isinstance(matrices[nm], (float, int)):
                averages[nm] = matrices[nm]
            else:
                # If demand weights are scalar then just calculate mean
                if isinstance(weights, (float, int)):
                    weights = None
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
    matrices = _check_matrices(
        matrices, ["walk", "wait", "ride", "fare", "num_int"]
    )

    # Multply matrices by given (or default) weighting factors
    factors = ec.RAIL_GC_FACTORS if factors is None else factors
    for nm in ["walk", "wait"]:
        fac = factors.get(nm, ec.RAIL_GC_FACTORS[nm])
        matrices[nm] = matrices[nm] * fac

    nm = "interchange_penalty"
    inter_factor = factors.get(nm, ec.RAIL_GC_FACTORS[nm])
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
    cost_file: Path,
    mode: str,
    zone_system: str,
    zone_translation_folder: Path,
    demand: pd.DataFrame = None,
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
    demand : pd.DataFrame, optional
        Relevant demand matrix (square format) for calculating
        weight average cost when converting between zone systems,
        only required if `zone_system` != `COMMON_ZONE_SYSTEM`.

    Returns
    -------
    pd.DataFrame
        Costs using the expected columns in `COST_LOOKUP`.

    Raises
    ------
    ValueError
        If any expected columns are missing.
    """
    mode = mode.lower()
    try:
        costs = pd.read_csv(cost_file, usecols=ec.COST_LOOKUP[mode].values())
    except ValueError as e:
        loc = str(e).find("columns expected")
        e_str = str(e)[loc:] if loc != -1 else str(e)
        raise ValueError(f"Columns missing from {mode} cost, {e_str}") from e
    costs.rename(
        columns={v: k for k, v in ec.COST_LOOKUP[mode].items()},
        inplace=True,
    )
    cost_cols = costs.columns.tolist()
    cost_cols.remove("origin")
    cost_cols.remove("destination")
    total_zeros = (costs[cost_cols] <= 0).all(axis=1).sum()
    print(
        f"{total_zeros} ({total_zeros / len(costs):.2%}) "
        "OD pairs have 0 in all cost values"
    )

    # Convert zone system if required
    if zone_system != ec.COMMON_ZONE_SYSTEM:
        lookup = du.get_zone_translation(
            zone_translation_folder,
            zone_system,
            ec.COMMON_ZONE_SYSTEM,
            return_dataframe=True,
        )
        if not isinstance(demand, pd.DataFrame):
            raise TypeError(
                f"'demand' is '{type(demand).__name__}', expected 'DataFrame'"
            )
        costs, _ = zt.translate_matrix(
            costs,
            lookup,
            [f"{zone_system}_zone_id", f"{ec.COMMON_ZONE_SYSTEM}_zone_id"],
            square_format=False,
            zone_cols=["origin", "destination"],
            aggregation_method="weighted_average",
            weights=zt.square_to_list(demand),
        )

    # Convert origin/destination columns to integers
    for c in ("origin", "destination"):
        costs[c] = pd.to_numeric(costs[c], downcast="integer")
    return costs.sort_values(["origin", "destination"])


def gen_cost_mode(
    costs: Union[pd.DataFrame, float], mode: str, **kwargs
) -> Union[np.array, float]:
    """Calculate generalised cost (GC) for a single mode using the relevant function.

    Parameters
    ----------
    costs : Union[pd.DataFrame, float]
        Cost dataframe, or value, for given mode.
    mode : str
        Name of the mode GC is being calculated for.
    kwargs : Keyword Arguments
        Passed to `gen_cost_car_mins` or `gen_cost_rail_mins`
        functions if `mode` is "car" or "rail", respectively,
        not used for any other modes. The following arguments
        are expected:
        For `gen_cost_car_mins`:
        - vt : float
        - vc : float
        For `gen_cost_rail_mins`:
        - vt : float
        - factors : Dict[str, float], optional

    Returns
    -------
    Union[np.array, float]
        The generlised cost as an array if `costs` is a DataFrame
        or a float if `costs` is a float.
    """
    mode = mode.lower()
    cost_to_array = lambda v: costs.pivot(
        index="origin", columns="destination", values=v
    ).values
    if mode == "car":
        gc = gen_cost_car_mins(
            {i: cost_to_array(i) for i in ("time", "dist", "toll")}, **kwargs
        )
    elif mode == "rail":
        gc = gen_cost_rail_mins(
            {
                i: cost_to_array(i)
                for i in ("walk", "wait", "ride", "fare", "num_int")
            },
            **kwargs,
        )
    else:
        gc = costs
    return gc


def calculate_gen_costs(
    costs: Dict[str, Union[pd.DataFrame, float]],
    gc_params: Dict[str, Dict[str, float]],
) -> Dict[str, Union[np.array, float]]:
    """Calculate the generalised costs for rail, car, bus, active and no-travel.

    Generalised cost is a scalar for bus, active and no-travel, will be set
    to cost.

    Parameters
    ----------
    costs : Dict[str, Union[pd.DataFrame, float]]
        Cost DataFrames for rail and car, other modes are optional
        but should be scalar values if given.
    gc_params : Dict[str, Dict[str, float]]
        Parameters for Value of time * occupancy and vehicle operating
        costs for car and just value of time for rail.

    Returns
    -------
    Dict[str, np.array]
        Generalised cost arrays for rail and car and scalar values
        for bus, active and no-travel.
    """
    gc = {}
    for m, cost in costs.items():
        gc[m] = gen_cost_mode(cost, m, **gc_params.get(m, {}))

    return gc


def read_gc_parameters(path: Path,
                       years: List[str],
                       modes: List[str],
                       purposes: List[str],
                       ) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Reads the generalised cost parameters CSV file.

    Parameters
    ----------
    path : Path
        Path to the parameters file.

    years : List[str]
        List of the years required.

    modes : List[str]
        List of the modes required.

    purposes : List[str]
        List of the purposes required.

    Returns
    -------
    gc_cost_components:
        Nested dictionary containing the parameters split
        by years, mode, purpose e.g.
        {
            "2018": {
                'commute': {
                    "car": {"vot": 16.2, "voc": 9.45},
                    "rail": {"vot": 16.4},
                },
            },
        }

    Raises
    ------
    ValueError:
        If any of the purposes or modes given are not valid

    ValueError:
        If there are any years, modes, or purposes missing
        from the file.

    ValueError:
        If none, or more than one line is found for a combination of
        year, mode, and purpose.
    """
    # Init
    gc_params = du.nested_dictionary(3, None)

    # TODO(BT): Move these into standard functions
    # Validate inputs
    valid_modes = ['car', 'rail']
    if not all([x in valid_modes for x in modes]):
        raise ValueError(
            "Given an invalid mode. Expected only %s\nGot %s"
            % (valid_modes, modes)
        )

    valid_purposes = ['commute', 'business', 'other']
    if not all([x in valid_purposes for x in purposes]):
        raise ValueError(
            "Given an invalid purpose. Expected only %s\nGot %s"
            % (valid_purposes, purposes)
        )

    # Read in and rename based on constants
    dtypes = dict(ec.GC_PARAMETERS_FILE.values())
    in_file = pd.read_csv(path, usecols=dtypes.keys(), dtype=dtypes)
    in_file.rename(
        columns={v[0]: k for k, v in ec.GC_PARAMETERS_FILE.items()},
        inplace=True,
    )

    # Find out if any of the wanted values are missing
    missing_years = [x for x in years if x not in in_file["year"].unique()]
    missing_modes = [x for x in modes if x not in in_file["mode"].unique()]
    missing_purposes = [x for x in purposes if x not in in_file["purpose"].unique()]

    # If things were missing, build an error message
    msg = ""
    if missing_years != list():
        msg += f"Years missing: {missing_years} "
    if missing_modes != list():
        msg += f"Year - mode pairs missing: {missing_modes}"
    if missing_purposes != list():
        msg += f"Year - mode - purpose pairs missing: {missing_purposes}"
    if msg != "":
        raise ValueError(msg + f" from: {path.name}")

    # Grab all the wanted in_file from the file
    for yr, m, p in itertools.product(years, modes, purposes):
        # Get the rows for these parameters
        mask = (
            (in_file["year"] == yr)
            & (in_file["mode"] == m)
            & (in_file["purpose"] == p)
        )
        data = in_file[mask]

        # Check that we got the right thing
        if len(data) == 0:
            raise ValueError(
                "No data found for:\nyr: %s\nm: %s\np: %s\n"
                % (yr, p, m)
            )

        if len(data) > 1:
            raise ValueError(
                "Multiple lines found for:\nyr: %s\nm: %s\np: %s\n"
                % (yr, p, m)
            )

        # Grab the data we want!
        cols = ec.GC_PARAMETERS_MODE[m]
        gc_params[yr][p][m] = dict(in_file.loc[mask, cols].iloc[0])

    return du.defaultdict_to_regular(gc_params)
