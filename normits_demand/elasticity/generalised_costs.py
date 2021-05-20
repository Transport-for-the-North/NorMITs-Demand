# -*- coding: utf-8 -*-
"""
    Module containing the functionality for performing the elasticty calculations to
    adjust the demand.
"""

##### IMPORTS #####
# Standard imports
import itertools

from pathlib import Path

from typing import Dict
from typing import List
from typing import Union


# Third party imports
import numpy as np
import pandas as pd

# Local imports
import normits_demand as nd
from normits_demand import constants as consts

from normits_demand.utils import general as du
from normits_demand.models import efs_zone_translator as zt
from normits_demand.elasticity import constants as ec


class CostBuilder:
    """Builds future year cost dataframes for scenarios."""

    _valid_modes = list(ec.MODE_ID.keys())
    _valid_purposes = list(consts.USER_CLASS_PURPOSES.keys())
    _valid_e_types = list(ec.GC_ELASTICITY_TYPES.keys())

    _e_types_dtypes = {
        'mode': str,
        'cost_component': str,
        'elasticity_type': str,
    }

    _cost_adjustment_dtypes = {
        "mode": str,
        "cost_component": str,
        "adj_type": str,
    }

    # Column name and dtype lookup for the GC parameters input file
    _gc_parameters_dtypes = {
        "yr": str,
        "m": str,
        "p": str,
        "vot": float,
        "voc": float,
    }

    _gc_parameters_mode = {
        'car': ['vot', 'voc'],
        'rail': ['vot'],
    }

    def __init__(self,
                 # Input files
                 vot_voc_path: nd.PathLike,
                 cost_adj_path: nd.PathLike,

                 # Parameters
                 years: List[int],
                 modes: List[str] = None,
                 purposes: List[str] = None,

                 # Optional input files
                 elasticity_types_path: nd.PathLike = None,
                 ):
        # Validate inputs
        try:
            valid_years = [int(x) for x in years]
        except ValueError:
            raise ValueError(
                "Expected a list of integer years. Got %s"
                % years
            )

        # Validate inputs
        modes = self._valid_modes if modes is None else modes
        if not all([x in self._valid_modes for x in modes]):
            raise ValueError(
                "Given an invalid mode. Expected only %s\nGot %s"
                % (self._valid_modes, modes)
            )

        purposes = self._valid_purposes if purposes is None else purposes
        if not all([x in self._valid_purposes for x in purposes]):
            raise ValueError(
                "Given an invalid purpose. Expected only %s\nGot %s"
                % (self._valid_purposes, purposes)
            )

        # Parameters
        self.years = valid_years
        self.years_str = [str(x) for x in valid_years]
        self.modes = modes
        self.purposes = purposes

        # Paths
        self.vot_voc_path = vot_voc_path
        self.cost_adj_path = cost_adj_path

        self.e_types = None
        if elasticity_types_path is not None:
            self.set_elasticity_types(elasticity_types_path)

        # Read in files
        self.cost_adj = pd.read_csv(cost_adj_path)

    def set_elasticity_types(self,
                             elasticity_types_path: nd.PathLike,
                             ) -> None:
        """

        Parameters
        ----------
        elasticity_types_path:
            Full path to the csv containing the mapping of modes and costs
            to the elasticities that should be used

        Returns
        -------
        None
        """
        # Read in the file
        dtypes = self._e_types_dtypes
        df = pd.read_csv(elasticity_types_path, usecols=dtypes.keys(), dtype=dtypes)
        self.e_types = df

    def get_vot_voc(self,
                    year_col: str = 'yr',
                    mode_col: str = 'm',
                    purpose_col: str = 'p',
                    vot_col: str = 'vot',
                    voc_col: str = 'voc',
                    ) -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
        """Reads the generalised cost parameters CSV file.

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
        rename = {
            year_col: 'yr',
            mode_col: 'm',
            purpose_col: 'p',
            vot_col: 'vot',
            voc_col: 'voc',
        }
        rev_rename = {v: k for k, v in rename.items()}

        # Read in and rename to standard names
        dtypes = {rev_rename[k]: v for k, v in self._gc_parameters_dtypes.items()}
        in_file = pd.read_csv(self.vot_voc_path, usecols=dtypes.keys(), dtype=dtypes)
        in_file.rename(columns=rename, inplace=True)

        # Find out if any of the wanted values are missing
        missing_years = [x for x in self.years_str if x not in in_file["yr"].unique()]
        missing_modes = [x for x in self.modes if x not in in_file["m"].unique()]
        missing_purposes = [x for x in self.purposes if x not in in_file["p"].unique()]

        # If things were missing, build an error message
        msg = ""
        if missing_years != list():
            msg += f"Years missing: {missing_years} "
        if missing_modes != list():
            msg += f"Year - mode pairs missing: {missing_modes}"
        if missing_purposes != list():
            msg += f"Year - mode - purpose pairs missing: {missing_purposes}"
        if msg != "":
            raise ValueError(msg + f" from: {self.vot_voc_path.name}")

        # Grab all the wanted in_file from the file
        for yr, m, p in itertools.product(self.years_str, self.modes, self.purposes):
            # Get the rows for these parameters
            mask = (
                (in_file["yr"] == yr)
                & (in_file["m"] == m)
                & (in_file["p"] == p)
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
            cols = self._gc_parameters_mode[m]
            gc_params[yr][p][m] = dict(in_file.loc[mask, cols].iloc[0])

        return du.defaultdict_to_regular(gc_params)

    def get_cost_changes(self,
                         ignore_cols: List[str] = None,
                         ) -> pd.DataFrame:
        """Read the elasticity cost changes file and check for missing data.

        Returns
        -------
        ignore_cols:
            Any additional cols in self.cost_adj_path that should be ignored
            when reading in.

        Raises
        ------
        KeyError
            If an elasticity_type is given that is not present
            in `GC_ELASTICITY_TYPES` lookup.

        nd.NormitsDemandError:
            If more than one elasticity type is found for a cost and mode
            combination

        """
        # Init
        dtypes = self._cost_adjustment_dtypes
        cost_adj = pd.read_csv(self.cost_adj_path, dtype=dtypes)

        # Assume all non-defined columns are years
        ignore_cols = list() if ignore_cols is None else ignore_cols
        ignore_cols += dtypes.keys()
        given_years = du.list_safe_remove(list(cost_adj), ignore_cols)

        # Make sure they are actually numbers left
        try:
            given_years = [int(x) for x in given_years]
        except ValueError:
            raise ValueError(
                "Cannot convert all years to integers. Tried to convert %s "
                "and failed." % given_years
            )

        # Check we can handle all the years asked for
        max_given_year = max(given_years)
        min_given_year = min(given_years)
        for year in self.years:
            if not (min_given_year <= year <= max_given_year):
                raise ValueError(
                    "Wanted year is out of range to extrapolate. Asked to get "
                    "year %s, but given data only from %s to %s.\n"
                    "From file: %s"
                    % (year, min_given_year, max_given_year, self.cost_adj_path)
                )

        # ## EXTRAPOLATE VALUES ## #
        for year in self.years:
            # Skip if we already have it
            if year in given_years:
                continue

            # Find the two numbers this year sits between
            lower = None
            upper = None
            for l, u in zip(given_years[:-1], given_years[1:]):
                if l < year < u:
                    lower = l
                    upper = u
                    break

            if lower is None or upper is None:
                raise nd.NormitsDemandError(
                    "Lower and/or Upper values have not been set. This "
                    "shouldn't be possible. Perhaps an error check has been "
                    "missed somewhere?"
                )

            # Extrapolate!
            year_diff = upper - lower
            val_diff = cost_adj[str(upper)] - cost_adj[str(lower)]
            wanted_year_diff = year - lower
            wanted_diff = val_diff * wanted_year_diff / year_diff
            cost_adj[str(year)] = cost_adj[str(upper)] + wanted_diff

        # ## ADD ELASTICITY TYPES ## #
        def get_e_type(x):
            m = (
                (self.e_types['mode'] == x['mode'])
                & (self.e_types['cost_component'] == x['cost_component'])
            )
            e_type = self.e_types[m]

            if len(e_type) > 1:
                raise nd.NormitsDemandError(
                    "Found more than one elasticity type for mode '%s' "
                    "and cost '%s'"
                    % (x['mode'], x['cost_component'])
                )

            elif len(e_type) == 1:
                x['e_type'] = e_type['elasticity_type'].values[0]

            else:
                # Must have found nothing
                x['e_type'] = 'none'

            return x

        cost_adj['e_type'] = 'none'
        cost_adj = cost_adj.apply(get_e_type, axis='columns')

        # Drop any that we don't have an elasticity for
        mask = (cost_adj['e_type'] == 'none')
        cost_adj = cost_adj[~mask].copy()

        # Check for unknown elasticity types
        e_types = cost_adj["e_type"].unique()
        unknown_types = [x for x in e_types if x not in self._valid_e_types]
        if unknown_types != list():
            raise KeyError(
                f"Unknown elasticity_type: {unknown_types}, "
                f"available types are: {self._valid_e_types}"
            )

        # ## CONVERT TO OUTPUT FORMAT ## #
        id_vars = ['e_type', 'adj_type']
        cost_adj = cost_adj.reindex(columns=id_vars + self.years_str)

        cost_adj = pd.melt(
            cost_adj,
            id_vars=id_vars,
            var_name='yr',
            value_name='change'
        )

        return cost_adj


##### FUNCTIONS #####
def _average_matrices(matrices: Dict[str, np.array],
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
    averages = dict()
    missing = list()
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


def gen_cost_car_mins(matrices: Dict[str, np.array],
                      voc: float,
                      vot: float,
                      ) -> np.array:
    """Calculate the generalised cost for cars in minutes.

    Parameters
    ----------
    matrices : Dict[str, np.array]
        The matrices expected are as follows:
        - time: time matrix in minutes;
        - dist: distance matrix in kilometres; and
        - toll: toll matrix in pence.
    voc : float
        The vehicle operating cost, in pence per kilometre.
    vot : float
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
        (matrices["time"])
        + ((voc / vot) * (matrices["dist"]))
        + (matrices["toll"] / vot)
    )


def gen_cost_rail_mins(matrices: Dict[str, np.array],
                       vot: float,
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
    vot : float
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

    # Multiply matrices by given (or default) weighting factors
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
        + (matrices["fare"] / vot)
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


def get_costs(cost_file: Path,
              mode: str,
              zone_system: str,
              zone_translation_folder: Path,
              translation_weights: pd.DataFrame = None,
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

    translation_weights : pd.DataFrame, optional
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
    # Init
    mode = mode.lower()

    # Try to load costs - let user know which columns are missing if cant
    try:
        costs = pd.read_csv(cost_file, usecols=ec.COST_LOOKUP[mode].values())
    except ValueError as e:
        loc = str(e).find("columns expected")
        e_str = str(e)[loc:] if loc != -1 else str(e)
        raise ValueError(f"Columns missing from {mode} cost, {e_str}") from e

    # Rename columns to standard format across modes
    costs.rename(
        columns={v: k for k, v in ec.COST_LOOKUP[mode].items()},
        inplace=True,
    )

    # Figure out the number of zeroes in the costs
    cost_cols = costs.columns.tolist()
    cost_cols.remove("origin")
    cost_cols.remove("destination")
    total_zeros = (costs[cost_cols] <= 0).all(axis=1).sum()
    print(
        f"{total_zeros} ({total_zeros / len(costs):.2%}) "
        "OD pairs are 0 for {mode} for costs"
    )

    # Convert zone system if required
    if zone_system != ec.COMMON_ZONE_SYSTEM:
        # Get the translation
        lookup = du.get_zone_translation(
            import_dir=zone_translation_folder,
            from_zone=zone_system,
            to_zone=ec.COMMON_ZONE_SYSTEM,
            return_dataframe=True,
        )

        # Check that weights are the right type
        if not isinstance(translation_weights, pd.DataFrame):
            raise TypeError(
                "'demand' is '%s', expected 'DataFrame'"
                % type(translation_weights).__name__
            )

        # Translate with weights!
        zone_systems = [zone_system, ec.COMMON_ZONE_SYSTEM]
        zone_cols = ["%s_zone_id" % x for x in zone_systems]
        costs, _ = zt.translate_matrix(
            matrix=costs,
            lookup=lookup,
            lookup_cols=zone_cols,
            square_format=False,
            zone_cols=["origin", "destination"],
            aggregation_method="weighted_average",
            weights=zt.square_to_list(translation_weights),
        )

    # Convert origin/destination columns to integers
    for c in ("origin", "destination"):
        costs[c] = pd.to_numeric(costs[c], downcast="integer")

    return costs.sort_values(["origin", "destination"])


def gen_cost_mode(costs: Union[pd.DataFrame, float],
                  mode: str,
                  **kwargs
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
        gc = gen_cost_car_mins({i: cost_to_array(i) for i in ("time", "dist", "toll")}, **kwargs)
    elif mode == "rail":
        gc = gen_cost_rail_mins({
            i: cost_to_array(i)
            for i in ("walk", "wait", "ride", "fare", "num_int")
        }, **kwargs)
    else:
        gc = costs
    return gc


def calculate_gen_costs(costs: Dict[str, Union[pd.DataFrame, float]],
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


# TODO(BT): Remove all references to this function
def read_gc_parameters(path: Path,
                       years: List[str],
                       modes: List[str],
                       purposes: List[str],
                       ) -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
    """Reads the generalised cost parameters CSV file.

    Wrapper around the cost builder - deprecated!!

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
    cost_builder = CostBuilder(
        years=years,
        modes=modes,
        purposes=purposes,
        vot_voc_path=path,
        cost_adj_path=None,
    )

    return cost_builder.get_vot_voc()
