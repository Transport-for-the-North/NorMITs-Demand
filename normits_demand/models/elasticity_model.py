# -*- coding: utf-8 -*-
"""
    Module containing the functions for applying elasticities to the demand
    matrices.
"""

##### IMPORTS #####
# Standard imports
from pathlib import Path
from collections import defaultdict

from typing import List
from typing import Dict
from typing import Tuple
from typing import Union


# Third party imports
import numpy as np
import pandas as pd
import tqdm

# Local imports
from normits_demand.utils import general as du

from normits_demand.models import efs_zone_translator as zt

from normits_demand.elasticity import utils as eu
from normits_demand.elasticity import generalised_costs as gc
from normits_demand.elasticity import constants as ec


##### CLASSES #####
class ElasticityModel:
    """Class for applying elasticity calculations to EFS demand."""

    def __init__(self,
                 input_folders: Dict[str, Path],
                 input_files: Dict[str, Path],
                 output_folders: Dict[str, Path],
                 output_years: List[int],
                 ):
        """Check input files and folders exist and create output folders.

        Parameters
        ----------
        input_folders : Dict[str, Path]
            Paths to the input folders, expects the following keys:
            - elasticity
            - translation
            - rail_demand
            - car_demand
            - rail_costs
            - car_costs
        input_files : Dict[str, Path]
            Paths to the input files, expects the following keys:
            - gc_parameters
            - cost_changes
        output_folders : Dict[str, Path]
            Paths to the output folders, expectes the following keys:
            - car
            - rail
            - others
        output_years : List[int]
            List of years to perform elasticity calculations for.
        """
        self._check_paths(
            input_folders,
            (
                "elasticity",
                "translation",
                "rail_demand",
                "car_demand",
                "rail_costs",
                "car_costs",
            ),
        )
        self._check_paths(
            input_files, ("gc_parameters", "cost_changes"), "file"
        )
        self.input_files = input_files

        self.elasticity_folder = input_folders["elasticity"]
        self.zone_translation_folder = input_folders["translation"]
        self.demand_folders = {
            m: [input_folders[f"{m}_{c}"] for c in ("demand", "costs")]
            for m in ("rail", "car")
        }

        self._check_paths(
            output_folders,
            ("car", "rail", "others"),
            create_folders=True,
        )
        self.output_folder = output_folders
        self.years = output_years

    # BACKLOG: Move to utils/IO
    #   labels: demand merge
    @staticmethod
    def _check_paths(paths: Dict[str, Path],
                     expected: List[str],
                     path_type: str = "folder",
                     create_folders: bool = False,
                     ) -> None:
        """Check if expected paths are given and exist.

        Parameters
        ----------
        paths : Dict[str, Path]
            Dictionary containing paths to the expected folders
            (or files).
        expected : List[str]
            List of expected keys in the `paths` dictionary.
        path_type : str, optional
            Type of paths being provided either 'folder' (default)
            or 'file'.
        create_folders : bool, optional
            If True will create folders that don't exist, if False
            (default) raises FileNotFoundError. If `path_type` is
            'file' then this parameter is ignored.

        Raises
        ------
        KeyError
            If any `expected` folders are missing from `folders`.
        FileNotFoundError
            If any of the paths in `folders` aren't directories
            and `create_folders` is False.
        """
        path_type = path_type.lower()
        if path_type == "folder":
            def check(p): return p.is_dir()
        elif path_type == "file":
            def check(p): return p.is_file()
        else:
            raise ValueError(
                f"path_type should be 'folder' or 'file' not '{path_type}'"
            )

        missing = list()
        not_dir = dict()
        for i in expected:
            if i not in paths.keys():
                missing.append(i)
                continue
            if not check(paths[i]):
                if create_folders and path_type == "folder":
                    paths[i].mkdir(parents=True)
                else:
                    not_dir[i] = paths[i]
        if missing:
            raise KeyError(f"Missing input {path_type}s: {missing}")
        if not_dir:
            raise FileNotFoundError(
                f"{path_type.capitalize()}s could not be found: {not_dir}"
            )

    def apply_all(self):
        """Performs elasticity calculations for all segments provided.

        Segment information is read from `SEGMENTS_FILE` which is
        expected to be found in elasticity folder given.
        """
        segments = eu.read_segments_file(
            self.elasticity_folder / ec.SEGMENTS_FILE
        )
        gc_params = gc.read_gc_parameters(
            self.input_files["gc_parameters"],
            self.years,
            list(ec.MODE_ID.keys()),
        )
        cost_changes = read_cost_changes(
            self.input_files["cost_changes"], self.years
        )
        # Redirect stdout and stderr to tqdm allows tqdm to control
        # how print statements are shown and stops the progress bar
        # formatting from breaking. Note: warnings.warn() messages
        # still cause formatting issues in terminal.
        with eu.std_out_err_redirect_tqdm() as orig_stdout:
            pbar = tqdm.tqdm(
                total=len(segments) * len(self.years),
                desc="Applying elasticities to segments",
                file=orig_stdout,
                dynamic_ncols=True,
                unit="segment",
            )
            for _, row in segments.iterrows():
                for yr in self.years:
                    elasticity_params = {
                        "purpose": str(row["Elast_Purp"]),
                        "market_share": row["Elast_MarketShare"],
                    }
                    demand_params = {
                        "trip_origin": row["EFS_PurpBase"],
                        "matrix_format": "pa",
                        "year": yr,
                        "purpose": str(row["EFS_SubPurpID"]),
                    }
                    if not np.isnan(row["EFS_SkillLevel"]):
                        seg = row["EFS_SkillLevel"]
                    else:
                        seg = row["EFS_IncLevel"]
                    demand_params["segment"] = f"{seg:.0f}"
                    try:
                        self.apply_elasticities(
                            demand_params,
                            elasticity_params,
                            gc_params[yr],
                            cost_changes.loc[cost_changes["year"] == yr],
                        )
                    except Exception as e:  # pylint: disable=broad-except
                        # Catching and printing all errors so program can
                        # continue with other segments
                        name = du.get_dist_name(**demand_params)
                        print(f"{name} - {e.__class__.__name__}: {e}")
                    pbar.update(1)
            pbar.close()

    def apply_elasticities(self,
                           demand_params: Dict[str, str],
                           elasticity_params: Dict[str, str],
                           gc_params: Dict[str, Dict[str, float]],
                           cost_changes: pd.DataFrame,
                           ) -> Dict[str, pd.DataFrame]:
        """Performs elasticity calculation for a single EFS segment.

        Parameters
        ----------
        demand_params : Dict[str, str]
            Parameters to define what demand matrix to
            use expected format:
            {
                "trip_origin": "hb",
                "matrix_format": "pa",
                "year": "2018",
                "purpose": "1",
                "segment": "1",
            }
        elasticity_params : Dict[str, str]
            Parameters to define what elasticity values to
            use, expected format:
            {
                "purpose": "Commuting",
                "market_share": "CarToRail_Moderate",
            }
        gc_params : Dict[str, Dict[str, float]]
            Parameters used in the generlised cost calculations,
            expected format:
            {
                "car": {"vt": 16.2, "vc": 9.45},
                "rail": {"vt": 16.4},
            }
        cost_changes : pd.DataFrame
            The cost changes to be applied to be applied,
            expects the following columns: elasticity_type,
            constraint_matrix_name and percentage_change.

        Returns
        -------
        Dict[str, pd.DataFrame]
            The adjusted demand for all modes.
        """
        elasticities = eu.read_elasticity_file(
            self.elasticity_folder / ec.ELASTICITIES_FILE, **elasticity_params
        )
        # Check if any elasticities aren't provided in ELASTICITIES_FILE
        missing = np.isin(
            cost_changes["elasticity_type"].unique(),
            elasticities["ElasticityType"],
            invert=True,
        )
        if sum(missing) > 0:
            missing = list(cost_changes["elasticity_type"].unique()[missing])
            raise ValueError(
                f"Elasticity values in {ec.ELASTICITIES_FILE} "
                f"missing for the following types: {missing}"
            )

        constraint_matrices = eu.get_constraint_matrices(
            self.elasticity_folder / ec.CONSTRAINTS_FOLDER,
            cost_changes["constraint_matrix_name"].unique().tolist(),
        )
        (
            base_demand,
            rail_ca_split,
            car_reverse,
            car_original,
        ) = self._get_demand(demand_params)
        base_costs = self._get_costs(
            demand_params["purpose"], {"car": car_original}
        )
        del car_original
        base_gc = gc.calculate_gen_costs(base_costs, gc_params)

        # Loop setup
        cols = [
            "elasticity_type",
            "constraint_matrix_name",
            "percentage_change",
        ]
        iterator = cost_changes[cols].itertuples(index=False, name=None)
        demand_adjustment = defaultdict(list)

        # Loop through cost changes file and calculate demand adjustment
        for elast_type, cstr_name, change in iterator:
            adj_dem = calculate_adjustment(
                base_demand,
                base_costs,
                base_gc,
                elasticities.loc[elasticities["ElasticityType"] == elast_type],
                elast_type,
                constraint_matrices[cstr_name],
                change,
                gc_params,
            )

            # Store all our adjustments to apply later
            for mode in adj_dem.keys():
                demand_adjustment[mode].append(adj_dem[mode])

        # Multiply base demand by adjustments for rail and car and convert to dataframe
        adjusted_demand = dict()
        for mode, base_vals in base_demand.items():
            # Check the adjustments exist!
            if mode not in demand_adjustment.keys():
                raise ValueError(
                    "We haven't calculated any demand adjustments for the "
                    "base demand in mode '%s'!!" % mode
                )

            # Adjust our base demand
            full_adjustment = np.prod(demand_adjustment[mode], axis=0)
            adjusted_demand[mode] = base_vals * full_adjustment


        # Split rail demand back into CA/NCA
        for nm, df in rail_ca_split.items():
            adjusted_demand[nm] = adjusted_demand["rail"] * df
        total_rail = adjusted_demand["ca1"] + adjusted_demand["ca2"]
        if not np.array_equal(adjusted_demand["rail"], total_rail):
            # Need to get maximum twice to get a single float
            diff = np.max(
                np.ravel(np.abs(adjusted_demand["rail"] - total_rail))
            )
            name = du.get_dist_name(
                **demand_params, mode=str(ec.MODE_ID["rail"])
            )
            print(
                f"{name}: when splitting adjusted rail demand into "
                "CA and NCA, NCA + CA != Total Rail, there is a "
                f"maximum difference of {diff:.1E}"
            )
        adjusted_demand.pop("rail")

        print("HERE")
        print(demand_params)
        # Write demand output
        self._write_demand(adjusted_demand, demand_params, car_reverse)
        return adjusted_demand

    def _get_demand(self,
                    demand_params: Dict[str, str]
                    ) -> Tuple[Dict[str, pd.DataFrame],
                               Dict[str, np.array],
                               pd.DataFrame,
                               pd.DataFrame]:
        """Read the rail and car demand, aggregating CA and NCA for rail.

        Parameters
        ----------
        demand_params : Dict[str, str]
            Parameters to be passed to `get_dist_name` function
            for getting the demand file name.

        Returns
        -------
        demand : Dict[str, pd.DataFrame]
            The demand data for car and rail modes read from file and
            demand values of 1 for bus, active and no_travel.
        rail_split : Dict[str, np.array]]
            The ratio of CA and NCA to total rail demand, allowing
            the demand to be split back into CA and NCA once
            elasticities are applied.
        car_reverse : pd.DataFrame
            Splitting factors and look for converting the car demand
            back to old zone system after calculations.
        car_original : pd.DataFrame
            Demand matrix for car before the zone system conversion.

        Raises
        ------
        KeyError
            If the CA and NCA demand for rail doesn't have
            the same zone index and columns.
        """
        demand = {}
        # Get rail demand and add CA and NCA together
        m = "rail"
        tmp = {}
        for ca in ("1", "2"):
            path = self.demand_folders[m][0] / du.get_dist_name(
                **demand_params,
                mode=str(ec.MODE_ID[m]),
                car_availability=ca,
                csv=True,
            )
            tmp[f"ca{ca}"], _, _ = eu.read_demand_matrix(
                path, self.zone_translation_folder, ec.MODE_ZONE_SYSTEM[m]
            )
        if not (
            tmp["ca1"].index.equals(tmp["ca2"].index)
            and tmp["ca1"].columns.equals(tmp["ca2"].columns)
        ):
            raise KeyError(
                du.get_dist_name(**demand_params, mode=ec.MODE_ID[m])
                + " does not have the same index for CA and NCA"
            )
        # Get demand for CA + NCA and calculate split for converting back
        demand[m] = tmp["ca1"] + tmp["ca2"]
        rail_split = {}
        for k, val in tmp.items():
            rail_split[k] = np.divide(
                val.values,
                demand[m].values,
                out=np.zeros_like(val, dtype=float),
                where=demand[m] != 0,
            )
        del tmp

        # Get car demand
        m = "car"
        path = self.demand_folders[m][0] / du.get_dist_name(
            **demand_params, mode=str(ec.MODE_ID[m]), csv=True
        )
        demand[m], car_reverse, car_original = eu.read_demand_matrix(
            path, self.zone_translation_folder, ec.MODE_ZONE_SYSTEM[m]
        )

        demand.update(dict.fromkeys(ec.OTHER_MODES, 1.0))
        return demand, rail_split, car_reverse, car_original

    def _get_costs(
        self, purpose: int, demand: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """Read the cost files for each mode in `MODE_ZONE_SYSTEM`.

        Doesn't get the costs for Bus, Active or Non-travel modes as these
        are defined as cost change in the elasticity calculations.

        Parameters
        ----------
        purpose : int
            Purpose ID to get the costs for.
        demand : Dict[str, pd.DataFrame]
            Demand to as weights for zone translation.

        Returns
        -------
        Dict[str, pd.DataFrame]
            The costs for each mode which is present in `MODE_ID`.
        """
        costs = {}
        for m, zone in ec.MODE_ZONE_SYSTEM.items():
            path = self.demand_folders[m][1] / ec.COST_NAMES.format(
                mode=m, purpose=purpose
            )
            costs[m] = gc.get_costs(
                path, m, zone, self.zone_translation_folder, demand.get(m)
            )

        costs.update(dict.fromkeys(ec.OTHER_MODES, 1.0))
        return costs

    def _write_demand(self,
                      adjusted_demand: Dict[str, pd.DataFrame],
                      demand_params: Dict[str, str],
                      car_reverse: pd.DataFrame,
                      ) -> None:
        """Write the adjusted demand to CSV files.

        The outputs are written to mode sub-folders in `self.output_folder`,
        the bus, active and no_travel modes are written to a single file as
        these are scalar values.

        Parameters
        ----------
        adjusted_demand : Dict[str, pd.DataFrame]
            Dictionary containing the adjusted demand for each mode.
        demand_params : Dict[str, str]
            The demand parameters to be passed to `get_dist_name` for
            creating the output filename.
        car_reverse : pd.DataFrame
            The lookup and splitting factors for converting car demand
            back to the original zone system, with the following columns:
            [original zone - origin, original zone - destination,
            common zone system - origin, common zone system
            - destination, splitting factor].
        """
        if car_reverse is not None:
            # Convert car demand back to original zone system
            original_zs = zt.square_to_list(adjusted_demand["car"])
            car_reverse.columns = ["orig_o", "orig_d", "o", "d", "split"]
            original_zs = original_zs.merge(
                car_reverse,
                on=["o", "d"],
                how="left",
                validate="1:m",
            )
            original_zs["value"] = original_zs["value"] * original_zs["split"]
            original_zs = (
                original_zs[["orig_o", "orig_d", "value"]]
                .groupby(["orig_o", "orig_d"], as_index=False)
                .sum()
            )
            original_zs.rename(
                columns={f"orig_{i}": i for i in "od"},
                inplace=True,
            )
            original_zs = original_zs.pivot(
                index="o",
                columns="d",
                values="value",
            )
            original_zs.index.name = adjusted_demand["car"].index.name
            original_zs.columns.name = adjusted_demand["car"].columns.name
            # Check conversion hasn't changed total
            totals = list()
            for x in (original_zs, adjusted_demand["car"]):
                totals.append(np.sum(x.values))
            if abs(totals[0] - totals[1]) > ec.MATRIX_TOTAL_TOLERANCE:
                name = du.get_dist_name(
                    **demand_params, mode=str(ec.MODE_ID["car"])
                )
                print(
                    f"{name}: 'car' matrix totals differ by "
                    f"{abs(totals[0] - totals[1]):.1E} when converting "
                    "back to original zone system"
                )
            adjusted_demand["car"] = original_zs

        # Write the demand for car and rail for both
        # car availabilities (ca1 and ca2)
        for m in ("car", "ca1", "ca2"):
            ca = None
            mode = m
            if m != "car":
                ca = m[2]
                mode = "rail"
            folder = self.output_folder[mode]
            name = du.get_dist_name(
                **demand_params,
                mode=str(ec.MODE_ID[mode]),
                car_availability=ca,
                csv=True,
            )
            du.safe_dataframe_to_csv(adjusted_demand[m], folder / name)

        # Write other modes to a single file
        folder = self.output_folder["others"]
        name = du.get_dist_name(**demand_params, csv=True)
        df = pd.DataFrame(
            [
                (k, adjusted_demand[k].mean())
                for k in ("bus", "active", "no_travel")
            ],
            columns=["mode", "mean_demand_adjustment"],
        )
        du.safe_dataframe_to_csv(df, folder / name, index=False)


##### FUNCTIONS #####
def calculate_adjustment(
    base_demand: Dict[str, pd.DataFrame],
    base_costs: Dict[str, pd.DataFrame],
    base_gc: Dict[str, pd.DataFrame],
    elasticities: pd.DataFrame,
    elasticity_type: str,
    cost_constraint: np.array,
    cost_change: float,
    gc_params: Dict[str, Dict[str, float]],
) -> Dict[str, np.array]:
    """Calculate the demand adjustment for a single cost change.

    Parameters
    ----------
    base_demand : Dict[str, pd.DataFrame]
        Base demand for all modes, with key being the
        mode name.
    base_costs : Dict[str, pd.DataFrame]
        Base costs for all modes, with key being the
        mode name.
    base_gc : Dict[str, pd.DataFrame]
        Base generalised cost for all modes.
    elasticities : pd.DataFrame
        Elasticity values for all mode combinations.
    elasticity_type : str
        The name of the elasticity cost change being
        applied.
    cost_constraint : np.array
        An array to define where the cost is applied,
        should be the same shape as `base_demand`.
    cost_change : float
        The percentage cost change being applied.
    gc_params : Dict[str, Dict[str, float]]
        The parameters used in the generlised cost
        calculations, there should be one set of
        values per mode.

    Returns
    -------
    Dict[str, np.array]
        The `base_demand` matrices after the
        elasticity calculation has been applied.

    Raises
    ------
    KeyError
        If an `elasticity_type` not present in the
        `GC_ELASTICITY_TYPES` lookup is given.
    """
    if elasticity_type not in ec.GC_ELASTICITY_TYPES:
        raise KeyError(
            f"Unknown elasticity_type: '{elasticity_type}', "
            f"expected one of {list(ec.GC_ELASTICITY_TYPES.keys())}"
        )

    chg_mode, cost_type = ec.GC_ELASTICITY_TYPES[elasticity_type]
    # Filter only elasticities involving the mode that changes
    elasticities = elasticities.loc[
        elasticities["ModeCostChg"].str.lower() == chg_mode
    ]

    # The cost and cost_factors are dependant on the cost that changes
    cost, cost_factor = _elasticity_gc_factors(
        base_costs[chg_mode],
        gc_params.get(chg_mode, {}),
        elasticity_type,
    )
    if cost_type != "gc":
        # Adjust the costs of the change mode and calculate adjusted GC
        adj_cost, adj_gc_params = adjust_cost(
            base_costs[chg_mode],
            gc_params.get(chg_mode, {}),
            elasticity_type,
            cost_change,
            cost_constraint,
        )
        adj_gc = gc.gen_cost_mode(adj_cost, chg_mode, **adj_gc_params)
        # Set GC ratio to 1 (no demand adjustment) wherever
        # base GC <= 0, as cost shouldn't be 0 (or negative)
        gc_ratio = np.divide(
            adj_gc,
            base_gc[chg_mode],
            where=base_gc[chg_mode] > 0,
            out=np.full_like(adj_gc, 1.0),
        )
    else:
        gc_ratio = np.where(cost_constraint == 1, cost_change, 1)

    cols = ["AffectedMode", "OwnElast"]
    demand_adjustment = {
        m.lower(): np.full_like(base_demand[m.lower()], 1.0)
        for m in elasticities[cols[0]].unique()
    }
    for aff_mode, elast in elasticities[cols].itertuples(
        index=False, name=None
    ):
        aff_mode = aff_mode.lower()
        if cost_type != "gc":
            # Calculate the generalised cost of the current elasticity
            gc_elast = gc.gen_cost_elasticity_mins(
                elast,
                base_gc[chg_mode],
                cost,
                base_demand[chg_mode],
                cost_factor,
            )
        else:
            # If the generalised costs itself is being changed then elasticity given
            # is the GC elasticity and the GC ratio will be equal to the change,
            # using cost constraint to only change certain cells
            gc_elast = elast

        demand_adjustment[aff_mode] = demand_adjustment[aff_mode] * np.power(
            gc_ratio,
            gc_elast,
            out=np.zeros_like(gc_ratio),
            where=gc_ratio != 0,  # 0^(-x) is undefined and 0^(+x)=0 so leave 0
        )

    return demand_adjustment


def adjust_cost(
    base_costs: Union[pd.DataFrame, float],
    gc_params: Dict[str, float],
    elasticity_type: str,
    cost_change: float,
    constraint_matrix: np.array = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Adjust the cost matrices or parameters for given `elasticity_type`.

    `GC_ELASTICITY_TYPES` is used as the lookup for what type of cost
    changes are applied.

    Parameters
    ----------
    base_costs : Union[pd.DataFrame, float]
        Base costs to be adjusted.
    gc_params : Dict[str, float]
        Generalised cost parameters to be adjusted.
    elasticity_type : str
        Elasticity type used with `GC_ELASTICITY_TYPES` lookup to
        determine the type of cost changes being applied.
    cost_change : float
        The cost change value to apply to `base_costs` or
        `gc_params`.
    constraint_matrix : np.array, optional
        Constraint to be used when adjusting `base_costs`, by
        default None.

    Returns
    -------
    adj_costs : Union[pd.DataFrame, float]
        Adjusted costs, or `base_costs` if no adjustment required.
    adj_gc_params : Dict[str, float]
        Adjusted GC parameters, or `gc_params` if no adjustement
        required.

    Raises
    ------
    KeyError
        If the cost to be adjusted isn't present in the `base_costs`
        or `gc_params`.
    """
    mode, cost_type = ec.GC_ELASTICITY_TYPES[elasticity_type]
    # Other modes have scalar costs and no GC params so are just
    # multiplied by change
    if not isinstance(base_costs, pd.DataFrame):
        return base_costs * cost_change, gc_params

    # Make sure costs are sorted so that the constraint matrix lines up correctly
    adj_cost = base_costs.copy().sort_values(["origin", "destination"])
    adj_gc_params = gc_params.copy()
    if cost_type in base_costs.columns:
        if constraint_matrix is None:
            constraint_matrix = 1.0
        np.multiply(
            adj_cost[cost_type].values,
            cost_change,
            out=adj_cost[cost_type].values,
            where=constraint_matrix.flatten() == 1,
            casting="unsafe",
        )
    elif cost_type in gc_params:
        adj_gc_params[cost_type] = adj_gc_params[cost_type] * cost_change
    else:
        raise KeyError(
            f"Cost type to be changed ({cost_type}) isn't present "
            f"in the base_costs or gc_params for {mode}"
        )
    return adj_cost, adj_gc_params


def _elasticity_gc_factors(
    base_costs: pd.DataFrame,
    gc_params: Dict[str, float],
    elasticity_type: str,
) -> Tuple[np.array, float]:
    """Return cost and cost_fator values for use in `gen_cost_elasticity_mins`.

    Determines the required parameters for calculating the GC elasticity
    based on the `elasticity_type` given, using `GC_ELASTICITY_TYPES` lookup.

    Parameters
    ----------
    base_costs : pd.DataFrame
        Base costs data for single mode.
    gc_params : Dict[str, float]
        Generalised cost calculation parameters for single mode.
    elasticity_type : str
        The name of the elasticity type.

    Returns
    -------
    np.array
        cost to be used in `gen_cost_elasticity_mins`.
    float
        cost_factor to be used in `gen_cost_elasticity_mins`.

    Raises
    ------
    ValueError
        If the elasticitytype given leads to an unknown combination
        of cost_type and mode.
    """

    def square_matrix(values_col: str) -> pd.DataFrame:
        """Convert costs column into a square matrix."""
        return base_costs.pivot(
            index="origin",
            columns="destination",
            values=values_col,
        )

    mode, cost_type = ec.GC_ELASTICITY_TYPES[elasticity_type]
    cost, factor = None, None
    if cost_type == "gc":
        cost = 1.0
    elif mode == "car":
        if cost_type == "time":
            cost = square_matrix(cost_type)
            factor = 1 / 60
        elif cost_type == "vc":
            cost = square_matrix("dist")
            factor = (gc_params["vc"] / gc_params["vt"]) / 1000
    elif mode == "rail":
        if cost_type == "ride":
            cost = square_matrix(cost_type)
        elif cost_type == "fare":
            cost = square_matrix(cost_type)
            factor = 1 / gc_params["vt"]
    elif mode in ec.OTHER_MODES:
        cost = 1.0

    if cost is None:
        raise ValueError(
            f"Unknown cost_type/mode combination: {cost_type}, {mode} not "
            "sure what factors are required for GC elasticity calculation"
        )
    return cost, factor


def read_cost_changes(path: Path, years: List[str]) -> pd.DataFrame:
    """Read the elasticity cost changes file and check for missing data.

    Parameters
    ----------
    path : Path
        Path to the cost changes file.
    years : List[str]
        List of years to be checked there is data for.

    Returns
    -------
    pd.DataFrame
        DataFrame containing all the cost change data with
        the following columns: year, elasticity_type,
        constraint_matrix_name and percentage_change.

    Raises
    ------
    KeyError
        If an elasticity_type is given that is not present
        in `GC_ELASTICITY_TYPES` lookup.
    ValueError
        If no data is present for one, or more, `years`.
    """
    dtypes = {
        "year": str,
        "elasticity_type": str,
        "constraint_matrix_name": str,
        "percentage_change": float,
    }
    df = pd.read_csv(path, usecols=dtypes.keys(), dtype=dtypes)

    unknown_types = []
    for i in df["elasticity_type"].unique():
        if i not in ec.GC_ELASTICITY_TYPES.keys():
            unknown_types.append(i)
    if unknown_types:
        raise KeyError(
            f"Unknown elasticity_type: {unknown_types}, "
            f"available types are: {list(ec.GC_ELASTICITY_TYPES.keys())}"
        )
    missing_years = [i for i in years if i not in df["year"].unique().tolist()]
    if missing_years:
        raise ValueError(f"Cost change not present for years: {missing_years}")

    return df
