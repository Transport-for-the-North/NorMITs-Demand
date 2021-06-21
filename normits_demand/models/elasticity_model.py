# -*- coding: utf-8 -*-
"""
Created on: Tue May 11 14:22:12 2021
Updated on:

Original author: Ben Taylor
Last update made by:
Other updates made by:

File purpose:
Module containing the functions for applying elasticities to demand matrices.
"""


# Built-in imports
import functools
import operator

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
import normits_demand as nd
from normits_demand import constants as consts
from normits_demand.utils import general as du
from normits_demand.utils import file_ops
from normits_demand.models import efs_zone_translator as zt
from normits_demand.concurrency import multiprocessing
from normits_demand.elasticity import utils as eu
from normits_demand.elasticity import generalised_costs as gc
from normits_demand.elasticity import constants as ec


class ElasticityModel:
    """Class for applying elasticity calculations to demand matrices."""

    _default_common_zone_system = ec.COMMON_ZONE_SYSTEM

    _valid_ca = consts.VALID_CA
    _valid_mode_names = list(ec.MODE_ID.keys())
    _valid_mode_nums = list(ec.MODE_ID.values())

    def __init__(self,
                 input_folders: Dict[str, Path],
                 input_files: Dict[str, Path],
                 output_folders: Dict[str, Path],
                 output_years: List[int],
                 base_year: int,
                 common_zone_system: str = None,
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

        base_year:
            The base year of the produced matrices. This is used to
            use the correct base year GC cost parameters when calculating
            adjustments into the future year.

        common_zone_system:
            The name of the common zone system to calculate the adjustments in.
            Demand will be translated into and out of this zone system if
            it is not already in this format.
        """
        # BACKLOG: Change ElasticityModel constructor. Make arguments explicit
        #  labels: elasticity
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

        self.import_home = input_folders["elasticity"]
        self.zone_translation_folder = input_folders["translation"]

        # Assign demand and cost dirs
        modes = ['car', 'rail']
        self.demand_dirs = {m: input_folders[f"{m}_demand"] for m in modes}
        self.cost_dirs = {m: input_folders[f"{m}_costs"] for m in modes}

        self._check_paths(
            output_folders,
            ("car", "rail", "others"),
            create_folders=True,
        )
        self.output_folder = output_folders
        self.years = output_years
        self.base_year = base_year

        if common_zone_system is None:
            self.common_zone_system = self._default_common_zone_system
        else:
            self.common_zone_system = common_zone_system

        # Set up the cost builder
        self.cost_builder = gc.CostBuilder(
            base_year=self.base_year,
            years=self.years,
            modes=list(ec.MODE_ID.keys()),
            purposes=ec.PURPOSES,
            vot_voc_path=self.input_files["gc_parameters"],
            cost_adj_path=self.input_files["cost_changes"],
            elasticity_types_path=self.import_home / ec.ETYPES_FNAME,
        )

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

    def apply_all_MP(self, process_count: int = consts.PROCESS_COUNT):
        """Performs elasticity calculations for all segments provided.

        Segment information is read from `SEGMENTS_FILE` which is
        expected to be found in elasticity folder given.
        """
        # Read in the cost changes
        print("Reading in the cost changes...")
        scalar_costs = self.cost_builder.get_vot_voc()
        cost_changes = self.cost_builder.get_cost_changes()

        # Read in the segments to loop around
        segments = eu.read_segments_file(self.import_home / ec.SEGMENTS_FILE)

        # Set up the arguments for each iteration
        kwarg_list = list()
        print("Setting up arguments...")
        for _, row in segments.iterrows():
            for yr in self.years:
                # Grab the elasticity params from the file
                elasticity_params = {
                    "purpose": str(row["elast_p"]),
                    "market_share": row["elast_market_share"],
                }

                # Grab the segment params from the file
                demand_seg_params = {
                    "trip_origin": row["trip_origin"],
                    "matrix_format": "pa",
                    "year": yr,
                    "purpose": str(row["p"]),
                }
                if row["p"] in consts.SOC_P:
                    demand_seg_params["segment"] = str(int(row["soc"]))
                elif row["p"] in consts.NS_P:
                    demand_seg_params["segment"] = str(int(row["ns"]))
                else:
                    raise nd.NormitsDemandError(
                        "purpose '%s' does not seem to be a soc or an "
                        "ns purpose!" % demand_seg_params['purpose']
                    )

                # Figure out which vot and voc costs to use
                uc = du.purpose_to_user_class(row['p'])

                kwarg_list.append({
                    'demand_params': demand_seg_params,
                    'elasticity_params': elasticity_params,
                    'base_year_gc_params': scalar_costs[str(self.base_year)][uc],
                    'future_gc_params': scalar_costs[yr][uc],
                    'cost_changes': cost_changes.loc[cost_changes["yr"] == yr],
                    'future_year': int(yr),
                    'fname_suffix': '_int',
                })

        # Set up progress bar kwargs to track
        pbar_kwargs = {
            'total': len(segments) * len(self.years),
            'desc': "Applying elasticities to segments",
            'unit': "segment",
            'colour': 'cyan',
        }

        # Call the functions
        print("Running!")
        multiprocessing.multiprocess(
            fn=self.apply_elasticities,
            kwargs=kwarg_list,
            pbar_kwargs=pbar_kwargs,
            process_count=process_count,
        )

    def apply_all(self):
        """Performs elasticity calculations for all segments provided.

        Segment information is read from `SEGMENTS_FILE` which is
        expected to be found in elasticity folder given.
        """

        # Read in the cost changes
        scalar_costs = self.cost_builder.get_vot_voc()
        cost_changes = self.cost_builder.get_cost_changes()

        # Redirects the outputs around pbar
        with du.std_out_err_redirect_tqdm() as orig_stdout:

            # Read in the segments to loop around
            segments = eu.read_segments_file(self.import_home / ec.SEGMENTS_FILE)

            # Set up pbar
            pbar = tqdm.tqdm(
                total=len(segments) * len(self.years),
                desc="Applying elasticities to segments",
                file=orig_stdout,
                dynamic_ncols=True,
                unit="segment",
            )

            # Loop through all of the defined segments
            for _, row in segments.iterrows():
                for yr in self.years:
                    # Grab the elasticity params from the file
                    elasticity_params = {
                        "purpose": str(row["elast_p"]),
                        "market_share": row["elast_market_share"],
                    }

                    # Grab the segment params from the file
                    demand_seg_params = {
                        "trip_origin": row["trip_origin"],
                        "matrix_format": "pa",
                        "year": yr,
                        "purpose": str(row["p"]),
                    }
                    if row["p"] in consts.SOC_P:
                        demand_seg_params["segment"] = str(int(row["soc"]))
                    elif row["p"] in consts.NS_P:
                        demand_seg_params["segment"] = str(int(row["ns"]))
                    else:
                        raise nd.NormitsDemandError(
                            "purpose '%s' does not seem to be a soc or an "
                            "ns purpose!" % demand_seg_params['purpose']
                        )

                    # Figure out which vot and voc costs to use
                    uc = du.purpose_to_user_class(row['p'])

                    # TODO(BT): REMOVE THIS!
                    self.apply_elasticities(
                        demand_params=demand_seg_params,
                        elasticity_params=elasticity_params,
                        base_year_gc_params=scalar_costs[str(self.base_year)][uc],
                        future_gc_params=scalar_costs[yr][uc],
                        cost_changes=cost_changes.loc[cost_changes["yr"] == yr],
                        fname_suffix='_int',
                    )

                    # Try to apply the elasticity
                    # try:
                    #     self.apply_elasticities(
                    #         demand_params=demand_seg_params,
                    #         elasticity_params=elasticity_params,
                    #         base_year_gc_params=scalar_costs[str(self.base_year)][uc],
                    #         future_gc_params=scalar_costs[yr][uc],
                    #         cost_changes=cost_changes.loc[cost_changes["yr"] == yr],
                    #     )
                    # except Exception as e:  # pylint: disable=broad-except
                    #     # Catching and printing all errors so program can
                    #     # continue with other segments
                    #     name = du.get_dist_name(**demand_seg_params)
                    #     print(f"{name} - {e.__class__.__name__}: {e}")
                    pbar.update(1)

            pbar.close()

    def apply_elasticities(self,
                           demand_params: Dict[str, str],
                           elasticity_params: Dict[str, str],
                           base_year_gc_params: Dict[str, Dict[str, float]],
                           future_gc_params: Dict[str, Dict[str, float]],
                           cost_changes: pd.DataFrame,
                           future_year: int,
                           fname_suffix: str = None,
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

        base_year_gc_params : Dict[str, Dict[str, float]]
            Parameters used in the generalised cost calculations,
            expected format:
            {
                "car": {"vt": 16.2, "vc": 9.45},
                "rail": {"vt": 16.4},
            }

        future_gc_params : Dict[str, Dict[str, float]]
            Parameters used in the generalised cost calculations,
            expected format:
            {
                "car": {"vt": 16.2, "vc": 9.45},
                "rail": {"vt": 16.4},
            }

        cost_changes : pd.DataFrame
            The cost changes to be applied to be applied,
            expects the following columns: e_type,
            adj_type and change.

        base_year:
            The base year that the base_year_gc_params were created for.

        future_year:
            The future year that the future_gc_params are related to. This is
            used, along with base_year to add inflation and congestion
            to car costs.

        fname_suffix:
            An optional suffix to add onto the filename when searching for
            the demand to read in.

        Returns
        -------
        Dict[str, pd.DataFrame]
            The adjusted demand for all modes.
        """
        # Init
        elasticities = eu.read_elasticity_file(
            self.import_home / ec.ELASTICITIES_FILE,
            **elasticity_params,
        )

        # ## CHECK THE ELASTICITIES WE WANT EXIST ## #
        to_use = cost_changes["e_type"].unique()
        available = elasticities["type"].unique()
        missing = [x for x in to_use if x not in available]

        if missing != list():
            raise ValueError(
                f"Elasticity values in {ec.ELASTICITIES_FILE} "
                f"missing for the following types: {missing}"
            )

        # ## LOAD IN THE CONSTRAINT MATRICES ## #
        path = self.import_home / ec.CONSTRAINTS_FOLDER
        needed_mats = cost_changes["adj_type"].unique().tolist()
        constraint_mats = eu.get_constraint_mats(path, needed_mats)

        # ## LOAD IN DEMAND FOR THIS SEGMENT ## #
        # common format and retain the translations to get back to original formats
        ret_vals = self._get_demand(demand_params, fname_suffix=fname_suffix)
        base_demand = ret_vals[0]
        rail_ca_split_factors = ret_vals[1]
        car_reverse_translation = ret_vals[2]
        car_original_mat = ret_vals[3]

        # ## GET COSTS FOR THIS SEGMENT ## #
        # Load in a common format, weight by demand where translations are needed
        translation_weights = {
            "car": car_original_mat,
            "rail": base_demand['rail'],
        }
        base_costs = self._get_costs(
            purpose=demand_params["purpose"],
            translation_weights=translation_weights
        )
        del car_original_mat

        # GC for each mode in the base year - all adjustments based from this
        base_year_gc = gc.calculate_gen_costs(base_costs, base_year_gc_params)

        # Loop setup
        cols = [
            "e_type",
            "adj_type",
            "change",
        ]
        iterator = cost_changes[cols].itertuples(index=False, name=None)
        demand_adjustment = defaultdict(list)

        # Loop through cost changes file and calculate demand adjustment
        for elast_type, cstr_name, change in iterator:
            adj_dem = calculate_own_cost_adjustment(
                demand=base_demand,
                base_costs=base_costs,
                base_year_gc=base_year_gc,
                elasticities=elasticities.loc[elasticities["type"] == elast_type],
                elasticity_type=elast_type,
                cost_constraint=constraint_mats[cstr_name],
                cost_change=change,
                future_scalar_costs=future_gc_params,
                base_year=self.base_year,
                future_year=future_year,
            )

            # Store all our demand by affected mode
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
        for nm, df in rail_ca_split_factors.items():
            adjusted_demand[nm] = adjusted_demand["rail"] * df

        # Check that the split demand is equal to the pre-split demand
        ca_names = list(rail_ca_split_factors.keys())
        ca_demand = [adjusted_demand[x] for x in ca_names]
        total_rail = functools.reduce(operator.add, ca_demand)

        if not np.array_equal(adjusted_demand["rail"], total_rail):
            # Need to get maximum twice to get a single float
            diff = np.max(
                np.ravel(np.abs(adjusted_demand["rail"] - total_rail))
            )
            name = du.get_dist_name(
                **demand_params, mode=str(ec.MODE_ID["rail"])
            )
            verbose = diff > 1e-5
            du.print_w_toggle(
                "%s: when splitting adjusted rail demand into CA and NCA, "
                "NCA + CA != Total Rail, there is a maximum difference "
                "of %.1e"
                % (name, diff),
                verbose=verbose
            )
        adjusted_demand.pop("rail")

        # Write demand output
        self._write_demand(
            adjusted_demand=adjusted_demand,
            demand_params=demand_params,
            car_keys=['car'],
            rail_keys=ca_names,
            car_reverse=car_reverse_translation,
            fname_suffix=fname_suffix,
        )
        return adjusted_demand

    def _read_demand_matrix(self,
                            path: Path,
                            zone_translation_folder: Path,
                            from_zone: str
                            ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Reads demand matrix and converts it to self.common_zone_system.

        Parameters
        ----------
        path : Path
            Path to the demand matrix.

        zone_translation_folder : Path
            Path to the folder contain zone lookups.

        from_zone : str
            The current zone system of the matrix, if this
            isn't self.common_zone_system then the matrix will
            be converted.

        Returns
        -------
        translated_demand:
            The demand matrix in the self.common_zone_system.

        translation_factors:
            Splitting factors for converting back to the from_zone
            zone system.

        original_demand:
            The demand matrix in the from_zone zone system.
        """
        # BACKLOG: Update elasticity to use new zone translations
        #  labels: elasticity, zone translation
        # Init
        demand = file_ops.read_df(path, index_col=0, find_similar=True)
        translation_factors = np.ones(demand.shape)
        orig_mat = demand.copy()

        # Convert column and index names to int
        demand.columns = pd.to_numeric(demand.columns, downcast="integer")
        demand.index = pd.to_numeric(demand.index, downcast="integer")

        # Translate if needed
        if from_zone != self.common_zone_system:
            # Set read in dtypes
            dtypes = {
                f"{from_zone}_zone_id": int,
                f"{self.common_zone_system}_zone_id": int,
                "split": float,
            }

            # Build translation path
            fname = ec.ZONE_LOOKUP_NAME.format(
                from_zone=from_zone,
                to_zone=self.common_zone_system
            )
            path = zone_translation_folder / fname

            # Load in the translation file
            translation = pd.read_csv(path, usecols=dtypes.keys(), dtype=dtypes)
            cols = [f"{from_zone}_zone_id", f"{self.common_zone_system}_zone_id"]

            # Try to translate!
            try:
                demand, translation_factors = zt.translate_matrix(
                    matrix=demand,
                    lookup=translation,
                    lookup_cols=cols,
                    split_column="split",
                )
            except zt.MatrixTotalError as e:
                # Print the error but continue with the translation to still
                # process current segment
                print(f"{path.stem} - {e.__class__.__name__}: {e}")
                demand, translation_factors = zt.translate_matrix(
                    matrix=demand,
                    lookup=translation,
                    lookup_cols=cols,
                    split_column="split",
                    check_total=False,
                )

        # Put the rows/columns back into order
        demand = demand.sort_index().sort_index(axis=1)
        return demand, translation_factors, orig_mat

    def _get_demand(self,
                    demand_params: Dict[str, str],
                    fname_suffix: str = None,
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

        fname_suffix:
            An optional suffix to add onto the filename when searching for
            the demand to read in

        Returns
        -------
        demand : Dict[str, pd.DataFrame]
            The demand data for car and rail modes read from file and
            demand values of 1 for bus, active and no_travel.

        rail_ca_split_factors : Dict[str, np.array]]
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
        # Init
        demand = dict()

        # ## LOAD THE RAIL DEMAND ## #
        m = "rail"

        # Load in all the ca mats
        tmp = dict()
        for ca in self._valid_ca:
            fname = du.get_dist_name(
                **demand_params,
                mode=str(ec.MODE_ID[m]),
                car_availability=str(ca),
                suffix=fname_suffix,
                csv=True,
            )
            path = self.demand_dirs[m] / fname
            tmp[ca], *_ = self._read_demand_matrix(
                path,
                self.zone_translation_folder,
                ec.MODE_ZONE_SYSTEM[m]
            )

        # Make sure all mats are the same shape
        ref_idx = tmp[self._valid_ca[0]].index
        ref_cols = tmp[self._valid_ca[0]].columns
        for k, v in tmp.items():
            match_idx = v.index.equals(ref_idx)
            match_cols = v.index.equals(ref_cols)
            if not (match_idx and match_cols):
                raise KeyError(
                    du.get_dist_name(**demand_params, mode=ec.MODE_ID[m])
                    + " does not have the same index for CA and NCA"
                )

        # Get demand for CA + NCA and calculate split for converting back
        demand[m] = functools.reduce(operator.add, tmp.values())
        rail_ca_split_factors = dict()
        for k, val in tmp.items():
            rail_ca_split_factors[k] = np.divide(
                val.values,
                demand[m].values,
                out=np.zeros_like(val, dtype=float),
                where=demand[m] != 0,
            )
        del tmp

        # ## LOAD THE CAR DEMAND ## #
        m = "car"

        # Build the path
        fname = du.get_dist_name(
            **demand_params,
            mode=str(ec.MODE_ID[m]),
            suffix=fname_suffix,
            csv=True
        )
        path = self.demand_dirs[m] / fname

        # Load in the matrix and translate
        demand[m], car_reverse, car_original = self._read_demand_matrix(
            path,
            self.zone_translation_folder,
            ec.MODE_ZONE_SYSTEM[m]
        )

        demand.update(dict.fromkeys(ec.OTHER_MODES, 1.0))
        return demand, rail_ca_split_factors, car_reverse, car_original

    def _get_costs(self,
                   purpose: int,
                   translation_weights: Dict[str, pd.DataFrame],
                   ) -> Dict[str, pd.DataFrame]:
        """Read the cost files for each mode in `ec.MODE_ZONE_SYSTEM`.

        Doesn't get the costs for Bus, Active or Non-travel modes as these
        are defined as cost change in the elasticity calculations.

        Parameters
        ----------
        purpose : int
            Purpose ID to get the costs for.

        translation_weights : Dict[str, pd.DataFrame]
            Weights for zone translation.

        Returns
        -------
        Dict[str, pd.DataFrame]
            The costs for each mode which is present in `MODE_ID`.
        """
        costs = dict()
        for m, zone in ec.MODE_ZONE_SYSTEM.items():
            # Get the path for this mode and purpose
            fname = ec.COST_NAMES.format(mode=m, purpose=purpose)
            path = self.cost_dirs[m] / fname

            # Load in the costs, translate if needed
            costs[m] = gc.get_costs(
                path,
                m,
                zone,
                self.zone_translation_folder,
                translation_weights.get(m)
            )

        # Add in Bus, Active or Non-travel as 1.0 as default
        costs.update(dict.fromkeys(ec.OTHER_MODES, 1.0))

        return costs

    def _write_demand(self,
                      adjusted_demand: Dict[str, pd.DataFrame],
                      demand_params: Dict[str, str],
                      car_keys: List[str],
                      rail_keys: List[str],
                      car_reverse: pd.DataFrame,
                      fname_suffix: str = None,
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

        fname_suffix:
            An optional suffix to add onto the filename when searching for
            the demand to read in

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
        for k in car_keys:
            mode = 'car'
            folder = self.output_folder[mode]
            name = du.get_dist_name(
                **demand_params,
                mode=str(ec.MODE_ID[mode]),
                suffix=fname_suffix,
                csv=True,
            )
            du.safe_dataframe_to_csv(adjusted_demand[k], folder / name)

        for k in rail_keys:
            mode = 'rail'
            ca = k
            folder = self.output_folder[mode]
            name = du.get_dist_name(
                **demand_params,
                mode=str(ec.MODE_ID[mode]),
                car_availability=str(ca),
                suffix=fname_suffix,
                csv=True,
            )
            du.safe_dataframe_to_csv(adjusted_demand[k], folder / name)

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
def calculate_own_cost_adjustment(demand: Dict[str, pd.DataFrame],
                                  base_costs: Dict[str, pd.DataFrame],
                                  base_year_gc: Dict[str, pd.DataFrame],
                                  elasticities: pd.DataFrame,
                                  elasticity_type: str,
                                  cost_constraint: np.array,
                                  cost_change: float,
                                  future_scalar_costs: Dict[str, Dict[str, float]],
                                  base_year: int,
                                  future_year: int,
                                  ) -> Dict[str, np.array]:
    """Calculate the demand adjustment for a single cost change.

    OWN COST

    Parameters
    ----------
    demand : Dict[str, pd.DataFrame]
        The demand to adjust based on cost changes.
        Starting demand for all modes, with key being the
        mode name.

    base_costs : Dict[str, pd.DataFrame]
        Base year costs for all modes, with key being the
        mode name.

    base_year_gc : Dict[str, pd.DataFrame]
        Base year generalised cost for all modes.

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

    future_scalar_costs : Dict[str, Dict[str, float]]
        The parameters used in the generalised cost
        calculations, there should be one set of
        values per mode.

    base_year:
        The base year that the base_costs were created for.

    future_year:
        The future year that the cost_change and future_scalar_costs
        is related to. This is used, along with base_year to add inflation
        and congestion to car costs.

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
        elasticities["changing_mode"].str.lower() == chg_mode
    ]

    # Figure out the GC ratio of change
    if cost_type != "gc":
        # Adjust the costs of the changing mode
        adj_cost, _ = adjust_cost(
            base_costs=base_costs[chg_mode],
            gc_params=future_scalar_costs.get(chg_mode, {}),
            elasticity_type=elasticity_type,
            cost_change=cost_change,
            constraint_matrix=cost_constraint,
            base_year=base_year,
            future_year=future_year,
        )

        # calculate the new adjusted GC
        adj_gc = gc.gen_cost_mode(
            adj_cost,
            chg_mode,
            **future_scalar_costs.get(chg_mode, {})
        )

        # Calculate the ratio of chance in GC
        gc_ratio = np.divide(
            adj_gc,
            base_year_gc[chg_mode],
            where=base_year_gc[chg_mode] > 0,
            out=np.full_like(adj_gc, 1.0),
        )

        # Set GC ratio to 1 (no demand adjustment) wherever
        # base GC <= 0, as cost shouldn't be 0 (or negative)
        gc_ratio = np.where(gc_ratio <= 0, 1, gc_ratio)

    else:
        adj_cost = base_costs[chg_mode].copy()
        adj_gc = base_year_gc[chg_mode].copy()
        gc_ratio = np.where(cost_constraint == 1, cost_change, 1)

    # Grab the cost that we are changing as a square matrix
    cost, cost_factor = _elasticity_gc_factors(
        adj_cost,
        future_scalar_costs.get(chg_mode, {}),
        elasticity_type,
    )

    # Initialise the adjustment matrices for each affected mode
    cols = ["affected_mode", "elast_value"]
    demand_adjustment = {
        m.lower(): np.full_like(demand[m.lower()], 1.0)
        for m in elasticities[cols[0]].unique()
    }

    # Calculate the own cost elasticity adjustments
    for aff_mode, elast in elasticities[cols].itertuples(index=False, name=None):
        aff_mode = aff_mode.lower()
        if cost_type != "gc":
            # Calculate the demand weighted generalised cost of the elasticity
            gc_elast = gc.gen_cost_elasticity_mins(
                elasticity=elast,
                gen_cost=adj_gc,
                cost=cost,
                demand=demand[chg_mode],
                cost_factor=cost_factor,
            )
        else:
            # If the generalised costs itself is being changed then elasticity given
            # is the GC elasticity and the GC ratio will be equal to the change,
            # using cost constraint to only change certain cells
            gc_elast = elast

        # Cannot do *= here as numpy won't broadcast types if we do
        # Sounds like bad practice. Need to figure out how the return is used!
        # Problem: when aff mode is scalar and chg mode is array gc_ratio
        # taking scalar * by array = array. Cannot assign array to scalar space
        demand_adjustment[aff_mode] = demand_adjustment[aff_mode] * np.power(
            gc_ratio,
            gc_elast,
            out=np.ones_like(gc_ratio),
            where=gc_ratio != 0,  # 0^(-x) is undefined and 0^(+x)=0 so leave 0
        )

    return demand_adjustment


def adjust_cost(base_costs: Union[pd.DataFrame, float],
                gc_params: Dict[str, float],
                elasticity_type: str,
                cost_change: float,
                base_year: int,
                future_year: int,
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

    base_year:
        The base year that the base_costs were created for.

    future_year:
        The future year that the cost_change is related to. This is used,
        along with base_year to add inflation and congestion to
        car costs.

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
    # Init
    mode, cost_type = ec.GC_ELASTICITY_TYPES[elasticity_type]

    # Other modes have scalar costs and no GC params so are just
    # multiplied by change
    if not isinstance(base_costs, pd.DataFrame):
        return base_costs * cost_change * constraint_matrix, gc_params

    # Make sure costs are sorted so that the constraint matrix lines up correctly
    adj_cost = base_costs.copy().sort_values(["origin", "destination"])

    # # TODO(BT): THIS IS A TEST!!
    # # Adjust time to reflect congestion relative to base
    # # Adjust toll in line with inflation
    # if mode == 'car':
    #     adj_cost['time'] *= 1.01 ** (future_year - base_year)
    #     adj_cost['toll'] *= 1.02 ** (future_year - base_year)

    adj_gc_params = gc_params.copy()
    # If cost component
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

    # if vot/voc
    # TODO(BT): Remove this later when we know it's never needed!
    elif cost_type in ['vot', 'voc']:
        # WE SHOULD NEVER ADJUST THESE OTHERWISE DOUBLE COUNTING LATER ON!
        raise nd.NormitsDemandError(
            "Scalar generalised cost parameters (vot/voc) should not be "
            "adjusted in the elasticity! DOUBLE COUNTING!!!"
        )
        # adj_gc_params[cost_type] = adj_gc_params[cost_type] * cost_change

    else:
        raise KeyError(
            f"Cost type to be changed ({cost_type}) isn't present "
            f"in the base_costs or gc_params for {mode}"
        )

    return adj_cost, adj_gc_params


def _elasticity_gc_factors(base_costs: pd.DataFrame,
                           gc_params: Dict[str, float],
                           elasticity_type: str,
                           ) -> Tuple[np.array, float]:
    """Return cost and cost_factor values for use in `gen_cost_elasticity_mins`.

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
        Cost in either minutes or monetary units.

    float
        cost_factor to be used in `gen_cost_elasticity_mins`.
        Factor to convert cost into minutes, if None (default)
        uses a factor of 1.0.

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
            factor = 1
        elif cost_type == "voc":
            cost = square_matrix("dist")
            factor = (gc_params["voc"] / gc_params["vot"])

    elif mode == "rail":
        if cost_type == "ride":
            cost = square_matrix(cost_type)
        elif cost_type == "fare":
            cost = square_matrix(cost_type)
            factor = 1 / gc_params["vot"]

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
