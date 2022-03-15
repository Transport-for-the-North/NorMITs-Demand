# -*- coding: utf-8 -*-
"""
    Script to convert the CSV distance skims from the rail (CUBE)
    model to the distribution model HB/NHB format.
"""

##### IMPORTS #####
from __future__ import annotations

# Standard imports
import argparse
import configparser
import dataclasses
import re
import sys
from pathlib import Path
from typing import NamedTuple

# Third party imports
import numpy as np
import pandas as pd
from tqdm import tqdm

# Local imports
sys.path.append("..")
# pylint: disable=import-error,wrong-import-position
import normits_demand as nd
from normits_demand.utils import file_ops
from normits_demand import logging as nd_log
from normits_demand import constants as nd_consts

# pylint: enable=import-error,wrong-import-position


##### CONSTANTS #####
LOG = nd_log.get_logger(
    nd_log.get_package_logger_name() + ".run_helper_scripts.extract_rail_costs"
)
LOG_FILE = "Rail_Costs.log"
COST_TYPES = {"dist": "Distance"}
TIME_PERIODS = {1: "AM", 2: "IP", 3: "PM", 4: "OP"}
MODEL_USER_CLASSES = {"commute": 1, "business": 2, "other": 3}

##### CLASSES #####
class CostFileNameError(nd.NormitsDemandError):
    """Error raised when parsing an invalid cost file name."""

    def __init__(self, path: Path, message: str, *args: object) -> None:
        self.path = path
        self.message = f"'{path.name}' {message}"
        super().__init__(self.message, *args)


class CostDetails(NamedTuple):
    """Time period and cost type details for a cost file."""

    time_slice: int
    cost_type: str


@dataclasses.dataclass
class ExtractRailCostsInputs:
    """Class to store and manage input parameters for `extract_rail_costs`.

    Parameters
    ----------
    model_name : str
        Name of the model the costs are extracted from.
    output_folder: Path
        Path to folder to save outputs.
    cost_folder: Path
        Path to folder containing cost files.
    """

    _CONFIG_SECTION = "EXTRACT RAIL COSTS PARAMETERS"

    model_name: str
    output_folder: Path
    cost_folder: Path

    def __post_init__(self) -> None:
        for f in ("output_folder", "cost_folder"):
            path = Path(getattr(self, f))
            if not path.is_dir():
                raise NotADirectoryError(f"{f} doesn't exist: {path}")
            setattr(self, f, path)

    def save(self, path: Path) -> None:
        """Save parameters to config file.

        Parameters
        ----------
        path : Path
            Path to file to save.
        """
        config = configparser.ConfigParser()
        config[self._CONFIG_SECTION] = {
            k: "" if v is None else str(v) for k, v in dataclasses.asdict(self).items()
        }

        with open(path, "wt") as file:
            config.write(file)
        LOG.info("Extract rail cost parameters saved to: %s", path)

    @classmethod
    def load(cls, path: Path) -> ExtractRailCostsInputs:
        """Load parameters from config file.

        Parameters
        ----------
        path : Path
            Path to config file.

        Returns
        -------
        ExtractRailCostsInputs
            Parameters loaded from `path`.

        Raises
        ------
        configparser.NoSectionError
            If the config file is missing a required section.
        """
        config = configparser.ConfigParser(converters={"path": Path})
        config.read(path)

        if not config.has_section(cls._CONFIG_SECTION):
            raise configparser.NoSectionError(cls._CONFIG_SECTION)
        section = config[cls._CONFIG_SECTION]

        params = {}
        params["model_name"] = section.get("model_name")
        params["output_folder"] = section.getpath("output_folder")
        params["cost_folder"] = section.getpath("cost_folder")
        return ExtractRailCostsInputs(**params)


##### FUNCTIONS #####
def get_arguments() -> argparse.Namespace:
    """Parse the commandline arguments.

    Positional arguments:
    - config: path to the config file

    Returns
    -------
    argparse.Namespace
        Argument values parsed from commandline.

    Raises
    ------
    FileNotFoundError
        If the config file doesn't exist.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Path to config file")
    args = parser.parse_args()

    if not args.config.is_file():
        raise FileNotFoundError(f"cannot find config file: {args.config}")
    return args


def parse_filename(path: Path) -> CostDetails:
    """Extracts information from the cost filename.

    Parameters
    ----------
    path : Path
        Path to the cost file.

    Returns
    -------
    CostDetails
        Information about the cost file given.

    Raises
    ------
    CostFileNameError
        If any of the information can't be extracted from the `path` given.
    """
    non_letter_number = r"(?:\W|_|\b)"
    cost_types = "|".join(COST_TYPES.keys())
    patterns = {
        "time_period": (
            non_letter_number + r"(?:((?:TS\d)|(?:AM|IP|PM|OP)))" + non_letter_number
        ),
        "cost_type": (non_letter_number + f"({cost_types})$"),
    }

    name_params = {}
    for nm, pat in patterns.items():
        match = re.search(pat, path.stem, re.I)
        if match:
            name_params[nm] = match.group(1)
        else:
            raise CostFileNameError(path, f"can't find {nm}")

    if name_params["time_period"].upper().startswith("TS"):
        try:
            time_slice = int(name_params["time_period"][2:])
        except ValueError as err:
            raise CostFileNameError(
                path, f"time slice ({name_params['time_period']}) isn't an integer"
            ) from err
    else:
        tp_to_ts = {v: k for k, v in TIME_PERIODS.items()}
        try:
            time_slice = tp_to_ts[name_params["time_period"]]
        except KeyError as err:
            raise CostFileNameError(
                path, f"time period ({name_params['time_period']}) isn't valid"
            ) from err

    return CostDetails(time_slice, name_params["cost_type"])


def find_costs(cost_folder: Path, cost_type: str = None) -> dict[CostDetails, Path]:
    """Find CSVs in `cost_folder` which match the expected cost filename format.

    Parameters
    ----------
    cost_folder : Path
        Folder containing cost CSVs.
    cost_type : str, optional
        Type of cost to return, if not given returns all costs found.

    Returns
    -------
    dict[CostDetails, Path]
        Paths to cost files with their details.

    See Also
    --------
    `COST_TYPES`: these keys show the options which can be
        provided to `cost_type`.
    """
    costs = {}
    for path in cost_folder.glob("*.csv"):
        try:
            details = parse_filename(path)
        except CostFileNameError as err:
            LOG.warning("%s: %s", err.__class__.__name__, err)
            continue
        if cost_type and details.cost_type != cost_type:
            continue
        costs[details] = path

    LOG.info("Found %s cost CSVs", len(costs))
    return costs


def _read_dist(path: Path, zoning: nd.ZoningSystem) -> dict[str, pd.DataFrame]:
    """Read distance cost file and split into square matrices by user class.

    Distances are expected to be a CSV with 5 columns:
    - origin
    - destination
    - business
    - commute
    - other

    Parameters
    ----------
    path : Path
        Path to the cost CSV to read.
    zoning : nd.ZoningSystem
        Zoning system the cost file is in.

    Returns
    -------
    dict[str, pd.DataFrame]
        Cost matrices for each user class with keys: 'business',
        'commute' and 'other'.
    """
    index_cols = ["origin", "destination"]
    data_cols = ["business", "commute", "other"]
    dtype = {
        **dict.fromkeys(index_cols, int),
        **dict.fromkeys(data_cols, float),
    }
    df = file_ops.read_df(
        path, usecols=range(5), dtype=dtype, names=dtype.keys(), header=None
    )
    df.set_index(index_cols, inplace=True)

    zones = zoning.unique_zones
    df = df.reindex(index=pd.MultiIndex.from_product((zones, zones)))
    nan = np.sum(df.isna().values)
    if nan > 0:
        LOG.error(
            "%s (%.1f%%) cells have missing costs in %s",
            nan,
            (nan / df.size) * 100,
            path.stem,
        )

    matrices = {}
    for c in data_cols:
        matrices[c] = df.loc[:, c].unstack(level=1)
    return matrices


def _userclass_purpose_lookup() -> dict[int, str]:
    """Lookup between purpose number and user class name."""
    lookup = {}
    for nm, purposes in nd_consts.USER_CLASS_PURPOSES.items():
        lookup.update(dict.fromkeys(purposes, nm))
    return lookup


def nhb_costs(
    dist_costs: dict[CostDetails, Path], output_folder: Path, model_name: str
) -> None:
    """Create the NHB cost files in the distribution model format.

    Parameters
    ----------
    dist_costs : dict[CostDetails, Path]
        Paths to cost files from `find_costs`.
    output_folder : Path
        Folder to save the NHB costs to.
    model_name : str
        Name of the model the costs are from.
    """
    LOG.info("Creating NHB distribution model costs in: %s", output_folder)
    zoning = nd.get_zoning_system(model_name)
    purp_to_uc = _userclass_purpose_lookup()
    MODE_N = nd.Mode.TRAIN.get_mode_num()

    pbar = tqdm(
        total=len(TIME_PERIODS) * len(nd_consts.ALL_NHB_P) * len(nd_consts.VALID_CA),
        desc="Creating NHB Costs",
        dynamic_ncols=True,
    )
    for ts in TIME_PERIODS:
        key = CostDetails(ts, "dist")
        try:
            costs = _read_dist(dist_costs[key], zoning)
        except KeyError:
            LOG.warning("no cost found for: %s", key)

        for purp in nd_consts.ALL_NHB_P:
            uc = purp_to_uc[purp]
            for ca in nd_consts.VALID_CA:
                out_file = (
                    output_folder
                    / f"nhb_{model_name}_cost_p{purp}_m{MODE_N}_ca{ca}_tp{ts}.csv.bz2"
                )

                file_ops.write_df(costs[uc], out_file)
                LOG.debug("Written: %s", out_file)
                pbar.update()

    pbar.close()


def hb_costs(
    dist_costs: dict[CostDetails, Path], output_folder: Path, model_name: str
) -> None:
    """Create the HB cost files in the distribution model format.

    Parameters
    ----------
    dist_costs : dict[CostDetails, Path]
        Paths to cost files from `find_costs`.
    output_folder : Path
        Folder to save the HB costs to.
    model_name : str
        Name of the model the costs are from.
    """
    LOG.info("Creating NHB distribution model costs in: %s", output_folder)
    zoning = nd.get_zoning_system(model_name)
    purp_to_uc = _userclass_purpose_lookup()
    MODE_N = nd.Mode.TRAIN.get_mode_num()

    # Calculate weighted average cost for all user classes
    tp_weights = {1: 3, 2: 6, 3: 3}
    costs = dict.fromkeys(["business", "commute", "other"], list())
    for ts, weight in tp_weights.items():
        key = CostDetails(ts, "dist")
        try:
            data = _read_dist(dist_costs[key], zoning)
        except KeyError:
            LOG.warning("no cost found for: %s", key)
            continue
        for nm, df in data.items():
            costs[nm].append(df * weight)
    # Sum all time periods together and calculate average
    costs = {nm: sum(c) / sum(tp_weights.values()) for nm, c in costs.items()}

    pbar = tqdm(
        total=len(nd_consts.ALL_HB_P) * len(nd_consts.VALID_CA),
        desc="Creating NHB Costs",
        dynamic_ncols=True,
    )
    for purp in nd_consts.ALL_HB_P:
        uc = purp_to_uc[purp]
        for ca in nd_consts.VALID_CA:
            out_file = (
                output_folder / f"hb_{model_name}_cost_p{purp}_m{MODE_N}_ca{ca}.csv.bz2"
            )

            file_ops.write_df(costs[uc], out_file)
            LOG.debug("Written: %s", out_file)
            pbar.update()


def main(init_logger: bool = True) -> None:
    """Reads the rail cost CSVs and converts to the distribution model format.

    Parameters
    ----------
    init_logger : bool, default True
        If True initialises the logger and creates a log file.
    """
    args = get_arguments()
    params = ExtractRailCostsInputs.load(args.config)

    if init_logger:
        nd_log.get_logger(
            nd_log.get_package_logger_name(),
            params.output_folder / LOG_FILE,
            "Running Extract Rail Costs",
        )
    params.save(params.output_folder / "Extract_rail_costs_parameters.ini")

    dist_costs = find_costs(params.cost_folder, "dist")

    # Create distribution model costs
    cost_output = params.output_folder / f"Distribution Model Costs/{params.model_name}"
    cost_output.mkdir(exist_ok=True, parents=True)
    nhb_costs(dist_costs, cost_output, params.model_name)
    hb_costs(dist_costs, cost_output, params.model_name)


##### MAIN #####
if __name__ == "__main__":
    main()
