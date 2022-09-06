# -*- coding: utf-8 -*-
"""
    Script to skim all SATURN assignments in a given folder to extract
    cost matrices and then convert to the HB and NHB purpose costs for
    the distribution model.
"""

##### IMPORTS #####
from __future__ import annotations

# Standard imports
import argparse
import configparser
import dataclasses
import itertools
import re
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

# Third party imports
import pandas as pd
from tqdm import tqdm

# Local imports
sys.path.append("..")
# pylint: disable=import-error,wrong-import-position
import normits_demand as nd
from normits_demand import logging as nd_log
from normits_demand import constants as nd_consts
from normits_demand.utils import file_ops
from normits_demand.matrices import ufm_converter

# pylint: enable=import-error,wrong-import-position


##### CONSTANTS #####
LOG = nd_log.get_logger(
    nd_log.get_package_logger_name() + ".run_helper_scripts.extract_saturn_costs"
)
LOG_FILE = "Skim_SATURN.log"
SKIM_TYPES = {"T": "Time", "D": "Distance", "M": "Toll", "P": "Penalty"}
TIME_PERIODS = {1: "AM", 2: "IP", 3: "PM", 4: "OP"}
MODEL_USER_CLASSES = {"commute": 1, "business": 2, "other": 3}
INTRAZONAL_COST_FACTOR = 0.5

##### CLASSES #####
class SkimFileNameError(nd.NormitsDemandError):
    """Error raised when parsing an invalid skim file name."""

    def __init__(self, path: Path, message: str, *args: object) -> None:
        self.path = path
        self.message = f"'{path.name}' {message}"
        super().__init__(self.message, *args)


class SkimDetails(NamedTuple):
    """Time period, skim type and user class details for a skim file."""

    time_slice: int
    skim_type: str
    user_class: int


@dataclasses.dataclass
class ExtractCostsInputs:
    """Class to store and manage input parameters for `extract_saturn_costs`.

    Parameters
    ----------
    model_name : str
        Name of the model costs are extracted from.
    output_folder : Path
        Path to folder to store outputs.
    run_skims : bool, default True
        Should SATURN skimming be ran.
    assignments_folder : Path, optional
        Folder containing SATURN assignments, required
        if `run_skims` is True.
    saturn_folder : Path, optional
        Folder containing SATURN EXES, required
        if `run_skims` is True.
    user_classes : set[int], optional
        User classes to get costs for, if not given
        gets costs for all user classes.
    """

    _CONFIG_SECTION = "EXTRACT COSTS PARAMETERS"

    model_name: str
    output_folder: Path
    run_skims: bool = True
    assignments_folder: Path = None
    saturn_folder: Path = None
    user_classes: set[int] = None

    def _run_skim_parameters(self) -> None:
        """Checks the required parameters when run skims is True."""
        for nm in ("assignments_folder", "saturn_folder"):
            if getattr(self, nm) is None:
                raise nd.NormitsDemandError(f"cannot run skims without {nm} parameter")
            path = Path(getattr(self, nm))
            if not path.is_dir():
                raise NotADirectoryError(f"{nm} doesn't exist: {path}")
            setattr(self, nm, path)

    def __post_init__(self) -> None:
        """Check parameters are valid."""
        if self.model_name is None:
            raise nd.NormitsDemandError("model_name parameter is required")

        self.run_skims = bool(self.run_skims)
        if self.run_skims:
            self._run_skim_parameters()

        self.output_folder = Path(self.output_folder)
        if not self.output_folder.is_dir():
            raise NotADirectoryError(
                f"output_folder doesn't exist: {self.output_folder}"
            )

        if self.user_classes:
            self.user_classes = {int(i) for i in self.user_classes}

    @property
    def skim_folder(self) -> Path:
        """Path: Path to the folder for saving skims."""
        return self.output_folder / "Skims"

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
        LOG.info("Extract cost parameters saved to: %s", path)

    @classmethod
    def load(cls, path: Path) -> ExtractCostsInputs:
        """Load parameters from config file.

        Parameters
        ----------
        path : Path
            Path to config file.

        Returns
        -------
        ExtractCostsInputs
            Parameters loaded from `path`.

        Raises
        ------
        configparser.NoSectionError
            If the config file is missing a required section.
        """
        converters = {
            "path": Path,
            "optpath": lambda p: None if p.strip() == "" else Path(p),
            "list": lambda s: None if s.strip() == "" else s.split(),
        }

        config = configparser.ConfigParser(converters=converters)
        config.read(path)

        if not config.has_section(cls._CONFIG_SECTION):
            raise configparser.NoSectionError(cls._CONFIG_SECTION)
        section = config[cls._CONFIG_SECTION]

        params = {}
        params["model_name"] = section.get("model_name")
        params["output_folder"] = section.getpath("output_folder")
        params["run_skims"] = section.getboolean("run_skims", fallback=cls.run_skims)
        params["assignments_folder"] = section.getoptpath("assignments_folder")
        params["saturn_folder"] = section.getoptpath("saturn_folder")
        params["user_classes"] = section.getlist("user_classes")
        return ExtractCostsInputs(**params)


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


def ufms_to_csvs(path: Path, converter: ufm_converter.UFMConverter) -> None:
    """Convert all UFMs with given starting `path` to CSVs.

    CSVs are produced in SATURNs TUBA 2 format.

    Parameters
    ----------
    path : Path
        Path stem for files to search for, finds any files in folder
        `path.parent` which start with `path.stem` and have the
        suffix '.UFM'.
    converter : normits_demand.matrices.ufm.UFMConverter
        Class for converting between UFMs and CSVs.
    """
    for mat in path.parent.glob(f"{path.stem}*.UFM"):
        converter.ufm_to_csv(mat)


def skim_assignments(
    assignments_folder: Path,
    skim_folder: Path,
    saturn_path: Path,
    user_classes: list[int] = None,
) -> None:
    """Skim all SATURN assignments (.UFS) files in `assignments_folder`.

    Creates a new folder (Skims) inside `assignments_folder` which
    contains UFM and CSVs for all the skims.

    Parameters
    ----------
    assignments_folder : Path
        Path to the folder containing all SATURN assignments to skim.
    skim_folder : Path
        Path to the folder to save the outputs.
    saturn_path : Path
        Path to the SATURN EXES folder.
    user_classes : list[int], optional
        List of individual user classes to run the skim for, if not
        given produces the skim for all userclasses combined.
    """
    skim_folder.mkdir(exist_ok=True)
    LOG.info("Skims saved to: %s", skim_folder)

    if user_classes is None:
        user_classes = [None]
    loop_params = itertools.product(assignments_folder.glob("*.UFS"), user_classes)
    pbar = tqdm(
        list(loop_params),
        desc="Skimming assignments",
        dynamic_ncols=True,
    )
    converter = ufm_converter.UFMConverter(saturn_path)
    for path, uc in pbar:
        path = path.resolve()
        if uc is not None:
            output = skim_folder / (path.stem + f"-SKIM_UC{uc}")
        else:
            output = skim_folder / (path.stem + "-SKIM")
        command = ["SKIM_ALL", str(path), str(output)]
        if uc is not None:
            command += ["UC", str(uc)]
        comp_proc = subprocess.run(
            command,
            capture_output=True,
            env=converter.environment,
            check=True,
            shell=True,
        )
        LOG.debug(
            "Skimmed: %s\n%s\n%s",
            path,
            ufm_converter.cmd_strip(comp_proc.stdout),
            ufm_converter.cmd_strip(comp_proc.stderr),
        )
        ufms_to_csvs(output, converter)


def parse_skim_name(path: Path) -> SkimDetails:
    """Parses a skim file name to extract details.

    Extracts time period, skim type and user class from file name.

    Parameters
    ----------
    path : Path
        Path to skim file.

    Returns
    -------
    SkimDetails
        Details extracted from file name.

    Raises
    ------
    SkimFileNameError
        If any details cannot be found in the file name.
    """
    non_letter_number = r"(?:\W|_|\b)"
    skim_types = "|".join(SKIM_TYPES.keys())
    patterns = {
        "time_period": (
            non_letter_number + r"(?:((?:TS\d)|(?:AM|IP|PM|OP)))" + non_letter_number
        ),
        "user_class": (non_letter_number + r"UC(\d+)"),
        "skim_type": (non_letter_number + f"({skim_types})$"),
    }
    name_params = {}
    for nm, pat in patterns.items():
        match = re.search(pat, path.stem, re.I)
        if match:
            name_params[nm] = match.group(1)
        else:
            raise SkimFileNameError(path, f"can't find {nm}")

    if name_params["time_period"].upper().startswith("TS"):
        try:
            time_slice = int(name_params["time_period"][2:])
        except ValueError as err:
            raise SkimFileNameError(
                path, f"time slice ({name_params['time_period']}) isn't an integer"
            ) from err
    else:
        tp_to_ts = {v: k for k, v in TIME_PERIODS.items()}
        try:
            time_slice = tp_to_ts[name_params["time_period"]]
        except KeyError as err:
            raise SkimFileNameError(
                path, f"time period ({name_params['time_period']}) isn't valid"
            ) from err
    try:
        user_class = int(name_params["user_class"])
    except ValueError as err:
        raise SkimFileNameError(
            path, f"user class ({name_params['user_class']}) isn't an integer"
        ) from err
    return SkimDetails(time_slice, name_params["skim_type"], user_class)


def find_csv_skims(
    skim_folder: Path, skim_type: str = None, user_classes: list[int] = None
) -> dict[SkimDetails, Path]:
    """Find CSV files with valid skim names in `skim_folder`.

    Expects skim files to be CSVs with SKIM somewhere in the file name.

    Parameters
    ----------
    skim_folder : Path
        Folder to search for CSVs in.
    skim_type : str, optional
        Only include skims with this type in the output,
        by default include all skims found.
    user_classes : list[int], optional
        Only include skims with given user class, by default
        include all skims found.

    Returns
    -------
    dict[SkimDetails, Path]
        Paths to all skim files found.
    """
    skims = {}
    for path in skim_folder.glob("*SKIM*.csv"):
        try:
            details = parse_skim_name(path)
        except SkimFileNameError as err:
            LOG.warning("%s: %s", err.__class__.__name__, err)
            continue
        if skim_type and details.skim_type != skim_type:
            continue
        if user_classes and details.user_class not in user_classes:
            continue
        skims[details] = path
    LOG.info("Found %s skim CSVs", len(skims))
    return skims


def _purp_to_user_class(purpose: int) -> int:
    """Convert `purpose` to model user class."""
    uc_name = None
    for nm, purps in nd_consts.USER_CLASS_PURPOSES.items():
        if purpose in purps:
            uc_name = nm.strip().lower()
            break
    else:
        raise nd.NormitsDemandError(f"unknown purpose {purpose}")
    if uc_name not in MODEL_USER_CLASSES:
        raise nd.NormitsDemandError(f"unknown user class {uc_name}")
    return MODEL_USER_CLASSES[uc_name]


def _read_skim(
    path: Path, zoning: nd.ZoningSystem, intra_factor: float
) -> pd.DataFrame:
    """Read cost skim and infill intrazonal costs.

    Cost skim is assumed to be a CSV in the SATURN TUBA 2 format.

    Parameters
    ----------
    path : Path
        Path to the cost skim.
    zoning : nd.ZoningSystem
        Zoning system of the cost skim.
    intra_factor : float
        Factor for multiplying the nearest neighbour cost by
        to infill intrazonal costs.

    Returns
    -------
    pd.DataFrame
        Cost matrix with intrazonals filled in.
    """
    # Read skim and make sure zone system is consitent with zoning
    zones = zoning.unique_zones
    df = file_ops.read_df(
        path, index_col=[0, 1], header=None, names=["origin", "destination", "cost"]
    )
    df = df.reindex(index=pd.MultiIndex.from_product((zones, zones)))
    # Convert to kms
    df.loc[:, "cost"] = df["cost"] / 1000

    # Calculate nearest neighbour cost
    row_min = df.groupby(level=0).min()
    col_min = df.groupby(level=1).min()
    nearest_neighbour = pd.concat([row_min, col_min], axis=1).min(axis=1)
    nearest_neighbour.index = pd.MultiIndex.from_arrays(
        (nearest_neighbour.index, nearest_neighbour.index)
    )
    intra_mask = df.index.get_level_values(0) == df.index.get_level_values(1)
    # Apply factored nearest neighbour cost to intrazonals
    df.loc[intra_mask, "cost"] = intra_factor * nearest_neighbour

    nan = df["cost"].isna().sum()
    if nan > 0:
        LOG.error(
            "%s (%.1f%%) cells have missing costs in %s",
            nan,
            (nan / len(df)) * 100,
            path.stem,
        )

    df = df.unstack()
    df = df.droplevel(0, axis=1)
    return df


def nhb_costs(
    dist_skims: dict[SkimDetails, Path],
    output_folder: Path,
    model_name: str,
    intrazonal_factor: float,
) -> None:
    """Create NHB cost files for the distribution model.

    Parameters
    ----------
    dist_skims : dict[SkimDetails, Path]
        Paths to cost skim CSVs.
    output_folder : Path
        Folder to save the NHB cost files to.
    model_name : str
        Name of the model the cost data is from.
    intrazonal_factor : float
        Factor to use for infilling intrazonal costs, this is
        multiplied by the zones nearest neighbour cost.
    """
    LOG.info("Creating NHB distribution model costs in: %s", output_folder)
    zoning = nd.get_zoning_system(model_name)

    time_periods = list(range(1, 5))
    pbar = tqdm(
        total=len(time_periods) * len(nd_consts.ALL_NHB_P),
        dynamic_ncols=True,
        desc="Creating NHB Costs",
    )
    for tp in time_periods:
        # Cache to avoid reading the same file twice
        # when multiple purposes use the same user class
        cache = {}

        for purp in nd_consts.ALL_NHB_P:
            uc = _purp_to_user_class(purp)
            out_file = output_folder / f"nhb_{model_name}_cost_p{purp}_m3_tp{tp}.csv.bz2"

            key = SkimDetails(tp, "D", uc)
            try:
                skim_path = dist_skims[key]
            except KeyError:
                LOG.warning("cannot find skim for %s", key)
                pbar.update()
                continue

            if key not in cache:
                cache[key] = _read_skim(skim_path, zoning, intrazonal_factor)
            file_ops.write_df(cache[key], out_file)
            LOG.debug("Written: %s", out_file)
            pbar.update()

    pbar.close()


def hb_costs(
    dist_skims: dict[SkimDetails, Path],
    output_folder: Path,
    model_name: str,
    intrazonal_factor: float,
) -> None:
    """Calculate HB costs from assignment distance skims.

    HB costs are calculated as a weighted average of all time periods.

    Parameters
    ----------
    dist_skims : dict[SkimDetails, Path]
        Paths to cost skim CSVs.
    output_folder : Path
        Folder to save the NHB cost files to.
    model_name : str
        Name of the model the cost data is from.
    intrazonal_factor : float
        Factor to use for infilling intrazonal costs, this is
        multiplied by the zones nearest neighbour cost.
    """
    LOG.info("Creating HB distribution model costs in: %s", output_folder)
    zoning = nd.get_zoning_system(model_name)
    tp_weights = {1: 3, 2: 6, 3: 3}
    LOG.info(
        "HB costs calculated using weighted average"
        " of each time period, weightings are %s",
        tp_weights,
    )

    pbar = tqdm(
        nd_consts.ALL_HB_P,
        dynamic_ncols=True,
        desc="Creating HB Costs",
    )
    # Cache to avoid reading the same file twice
    # when multiple purposes use the same user class
    cache = {}

    for purp in pbar:
        uc = _purp_to_user_class(purp)
        out_file = output_folder / f"hb_{model_name}_cost_p{purp}_m3.csv.bz2"

        if uc not in cache:
            skim_paths = {}
            try:
                key = None
                for tp in tp_weights:
                    key = SkimDetails(tp, "D", uc)
                    skim_paths[tp] = dist_skims[key]
            except KeyError:
                LOG.warning("cannot find skim for %s", key)
                continue

            costs = []
            for tp, path in skim_paths.items():
                df = _read_skim(path, zoning, intrazonal_factor)
                costs.append(df * tp_weights[tp])
            cache[uc] = sum(costs) / sum(tp_weights.values())

        file_ops.write_df(cache[uc], out_file)
        LOG.debug("Written: %s", out_file)


def main(init_logger: bool = True) -> None:
    """Get commandline arguments and produce skims and cost files."""
    args = get_arguments()
    params = ExtractCostsInputs.load(args.config)

    if init_logger:
        nd_log.get_logger(
            nd_log.get_package_logger_name(),
            params.output_folder / LOG_FILE,
            "Running Extract SATURN Costs",
        )
    params.save(params.output_folder / "Extract_costs_parameters.ini")

    if params.run_skims:
        skim_assignments(
            params.assignments_folder,
            params.skim_folder,
            params.saturn_folder,
            params.user_classes,
        )
    dist_skims = find_csv_skims(params.skim_folder, "D", params.user_classes)

    # Create distribution model costs
    cost_output = params.output_folder / f"Distribution Model Costs/{params.model_name}"
    cost_output.mkdir(exist_ok=True, parents=True)
    nhb_costs(dist_skims, cost_output, params.model_name, INTRAZONAL_COST_FACTOR)
    hb_costs(dist_skims, cost_output, params.model_name, INTRAZONAL_COST_FACTOR)


##### MAIN #####
if __name__ == "__main__":
    main()
