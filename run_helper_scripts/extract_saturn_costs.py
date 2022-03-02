# -*- coding: utf-8 -*-
"""
    Script to skim all SATURN assignments in a given folder
    to extract cost matrices.
"""

##### IMPORTS #####
from __future__ import annotations

# Standard imports
import argparse
import configparser
import dataclasses
import itertools
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

# Third party imports
from tqdm import tqdm

# Local imports
sys.path.append("..")
# pylint: disable=import-error,wrong-import-position
from normits_demand import logging as nd_log

# pylint: enable=import-error,wrong-import-position

##### CONSTANTS #####
LOG = nd_log.get_logger(
    nd_log.get_package_logger_name() + ".run_helper_scripts.skim_saturn_assignments"
)
LOG_FILE = "Skim_SATURN.log"
SKIM_TYPES = {"T": "Time", "D": "Distance", "M": "Toll", "P": "Penalty"}
TIME_PERIODS = {1: "AM", 2: "IP", 3: "PM", 4: "OP"}
SKIM_FOLDER = "Skims"
OUTPUT_FOLDER_FORMAT = "{import_home}/modal/car_and_passenger/costs/{model_name}"

##### CLASSES #####
class SkimFileNameError(Exception):
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

    output_folder: Path
    run_skims: bool = True
    assignments_folder: Path = None
    saturn_folder: Path = None
    user_classes: set[int] = None

    def _run_skim_parameters(self) -> None:
        """Checks the required parameters when run skims is True."""
        for nm in ("assignments_folder", "saturn_folder"):
            if getattr(self, nm) is None:
                raise ValueError(f"cannot run skims without {nm} parameter")
            path = Path(getattr(self, nm))
            if not path.is_dir():
                raise NotADirectoryError(f"{nm} doesn't exist: {path}")
            setattr(self, nm, path)

    def __post_init__(self) -> None:
        """Check parameters are valid."""
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
        return self.output_folder / SKIM_FOLDER

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
        params["output_folder"] = section.getpath("output_folder")
        params["run_skims"] = section.getboolean("run_skims", fallback=cls.run_skims)
        params["assignments_folder"] = section.getoptpath("assignments_folder")
        params["saturn_folder"] = section.getoptpath("saturn_folder")
        params["user_classes"] = section.getlist("user_classes")
        return ExtractCostsInputs(**params)


##### FUNCTIONS #####
def update_env(saturn_path: Path) -> os._Environ:  # pylint: disable=protected-access
    """Creates a copy of environment variables and adds SATURN path.

    Parameters
    ----------
    saturn_path: Path
        Path to folder containing SATURN batch files.

    Returns
    -------
    os._Environ
        A copy of `os.environ` with the `saturn_path` added to the
        "PATH" variable.

    Raises
    ------
    NotADirectoryError
        If `saturn_path` isn't an existing folder.
    """
    if not saturn_path.is_dir():
        raise NotADirectoryError(saturn_path)
    new_env = os.environ.copy()
    sat_paths = rf'"{saturn_path.resolve()}";"{saturn_path.resolve()}\BATS";'
    new_env["PATH"] = sat_paths + new_env["PATH"]
    return new_env


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


def _cmd_strip(stdout: bytes) -> str:
    """Convert to str and strip newlines from subprocess `stdout` or `stderr`."""
    return stdout.decode().strip().replace("\r\n", "\n")


def ufms_to_csvs(
    path: Path, saturn_env: os._Environ  # pylint: disable=protected-access
) -> None:
    """Convert all UFMs with given starting `path` to CSVs.

    CSVs are produced in SATURNs TUBA 2 format.

    Parameters
    ----------
    path : Path
        Path stem for files to search for, finds any files in folder
        `path.parent` which start with `path.stem` and have the
        suffix '.UFM'.
    saturn_env : os._Environ
        Environment with the SATURN path included.
    """
    for mat in path.parent.glob(f"{path.stem}*.UFM"):
        csv = mat.with_suffix(".csv")
        comp_proc = subprocess.run(
            ["UFM2TBA2", str(mat), str(csv)],
            capture_output=True,
            env=saturn_env,
            check=True,
            shell=True,
        )
        LOG.debug(
            "Converting to CSV: %s\n%s\n%s",
            mat,
            _cmd_strip(comp_proc.stdout),
            _cmd_strip(comp_proc.stderr),
        )


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
    saturn_env = update_env(saturn_path)
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
            env=saturn_env,
            check=True,
            shell=True,
        )
        LOG.debug(
            "Skimmed: %s\n%s\n%s",
            path,
            _cmd_strip(comp_proc.stdout),
            _cmd_strip(comp_proc.stderr),
        )
        ufms_to_csvs(output, saturn_env)


def parse_skim_name(path: Path) -> SkimDetails:
    """Parser a skim file name to extract details.

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
        if skim_type and details.skim_type == skim_type:
            continue
        if user_classes and details.user_class not in user_classes:
            continue
        skims[details] = path
    LOG.info("Found %s skim CSVs", len(skims))
    return skims


def main(init_logger: bool = True) -> None:
    """Get commandline arguments and produce skims."""
    args = get_arguments()
    params = ExtractCostsInputs.load(args.config)

    if init_logger:
        nd_log.get_logger(
            nd_log.get_package_logger_name(),
            params.output_folder / LOG_FILE,
            "Running SATURN Skims",
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
    # TODO Convert skims into distribution model cost format, output to cost folder


##### MAIN #####
if __name__ == "__main__":
    main()
