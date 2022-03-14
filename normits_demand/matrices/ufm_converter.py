# -*- coding: utf-8 -*-
"""
    Module containing functionality to convert CSV matrices into
    SATURN's UFM files.
"""

##### IMPORTS #####
# Standard imports
import os
import subprocess
from pathlib import Path

# Third party imports

# Local imports
from normits_demand import logging as nd_log

##### CONSTANTS #####
LOG = nd_log.get_logger(__name__)

##### CLASSES #####
class UFMConverter:
    """Class for converting matrices to and from SATURN's UFM file format.

    Parameters
    ----------
    saturn_folder : Path
        Path to the folder containing SATURN's EXES and batch files.
    """

    def __init__(self, saturn_folder: Path) -> None:
        self._saturn_folder = Path(saturn_folder)
        if not self._saturn_folder.is_dir():
            raise NotADirectoryError(
                f"saturn_folder isn't an existing folder: {self._saturn_folder}"
            )
        self._environment = None

    @property
    def environment(self) -> os._Environ:  # pylint: disable=protected-access
        """os._Environ : environment variables dictionary with `saturn_folder`
        added to "PATH".
        """
        if self._environment is None:
            self._environment = update_env(self._saturn_folder)
        return self._environment

    def ufm_to_tba2(self, ufm: Path, csv: Path = None) -> Path:
        """Convert UFM file to a CSV in TUBA 2 format.

        TUBA 2 format is 3 columns without a header row containing
        the origin zone number, destination zone number and the
        matrix value. This format excludes any cells which are 0.

        Parameters
        ----------
        ufm : Path
            Path to UFM file to convert.
        csv : Path, optional
            Path to output CSV file, defaults to `ufm.csv`
            if not given.

        Returns
        -------
        Path
            Path to output CSV file.

        Raises
        ------
        FileNotFoundError
            If the output CSV file isn't created.
        """
        ufm = ufm.resolve()
        if csv is None:
            csv = ufm.with_suffix(".csv")
        else:
            csv = csv.resolve()

        LOG.debug("Converting UFM to TUBA 2 CSV: %s", ufm)
        comp_proc = subprocess.run(
            ["UFM2TBA2", str(ufm), str(csv)],
            capture_output=True,
            env=self.environment,
            check=True,
            shell=True,
        )

        msg_data = (csv, cmd_strip(comp_proc.stdout), cmd_strip(comp_proc.stderr))
        if csv.exists():
            LOG.debug("Created CSV: %s\n%s\n%s", *msg_data)
        else:
            LOG.error("Failed to create CSV: %s\n%s\n%s", *msg_data)
            raise FileNotFoundError(f"error creating {csv}")
        return csv

    def square_csv_to_ufm(self, csv: Path, ufm: Path = None, title: str = None) -> Path:
        """Convert CSV in square format to UFM file.

        CSV should be a square matrix with no header row and the first
        column containing zone name.

        Parameters
        ----------
        csv : Path
            Path to CSV file.
        ufm : Path, optional
            Path to output CSV file, defaults to `csv.UFM`
            if not given.
        title : str, optional
            Title of the matrix (UFM metadata), defaults to
            `csv` name if not given.

        Returns
        -------
        Path
            Path to output UFM.

        Raises
        ------
        FileNotFoundError
            If the output UFM file isn't created.
        """
        # Should be square matrix with zone names in first column and no header rows
        LOG.debug("Converting CSV in square format to UFM: %s", csv)
        if ufm is None:
            ufm = csv.with_suffix(".UFM")
        if title is None:
            title = csv.stem

        key_data = [
            1,  # Select input data
            csv.resolve(),
            1,  # Read file with default format (spreadsheet)
            5,  # First column is zone name
            0,  # Read data
            14,  # Dump to UFM file
            1,  # Output as is to UFM
            ufm.resolve(),
            title,  # Matrix title
            1,  # Close output file
            0,  # Exit
            "Y",
        ]
        key_path = csv.with_suffix(".KEY")
        with open(key_path, "wt") as file:
            file.writelines(f"{l}\n" for l in key_data)
        LOG.debug("Written KEY file: %s", key_path)

        comp_proc = subprocess.run(
            ["MX", "I", "KEY", str(key_path), "VDU", str(key_path.with_suffix(".VDU"))],
            capture_output=True,
            env=self.environment,
            check=True,
            shell=True,
        )
        msg_data = (ufm, cmd_strip(comp_proc.stdout), cmd_strip(comp_proc.stderr))
        if ufm.exists():
            LOG.debug("Created UFM: %s\n%s\n%s", *msg_data)
        else:
            LOG.error("Failed creating UFM: %s\n%s\n%s", *msg_data)
            raise FileNotFoundError(f"error creating {csv}")
        return csv


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


def cmd_strip(stdout: bytes) -> str:
    """Convert to str and strip newlines from subprocess `stdout` or `stderr`."""
    return stdout.decode().strip().replace("\r\n", "\n")
