# -*- coding: utf-8 -*-
"""
    Script to skim all SATURN assignments in a given folder
    to extract cost matrices.
"""

##### IMPORTS #####
# Standard imports
import argparse
import itertools
import logging
import os
import subprocess
from pathlib import Path

# Third party imports
from tqdm import tqdm


##### CONSTANTS #####
LOG = logging.getLogger(__name__)
SKIM_TYPES = {"T": "Time", "D": "Distance", "M": "Toll", "P": "Penalty"}

##### CLASSES #####

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
    - `saturn_folder`: Path to SATURN EXES folder
    - `assignments_folder`: Path to folder containing assignments
    Optional arguments:
    - `user_classes`: List of individual user classes to run the skim
      for, if not given produces the skim for all userclasses combined.

    Returns
    -------
    argparse.Namespace
        Argument values parsed from commandline.

    Raises
    ------
    NotADirectoryError
        If the input folders don't exist.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("saturn_folder", type=Path, help="Path to SATURN EXES folder")
    parser.add_argument(
        "assignments_folder", type=Path, help="Path to folder containing assignments"
    )
    parser.add_argument(
        "-u",
        "--user_classes",
        type=int,
        nargs="+",
        help="List of individual user classes to run the skim for, "
        "if not given produces the skim for all userclasses combined.",
    )

    args = parser.parse_args()
    for nm in ("saturn_folder", "assignments_folder"):
        path = getattr(args, nm)
        if not path.is_dir():
            raise NotADirectoryError(f"{nm} is not an existing folder: {path}")
    return args


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
        tqdm.write(
            "\n".join(
                [
                    f"Converting: {mat}",
                    comp_proc.stdout.decode(),
                    comp_proc.stderr.decode(),
                ]
            )
        )


def skim_assignments(
    assignments_folder: Path,
    saturn_path: Path,
    user_classes: list[int] = None,
) -> Path:
    """Skim all SATURN assignments (.UFS) files in `assignments_folder`.

    Creates a new folder (Skims) inside `assignments_folder` which
    contains UFM and CSVs for all the skims.

    Parameters
    ----------
    assignments_folder : Path
        Path to the folder containing all SATURN assignments to skim.
    saturn_path : Path
        Path to the SATURN EXES folder.
    user_classes : list[int], optional
        List of individual user classes to run the skim for, if not
        given produces the skim for all userclasses combined.

    Returns
    -------
    Path
        Path to the 'Skims' folder containing outputs.
    """
    skim_folder = assignments_folder / "Skims"
    skim_folder.mkdir(exist_ok=True)

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
        tqdm.write(
            "\n".join(
                [
                    f"Skimmed: {path}\n",
                    comp_proc.stdout.decode(),
                    comp_proc.stderr.decode(),
                    "Converting UFMs to CSVs",
                ]
            )
        )
        ufms_to_csvs(output, saturn_env)
    return skim_folder


def main() -> None:
    """Get commandline arguments and produce skims."""
    args = get_arguments()
    skim_assignments(args.assignments_folder, args.saturn_folder, args.user_classes)
    # TODO Convert skims into distribution model cost format, output to cost folder


##### MAIN #####
if __name__ == "__main__":
    main()
