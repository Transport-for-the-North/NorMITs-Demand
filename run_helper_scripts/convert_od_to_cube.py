# -*- coding: utf-8 -*-
"""
Script to convert the train distribution model outputs to the CUBE format for MiRANDA.

See Also
--------
`normits_demand.matrices.cube_mat_converter.CUBEMatConverter`
    Class for performing the conversion.
"""

##### IMPORTS #####
# Standard imports
import collections
import dataclasses
from pathlib import Path
import pprint
import sys
from typing import Iterator, NamedTuple


# Third party imports
import pandas as pd
from tqdm import tqdm

# Local imports
sys.path.append("..")
# pylint: disable=import-error,wrong-import-position
import normits_demand as nd
from normits_demand import logging as nd_log
from normits_demand.matrices import cube_mat_converter
from normits_demand.utils import general

# pylint: enable=import-error,wrong-import-position


##### CONSTANTS #####
LOG = nd_log.get_logger(
    nd_log.get_package_logger_name() + ".run_helper_scripts.convert_od_to_cube"
)
LOG_FILE = "OD_to_CUBE.log"
MIRANDA_USERCLASSES = [
    "EB_CAF",
    "EB_CAT",
    "EB_NCA",
    "Com_CAF",
    "Com_CAT",
    "Com_NCA",
    "Oth_CAF",
    "Oth_CAT",
    "Oth_NCA",
]
TIME_PERIODS = {1: "AM", 2: "IP", 3: "PM"}

##### CLASSES #####
@dataclasses.dataclass
class CSVToMatParameters:
    """Parameters for converting multiple CSVs to a single CUBE .mat file.

    Attributes
    ----------
    matrix_name: str
        Name of the output CUBE .mat file to create.
    csv_paths: dict[str, Path]
        Paths to the CSVs to be used for creating the matrix,
        the CSVs should have 3 columns: origin, destination and trips
        with no header row. The dictionary keys provide the name for
        the matrix level in the output file.
    matrix_infill: float, default 1
        Factor to divide matrix by upon creation.
    """

    matrix_name: str
    csv_paths: dict[str, Path]
    matrix_infill: float = 1


@dataclasses.dataclass
class ODToCUBEParameters:
    """Parameters for running OD to CUBE conversion process.

    Attributes
    ----------
    cube_voyager_path: Path
        Path to the CUBE Voyager executable.
    model_name: str
        Name of the model matrices are from.
    output_folder: Path
        Folder to save output file.
    full_od_folder: Path
        Folder containing the full OD matrix outputs from
        the distribution model.
    distribution_iteration: str
        Name of the distribution model iteration e.g. 'iter9.6.1'.
    """

    cube_voyager_path: Path
    model_name: str
    output_folder: Path
    full_od_folder: Path
    distribution_iteration: str


class MatrixDetails(NamedTuple):
    """Time period and user class for a matrix."""

    time_period: int
    user_class: str


##### FUNCTIONS #####
def get_file_lookup(folder: Path) -> dict[MatrixDetails, list[Path]]:
    """Find all matrices in folder and get user class and time period information.

    Parameters
    ----------
    folder : Path
        Folder containing full OD matrices output from distribution model.

    Returns
    -------
    dict[MatrixDetails, list[Path]]
        List of all matrix files for each MiRANDA time period and userclass.
    """
    uc_lookup = {1: "Com", **dict.fromkeys((2, 12), "EB")}
    all_purposes = set(sum(list(nd.TripOrigin.get_purpose_dict().values()), []))
    uc_lookup.update(dict.fromkeys(all_purposes - set(uc_lookup), "Oth"))

    file_lookup = collections.defaultdict(list)
    for file in folder.iterdir():
        params = general.fname_to_calib_params(
            file.stem, get_trip_origin=True, get_matrix_format=True
        )

        if params["tp"] not in TIME_PERIODS:
            continue

        try:
            if params["ca"] == 1 or params["trip_origin"] == "nhb":
                ca = "NCA"
            elif params["ca"] != 2:
                raise ValueError(f"unknown ca value of {params['ca']}")
            elif params["matrix_format"] == "od_from" and params["trip_origin"] == "hb":
                ca = "CAF"
            elif params["matrix_format"] == "od_to" and params["trip_origin"] == "hb":
                ca = "CAT"
            else:
                raise ValueError(f"unknown parameters {params}")
        except ValueError as err:
            print(file.name, err)

        uc = uc_lookup[params["p"]]
        file_lookup[MatrixDetails(params["tp"], f"{uc}_{ca}")].append(file)

    return file_lookup


def combine_full_matrices(
    files_lookup: dict[MatrixDetails, list[Path]],
    output_folder: Path,
    overwrite: bool = False,
) -> dict[str, dict[str, Path]]:
    """Combine purpose matrices into the MiRANDA user classes.

    Parameters
    ----------
    files_lookup : dict[MatrixDetails, list[Path]]
        List of all matrix files for each MiRANDA time period and userclass,
        from `get_file_lookup`.
    output_folder : Path
        Folder to save MiRANDA user class matrices in, as CSVs.
    overwrite : bool, default False
        Recreate the combined output matrix even if it already exists,
        otherwise don't bother.

    Returns
    -------
    dict[str, dict[str, Path]]
        Paths to the MiRANDA user class matrices, where the main
        dictionary key is the time period and the nested dictionary
        key is the MiRANDA user class.

    Raises
    ------
    KeyError
        If a `MIRANDA_USERCLASSES` is missing.
    """
    output_folder.mkdir(exist_ok=True)

    new_matrices: dict[str, dict[str, Path]] = collections.defaultdict(dict)
    pbar: Iterator[tuple[MatrixDetails, list[Path]]] = tqdm(
        files_lookup.items(), desc="Combining OD matrices", dynamic_ncols=True
    )
    for details, files in pbar:
        tp = TIME_PERIODS[details.time_period]
        out = output_folder / f"{tp}_{details.user_class}.csv"

        if overwrite or not out.exists():
            matrix: pd.DataFrame = pd.read_csv(files[0], index_col=0)
            for path in files[1:]:
                matrix += pd.read_csv(path, index_col=0)
            matrix.index.name = "origin"
            matrix.columns.name = "destination"  # pylint: disable=no-member
            matrix = matrix.stack().to_frame("trips")  # pylint: disable=no-member
            matrix.reset_index().to_csv(out, index=False, header=False)

        new_matrices[tp][details.user_class] = out

    # Sort internal dict
    for tp, paths in new_matrices.items():
        try:
            new_matrices[tp] = {k: paths[k] for k in MIRANDA_USERCLASSES}
        except KeyError as err:
            raise KeyError(f"{tp} missing user class {err}") from err
    return new_matrices


def main(params: ODToCUBEParameters, init_logger: bool = True) -> None:
    """Run OD CSV matrix to CUBE .mat format converter.

    Parameters
    ----------
    params : ODToCUBEParameters
        Parameters for running the process.
    init_logger : bool, default True
        Initialise NorMITs demand logger.
    """
    params.output_folder.mkdir(exist_ok=True, parents=True)
    if init_logger:
        nd_log.get_logger(
            nd_log.get_package_logger_name(),
            params.output_folder / LOG_FILE,
            "Running OD to CUBE Matrices",
        )

    # Convert full OD matrices to MiRANDA userclasses
    od_files = get_file_lookup(params.full_od_folder)
    LOG.info(
        "File lookup:\n%s",
        pprint.pformat({k: [f.name for f in files] for k, files in od_files.items()}),
    )
    uc_matrices = combine_full_matrices(od_files, params.output_folder / "CSVs")
    LOG.info(
        "%s user class matrices:\n%s",
        params.model_name,
        pprint.pformat(
            {tp: {k: f.name for k, f in files.items()} for tp, files in uc_matrices.items()}
        ),
    )

    # Convert CSVs to CUBE matrices
    zoning = nd.get_zoning_system(params.model_name)
    converter = cube_mat_converter.CUBEMatConverter(params.cube_voyager_path)

    LOG.info("Converting to CUBE matrices")
    for tp, csvs in uc_matrices.items():
        converter.csv_to_mat(
            len(zoning.unique_zones),
            csvs,
            params.output_folder / f"{tp.upper()}_SupplyGroup_PTAssign.mat",
        )


##### MAIN #####
if __name__ == "__main__":
    # TODO Replace testing code
    dm_iteration = "iter9.6.2"
    parameters = ODToCUBEParameters(
        cube_voyager_path=Path(r"C:\Program Files\Citilabs\CubeVoyager\VOYAGER.EXE"),
        model_name="miranda",
        output_folder=Path(
            r"C:\WSP_Projects\MidMITs\02 MidMITs\Outputs"
            fr"\MiRANDA Assignments\{dm_iteration}\CUBE Matrices"
        ),
        full_od_folder=Path(
            fr"T:\MidMITs Demand\Distribution Model\{dm_iteration}"
            r"\train\Final Outputs\Full OD Matrices"
        ),
        distribution_iteration=dm_iteration,
    )
    main(parameters)
