# -*- coding: utf-8 -*-
"""
    Script to convert the train distribution model outputs to the
    CUBE format for MiRANDA.

    See Also
    --------
    `normits_demand.matrices.cube_mat_converter.CUBEMatConverter`
        Class for performing the conversion.
"""

##### IMPORTS #####
# Standard imports
import dataclasses
from pathlib import Path
import sys

# Third party imports

# Local imports
sys.path.append("..")
# pylint: disable=import-error,wrong-import-position
import normits_demand as nd
from normits_demand import logging as nd_log
from normits_demand.matrices import cube_mat_converter

# pylint: enable=import-error,wrong-import-position


##### CONSTANTS #####
LOG = nd_log.get_logger(
    nd_log.get_package_logger_name() + ".run_helper_scripts.convert_od_to_cube"
)
LOG_FILE = "OD_to_CUBE.log"

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
    csv_to_mat: list[CSVToMatParameters]
        List of parameters to use for conversion, conversion
        will be done separately for each item in the list.
    """

    cube_voyager_path: Path
    model_name: str
    output_folder: Path
    csv_to_mat: list[CSVToMatParameters]


##### FUNCTIONS #####
def main(params: ODToCUBEParameters, init_logger: bool = True) -> None:
    """Run OD CSV matrix to CUBE .mat format converter.

    _extended_summary_

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

    zoning = nd.get_zoning_system(params.model_name)

    converter = cube_mat_converter.CUBEMatConverter(params.cube_voyager_path)

    for csv_params in params.csv_to_mat:
        converter.csv_to_mat(
            len(zoning.unique_zones),
            csv_params.csv_paths,
            params.output_folder / csv_params.matrix_name,
            csv_params.matrix_infill,
        )


##### MAIN #####
if __name__ == "__main__":
    # TODO Replace testing code
    csv_folder = Path(
        r"C:\WSP_Projects\MidMITs\02 MidMITs\Outputs\MiRANDA Assignments\CUBE Mat Conversion"
    )
    parameters = ODToCUBEParameters(
        cube_voyager_path=Path(r"C:\Program Files\Citilabs\CubeVoyager\VOYAGER.EXE"),
        model_name="miranda",
        output_folder=csv_folder,
        csv_to_mat=[
            CSVToMatParameters(
                "test.mat", {f"test{u}": csv_folder / f"test_{u}.csv" for u in (1, 2, 3)}
            )
        ],
    )
    main(parameters)
