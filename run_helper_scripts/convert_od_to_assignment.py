# -*- coding: utf-8 -*-
"""
    Script to convert compiled OD PCU matrices into stacked UFMs ready
    for SATURN assignments.
"""

##### IMPORTS #####
# Standard imports
import dataclasses
import sys
from pathlib import Path

# Third party imports
from tqdm import tqdm

# Local imports
sys.path.append("..")
# pylint: disable=import-error,wrong-import-position
from normits_demand import logging as nd_log
from normits_demand.matrices import ufm_converter
from normits_demand.utils import file_ops

# pylint: enable=import-error,wrong-import-position


##### CONSTANTS #####
LOG = nd_log.get_logger(
    nd_log.get_package_logger_name() + ".run_helper_scripts.convert_od_to_assignment"
)
LOG_FILE = "OD_to_assignment.log"

##### CLASSES #####
@dataclasses.dataclass
class ODToAssignmentParameters:
    """Parameters for running `convert_od_to_assignment` module."""

    # TODO Add functionality to save/load to config
    output_folder: Path
    model_name: str
    matrix_paths: dict[str, list[Path]]
    saturn_folder: Path


##### FUNCTIONS #####
def _csv_to_ufm(
    csv: Path,
    output_folder: Path,
    matrix_name: str,
    converter: ufm_converter.UFMConverter,
) -> Path:
    """Converts CSV file to UFM.

    Parameters
    ----------
    csv : Path
        Path to CSV file which is a square matrix with the first
        row and column containing the zone names.
    output_folder : Path
        Folder to save UFM to.
    matrix_name : str
        Name of output UFM file.
    converter : ufm_converter.UFMConverter
        Instance of UFMConverter class which is used to call
        the SATURN functionality to do the conversion.

    Returns
    -------
    Path
        Path to the output UFM file.

    See Also
    --------
    `ufm_converter.UFMConverter.square_csv_to_ufm`: which performs the conversion.
    """
    LOG.debug("Converting matrix to UFM: %s", csv)

    # Drop header line
    df = file_ops.read_df(csv, find_similar=True, index_col=[0])
    no_header = output_folder / (csv.stem + "-no_header.csv")
    df.to_csv(no_header, header=False)
    del df

    ufm = converter.square_csv_to_ufm(
        no_header, output_folder / f"{matrix_name}.UFM", matrix_name
    )
    return ufm


def main(params: ODToAssignmentParameters, init_logger: bool = True) -> None:
    """Run conversion from OD PCU matrices to stacked assignment UFMs.

    Parameters
    ----------
    params : ODToAssignmentParameters
        Parameters for running the conversion.
    init_logger : bool, default True
        If True, initialises the NorMITs demand logger.
    """
    params.output_folder.mkdir(exist_ok=True, parents=True)
    if init_logger:
        nd_log.get_logger(
            nd_log.get_package_logger_name(),
            params.output_folder / LOG_FILE,
            "Running OD to Assignment Matrices",
        )

    converter = ufm_converter.UFMConverter(params.saturn_folder)
    pbar = tqdm(
        total=len(params.matrix_paths),
        desc="Creating assignment matrices",
        dynamic_ncols=True,
    )
    for ts, uc_paths in params.matrix_paths.items():

        matrices = []
        for uc, path in enumerate(uc_paths, 1):
            if path.suffix.lower() != ".ufm":
                matrix_name = f"{params.model_name}_TS{ts}_UC{uc}"
                path = _csv_to_ufm(path, params.output_folder, matrix_name, converter)
            matrices.append(path)

        converter.stack(
            matrices, params.output_folder / f"{params.model_name}_stacked_TS{ts}.UFM"
        )
        pbar.update()
    pbar.close()


##### MAIN #####
if __name__ == "__main__":
    distribution_model_iteration = "9.7-COVID.1"
    year = 2021
    mat_folder = Path(
        fr"T:\MidMITs Demand\Distribution Model\iter{distribution_model_iteration}"
        r"\car_and_passenger\Final Outputs\Compiled OD Matrices\PCU"
    )
    prior_folder = Path(
        r"C:\WSP_Projects\MidMITs\00 From Client\MiHAM From Motts"
        r"\2. Assignment Matrices\Prior Matrices"
    )
    matrix_paths = {}
    for t in (1, 2, 3):
        ucs = [
            mat_folder / f"synthetic_od_{u}_yr{year}_m3_tp{t}.csv"
            for u in ("business", "commute", "other")
        ]
        ucs += [prior_folder / f"MD_b15_mat_v032_TS1_UC{u}.UFM" for u in (4, 5)]
        matrix_paths[t] = ucs

    parameters = ODToAssignmentParameters(
        output_folder=mat_folder / "UFMs",
        model_name="miham",
        matrix_paths=matrix_paths,
        saturn_folder=Path(r"C:\SATWIN\XEXES 11.5.05J Beta"),
    )
    main(parameters)
