# -*- coding: utf-8 -*-
"""Run Script for the EDGE CUBE extractor."""
# ## IMPORTS ## #
# Standard imports
import os
import sys
import shutil

# Third party imports
from pathlib import Path
from tqdm import tqdm


# Local imports
sys.path.append(".")
sys.path.append("..")
# pylint: disable=import-error,wrong-import-position
from normits_demand import logging as nd_log
from normits_demand.tools import edge_cube_extractor
from normits_demand.utils import file_ops

# TODO (MI) Setup user inputs using BaseConfig

# ## USER INPUTS ## #
# cube catalogue setup
CUBE_EXE = Path("C:/Program Files/Citilabs/CubeVoyager/VOYAGER.EXE")
CUBE_CAT_PATH = Path(r"C:\NorMITs\NorTMS_T3_Model_v8.16b")
CAT_RUN_DIR = "Runs"
CUBE_RUN_ID = "ILP_2018"

# Input files
TLC_OVERWRITE_PATH = Path(r"C:\NorMITs\inputs\TLC_Overwrite_MOIRA.csv")

# Output location
OUT_PATH = Path(r"C:\NorMITs\outputs")

# ## CONSTANTS ## #
# logger
LOG_FILE = "Export_BaseMatrices_Logfile.Log"
LOG = nd_log.get_logger(f"{nd_log.get_package_logger_name()}.run_edge_cube_extractor")

# Derived from inputs
CUBE_CAT_RUN_PATH = CUBE_CAT_PATH / CAT_RUN_DIR / CUBE_RUN_ID

# ## CLASSES ## #


# ## FUNCTIONS ## #
def run_extractor():
    """Process Fixed objects."""
    # time periods
    periods = ["AM", "IP", "PM", "OP"]

    # copy Cube files
    for period in periods:
        # read distance matrix
        file_ops.check_file_exists(
            CUBE_CAT_RUN_PATH / f"Outputs/BaseAssign/{period}_stn2stn_costs.csv"
        )
        shutil.copy2(
            CUBE_CAT_RUN_PATH / f"Outputs/BaseAssign/{period}_stn2stn_costs.csv",
            OUT_PATH / f"{period}_stn2stn_costs.csv",
        )

        # read iRSj props
        file_ops.check_file_exists(
            CUBE_CAT_RUN_PATH / f"Outputs/BaseAssign/{period}_iRSj_probabilities.h5"
        )
        shutil.copy2(
            CUBE_CAT_RUN_PATH / f"Outputs/BaseAssign/{period}_iRSj_probabilities.h5",
            OUT_PATH / f"{period}_iRSj_probabilities.h5",
        )

        LOG.info("Distance and Probability matrices for period %s has been copied", period)

    # produce TLC lookup
    file_ops.check_file_exists(TLC_OVERWRITE_PATH)
    tlc_overwrite = file_ops.read_df(TLC_OVERWRITE_PATH)
    stns_tlc = edge_cube_extractor.stnzone_2_stn_tlc(
        CUBE_CAT_RUN_PATH / "Inputs/Network/Station_Connectors.csv",
        CUBE_CAT_RUN_PATH / "Inputs/Network/TfN_Rail_Nodes.csv",
        CUBE_CAT_RUN_PATH / "Inputs/Network/External_Station_Nodes.csv",
        tlc_overwrite,
    )
    # write TLC Lookup
    file_ops.write_df(stns_tlc, OUT_PATH / "TLCs.csv", index=False)

    # PT Demand to time periods F/T
    edge_cube_extractor.pt_demand_from_to(CUBE_EXE, CUBE_CAT_PATH, CUBE_CAT_RUN_PATH, OUT_PATH)
    LOG.info("NoRMS matrices converted to OMX successfully")

    # export to CSVs
    for period in tqdm(periods, desc="Time Periods Loop ", unit="Period"):
        edge_cube_extractor.export_mat_2_csv_via_omx(
            CUBE_EXE, OUT_PATH / f"PT_{period}.MAT", OUT_PATH, f"{period}", f"{period}"
        )
        LOG.info(f"{period} NoRMS matrices exported to CSVs")

    LOG.info("#" * 80)
    LOG.info("Process Finished Successfully")
    LOG.info("#" * 80)


def main():
    """Main Function."""
    # Set up a logger to capture all log outputs and warnings
    nd_log.get_logger(
        logger_name=nd_log.get_package_logger_name(),
        log_file_path=os.path.join(OUT_PATH, LOG_FILE),
        instantiate_msg="Export NoRMS Base Demand",
        log_version=True,
    )
    nd_log.capture_warnings(file_handler_args=dict(log_file=OUT_PATH / LOG_FILE))
    run_extractor()


if __name__ == "__main__":
    main()
