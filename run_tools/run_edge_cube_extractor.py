# -*- coding: utf-8 -*-
"""Run Script for the EDGE CUBE extractor"""
# ## IMPORTS ## #
# Standard imports
import os
import sys
import shutil

# Third party imports
import pandas as pd
from tqdm import tqdm

# Local imports
sys.path.append(".")
sys.path.append("..")
# pylint: disable=import-error,wrong-import-position
from normits_demand import logging as nd_log
from normits_demand.tools import edge_cube_extractor
# pylint: enable=import-error,wrong-import-position

# ## USER INPUTS ## #
# cube catalogue setup
CUBE_EXE = "C:/Program Files/Citilabs/CubeVoyager/VOYAGER.EXE"
CUBE_CAT_PATH = r"E:\Norms\NorTMS_T3_Model_v8.17"
CAT_RUN_DIR = "Runs"
CUBE_RUN_ID = "ILP_2018"

# Input files
TLC_OVERWRITE_PATH = r"I:\NorMITs Demand\import\edge_replicant\apply_growth\pre-processing\TLC_Overwrite.csv"

# Output location
OUT_PATH = r"E:\Norms\Edge growth\output"

# ## CONSTANTS ## #
# logger
LOG_FILE = "Export_BaseMatrices_Logfile.Log"
LOG = nd_log.get_logger(f"{nd_log.get_package_logger_name()}.run_edge_cube_extractor")

# Derived from inputs
CUBE_CAT_RUN_PATH = os.path.join(CUBE_CAT_PATH, CAT_RUN_DIR)

# ## CLASSES ## #


# ## FUNCTIONS ## #
def run_extractor():
    """Process Fixed objects"""
    # time periods
    periods = ["AM", "IP", "PM", "OP"]

    # copy Cube files
    for period in periods:
        # read distance matrix
        edge_cube_extractor.check_file_exists(f"{CUBE_CAT_RUN_PATH}/{CUBE_RUN_ID}/Outputs/BaseAssign/{period}_stn2stn_costs.csv")
        shutil.copy2(
            f"{CUBE_CAT_RUN_PATH}/{CUBE_RUN_ID}/Outputs/BaseAssign/{period}_stn2stn_costs.csv",
            f"{OUT_PATH}/{period}_stn2stn_costs.csv",
        )
        # read iRSj props
        edge_cube_extractor.check_file_exists(
            f"{CUBE_CAT_RUN_PATH}/{CUBE_RUN_ID}/Outputs/BaseAssign/{period}_iRSj_probabilities.h5"
        )
        shutil.copy2(
            f"{CUBE_CAT_RUN_PATH}/{CUBE_RUN_ID}/Outputs/BaseAssign/{period}_iRSj_probabilities.h5",
            f"{OUT_PATH}/{period}_iRSj_probabilities.h5",
        )

        LOG.info(f"Distance and Probability matrices for period {period} has been copied")

    # produce TLC lookup
    edge_cube_extractor.check_file_exists(TLC_OVERWRITE_PATH)
    tlc_overwrite = pd.read_csv(TLC_OVERWRITE_PATH)
    stns_tlc = edge_cube_extractor.stnzone_2_stn_tlc(
        f"{CUBE_CAT_RUN_PATH}/{CUBE_RUN_ID}/Inputs/Network/Station_Connectors.csv",
        f"{CUBE_CAT_RUN_PATH}/{CUBE_RUN_ID}/Inputs/Network/TfN_Rail_Nodes.csv",
        f"{CUBE_CAT_RUN_PATH}/{CUBE_RUN_ID}/Inputs/Network/External_Station_Nodes.csv",
        tlc_overwrite,
    )
    # write TLC Lookup
    stns_tlc.to_csv(f"{OUT_PATH}/TLCs.csv", index=False)

    # PT Demand to time periods F/T
    edge_cube_extractor.pt_demand_from_to(CUBE_EXE, CUBE_CAT_PATH, CUBE_CAT_RUN_PATH + "/" + CUBE_RUN_ID, OUT_PATH)
    LOG.info("NoRMS matrices converted to OMX successfully")

    # export to CSVs
    for period in tqdm(periods, desc="Time Periods Loop ", unit="Period"):
        edge_cube_extractor.export_mat_2_csv_via_omx(
            CUBE_EXE, OUT_PATH + f"/PT_{period}.MAT", OUT_PATH, f"{period}", f"{period}"
        )
        LOG.info(f"{period} NoRMS matrices exported to CSVs")

    LOG.info("#" * 80)
    LOG.info("Process Finished Successfully")
    LOG.info("#" * 80)


def main():
    # Set up a logger to capture all log outputs and warnings
    nd_log.get_logger(
        logger_name=nd_log.get_package_logger_name(),
        log_file_path=os.path.join(OUT_PATH, LOG_FILE),
        instantiate_msg="Running TLD Builder",
        log_version=True,
    )
    nd_log.capture_warnings(
        file_handler_args=dict(log_file=os.path.join(OUT_PATH, LOG_FILE))
    )
    run_extractor()


if __name__ == "__main__":
    main()
