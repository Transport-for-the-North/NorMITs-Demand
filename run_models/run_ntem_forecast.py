# -*- coding: utf-8 -*-
"""
    Module for running the NTEM forecast.
"""

##### IMPORTS #####
# Standard imports
from pathlib import Path

# Third party imports

# Local imports
from normits_demand.models import ntem_forecast
from normits_demand import efs_constants as efs_consts
from normits_demand import logging as nd_log

##### CONSTANTS #####
LOG_FILE = "NTEM_forecast.log"
LOG = nd_log.get_logger(
    nd_log.get_package_logger_name() + ".run_models.run_ntem_forecast"
)
IMPORT_PATH = Path(r"I:\NorMITs Demand\import")
MODEL_NAME = "noham"

##### CLASSES #####


##### FUNCTIONS #####
def main():
    tempro_data = ntem_forecast.TEMProData(
        [efs_consts.BASE_YEAR] + efs_consts.FUTURE_YEARS
    )
    print(tempro_data.data.head())
    tempro_data.data.info()
    
    future_tempro = ntem_forecast.grow_tempro_data(tempro_data)
    future_tempro.save(Path(r"C:\WSP_Projects\TfN Secondment\NorMITs-Demand\Test Outputs\TEMProForecasts"))

    ntem_inputs = ntem_forecast.NTEMImportMatrices(
        IMPORT_PATH,
        efs_consts.BASE_YEAR,
        MODEL_NAME,
    )
    ntem_inputs.hb_paths
    ntem_inputs.nhb_paths


##### MAIN #####
if __name__ == '__main__':
    # Add log file output to main package logger
    nd_log.get_logger(
        nd_log.get_package_logger_name(), LOG_FILE, "Running NTEM forecast"
    )
    main()
