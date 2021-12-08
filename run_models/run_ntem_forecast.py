# -*- coding: utf-8 -*-
"""
    Module for running the NTEM forecast.
"""

##### IMPORTS #####
# Standard imports

# Third party imports

# Local imports
from normits_demand.models import ntem_forecast
from normits_demand import efs_constants as efs_consts
from normits_demand import logging as nd_log

##### CONSTANTS #####
LOG_FILE = "NTEM_forecast.log"
LOG = nd_log.get_logger(nd_log.get_package_logger_name() + ".run_models.run_ntem_forecast")

##### CLASSES #####


##### FUNCTIONS #####
def main():
    tempro_data = ntem_forecast.TEMProData(
        [efs_consts.BASE_YEAR] + efs_consts.FUTURE_YEARS
    )
    import io
    text = io.StringIO()
    tempro_data.data.info(buf=text)
    tempro_data.get(purposes=[1, 2, 3], modes=[1],
                    time_periods=[2]).info(buf=text)
    LOG.info("Tempro data\n" + text.getvalue())


##### MAIN #####
if __name__ == '__main__':
    # Add log file output to main package logger
    nd_log.get_logger(nd_log.get_package_logger_name(), LOG_FILE, "Running NTEM forecast")
    main()
