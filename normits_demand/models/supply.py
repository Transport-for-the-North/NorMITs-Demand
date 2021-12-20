# -*- coding: utf-8 -*-
"""
Created on: 02/09/2021
Updated on:

Original author: Ben Taylor
Last update made by:
Other updates made by:

File purpose:

"""
# Built-Ins
import os
import warnings

# Third Party
import pandas as pd
import numpy as np

# Local Imports
import normits_demand as nd
from normits_demand.supply import PostGresConector
from normits_demand.utils import timing
from normits_demand.utils import file_ops
from normits_demand.utils import pandas_utils as pd_utils


class NorMITsSupply(PostGresConector):
    _log_fname = "NorMITs_Supply_log.log"
    # Constants
    _sql_col = "skim_id, scenario_id, users_id,value_id," \
               "skim_type_id,time_period,origin_zone,destination_zone,skim_value"
    _skims_index = "origin_zone"
    _skims_col = "destination_zone"
    _skims_val = "skim_value"

    def __init__(self,
                 user: str,
                 password: str,
                 server: str,
                 database: str,
                 port: str,
                 query_fname: str,
                 scenario_id: int,
                 users_id: int,
                 value_id: int,
                 skim_type_id: int,
                 time_period: str,
                 export_home: nd.PathLike,
                 ) -> None:
        """
        Sets up and validates arguments for the NorMITs Supply model.

        Parameters
        ----------
        query_fname:
        skim_type_id:
        scenario_id:
        users_id:
        value_id:
        time_period:
        """
        # Assign
        self.query_fname = query_fname
        self.skim_type_id = skim_type_id
        self.scenario_id = scenario_id
        self.users_id = users_id
        self.value_id = value_id
        self.time_period = time_period
        self.export_home = export_home

        # Connect to the sql server
        super().__init__(
            user=user,
            password=password,
            server=server,
            database=database,
            port=port,
        )
        # Create a logger
        logger_name = "%s.%s" % (nd.get_package_logger_name(), self.__class__.__name__)
        log_file_path = os.path.join(self.export_home, self._log_fname)
        self._logger = nd.get_logger(
            logger_name=logger_name,
            log_file_path=log_file_path,
            instantiate_msg="Initialised NorMITs Supply Model",
        )

    def run(self,
            od_to_pa_conversion: bool = False
            ) -> None:
        """
        Runs the NorMITs Supply Model.

        Completes the following steps:
            - Runs query on the connected database.
            - Stores the resulting skims as numpy array.
            - Optionally converts OD to PA using existing tour proportions, if
              od_to_pa_conversion is True.

        Parameters
        ----------
        od_to_pa_conversion:
            Whether to convert Origin Destinations to Production Attractions.

        Returns
        -------
        None
        """
        # Initialise timing

        start_time = timing.current_milli_time()
        self._logger.info("Starting NorMITs Supply Model")

        # Run query on database
        self._logger.info("Running query on the connected database")
        skims = self._cost_request(self.query_fname, self.skim_type_id, self.scenario_id,
                                   self.users_id, self.value_id, self.time_period)
        print(skims)

    def _cost_request(self,
                      query_fname: str,
                      skim_type_id: int,
                      scenario_id: int,
                      users_id: int,
                      value_id: int,
                      time_period: str,
                      ):
        print(time_period)
        query_name = "SELECT * FROM " + query_fname + " WHERE scenario_id = " + str(
            scenario_id) + " AND users_id = " + str(
            users_id) + " AND value_id = " + str(value_id) + " AND skim_type_id = " + str(
            skim_type_id)

        skims = self.query(query_name)

        skims = pd_utils.long_df_to_wide_ndarray(skims, self._skims_index, self._skims_col, self._skims_val)
        return skims
