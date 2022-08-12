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
from typing import Dict

import pandas as pd
import numpy as np

# Local Imports
import normits_demand as nd
from normits_demand.supply import PostGresConector
from normits_demand.utils import timing
from normits_demand.utils import file_ops
from normits_demand.utils import pandas_utils as pd_utils
from normits_demand.matrices import utils as mat_utils
from normits_demand.core import zoning


class NorMITsSupply(PostGresConector):
    # Constants
    _sql_col = "skim_id, scenario_id, users_id,value_id," \
               "skim_type_id,time_period,origin_zone,destination_zone,skim_value"
    _skims_index = "origin_zone"
    _skims_col = "destination_zone"
    _skims_val = "skim_value"
    _zone_id = {1: "Norms",
                2: "Noham"}
    _time_period_od = ['TS1', 'TS2', 'TS3', 'TS4']
    _time_period_pa = [1, 2, 3, 4]

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
        user:
            Username of database user.

        password:
            Password of database user.

        server:
            Server address.

        database:
            Postgres database name.

        port:
            Port number for the database.

        query_fname:
            File name of the table in which the query has to be run.

        skim_type_id:
            Type of skim required. Distance, time, toll etc. Ranges from 1 to 6.

        scenario_id:
            Type of scenario needed. NoRMS, NoHAM or TRACC. Ranges from 1 to 3.

        users_id:
            Type of user class required. Car Business, Car commute,
            LGV, HGV etc. Ranges from 1 to 19.

        value_id:
            Cost value required. Ranges from 1 to 67.

        time_period:
            Time period needed. TS1, TS2, TS3, TS4, allday and AM.
        """
        # Assign
        self.query_fname = query_fname
        self.skim_type_id = skim_type_id
        self.scenario_id = scenario_id
        self.users_id = users_id
        self.value_id = value_id
        self.time_period = time_period
        self.export_home = export_home
        self.zone_list = zoning._get_zones(self._zone_id[self.scenario_id])[0]
        self.col_index = self.zone_list.tolist()

        # Connect to the sql server
        super().__init__(
            user=user,
            password=password,
            server=server,
            database=database,
            port=port,
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
        print("Starting NorMITs Supply Model")

        # Run query on database
        print("Running query on the connected database")
        skims = self._cost_request()
        print(skims)
        print(skims.sum())
        skims1 = pd.DataFrame(skims)
        # skims1.to_csv(r"I:\NorMITs Demand\import\noham\post_me_tour_proportions\fh_th_factors\trial1.csv", index=False)
        skims1.to_csv(r"E:\supply\trial1.csv", index=False)

    def _cost_request(self):
        """
        Runs the cost query and return the output as numpy.

        Parameters
        ----------
        None.

        Returns
        -------
        skims:
        Returns skims based on the query as numpy

        """

        query_name = "SELECT * FROM " + self.query_fname + " WHERE scenario_id = " + str(
            self.scenario_id) + " AND users_id = " + str(
            self.users_id) + " AND value_id = " + str(self.value_id) + " AND skim_type_id = " + str(
            self.skim_type_id)

        skims = self.query(query_name)
        print(skims)
        exit()

        od_matrix = {}
        j = 1
        for i in self._time_period_od:
            new_df = skims[skims['time_period'] == i]
            od_matrix[j] = pd_utils.long_df_to_wide_ndarray(new_df, self._skims_index, self._skims_col, self._skims_val,
                                                     self.col_index, self.col_index)
            j += 1
        return od_matrix

    def od_pa_via_tour_props(self,
                             od_matrix: Dict,
                             fh_factor_dict: Dict,
                             n_od_vals: int,
                             tp_needed: list,

                             ):
        mat_utils.check_fh_th_factors(
            factor_dict=fh_factor_dict,
            tp_needed=tp_needed,
            n_row_col=n_od_vals,
        )
        pa = {}
        pa_total = np.zeros(n_od_vals, n_od_vals)
        for tp in tp_needed:
            pa[tp] = od_matrix[tp] * fh_factor_dict[tp]
            pa_total += pa[tp]
