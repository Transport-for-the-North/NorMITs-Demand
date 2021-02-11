# -*- coding: utf-8 -*-
"""
Created on: Wed 10 15:28:32 2020
Updated on:

Original author: Ben Taylor
Last update made by:
Other updates made by:

File purpose:
Defines automatic audits of EFS outputs to NTEM data and returning reports
"""
# built-ins
import os
import pathlib

from typing import Union

# 3rd party

# Local imports
import normits_demand as nd
from normits_demand.utils import general as du


class EfsAudits:
    # TODO(Ben Taylor): Write EfsAudits docs

    _demand_dir_name = nd.ExternalForecastSystem.out_dir

    def __init__(self,
                 import_home: Union[pathlib.Path, str],
                 export_home: Union[pathlib.Path, str],
                 base_year: str,
                 model_name: str,
                 iter_num: int,
                 scenario_name: str,
                 demand_version: nd.ExternalForecastSystem.__version__,
                 demand_dir_name: _demand_dir_name,
                 ):
        # Init
        if not isinstance(pathlib.Path, import_home):
            import_home = pathlib.Path(import_home)

        if not isinstance(pathlib.Path, export_home):
            export_home = pathlib.Path(export_home)

        self.iter_name = du.create_iter_name(iter_num)
        
        # build IO paths
        self.imports, self.exports, _ = du.build_io_paths(
            import_location=import_home,
            export_location=export_home,
            base_year=base_year,
            model_name=model_name,
            iter_name=self.iter_name,
            scenario_name=scenario_name,
            demand_version=demand_version,
            demand_dir_name=demand_dir_name,
        )