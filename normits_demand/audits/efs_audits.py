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
from normits_demand import efs_constants as consts

from normits_demand.utils import general as du
from normits_demand.utils import file_ops


class EfsAudits:
    # TODO(Ben Taylor): Write EfsAudits docs

    def __init__(self,
                 import_home: Union[pathlib.Path, str],
                 export_home: Union[pathlib.Path, str],
                 model_name: str,
                 iter_num: int,
                 scenario_name: str,
                 demand_version: str = nd.ExternalForecastSystem.__version__,
                 demand_dir_name: str = nd.ExternalForecastSystem.out_dir,
                 synth_zoning_system: str = 'msoa',
                 ):
        # Init
        if not isinstance(import_home, pathlib.Path):
            import_home = pathlib.Path(import_home)

        if not isinstance(export_home, pathlib.Path):
            export_home = pathlib.Path(export_home)

        self.iter_name = du.create_iter_name(iter_num)
        
        # build IO paths
        self.imports, self.exports, _ = du.build_io_paths(
            import_location=import_home,
            export_location=export_home,
            model_name=model_name,
            iter_name=self.iter_name,
            scenario_name=scenario_name,
            demand_version=demand_version,
            demand_dir_name=demand_dir_name,
        )

        # Build paths to the production and attraction vectors
        hb_p_fname = consts.PRODS_FNAME % (synth_zoning_system, 'hb')
        nhb_p_fname = consts.PRODS_FNAME % (synth_zoning_system, 'nhb')
        hb_a_fname = consts.ATTRS_FNAME % (synth_zoning_system, 'hb')
        nhb_a_fname = consts.ATTRS_FNAME % (synth_zoning_system, 'nhb')

        self.base_hb_p_path = os.path.join(self.imports['productions'], hb_p_fname)
        self.base_nhb_p_path = os.path.join(self.imports['productions'], nhb_p_fname)
        self.base_hb_a_path = os.path.join(self.imports['attractions'], hb_a_fname)
        self.base_nhb_a_path = os.path.join(self.imports['attractions'], nhb_a_fname)

        # Build paths to post-EG production and attraction vectors
        hb_p_fname = consts.PRODS_FNAME % (synth_zoning_system, 'hb_exc')
        hb_a_fname = consts.ATTRS_FNAME % (synth_zoning_system, 'hb_exc')

        self.post_eg_hb_p_path = os.path.join(self.imports['productions'], hb_p_fname)
        self.post_eg_hb_a_path = os.path.join(self.imports['attractions'], hb_p_fname)

    def compare_base_pa_vectors_to_ntem(self) -> None:
        # Make sure the files we need exist
        vector_paths = [
            self.base_hb_p_path,
            self.base_nhb_p_path,
            self.base_hb_a_path,
            self.base_nhb_a_path,
        ]

        for path in vector_paths:
            file_ops.check_file_exists(path)

        # Compare each path to NTEM and generate a report
        trip_origin = 'hb'
        vec_type = 'productions'

        

