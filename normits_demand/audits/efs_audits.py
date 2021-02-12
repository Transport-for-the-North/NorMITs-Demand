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

from typing import List
from typing import Dict
from typing import Union
from typing import Tuple

# 3rd party
import pandas as pd

# Local imports
import normits_demand as nd
from normits_demand import efs_constants as consts

from normits_demand.utils import general as du
from normits_demand.utils import file_ops
from normits_demand.constraints import ntem_control


class EfsAudits:
    # TODO(Ben Taylor): Write EfsAudits docs

    ntem_control_cols = ['p', 'm']

    def __init__(self,
                 import_home: Union[pathlib.Path, str],
                 export_home: Union[pathlib.Path, str],
                 model_name: str,
                 iter_num: int,
                 scenario_name: str,
                 years_needed: List[str],
                 demand_version: str = nd.ExternalForecastSystem.__version__,
                 demand_dir_name: str = nd.ExternalForecastSystem.out_dir,
                 synth_zoning_system: str = 'msoa',
                 ):
        # Init
        if not isinstance(import_home, pathlib.Path):
            import_home = pathlib.Path(import_home)

        if not isinstance(export_home, pathlib.Path):
            export_home = pathlib.Path(export_home)

        # Validate inputs
        self.model_name = du.validate_model_name(model_name)
        self.scenario_name = du.validate_scenario_name(scenario_name)
        self.synth_zoning_system = du.validate_zoning_system(synth_zoning_system)

        self.iter_name = du.create_iter_name(iter_num)
        self.years_needed = years_needed
        
        # build IO paths
        self.efs_imports, self.efs_exports, _ = du.build_efs_io_paths(
            import_location=import_home,
            export_location=export_home,
            model_name=model_name,
            iter_name=self.iter_name,
            scenario_name=scenario_name,
            demand_version=demand_version,
            demand_dir_name=demand_dir_name
        )

        self.synth_zone_name = self.synth_zoning_system
        self.model_zone_name = self.model_name

        self.imports, self.exports = self._build_io_paths()

    def _build_io_paths(self,
                        ntem_totals_path: pathlib.Path = None,
                        zone_conversion_path: pathlib.Path = None,
                        ) -> Tuple[Dict[str, Union[str, Dict[str, str]]],
                                   Dict[str, str]]:
        """
        Returns dictionaries of IO paths for this class
        """
        # Set up any paths that haven't been already
        if ntem_totals_path is None:
            ntem_totals_path = self.efs_imports['ntem_control']

        if zone_conversion_path is None:
            zone_conversion_path = self.efs_imports['zone_translation']['no_overlap']

        # ## BUILD THE IMPORTS ## #
        # Base vector imports
        hb_p_fname = consts.PRODS_FNAME % (self.synth_zone_name, 'hb')
        nhb_p_fname = consts.PRODS_FNAME % (self.synth_zone_name, 'nhb')
        hb_a_fname = consts.ATTRS_FNAME % (self.synth_zone_name, 'hb')
        nhb_a_fname = consts.ATTRS_FNAME % (self.synth_zone_name, 'nhb')

        base_vectors = {
            'hb_p': os.path.join(self.efs_exports['productions'], hb_p_fname),
            'nhb_p': os.path.join(self.efs_exports['productions'], nhb_p_fname),
            'hb_a': os.path.join(self.efs_exports['attractions'], hb_a_fname),
            'nhb_a': os.path.join(self.efs_exports['attractions'], nhb_a_fname),
        }

        # post exceptional growth imports
        hb_p_fname = consts.PRODS_FNAME % (self.synth_zone_name, 'hb_exc')
        hb_a_fname = consts.ATTRS_FNAME % (self.synth_zone_name, 'hb_exc')

        eg_vectors = {
            'hb_p': os.path.join(self.efs_exports['productions'], hb_p_fname),
            'hb_a': os.path.join(self.efs_exports['attractions'], hb_a_fname),
        }

        # matrix imports
        matrices = {
            'pa_24': self.efs_exports['pa_24'],
            'pa': self.efs_exports['pa'],
            'od': self.efs_exports['od'],
        }

        # ntem control imports
        ntem_imports = {
            'totals': ntem_totals_path,
            'msoa_to_lad': os.path.join(zone_conversion_path, 'lad_to_msoa.csv'),
            'model_to_lad': os.path.join(zone_conversion_path, '%s_to_msoa.csv' % self.model_name),
        }

        # Finally, build the outer imports dict!
        imports = {
            'base_vectors': base_vectors,
            'eg_vectors': eg_vectors,
            'matrices': matrices,
            'ntem': ntem_imports,
        }

        # Check all the import paths exist

        exports = {}

        return imports, exports

    def compare_base_pa_vectors_to_ntem(self) -> None:
        # Make sure the files we need exist
        path_dict = self.imports['base_vectors']

        for _, path in path_dict.items():
            file_ops.check_file_exists(path)

        # Compare each path to NTEM and generate a report
        trip_origin = 'hb'
        vec_type = 'productions'

        # Read in the lad<->msoa conversion
        msoa_to_lad = pd.read_csv(self.imports['ntem']['msoa_to_lad'])
        
        # Read in the vector - we need to add a mode column if not there
        vector = pd.read_csv(path_dict['hb_p'])
        if 'm' not in list(vector):
            model_modes = consts.MODEL_MODES[self.model_name]
            if len(model_modes) > 1:
                raise ValueError(
                    "The vector loaded in from '%s' does not contain a mode "
                    "column (label: 'm'). I tried to infer the mode based "
                    "on the model_name given, but more than one mode exists! "
                    "I don't know how to proceed."
                    % (str(path_dict['hb_p']))
                )

            # Add the missing mode column in
            vector['m'] = model_modes[0]

        # See how close the vector is to NTEM in each year
        for year in self.years_needed:
            # Get the ntem control for this year
            ntem_fname = consts.NTEM_CONTROL_FNAME % ('pa', year)
            ntem_path = os.path.join(self.imports['ntem']['totals'], ntem_fname)
            ntem_totals = pd.read_csv(ntem_path)

            _, audit, adjustments, lad_totals = ntem_control.control_to_ntem(
                vector,
                ntem_totals,
                msoa_to_lad,
                group_cols=self.ntem_control_cols,
                base_value_name=year,
                ntem_value_name=vec_type,
                purpose=trip_origin
            )

            print(audit)
            print()
            print(adjustments)

