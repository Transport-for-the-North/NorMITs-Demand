# -*- coding: utf-8 -*-
"""
Created on: Wed 10 15:28:32 2020
Updated on:

Original author: Ben Taylor
Last update made by:
Other updates made by:

File purpose:
Defines automatic reports of EFS outputs to NTEM data and returning reports
"""
# built-ins
import os
import pathlib
import itertools

from typing import List
from typing import Dict
from typing import Union
from typing import Tuple

# 3rd party
import pandas as pd

from tqdm import tqdm

# Local imports
import normits_demand as nd
from normits_demand import efs_constants as consts

from normits_demand.utils import general as du
from normits_demand.utils import file_ops
from normits_demand.constraints import ntem_control


class EfsReporter:
    # TODO(Ben Taylor): Write EfsReporter docs

    ntem_control_cols = ['p', 'm']

    _vector_types = ['productions', 'attractions']
    _trip_origins = consts.VALID_TRIP_ORIGINS

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
            'hb_productions': os.path.join(self.efs_exports['productions'], hb_p_fname),
            'nhb_productions': os.path.join(self.efs_exports['productions'], nhb_p_fname),
            'hb_attractions': os.path.join(self.efs_exports['attractions'], hb_a_fname),
            'nhb_attractions': os.path.join(self.efs_exports['attractions'], nhb_a_fname),
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

        # ## BUILD THE EXPORTS ## #
        export_home = os.path.join(self.efs_exports['reports'], 'EFS Reporter')
        exports = {
            'home': export_home
        }

        # Create any paths that don't already exist
        for _, path in exports.items():
            du.create_folder(path, chDir=False)

        return imports, exports

    def compare_base_pa_vectors_to_ntem(self) -> pd.DataFrame:
        """
        Generates a report of the base P/A Vectors to NTEM data

        Returns
        -------
        report:
            A copy of the report comparing the base vectors to NTEM
        """
        # Init
        out_col_order = ['Vector Type', 'Trip Origin', 'Year']
        out_col_order += ['NTEM', 'Achieved', 'NTEM - Achieved', '% Difference']

        # Make sure the files we need exist
        path_dict = self.imports['base_vectors']

        for _, path in path_dict.items():
            file_ops.check_file_exists(path)

        # Read in the lad<->msoa conversion
        msoa_to_lad = pd.read_csv(self.imports['ntem']['msoa_to_lad'])

        # Compare every base year vector to NTEM and create a report
        report_ph = list()
        base_vector_iterator = itertools.product(self._vector_types, self._trip_origins)
        for vector_type, trip_origin in base_vector_iterator:
            # Read in the correct vector
            vector_name = '%s_%s' % (trip_origin, vector_type)
            vector = pd.read_csv(path_dict[vector_name])

            # We need to add a mode column if not there
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

            report_ph.append(compare_vector_to_ntem(
                vector=vector,
                compare_cols=self.years_needed,
                compare_cols_name='Year',
                zone_to_lad=msoa_to_lad,
                ntem_totals_dir=self.imports['ntem']['totals'],
                vector_type=vector_type,
                trip_origin=trip_origin,
                constraint_cols=self.ntem_control_cols,
            ))

        # Convert to a dataframe for output
        report = pd.concat(report_ph)

        # Write the report to disk
        fname = "base_vector_report.csv"
        out_path = os.path.join(self.exports['home'], fname)
        report.to_csv(out_path, index=False)

        return report


# TODO: Move compare_vector_to_ntem() to a general pa_reporting module
def compare_vector_to_ntem(vector: pd.DataFrame,
                           compare_cols: List[str],
                           compare_cols_name: str,
                           zone_to_lad: pd.DataFrame,
                           ntem_totals_dir: Union[pathlib.Path, str],
                           vector_type: str,
                           trip_origin: str,
                           constraint_cols: List[str] = None,
                           compare_year: str = None,
                           ) -> pd.DataFrame:
    """
    Returns a report comparing the base P/A Vectors to NTEM data

    Parameters
    ----------
    vector:
        The vector to compare to NTEM

    compare_cols:
        The columns of vector to compare to NTEM

    compare_cols_name:
        The name to give to the compare_cols column in the report

    zone_to_lad:
        A DataFrame converting the vector zone system to LAD zone system

    ntem_totals_dir:
        The path to a directory containing the NTEM data to compare to

    vector_type:
        The type of vector. Either 'productions' or 'attractions'

    trip_origin:
        The trip origin of vector. Either 'hb' or 'nhb.

    constraint_cols:
        The columns of vector to compare to NTEM. If left as None, defaults
        to ['p', 'm']

    compare_year:
        The year to use when comparing to NTEM. If left as None, compare_cols
        is assumed to contain the years

    Returns
    -------
    report:
        A copy of the report comparing the given vector to NTEM. It will
        contain the following columns:
        ['Vector Type', 'Trip Origin', 'Year', 'NTEM', 'Achieved',
         'NTEM - Achieved', '% Difference']
    """
    # Init
    out_col_order = ['Vector Type', 'Trip Origin', 'Year']
    out_col_order += ['NTEM', 'Achieved', 'NTEM - Achieved', '% Difference']

    # If compare_year is None, assume compare_cols is years
    if compare_year is None:
        col_years = compare_cols
    else:
        col_years = [compare_year] * len(compare_cols)

    if constraint_cols is None:
        constraint_cols = ['p', 'm']

    # See how close the vector is to NTEM in each year
    report_ph = list()
    for col, year in zip(compare_cols, col_years):
        # Get the ntem control for this year
        ntem_fname = consts.NTEM_CONTROL_FNAME % ('pa', year)
        ntem_path = os.path.join(ntem_totals_dir, ntem_fname)
        ntem_totals = pd.read_csv(ntem_path)

        # Get an report of how close we are to ntem
        _, report, *_ = ntem_control.control_to_ntem(
            control_df=vector,
            ntem_totals=ntem_totals,
            zone_to_lad=zone_to_lad,
            constraint_cols=constraint_cols,
            base_value_name=col,
            ntem_value_name=vector_type,
            trip_origin=trip_origin,
            verbose=False,
        )

        # Generate the report based on the one we got back
        del report['after']
        report['NTEM'] = report.pop('target')
        report['Achieved'] = report.pop('before')
        report['NTEM - Achieved'] = report['NTEM'] - report['Achieved']
        report['% Difference'] = report['NTEM - Achieved'] / report['NTEM'] * 100

        # Add more info to the report and store in the placeholder
        report[compare_cols_name] = col
        report['Vector Type'] = vector_type
        report['Trip Origin'] = trip_origin

        report_ph.append(report)

    # Convert to a dataframe for output
    report = pd.DataFrame(report_ph)
    report = report.reindex(columns=out_col_order)

    return report
