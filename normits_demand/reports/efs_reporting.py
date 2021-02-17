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

from typing import Any
from typing import List
from typing import Dict
from typing import Union
from typing import Tuple

# 3rd party
import pandas as pd

# Local imports
import normits_demand as nd
from normits_demand import constants as consts

from normits_demand.validation import checks
from normits_demand.constraints import ntem_control
from normits_demand.matrices import matrix_processing as mat_p

from normits_demand.utils import file_ops
from normits_demand.utils import general as du


class EfsReporter:
    # TODO(Ben Taylor): Write EfsReporter docs

    # BACKLOG: Generate reports for future year mode shares
    #  labels: EFS

    ntem_control_dtypes = {'p': int, 'm': int}
    ntem_control_cols = list(ntem_control_dtypes.keys())

    _vector_types = ['productions', 'attractions']
    _trip_origins = consts.TRIP_ORIGINS
    _zone_col_base = '%s_zone_id'

    _ntem_report_cols = [
        'Vector Type',
        'Trip Origin',
        'Year',
        'NTEM',
        'Achieved',
        'NTEM - Achieved',
        '% Difference'
    ]

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

        # Load in the data we'll need later
        self._load_ie_zonal_info()

        self.reporting_subsets = {
            'internal': self.model_internal_zones,
            'external': self.model_external_zones,
        }

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

        # Translated base vector imports
        hb_p_fname = consts.PRODS_FNAME % (self.model_zone_name, 'hb')
        nhb_p_fname = consts.PRODS_FNAME % (self.model_zone_name, 'nhb')
        hb_a_fname = consts.ATTRS_FNAME % (self.model_zone_name, 'hb')
        nhb_a_fname = consts.ATTRS_FNAME % (self.model_zone_name, 'nhb')

        translated_base_vectors = {
            'hb_productions': os.path.join(self.efs_exports['productions'], hb_p_fname),
            'nhb_productions': os.path.join(self.efs_exports['productions'], nhb_p_fname),
            'hb_attractions': os.path.join(self.efs_exports['attractions'], hb_a_fname),
            'nhb_attractions': os.path.join(self.efs_exports['attractions'], nhb_a_fname),
        }

        # TODO: Add in NHB exceptional growth when we have the files
        # post exceptional growth imports
        hb_p_fname = consts.PRODS_FNAME % (self.model_zone_name, 'hb_exc')
        hb_a_fname = consts.ATTRS_FNAME % (self.model_zone_name, 'hb_exc')

        eg_vectors = {
            'hb_productions': os.path.join(self.efs_exports['productions'], hb_p_fname),
            'hb_attractions': os.path.join(self.efs_exports['attractions'], hb_a_fname),
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
            'model_to_lad': os.path.join(zone_conversion_path, '%s_to_lad.csv' % self.model_name),
        }

        # Finally, build the outer imports dict!
        imports = {
            'base_vectors': base_vectors,
            'translated_base_vectors': translated_base_vectors,
            'eg_vectors': eg_vectors,
            'matrices': matrices,
            'ntem': ntem_imports,
        }

        # ## BUILD THE EXPORTS ## #
        export_home = os.path.join(self.efs_exports['reports'], 'EFS Reporter')

        # Build the cache paths
        cache_home = os.path.join(export_home, 'cache')
        cache_paths = {
            'pa_24': os.path.join(cache_home, 'pa_24'),
            'pa': os.path.join(cache_home, 'pa'),
            'od': os.path.join(cache_home, 'od'),
        }
        for _, path in cache_paths.items():
            du.create_folder(path, chDir=False)

        # Finally, build the outer exports dict!
        exports = {
            'home': export_home,
            'cache': cache_paths,
        }

        # Create any paths that don't already exist
        for _, path in exports.items():
            if isinstance(path, dict):
                continue
            du.create_folder(path, chDir=False)

        return imports, exports

    def _load_ie_zonal_info(self):
        """
        Populates self.model_internal_zones and self.model_external_zones
        """
        # Init
        zone_col = self._zone_col_base % self.model_name

        int_fname = consts.INTERNAL_AREA % self.model_name
        int_path = os.path.join(self.efs_imports['model_schema'], int_fname)
        self.model_internal_zones = pd.read_csv(int_path)[zone_col].tolist()

        ext_fname = consts.EXTERNAL_AREA % self.model_name
        ext_path = os.path.join(self.efs_imports['model_schema'], ext_fname)
        self.model_external_zones = pd.read_csv(ext_path)[zone_col].tolist()

    def _compare_vector_to_ntem(self,
                                vector: pd.DataFrame,
                                zone_to_lad: pd.DataFrame,
                                vector_type: str,
                                matrix_format: str,
                                trip_origin: str,
                                base_zone_name: str,
                                report_subsets: Dict[str, Any] = None,
                                subset_col_name: str = 'Subset',
                                ) -> pd.DataFrame:
        """
        An internal wrapper around compare_vector_to_ntem()
        """
        # We need to add a mode column if not there
        if 'm' not in list(vector):
            model_modes = consts.MODEL_MODES[self.model_name]
            if len(model_modes) > 1:
                raise ValueError(
                    "The given vector does not contain a mode "
                    "column (label: 'm'). I tried to infer the mode based "
                    "on the model_name given, but more than one mode exists! "
                    "I don't know how to proceed."
                )

            # Add the missing mode column in
            vector['m'] = model_modes[0]

        return compare_vector_to_ntem(
            vector=vector,
            compare_cols=self.years_needed,
            compare_cols_name='Year',
            zone_to_lad=zone_to_lad,
            ntem_totals_dir=self.imports['ntem']['totals'],
            vector_type=vector_type,
            matrix_format=matrix_format,
            trip_origin=trip_origin,
            base_zone_name=base_zone_name,
            constraint_cols=self.ntem_control_cols,
            constraint_dtypes=self.ntem_control_dtypes,
            report_subsets=report_subsets,
            subset_col_name=subset_col_name,
        )

    def _generate_vector_report(self,
                                vector_dict: Dict[str, pd.DataFrame],
                                vector_zone_col: str,
                                zone_to_lad: pd.DataFrame,
                                matrix_format: str,
                                output_path: pathlib.Path,
                                vector_types: List[str] = None,
                                trip_origins: List[str] = None,
                                report_subsets: Dict[str, Any] = None,
                                subset_col_name: str = 'Subset',
                                ) -> pd.DataFrame:
        """
        Generates a report comparing the given vectors to NTEM

        A copy of the report is also written to disk at output_path

        Parameters
        ----------
        vector_dict:
            A dictionary of keys to vector pd.DataFrames where the keys are
            underscore separated vector_types and trip_origins

        vector_zone_col:
            name of zoning column used in vector_dict. Same column name
            should be used in zone_to_lad for translation.

        zone_to_lad:
            DF of translation between control_df zone system and LAD


        matrix_format:
            The format of the vectors being passed in. Either 'pa' or 'od'

        output_path:
            A full path, including filename, of where to write the produced
            report to disk

        vector_types:
            The vector types to look for in vector_dict. If left as None,
            defaults to self._vector_types.

        trip_origins:
            The trip origins to look for in vector_dict. If left as None,
            defaults to self._trip_origins

        Returns
        -------
        report:
            A copy of the generated report inside a pandas dataframe.
        """
        # Init
        vector_types = self._vector_types if vector_types is None else vector_types
        trip_origins = self._trip_origins if trip_origins is None else trip_origins

        # Compare every base year vector to NTEM and create a report
        report_ph = list()
        base_vector_iterator = itertools.product(vector_types, trip_origins)
        for vector_type, trip_origin in base_vector_iterator:
            # Read in the correct vector
            vector_name = '%s_%s' % (trip_origin, vector_type)

            report_ph.append(self._compare_vector_to_ntem(
                vector=vector_dict[vector_name],
                zone_to_lad=zone_to_lad,
                vector_type=vector_type,
                matrix_format=matrix_format,
                trip_origin=trip_origin,
                base_zone_name=vector_zone_col,
                report_subsets=report_subsets,
                subset_col_name=subset_col_name,
            ))

        # Convert to a dataframe for output
        report = pd.concat(report_ph)

        # Write the report to disk
        report.to_csv(output_path, index=False)

        return report

    def compare_base_pa_vectors_to_ntem(self) -> pd.DataFrame:
        """
        Generates a report of the base P/A Vectors to NTEM data

        Returns
        -------
        report:
            A copy of the report comparing the base vectors to NTEM
        """
        # Init
        matrix_format = 'pa'
        output_fname = "base_vector_report.csv"

        # Make sure the files we need exist
        path_dict = self.imports['base_vectors']

        for _, path in path_dict.items():
            file_ops.check_file_exists(path)

        # Load in the vectors
        vector_dict = {k: pd.read_csv(v) for k, v in path_dict.items()}

        return self._generate_vector_report(
            vector_dict=vector_dict,
            vector_zone_col=self._zone_col_base % self.synth_zone_name,
            zone_to_lad=pd.read_csv(self.imports['ntem']['msoa_to_lad']),
            matrix_format=matrix_format,
            output_path=os.path.join(self.exports['home'], output_fname),
            vector_types=self._vector_types,
            trip_origins=self._trip_origins,
        )

    def compare_translated_base_pa_vectors_to_ntem(self) -> pd.DataFrame:
        """
        Generates a report of the base P/A Vectors to NTEM data

        Returns
        -------
        report:
            A copy of the report comparing the base vectors to NTEM
        """
        # Init
        matrix_format = 'pa'
        output_fname = "translated_base_vector_report.csv"

        # Make sure the files we need exist
        path_dict = self.imports['translated_base_vectors']

        for _, path in path_dict.items():
            file_ops.check_file_exists(path)

        # Load in the vectors
        vector_dict = {k: pd.read_csv(v) for k, v in path_dict.items()}

        return self._generate_vector_report(
            vector_dict=vector_dict,
            vector_zone_col=self._zone_col_base % self.model_zone_name,
            zone_to_lad=pd.read_csv(self.imports['ntem']['model_to_lad']),
            matrix_format=matrix_format,
            output_path=os.path.join(self.exports['home'], output_fname),
            vector_types=self._vector_types,
            trip_origins=self._trip_origins,
            report_subsets=self.reporting_subsets,
        )

    def compare_eg_pa_vectors_to_ntem(self) -> pd.DataFrame:
        """
        Generates a report of the post exceptional growth P/A vectors
        compared to NTEM data

        Returns
        -------
        report:
            A copy of the report comparing the post exceptional growth
            P/A vectors to NTEM.
        """
        # Init
        matrix_format = 'pa'
        output_fname = "exceptional_growth_vector_report.csv"

        # Make sure the files we need exist
        path_dict = self.imports['eg_vectors']

        for _, path in path_dict.items():
            file_ops.check_file_exists(path)

        # Load in the vectors
        vector_dict = {k: pd.read_csv(v) for k, v in path_dict.items()}

        return self._generate_vector_report(
            vector_dict=vector_dict,
            vector_zone_col=self._zone_col_base % self.model_zone_name,
            zone_to_lad=pd.read_csv(self.imports['ntem']['model_to_lad']),
            matrix_format=matrix_format,
            output_path=os.path.join(self.exports['home'], output_fname),
            vector_types=self._vector_types,
            trip_origins=['hb'],
            report_subsets=self.reporting_subsets,
        )

    def compare_pa_matrices_to_ntem(self) -> pd.DataFrame:
        """
        Generates a report of the PA matrices compared to NTEM data

        Returns
        -------
        report:
            A copy of the report comparing the PA matrices to NTEM
        """
        # Init
        matrix_format = 'pa'
        output_fname = "base_24hr_pa_matrices_report.csv"
        vector_order = [
            'hb_productions',
            'nhb_productions',
            'hb_attractions',
            'nhb_attractions',
        ]

        # Convert matrices into vector
        vectors = mat_p.maybe_convert_matrices_to_vector(
            mat_import_dir=self.imports['matrices']['pa_24'],
            years_needed=self.years_needed,
            cache_path=self.exports['cache']['pa_24'],
            matrix_format=matrix_format,
        )

        # Assign to a dictionary for accessing
        vector_dict = {name: vec for name, vec in zip(vector_order, vectors)}

        # ## GENERATE THE REPORT ## #
        return self._generate_vector_report(
            vector_dict=vector_dict,
            vector_zone_col=self._zone_col_base % self.model_zone_name,
            zone_to_lad=pd.read_csv(self.imports['ntem']['model_to_lad']),
            matrix_format=matrix_format,
            output_path=os.path.join(self.exports['home'], output_fname),
            vector_types=self._vector_types,
            trip_origins=self._trip_origins,
            report_subsets=self.reporting_subsets,
        )


# TODO: Move compare_vector_to_ntem() to a general pa_reporting module
def compare_vector_to_ntem(vector: pd.DataFrame,
                           compare_cols: List[str],
                           compare_cols_name: str,
                           zone_to_lad: pd.DataFrame,
                           ntem_totals_dir: Union[pathlib.Path, str],
                           vector_type: str,
                           trip_origin: str,
                           matrix_format: str,
                           base_zone_name: str,
                           constraint_cols: List[str] = None,
                           constraint_dtypes: Dict[str, Any] = None,
                           compare_year: str = None,
                           report_subsets: Dict[str, Any] = None,
                           subset_col_name: str = 'Subset',
                           ) -> pd.DataFrame:
    """
    Returns a report comparing the given vector to NTEM data

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
        The type of vector. Either 'productions', 'attractions', 'origin', 'destination'.

    trip_origin:
        The trip origin of vector. Either 'hb' or 'nhb.

    matrix_format:
        The format of the vector being compared. Either 'pa' or 'od'.

    base_zone_name:
        The name of column containing the zone system data for the vector and
        zone_to_lad given.

    constraint_cols:
        The columns of vector to compare to NTEM. If left as None, defaults
        to ['p', 'm']

    compare_year:
        The year to use when comparing to NTEM. If left as None, compare_cols
        is assumed to contain the years

    report_subsets:


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
    if report_subsets is not None:
        out_col_order += [subset_col_name]
    out_col_order += ['NTEM', 'Achieved', 'Achieved - NTEM', '% Difference']

    # validation
    vector_type = checks.validate_vector_type(vector_type)
    trip_origin = checks.validate_trip_origin(trip_origin)
    matrix_format = checks.validate_matrix_format(matrix_format)

    # If compare_year is None, assume compare_cols is years
    if compare_year is None:
        col_years = compare_cols
    else:
        col_years = [compare_year] * len(compare_cols)

    if constraint_cols is None:
        constraint_cols = ['p', 'm']

    segment_cols = du.list_safe_remove(list(vector), col_years)

    # We'll need this to build reports in the loop below
    def build_report(rep):
        if 'after' in rep:
            del rep['after']
        rep['NTEM'] = rep.pop('target')
        rep['Achieved'] = rep.pop('before')
        rep['Achieved - NTEM'] = rep['Achieved'] - rep['NTEM']
        rep['% Difference'] = rep['Achieved - NTEM'] / rep['NTEM'] * 100

        # Add more info to the report and store in the placeholder
        rep[compare_cols_name] = col
        rep['Vector Type'] = vector_type
        rep['Trip Origin'] = trip_origin

        return rep

    # See how close the vector is to NTEM in each year
    report_ph = list()
    for col, year in zip(compare_cols, col_years):
        # Get the ntem control for this year
        ntem_fname = consts.NTEM_CONTROL_FNAME % (matrix_format, year)
        ntem_path = os.path.join(ntem_totals_dir, ntem_fname)
        ntem_totals = pd.read_csv(ntem_path)

        # Get an report of how close we are to ntem
        adj_vector, report, *_ = ntem_control.control_to_ntem(
            control_df=vector,
            ntem_totals=ntem_totals,
            zone_to_lad=zone_to_lad,
            constraint_cols=constraint_cols,
            constraint_dtypes=constraint_dtypes,
            base_value_name=col,
            ntem_value_name=vector_type,
            trip_origin=trip_origin,
            base_zone_name=base_zone_name,
            group_cols=segment_cols,
            verbose=False,
        )

        # If we aren't dong subsets, just report in the whole thing
        if report_subsets is None:
            report_ph.append(build_report(report))
            continue

        # Build a report for each subset
        for name, vals in report_subsets.items():
            # Grab the subsets for this report
            needed_cols = segment_cols + [col]
            mask = vector[base_zone_name].isin(vals)

            vector_subset = vector[mask].reindex(columns=needed_cols)
            adj_vector_subset = adj_vector[mask].reindex(columns=needed_cols)

            # Build a replicant of the control report
            report = {
                'before': vector_subset[col].sum(),
                'target': adj_vector_subset[col].sum(),
            }

            # Create report
            report = build_report(report)
            report[subset_col_name] = name
            report_ph.append(report)

    # Convert to a dataframe for output
    report = pd.DataFrame(report_ph)
    report = report.reindex(columns=out_col_order)

    return report
