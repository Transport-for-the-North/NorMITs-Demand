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
import numpy as np
import pandas as pd
from tqdm import tqdm

# Local imports
import normits_demand as nd
from normits_demand import constants as consts

from normits_demand.validation import checks
from normits_demand.constraints import ntem_control

from normits_demand.matrices import matrix_processing as mat_p
from normits_demand.matrices import utils as mat_utils

from normits_demand.utils import file_ops
from normits_demand.utils import general as du
from normits_demand.utils import utils as tms_utils

from normits_demand.reports import reports_audits as tms_reports


class EfsReporter:
    # TODO(Ben Taylor): Write EfsReporter docs

    # BACKLOG: Generate reports for future year mode shares
    #  labels: EFS

    ntem_control_dtypes = {'p': int, 'm': int}
    ntem_control_cols = list(ntem_control_dtypes.keys())

    _pa_vector_types = ['productions', 'attractions']
    _od_vector_types = ['origins', 'destinations']
    _trip_origins = consts.TRIP_ORIGINS
    _zone_col_base = '%s_zone_id'

    # trip origin, year, purpose
    _band_distances_fname = '%s_yr%s_p%s_band_distances.csv'
    _band_shares_fname = '%s_yr%s_p%s_band_shares.csv'

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
                 iter_num: str,
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
        self.base_year, self.future_years = du.split_base_future_years_str(years_needed)
        
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
        self.model_zone_col = "%s_zone_id" % model_name

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

        # BACKLOG: Make tlb paths relative in efs reporter
        #  labels: EFS, demand merge
        tlb_atl_path = r'I:\NorMITs Synthesiser\import\trip_length_bands\north\standard_segments'

        # ## BUILD THE IMPORTS ## #
        # Raw vectors
        hb_p_fname = consts.PRODS_FNAME % (self.synth_zone_name, 'raw_hb')
        nhb_p_fname = consts.PRODS_FNAME % (self.synth_zone_name, 'raw_nhb')
        hb_a_fname = consts.ATTRS_FNAME % (self.synth_zone_name, 'raw_hb')
        nhb_a_fname = consts.ATTRS_FNAME % (self.synth_zone_name, 'raw_nhb')

        raw_vectors = {
            'hb_productions': os.path.join(self.efs_exports['productions'], hb_p_fname),
            'nhb_productions': os.path.join(self.efs_exports['productions'], nhb_p_fname),
            'hb_attractions': os.path.join(self.efs_exports['attractions'], hb_a_fname),
            'nhb_attractions': os.path.join(self.efs_exports['attractions'], nhb_a_fname),
        }

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
        nhb_p_fname = consts.PRODS_FNAME % (self.model_zone_name, 'nhb_exc')
        hb_a_fname = consts.ATTRS_FNAME % (self.model_zone_name, 'hb_exc')
        nhb_a_fname = consts.ATTRS_FNAME % (self.model_zone_name, 'nhb_exc')

        eg_vectors = {
            'hb_productions': os.path.join(self.efs_exports['productions'], hb_p_fname),
            'nhb_productions': os.path.join(self.efs_exports['productions'], nhb_p_fname),
            'hb_attractions': os.path.join(self.efs_exports['attractions'], hb_a_fname),
            'nhb_attractions': os.path.join(self.efs_exports['attractions'], nhb_a_fname),
        }

        # matrix imports
        matrices = {
            'pa_24': self.efs_exports['pa_24'],
            'pa_24_bespoke': self.efs_exports['pa_24_bespoke'],
            'pa': self.efs_exports['pa'],
            'od': self.efs_exports['od'],
            'post_me': self.efs_imports['decomp_post_me'],
        }

        # ntem control imports
        ntem_imports = {
            'totals': ntem_totals_path,
            'msoa_to_lad': os.path.join(zone_conversion_path, 'lad_to_msoa.csv'),
            'model_to_lad': os.path.join(zone_conversion_path, '%s_to_lad.csv' % self.model_name),
        }

        # Finally, build the outer imports dict!
        imports = {
            'post_me_pa': self.efs_imports['decomp_post_me'],
            'tlb': tlb_atl_path,
            'raw_vectors': raw_vectors,
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
            'pa_24_bespoke': os.path.join(cache_home, 'bespoke_pa_24'),
            'post_me': os.path.join(cache_home, 'post_me'),
            'pa': os.path.join(cache_home, 'pa'),
            'od': os.path.join(cache_home, 'od'),
            'post_me_tlb': os.path.join(cache_home, 'post_me_tlb'),
            'pa_24_tlb': os.path.join(cache_home, 'pa_24_tlb'),
        }
        for _, path in cache_paths.items():
            du.create_folder(path, chDir=False, verbose=False)

        # Build the tlb paths
        tlb_home = os.path.join(export_home, 'tlb_mats')
        tlb_paths = {
            'post_me': os.path.join(tlb_home, 'post_me'),
            'pa_24': os.path.join(tlb_home, 'pa_24'),
        }
        for _, path in tlb_paths.items():
            du.create_folder(path, chDir=False, verbose=False)

        # Finally, build the outer exports dict!
        exports = {
            'home': export_home,
            'modal': os.path.join(export_home, 'modal'),
            'tlb': tlb_paths,
            'cache': cache_paths,
        }

        # Create any paths that don't already exist
        for _, path in exports.items():
            if isinstance(path, dict):
                continue
            du.create_folder(path, chDir=False, verbose=False)

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
                                comparison_vector_dict: Dict[str, pd.DataFrame] = None,
                                comparison_vector_name: str = 'comparison',
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

        comparison_vector_dict:
            A dictionary of vectors to compare vector_dict vectors to instead
            of NTEM. If left as None, a comparison to NTEM is created.

        comparison_vector_name:
            If comparison_vector_dict is not None, this name will be used
            in the report to determine which column contains the data
            for comparison_vector_dict vectors.

        report_subsets:
            A dictionary of subset names to zonal subsets to generate reports for.
            The subset names will be placed in subset_col_name of the report.
            If left as None, no subsets are used in the report.

        subset_col_name:
            The name to give to the column that the subset names from
            report_subsets will go into.

        Returns
        -------
        report:
            A copy of the generated report inside a pandas dataframe.
        """
        # Init
        vector_types = self._pa_vector_types if vector_types is None else vector_types
        trip_origins = self._trip_origins if trip_origins is None else trip_origins

        # Validate the comparison dict
        if comparison_vector_dict is not None:
            for k in vector_dict.keys():
                if k not in comparison_vector_dict.keys():
                    raise ValueError(
                        "Cannot find key '%s' in the comparison dict to "
                        "compare the vector to." % k
                    )

        # Compare every base year vector to NTEM and create a report
        report_ph = list()
        base_vector_iterator = itertools.product(vector_types, trip_origins)
        for vector_type, trip_origin in base_vector_iterator:
            # Read in the correct vector
            vector_name = '%s_%s' % (trip_origin, vector_type)

            if comparison_vector_dict is None:
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
            else:
                raise NotImplementedError

        # Convert to a dataframe for output
        report = pd.concat(report_ph)

        # Write the report to disk
        report.to_csv(output_path, index=False)

        return report

    def _get_distance(self, p: int) -> pd.DataFrame:
        # TODO(BT): Horrible function call. Tidy up
        distance, _ = tms_utils.get_costs(
            self.efs_imports['model_home'],
            {'p': p},
            tp='24hr'
        )

        return pd.pivot(
            data=distance,
            index='p_zone',
            columns='a_zone',
            values='cost',
        )

    def _get_hb_trip_length_bands(self, p: int) -> pd.DataFrame:
        tlb = tms_utils.get_trip_length_bands(
            import_folder=self.imports['tlb'],
            calib_params={'p': p, 'm': consts.MODEL_MODES[self.model_name][0]},
            trip_origin='hb',
            segmentation=None,  # Not used!
            echo=False,
        )

        return tlb

    def run(self,
            run_raw_vector_report: bool = True,
            compare_trip_lengths: bool = True,
            ) -> None:
        """
        Runs all the report generation functions.

        Runs:
            compare_base_pa_vectors_to_ntem()
            compare_translated_base_pa_vectors_to_ntem()
            compare_eg_pa_vectors_to_ntem()
            compare_pa_matrices_to_ntem()
            compare_bespoke_pa_matrices_to_ntem()
            compare_tp_pa_matrices_to_ntem()
            compare_od_matrices_to_ntem()

        Returns
        -------
        None
        """
        if run_raw_vector_report:
            print("Generating report across all modes...")
            self.compare_raw_pa_vectors_to_ntem()

            print("Generating a report per mode...")
            self.compare_raw_pa_vectors_to_ntem_by_mode()

        print("Generating %s specific reports..." % self.model_name)
        # self.compare_base_pa_vectors_to_ntem()
        # self.compare_translated_base_pa_vectors_to_ntem()
        # self.compare_eg_pa_vectors_to_ntem()
        # self.analyse_compiled_matrices()

        if compare_trip_lengths:
            print("Generating trip length reports...")
            self.compare_trip_lengths()

        # Compare pre-furness vectors to post-ME
        # self.compare_eg_pa_vectors_to_post_me()

        # Matrix compare to NTEM
        self.compare_pa_matrices_to_ntem()
        self.compare_bespoke_pa_matrices_to_ntem()
        self.compare_tp_pa_matrices_to_ntem()
        self.compare_od_matrices_to_ntem()

    def _generate_trip_band_report_by_purpose(self,
                                              distance_dict: Dict[int, pd.DataFrame],
                                              raw_mat_import: nd.PathLike,
                                              cache_export: nd.PathLike,
                                              report_export: nd.PathLike,
                                              years_needed: List[str],
                                              trip_origin: str,
                                              matrix_format: str,
                                              mode: int,
                                              ) -> List[Dict[str, Any]]:
        # TODO(BT): Write generate_trip_band_report_by_purpose docs
        # Set up the progress bar
        pbar = tqdm(
            total=len(years_needed) * len(distance_dict.keys()),
            desc="Generating TLB reports"
        )
        avg_trip_lengths = list()
        for year in years_needed:
            for p, distance in distance_dict.items():
                # Aggregate and read in the matrix
                mat_p.aggregate_matrices(
                    import_dir=raw_mat_import,
                    export_dir=cache_export,
                    trip_origin=trip_origin,
                    matrix_format=matrix_format,
                    years_needed=[int(year)],
                    p_needed=[p],
                    m_needed=[mode],
                    compress_out=True,
                )

                # Read the matrix back in
                fname = du.get_dist_name(
                    trip_origin=trip_origin,
                    matrix_format=matrix_format,
                    year=year,
                    purpose=str(p),
                    mode=str(mode),
                    compressed=True,
                )
                path = os.path.join(cache_export, fname)
                df = file_ops.read_df(path, index_col=0)

                # filter to just the internal area
                int_mask = mat_utils.get_internal_mask(df, self.model_internal_zones)
                internal_pa = int_mask * df

                # Read in the trip length bands
                trip_len_bands = self._get_hb_trip_length_bands(p)

                # Generate trip length data
                reports = tms_reports.get_trip_length_by_band(
                    trip_len_bands,
                    distance.values,
                    internal_pa.values,
                )
                band_dist, band_share, avg_trip_len = reports

                # Write files to disk
                fnames = [
                    self._band_distances_fname % (trip_origin, year, p),
                    self._band_shares_fname % (trip_origin, year, p),
                ]
                for report, fname in zip([band_dist, band_share], fnames):
                    path = os.path.join(report_export, fname)
                    report.to_csv(path, index=False)

                # Store a report to return
                avg_trip_lengths.append({
                    'name': '%s_yr%s_p%s' % (trip_origin, year, p),
                    'avg_trip_length': avg_trip_len,
                })
                pbar.update(1)

        pbar.close()

        return avg_trip_lengths

    def analyse_compiled_matrices(self) -> None:
        """
        Generates a report analysing the trips in the compiled
        matrices

        Returns
        -------
        None
        """
        # Init
        report_name = "compiled_matrices_trips.csv"
        import_dir = self.efs_exports['compiled_od_pcu']
        mat_fnames = file_ops.list_files(import_dir)

        # Generate the report
        report = list()
        for fname in mat_fnames:
            # Read in the matrix
            path = os.path.join(import_dir, fname)
            mat = file_ops.read_df(path, index_col=0, find_similar=True)

            if mat.shape[0] != mat.shape[1]:
                raise nd.NormitsDemandError(
                    "The read in matrix isn't square! Read in %s and got "
                    "shape %s" % (fname, mat.shape)
                )

            # Generate a mask for inter and intra
            intra_mask = np.diag([1] * mat.shape[0])
            inter_mask = 1 - intra_mask

            # Calculate totals
            intra_trips = (mat.values * intra_mask).sum()
            inter_trips = (mat.values * inter_mask).sum()

            # Add to report
            report.append({
                'File Name': fname,
                'inter_zonal': inter_trips,
                'intra_zonal': intra_trips,
            })

        # Write out the report
        path = os.path.join(self.exports['home'], report_name)
        pd.DataFrame(report).to_csv(path, index=False)

    def compare_trip_lengths(self) -> None:
        """
        Generates a report comparing post-me trip lengths to the
        trip lengths being returned in the furness (24hr PA)
        """
        # Init
        atl_name = "average_trip_lengths.csv"

        # Make sure the files we need exist
        path_dict = self.imports['post_me_pa']
        file_ops.check_path_exists(path_dict)

        # Build a dictionary of distances
        print("Reading distances...")
        distances = {p: self._get_distance(p) for p in consts.ALL_HB_P}

        # Generate post-me reports
        avg_trip_lengths = self._generate_trip_band_report_by_purpose(
            distance_dict=distances,
            raw_mat_import=self.imports['post_me_pa'],
            cache_export=self.exports['cache']['post_me_tlb'],
            report_export=self.exports['tlb']['post_me'],
            years_needed=[self.base_year],
            trip_origin='hb',
            matrix_format='pa',
            mode=consts.MODEL_MODES[self.model_name][0],
        )
        atl_path = os.path.join(self.exports['tlb']['post_me'], atl_name)
        pd.DataFrame(avg_trip_lengths).to_csv(atl_path, index=False)

        # Generate post-furness reports
        avg_trip_lengths = self._generate_trip_band_report_by_purpose(
            distance_dict=distances,
            raw_mat_import=self.imports['matrices']['pa_24'],
            cache_export=self.exports['cache']['pa_24_tlb'],
            report_export=self.exports['tlb']['pa_24'],
            years_needed=self.years_needed,
            trip_origin='hb',
            matrix_format='pa',
            mode=consts.MODEL_MODES[self.model_name][0],
        )
        atl_path = os.path.join(self.exports['tlb']['post_me'], atl_name)
        pd.DataFrame(avg_trip_lengths).to_csv(atl_path, index=False)

        # ## GENERATE SUMMARY REPORTS ## #

        # Make the base report
        band_share_reports = dict()
        for purpose in consts.ALL_HB_P:
            fname = self._band_shares_fname % ('hb', self.base_year, purpose)
            rep = pd.read_csv(os.path.join(self.exports['tlb']['post_me'], fname))
            rep = rep.rename(columns={'bs': 'post_me'})
            band_share_reports[purpose] = rep

        # Make the progress bar
        pbar = tqdm(
            total=len(self.years_needed) * len(consts.ALL_HB_P),
            desc="Generating summary reports"
        )
        # Tack on the other years
        for year in self.years_needed:
            for purpose in band_share_reports.keys():
                # Read in the report
                fname = self._band_shares_fname % ('hb', year, purpose)
                rep = pd.read_csv(os.path.join(self.exports['tlb']['pa_24'], fname))

                # Add to report
                band_share_reports[purpose][year] = rep['bs']

                pbar.update(1)
        pbar.close()

        # Write out the reports
        for purpose, report in band_share_reports.items():
            fname = self._band_shares_fname % ('hb', 'all', purpose)
            path = os.path.join(self.exports['home'], fname)
            report.to_csv(path, index=False)

    def compare_raw_pa_vectors_to_ntem(self) -> pd.DataFrame:
        """
        Generates a report of the base P/A Vectors to NTEM data

        Returns
        -------
        report:
            A copy of the report comparing the base vectors to NTEM
        """
        # Init
        matrix_format = 'pa'
        output_fname = "raw_vector_report.csv"

        # Make sure the files we need exist
        path_dict = self.imports['raw_vectors']

        for _, path in path_dict.items():
            file_ops.check_file_exists(path)

        # Load in the vectors
        vector_dict = {k: pd.read_csv(v) for k, v in path_dict.items()}

        # Filter down the vectors for speed
        group_cols = ['msoa_zone_id', 'p', 'm']
        index_cols = group_cols.copy() + self.years_needed

        for k, v in vector_dict.items():
            v = v.reindex(columns=index_cols)
            v = v.groupby(group_cols).sum().reset_index()
            vector_dict[k] = v

        return self._generate_vector_report(
            vector_dict=vector_dict,
            vector_zone_col=self._zone_col_base % self.synth_zone_name,
            zone_to_lad=pd.read_csv(self.imports['ntem']['msoa_to_lad']),
            matrix_format=matrix_format,
            output_path=os.path.join(self.exports['home'], output_fname),
            vector_types=self._pa_vector_types,
            trip_origins=self._trip_origins,
        )

    def compare_raw_pa_vectors_to_ntem_by_mode(self,
                                               verbose: bool = True
                                               ) -> pd.DataFrame:
        """
        Generates a report of the base P/A Vectors to NTEM data

        Returns
        -------
        report:
            A copy of the report comparing the base vectors to NTEM
        """
        # Init
        matrix_format = 'pa'
        base_output_fname = "m%s_raw_vector_report.csv"

        # Make sure the files we need exist
        path_dict = self.imports['raw_vectors']

        for _, path in path_dict.items():
            file_ops.check_file_exists(path)

        # Load in the vectors
        vector_dict = {k: pd.read_csv(v) for k, v in path_dict.items()}

        # Filter down the vectors for speed
        group_cols = ['msoa_zone_id', 'p', 'm']
        index_cols = group_cols.copy() + self.years_needed

        for k, v in vector_dict.items():
            v = v.reindex(columns=index_cols)
            v = v.groupby(group_cols).sum().reset_index()
            vector_dict[k] = v

        # Generate a report per mode
        report_ph = list()
        desc = "Building modal reports"
        for mode in tqdm(consts.ALL_MODES, desc=desc, disable=(not verbose)):
            # extract just the data for this mode
            m_vector_dict = {k: v[v['m'] == mode].copy() for k, v in vector_dict.items()}

            output_fname = base_output_fname % str(mode)
            output_path = os.path.join(self.exports['modal'], output_fname)

            report = self._generate_vector_report(
                vector_dict=m_vector_dict,
                vector_zone_col=self._zone_col_base % self.synth_zone_name,
                zone_to_lad=pd.read_csv(self.imports['ntem']['msoa_to_lad']),
                matrix_format=matrix_format,
                output_path=output_path,
                vector_types=self._pa_vector_types,
                trip_origins=self._trip_origins,
            )

            report['m'] = mode
            report_ph.append(report)

        # Write the concatenated report to disk
        full_report = pd.concat(report_ph)
        path = os.path.join(self.exports['home'], 'raw_vector_modal_report.csv')
        full_report.to_csv(path, index=False)

        return full_report

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
            vector_types=self._pa_vector_types,
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
            vector_types=self._pa_vector_types,
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
            vector_types=self._pa_vector_types,
            trip_origins=['hb', 'nhb'],
            report_subsets=self.reporting_subsets,
        )

    def compare_eg_pa_vectors_to_post_me(self) -> pd.DataFrame:
        # Init
        matrix_format = 'pa'
        output_fname = "eg_pa_vectors_to_postme_report.csv"
        zonal_output_fname = "eg_pa_vectors_to_postme_zonal_report.csv"
        out_path = os.path.join(self.exports['home'], output_fname)
        zonal_out_path = os.path.join(self.exports['home'], zonal_output_fname)
        vector_order = [
            'hb_productions',
            'nhb_productions',
            'hb_attractions',
            'nhb_attractions',
        ]

        # Make sure the EG PA vectors exist
        path_dict = self.imports['eg_vectors']
        for _, path in path_dict.items():
            file_ops.check_file_exists(path)

        # Read in the EG PA vectors
        eg_pa_dict = {k: pd.read_csv(v) for k, v in path_dict.items()}

        # Make sure the files we need exist
        path_dict = self.imports['translated_base_vectors']
        for _, path in path_dict.items():
            file_ops.check_file_exists(path)

        # Convert post-me matrices into vector
        vectors = mat_p.maybe_convert_matrices_to_vector(
            mat_import_dir=self.imports['matrices']['post_me'],
            years_needed=[self.base_year],
            cache_path=self.exports['cache']['post_me'],
            matrix_format=matrix_format,
            model_zone_col=self.model_zone_col,
            internal_zones=self.model_internal_zones,
        )
        # Assign to a dictionary for accessing
        post_me_dict = {name: vec for name, vec in zip(vector_order, vectors)}

        # Perform a high level comparison
        report = list()
        zonal_report = list()
        for to, vec_type in itertools.product(['hb', 'nhb'], ['productions', 'attractions']):
            vec_name = '%s_%s' % (to, vec_type)

            post_me_vec = post_me_dict[vec_name]
            eg_pa_vec = eg_pa_dict[vec_name]

            # Generate all zones report
            post_me_total = post_me_vec[self.base_year].sum()
            eg_pa_total = eg_pa_vec[self.base_year].sum()
            diff = eg_pa_total - post_me_total

            report.append({
                'Name': vec_name,
                'eg_pa_vec': eg_pa_total,
                'post_me': post_me_total,
                'diff': diff,
                '% diff': diff / post_me_total * 100,
            })

            # ## ZONE SPECIFIC REPORT ## #
            post_me_vec = post_me_vec.rename(columns={self.base_year: 'post_me'})
            eg_pa_vec = eg_pa_vec.rename(columns={self.base_year: 'pa_vec'})

            # Grab just the data we need
            merge_cols = [self.model_zone_col, 'p']
            post_me_vec = post_me_vec.reindex(columns=merge_cols + ['post_me'])
            post_me_vec = post_me_vec.groupby(merge_cols).sum().reset_index()

            eg_pa_vec = eg_pa_vec.reindex(columns=merge_cols + ['pa_vec'])
            eg_pa_vec = eg_pa_vec.groupby(merge_cols).sum().reset_index()

            # Stick together
            vec = pd.merge(
                post_me_vec,
                eg_pa_vec,
                how='outer',
                on=merge_cols
            ).fillna(0)

            # Calculate differences
            diff = vec['pa_vec'] - vec['post_me']
            vec[vec_name + '_diff'] = diff
            vec[vec_name + '_%diff'] = diff / vec['post_me'] * 100

            vec = vec.drop(columns=['post_me', 'pa_vec'])
            vec = vec.set_index(merge_cols)
            zonal_report.append(vec)

        # Write out reports
        report = pd.DataFrame(report)

        report.to_csv(out_path, index=False)
        pd.concat(zonal_report, axis=1).to_csv(zonal_out_path)

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
            model_zone_col=self.model_zone_col,
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
            vector_types=self._pa_vector_types,
            trip_origins=self._trip_origins,
            report_subsets=self.reporting_subsets,
        )

    def compare_bespoke_pa_matrices_to_ntem(self) -> pd.DataFrame:
        """
        Generates a report of the PA matrices (after bespoke zone integration)
        compared to NTEM data.

        Returns
        -------
        report:
            A copy of the report comparing the PA matrices to NTEM
        """
        # Init
        matrix_format = 'pa'
        output_fname = "bespoke_24hr_pa_matrices_report.csv"
        vector_order = [
            'hb_productions',
            'nhb_productions',
            'hb_attractions',
            'nhb_attractions',
        ]

        # Convert matrices into vector
        vectors = mat_p.maybe_convert_matrices_to_vector(
            mat_import_dir=self.imports['matrices']['pa_24_bespoke'],
            years_needed=self.years_needed,
            cache_path=self.exports['cache']['pa_24_bespoke'],
            matrix_format=matrix_format,
            model_zone_col=self.model_zone_col,
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
            vector_types=self._pa_vector_types,
            trip_origins=self._trip_origins,
            report_subsets=self.reporting_subsets,
        )

    def compare_tp_pa_matrices_to_ntem(self) -> pd.DataFrame:
        """
        Generates a report of the time period split PA matrices compared
        to NTEM data

        Returns
        -------
        report:
            A copy of the report comparing the time period split PA matrices
            to NTEM
        """
        # Init
        matrix_format = 'pa'
        output_fname = "base_tp_pa_matrices_report.csv"
        vector_order = [
            'hb_productions',
            'nhb_productions',
            'hb_attractions',
            'nhb_attractions',
        ]

        # Convert matrices into vector
        vectors = mat_p.maybe_convert_matrices_to_vector(
            mat_import_dir=self.imports['matrices']['pa'],
            years_needed=self.years_needed,
            cache_path=self.exports['cache']['pa'],
            matrix_format=matrix_format,
            model_zone_col=self.model_zone_col,
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
            vector_types=self._pa_vector_types,
            trip_origins=self._trip_origins,
            report_subsets=self.reporting_subsets,
        )

    def compare_od_matrices_to_ntem(self) -> pd.DataFrame:
        """
        Generates a report of the OD matrices compared to NTEM data

        Returns
        -------
        report:
            A copy of the report comparing the OD matrices to NTEM
        """
        # Init
        matrix_format = 'od'
        output_fname = "base_tp_od_matrices_report.csv"
        vector_order = [
            'hb_origins',
            'nhb_origins',
            'hb_destinations',
            'nhb_destinations',
        ]

        # Convert matrices into vector
        vectors = mat_p.maybe_convert_matrices_to_vector(
            mat_import_dir=self.imports['matrices']['od'],
            years_needed=self.years_needed,
            cache_path=self.exports['cache']['od'],
            matrix_format=matrix_format,
            model_zone_col=self.model_zone_col,
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
            vector_types=self._od_vector_types,
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

    constraint_dtypes:
        A dictionary of constraint col_names to constraint col dtypes.
        If left as None, defaults to all dtypes being str.
        e.g. {'p': str, 'm': str}.

    compare_year:
        The year to use when comparing to NTEM. If left as None, compare_cols
        is assumed to contain the years

    report_subsets:
        A dictionary of subset names to zonal subsets to generate reports for.
        The subset names will be placed in subset_col_name of the report.
        If left as None, no subsets are used in the report.

    subset_col_name:
        The name to give to the column that the subset names from
        report_subsets will go into.

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
            adj_mask = adj_vector[base_zone_name].isin(vals)

            vector_subset = vector[mask].reindex(columns=needed_cols)
            adj_vector_subset = adj_vector[adj_mask].reindex(columns=needed_cols)

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
