# -*- coding: utf-8 -*-
"""
Created on: Friday July 2nd 2021
Updated on: Wednesday July 21st 2021

Original author: Nirmal Kumar
Last update made by: Ben Taylor
Other updates made by: Ben Taylor

File purpose:
Attraction Models for NoTEM
"""

# Allow class self type hinting
from __future__ import annotations

# Builtins
import os
import math

from typing import Dict
from typing import List
from typing import Tuple

# Third party imports
import pandas as pd

# local imports
import normits_demand as nd

from normits_demand import efs_constants as consts

from normits_demand.utils import general as du
from normits_demand.utils import file_ops
from normits_demand.utils import timing


class HBAttractionModel:
    # Constants
    _trip_origin = 'hb'
    _zoning_system = 'msoa'

    # Segmentation names
    _pure_attractions = 'pure_attractions'
    _fully_segmented = 'fully_segmented'
    _notem_segmented = 'notem_segmented'

    # Define wanted columns
    _target_col_dtypes = {
        # TODO: Check with BT on how to deal with year in column name
        'land_use': {
            'msoa_zone_id': str,
            'employment_cat': str,
            'soc': int,
            '2018': float
        },
        'trip_rate': {
            'msoa_zone_id': str,
            'employment_cat': str,
            'purpose': int,
            'soc': int,
            'trip_rate': float
        },
        'mode_split': {
            'msoa_zone_id': str,
            'p': int,
            'm': int,
            'mode_share': float
        },
    }

    # Define segment renames needed
    _seg_rename = {
        'employment_cat': 'e_cat',
        '2018': 'emp',
        'area_type': 'tfn_at',
        'purpose': 'p',
        'mode_share': 'split'
    }

    # Define output fnames
    _base_output_fname = '%s_%s_%s_%d_dvec.pkl'
    _base_report_fname = '%s_%s_%d_%s.csv'

    def __init__(self,
                 land_use_paths: Dict[int, nd.PathLike],
                 control_production_paths: Dict[int, nd.PathLike],
                 attraction_trip_rates_path: str,
                 mode_splits_path: str,
                 export_path: str,
                 constraint_paths: Dict[int, nd.PathLike] = None,
                 process_count: int = consts.PROCESS_COUNT
                 ) -> None:
        """
        Sets up and validates arguments for the Attraction model.

        Parameters
        ----------
        land_use_paths:
            Dictionary of {year: land_use_employment_data} pairs.

        control_production_paths:
            Dictionary of {year: path_to_production_to_control_to} pairs.
            These paths should be gotten from nd.HBProduction model.
            Must contain the same keys as land_use_paths, but it can contain
            more (any extras will be ignored).
            These productions will be used to control the produced attractions.

        attraction_trip_rates_path:
            The path to the attraction trip rates.
            Should have the columns as defined in:
            HBAttractionModel._target_cols['trip_rate']

        mode_splits_path:
            The path to attraction mode split.
            Should have the columns as defined in:
            HBAttractionModel._target_cols['mode_split']

        export_path:
            Path to export attraction outputs.

        constraint_paths:
            Dictionary of {year: constraint_path} pairs.
            Must contain the same keys as land_use_paths, but it can contain
            more (any extras will be ignored).
            If set - will be used to constrain the attractions - a report will
            be written before and after.

        process_count:
            The number of processes to create in the Pool. Typically this
            should not exceed the number of cores available.
            Defaults to consts.PROCESS_COUNT.
        """
        # Validate inputs
        [file_ops.check_file_exists(x) for x in land_use_paths.values()]
        [file_ops.check_file_exists(x) for x in control_production_paths.values()]
        file_ops.check_file_exists(attraction_trip_rates_path)
        file_ops.check_file_exists(mode_splits_path)
        file_ops.check_path_exists(export_path)

        if constraint_paths is not None:
            [file_ops.check_file_exists(x) for x in constraint_paths.values()]

        for year in land_use_paths.keys():
            if year not in control_production_paths.keys():
                raise ValueError(
                    "Year %d found in land_use_paths\n"
                    "But not found in control_production_paths"
                    % year
                )
            if constraint_paths is not None:
                if year not in constraint_paths.keys():
                    raise ValueError(
                        "Year %d found in land_use_paths\n"
                        "But not found in constraint_paths"
                        % year
                    )

        # Assign
        self.land_use_paths = land_use_paths
        self.control_production_paths = control_production_paths
        self.attraction_trip_rates_path = attraction_trip_rates_path
        self.mode_splits_path = mode_splits_path
        self.constraint_paths = constraint_paths
        self.export_path = export_path
        self.report_path = os.path.join(export_path, "Reports")
        self.process_count = process_count
        self.years = list(self.land_use_paths.keys())

        # Create paths
        du.create_folder(self.report_path, verbose=False)

        # Initialise Output paths
        # TODO(BT): Convert output paths into dictionaries
        #  something like: self.reports['pure_attractions'][year]
        self._create_output_paths(self.export_path, self.years)
        self._create_pure_attractions_report_paths(self.report_path, self.years)
        self._create_notem_segmented_report_paths(self.report_path, self.years)

    def run(self,
            export_pure_attractions: bool = True,
            export_fully_segmented: bool = False,
            export_notem_segmentation: bool = True,
            export_reports: bool = True,
            verbose: bool = False,
            ) -> None:
        """
        Runs the HB Attraction model.

        Completes the following steps for each year:
            - Reads in the land use employment data given in the constructor.
            - Reads in the trip rates data given in the constructor.
            - Multiplies the employment and trip rates on relevant segments,
              producing "pure attractions".
            - Optionally writes out a pickled DVector of "pure attractions" at
              self.pure_attractions_out[year]
            - Optionally writes out a number of "pure attractions" reports, if
              reports is True.
            - Reads in the mode splits given in the constructor.
            - Multiplies the "pure attractions" and mode splits on relevant
              segments, producing "fully segmented attractions".
            - Checks the attraction totals before and after mode split and throws
              error if they don't match.
            - Optionally writes out a pickled DVector of "fully segmented attractions"
              at self.fully_segmented_paths[year].
            - Balances "fully segmented attractions" to production notem segmentation,
              producing "notem segmented" attractions.
            - Optionally writes out a pickled DVector of "notem segmented attractions"
              at self.notem_segmented_paths[year].
            - Optionally writes out a number of "notem segmented" reports, if
              reports is True.

        Parameters
        ----------
        export_pure_attractions:
            Whether to export the pure attractions to disk or not.
            Will be written out to: self.pure_attractions_out[year]

        export_fully_segmented:
            Whether to export the fully segmented attractions to disk or not.
            Will be written out to: self.fully_segmented_paths[year]

        export_notem_segmentation:
            Whether to export the notem segmented demand to disk or not.
            Will be written out to: self.notem_segmented_paths[year]

        export_reports:
            Whether to output reports while running. All reports will be
            written out to self.report_path.

        verbose:
            Whether to print progress bars during processing or not.

        Returns
        -------
        None
        """
        # Initialise timing
        # TODO(BT): Properly integrate logging
        start_time = timing.current_milli_time()
        du.print_w_toggle(
            "Starting HB Attraction Model at: %s" % timing.get_datetime(),
            verbose=verbose
        )

        # Generate the attractions for each year
        for year in self.years:
            # ## GENERATE PURE ATTRACTIONS ## #
            du.print_w_toggle("Loading the employment data...", verbose=verbose)
            emp_dvec = self._read_land_use_data(year, verbose=verbose)

            du.print_w_toggle("Applying trip rates...", verbose=verbose)
            pure_attractions = self._generate_attractions(emp_dvec, verbose=verbose)

            if export_pure_attractions:
                du.print_w_toggle("Exporting pure attractions to disk...", verbose=verbose)
                pure_attractions.to_pickle(self.pure_attractions_paths[year])

            if export_reports:
                du.print_w_toggle(
                    "Exporting pure attractions reports disk...\n"
                    "Total Attractions for year %d: %.4f"
                    % (year, pure_attractions.sum()),
                    verbose=verbose
                )

                self._write_reports(
                    dvec=pure_attractions,
                    segment_totals_path=self.pd_report_segment_paths[year],
                    ca_sector_path=self.pd_report_ca_sector_paths[year],
                    ie_sector_path=self.pd_report_ie_sector_paths[year],
                )

            # ## SPLIT PURE ATTRACTIONS BY MODE ## #
            du.print_w_toggle("Splitting by mode...", verbose=verbose)
            fully_segmented = self._split_by_mode(pure_attractions)

            self._attractions_total_check(
                pure_attractions=pure_attractions,
                fully_segmented_attractions=fully_segmented,
            )

            # Output attractions before any aggregation
            if export_fully_segmented:
                du.print_w_toggle(
                    "Exporting fully segmented attractions to disk...",
                    verbose=verbose,
                )
                fully_segmented.to_pickle(self.fully_segmented_paths[year])

            # Control the attractions to the productions - this also adds in
            # some segmentation to bring it in line with the productions
            notem_segmented = self._attractions_balance(
                a_dvec=fully_segmented,
                p_dvec_path=self.control_production_paths[year],
            )

            if export_notem_segmentation:
                du.print_w_toggle("Exporting notem segmented attractions to disk...", verbose=verbose)
                notem_segmented.to_pickle(self.notem_segmented_paths[year])

            if export_reports:
                du.print_w_toggle(
                    "Exporting notem segmented attractions reports disk...\n"
                    "Total Attractions for year %d: %.4f"
                    % (year, notem_segmented.sum()),
                    verbose=verbose
                )

                self._write_reports(
                    dvec=notem_segmented,
                    segment_totals_path=self.notem_report_segment_paths[year],
                    ca_sector_path=self.notem_report_ca_sector_paths[year],
                    ie_sector_path=self.notem_report_ie_sector_paths[year],
                )

            # TODO: Bring in constraints (Validation)
            #  Output some audits of what attractions was before and after control
            #  By segment.
            if self.constraint_paths is not None:
                raise NotImplemented(
                    "No code implemented to constrain attractions."
                )

            # End timing
            end_time = timing.current_milli_time()
            du.print_w_toggle("Finished HB Attraction Model at: %s" % timing.get_datetime(),
                              verbose=verbose)
            du.print_w_toggle("HB Attraction Model took: %s"
                              % timing.time_taken(start_time, end_time), verbose=verbose)

    def _read_land_use_data(self,
                            year: int,
                            verbose: bool,
                            ) -> nd.DVector:
        """
        Reads in the land use data for year and converts it into a Dvector.

        Parameters
        ----------
        year:
            The year to get attraction data for.

        verbose:
            Passed into the DVector.

        Returns
        -------
        emp_dvec:
            Returns employment as a Dvector
        """
        # Define the zoning and segmentations we want to use
        msoa_zoning = nd.get_zoning_system('msoa')
        emp_seg = nd.get_segmentation_level('lu_emp')

        # Read the land use data corresponding to the year
        emp = du.safe_read_csv(
            file_path=self.land_use_paths[year],
            usecols=self._target_col_dtypes['land_use'].keys(),
            dtype=self._target_col_dtypes['land_use'],
        )

        # Instantiate
        return nd.DVector(
            zoning_system=msoa_zoning,
            segmentation=emp_seg,
            import_data=emp.rename(columns=self._seg_rename),
            zone_col="msoa_zone_id",
            val_col="emp",
            verbose=verbose,
        )

    def _generate_attractions(self,
                              emp_dvec: nd.DVector,
                              verbose: bool = True,
                              ) -> nd.DVector:
        """
        Applies trip rates to the given HB employment.

        Parameters
        ----------
        emp_dvec:
            Dvector containing the employment.

        verbose:
            Whether to print a progress bar while applying the splits or not.

        Returns
        -------
        pure_attraction:
            Returns the product of employment and attraction trip rate Dvector.
            ie., pure attraction
        """
        # Define the zoning and segmentations we want to use
        msoa_zoning = nd.get_zoning_system('msoa')
        pure_attractions_ecat_seg = nd.get_segmentation_level('pure_attractions_ecat')
        pure_attractions_seg = nd.get_segmentation_level('pure_attractions')

        # Reading trip rates
        du.print_w_toggle("Reading in files...", verbose=verbose)
        trip_rates = du.safe_read_csv(
            self.attraction_trip_rates_path,
            usecols=self._target_col_dtypes['trip_rate'].keys(),
            dtype=self._target_col_dtypes['trip_rate'],
        )

        # ## CREATE THE TRIP RATES DVEC ## #
        du.print_w_toggle("Creating trip rates DVec...", verbose=verbose)

        # Instantiate
        trip_rates_dvec = nd.DVector(
            zoning_system=msoa_zoning,
            segmentation=pure_attractions_ecat_seg,
            import_data=trip_rates.rename(columns=self._seg_rename),
            zone_col="msoa_zone_id",
            val_col="trip_rate",
            verbose=verbose,
        )

        # ## MULTIPLY TOGETHER ## #
        # Remove un-needed ecat column too
        pure_attractions_ecat = emp_dvec * trip_rates_dvec
        return pure_attractions_ecat.aggregate(pure_attractions_seg)

    # TODO: module _write_reports is common for both production and attraction models, so it can be grouped elsewhere
    @staticmethod
    def _write_reports(dvec: nd.DVector,
                       segment_totals_path: nd.PathLike,
                       ca_sector_path: nd.PathLike,
                       ie_sector_path: nd.PathLike,
                       ) -> None:
        """
        Writes segment, CA sector, and IE sector reports to disk.

        Parameters
        ----------
        dvec:
            The Dvector to write the reports for.

        segment_totals_path:
            Path to write the segment totals report to.

        ca_sector_path:
            Path to write the CA sector report to.

        ie_sector_path:
            Path to write the IE sector report to.

        Returns
        -------
        None
        """
        # Segment totals report
        df = dvec.sum_zoning().to_df()
        df.to_csv(segment_totals_path, index=False)

        # Segment by CA Sector total reports
        tfn_ca_sectors = nd.get_zoning_system('ca_sector_2020')
        df = dvec.translate_zoning(tfn_ca_sectors)
        df.to_df().to_csv(ca_sector_path, index=False)

        # Segment by IE Sector total reports
        ie_sectors = nd.get_zoning_system('ie_sector')
        df = dvec.translate_zoning(ie_sectors).to_df()
        df.to_csv(ie_sector_path, index=False)

    def _split_by_mode(self,
                       attractions: nd.DVector,
                       ) -> nd.DVector:
        """
        Applies mode splits to the given balanced pure attractions.

        Parameters
        ----------
        attractions:
            Dvector containing the attractions to split.

        Returns
        -------
        full_segmented_attractions:
            A DVector containing pure_attractions split by mode.
        """
        # Define the segmentation we want to use
        m_pure_attractions_seg = nd.get_segmentation_level('hb_p_m')
        msoa_zoning = nd.get_zoning_system('msoa')

        # Create the mode-time splits DVector
        mode_splits = pd.read_csv(
            self.mode_splits_path,
            usecols=self._target_col_dtypes['mode_split'].keys(),
            dtype=self._target_col_dtypes['mode_split'],
        )

        mode_splits_dvec = nd.DVector(
            zoning_system=msoa_zoning,
            segmentation=m_pure_attractions_seg,
            import_data=mode_splits.rename(columns=self._seg_rename),
            val_col="split",
            zone_col="msoa_zone_id",
        )

        return attractions * mode_splits_dvec

    @staticmethod
    def _attractions_total_check(pure_attractions: nd.DVector,
                                 fully_segmented_attractions: nd.DVector,
                                 rel_tol: float = 0.0001,
                                 ) -> None:
        """
        Checks if totals match

        Checks if the attraction totals are matching before and
        after mode split and returns error message if they are unequal.

        Parameters
        -----------
        pure_attractions:
            Dvector containing pure attractions.

        fully_segmented_attractions:
            Dvector containing attractions after mode split.

        rel_tol:
            the relative tolerance – it is the maximum allowed difference
            between the sum of pure_attractions and fully_segmented_attractions,
            relative to the larger absolute value of pure_attractions or
            fully_segmented_attractions. By default, this is set to 0.0001,
            meaning the values must be within 0.01% of each other.
        """
        # Init
        pa_sum = pure_attractions.sum()
        fsa_sum = fully_segmented_attractions.sum()

        # check
        if not math.isclose(pa_sum, fsa_sum, rel_tol=rel_tol):
            raise ValueError(
                "The attraction totals before and after mode split are not same.\n"
                "Expected %f\n"
                "Got %f"
                % (pa_sum, fsa_sum)
            )

    def _create_output_paths(self,
                             export_path: nd.PathLike,
                             years: List[int],
                             ) -> None:
        """
        Creates output file paths for different segmentations.

        Creates output file names for pure attractions, fully segmented
        and notem segmented HB attraction outputs for the list of years.

        Parameters
        ----------
        export_path:
            Location where the output files are to be created.

        years:
            Contains the list of years for which the attraction model is run.

        Returns
        -------
        None
        """
        # Init
        base_fname = self._base_output_fname
        fname_parts = [self._trip_origin, self._zoning_system]

        self.pure_attractions_paths = dict()
        self.fully_segmented_paths = dict()
        self.notem_segmented_paths = dict()

        for year in years:
            # Pure attractions path
            fname = base_fname % (*fname_parts, self._pure_attractions, year)
            self.pure_attractions_paths[year] = os.path.join(export_path, fname)

            # Fully Segmented path
            fname = base_fname % (*fname_parts, self._fully_segmented, year)
            self.fully_segmented_paths[year] = os.path.join(export_path, fname)

            # Notem Segmented path
            fname = base_fname % (*fname_parts, self._notem_segmented, year)
            self.notem_segmented_paths[year] = os.path.join(export_path, fname)

    # TODO: module _create_report_paths is common for both production and attraction models,
    #  so it can be grouped elsewhere
    def _create_report_paths(self,
                             report_path: nd.PathLike,
                             years: List[int],
                             report_name: str,
                             ) -> Tuple[Dict[int, str], Dict[int, str], Dict[int, str]]:
        """
        Creates report file paths for each of years

        Parameters
        ----------
        report_path:
           The home path (directory) where all the reports should go

        years:
           A list of years to generate report paths for

        report_name:
            The name to use in the report filename. Filenames will be named
            as: [report_name, year, report_type], joined with '_'.

        Returns
        -------
        segment_total_paths:
            A dictionary of paths where the key is the year and the value is
            the path to the segment total reports for year.

        ca_sector_total_paths:
            A dictionary of paths where the key is the year and the value is
            the path to the ca sector segment total reports for year.

        ie_sector_total_paths:
            A dictionary of paths where the key is the year and the value is
            the path to the IE sector segment total reports for year.
        """
        # Init
        base_fname = self._base_report_fname
        fname_parts = [self._trip_origin, report_name]

        segment_total_paths = dict()
        ca_sector_paths = dict()
        ie_sector_paths = dict()

        for year in years:
            # Segment totals
            fname = base_fname % (*fname_parts, year, "segment_totals")
            segment_total_paths[year] = os.path.join(report_path, fname)

            # CA sector totals
            fname = base_fname % (*fname_parts, year, "ca_sector_totals")
            ca_sector_paths[year] = os.path.join(report_path, fname)

            # IE sector totals
            fname = base_fname % (*fname_parts, year, "ie_sector_totals")
            ie_sector_paths[year] = os.path.join(report_path, fname)

        return segment_total_paths, ca_sector_paths, ie_sector_paths

    def _create_pure_attractions_report_paths(self,
                                              report_path: nd.PathLike,
                                              years: List[int],
                                              ) -> None:
        """
        Creates pure attractions report file paths for each of years

        Parameters
        ----------
        report_path:
            The home path (directory) where all the reports should go

        years:
            A list of years to generate report paths for

        Returns
        -------
        None
        """
        paths = self._create_report_paths(report_path, years, self._pure_attractions)
        self.pd_report_segment_paths = paths[0]
        self.pd_report_ca_sector_paths = paths[1]
        self.pd_report_ie_sector_paths = paths[2]

    def _create_notem_segmented_report_paths(self,
                                             report_path: nd.PathLike,
                                             years: List[int],
                                             ) -> None:
        """
        Creates notem_segmented report file paths for each of years

        Parameters
        ----------
        report_path:
            The home path (directory) where all the reports should go

        years:
            A list of years to generate report paths for

        Returns
        -------
        None
        """
        paths = self._create_report_paths(report_path, years, self._notem_segmented)
        self.notem_report_segment_paths = paths[0]
        self.notem_report_ca_sector_paths = paths[1]
        self.notem_report_ie_sector_paths = paths[2]

    @staticmethod
    def _attractions_balance(a_dvec: nd.DVector,
                             p_dvec_path: str,
                             ) -> nd.DVector:
        """
        Balances attractions to production segmentation

        Parameters
        ----------
        a_dvec:
            The attractions Dvector to control.

        p_dvec_path:
            The path to the production Dvector to balance the attraction
            DVector to.

        Returns
        -------
        balanced_a_dvec:
            a_dvec controlled to p_dvec
        """
        # Read in the productions DVec from disk
        p_dvec = nd.from_pickle(p_dvec_path)

        # Split a_dvec into p_dvec segments and balance
        a_dvec = a_dvec.split_segmentation_like(p_dvec)
        return a_dvec.balance_at_segments(p_dvec, split_weekday_weekend=True)


class NHBAttractionModel:
    # Constants
    _trip_origin = 'nhb'
    _zoning_system = 'msoa'

    # Segmentation names
    _pure_attractions = 'pure_attractions'
    _notem_segmented = 'notem_segmented'

    # Define output fnames
    _base_output_fname = '%s_%s_%s_%d_dvec.pkl'
    _base_report_fname = '%s_%s_%d_%s.csv'

    def __init__(self,
                 hb_attractions: Dict[int, nd.PathLike],
                 nhb_productions: Dict[int, nd.PathLike],
                 export_path: str,
                 constraint_paths: Dict[int, nd.PathLike] = None,
                 process_count: int = consts.PROCESS_COUNT
                 ) -> None:
        """
        Sets up and validates arguments for the NHB Attraction model.

        Parameters
        ----------
        hb_attractions:
            Dictionary of {year: notem_segmented_HB_attractions_data} pairs.
            These paths should come from nd.HBAttraction model and should
            be pickled Dvector paths.

        nhb_productions:
            Dictionary of {year: notem_segmented_NHB_productions_data} pairs.
            These paths should come from nd.NHBProduction model and should
            be pickled Dvector paths.
            These productions will be used to control the produced attractions.

        export_path:
            Path to export NHB attraction outputs.

        constraint_paths:
            Dictionary of {year: constraint_path} pairs.
            Must contain the same keys as land_use_paths, but it can contain
            more (any extras will be ignored).
            If set - will be used to constrain the attractions - a report will
            be written before and after.

        process_count:
            The number of processes to create in the Pool. Typically this
            should not exceed the number of cores available.
            Defaults to consts.PROCESS_COUNT.
        """

        # Check that the paths we need exist!
        [file_ops.check_file_exists(x) for x in hb_attractions.values()]
        [file_ops.check_file_exists(x) for x in nhb_productions.values()]
        file_ops.check_path_exists(export_path)

        if constraint_paths is not None:
            [file_ops.check_file_exists(x) for x in constraint_paths.values()]

        # Validate that we have data for all the years we're running for
        for year in hb_attractions.keys():
            if year not in nhb_productions.keys():
                raise ValueError(
                    "Year %d found in notem segmented hb_attractions_paths\n"
                    "But not found in notem segmented nhb_productions_paths"
                    % year
                )

            if constraint_paths is not None:
                if year not in constraint_paths.keys():
                    raise ValueError(
                        "Year %d found in notem segmented hb_attractions_paths\n"
                        "But not found in constraint_paths"
                        % year
                    )

        # Assign
        self.hb_attractions = hb_attractions
        self.nhb_productions = nhb_productions
        self.constraint_paths = constraint_paths
        self.export_path = export_path
        self.report_path = os.path.join(export_path, "Reports")
        self.process_count = process_count
        self.years = list(self.hb_attractions.keys())

        # Create paths
        du.create_folder(self.report_path, verbose=False)

        # Initialise Output paths
        # TODO(BT): Convert output paths into dictionaries
        #  something like: self.reports['pure_attractions'][year]
        self._create_output_paths(self.export_path, self.years)
        self._create_notem_segmented_report_paths(self.report_path, self.years)

    def run(self,
            export_nhb_pure_attractions: bool = False,
            export_notem_segmentation: bool = False,
            export_reports: bool = False,
            verbose: bool = False,
            ) -> None:
        """
        Runs the NHB Attraction model.

        Completes the following steps for each year:
            - Reads in the notem segmented HB attractions compressed pickle
              given in the constructor.
            - Changes HB purposes to NHB purposes.
            - Optionally writes out a pickled DVector of "pure attractions" at
              self.pure_attractions_out[year]
            - Optionally writes out a number of "pure attractions" reports, if
              reports is True.
            - Balances "fully segmented attractions" to production notem segmentation,
              producing "notem segmented" NHB attractions.
            - Optionally writes out a pickled DVector of "notem segmented attractions"
              at self.notem_segmented_paths[year].
            - Optionally writes out a number of "notem segmented" reports, if
              reports is True.

        Parameters
        ----------
        export_nhb_pure_attractions:
            Whether to export the pure attractions to disk or not.
            Will be written out to: self.pure_attractions_out[year]

        export_notem_segmentation:
            Whether to export the notem segmented demand to disk or not.
            Will be written out to: self.notem_segmented_paths[year]

        export_reports:
            Whether to output reports while running. All reports will be
            written out to self.report_path.

        verbose:
            Whether to print progress bars during processing or not.

        Returns
        -------
        None
        """
        # Initialise timing
        # TODO(BT): Properly integrate logging
        start_time = timing.current_milli_time()
        du.print_w_toggle(
            "Starting NHB Attraction Model at: %s" % timing.get_datetime(),
            verbose=verbose
        )

        # Generate the nhb attractions for each year
        for year in self.years:
            # ## GENERATE PURE ATTRACTIONS ## #
            du.print_w_toggle("Loading the HB attraction data...", verbose=verbose)
            pure_nhb_attr = self._create_nhb_attraction_data(year, verbose=verbose)

            if export_nhb_pure_attractions:
                du.print_w_toggle("Exporting NHB pure attractions to disk...", verbose=verbose)
                pure_nhb_attr.to_pickle(self.pure_attractions_paths[year])

            if export_reports:
                du.print_w_toggle(
                    "Exporting pure NHB attractions reports to disk...\n"
                    "Total Attractions for year %d: %.4f"
                    % (year, pure_nhb_attr.sum()),
                    verbose=verbose
                )

                self._write_reports(
                    dvec=pure_nhb_attr,
                    segment_totals_path=self.pd_report_segment_paths[year],
                    ca_sector_path=self.pd_report_ca_sector_paths[year],
                    ie_sector_path=self.pd_report_ie_sector_paths[year],
                )

            # Control the attractions to the productions
            notem_segmented = self._attractions_balance(
                a_dvec=pure_nhb_attr,
                p_dvec_path=self.nhb_productions[year],
            )

            if export_notem_segmentation:
                du.print_w_toggle("Exporting notem segmented attractions to disk...", verbose=verbose)
                notem_segmented.to_pickle(self.notem_segmented_paths[year])

            if export_reports:
                du.print_w_toggle(
                    "Exporting notem segmented attractions reports to disk...\n"
                    "Total Attractions for year %d: %.4f"
                    % (year, notem_segmented.sum()),
                    verbose=verbose
                )

                self._write_reports(
                    dvec=notem_segmented,
                    segment_totals_path=self.notem_report_segment_paths[year],
                    ca_sector_path=self.notem_report_ca_sector_paths[year],
                    ie_sector_path=self.notem_report_ie_sector_paths[year],
                )

            # TODO: Bring in constraints (Validation)
            #  Output some audits of what attractions was before and after control
            #  By segment.
            if self.constraint_paths is not None:
                raise NotImplemented(
                    "No code implemented to constrain attractions."
                )

            # End timing
            end_time = timing.current_milli_time()
            du.print_w_toggle("Finished NHB Attraction Model at: %s" % timing.get_datetime(),
                              verbose=verbose)
            du.print_w_toggle("NHB Attraction Model took: %s"
                              % timing.time_taken(start_time, end_time), verbose=verbose)

    def _create_nhb_attraction_data(self,
                                    year: int,
                                    verbose: bool,
                                    ) -> nd.DVector:
        """
        Reads in HB attractions converts it into a NHB attractions Dvector.

        - Reads the HB attractions compressed pickle.
        - Removes p1 and p7 from the HB purposes.
        - Adds 10 to the remaining purposes to create NHB purposes.
        - Returns its DVector

        Parameters
        ----------
        year:
            The year to get HB attractions data for.

        verbose:
            Passed into the DVector.

        Returns
        -------
        nhb_attr_dvec:
            Returns NHB attractions as a Dvector
        """
        # Define the zoning and segmentations we want to use
        msoa_zoning = nd.get_zoning_system('msoa')
        nhb_notem_seg = nd.get_segmentation_level('nhb_notem_output')

        # Reading the notem segmented HB attractions compressed pickle
        hb_attr_notem = nd.from_pickle(self.hb_attractions[year])
        hb_attr_notem_df = hb_attr_notem.to_df()

        # Removing p1 and p7
        hb_attr_notem_df = hb_attr_notem_df.loc[(hb_attr_notem_df['p'].astype(int) != 1) &
                                                (hb_attr_notem_df['p'].astype(int) != 7)]
        # Adding 10 to the remaining purposes
        hb_attr_notem_df['p'] = hb_attr_notem_df['p'].astype(int)+10

        # Instantiate
        return nd.DVector(
            zoning_system=msoa_zoning,
            segmentation=nhb_notem_seg,
            import_data=hb_attr_notem_df,
            zone_col="zone",
            val_col="val",
            verbose=verbose,
        )

    # TODO: module _write_reports is common for both production and attraction models, so it can be grouped elsewhere
    @staticmethod
    def _write_reports(dvec: nd.DVector,
                       segment_totals_path: nd.PathLike,
                       ca_sector_path: nd.PathLike,
                       ie_sector_path: nd.PathLike,
                       ) -> None:
        """
        Writes segment, CA sector, and IE sector reports to disk.

        Parameters
        ----------
        dvec:
            The Dvector to write the reports for.

        segment_totals_path:
            Path to write the segment totals report to.

        ca_sector_path:
            Path to write the CA sector report to.

        ie_sector_path:
            Path to write the IE sector report to.

        Returns
        -------
        None
        """
        # Segment totals report
        df = dvec.sum_zoning().to_df()
        df.to_csv(segment_totals_path, index=False)

        # Segment by CA Sector total reports
        tfn_ca_sectors = nd.get_zoning_system('ca_sector_2020')
        df = dvec.translate_zoning(tfn_ca_sectors)
        df.to_df().to_csv(ca_sector_path, index=False)

        # Segment by IE Sector total reports
        ie_sectors = nd.get_zoning_system('ie_sector')
        df = dvec.translate_zoning(ie_sectors).to_df()
        df.to_csv(ie_sector_path, index=False)

    def _create_output_paths(self,
                             export_path: nd.PathLike,
                             years: List[int],
                             ) -> None:
        """
        Creates output file paths for different segmentations.

        Creates output file names for pure attractions
        and notem segmented HB attraction outputs for the list of years.

        Parameters
        ----------
        export_path:
            Location where the output files are to be created.

        years:
            Contains the list of years for which the attraction model is run.

        Returns
        -------
        None
        """
        # Init
        base_fname = self._base_output_fname
        fname_parts = [self._trip_origin, self._zoning_system]

        self.pure_attractions_paths = dict()
        self.notem_segmented_paths = dict()

        for year in years:
            # Pure attractions path
            fname = base_fname % (*fname_parts, self._pure_attractions, year)
            self.pure_attractions_paths[year] = os.path.join(export_path, fname)

            # Notem Segmented path
            fname = base_fname % (*fname_parts, self._notem_segmented, year)
            self.notem_segmented_paths[year] = os.path.join(export_path, fname)

    # TODO: module _create_report_paths is common for both production and attraction models,
    #  so it can be grouped elsewhere
    def _create_report_paths(self,
                             report_path: nd.PathLike,
                             years: List[int],
                             report_name: str,
                             ) -> Tuple[Dict[int, str], Dict[int, str], Dict[int, str]]:
        """
        Creates report file paths for each of years

        Parameters
        ----------
        report_path:
           The home path (directory) where all the reports should go

        years:
           A list of years to generate report paths for

        report_name:
            The name to use in the report filename. Filenames will be named
            as: [report_name, year, report_type], joined with '_'.

        Returns
        -------
        segment_total_paths:
            A dictionary of paths where the key is the year and the value is
            the path to the segment total reports for year.

        ca_sector_total_paths:
            A dictionary of paths where the key is the year and the value is
            the path to the ca sector segment total reports for year.

        ie_sector_total_paths:
            A dictionary of paths where the key is the year and the value is
            the path to the IE sector segment total reports for year.
        """
        # Init
        base_fname = self._base_report_fname
        fname_parts = [self._trip_origin, report_name]

        segment_total_paths = dict()
        ca_sector_paths = dict()
        ie_sector_paths = dict()

        for year in years:
            # Segment totals
            fname = base_fname % (*fname_parts, year, "segment_totals")
            segment_total_paths[year] = os.path.join(report_path, fname)

            # CA sector totals
            fname = base_fname % (*fname_parts, year, "ca_sector_totals")
            ca_sector_paths[year] = os.path.join(report_path, fname)

            # IE sector totals
            fname = base_fname % (*fname_parts, year, "ie_sector_totals")
            ie_sector_paths[year] = os.path.join(report_path, fname)

        return segment_total_paths, ca_sector_paths, ie_sector_paths

    def _create_pure_attractions_report_paths(self,
                                              report_path: nd.PathLike,
                                              years: List[int],
                                              ) -> None:
        """
        Creates pure attractions report file paths for each of years

        Parameters
        ----------
        report_path:
            The home path (directory) where all the reports should go

        years:
            A list of years to generate report paths for

        Returns
        -------
        None
        """
        paths = self._create_report_paths(report_path, years, self._pure_attractions)
        self.pd_report_segment_paths = paths[0]
        self.pd_report_ca_sector_paths = paths[1]
        self.pd_report_ie_sector_paths = paths[2]

    def _create_notem_segmented_report_paths(self,
                                             report_path: nd.PathLike,
                                             years: List[int],
                                             ) -> None:
        """
        Creates notem_segmented report file paths for each of years

        Parameters
        ----------
        report_path:
            The home path (directory) where all the reports should go

        years:
            A list of years to generate report paths for

        Returns
        -------
        None
        """
        paths = self._create_report_paths(report_path, years, self._notem_segmented)
        self.notem_report_segment_paths = paths[0]
        self.notem_report_ca_sector_paths = paths[1]
        self.notem_report_ie_sector_paths = paths[2]

    @staticmethod
    def _attractions_balance(a_dvec: nd.DVector,
                             p_dvec_path: str,
                             ) -> nd.DVector:
        """
        Balances attractions to production segmentation

        Parameters
        ----------
        a_dvec:
            The attractions Dvector to control.

        p_dvec_path:
            The path to the production Dvector to balance the attraction
            DVector to.

        Returns
        -------
        balanced_a_dvec:
            a_dvec controlled to p_dvec
        """
        # Read in the productions DVec from disk
        p_dvec = nd.from_pickle(p_dvec_path)

        # Balance a_dvec with p_dvec
        return a_dvec.balance_at_segments(p_dvec, split_weekday_weekend=False)
