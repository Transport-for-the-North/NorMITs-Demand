"""
Created on: 02/07/2021

File purpose: Attraction Model for NoTEM

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
from normits_demand.utils import file_ops as ops
from normits_demand.utils import timing


class HBAttractionModel:
    # Constants
    _trip_origin = 'hb'
    _zoning_system = 'msoa'

    # Segmentation names
    _pure_attractions = 'pure_attractions'
    _fully_segmented = 'fully_segmented'

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
                 notem_segmented_productions: str,
                 trip_attraction_rates_path: str,
                 mode_controls_path: str,
                 constraint_paths: Dict[int, nd.PathLike],
                 export_path: str,
                 process_count: int = consts.PROCESS_COUNT
                 ) -> HBAttractionModel:
        """
        Validates and assigns the attributes needed for NoTEM Attraction model.

        Parameters
        ----------
        land_use_paths:
            Dictionary containing different years and the corresponding
            employment path as key and value respectively.

        notem_segmented_productions:
            Contains path to the pickled notem segmented productions which
            is used for balancing attraction.

        trip_attraction_rates_path:
            Contains path to attraction trip rate.

        mode_controls_path:
            Contains path to mode split.

        constraint_paths:
            Dictionary containing different years and the corresponding
            constraint path as key and value respectively.

        export_path:
            Path to export attraction outputs.

        process_count:
            The number of processes to create in the Pool. Typically this
            should not exceed the number of cores available.
            Defaults to consts.PROCESS_COUNT.

        """
        # TODO(BT): Document attributes (partially done by NK)
        # Validate inputs
        [ops.check_file_exists(x) for x in land_use_paths.values()]
        [ops.check_file_exists(x) for x in constraint_paths.values()]
        ops.check_file_exists(notem_segmented_productions)
        ops.check_file_exists(trip_attraction_rates_path)
        ops.check_file_exists(mode_controls_path)
        ops.check_path_exists(export_path)

        # Assign
        self.land_use_paths = land_use_paths
        self.notem_segmented_productions = notem_segmented_productions
        self.trip_att_rates_path = trip_attraction_rates_path
        self.mode_controls_path = mode_controls_path
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
        self._create_full_segmented_report_paths(self.report_path, self.years)

    def run(self,
            export_pure_attractions: bool = False,
            export_fully_segmented: bool = False,
            export_reports: bool = False,
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
            - Optionally writes out a pickled DVector of "fully segmented attractions"
              at self.fully_segmented_paths[year] if export_fully_segmented
              is True.


        Parameters
        ----------
        export_pure_attractions:
            Whether to export the pure attractions to disk or not.
            Will be written out to: self.pure_attractions_out[year]

        export_fully_segmented:
            Whether to export the fully segmented attractions to disk or not.
            Will be written out to: self.fully_segmented_paths[year]

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
            # pu=pure_attractions.to_df()
            # pu.to_csv(r"C:\Data\Nirmal_Atkins\pure_attractions_new.csv",index=False)

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
            mode_split = self._split_by_mode(pure_attractions)

            self._attractions_total_check(
                pure_attractions=pure_attractions,
                fully_segmented_attractions=mode_split,
            )

            # Output attractions before any aggregation
            if export_fully_segmented:
                du.print_w_toggle(
                    "Exporting fully segmented attractions to disk...",
                    verbose=verbose,
                )
                mode_split.to_pickle(self.fully_segmented_paths[year])

            # TODO: Balance pure attractions - report on the balanced
            controlled = self._attractions_balance(
                p_dvec=self.notem_segmented_productions,
                a_dvec=mode_split,
            )

            if export_reports:
                du.print_w_toggle(
                    "Exporting controlled attractions reports disk...\n"
                    "Total Attractions for year %d: %.4f"
                    % (year, controlled.sum()),
                    verbose=verbose
                )

                self._write_reports(
                    dvec=controlled,
                    segment_totals_path=self.full_report_segment_paths[year],
                    ca_sector_path=self.full_report_ca_sector_paths[year],
                    ie_sector_path=self.full_report_ie_sector_paths[year],
                )

            # TODO: Bring in constraints (Validation)
            #  Output some audits of what attractions was before and after control
            #  By segment.

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
        Reads in the land use data for year and converts it to Dvector.

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
            self.trip_att_rates_path,
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
        m_pure_attractions_seg = nd.get_segmentation_level('p_m')
        msoa_zoning = nd.get_zoning_system('msoa')

        # Create the mode-time splits DVector
        mode_splits = pd.read_csv(
            self.mode_controls_path,
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
        Creates output file names for pure attractions, fully segmented and aggregated
        HB attraction outputs for the list of years

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

    def _create_full_segmented_report_paths(self,
                                            report_path: nd.PathLike,
                                            years: List[int],
                                            ) -> None:
        """
        Creates fully_segmented report file paths for each of years

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
        paths = self._create_report_paths(report_path, years, self._fully_segmented)
        self.full_report_segment_paths = paths[0]
        self.full_report_ca_sector_paths = paths[1]
        self.full_report_ie_sector_paths = paths[2]

    def _attractions_balance(self,
                             p_dvec: str,
                             a_dvec: nd.DVector,
                             ) -> nd.DVector:

        p_dvec = nd.from_pickle(p_dvec)

        if a_dvec.segmentation.name != 'p_m_soc':
            seg = nd.get_segmentation_level('p_m_soc')
            a_dvec = a_dvec.aggregate(seg)

        # Split a_dvec into p_dvec segments
        print("Attrs:", a_dvec.sum())
        a_dvec = a_dvec.split_segmentation_like(p_dvec)
        print("Attrs:", a_dvec.sum())

        # Control across segments
        a_dvec = a_dvec.balance_at_segments(p_dvec, split_weekday_weekend=True)
        print("Prods:", p_dvec.sum())
        print("Attrs:", a_dvec.sum())

        return a_dvec
