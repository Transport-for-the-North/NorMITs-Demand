"""
Created on: 19/06/2021

File purpose: Production Model for NoTEM

"""
# Allow class self type hinting
from __future__ import annotations

# Builtins
import os

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


class HBProductionModel:
    # Constants
    _trip_origin = 'hb'
    _zoning_system = 'msoa'
    _return_segmentation_name = 'hb_notem_output'

    # Segmentation names
    _pure_demand = 'pure_demand'
    _fully_segmented = 'fully_segmented'
    _notem_segmented = 'notem_segmented'

    # Define wanted columns
    _target_cols = {
        'land_use': ['msoa_zone_id', 'area_type', 'tfn_traveller_type', 'people'],
        'trip_rate': ['tfn_tt', 'tfn_at', 'p', 'trip_rate'],
        'm_tp': ['p', 'tfn_tt', 'tfn_at', 'm', 'tp', 'split'],
    }

    # Define segment renames needed
    _seg_rename = {
        'tfn_traveller_type': 'tfn_tt',
        'area_type': 'tfn_at',
    }

    # Define output fnames
    _base_output_fname = '%s_%s_%s_%d_dvec.pkl'
    _base_report_fname = '%s_%s_%d_%s.csv'

    def __init__(self,
                 land_use_paths: Dict[int, nd.PathLike],
                 trip_rates_path: str,
                 mode_time_splits_path: str,
                 constraint_paths: Dict[int, nd.PathLike],
                 export_path: str,
                 process_count: int = consts.PROCESS_COUNT
                 ):
        # TODO(BT): DOcument attributes
        # Validate inputs
        [ops.check_file_exists(x) for x in land_use_paths.values()]
        [ops.check_file_exists(x) for x in constraint_paths.values()]
        ops.check_file_exists(trip_rates_path)
        ops.check_file_exists(mode_time_splits_path)
        ops.check_path_exists(export_path)

        # Assign
        self.land_use_paths = land_use_paths
        self.trip_rates_path = trip_rates_path
        self.mode_time_splits_path = mode_time_splits_path
        self.constraint_paths = constraint_paths
        self.export_path = export_path
        self.report_path = os.path.join(export_path, "Reports")
        self.process_count = process_count
        self.years = list(self.land_use_paths.keys())

        # Create paths
        du.create_folder(self.report_path, verbose=False)

        # Initialise Output paths
        # TODO(BT): Convert output paths into dictionaries
        #  something like: self.reports['pure_demand'][year]
        self._create_output_paths(self.export_path, self.years)
        self._create_pure_demand_report_paths(self.report_path, self.years)
        self._create_notem_segmented_report_paths(self.report_path, self.years)

    def run(self,
            export_pure_demand: bool = True,
            export_fully_segmented: bool = False,
            export_notem_segmentation: bool = True,
            export_reports: bool = True,
            verbose: bool = False,
            ) -> None:
        """
        Runs the HB Production model.

        Completes the following steps for each year:
            - Reads in the land use population data given in the constructor.
            - Reads in the trip rates data given in the constructor.
            - Multiplies the population and trip rates on relevant segments,
              producing "pure demand".
            - Optionally writes out a pickled DVector of "pure demand" at
              self.pure_demand_out[year]
            - Optionally writes out a number of "pure demand" reports, if
              reports is True.
            - Reads in the mode-time splits given in the constructor.
            - Multiplies the "pure demand" and mode-time splits on relevant
              segments, producing "fully segmented demand".
            - Optionally writes out a pickled DVector of "fully segmented demand"
              at self.fully_segmented_paths[year] if export_fully_segmented
              is True.
            - Aggregates this demand into hb_notem_full_tfn segmentation,
              producing "notem segmented demand".
            - Optionally writes out a number of "notem segmented demand"
              reports, if reports is True.
            - Optionally writes out a pickled DVector of "notem segmented demand"
              at self.notem_segmented_paths[year] if export_notem_segmentation
              is True.
            - Finally, returns "notem segmented demand" as a DVector.

        Parameters
        ----------
        export_pure_demand:
            Whether to export the pure demand to disk or not.
            Will be written out to: self.pure_demand_out[year]

        export_fully_segmented:
            Whether to export the fully segmented demand to disk or not.
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
            "Starting HB Production Model at: %s" % timing.get_datetime(),
            verbose=verbose
        )

        # Generate the productions for each year
        for year in self.years:
            # ## GENERATE PURE DEMAND ## #
            du.print_w_toggle("Loading the population data...", verbose=verbose)
            pop_dvec = self._read_land_use_data(year, verbose)

            du.print_w_toggle("Applying trip rates...", verbose=verbose)
            pure_demand = self._generate_productions(pop_dvec, verbose)

            if export_pure_demand:
                du.print_w_toggle("Exporting pure demand to disk...", verbose=verbose)
                pure_demand.to_pickle(self.pure_demand_paths[year])

            if export_reports:
                du.print_w_toggle(
                    "Exporting pure demand reports disk...\n"
                    "Total Productions for year %d: %.4f"
                    % (year, pure_demand.sum()),
                    verbose=verbose
                )

                tfn_agg_at_seg = nd.get_segmentation_level('pure_demand_reporting')
                self._write_reports(
                    dvec=pure_demand.aggregate(tfn_agg_at_seg),
                    segment_totals_path=self.pd_report_segment_paths[year],
                    ca_sector_path=self.pd_report_ca_sector_paths[year],
                    ie_sector_path=self.pd_report_ie_sector_paths[year],
                )

            # ## SPLIT PURE DEMAND BY MODE AND TIME ## #
            du.print_w_toggle("Splitting by mode and time...", verbose=verbose)
            fully_segmented = self._split_by_tp_and_mode(pure_demand)

            # Output productions before any aggregation
            if export_fully_segmented:
                du.print_w_toggle(
                    "Exporting fully segmented productions to disk...",
                    verbose=verbose,
                )
                fully_segmented.to_pickle(self.fully_segmented_paths[year])

            # ## AGGREGATE INTO RETURN SEGMENTATION ## #
            return_seg = nd.get_segmentation_level(self._return_segmentation_name)
            productions = fully_segmented.aggregate(
                out_segmentation=return_seg,
                split_tfntt_segmentation=True
            )

            if export_notem_segmentation:
                du.print_w_toggle(
                    "Exporting notem segmented demand to disk...",
                    verbose=verbose
                )
                productions.to_pickle(self.notem_segmented_paths[year])

            if export_reports:
                du.print_w_toggle(
                    "Exporting notem segmented reports disk...\n"
                    "Total Productions for year %d: %.4f"
                    % (year, productions.sum()),
                    verbose=verbose
                )

                self._write_reports(
                    dvec=productions,
                    segment_totals_path=self.notem_report_segment_paths[year],
                    ca_sector_path=self.notem_report_ca_sector_paths[year],
                    ie_sector_path=self.notem_report_ie_sector_paths[year],
                )

            # TODO: Bring in constraints (Validation)
            #  Output some audits of what demand was before and after control
            #  By segment.

            # End timing
            end_time = timing.current_milli_time()
            du.print_w_toggle("Finished HB Production Model at: %s" % timing.get_datetime(),
                              verbose=verbose)
            du.print_w_toggle("HB Production Model took: %s"
                              % timing.time_taken(start_time, end_time), verbose=verbose)

    def _read_land_use_data(self,
                            year: int,
                            verbose: bool,
                            ) -> nd.DVector:
        """
        Reads in the land use data for year and converts it to Dvector

        Parameters
        ----------
        year:
            The year to get population data for.

        verbose:
            Passed into the DVector.

        Returns
        -------
        pop_dvec:
            Returns the population Dvector
        """
        # Define the zoning and segmentations we want to use
        msoa_zoning = nd.get_zoning_system('msoa')
        pop_seg = nd.get_segmentation_level('lu_pop')

        # Read the land use data corresponding to the year
        pop = du.safe_read_csv(
            file_path=self.land_use_paths[year],
            usecols=self._target_cols['land_use']
        )

        # Instantiate
        return nd.DVector(
            zoning_system=msoa_zoning,
            segmentation=pop_seg,
            import_data=pop.rename(columns=self._seg_rename),
            zone_col="msoa_zone_id",
            val_col="people",
            verbose=verbose,
        )

    def _generate_productions(self,
                              population: nd.DVector,
                              verbose: bool,
                              ) -> nd.DVector:
        """
        Applies trip rate split on the given HB productions

        Parameters
        ----------
        population:
            Dvector containing the population.

        verbose:
            Whether to print a progress bar while applying the splits or not

        Returns
        -------
        pure_demand:
            Returns the product of population and trip rate Dvector
            ie., pure demand
        """

        # Define the zoning and segmentations we want to use
        pure_demand_seg = nd.get_segmentation_level('pure_demand')

        # Reading trip rates
        du.print_w_toggle("Reading in files...", verbose=verbose)
        trip_rates = du.safe_read_csv(self.trip_rates_path, usecols=self._target_cols['trip_rate'])

        # ## CREATE THE TRIP RATES DVEC ## #
        du.print_w_toggle("Creating trip rates DVec...", verbose=verbose)

        # Instantiate
        trip_rates_dvec = nd.DVector(
            zoning_system=None,
            segmentation=pure_demand_seg,
            import_data=trip_rates.rename(columns=self._seg_rename),
            val_col="trip_rate",
            verbose=verbose,
        )

        # ## MULTIPLY TOGETHER ## #
        return population * trip_rates_dvec

    @staticmethod
    def _write_reports(dvec: nd.DVector,
                       segment_totals_path: nd.PathLike,
                       ca_sector_path: nd.PathLike,
                       ie_sector_path: nd.PathLike,
                       ) -> None:
        """
        Writes segment, CA sector, and IE sector reports to disk

        Parameters
        ----------
        dvec:
            The Dvector to write the reports for

        segment_totals_path:
            Path to write the segment totals report to

        ca_sector_path:
            Path to write the CA sector report to

        ie_sector_path:
            Path to write the IE sector report to

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

    def _split_by_tp_and_mode(self,
                              pure_demand: nd.DVector,
                              ) -> nd.DVector:
        """
        Applies time period and mode splits to the given pure demand.

        Parameters
        ----------
        pure_demand:
            Dvector containing the pure demand to split.

        Returns
        -------
        full_segmented_demand:
            A DVector containing pure_demand split by mode and time.
        """
        # Define the segmentation we want to use
        m_tp_pure_demand_seg = nd.get_segmentation_level('full_tfntt_tfnat')
        notem_seg = nd.get_segmentation_level('full_tfntt')

        # Create the mode-time splits DVector
        mode_time_splits = pd.read_csv(
            self.mode_time_splits_path,
            usecols=self._target_cols['m_tp']
        )

        mode_time_splits_dvec = nd.DVector(
            zoning_system=None,
            segmentation=m_tp_pure_demand_seg,
            import_data=mode_time_splits,
            val_col="split",
        )

        return pure_demand.multiply_and_aggregate(
            other=mode_time_splits_dvec,
            out_segmentation=notem_seg,
        )

    def _create_output_paths(self,
                             export_path: nd.PathLike,
                             years: List[int],
                             ) -> None:
        """
        Creates output file names for pure demand, fully segmented and aggregated
        HB production outputs for the list of years

        Parameters
        ----------
        export_path:
            Location where the output files are to be created.

        years:
            Contains the list of years for which the production model is run.

        Returns
        -------
        None
        """
        # Init
        base_fname = self._base_output_fname
        fname_parts = [self._trip_origin, self._zoning_system]

        self.pure_demand_paths = dict()
        self.fully_segmented_paths = dict()
        self.notem_segmented_paths = dict()

        for year in years:
            # Pure demand path
            fname = base_fname % (*fname_parts, self._pure_demand, year)
            self.pure_demand_paths[year] = os.path.join(export_path, fname)

            # Fully Segmented path
            fname = base_fname % (*fname_parts, self._fully_segmented, year)
            self.fully_segmented_paths[year] = os.path.join(export_path, fname)

            # NoTEM Segmented path
            fname = base_fname % (*fname_parts, self._notem_segmented, year)
            self.notem_segmented_paths[year] = os.path.join(export_path, fname)

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

    def _create_pure_demand_report_paths(self,
                                         report_path: nd.PathLike,
                                         years: List[int],
                                         ) -> None:
        """
        Creates pure demand report file paths for each of years

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
        paths = self._create_report_paths(report_path, years, self._pure_demand)
        self.pd_report_segment_paths = paths[0]
        self.pd_report_ca_sector_paths = paths[1]
        self.pd_report_ie_sector_paths = paths[2]

    def _create_notem_segmented_report_paths(self,
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
        self.notem_report_segment_paths = paths[0]
        self.notem_report_ca_sector_paths = paths[1]
        self.notem_report_ie_sector_paths = paths[2]

    def _generate_productions(self,
                              population: nd.DVector,
                              verbose: bool,
                              ) -> nd.DVector:
        """
        Applies trip rate split on the given HB productions

        Parameters
        ----------
        population:
            Dvector containing the population.

        verbose:
            Whether to print a progress bar while applying the splits or not

        Returns
        -------
        pure_demand:
            Returns the product of population and trip rate Dvector
            ie., pure demand
        """

        # Define the zoning and segmentations we want to use
        pure_demand_seg = nd.get_segmentation_level('pure_demand')

        # Reading trip rates
        du.print_w_toggle("Reading in files...", verbose=verbose)
        trip_rates = du.safe_read_csv(self.trip_rates_path, usecols=self._target_cols['trip_rate'])

        # ## CREATE THE TRIP RATES DVEC ## #
        du.print_w_toggle("Creating trip rates DVec...", verbose=verbose)

        # Instantiate
        trip_rates_dvec = nd.DVector(
            zoning_system=None,
            segmentation=pure_demand_seg,
            import_data=trip_rates.rename(columns=self._seg_rename),
            val_col="trip_rate",
            verbose=verbose,
        )
        # ## MULTIPLY TOGETHER ## #
        return population * trip_rates_dvec

    @staticmethod
    def _write_reports(dvec: nd.DVector,
                       segment_totals_path: nd.PathLike,
                       ca_sector_path: nd.PathLike,
                       ie_sector_path: nd.PathLike,
                       ) -> None:
        """
        Writes segment, CA sector, and IE sector reports to disk

        Parameters
        ----------
        dvec:
            The Dvector to write the reports for

        segment_totals_path:
            Path to write the segment totals report to

        ca_sector_path:
            Path to write the CA sector report to

        ie_sector_path:
            Path to write the IE sector report to

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

    def _split_by_tp_and_mode(self,
                              pure_demand: nd.DVector,
                              ) -> nd.DVector:
        """
        Applies time period and mode splits to the given pure demand.

        Parameters
        ----------
        pure_demand:
            Dvector containing the pure demand to split.

        Returns
        -------
        full_segmented_demand:
            A DVector containing pure_demand split by mode and time.
        """
        # Define the segmentation we want to use
        m_tp_pure_demand_seg = nd.get_segmentation_level('full_tfntt_tfnat')
        notem_seg = nd.get_segmentation_level('full_tfntt')

        # Create the mode-time splits DVector
        mode_time_splits = pd.read_csv(
            self.mode_time_splits_path,
            usecols=self._target_cols['m_tp']
        )

        mode_time_splits_dvec = nd.DVector(
            zoning_system=None,
            segmentation=m_tp_pure_demand_seg,
            import_data=mode_time_splits,
            val_col="split",
        )

        return pure_demand.multiply_and_aggregate(
            other=mode_time_splits_dvec,
            out_segmentation=notem_seg,
        )

    def _create_output_paths(self,
                             export_path: nd.PathLike,
                             years: List[int],
                             ) -> None:
        """
        Creates output file names for pure demand, fully segmented and aggregated
        HB production outputs for the list of years

        Parameters
        ----------
        export_path:
            Location where the output files are to be created.

        years:
            Contains the list of years for which the production model is run.

        Returns
        -------
        None
        """
        # Init
        base_fname = self._base_output_fname
        fname_parts = [self._trip_origin, self._zoning_system]

        self.pure_demand_paths = dict()
        self.fully_segmented_paths = dict()
        self.notem_segmented_paths = dict()

        for year in years:
            # Pure demand path
            fname = base_fname % (*fname_parts, self._pure_demand, year)
            self.pure_demand_paths[year] = os.path.join(export_path, fname)

            # Fully Segmented path
            fname = base_fname % (*fname_parts, self._fully_segmented, year)
            self.fully_segmented_paths[year] = os.path.join(export_path, fname)

            # NoTEM Segmented path
            fname = base_fname % (*fname_parts, self._notem_segmented, year)
            self.notem_segmented_paths[year] = os.path.join(export_path, fname)

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

    def _create_pure_demand_report_paths(self,
                                         report_path: nd.PathLike,
                                         years: List[int],
                                         ) -> None:
        """
        Creates pure demand report file paths for each of years

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
        paths = self._create_report_paths(report_path, years, self._pure_demand)
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


class NHBProductionModel:
    # Constants
    _trip_origin = 'hb'
    _zoning_system = 'msoa'
    _return_segmentation_name = 'hb_notem_output'

    # Segmentation names
    _pure_demand = 'pure_demand'
    _fully_segmented = 'fully_segmented'
    _notem_segmented = 'notem_segmented'

    # Define wanted columns
    _target_cols = {
        'land_use': ['msoa_zone_id', 'area_type'],
        'trip_rate': ['tfn_tt', 'tfn_at', 'p', 'trip_rate'],
        'm_tp': ['p', 'tfn_tt', 'tfn_at', 'm', 'tp', 'split'],
    }

    # Define segment renames needed
    _seg_rename = {
        'tfn_traveller_type': 'tfn_tt',
        'area_type': 'tfn_at',
    }

    # Define output fnames
    _base_output_fname = '%s_%s_%s_%d_dvec.pkl'
    _base_report_fname = '%s_%s_%d_%s.csv'

    def __init__(self,
                 hb_attractions: Dict[int, nd.PathLike],
                 land_use_paths: Dict[int, nd.PathLike],
                 nhb_trip_rates_path: str,
                 nhb_time_splits_path: str,
                 export_path: str,
                 constraint_paths: Dict[int, nd.PathLike] = None,
                 process_count: int = consts.PROCESS_COUNT
                 ):
        # TODO(BT): Document attributes
        # Validate inputs
        [ops.check_file_exists(x) for x in hb_attractions.values()]
        [ops.check_file_exists(x) for x in land_use_paths.values()]
        if constraint_paths is not None:
            [ops.check_file_exists(x) for x in constraint_paths.values()]
        ops.check_file_exists(nhb_trip_rates_path)
        ops.check_file_exists(nhb_time_splits_path)
        ops.check_path_exists(export_path)

        # Assign
        self.hb_attractions = hb_attractions
        self.land_use_paths = land_use_paths
        self.nhb_trip_rates_path = nhb_trip_rates_path
        self.nhb_time_splits_path = nhb_time_splits_path
        self.constraint_paths = constraint_paths
        self.export_path = export_path
        self.report_path = os.path.join(export_path, "Reports")
        self.process_count = process_count
        self.years = list(self.land_use_paths.keys())

        # Create paths
        du.create_folder(self.report_path, verbose=False)

        # Initialise Output paths
        # TODO(BT): Convert output paths into dictionaries
        #  something like: self.reports['pure_demand'][year]
        self._create_output_paths(self.export_path, self.years)
        self._create_pure_demand_report_paths(self.report_path, self.years)
        self._create_notem_segmented_report_paths(self.report_path, self.years)

    def run(self,
            export_pure_demand: bool = True,
            export_fully_segmented: bool = False,
            export_notem_segmentation: bool = True,
            export_reports: bool = True,
            verbose: bool = False,
            ) -> None:
        """
        Runs the NHB Production model.

        Completes the following steps for each year:
            - Reads in the land use population data given in the constructor.
            - Reads in the trip rates data given in the constructor.
            - Multiplies the population and trip rates on relevant segments,
              producing "pure demand".
            - Optionally writes out a pickled DVector of "pure demand" at
              self.pure_demand_out[year]
            - Optionally writes out a number of "pure demand" reports, if
              reports is True.
            - Reads in the mode-time splits given in the constructor.
            - Multiplies the "pure demand" and mode-time splits on relevant
              segments, producing "fully segmented demand".
            - Optionally writes out a pickled DVector of "fully segmented demand"
              at self.fully_segmented_paths[year] if export_fully_segmented
              is True.
            - Aggregates this demand into hb_notem_full_tfn segmentation,
              producing "notem segmented demand".
            - Optionally writes out a number of "notem segmented demand"
              reports, if reports is True.
            - Optionally writes out a pickled DVector of "notem segmented demand"
              at self.notem_segmented_paths[year] if export_notem_segmentation
              is True.
            - Finally, returns "notem segmented demand" as a DVector.

        Parameters
        ----------
        export_pure_demand:
            Whether to export the pure demand to disk or not.
            Will be written out to: self.pure_demand_out[year]

        export_fully_segmented:
            Whether to export the fully segmented demand to disk or not.
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
            "Starting NHB Production Model at: %s" % timing.get_datetime(),
            verbose=verbose
        )

        # Generate the productions for each year
        for year in self.years:
            # ## GENERATE PURE DEMAND ## #
            du.print_w_toggle("Loading the HB attraction data...", verbose=verbose)
            hb_attr_dvec = self._transform_attractions(year, verbose)

            du.print_w_toggle("Applying trip rates...", verbose=verbose)
            pure_demand = self._generate_productions(pop_dvec, verbose)

            if export_pure_demand:
                du.print_w_toggle("Exporting pure demand to disk...", verbose=verbose)
                pure_demand.to_pickle(self.pure_demand_paths[year])

            if export_reports:
                du.print_w_toggle(
                    "Exporting pure demand reports disk...\n"
                    "Total Productions for year %d: %.4f"
                    % (year, pure_demand.sum()),
                    verbose=verbose
                )

                tfn_agg_at_seg = nd.get_segmentation_level('pure_demand_reporting')
                self._write_reports(
                    dvec=pure_demand.aggregate(tfn_agg_at_seg),
                    segment_totals_path=self.pd_report_segment_paths[year],
                    ca_sector_path=self.pd_report_ca_sector_paths[year],
                    ie_sector_path=self.pd_report_ie_sector_paths[year],
                )

            # ## SPLIT PURE DEMAND BY MODE AND TIME ## #
            du.print_w_toggle("Splitting by mode and time...", verbose=verbose)
            fully_segmented = self._split_by_tp_and_mode(pure_demand)

            # Output productions before any aggregation
            if export_fully_segmented:
                du.print_w_toggle(
                    "Exporting fully segmented productions to disk...",
                    verbose=verbose,
                )
                fully_segmented.to_pickle(self.fully_segmented_paths[year])

            # ## AGGREGATE INTO RETURN SEGMENTATION ## #
            return_seg = nd.get_segmentation_level(self._return_segmentation_name)
            productions = fully_segmented.aggregate(
                out_segmentation=return_seg,
                split_tfntt_segmentation=True
            )

            if export_notem_segmentation:
                du.print_w_toggle(
                    "Exporting notem segmented demand to disk...",
                    verbose=verbose
                )
                productions.to_pickle(self.notem_segmented_paths[year])

            if export_reports:
                du.print_w_toggle(
                    "Exporting notem segmented reports disk...\n"
                    "Total Productions for year %d: %.4f"
                    % (year, productions.sum()),
                    verbose=verbose
                )

                self._write_reports(
                    dvec=productions,
                    segment_totals_path=self.notem_report_segment_paths[year],
                    ca_sector_path=self.notem_report_ca_sector_paths[year],
                    ie_sector_path=self.notem_report_ie_sector_paths[year],
                )

            # TODO: Bring in constraints (Validation)
            #  Output some audits of what demand was before and after control
            #  By segment.
            if self.constraint_paths is not None:
                raise NotImplemented(
                    "No code implemented to constrain productions."
                )

            # End timing
            end_time = timing.current_milli_time()
            du.print_w_toggle("Finished HB Production Model at: %s" % timing.get_datetime(),
                              verbose=verbose)
            du.print_w_toggle("HB Production Model took: %s"
                              % timing.time_taken(start_time, end_time), verbose=verbose)

    def _transform_attractions(self,
                               year: int,
                               verbose: bool,
                               ) -> nd.DVector:
        """
        Reads in the HB attractions and aggregates away time period

        Parameters
        ----------
        year:
            The year to get HB attractions data for.

        verbose:
            Passed into the DVector.

        Returns
        -------
        pop_dvec:
            Returns the population Dvector
        """
        # Define the zoning and segmentations we want to use
        msoa_zoning = nd.get_zoning_system('msoa')
        hb_attr_seg = nd.get_segmentation_level('hb_notem_without_tp')
        hb_attr_notem = ops.read_pickle(self.hb_attractions[year])
        hb_attr = hb_attr_notem.aggregate(hb_attr_seg)
        hb_attr_df = hb_attr.to_df()

        # Read the land use data corresponding to the year
        pop = du.safe_read_csv(
            file_path=self.land_use_paths[year],
            usecols=self._target_cols['land_use']
        )
        hb_attr_at_df = pd.merge(hb_attr_df, pop, on="zone", how="left")
        
        # Instantiate
        return nd.DVector(
            zoning_system=msoa_zoning,
            segmentation=pop_seg,
            import_data=pop.rename(columns=self._seg_rename),
            zone_col="msoa_zone_id",
            val_col="people",
            verbose=verbose,
        )

    def _generate_productions(self,
                              population: nd.DVector,
                              verbose: bool,
                              ) -> nd.DVector:
        """
        Applies trip rate split on the given HB productions

        Parameters
        ----------
        population:
            Dvector containing the population.

        verbose:
            Whether to print a progress bar while applying the splits or not

        Returns
        -------
        pure_demand:
            Returns the product of population and trip rate Dvector
            ie., pure demand
        """

        # Define the zoning and segmentations we want to use
        pure_demand_seg = nd.get_segmentation_level('pure_demand')

        # Reading trip rates
        du.print_w_toggle("Reading in files...", verbose=verbose)
        trip_rates = du.safe_read_csv(self.trip_rates_path, usecols=self._target_cols['trip_rate'])

        # ## CREATE THE TRIP RATES DVEC ## #
        du.print_w_toggle("Creating trip rates DVec...", verbose=verbose)

        # Instantiate
        trip_rates_dvec = nd.DVector(
            zoning_system=None,
            segmentation=pure_demand_seg,
            import_data=trip_rates.rename(columns=self._seg_rename),
            val_col="trip_rate",
            verbose=verbose,
        )
        # ## MULTIPLY TOGETHER ## #
        return population * trip_rates_dvec

    @staticmethod
    def _write_reports(dvec: nd.DVector,
                       segment_totals_path: nd.PathLike,
                       ca_sector_path: nd.PathLike,
                       ie_sector_path: nd.PathLike,
                       ) -> None:
        """
        Writes segment, CA sector, and IE sector reports to disk

        Parameters
        ----------
        dvec:
            The Dvector to write the reports for

        segment_totals_path:
            Path to write the segment totals report to

        ca_sector_path:
            Path to write the CA sector report to

        ie_sector_path:
            Path to write the IE sector report to

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

    def _split_by_tp_and_mode(self,
                              pure_demand: nd.DVector,
                              ) -> nd.DVector:
        """
        Applies time period and mode splits to the given pure demand.

        Parameters
        ----------
        pure_demand:
            Dvector containing the pure demand to split.

        Returns
        -------
        full_segmented_demand:
            A DVector containing pure_demand split by mode and time.
        """
        # Define the segmentation we want to use
        m_tp_pure_demand_seg = nd.get_segmentation_level('full_tfntt_tfnat')
        notem_seg = nd.get_segmentation_level('full_tfntt')

        # Create the mode-time splits DVector
        mode_time_splits = pd.read_csv(
            self.mode_time_splits_path,
            usecols=self._target_cols['m_tp']
        )

        mode_time_splits_dvec = nd.DVector(
            zoning_system=None,
            segmentation=m_tp_pure_demand_seg,
            import_data=mode_time_splits,
            val_col="split",
        )

        return pure_demand.multiply_and_aggregate(
            other=mode_time_splits_dvec,
            out_segmentation=notem_seg,
        )

    def _create_output_paths(self,
                             export_path: nd.PathLike,
                             years: List[int],
                             ) -> None:
        """
        Creates output file names for pure demand, fully segmented and aggregated
        HB production outputs for the list of years

        Parameters
        ----------
        export_path:
            Location where the output files are to be created.

        years:
            Contains the list of years for which the production model is run.

        Returns
        -------
        None
        """
        # Init
        base_fname = self._base_output_fname
        fname_parts = [self._trip_origin, self._zoning_system]

        self.pure_demand_paths = dict()
        self.fully_segmented_paths = dict()
        self.notem_segmented_paths = dict()

        for year in years:
            # Pure demand path
            fname = base_fname % (*fname_parts, self._pure_demand, year)
            self.pure_demand_paths[year] = os.path.join(export_path, fname)

            # Fully Segmented path
            fname = base_fname % (*fname_parts, self._fully_segmented, year)
            self.fully_segmented_paths[year] = os.path.join(export_path, fname)

            # NoTEM Segmented path
            fname = base_fname % (*fname_parts, self._notem_segmented, year)
            self.notem_segmented_paths[year] = os.path.join(export_path, fname)

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

    def _create_pure_demand_report_paths(self,
                                         report_path: nd.PathLike,
                                         years: List[int],
                                         ) -> None:
        """
        Creates pure demand report file paths for each of years

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
        paths = self._create_report_paths(report_path, years, self._pure_demand)
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
