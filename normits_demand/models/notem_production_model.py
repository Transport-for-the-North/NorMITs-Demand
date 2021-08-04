# -*- coding: utf-8 -*-
"""
Created on: Friday June 18th 2021
Updated on: Wednesday July 21st 2021

Original author: Nirmal Kumar
Last update made by: Ben Taylor
Other updates made by: Ben Taylor

File purpose:
Production Models for NoTEM
"""
# Allow class self type hinting
from __future__ import annotations

# Builtins
import os

from typing import Dict
from typing import List

# Third party imports
import pandas as pd

# local imports
import normits_demand as nd

from normits_demand import efs_constants as consts

from normits_demand.utils import timing
from normits_demand.utils import file_ops
from normits_demand.utils import general as du
from normits_demand.utils import pandas_utils as pd_utils

from normits_demand.pathing import HBProductionModelPaths
from normits_demand.pathing import NHBProductionModelPaths
from normits_demand.pathing import WriteReports


class HBProductionModel(HBProductionModelPaths, WriteReports):
    """The Home-Based Production Model of NoTEM

    The production model can be ran by calling the class run() method.

    Attributes
    ----------
    land_use_paths: Dict[int, nd.PathLike]:
        Dictionary of {year: land_use_employment_data} pairs. As passed
        into the constructor.

    trip_rates_path: str
        The path to the production trip rates. As passed into the constructor.

    mode_time_splits_path: str
        The path to production mode-time splits. As passed into the
        constructor.

    constraint_paths: Dict[int, nd.PathLike]
        Dictionary of {year: constraint_path} pairs. As passed into the
        constructor.

    process_count: int
        The number of processes to create in the Pool. As passed into the
        constructor.

    years: List[int]
        A list of years that the model will run for. Derived from the keys of
        land_use_paths

    See HBProductionModelPaths for documentation on:
        "path_years, export_home, report_home, export_paths, report_paths"
    """
    # Constants
    _return_segmentation_name = 'hb_notem_output'

    # Define wanted columns
    _target_col_dtypes = {
        'land_use': {
            'msoa_zone_id': str,
            'area_type': int,
            'tfn_traveller_type': int,
            'people': float
        },
        'trip_rate': {
            'tfn_tt': int,
            'tfn_at': int,
            'p': int,
            'trip_rate': float
        },
        'm_tp': {
            'p': int,
            'tfn_tt': int,
            'tfn_at': int,
            'm': int,
            'tp': int,
            'split': float
        },
    }

    # Define segment renames needed
    _seg_rename = {
        'tfn_traveller_type': 'tfn_tt',
        'area_type': 'tfn_at',
    }

    def __init__(self,
                 land_use_paths: Dict[int, nd.PathLike],
                 trip_rates_path: str,
                 mode_time_splits_path: str,
                 export_home: str,
                 constraint_paths: Dict[int, nd.PathLike] = None,
                 process_count: int = consts.PROCESS_COUNT
                 ) -> None:
        """
        Sets up and validates arguments for the Production model.

        Parameters
        ----------
        land_use_paths:
            Dictionary of {year: land_use_employment_data} pairs.

        trip_rates_path:
            The path to the production trip rates.
            Should have the columns as defined in:
            HBProductionModel._target_cols['trip_rate']

        mode_time_splits_path:
            The path to production mode-time splits.
            Should have the columns as defined in:
            HBProductionModel._target_cols['m_tp']

        export_home:
            Path to export production outputs.

        constraint_paths:
            Dictionary of {year: constraint_path} pairs.
            Must contain the same keys as land_use_paths, but it can contain
            more (any extras will be ignored).
            If set - will be used to constrain the productions - a report will
            be written before and after.

        process_count:
            The number of processes to create in the Pool. Typically this
            should not exceed the number of cores available.
            Defaults to consts.PROCESS_COUNT.
        """
        # Check that the paths we need exist!
        [file_ops.check_file_exists(x) for x in land_use_paths.values()]
        file_ops.check_file_exists(trip_rates_path)
        file_ops.check_file_exists(mode_time_splits_path)
        if constraint_paths is not None:
            [file_ops.check_file_exists(x) for x in constraint_paths.values()]

        # Validate that we have data for all the years we're running for
        for year in land_use_paths.keys():
            if constraint_paths is not None:
                if year not in constraint_paths.keys():
                    raise ValueError(
                        "Year %d found in land_use_paths\n"
                        "But not found in constraint_paths"
                        % year
                    )

        # Assign
        self.land_use_paths = land_use_paths
        self.trip_rates_path = trip_rates_path
        self.mode_time_splits_path = mode_time_splits_path
        self.constraint_paths = constraint_paths
        self.process_count = process_count
        self.years = list(self.land_use_paths.keys())

        # Make sure the reports paths exists
        report_home = os.path.join(export_home, "Reports")
        file_ops.create_folder(report_home)

        # Build the output paths
        super().__init__(
            path_years=self.years,
            export_home=export_home,
            report_home=report_home,
        )

    def run(self,
            export_pure_demand: bool = False,
            export_fully_segmented: bool = False,
            export_notem_segmentation: bool = False,
            export_reports: bool = False,
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
              self.export_paths.pure_demand[year]
            - Optionally writes out a number of "pure demand" reports, if
              reports is True.
            - Reads in the mode-time splits given in the constructor.
            - Multiplies the "pure demand" and mode-time splits on relevant
              segments, producing "fully segmented demand".
            - Optionally writes out a pickled DVector of "fully segmented demand"
              at self.export_paths.fully_segmented[year] if export_fully_segmented
              is True.
            - Aggregates this demand into hb_notem_full_tfn segmentation,
              producing "notem segmented demand".
            - Optionally writes out a number of "notem segmented demand"
              reports, if reports is True.
            - Optionally writes out a pickled DVector of "notem segmented demand"
              at self.export_paths.notem_segmented[year] if export_notem_segmentation
              is True.
            - Finally, returns "notem segmented demand" as a DVector.

        Parameters
        ----------
        export_pure_demand:
            Whether to export the pure demand to disk or not.
            Will be written out to: self.export_paths.pure_demand[year]

        export_fully_segmented:
            Whether to export the fully segmented demand to disk or not.
            Will be written out to: self.export_paths.fully_segmented[year]

        export_notem_segmentation:
            Whether to export the notem segmented demand to disk or not.
            Will be written out to: self.export_paths.notem_segmented[year]

        export_reports:
            Whether to output reports while running. All reports will be
            written out to self.report_home.

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
            year_start_time = timing.current_milli_time()

            # ## GENERATE PURE DEMAND ## #
            du.print_w_toggle("Loading the population data...", verbose=verbose)
            pop_dvec = self._read_land_use_data(year, verbose)

            du.print_w_toggle("Applying trip rates...", verbose=verbose)
            pure_demand = self._generate_productions(pop_dvec, verbose)

            if export_pure_demand:
                du.print_w_toggle("Exporting pure demand to disk...", verbose=verbose)
                pure_demand.to_pickle(self.export_paths.pure_demand[year])

            if export_reports:
                du.print_w_toggle(
                    "Exporting pure demand reports to disk...",
                    verbose=verbose
                )

                tfn_agg_at_seg = nd.get_segmentation_level('pure_demand_reporting')
                pure_demand_paths = self.report_paths.pure_demand
                self.write_reports(
                    dvec=pure_demand.aggregate(tfn_agg_at_seg),
                    segment_totals_path=pure_demand_paths.segment_total[year],
                    ca_sector_path=pure_demand_paths.ca_sector[year],
                    ie_sector_path=pure_demand_paths.ie_sector[year],
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
                fully_segmented.to_pickle(self.export_paths.fully_segmented[year])

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
                productions.to_pickle(self.export_paths.notem_segmented[year])

            if export_reports:
                du.print_w_toggle(
                    "Exporting notem segmented reports to disk...",
                    verbose=verbose
                )

                notem_segmented_paths = self.report_paths.notem_segmented
                self.write_reports(
                    dvec=productions,
                    segment_totals_path=notem_segmented_paths.segment_total[year],
                    ca_sector_path=notem_segmented_paths.ca_sector[year],
                    ie_sector_path=notem_segmented_paths.ie_sector[year],
                )

            # TODO: Bring in constraints (Validation)
            #  Output some audits of what demand was before and after control
            #  By segment.
            if self.constraint_paths is not None:
                raise NotImplemented(
                    "No code implemented to constrain productions."
                )

            # Print timing stats for the year
            year_end_time = timing.current_milli_time()
            time_taken = timing.time_taken(year_start_time, year_end_time)
            du.print_w_toggle(
                "HB Productions in year %s took: %s\n" % (year, time_taken),
                verbose=verbose
            )

        # End timing
        end_time = timing.current_milli_time()
        time_taken = timing.time_taken(start_time, end_time)
        du.print_w_toggle(
            "HB Production Model took: %s\n"
            "Finished at: %s" % (time_taken, end_time),
            verbose=verbose
        )

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
        pop = file_ops.read_df(
            path=self.land_use_paths[year],
            find_similar=True,
        )
        pop = pd_utils.reindex_cols(pop, self._target_col_dtypes['land_use'].keys())
        for col, dt in self._target_col_dtypes['land_use'].items():
            pop[col] = pop[col].astype(dt)

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
        trip_rates = du.safe_read_csv(
            self.trip_rates_path,
            usecols=self._target_col_dtypes['trip_rate'].keys(),
            dtype=self._target_col_dtypes['trip_rate'],
        )

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
            usecols=self._target_col_dtypes['m_tp'].keys(),
            dtype=self._target_col_dtypes['m_tp'],
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


class NHBProductionModel(NHBProductionModelPaths, WriteReports):
    """The Non Home-Based Production Model of NoTEM

        The production model can be ran by calling the class run() method.

        Attributes
        ----------
        hb_attractions_paths:
            Dictionary of {year: notem_segmented_HB_attractions_data} pairs.
            As passed into the constructor.

        land_use_paths: Dict[int, nd.PathLike]:
            Dictionary of {year: land_use_employment_data} pairs. As passed
            into the constructor.

        nhb_trip_rates_path: str
            The path to the NHB production trip rates. As passed into the constructor.

        nhb_time_splits_path: str
            The path to the NHB production time splits. As passed into the
            constructor.

        constraint_paths: Dict[int, nd.PathLike]
            Dictionary of {year: constraint_path} pairs. As passed into the
            constructor.

        process_count: int
            The number of processes to create in the Pool. As passed into the
            constructor.

        years: List[int]
            A list of years that the model will run for. Derived from the keys of
            land_use_paths

        See NHBProductionModelPaths for documentation on:
            "path_years, export_home, report_home, export_paths, report_paths"
        """

    # Define wanted columns
    _target_col_dtypes = {
        'land_use': {
            'msoa_zone_id': str,
            'area_type': int
        },
        'nhb_trip_rate': {
            'nhb_p': int,
            'nhb_m': int,
            'p': int,
            'm': int,
            'tfn_at': int,
            'nhb_trip_rate': float
        },
        'tp': {
            'nhb_p': int,
            'nhb_m': int,
            'tfn_at': int,
            'tp': int,
            'split': float
        },
    }

    # Define segment renames needed
    _seg_rename = {
        'area_type': 'tfn_at',
    }

    def __init__(self,
                 hb_attractions_paths: Dict[int, nd.PathLike],
                 land_use_paths: Dict[int, nd.PathLike],
                 nhb_trip_rates_path: str,
                 nhb_time_splits_path: str,
                 export_home: str,
                 constraint_paths: Dict[int, nd.PathLike] = None,
                 process_count: int = consts.PROCESS_COUNT
                 ) -> None:
        """
        Sets up and validates arguments for the NHB Production model.

        Parameters
        ----------
        hb_attractions_paths:
            Dictionary of {year: notem_segmented_HB_attractions_data} pairs.
            These paths should come from nd.HBAttraction model and should
            be pickled Dvector paths.

        land_use_paths:
            Dictionary of {year: land_use_population_data} pairs.

        nhb_trip_rates_path:
            The path to the NHB production trip rates.
            Should have the columns as defined in:
            NHBProductionModel._target_cols['nhb_trip_rate']

        nhb_time_splits_path:
            The path to NHB production time split.
            Should have the columns as defined in:
            NHBProductionModel._target_cols['tp']

        export_home:
            Path to export NHB Production outputs.

        constraint_paths:
            Dictionary of {year: constraint_path} pairs.
            Must contain the same keys as land_use_paths, but it can contain
            more (any extras will be ignored).
            If set - will be used to constrain the productions - a report will
            be written before and after.

        process_count:
            The number of processes to create in the Pool. Typically this
            should not exceed the number of cores available.
            Defaults to consts.PROCESS_COUNT.
        """
        # Check that the paths we need exist!
        [file_ops.check_file_exists(x) for x in hb_attractions_paths.values()]
        [file_ops.check_file_exists(x) for x in land_use_paths.values()]
        file_ops.check_file_exists(nhb_trip_rates_path)
        file_ops.check_file_exists(nhb_time_splits_path)

        if constraint_paths is not None:
            [file_ops.check_file_exists(x) for x in constraint_paths.values()]

        # Validate that we have data for all the years we're running for
        for year in hb_attractions_paths.keys():
            if year not in land_use_paths.keys():
                raise ValueError(
                    "Year %d found given attractions: hb_attractions_paths\n"
                    "But not found in land_use_paths"
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
        self.hb_attractions = hb_attractions_paths
        self.land_use_paths = land_use_paths
        self.nhb_trip_rates_path = nhb_trip_rates_path
        self.nhb_time_splits_path = nhb_time_splits_path
        self.constraint_paths = constraint_paths
        self.process_count = process_count
        self.years = list(self.hb_attractions.keys())

        # Make sure the reports paths exists
        report_home = os.path.join(export_home, "Reports")
        file_ops.create_folder(report_home)

        # Build the output paths
        super().__init__(
            path_years=self.years,
            export_home=export_home,
            report_home=report_home,
        )

    def run(self,
            export_nhb_pure_demand: bool = False,
            export_fully_segmented: bool = False,
            export_notem_segmentation: bool = False,
            export_reports: bool = False,
            verbose: bool = False,
            ) -> None:
        """
        Runs the NHB Production model.

        Completes the following steps for each year:
            - Reads in the notem segmented HB attractions compressed pickle
              given in the constructor.
            - Removes time period segmentation from the above data.
            - Reads in the land use population data given in the constructor,
              extracts the mapping of msoa_zone_id to tfn_at.
            - Reads in the NHB trip rates data given in the constructor.
            - Multiplies the HB attractions and NHB trip rates on relevant segments,
              producing "pure NHB demand".
            - Optionally writes out a pickled DVector of "pure NHB demand" at
              self.export_paths.pure_demand[year]
            - Optionally writes out a number of "pure demand" reports, if
              reports is True.
            - Reads in the time splits given in the constructor.
            - Multiplies the "pure NHB demand" and time splits on relevant
              segments, producing "fully segmented demand".
            - Optionally writes out a pickled DVector of "fully segmented demand"
              at self.export_paths.fully_segmented[year] if export_fully_segmented
              is True.
            - Renames nhb_p and nhb_m as p and m respectively,
              producing "notem segmented demand".
            - Optionally writes out a number of "notem segmented demand"
              reports, if reports is True.
            - Optionally writes out a pickled DVector of "notem segmented demand"
              at self.export_paths.notem_segmented[year] if export_notem_segmentation
              is True.

        Parameters
        ----------
        export_nhb_pure_demand:
            Whether to export the pure NHB demand to disk or not.
            Will be written out to: self.export_paths.pure_demand[year]

        export_fully_segmented:
            Whether to export the fully segmented demand to disk or not.
            Will be written out to: self.export_paths.fully_segmented[year]

        export_notem_segmentation:
            Whether to export the notem segmented demand to disk or not.
            Will be written out to: self.export_paths.notem_segmented[year]

        export_reports:
            Whether to output reports while running. All reports will be
            written out to self.report_home.

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

        # Generate the nhb productions for each year
        for year in self.years:
            year_start_time = timing.current_milli_time()

            # ## GENERATE PURE DEMAND ## #
            du.print_w_toggle("Loading the HB attraction data...", verbose=verbose)
            hb_attr_dvec = self._transform_attractions(year, verbose)

            du.print_w_toggle("Applying trip rates...", verbose=verbose)
            pure_nhb_demand = self._generate_nhb_productions(hb_attr_dvec, verbose)

            if export_nhb_pure_demand:
                du.print_w_toggle("Exporting NHB pure demand to disk...", verbose=verbose)
                pure_nhb_demand.to_pickle(self.export_paths.pure_demand[year])

            if export_reports:
                du.print_w_toggle(
                    "Exporting NHB pure demand reports to disk...\n",
                    verbose=verbose
                )

                tfn_agg_at_seg = nd.get_segmentation_level('pure_nhb_demand_reporting')
                pure_demand_paths = self.report_paths.pure_demand
                self.write_reports(
                    dvec=pure_nhb_demand.aggregate(tfn_agg_at_seg),
                    segment_totals_path=pure_demand_paths.segment_total[year],
                    ca_sector_path=pure_demand_paths.ca_sector[year],
                    ie_sector_path=pure_demand_paths.ie_sector[year],
                )

            # ## SPLIT NHB PURE DEMAND BY TIME ## #
            du.print_w_toggle("Splitting by time...", verbose=verbose)
            fully_segmented = self._split_by_tp(pure_nhb_demand)

            if export_fully_segmented:
                du.print_w_toggle(
                    "Exporting fully segmented demand to disk...",
                    verbose=verbose
                )
                fully_segmented.to_pickle(self.export_paths.fully_segmented[year])

            # Renaming
            notem_segmented = self._rename(fully_segmented, verbose)

            if export_notem_segmentation:
                du.print_w_toggle(
                    "Exporting notem segmented demand to disk...",
                    verbose=verbose
                )
                notem_segmented.to_pickle(self.export_paths.notem_segmented[year])

            if export_reports:
                du.print_w_toggle(
                    "Exporting notem segmented reports to disk...\n",
                    verbose=verbose
                )

                notem_segmented_paths = self.report_paths.notem_segmented
                self.write_reports(
                    dvec=notem_segmented,
                    segment_totals_path=notem_segmented_paths.segment_total[year],
                    ca_sector_path=notem_segmented_paths.ca_sector[year],
                    ie_sector_path=notem_segmented_paths.ie_sector[year],
                )

            # TODO: Bring in constraints (Validation)
            #  Output some audits of what demand was before and after control
            #  By segment.
            if self.constraint_paths is not None:
                raise NotImplemented(
                    "No code implemented to constrain productions."
                )

            # Print timing stats for the year
            year_end_time = timing.current_milli_time()
            time_taken = timing.time_taken(year_start_time, year_end_time)
            du.print_w_toggle(
                "HB Productions in year %s took: %s\n" % (year, time_taken),
                verbose=verbose
            )

        # End timing
        end_time = timing.current_milli_time()
        time_taken = timing.time_taken(start_time, end_time)
        du.print_w_toggle(
            "NHB Production Model took: %s\n"
            "Finished at: %s" % (time_taken, end_time),
            verbose=verbose
        )

    def _transform_attractions(self,
                               year: int,
                               verbose: bool,
                               ) -> nd.DVector:
        """
        Removes time period and adds tfn_at to HB attraction DVector

        - Reads the HB attractions compressed pickle.
        - Removes time period from segmentation.
        - Extracts the mapping of msoa_zone_id to tfn_at from land use.
        - Adds tfn_at to the HB attraction and returns its DVector.

        Parameters
        ----------
        year:
            The year to get HB attractions data for.

        verbose:
            Passed into the DVector.

        Returns
        -------
        hb_attr_dvec:
            Returns the HB attraction Dvector with tfn_at.
        """
        # Define the zoning and segmentations we want to use
        msoa_zoning = nd.get_zoning_system('msoa')
        hb_notem_no_output_seg = nd.get_segmentation_level('hb_notem_output_no_tp')
        hb_notem_tfnat_seg = nd.get_segmentation_level('notem_hb_tfnat_p_m_g_soc_ns_ca')

        # ## READ IN AND VALIDATE THE LAND USE DATA ## #
        # Reading the land use data
        pop = du.safe_read_csv(
            file_path=self.land_use_paths[year],
            usecols=self._target_col_dtypes['land_use'].keys(),
            dtype=self._target_col_dtypes['land_use'],
        )
        pop.columns = ['zone', 'tfn_at']
        pop = pop.drop_duplicates()

        # Set up for validations
        pop_zones = set(pop['zone'].unique().tolist())
        unique_zones = set(msoa_zoning.unique_zones)

        # Check that we have all the zones we need
        missing_zones = unique_zones - pop_zones
        if len(missing_zones) > 0:
            raise ValueError(
                "The given land use data does not have tfn_at data for all "
                "MSOAs!\n"
                "Missing zones: %s"
                % missing_zones
            )

        # Check that we don't have any extra zones
        extra_zones = pop_zones - unique_zones
        if len(extra_zones) > 0:
            raise ValueError(
                "The given land use data contains zones data for zones not in "
                "the MSOA zoning system. Not sure how to proceed.\n"
                "Extra zones: %s"
                % extra_zones
            )

        # ## CONVERT THE ATTRACTIONS INTO DESIRED FORMAT ## #
        # Read the notem segmented compressed pickle
        hb_attr_notem = nd.from_pickle(self.hb_attractions[year])

        # Remove time period from segmentation
        hb_attr = hb_attr_notem.aggregate(hb_notem_no_output_seg)
        hb_attr_df = hb_attr.to_df()

        # Add tfn area type to the segmentation
        hb_attr_at_df = pd.merge(hb_attr_df, pop, on="zone", how="left")

        # Instantiate
        return nd.DVector(
            zoning_system=msoa_zoning,
            segmentation=hb_notem_tfnat_seg,
            import_data=hb_attr_at_df.rename(columns=self._seg_rename),
            zone_col="zone",
            val_col="value",
            verbose=verbose,
        )

    def _generate_nhb_productions(self,
                                  hb_attractions: nd.DVector,
                                  verbose: bool,
                                  ) -> nd.DVector:
        """
        Applies NHB trip rates to hb_attractions

        Parameters
        ----------
        hb_attractions:
            Dvector containing the data to apply the trip rates to.

        verbose:
            Whether to print a progress bar while applying the splits or not

        Returns
        -------
        pure_NHB_demand:
            Returns the product of HB attractions and NHB trip rate Dvector
            ie., pure NHB demand
        """

        # Define the zoning and segmentations we want to use
        nhb_trip_rate_seg = nd.get_segmentation_level('nhb_trip_rate')

        # Reading NHB trip rates
        du.print_w_toggle("Reading in files...", verbose=verbose)
        trip_rates = du.safe_read_csv(
            file_path=self.nhb_trip_rates_path,
            usecols=self._target_col_dtypes['nhb_trip_rate'].keys(),
            dtype=self._target_col_dtypes['nhb_trip_rate'],
        )

        # Create the NHB Trip Rates DVec
        trip_rates_dvec = nd.DVector(
            zoning_system=None,
            segmentation=nhb_trip_rate_seg,
            import_data=trip_rates.rename(columns=self._seg_rename),
            val_col="nhb_trip_rate",
            verbose=verbose,
        )

        # Multiply
        return hb_attractions * trip_rates_dvec

    def _split_by_tp(self,
                     pure_nhb_demand: nd.DVector,
                     ) -> nd.DVector:
        """
        Applies time period splits to the given pure nhb demand.

        Parameters
        ----------
        pure_nhb_demand:
            Dvector containing the pure nhb demand to split.

        Returns
        -------
        full_segmented_demand:
            A DVector containing pure_demand split by time.
        """
        # Define the segmentation we want to use
        tp_pure_nhb_demand_seg = nd.get_segmentation_level('nhb_tfnat_p_m_tp')
        fully_seg = nd.get_segmentation_level('full_nhb')

        # Read the time splits factor
        time_splits = pd.read_csv(
            self.nhb_time_splits_path,
            usecols=self._target_col_dtypes['tp'].keys(),
            dtype=self._target_col_dtypes['tp'],
        )

        # Instantiate
        time_splits_dvec = nd.DVector(
            zoning_system=None,
            segmentation=tp_pure_nhb_demand_seg,
            import_data=time_splits,
            val_col="split",
        )

        # Multiply together #
        return pure_nhb_demand.multiply_and_aggregate(
            other=time_splits_dvec,
            out_segmentation=fully_seg,
        )

    @staticmethod
    def _rename(full_segmentation: nd.DVector,
                verbose: bool
                ) -> nd.DVector:
        """
        Renames nhb_p and nhb_m as m and p respectively in full segmentation

        Parameters
        ----------
        full_segmentation:
            fully segmented NHB productions containing nhb_p and nhb_m as column names

        verbose:
            Whether to print messages while running this module

        Returns
        -------
        notem_segmented:
            Returns the notem segmented NHB production DVector
        """

        # Define the zoning and segmentations we want to use
        nhb_prod_seg = nd.get_segmentation_level('nhb_notem_output')

        du.print_w_toggle("Renaming...", verbose=verbose)

        # Rename columns
        return full_segmentation.aggregate(nhb_prod_seg)
