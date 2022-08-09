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
import dataclasses
import os
import pathlib
import warnings

from typing import Dict, List, Optional

# Third party imports
import pandas as pd

# local imports
import normits_demand as nd
from normits_demand import constants as consts

from normits_demand.utils import timing
from normits_demand.utils import file_ops
from normits_demand.utils import general as du
from normits_demand.utils import pandas_utils as pd_utils

from normits_demand.pathing import HBProductionModelPaths
from normits_demand.pathing import NHBProductionModelPaths


class HBProductionModel(HBProductionModelPaths):
    _log_fname = "HBProductionModel_log.log"
    """The Home-Based Production Model of NoTEM

    The production model can be ran by calling the class run() method.

    Attributes
    ----------
    population_paths: Dict[int, nd.PathLike]:
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
    __version__ = nd.__version__
    _return_segmentation_name = 'notem_hb_output'

    # Define wanted columns
    _target_col_dtypes = {
        'pop': {
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
                 population_paths: Dict[int, nd.PathLike],
                 trip_rates_path: str,
                 mode_time_splits_path: str,
                 export_home: str,
                 constraint_paths: Dict[int, nd.PathLike] = None,
                 process_count: int = consts.PROCESS_COUNT,
                 trip_end_adjustments: Optional[List[TripEndAdjustmentFactors]] = None,
                 ) -> None:
        """
        Sets up and validates arguments for the Production model.

        Parameters
        ----------
        population_paths:
            Dictionary of {year: population_data} pairs.
            HBProductionModel._target_col_dtypes['pop']

        trip_rates_path:
            The path to the production trip rates.
            Should have the columns as defined in:
            HBProductionModel._target_col_dtypes['trip_rate']

        mode_time_splits_path:
            The path to production mode-time splits.
            Should have the columns as defined in:
            HBProductionModel._target_col_dtypes['m_tp']

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

        trip_end_adjustments: List[TripEndAdjustmentFactors], optional
            List of all adjustment factors to apply to the trip ends. Adjustments
            are applied one after another at to the productions in the output
            segmentation.
        """
        # Check that the paths we need exist!
        [file_ops.check_file_exists(x, find_similar=True) for x in population_paths.values()]
        file_ops.check_file_exists(trip_rates_path, find_similar=True)
        file_ops.check_file_exists(mode_time_splits_path, find_similar=True)
        if constraint_paths is not None:
            [file_ops.check_file_exists(x, find_similar=True) for x in constraint_paths.values()]

        # Validate that we have data for all the years we're running for
        for year in population_paths.keys():
            if constraint_paths is not None:
                if year not in constraint_paths.keys():
                    raise ValueError(
                        "Year %d found in land_use_paths\n"
                        "But not found in constraint_paths"
                        % year
                    )

        # Assign
        self.population_paths = population_paths
        self.trip_rates_path = trip_rates_path
        self.mode_time_splits_path = mode_time_splits_path
        self.constraint_paths = constraint_paths
        self.process_count = process_count
        self.years = list(self.population_paths.keys())
        self.adjustment_factors = trip_end_adjustments

        # Make sure the reports paths exists
        report_home = os.path.join(export_home, "Reports")
        file_ops.create_folder(report_home)

        # Build the output paths
        super().__init__(
            path_years=self.years,
            export_home=export_home,
            report_home=report_home,
        )
        # Create a logger
        logger_name = "%s.%s" % (nd.get_package_logger_name(), self.__class__.__name__)
        log_file_path = os.path.join(self.export_home, self._log_fname)
        self._logger = nd.get_logger(
            logger_name=logger_name,
            log_file_path=log_file_path,
            instantiate_msg="Initialised HB Production Model",
        )

    def run(self,
            export_pure_demand: bool = False,
            export_fully_segmented: bool = False,
            export_notem_segmentation: bool = True,
            export_reports: bool = True,
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
            - Aggregates this demand into self._return_segmentation_name segmentation,
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

        Returns
        -------
        None
        """
        # Initialise timing

        start_time = timing.current_milli_time()
        self._logger.info("Starting HB Production Model")

        # Generate the productions for each year
        for year in self.years:
            year_start_time = timing.current_milli_time()
            # ## GENERATE PURE DEMAND ## #
            self._logger.info("Loading the population data")
            pop_dvec = self._read_land_use_data(year)

            self._logger.info("Applying trip rates")
            pure_demand = self._generate_productions(pop_dvec)

            if export_pure_demand:
                self._logger.info("Exporting pure demand to disk")
                pure_demand.save(self.export_paths.pure_demand[year])

            if export_reports:
                self._logger.info("Exporting pure demand reports to disk")
                report_seg = nd.get_segmentation_level('notem_hb_productions_pure_report')
                pure_demand_paths = self.report_paths.pure_demand
                pure_demand.aggregate(report_seg).write_sector_reports(
                    segment_totals_path=pure_demand_paths.segment_total[year],
                    ca_sector_path=pure_demand_paths.ca_sector[year],
                    ie_sector_path=pure_demand_paths.ie_sector[year],
                )

            # ## SPLIT PURE DEMAND BY MODE AND TIME ## #
            self._logger.info("Splitting by mode and time")
            fully_segmented = self._split_by_tp_and_mode(pure_demand)

            # ## PRODUCTIONS TOTAL CHECK ## #
            if not pure_demand.sum_is_close(fully_segmented):
                msg = (
                    "The production totals before and after mode time split are not same.\n"
                    "Expected %f\n"
                    "Got %f"
                    % (pure_demand.sum(), fully_segmented.sum())
                )
                self._logger.warning(msg)
                warnings.warn(msg)

            # Output productions before any aggregation
            if export_fully_segmented:
                self._logger.info("Exporting fully segmented productions to disk.")
                fully_segmented.save(self.export_paths.fully_segmented[year])

            # ## AGGREGATE INTO RETURN SEGMENTATION ## #
            return_seg = nd.get_segmentation_level(self._return_segmentation_name)
            productions = fully_segmented.aggregate(
                out_segmentation=return_seg,
                split_tfntt_segmentation=True
            )

            if self.adjustment_factors is not None:
                self._logger.info("Exporting pre-adjustment notem segmented demand to disk")
                path = pathlib.Path(self.export_paths.notem_segmented[year])
                productions.save(
                    path.with_name(path.stem + f"_pre-adjustment{''.join(path.suffixes)}")
                )

                productions = self._trip_end_adjustment(productions)

            if export_notem_segmentation:
                self._logger.info("Exporting notem segmented demand to disk")
                productions.save(self.export_paths.notem_segmented[year])

            if export_reports:
                self._logger.info("Exporting notem segmented reports to disk")
                notem_segmented_paths = self.report_paths.notem_segmented
                productions.write_sector_reports(
                    segment_totals_path=notem_segmented_paths.segment_total[year],
                    ca_sector_path=notem_segmented_paths.ca_sector[year],
                    ie_sector_path=notem_segmented_paths.ie_sector[year],
                    lad_report_path=notem_segmented_paths.lad_report[year],
                    lad_report_seg=nd.get_segmentation_level('hb_p_m_tp_week'),
                )

            # TODO: Bring in constraints (Validation)
            #  Output some audits of what demand was before and after control
            #  By segment.
            if self.constraint_paths is not None:
                msg = "No code implemented to constrain productions"
                self._logger.error(msg)
                raise NotImplementedError(msg)

            # Print timing stats for the year
            year_end_time = timing.current_milli_time()
            time_taken = timing.time_taken(year_start_time, year_end_time)
            self._logger.info("HB Productions in year %s took: %s\n" % (year, time_taken))

        # End timing
        end_time = timing.current_milli_time()
        time_taken = timing.time_taken(start_time, end_time)
        self._logger.info("HB Production Model took:%s" % time_taken)
        self._logger.info("HB Production Model Finished")

    def _read_land_use_data(self,
                            year: int,
                            ) -> nd.DVector:
        """
        Reads in the land use data for year and converts it to Dvector

        Parameters
        ----------
        year:
            The year to get population data for.

        Returns
        -------
        pop_dvec:
            Returns the population Dvector
        """
        # Define the zoning and segmentations we want to use
        msoa_zoning = nd.get_zoning_system('msoa')
        pop_seg = nd.get_segmentation_level('notem_lu_pop')

        # Read the land use data corresponding to the year
        pop = file_ops.read_df(
            path=self.population_paths[year],
            find_similar=True,
        )

        # TODO(BT): Remove this in Land Use 4.0 Update
        # Little hack until Land Use is updated
        if str(year) in list(pop):
            pop = pop.rename(columns={str(year): 'people'})

        pop = pd_utils.reindex_cols(pop, self._target_col_dtypes['pop'].keys())
        for col, dt in self._target_col_dtypes['pop'].items():
            pop[col] = pop[col].astype(dt)

        # Instantiate
        return nd.DVector(
            zoning_system=msoa_zoning,
            segmentation=pop_seg,
            import_data=pop.rename(columns=self._seg_rename),
            zone_col="msoa_zone_id",
            val_col="people",
        )

    def _generate_productions(self,
                              population: nd.DVector,
                              ) -> nd.DVector:
        """
        Applies trip rate split on the given HB productions

        Parameters
        ----------
        population:
            DVector containing the population.

        Returns
        -------
        pure_demand:
            Returns the product of population and trip rate Dvector
            ie., pure demand
        """

        # Define the zoning and segmentations we want to use
        pure_hb_prod = nd.get_segmentation_level('notem_hb_productions_pure')

        # Reading trip rates
        trip_rates = du.safe_read_csv(
            self.trip_rates_path,
            usecols=self._target_col_dtypes['trip_rate'].keys(),
            dtype=self._target_col_dtypes['trip_rate'],
        )

        # ## CREATE THE TRIP RATES DVEC ## #

        # Instantiate
        trip_rates_dvec = nd.DVector(
            zoning_system=None,
            segmentation=pure_hb_prod,
            import_data=trip_rates.rename(columns=self._seg_rename),
            val_col="trip_rate",
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
        m_tp_splits_seg = nd.get_segmentation_level('notem_hb_productions_full_tfnat')

        full_seg = nd.get_segmentation_level('notem_hb_productions_full')
        # Create the mode-time splits DVector
        mode_time_splits = pd.read_csv(
            self.mode_time_splits_path,
            usecols=self._target_col_dtypes['m_tp'].keys(),
            dtype=self._target_col_dtypes['m_tp'],
        )

        mode_time_splits_dvec = nd.DVector(
            zoning_system=None,
            segmentation=m_tp_splits_seg,
            time_format='avg_week',
            import_data=mode_time_splits,
            val_col="split",
        )

        return pure_demand.multiply_and_aggregate(
            other=mode_time_splits_dvec,
            out_segmentation=full_seg,
        )

    def _trip_end_adjustment(self, trip_ends: nd.DVector) -> nd.DVector:
        """Multiply `trip_ends` by `adjustment_factors`.

        Trip ends are multiplied by all adjustment factors in
        the list one after another.

        Parameters
        ----------
        trip_ends : nd.DVector
            Productions trip ends for adjustment.

        Returns
        -------
        nd.DVector
            Productions trip ends after applying all adjustments.
        """
        for adjustment in self.adjustment_factors:
            self._logger.info("adjusting trip ends with %s", adjustment.file)

            adjust_dvec = adjustment.dvector
            if adjustment.zoning != trip_ends.zoning_system:
                adjust_dvec = adjustment.dvector.translate_zoning(
                    trip_ends.zoning_system, weighting="no_weight"
                )

            trip_ends = trip_ends * adjust_dvec

        return trip_ends


class NHBProductionModel(NHBProductionModelPaths):
    _log_fname = "NHBProductionModel_log.log"
    """The Non Home-Based Production Model of NoTEM

        The production model can be ran by calling the class run() method.

        Attributes
        ----------
        hb_attraction_paths:
            Dictionary of {year: notem_segmented_HB_attractions_data} pairs.
            As passed into the constructor.

        population_paths: Dict[int, nd.PathLike]:
            Dictionary of {year: land_use_employment_data} pairs. As passed
            into the constructor.

        trip_rates_path: str
            The path to the NHB production trip rates. As passed into the constructor.

        time_splits_path: str
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
    # Constants
    __version__ = nd.__version__
    _return_segmentation_name = 'notem_nhb_output'

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
                 hb_attraction_paths: Dict[int, nd.PathLike],
                 population_paths: Dict[int, nd.PathLike],
                 trip_rates_path: str,
                 time_splits_path: str,
                 export_home: str,
                 constraint_paths: Dict[int, nd.PathLike] = None,
                 process_count: int = consts.PROCESS_COUNT
                 ) -> None:
        """
        Sets up and validates arguments for the NHB Production model.

        Parameters
        ----------
        hb_attraction_paths:
            Dictionary of {year: notem_segmented_HB_attractions_data} pairs.
            These paths should come from nd.HBAttraction model and should
            be pickled Dvector paths.

        population_paths:
            Dictionary of {year: land_use_population_data} pairs.

        trip_rates_path:
            The path to the NHB production trip rates.
            Should have the columns as defined in:
            NHBProductionModel._target_cols['nhb_trip_rate']

        time_splits_path:
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
        [file_ops.check_file_exists(x) for x in hb_attraction_paths.values()]
        [file_ops.check_file_exists(x, find_similar=True) for x in population_paths.values()]
        file_ops.check_file_exists(trip_rates_path, find_similar=True)
        file_ops.check_file_exists(time_splits_path, find_similar=True)

        if constraint_paths is not None:
            [file_ops.check_file_exists(x, find_similar=True) for x in constraint_paths.values()]

        # Validate that we have data for all the years we're running for
        for year in hb_attraction_paths.keys():
            if year not in population_paths.keys():
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
        self.hb_attraction_paths = hb_attraction_paths
        self.population_paths = population_paths
        self.trip_rates_path = trip_rates_path
        self.time_splits_path = time_splits_path
        self.constraint_paths = constraint_paths
        self.process_count = process_count
        self.years = list(self.hb_attraction_paths.keys())

        # Make sure the reports paths exists
        report_home = os.path.join(export_home, "Reports")
        file_ops.create_folder(report_home)

        # Build the output paths
        super().__init__(
            path_years=self.years,
            export_home=export_home,
            report_home=report_home,
        )
        # Create a logger
        logger_name = "%s.%s" % (nd.get_package_logger_name(), self.__class__.__name__)
        log_file_path = os.path.join(self.export_home, self._log_fname)
        self._logger = nd.get_logger(
            logger_name=logger_name,
            log_file_path=log_file_path,
            instantiate_msg="Initialised NHB Production Model",
        )

    def run(self,
            export_nhb_pure_demand: bool = False,
            export_fully_segmented: bool = False,
            export_notem_segmentation: bool = True,
            export_reports: bool = True,
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

        Returns
        -------
        None
        """
        # Initialise timing

        start_time = timing.current_milli_time()
        self._logger.info("Starting NHB Production Model")

        # Generate the nhb productions for each year
        for year in self.years:
            year_start_time = timing.current_milli_time()

            # ## GENERATE PURE DEMAND ## #
            self._logger.info("Loading the HB attraction data")
            hb_attr_dvec = self._transform_attractions(year)

            self._logger.info("Applying trip rates")
            pure_nhb_demand = self._generate_nhb_productions(hb_attr_dvec)

            if export_nhb_pure_demand:
                self._logger.info("Exporting NHB pure demand to disk")
                pure_nhb_demand.save(self.export_paths.pure_demand[year])

            if export_reports:
                self._logger.info("Exporting NHB pure demand reports to disk")
                report_seg = nd.get_segmentation_level('notem_nhb_productions_pure_report')
                pure_demand_paths = self.report_paths.pure_demand
                pure_nhb_demand.aggregate(report_seg).write_sector_reports(
                    segment_totals_path=pure_demand_paths.segment_total[year],
                    ca_sector_path=pure_demand_paths.ca_sector[year],
                    ie_sector_path=pure_demand_paths.ie_sector[year],
                )

            # ## SPLIT NHB PURE DEMAND BY TIME ## #
            self._logger.info("Splitting by time")
            fully_segmented = self._split_by_tp(pure_nhb_demand)

            # ## PRODUCTIONS TOTAL CHECK ## #
            if not pure_nhb_demand.sum_is_close(fully_segmented):
                msg = (
                    "The NHB production totals before and after time split are not same.\n"
                    "Expected %f\n"
                    "Got %f"
                    % (pure_nhb_demand.sum(), fully_segmented.sum())
                )
                self._logger.warning(msg)
                warnings.warn(msg)

            if export_fully_segmented:
                self._logger.info("Exporting fully segmented demand to disk")
                fully_segmented.save(self.export_paths.fully_segmented[year])

            # Renaming
            notem_segmented = self._rename(fully_segmented)

            # ## PRODUCTIONS TOTAL CHECK ## #
            if not fully_segmented.sum_is_close(notem_segmented):
                msg = (
                    "The NHB production totals before and after rename to "
                    "output segmentation are not same.\n"
                    "Expected %f\n"
                    "Got %f"
                    % (pure_nhb_demand.sum(), fully_segmented.sum())
                )
                self._logger.warning(msg)
                warnings.warn(msg)

            if export_notem_segmentation:
                self._logger.info("Exporting notem segmented demand to disk")
                notem_segmented.save(self.export_paths.notem_segmented[year])

            if export_reports:
                self._logger.info("Exporting notem segmented reports to disk\n")
                notem_segmented_paths = self.report_paths.notem_segmented
                notem_segmented.write_sector_reports(
                    segment_totals_path=notem_segmented_paths.segment_total[year],
                    ca_sector_path=notem_segmented_paths.ca_sector[year],
                    ie_sector_path=notem_segmented_paths.ie_sector[year],
                    lad_report_path=notem_segmented_paths.lad_report[year],
                    lad_report_seg=nd.get_segmentation_level('nhb_p_m_tp_week'),
                )

            # TODO: Bring in constraints (Validation)
            #  Output some audits of what demand was before and after control
            #  By segment.
            if self.constraint_paths is not None:
                msg = "No code implemented to constrain productions"
                self._logger.error(msg)
                raise NotImplementedError(msg)

            # Print timing stats for the year
            year_end_time = timing.current_milli_time()
            time_taken = timing.time_taken(year_start_time, year_end_time)
            self._logger.info("NHB Productions in year %s took: %s\n" % (year, time_taken))

        # End timing
        end_time = timing.current_milli_time()
        time_taken = timing.time_taken(start_time, end_time)
        self._logger.info("NHB Production Model took:%s" % time_taken)
        self._logger.info("NHB Production Model Finished")

    def _transform_attractions(self,
                               year: int,
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

        Returns
        -------
        hb_attr_dvec:
            Returns the HB attraction Dvector with tfn_at.
        """
        # Define the zoning and segmentations we want to use
        msoa_zoning = nd.get_zoning_system('msoa')
        notem_no_tp_seg = nd.get_segmentation_level('notem_hb_output_no_tp')

        # ## READ IN AND VALIDATE THE LAND USE DATA ## #
        # Reading the land use data
        # Read the land use data corresponding to the year
        pop = file_ops.read_df(
            path=self.population_paths[year],
            find_similar=True,
        )
        pop = pd_utils.reindex_cols(pop, self._target_col_dtypes['land_use'].keys())
        for col, dtype in self._target_col_dtypes['land_use'].items():
            pop[col] = pop[col].astype(dtype)

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

        # Convert area_types into a DVector
        pop['value'] = 1
        area_type = nd.DVector(
            zoning_system=msoa_zoning,
            segmentation=nd.get_segmentation_level('tfn_at'),
            import_data=pop,
            zone_col="zone",
            val_col="value",
        )

        # ## CONVERT THE ATTRACTIONS INTO DESIRED FORMAT ## #
        # Read the notem segmented compressed pickle
        hb_attr_notem = nd.DVector.load(self.hb_attraction_paths[year])

        # Remove time period and add in tfn_at
        hb_attr = hb_attr_notem.aggregate(notem_no_tp_seg)
        return hb_attr.expand_segmentation(area_type)

    def _generate_nhb_productions(self,
                                  hb_attractions: nd.DVector,
                                  ) -> nd.DVector:
        """
        Applies NHB trip rates to hb_attractions

        Parameters
        ----------
        hb_attractions:
            Dvector containing the data to apply the trip rates to.

        Returns
        -------
        pure_NHB_demand:
            Returns the product of HB attractions and NHB trip rate Dvector
            ie., pure NHB demand
        """

        # Define the zoning and segmentations we want to use
        nhb_trip_rate_seg = nd.get_segmentation_level('notem_nhb_trip_rate')
        pure_seg = nd.get_segmentation_level('notem_nhb_productions_pure')

        # Reading NHB trip rates
        trip_rates = du.safe_read_csv(
            file_path=self.trip_rates_path,
            usecols=self._target_col_dtypes['nhb_trip_rate'].keys(),
            dtype=self._target_col_dtypes['nhb_trip_rate'],
        )

        # Create the NHB Trip Rates DVec
        trip_rates_dvec = nd.DVector(
            zoning_system=None,
            segmentation=nhb_trip_rate_seg,
            import_data=trip_rates.rename(columns=self._seg_rename),
            val_col="nhb_trip_rate",
        )

        # Multiply
        return hb_attractions.multiply_and_aggregate(trip_rates_dvec, pure_seg)

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
        nhb_time_splits_seg = nd.get_segmentation_level('notem_nhb_tfnat_p_m_tp')
        full_seg = nd.get_segmentation_level('notem_nhb_productions_full')

        # Read the time splits factor
        time_splits = pd.read_csv(
            self.time_splits_path,
            usecols=self._target_col_dtypes['tp'].keys(),
            dtype=self._target_col_dtypes['tp'],
        )

        # Instantiate
        time_splits_dvec = nd.DVector(
            zoning_system=None,
            segmentation=nhb_time_splits_seg,
            time_format='avg_week',
            import_data=time_splits,
            val_col="split",
        )

        # Multiply together #
        return pure_nhb_demand.multiply_and_aggregate(
            other=time_splits_dvec,
            out_segmentation=full_seg,
        )

    def _rename(self, full_segmentation: nd.DVector) -> nd.DVector:
        """
        Renames nhb_p and nhb_m as m and p respectively in full segmentation

        Parameters
        ----------
        full_segmentation:
            fully segmented NHB productions containing nhb_p and nhb_m as column names

        Returns
        -------
        notem_segmented:
            Returns the notem segmented NHB production DVector
        """
        nhb_prod_seg = nd.get_segmentation_level(self._return_segmentation_name)
        return full_segmentation.aggregate(nhb_prod_seg)


@dataclasses.dataclass
class TripEndAdjustmentFactors:
    """Stores (and reads) the trip end adjustment factors data.

    Attributes
    ----------
    file : pathlib.Path
        CSV file containing the adjustment factors, with
        columns containing the zone IDs, segment data and
        finally the factors.
    segmentation : nd.SegmentationLevel
        Segmentation level that the data in `file` is in.
    zoning : nd.ZoningSystem
        Zone system that the data in `file` is in.
    time_format : nd.TimeFormat
        Time format that the data in `file` is in.
    dvector : nd.DVector
    """
    file: pathlib.Path
    segmentation: nd.SegmentationLevel
    zoning: nd.ZoningSystem
    time_format: nd.TimeFormat

    def __post_init__(self) -> None:
        """Check given `file` exists.

        Raises
        ------
        FileNotFoundError
            If `self.file` isn't a path to an existing file.
        """
        self._dvector: Optional[nd.DVector] = None

        self.file = pathlib.Path(self.file)
        if not self.file.is_file():
            raise FileNotFoundError(
                f"adjustment factors file doesn't exist: {self.file}"
            )

    @property
    def dvector(self) -> nd.DVector:
        """Read data from file and return as a DVector."""
        if self._dvector is None:
            data = file_ops.read_df(self.file)
            self._dvector = nd.DVector(
                segmentation=self.segmentation,
                import_data=data,
                zoning_system=self.zoning,
                time_format=self.time_format,
                infill=1,
            )
        return self._dvector
