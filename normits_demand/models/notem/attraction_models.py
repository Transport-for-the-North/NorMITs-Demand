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
import warnings

from typing import Dict

# Third party imports
import pandas as pd

# local imports
import normits_demand as nd

from normits_demand import constants as consts

from normits_demand.utils import timing
from normits_demand.utils import file_ops
from normits_demand.utils import general as du
from normits_demand.utils import pandas_utils as pd_utils

from normits_demand.pathing import HBAttractionModelPaths
from normits_demand.pathing import NHBAttractionModelPaths


class HBAttractionModel(HBAttractionModelPaths):
    _log_fname = "HBAttractionModel_log.log"
    """The Home-Based Attraction Model of NoTEM

        The attraction model can be ran by calling the class run() method.

        Attributes
        ----------
        employment_paths: Dict[int, nd.PathLike]:
            Dictionary of {year: land_use_employment_data} pairs. As passed
            into the constructor.

        production_balance_paths: Dict[int, nd.PathLike]:
            Dictionary of {year: path_to_production_to_control_to} pairs. As passed
            into the constructor.

        trip_weights_path: str
            The path to the attraction trip weights. As passed into the constructor.

        mode_splits_path: str
            The path to attraction mode splits. As passed into the constructor.

        constraint_paths: Dict[int, nd.PathLike]
            Dictionary of {year: constraint_path} pairs. As passed into the
            constructor.

        process_count: int
            The number of processes to create in the Pool. As passed into the
            constructor.

        years: List[int]
            A list of years that the model will run for. Derived from the keys of
            land_use_paths

        See HBAttractionModelPaths for documentation on:
            "path_years, export_home, report_home, export_paths, report_paths"
        """
    # Constants
    __version__ = nd.__version__

    # Define wanted columns
    _target_col_dtypes = {
        'employment': {
            'msoa_zone_id': str,
            'employment_cat': str,
            'soc': int,
            'people': float
        },
        'trip_weight': {
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
        'people': 'people',
        'area_type': 'tfn_at',
        'purpose': 'p',
        'mode_share': 'split'
    }

    def __init__(self,
                 employment_paths: Dict[int, nd.PathLike],
                 production_balance_paths: Dict[int, nd.PathLike],
                 trip_weights_path: str,
                 mode_splits_path: str,
                 export_home: str,
                 constraint_paths: Dict[int, nd.PathLike] = None,
                 process_count: int = consts.PROCESS_COUNT
                 ) -> None:
        """
        Sets up and validates arguments for the Attraction model.

        Parameters
        ----------
        employment_paths:
            Dictionary of {year: land_use_employment_data} pairs.
            Should have the columns as defined in:
            HBAttractionModel._target_cols['employment']

        production_balance_paths:
            Dictionary of {year: path_to_production_to_control_to} pairs.
            These paths should be gotten from nd.HBProduction model.
            Must contain the same keys as land_use_paths, but it can contain
            more (any extras will be ignored).
            These productions will be used to control the produced attractions.

        trip_weights_path:
            The path to the attraction trip weights.
            Should have the columns as defined in:
            HBAttractionModel._target_cols['trip_weight']

        mode_splits_path:
            The path to attraction mode split.
            Should have the columns as defined in:
            HBAttractionModel._target_cols['mode_split']

        export_home:
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
        # Check that the paths we need exist!
        [file_ops.check_file_exists(x, find_similar=True) for x in employment_paths.values()]
        [file_ops.check_file_exists(x) for x in production_balance_paths.values()]
        file_ops.check_file_exists(trip_weights_path)
        file_ops.check_file_exists(mode_splits_path)

        if constraint_paths is not None:
            [file_ops.check_file_exists(x, find_similar=True) for x in constraint_paths.values()]

        # Validate that we have data for all the years we're running for
        for year in employment_paths.keys():
            if year not in production_balance_paths.keys():
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
        self.employment_paths = employment_paths
        self.production_balance_paths = production_balance_paths
        self.trip_weights_path = trip_weights_path
        self.mode_splits_path = mode_splits_path
        self.constraint_paths = constraint_paths
        self.process_count = process_count
        self.years = list(self.employment_paths.keys())

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
            instantiate_msg="Initialised HB Attraction Model",
        )

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
              self.export_paths.pure_demand[year]
            - Optionally writes out a number of "pure attractions" reports, if
              reports is True.
            - Reads in the mode splits given in the constructor.
            - Multiplies the "pure attractions" and mode splits on relevant
              segments, producing "fully segmented attractions".
            - Checks the attraction totals before and after mode split and throws
              error if they don't match.
            - Optionally writes out a pickled DVector of "fully segmented attractions"
              at self.export_paths.fully_segmented[year].
            - Balances "fully segmented attractions" to production notem segmentation,
              producing "notem segmented" attractions.
            - Optionally writes out a pickled DVector of "notem segmented attractions"
              at self.export_paths.notem_segmented[year].
            - Optionally writes out a number of "notem segmented" reports, if
              reports is True.

        Parameters
        ----------
        export_pure_attractions:
            Whether to export the pure attractions to disk or not.
            Will be written out to: self.export_paths.pure_demand[year]

        export_fully_segmented:
            Whether to export the fully segmented attractions to disk or not.
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
        
        start_time = timing.current_milli_time()
        self._logger.info("Starting HB Attraction Model")

        # Generate the attractions for each year
        for year in self.years:
            year_start_time = timing.current_milli_time()

            # ## GENERATE PURE ATTRACTIONS ## #
            self._logger.info("Loading the employment data")
            emp_dvec = self._read_land_use_data(year, verbose=verbose)

            self._logger.info("Applying trip rates")
            pure_attractions = self._generate_attractions(emp_dvec, verbose=verbose)

            if export_pure_attractions:
                self._logger.info("Exporting pure attractions to disk")
                pure_attractions.to_pickle(self.export_paths.pure_demand[year])

            if export_reports:
                self._logger.info("Exporting pure demand reports to disk")               
                pure_demand_paths = self.report_paths.pure_demand
                pure_attractions.write_sector_reports(
                    segment_totals_path=pure_demand_paths.segment_total[year],
                    ca_sector_path=pure_demand_paths.ca_sector[year],
                    ie_sector_path=pure_demand_paths.ie_sector[year],
                )

            # ## SPLIT PURE ATTRACTIONS BY MODE ## #
            self._logger.info("Splitting by mode")
            fully_segmented = self._split_by_mode(pure_attractions)

            # ## ATTRACTIONS TOTAL CHECK ## #
            if not pure_attractions.sum_is_close(fully_segmented):
                msg = (
                    "The attraction totals before and after mode split are not same.\n"
                    "Expected %f\n"
                    "Got %f"
                    % (pure_attractions.sum(), fully_segmented.sum())
                )                
                self._logger.warning(msg)
                warnings.warn(msg)

            # Output attractions before any aggregation
            if export_fully_segmented:
                self._logger.info("Exporting fully segmented attractions to disk")
                fully_segmented.to_pickle(self.export_paths.fully_segmented[year])

            # Control the attractions to the productions - this also adds in
            # some segmentation to bring it in line with the productions
            notem_segmented = self._attractions_balance(
                a_dvec=fully_segmented,
                p_dvec_path=self.production_balance_paths[year],
            )

            if export_notem_segmentation:
                self._logger.info("Exporting notem segmented attractions to disk")
                notem_segmented.to_pickle(self.export_paths.notem_segmented[year])

            if export_reports:
                self._logger.info("Exporting notem segmented reports to disk")
                notem_segmented_paths = self.report_paths.notem_segmented
                notem_segmented.write_sector_reports(
                    segment_totals_path=notem_segmented_paths.segment_total[year],
                    ca_sector_path=notem_segmented_paths.ca_sector[year],
                    ie_sector_path=notem_segmented_paths.ie_sector[year],
                )

            # TODO: Bring in constraints (Validation)
            #  Output some audits of what attractions was before and after control
            #  By segment.
            if self.constraint_paths is not None:
                msg = "No code implemented to constrain attractions."
                self._logger.error(msg)
                raise NotImplementedError(msg)  
              
            # Print timing stats for the year
            year_end_time = timing.current_milli_time()
            time_taken = timing.time_taken(year_start_time, year_end_time)
            self._logger.info("HB Attraction in year %s took: %s\n" % (year, time_taken))

        # End timing
        end_time = timing.current_milli_time()
        time_taken = timing.time_taken(start_time, end_time)
        self._logger.info("HB Attraction Model took: %s" % time_taken)
        self._logger.info("HB Attraction Model Finished")        

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
        emp_seg = nd.get_segmentation_level('notem_lu_emp')

        # Read the land use data corresponding to the year
        emp = file_ops.read_df(
            path=self.employment_paths[year],
            find_similar=True,
        )

        # TODO(BT): Remove this in Land Use 4.0 Update
        # Little hack until Land Use is updated
        if str(year) in list(emp):
            emp = emp.rename(columns={str(year): 'people'})

        emp = pd_utils.reindex_cols(emp, self._target_col_dtypes['employment'].keys())
        for col, dt in self._target_col_dtypes['employment'].items():
            emp[col] = emp[col].astype(dt)

        # Instantiate
        return nd.DVector(
            zoning_system=msoa_zoning,
            segmentation=emp_seg,
            import_data=emp.rename(columns=self._seg_rename),
            zone_col="msoa_zone_id",
            val_col="people",
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
        trip_weights_seg = nd.get_segmentation_level('notem_hb_attractions_trip_weights')
        pure_attractions_seg = nd.get_segmentation_level('notem_hb_attractions_pure')

        # ## CREATE THE TRIP RATES DVEC ## #
        # Reading trip rates
        trip_rates = du.safe_read_csv(
            self.trip_weights_path,
            usecols=self._target_col_dtypes['trip_weight'].keys(),
            dtype=self._target_col_dtypes['trip_weight'],
        )

        # make DVec
        trip_weights_dvec = nd.DVector(
            zoning_system=msoa_zoning,
            segmentation=trip_weights_seg,
            import_data=trip_rates.rename(columns=self._seg_rename),
            zone_col="msoa_zone_id",
            val_col="trip_rate",
        )

        # ## MULTIPLY TOGETHER ## #
        # Remove un-needed ecat column too
        return emp_dvec.multiply_and_aggregate(trip_weights_dvec, pure_attractions_seg)

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
        p_dvec = nd.read_pickle(p_dvec_path)

        # Split a_dvec into p_dvec segments and balance
        a_dvec = a_dvec.split_segmentation_like(p_dvec)
        return a_dvec.balance_at_segments(p_dvec, split_weekday_weekend=True)


class NHBAttractionModel(NHBAttractionModelPaths):
    _log_fname = "NHBAttractionModel_log.log"
    """The Non Home-Based Attraction Model of NoTEM

        The attraction model can be ran by calling the class run() method.

        Attributes
        ----------
        hb_attraction_paths: Dict[int, nd.PathLike]:
            Dictionary of {year: notem_segmented_HB_attractions_data} pairs. As passed
            into the constructor.

        nhb_production_paths: Dict[int, nd.PathLike]:
            Dictionary of {year: notem_segmented_NHB_productions_data} pairs. As passed
            into the constructor.

        constraint_paths: Dict[int, nd.PathLike]
            Dictionary of {year: constraint_path} pairs. As passed into the
            constructor.

        process_count: int
            The number of processes to create in the Pool. As passed into the
            constructor.

        See NHBAttractionModelPaths for documentation on:
            "path_years, export_home, report_home, export_paths, report_paths"
        """
    # Constants
    __version__ = nd.__version__

    def __init__(self,
                 hb_attraction_paths: Dict[int, nd.PathLike],
                 nhb_production_paths: Dict[int, nd.PathLike],
                 export_home: str,
                 constraint_paths: Dict[int, nd.PathLike] = None,
                 process_count: int = consts.PROCESS_COUNT
                 ) -> None:
        """
        Sets up and validates arguments for the NHB Attraction model.

        Parameters
        ----------
        hb_attraction_paths:
            Dictionary of {year: notem_segmented_HB_attractions_data} pairs.
            These paths should come from nd.HBAttraction model and should
            be pickled Dvector paths.

        nhb_production_paths:
            Dictionary of {year: notem_segmented_NHB_productions_data} pairs.
            These paths should come from nd.NHBProduction model and should
            be pickled Dvector paths.
            These productions will be used to control the produced attractions.

        export_home:
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
        [file_ops.check_file_exists(x) for x in hb_attraction_paths.values()]
        [file_ops.check_file_exists(x) for x in nhb_production_paths.values()]

        if constraint_paths is not None:
            [file_ops.check_file_exists(x, find_similar=True) for x in constraint_paths.values()]

        # Validate that we have data for all the years we're running for
        for year in hb_attraction_paths.keys():
            if year not in nhb_production_paths.keys():
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
        self.hb_attraction_paths = hb_attraction_paths
        self.nhb_production_paths = nhb_production_paths
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
            instantiate_msg="Initialised NHB Attraction Model",
        )

    def run(self,
            export_nhb_pure_attractions: bool = True,
            export_notem_segmentation: bool = True,
            export_reports: bool = True,
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
            Will be written out to: self.export_paths.pure_demand[year]

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
        start_time = timing.current_milli_time()
        self._logger.info("Starting NHB Attraction Model")

        # Generate the nhb attractions for each year
        for year in self.years:
            year_start_time = timing.current_milli_time()

            # ## GENERATE PURE ATTRACTIONS ## #
            self._logger.info("Loading the HB attraction data")
            pure_nhb_attr = self._create_nhb_attraction_data(year, verbose=verbose)

            if export_nhb_pure_attractions:
                self._logger.info("Exporting NHB pure attractions to disk")
                pure_nhb_attr.to_pickle(self.export_paths.pure_demand[year])

            if export_reports:
                self._logger.info("Exporting pure NHB attractions reports to disk")
                pure_demand_paths = self.report_paths.pure_demand
                pure_nhb_attr.write_sector_reports(
                    segment_totals_path=pure_demand_paths.segment_total[year],
                    ca_sector_path=pure_demand_paths.ca_sector[year],
                    ie_sector_path=pure_demand_paths.ie_sector[year],
                )

            # Control the attractions to the productions
            notem_segmented = self._attractions_balance(
                a_dvec=pure_nhb_attr,
                p_dvec_path=self.nhb_production_paths[year],
            )

            if export_notem_segmentation:
                self._logger.info("Exporting notem segmented attractions to disk")
                notem_segmented.to_pickle(self.export_paths.notem_segmented[year])

            if export_reports:
                self._logger.info("Exporting notem segmented attractions reports to disk")
                notem_segmented_paths = self.report_paths.notem_segmented
                notem_segmented.write_sector_reports(
                    segment_totals_path=notem_segmented_paths.segment_total[year],
                    ca_sector_path=notem_segmented_paths.ca_sector[year],
                    ie_sector_path=notem_segmented_paths.ie_sector[year],
                )

            # TODO: Bring in constraints (Validation)
            #  Output some audits of what attractions was before and after control
            #  By segment.
            if self.constraint_paths is not None:
                msg = "No code implemented to constrain productions"                              
                self._logger.error(msg)
                raise NotImplementedError(msg)            
                
            # Print timing stats for the year
            year_end_time = timing.current_milli_time()
            time_taken = timing.time_taken(year_start_time, year_end_time)
            self._logger.info("NHB Attraction in year %s took: %s\n" % (year, time_taken))

        # End timing
        end_time = timing.current_milli_time()
        time_taken = timing.time_taken(start_time, end_time)
        self._logger.info("NHB Attraction Model took: %s" % time_taken)
        self._logger.info("NHB Attraction Model Finished")

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
        segmentation = nd.get_segmentation_level('notem_nhb_output')

        # Reading the notem segmented HB attractions compressed pickle
        hb_attr_notem = nd.read_pickle(self.hb_attraction_paths[year])
        hb_attr_notem_df = hb_attr_notem.to_df()

        # Removing p1 and p7
        mask = (hb_attr_notem_df['p'].astype(int) == 7)
        hb_attr_notem_df = hb_attr_notem_df[~mask].copy().reset_index(drop=True)

        # Adding 10 to the remaining purposes
        hb_attr_notem_df['p'] = hb_attr_notem_df['p'].astype(int) + 10

        # Instantiate
        return nd.DVector(
            zoning_system=hb_attr_notem.zoning_system,
            segmentation=segmentation,
            time_format='avg_week',
            import_data=hb_attr_notem_df,
            zone_col=hb_attr_notem.zoning_system.col_name,
            val_col="val",
        )

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
        p_dvec = nd.read_pickle(p_dvec_path)

        # Balance a_dvec with p_dvec
        return a_dvec.balance_at_segments(p_dvec, split_weekday_weekend=True)
