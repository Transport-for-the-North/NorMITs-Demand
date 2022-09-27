# -*- coding: utf-8 -*-
"""
Created on: Wednesday Sept 22nd 2021

Original author: Nirmal Kumar

File purpose:
Tram inclusion on NoTEM outputs
"""
# Allow class self type hinting
from __future__ import annotations

# Builtins
import os

from typing import Any
from typing import List
from typing import Dict
from typing import Tuple

# Third party imports
import pandas as pd
import numpy as np

# local imports
import normits_demand as nd
from normits_demand import core as nd_core

from normits_demand.utils import timing
from normits_demand.utils import file_ops
from normits_demand.utils import general as du
from normits_demand.utils import math_utils
from normits_demand.utils import pandas_utils as pd_utils

from normits_demand.pathing import TramExportPaths


class TramModel(TramExportPaths):
    """
    Tram Inclusion on NoTEM outputs
    """
    # Constants
    _base_year = 2018
    _train_mode = nd.Mode.TRAIN.get_mode_num()
    _tram_mode = nd.Mode.TRAM.get_mode_num()
    _zoning_name = 'msoa'
    _lad_report_zoning_name = 'lad_2020'

    # Filenames
    _running_report_fname = 'running_parameters.txt'
    _log_fname = "tram_log.log"

    # Col names
    _tram_segment_cols = ['p', 'm', 'ca']
    _lad_comparison_col_names = ['p', 'm', 'tp']
    _val_col = 'val'

    # Running segmentations
    _hb_tram_seg = 'hb_p_m_ca'
    _nhb_tram_seg = 'nhb_p_m_ca'

    # Define report names
    _hb_reports = {
        'segment_total': 'hb_notem_segmented_%d_segment_totals.csv',
        'ca_sector': 'hb_notem_segmented_%d_ca_sector_totals.csv',
        'ie_sector': 'hb_notem_segmented_%d_ie_sector_totals.csv',
    }
    _nhb_reports = {
        'segment_total': 'nhb_notem_segmented_%d_segment_totals.csv',
        'ca_sector': 'nhb_notem_segmented_%d_ca_sector_totals.csv',
        'ie_sector': 'nhb_notem_segmented_%d_ie_sector_totals.csv',
    }

    # Define wanted columns
    _target_col_dtypes = {
        'tram': {
            'msoa_zone_id': str,
            'p': int,
            'ca': int,
            'm': int,
            'trips': float
        },
        'notem': {
            'msoa_zone_id': str,
            'p': int,
            'ca': int,
            'm': int,
            'val': float
        },
        'notem_north': {
            'ie_sector_zone_id': int,
            'p': int,
            'ca': int,
            'm': int,
            'val': float
        }
    }

    def __init__(self,
                 years: List[int],
                 scenario: nd_core.Scenario,
                 iteration_name: str,
                 import_builder: nd.pathing.TramImportPathsBase,
                 export_home: nd.PathLike,
                 tram_competitors: List[nd.Mode],
                 hb_balance_zoning: nd.BalancingZones = None,
                 nhb_balance_zoning: nd.BalancingZones = None,
                 ):
        """
        Assigns the attributes needed for tram inclusion model.

        Parameters
        ----------
        years:
            List of years to run tram inclusion for. Will assume that the smallest
            year is the base year.

        scenario:
            The scenario to run for.

        iteration_name:
            The name of this iteration of the NoTEM models. Will have 'iter'
            prepended to create the folder name. e.g. if iteration_name was
            set to '3i' the iteration folder would be called 'iter3i'.

        import_builder:
            The home location where all the tram related import files are located.

        export_home:
            The home where all the export paths should be built from. See
            nd.pathing.NoTEMExportPaths for more info on how these paths
            will be built.

        tram_competitors:
            A list of the Modes which would be competing with Tram for trips.
            These are the modes which will be used to remove trips from in
            order to add in tram trips

        hb_balance_zoning:
            The zoning systems to balance the home-based trip ends
            at. A translation must exist
            between this and the running zoning system, which is MSOA by default.
            If left as None, then no spatial balance is done, only a segmental balance.

        nhb_balance_zoning:
            The zoning systems to balance the non-home-based trip ends
            at. A translation must exist
            between this and the running zoning system, which is MSOA by default.
            If left as None, then no spatial balance is done, only a segmental balance.
        """
        # Validate inputs
        if not isinstance(import_builder, nd.pathing.TramImportPathsBase):
            raise ValueError(
                'import_builder is not the correct type. Expected '
                '"nd.pathing.TramImportPathsBase", but got %s'
                % type(import_builder)
            )

        # Assign
        self.years = years
        self.base_year = min(self.years)
        self.import_builder = import_builder
        self.base_train = pd.DataFrame()
        self.tram_competitors = tram_competitors
        self.hb_balance_zoning = hb_balance_zoning
        self.nhb_balance_zoning = nhb_balance_zoning

        # Generate the zoning system
        self.zoning_system = nd.get_zoning_system(self._zoning_name)
        self._zoning_system_col = self.zoning_system.col_name

        # Generate the export paths
        super().__init__(
            export_home=export_home,
            path_years=self.years,
            scenario=scenario,
            iteration_name=iteration_name,
        )

        # Create a logger
        logger_name = "%s.%s" % (nd.get_package_logger_name(), self.__class__.__name__)
        log_file_path = os.path.join(self.export_home, self._log_fname)
        self._logger = nd.get_logger(
            logger_name=logger_name,
            log_file_path=log_file_path,
            instantiate_msg="Initialised new Tram Logger",
        )
        self._write_running_report()

        # Validate the base year
        base_year = min(years)
        if base_year != self._base_year:
            self._logger.warning(
                "The minimum year given is not the same as the base year "
                "defined in the model! The model says the base year is %s, "
                "whereas the years define the base year as %s."
                % (self._base_year, base_year)
            )

    def _write_running_report(self):
        """
        Outputs a simple report detailing inputs and outputs
        """
        # Define the lines to output
        out_lines = [
            'Code Version: %s' % str(nd.__version__),
            'NoTEM/Tram Iteration: %s' % str(self.iteration_name),
            'Scenario: %s' % str(self.scenario.value),
            '',
            '### HB Productions ###',
            'import_files: %s' % self.import_builder.generate_hb_production_imports(),
            'vector_export: %s' % self.hb_production.export_paths.home,
            'report_export: %s' % self.hb_production.report_paths.home,
            '',
            '### HB Attractions ###',
            'import_files: %s' % self.import_builder.generate_hb_attraction_imports(),
            'vector_export: %s' % self.hb_attraction.export_paths.home,
            'report_export: %s' % self.hb_attraction.report_paths.home,
            '',
            '### NHB Productions ###',
            'import_files: %s' % self.import_builder.generate_nhb_production_imports(),
            'vector_export: %s' % self.nhb_production.export_paths.home,
            'report_export: %s' % self.nhb_production.report_paths.home,
            '',
            '### NHB Attractions ###',
            'import_files: %s' % self.import_builder.generate_nhb_attraction_imports(),
            'vector_export: %s' % self.nhb_attraction.export_paths.home,
            'report_export: %s' % self.nhb_attraction.report_paths.home,
        ]

        # Write out to disk
        output_path = os.path.join(self.export_home, self._running_report_fname)
        with open(output_path, 'w') as out:
            out.write('\n'.join(out_lines))

    def run_tram(self,
                 generate_all: bool = False,
                 generate_hb: bool = False,
                 generate_hb_production: bool = False,
                 generate_hb_attraction: bool = False,
                 generate_nhb: bool = False,
                 generate_nhb_production: bool = False,
                 generate_nhb_attraction: bool = False,
                 before_after_report: bool = False,
                 ) -> None:
        """
        Generates the inputs required for tram inclusion run.

        Parameters
        ----------
        generate_all:
            Runs both home based and non home based trip end models.

        generate_hb:
            Runs the home based trip end models only.

        generate_hb_production:
            Runs the home based production trip end model only.

        generate_hb_attraction:
            Runs the home based attraction trip end model only.

        generate_nhb:
            Runs the non home based trip end models only.

        generate_nhb_production:
            Runs the non home based production trip end model only.

        generate_nhb_attraction:
            Runs the non home based attraction trip end model only.

        before_after_report:
            Whether to generate a report comparing the trip end outputs
            before and after the tram infill. These reports will be generated
            at the LAD level.

        Returns
        -------
        None
        """

        start_time = timing.current_milli_time()
        self._logger.info("Starting a new run of the Tram Model")

        # Determine which models to run
        if generate_all:
            generate_hb = True
            generate_nhb = True

        if generate_hb:
            generate_hb_production = True
            generate_hb_attraction = True

        if generate_nhb:
            generate_nhb_production = True
            generate_nhb_attraction = True

        self._logger.debug("Running hb productions: %s" % generate_hb_production)
        self._logger.debug("Running nhb productions: %s" % generate_nhb_production)
        self._logger.debug("Running hb attractions: %s" % generate_hb_attraction)
        self._logger.debug("Running nhb attractions: %s" % generate_nhb_attraction)
        self._logger.debug("Running before/after reports: %s" % before_after_report)
        self._logger.debug("")

        # Run the models
        if generate_hb_production:
            self._generate_hb_production(before_after_report)

        if generate_hb_attraction:
            self._generate_hb_attraction(before_after_report)

        if generate_nhb_production:
            self._generate_nhb_production(before_after_report)

        if generate_nhb_attraction:
            self._generate_nhb_attraction(before_after_report)

        end_time = timing.current_milli_time()
        time_taken = timing.time_taken(start_time, end_time)
        self._logger.info("Tram Model run complete! Took %s" % time_taken)

    def _generate_hb_production(self, before_after_report: bool) -> None:
        """
        Runs tram inclusion for home based Production trip end models
        """
        self._logger.info("Generating HB Production imports")
        import_files = self.import_builder.generate_hb_production_imports()
        before_lad_report_paths = import_files.pop('before_lad_report_paths')

        # Runs the home based Production model
        self._logger.info("Adding Tram into HB Productions")
        paths = self.hb_production
        vector_reports = paths.report_paths.vector_reports
        self._add_tram(
            **import_files,
            export_paths=paths.export_paths.notem_segmented,
            tram_growth_factors=paths.report_paths.tram_growth_factors,
            more_tram_msoa=paths.report_paths.more_tram_msoa,
            more_tram_north=paths.report_paths.more_tram_north,
            mode_adj_factors=paths.report_paths.mode_adj_factors,
            report_segment_total_paths=vector_reports.segment_total,
            report_ca_sector_paths=vector_reports.ca_sector,
            report_ie_sector_paths=vector_reports.ie_sector,
            lad_report_paths=vector_reports.lad_report,
        )

        # Generate a LAD report before and after tram infill
        if before_after_report:
            self._before_after_report(
                before_lad_report_paths=before_lad_report_paths,
                after_lad_report_paths=vector_reports.lad_report,
                output_paths=paths.report_paths.comparison_report,
                tram_only_output_paths=paths.report_paths.tram_comparison_report,
            )

    def _generate_hb_attraction(self, before_after_report: bool) -> None:
        """
        Runs tram inclusion for home based Attraction trip end model
        """
        self._logger.info("Generating HB Attraction imports")
        import_files = self.import_builder.generate_hb_attraction_imports()
        before_lad_report_paths = import_files.pop('before_lad_report_paths')

        self._logger.info("Adding Tram into HB Attractions")
        paths = self.hb_attraction
        vector_reports = paths.report_paths.vector_reports
        self._add_tram(
            **import_files,
            export_paths=paths.export_paths.notem_segmented,
            tram_growth_factors=paths.report_paths.tram_growth_factors,
            more_tram_msoa=paths.report_paths.more_tram_msoa,
            more_tram_north=paths.report_paths.more_tram_north,
            mode_adj_factors=paths.report_paths.mode_adj_factors,
            report_segment_total_paths=vector_reports.segment_total,
            report_ca_sector_paths=vector_reports.ca_sector,
            report_ie_sector_paths=vector_reports.ie_sector,
            lad_report_paths=vector_reports.lad_report,
            balance_paths=self.hb_production.export_paths.notem_segmented,
            split_weekday_weekend=True,
        )

        # Generate a LAD report before and after tram infill
        if before_after_report:
            self._before_after_report(
                before_lad_report_paths=before_lad_report_paths,
                after_lad_report_paths=vector_reports.lad_report,
                output_paths=paths.report_paths.comparison_report,
                tram_only_output_paths=paths.report_paths.tram_comparison_report,
            )

    def _generate_nhb_production(self, before_after_report: bool) -> None:
        """
        Runs tram inclusion for non-home based Production trip end model
        """
        self._logger.info("Generating NHB Production imports")
        import_files = self.import_builder.generate_nhb_production_imports()
        before_lad_report_paths = import_files.pop('before_lad_report_paths')

        self._logger.info("Adding Tram into NHB Productions")
        paths = self.nhb_production
        vector_reports = paths.report_paths.vector_reports
        self._add_tram(
            **import_files,
            export_paths=paths.export_paths.notem_segmented,
            tram_growth_factors=paths.report_paths.tram_growth_factors,
            more_tram_msoa=paths.report_paths.more_tram_msoa,
            more_tram_north=paths.report_paths.more_tram_north,
            mode_adj_factors=paths.report_paths.mode_adj_factors,
            report_segment_total_paths=vector_reports.segment_total,
            report_ca_sector_paths=vector_reports.ca_sector,
            report_ie_sector_paths=vector_reports.ie_sector,
            lad_report_paths=vector_reports.lad_report,
        )

        # Generate a LAD report before and after tram infill
        if before_after_report:
            self._before_after_report(
                before_lad_report_paths=before_lad_report_paths,
                after_lad_report_paths=vector_reports.lad_report,
                output_paths=paths.report_paths.comparison_report,
                tram_only_output_paths=paths.report_paths.tram_comparison_report,
            )

    def _generate_nhb_attraction(self, before_after_report: bool) -> None:
        """
        Runs tram inclusion for non home based Attraction trip end models.
        """
        self._logger.info("Generating NHB Attraction imports")
        import_files = self.import_builder.generate_nhb_attraction_imports()
        before_lad_report_paths = import_files.pop('before_lad_report_paths')

        self._logger.info("Adding Tram into NHB Attractions")
        paths = self.nhb_attraction
        vector_reports = paths.report_paths.vector_reports
        self._add_tram(
            **import_files,
            export_paths=paths.export_paths.notem_segmented,
            tram_growth_factors=paths.report_paths.tram_growth_factors,
            more_tram_msoa=paths.report_paths.more_tram_msoa,
            more_tram_north=paths.report_paths.more_tram_north,
            mode_adj_factors=paths.report_paths.mode_adj_factors,
            report_segment_total_paths=vector_reports.segment_total,
            report_ca_sector_paths=vector_reports.ca_sector,
            report_ie_sector_paths=vector_reports.ie_sector,
            lad_report_paths=vector_reports.lad_report,
            balance_paths=self.nhb_production.export_paths.notem_segmented,
            split_weekday_weekend=False,
        )

        # Generate a LAD report before and after tram infill
        if before_after_report:
            self._before_after_report(
                before_lad_report_paths=before_lad_report_paths,
                after_lad_report_paths=vector_reports.lad_report,
                output_paths=paths.report_paths.comparison_report,
                tram_only_output_paths=paths.report_paths.tram_comparison_report,
            )

    def _infill_tram(self,
                     trip_end: pd.DataFrame,
                     tram_data: pd.DataFrame,
                     ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Internal function on self._add_tram()"""
        tram_zones = tram_data['msoa_zone_id'].unique().tolist()
        north_zones = self.zoning_system.internal_zones
        non_north_zones = self.zoning_system.external_zones

        # Runs tram infill for 275 internal tram msoa zones
        tram_zone_infilled, more_tram_report = self._infill_tram_zones(
            trip_end=trip_end,
            tram_data=tram_data,
            tram_zones=tram_zones,
            tram_competitors=self.tram_competitors,
        )

        # Runs tram infill for entire north
        tram_north_infilled, more_tram_north_report = self._infill_tram_north(
            trip_end=trip_end,
            tram_data=tram_data,
            north_zones=north_zones,
            tram_competitors=self.tram_competitors,
        )

        # Runs tram infill for internal non tram msoa zones
        # Returns the untouched external zones too
        non_tram_infilled, non_tram_adj_factors = self._infill_non_tram_zones(
            trip_end=trip_end,
            tram_zone_infilled=tram_zone_infilled,
            tram_north_infilled=tram_north_infilled,
            tram_zones=tram_zones,
            north_zones=north_zones,
            non_north_zones=non_north_zones,
            tram_competitors=self.tram_competitors,
        )

        df_list = [non_tram_infilled, tram_zone_infilled]
        tram_infilled_vector = pd.concat(df_list, ignore_index=True)

        return (
            tram_infilled_vector,
            more_tram_report,
            more_tram_north_report,
            non_tram_adj_factors
        )

    def _add_tram(self,
                  tram_import_path: nd.PathLike,
                  trip_origin: str,
                  vector_import_paths: Dict[int, nd.PathLike],
                  export_paths: Dict[int, nd.PathLike],
                  tram_growth_factors: Dict[int, nd.PathLike],
                  more_tram_msoa: Dict[int, nd.PathLike],
                  more_tram_north: Dict[int, nd.PathLike],
                  mode_adj_factors: Dict[int, nd.PathLike],
                  report_segment_total_paths: Dict[int, nd.PathLike],
                  report_ca_sector_paths: Dict[int, nd.PathLike],
                  report_ie_sector_paths: Dict[int, nd.PathLike],
                  lad_report_paths: Dict[int, nd.PathLike],
                  balance_paths: Dict[int, nd.PathLike] = None,
                  split_weekday_weekend: bool = True,
                  ) -> None:
        """
        TODO(BT): Update Tram Docs
        Runs the tram inclusion for the notem trip end output.

        Completes the following steps for each year:
            - Reads in the base year tram data given.
            - Reads in the notem segmented trip end output compressed pickle
              given.
            - Aggregates above to the required segmentation.
            - Runs the tram infill for the 275 internal msoa zones with tram data.
            - Calculates the future growth of tram based on rail growth and applies it to tram.
            - Runs the tram infill for the entire north.
            - Runs the tram infill for the remaining internal non tram msoa zones
            - Combines the tram infill with the external msoa zone data and
              converts them to Dvector.
            - Disaggregates the Dvector to the notem segmentation.

        Parameters
        ----------
        tram_import_path:
            The path to the base year tram data.

        trip_origin:
            Whether the trip origin is 'hb' or 'nhb'.

        vector_import_paths:
            Dictionary of {year: notem_segmented_trip_end data} pairs.

        export_paths:
            Path with filename for tram included notem segmented trip end output.

        balance_paths:
            Dictionary of {year: balance_path} pairs. provides the paths to
            balance the final produced vector to. Usually used when infilling
            attractions, so they can be balanced back to the productions.

        Returns
        -------
        None

        """
        # Init
        start_time = timing.current_milli_time()

        # Check that the paths we need exist!
        file_ops.check_file_exists(tram_import_path)
        [file_ops.check_file_exists(x) for x in vector_import_paths.values()]

        # Load in the base year data
        base_vector = self._read_vector_data(
            trip_origin=trip_origin,
            vector_path=vector_import_paths[self.base_year],
        )

        # Add Tram in for each year, save the output and report
        for year in self.years:
            year_start = timing.current_milli_time()

            # Load in the starting vector
            starting_trip_end = self._read_vector_data(
                trip_origin=trip_origin,
                vector_path=vector_import_paths[year],
            )

            # Load in the tram data, and grow into this year
            tram_data = self._read_tram_data(tram_import_path)
            tram_data, growth_factors = self._grow_tram_by_rail(
                base_data=base_vector,
                future_data=starting_trip_end,
                base_tram=tram_data,
            )

            # Replace val with future val
            tram_data = tram_data.drop(columns=self._val_col)
            tram_data = tram_data.rename(columns={'future_val': self._val_col})

            # Infill the tram data into tram and non-tram areas
            returns = self._infill_tram(
                trip_end=starting_trip_end,
                tram_data=tram_data,
            )

            tram_infilled_vector = returns[0]
            more_tram_report, more_tram_north_report, non_tram_adj_factors = returns[1:]

            # ## WRITE OUT RUNNING REPORTS ## #
            growth_factors.to_csv(tram_growth_factors[year], index=False)
            more_tram_report.to_csv(more_tram_msoa[year], index=False)
            more_tram_north_report.to_csv(more_tram_north[year], index=False)
            non_tram_adj_factors.to_csv(mode_adj_factors[year], index=False)

            # ## MAKE SURE NOTHING HAS BEEN DROPPED ## #
            # We want to do this first as it's faster with less segments
            expected_total = starting_trip_end[self._val_col].values.sum()
            final_total = tram_infilled_vector[self._val_col].values.sum()
            if not math_utils.is_almost_equal(expected_total, final_total, rel_tol=0.001):
                raise ValueError(
                    "Some demand seems to have gone missing while infilling tram!\n"
                    "Starting demand: %s\n"
                    "Ending demand: %s"
                    % (expected_total, final_total)
                )

            # ## CONVERT TO DVECTOR ## #
            # TODO(NK, BT): This can be done much smarter! Class variables
            #  should know what this looks like before and after
            if trip_origin == 'hb':
                tram_seg = nd.get_segmentation_level('hb_p_m7_ca')
                out_seg = nd.get_segmentation_level('tram_hb_output')
                lad_report_seg = nd.get_segmentation_level('hb_p_m7_tp_week')

            elif trip_origin == 'nhb':
                tram_seg = nd.get_segmentation_level('nhb_p_m7_ca')
                out_seg = nd.get_segmentation_level('tram_nhb_output')
                lad_report_seg = nd.get_segmentation_level('nhb_p_m7_tp_week')

            else:
                raise ValueError(
                    "trip_origin is not the correct type. "
                    "Expected trip_origin is hb or nhb, got %s"
                    % trip_origin
                )

            tram_dvec = nd.DVector(
                zoning_system=nd.get_zoning_system('msoa'),
                segmentation=tram_seg,
                import_data=tram_infilled_vector,
                zone_col=self._zoning_system_col,
                val_col=self._val_col,
            )

            # ## CONVERT BACK TO ORIGINAL SEGMENTATION ## #
            # Original DVec at full segmentation
            orig_dvec = nd.DVector.load(vector_import_paths[year])

            # Need to add in m7 - get its segments from rail
            orig_dvec = orig_dvec.duplicate_segment_like(
                segment_dict={'m': nd.Mode.TRAM.get_mode_num()},
                like_segment_dict={'m': nd.Mode.BUS.get_mode_num()},
                out_segmentation=out_seg,
            )

            # Add segments back in from original input
            tram_dvec = tram_dvec.split_segmentation_like(orig_dvec, zonal_average=False)

            # Balance if paths given
            if balance_paths is not None:
                tram_dvec = self.balance_dvecs(
                    dvec=tram_dvec,
                    trip_origin=trip_origin,
                    balance=nd.DVector.load(balance_paths[year]),
                    split_weekday_weekend=split_weekday_weekend,
                )

            # ## WRITE OUT THE DVEC AND REPORTS ## #
            self._logger.info("Writing Produced Tram data to disk")
            tram_dvec.save(export_paths[year])

            self._logger.info("Writing Tram reports to disk")
            tram_dvec.write_sector_reports(
                segment_totals_path=report_segment_total_paths[year],
                ca_sector_path=report_ca_sector_paths[year],
                ie_sector_path=report_ie_sector_paths[year],
                lad_report_path=lad_report_paths[year],
                lad_report_seg=lad_report_seg,
            )

            # Print timing for each trip end model
            trip_end_end = timing.current_milli_time()
            time_taken = timing.time_taken(year_start, trip_end_end)
            self._logger.info("Tram Model for year %d took: %s\n" % (year, time_taken))

        # End timing
        end_time = timing.current_milli_time()
        time_taken = timing.time_taken(start_time, end_time)
        self._logger.info("Tram Model took: %s" % time_taken)

    def balance_dvecs(self,
                      dvec: nd.DVector,
                      trip_origin: str,
                      balance: nd.DVector,
                      split_weekday_weekend: bool,
                      ) -> nd.DVector:
        """Balances given Dvec to balance using dvec.balance_at_segments()"""
        # Init
        if trip_origin == 'hb':
            balance_zoning = self.hb_balance_zoning
        elif trip_origin == 'nhb':
            balance_zoning = self.nhb_balance_zoning
        else:
            raise ValueError(
                "Trip origin '%s' is not recognised" % trip_origin
            )

        balanced_dvec = dvec.balance_at_segments(
            balance,
            balance_zoning=balance_zoning,
            split_weekday_weekend=split_weekday_weekend,
        )

        # ## ATTRACTIONS TOTAL CHECK ## #
        if not balanced_dvec.sum_is_close(balance):
            msg = (
                "The vector total after balancing is not similar enough. Are "
                "some zones being dropped in the zonal translation?\n"
                "Expected %f\n"
                "Got %f"
                % (balance.sum(), balanced_dvec.sum())
            )
            self._logger.warning(msg)

        return balanced_dvec

    @staticmethod
    def _calculate_mode_shares(df, group_cols, val_cols):
        df = df.copy()

        for col in val_cols:
            # Create the sum of the group
            sum_col = '%s_sum' % col
            share_col = '%s_share' % col
            df[sum_col] = df.groupby(group_cols)[col].transform('sum')

            # Get the share
            df[share_col] = df[col] / df[sum_col]
            df = df.drop(columns=sum_col)

        return df

    def _before_after_report(self,
                             before_lad_report_paths: Dict[int, nd.PathLike],
                             after_lad_report_paths: Dict[int, nd.PathLike],
                             output_paths: Dict[int, nd.PathLike],
                             tram_only_output_paths: Dict[int, nd.PathLike],
                             ) -> None:
        """Generates reports comparing before and after reports"""
        lad_zoning = nd.get_zoning_system(self._lad_report_zoning_name)
        merge_cols = [lad_zoning.col_name] + self._lad_comparison_col_names

        for year in self.years:
            # Read in and tidy reports
            def get_df(path, col_name):
                df = nd.read_df(path)
                df = df.rename(columns={self._val_col: col_name})
                return pd_utils.reindex_and_groupby_sum(
                    df=df,
                    index_cols=merge_cols + [col_name],
                    value_cols=[col_name]
                )
            before_report = get_df(before_lad_report_paths[year], 'before')
            after_report = get_df(after_lad_report_paths[year], 'after')

            # Merge
            report = pd.merge(
                before_report,
                after_report,
                how='outer',
                on=merge_cols
            ).fillna(0)

            # Set up some col definitions
            group_cols = merge_cols.copy()
            group_cols.remove('m')
            val_cols = ['before', 'after']

            # Get report of all LADs
            all_report = self._calculate_mode_shares(report, group_cols, val_cols)
            nd.write_df(all_report, output_paths[year], index=False)

            # Filter down to only the zones with tram data
            mask = (report['m'] == self._tram_mode) & (report['after'] > 0)
            tram_zones = report[mask][lad_zoning.col_name].unique().tolist()

            # Generate a tram only report
            mask = report[lad_zoning.col_name].isin(tram_zones)
            tram_report = report[mask].copy()
            tram_report = self._calculate_mode_shares(tram_report, group_cols, val_cols)
            nd.write_df(tram_report, tram_only_output_paths[year], index=False)

    def _read_tram_data(self,
                        tram_data_path: nd.PathLike,
                        ) -> pd.DataFrame:
        """
        Reads in the tram and vector data.

        Parameters
        ----------

        tram_data_path:
            The path to the base year tram data.

        Returns
        -------
        tram_data:
            Returns the tram data as dataframe.
        """
        # Init
        tram_target_cols = self._target_col_dtypes['tram']

        # Read in dataframe
        tram_data = file_ops.read_df(path=tram_data_path, find_similar=True)
        tram_data['m'] = nd.Mode.TRAM.get_mode_num()
        tram_data = pd_utils.reindex_cols(tram_data, tram_target_cols)

        # Make sure the input data is in the correct data types
        for col, dt in self._target_col_dtypes['tram'].items():
            tram_data[col] = tram_data[col].astype(dt)

        tram_data.rename(columns={'trips': self._val_col}, inplace=True)

        return tram_data

    def _read_vector_data(self,
                          trip_origin: str,
                          vector_path: nd.PathLike
                          ) -> pd.DataFrame:
        """
        Reads in the tram and vector data.

        Parameters
        ----------
        trip_origin:
            Whether the trip origin is 'hb' or 'nhb'.

        vector_path:
            The path to DVector pickle to read in

        Returns
        -------
        vector:
            Returns the DVector in tram segmentation ['p','m','ca'], converted
            to a pandas dataframe.
        """
        # Read in the vector
        vector_dvec = nd.DVector.load(vector_path)

        # Aggregate the dvector to the required segmentation
        if trip_origin == 'hb':
            seg_name = self._hb_tram_seg
        elif trip_origin == 'nhb':
            seg_name = self._nhb_tram_seg
        else:
            raise ValueError(
                "trip_origin is not the correct type. Expected trip_origin "
                "either 'hb' or 'nhb'. Got %s"
                % trip_origin
            )

        # Aggregate
        segmentation = nd.get_segmentation_level(seg_name)
        vector_dvec = vector_dvec.aggregate(segmentation)

        # Convert to a dataframe and return
        df = vector_dvec.to_df()
        return df.rename(columns={vector_dvec.val_col: self._val_col})

    def _infill_tram_zones(self,
                           trip_end: pd.DataFrame,
                           tram_data: pd.DataFrame,
                           tram_zones: List[Any],
                           tram_competitors: List[nd.Mode],
                           mode_col: str = 'm',
                           ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Tram infill for the MSOAs with tram data in the north.

        Parameters
        ----------
        trip_end:
            A pandas DataFrame of the date that we should be infilling with
            tram data
            
        tram_data:
            Dataframe containing tram data at MSOA level for the north.

        tram_zones:
            A list of the zones that contain tram data. Must be the same format
            as vector[zone_col] and tram_data[zone_col]

        tram_competitors:
            A list of the Modes which would be competing with Tram for trips.
            These are the modes which will be used to remove trips from in
            order to add in tram trips

        mode_col:
            The name of the columns in notem_tram_seg and tram_data that
            refers to the mode segment.

        Returns
        -------
        infilled_tram_zones:
            Returns the dataframe after tram infill.

        more_tram_report:
            A report showing where there was more tram predicted than rail
            trips in vector
        """
        # Init
        tram_data = tram_data.copy()

        # Keep only the vector data in tram zones
        mask = trip_end[self._zoning_system_col].isin(tram_zones)
        trip_end = trip_end[mask].copy()

        # Infills tram data
        infilled_tram_zones, more_tram_report = self._infill_internal(
            trip_end=trip_end,
            tram_vector=tram_data,
            tram_competitors=tram_competitors,
            non_val_cols=[self._zoning_system_col] + self._tram_segment_cols,
            mode_col=mode_col,
        )

        return infilled_tram_zones, more_tram_report

    def _infill_tram_north(self,
                           trip_end: pd.DataFrame,
                           tram_data: pd.DataFrame,
                           north_zones: List[Any],
                           tram_competitors: List[nd.Mode],
                           mode_col: str = 'm',
                           ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Tram infill for the MSOAs with tram data in the north.

        Parameters
        ----------
        trip_end:
            A pandas DataFrame of the date that we should be infilling with
            tram data

        tram_data:
            Dataframe containing tram data at MSOA level for the north.

        tram_competitors:
            A list of the Modes which would be competing with Tram for trips.
            These are the modes which will be used to remove trips from in
            order to add in tram trips

        mode_col:
            The name of the columns in notem_tram_seg and tram_data that
            refers to the mode segment.

        Returns
        -------
        infilled_tram_zones:
            Returns the dataframe after tram infill.

        more_tram_report:
            A report showing where there was more tram predicted than rail
            trips in vector
        """

        # Init
        tram_data = tram_data.copy()

        # Keep only the vector data in tram zones
        mask = trip_end[self._zoning_system_col].isin(north_zones)
        trip_end = trip_end[mask].copy()

        # Aggregate tram and vector data to north level
        index_cols = self._tram_segment_cols + [self._val_col]

        trip_end = pd_utils.reindex_and_groupby_sum(trip_end, index_cols, [self._val_col])
        tram_data = pd_utils.reindex_and_groupby_sum(tram_data, index_cols, [self._val_col])

        # Infills tram data
        tram_north_infilled, more_tram_report = self._infill_internal(
            trip_end=trip_end,
            tram_vector=tram_data,
            tram_competitors=tram_competitors,
            non_val_cols=self._tram_segment_cols,
            mode_col=mode_col,
        )

        return tram_north_infilled, more_tram_report

    def _grow_tram_by_rail(self,
                           base_data: pd.DataFrame,
                           future_data: pd.DataFrame,
                           base_tram: pd.DataFrame,
                           future_tram_col: str = 'future_val',
                           ) -> pd.DataFrame:
        """
        Calculation of future year tram growth

        Parameters
        ----------
        base_data:
            Dataframe containing base year rail data.

        future_data:
            Dataframe containing future year rail data.

        base_tram:
            Dataframe containing base year tram data.

        future_tram_col:
            Column name

        Returns
        -------
        future_tram:
            base_tram with a new column added named future_tram_col. This
            will be the same as base_tram * growth_factors. Note that the
            return dataframe might not be in the same order as the
            passed in base_tram

        growth_factors:
            The growth factors used to generate future year tram
        """
        # TODO(BT): EFS style handling of growth
        # Init
        segment_cols = du.list_safe_remove(self._tram_segment_cols, ['m'])
        join_cols = [self._zoning_system_col] + segment_cols
        index_cols = join_cols + [self._val_col]

        # ## GET ALL DATA INTO ONE DF ## #
        # Tidy up
        def tidy_df(df, name, mode):
            df = df.copy()
            df = df[df['m'] == mode].copy()
            df = df.drop(columns=['m'])
            df = pd_utils.reindex_cols(df, index_cols)
            df = df.sort_values(join_cols)
            df = df.rename(columns={self._val_col: name})
            return df

        base_rail = tidy_df(base_data, 'base_rail', self._train_mode)
        future_rail = tidy_df(future_data, 'future_rail', self._train_mode)
        base_tram = tidy_df(base_tram, 'base_tram', self._tram_mode)

        # Merge all together
        all_data = pd.merge(base_rail, future_rail, on=join_cols, how='outer').fillna(0)
        all_data = pd.merge(all_data, base_tram, on=join_cols, how='right').fillna(0)

        # ## CALCULATE GROWTH FACTORS ## #
        all_data['rail_growth'] = np.divide(
            all_data['future_rail'].values,
            all_data['base_rail'].values,
            out=np.ones_like(all_data['base_rail'].values),
            where=all_data['base_rail'].values != 0,
        )

        # Calculate the future tram and tidy up
        all_data[future_tram_col] = all_data['base_tram'] * all_data['rail_growth']
        future_tram = pd_utils.reindex_cols(all_data, join_cols + ['base_tram', future_tram_col])
        future_tram = future_tram.rename(columns={'base_tram': self._val_col})
        future_tram['m'] = self._tram_mode

        # Extract the growth factors alone
        growth_df = pd_utils.reindex_cols(all_data, join_cols + ['rail_growth'])

        return future_tram, growth_df

    def _infill_non_tram_zones(self,
                               trip_end: pd.DataFrame,
                               tram_zone_infilled: pd.DataFrame,
                               tram_north_infilled: pd.DataFrame,
                               tram_zones: List[str],
                               north_zones: List[str],
                               non_north_zones: List[str],
                               tram_competitors: List[nd.Mode],
                               mode_col: str = 'm',
                               ):
        """
        Infills tram for MSOAs without tram data in the internal area (north area).

        Parameters
        ----------
        trip_end:
            The vector in infill the tram data into

        tram_zone_infilled:
            Vector, infilled with tram_data, at the zone level

        tram_north_infilled:
            Vector, infilled with tram_data at the north level

        tram_zones:
            A list of the zones that contain tram data

        north_zones:
            A list of the zones within the North

        non_north_zones:
            A list of the zones not within the North

        tram_competitors:
            A list of the Modes which would be competing with Tram for trips.
            These are the modes which will be used to remove trips from in
            order to add in tram trips

        mode_col:
            The name of the columns in north_wo_infill, non_tram_north, and
            dvec_tram_seg that refers to the mode segment.

        Returns
        -------
        non_tram_infilled:
            Returns the dataframe after non-tram infill for msoa zones for
            all external area, but only non-tram zones for internal area
        """
        # Init
        compet_mode_vals = [x.get_mode_num() for x in tram_competitors]
        non_val_cols = [self._zoning_system_col] + self._tram_segment_cols

        # ## SPLIT ORIGINAL VECTOR INTO PARTS ## #
        # Split the original vector into north and non-north
        non_north_mask = trip_end[self._zoning_system_col].isin(non_north_zones)
        non_north_vector = trip_end[non_north_mask].copy()

        north_mask = trip_end[self._zoning_system_col].isin(north_zones)
        north_vector = trip_end[north_mask].copy()

        # Filter down to non-tram zones
        tram_mask = north_vector[self._zoning_system_col].isin(tram_zones)
        north_no_tram_vector = north_vector[~tram_mask].copy()

        # ## STEP 1: CALCULATE NORTH AVERAGE MODE SHARE ADJUSTMENTS ## #
        # Aggregate vectors to northern level
        index_cols = self._tram_segment_cols + [self._val_col]

        kwargs = {'index_cols': index_cols, 'value_cols': [self._val_col]}
        non_tram_north = pd_utils.reindex_and_groupby_sum(north_no_tram_vector, **kwargs)
        agg_tram_zone_infilled = pd_utils.reindex_and_groupby_sum(tram_zone_infilled, **kwargs)

        # CALCULATE THE ADJUSTED NON-TRAM NORTH AVERAGE
        agg_tram_zone_infilled.rename(columns={self._val_col: 'tram_val'}, inplace=True)
        adj_non_tram_north = pd.merge(
            left=tram_north_infilled,
            right=agg_tram_zone_infilled,
            how='outer',
            on=self._tram_segment_cols,
        ).fillna(0)

        # Calculate, then keep only what we need
        adj_non_tram_north[self._val_col] -= adj_non_tram_north['tram_val']
        cols = self._tram_segment_cols + [self._val_col]
        adj_non_tram_north = pd_utils.reindex_cols(adj_non_tram_north, cols)

        # CALCULATE THE DIFFERENCE BETWEEN ADJUSTED AND ORIGINAL
        # Remove tram trips for merge
        adj_non_tram_north = adj_non_tram_north.rename(columns={self._val_col: 'adj_val'})
        tram_mask = adj_non_tram_north[mode_col] == self._tram_mode
        adj_non_tram_north = adj_non_tram_north[~tram_mask]

        # Stick into one df
        north_df = pd.merge(
            left=non_tram_north,
            right=adj_non_tram_north,
            how='outer',
            on=self._tram_segment_cols,
        ).fillna(0)

        # Filter down to just the competitor modes
        compet_mask = north_df[mode_col].isin(compet_mode_vals)
        north_df = north_df[compet_mask].copy()

        # Calculate the average adjustment factor
        north_df['adj_factor'] = north_df['adj_val'] / north_df[self._val_col]

        # Filter down to all we need
        cols = self._tram_segment_cols + ['adj_factor']
        non_tram_adj_factors = pd_utils.reindex_cols(north_df, cols)

        # ## STEP 2. ADJUST NON-TRAM MSOA BY AVERAGE NORTH ADJUSTMENT ## #
        # Attach the avg north adj factors
        north_no_tram_vector = pd.merge(
            left=north_no_tram_vector,
            right=non_tram_adj_factors,
            how='left',
            on=self._tram_segment_cols,
        ).fillna(1)

        # Adjust!
        north_no_tram_vector['new_val'] = north_no_tram_vector[self._val_col].copy()
        north_no_tram_vector['new_val'] *= north_no_tram_vector['adj_factor'].copy()
        north_no_tram_vector = north_no_tram_vector.drop(columns='adj_factor')

        # ## STEP 3. APPLY NEW MSOA MODE SHARES TO OLD TOTALS ## #
        # split into competitor and non-competitor
        compet_mask = north_no_tram_vector[mode_col].isin(compet_mode_vals)
        compet_df = north_no_tram_vector[compet_mask].copy()
        non_compet_df = north_no_tram_vector[~compet_mask].copy()

        # Calculate the new mode shares
        group_cols = du.list_safe_remove(non_val_cols, [mode_col])
        temp = compet_df.groupby(group_cols)
        compet_df['val_sum'] = temp[self._val_col].transform('sum')
        compet_df['new_val_sum'] = temp['new_val'].transform('sum')
        compet_df['new_mode_shares'] = compet_df['new_val'] / compet_df['new_val_sum']

        # Adjust new_vals to reflect old totals and new mode shares
        compet_df['new_val'] = compet_df['new_mode_shares'] * compet_df['val_sum']

        # Stick the competitor and non-competitor back together
        compet_df = pd_utils.reindex_cols(compet_df, list(non_compet_df))
        df_list = [compet_df, non_compet_df]
        north_no_tram_vector = pd.concat(df_list, ignore_index=True)

        # ## STEP 4. TIDY UP DFs. BRING EVERYTHING BACK TOGETHER ## #
        # Add the external back on
        non_north_vector['new_val'] = non_north_vector[self._val_col].copy()
        df_list = [north_no_tram_vector, non_north_vector]
        adj_non_tram_vector = pd.concat(df_list, ignore_index=True)

        # Check we haven't dropped anything!
        expected_total = adj_non_tram_vector[self._val_col].values.sum()
        final_total = adj_non_tram_vector['new_val'].values.sum()
        if not math_utils.is_almost_equal(expected_total, final_total):
            raise ValueError(
                "Some demand seems to have gone missing while infilling "
                "non tram zones!\n"
                "Starting demand: %s\n"
                "Ending demand: %s"
                % (expected_total, final_total)
            )

        new_df = adj_non_tram_vector.drop(columns=[self._val_col])
        new_df = new_df.rename(columns={'new_val': self._val_col})
        new_df = new_df.sort_values(non_val_cols).reset_index(drop=True)

        return new_df, non_tram_adj_factors

    def _infill_internal(self,
                         trip_end: pd.DataFrame,
                         tram_vector: pd.DataFrame,
                         tram_competitors: List[nd.Mode],
                         non_val_cols: List[str],
                         mode_col: str = 'm',
                         ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Creates a subset of the given dataframe using a condition.

        Parameters
        ----------
        trip_end:
            The original Vector to add tram_vector into.

        tram_vector:
            the vector of tram data to infill into vector.

        tram_competitors:
            The modes which are considered competitors of tram. These modes
            will be adjusted to infill the tram trips

        Returns
        -------
        notem_sub_df:
            Returns the subset of the original dataframe.

        more_tram_report:
            A report of all the places tram trips were higher than rail trips

        """
        # Init
        df = pd.concat([trip_end, tram_vector], ignore_index=True)

        # Create needed masks
        train_mask = df[mode_col] == nd.Mode.TRAIN.get_mode_num()
        tram_mask = df[mode_col] == nd.Mode.TRAM.get_mode_num()

        # Get just the train and tram trips
        train_df = df[train_mask].copy().reset_index(drop=True)
        tram_df = df[tram_mask].copy().reset_index(drop=True)

        # ## REMOVE TRAM TRIPS FROM TRAIN ## #
        common_cols = du.list_safe_remove(non_val_cols, [mode_col])

        # Combine into one df for comparisons
        train_df.drop(columns=mode_col, inplace=True)
        train_df.rename(columns={self._val_col: 'train'}, inplace=True)

        tram_df.drop(columns=mode_col, inplace=True)
        tram_df.rename(columns={self._val_col: 'tram'}, inplace=True)

        tram_train_df = pd.merge(
            left=train_df,
            right=tram_df,
            on=common_cols,
            how='outer',
        ).fillna(0)

        # Generate a report where there are more tram than train trips
        more_tram_than_train = tram_train_df['tram'] > tram_train_df['train']
        more_tram_report = tram_train_df[more_tram_than_train].copy()

        # TODO(BT): Report where tram is > 50% rail

        # Curtail tram trips where they're higher than train
        tram_train_df['new_tram'] = tram_train_df['tram'].mask(
            more_tram_than_train,
            tram_train_df['train'],
        )

        # Remove tram trips from train
        tram_train_df['new_train'] = tram_train_df['train'].copy()
        tram_train_df['new_train'] -= tram_train_df['new_tram']

        # Get back into input format
        train_cols = common_cols + ['train', 'new_train']
        train_df = pd_utils.reindex_cols(tram_train_df, train_cols)
        train_df['m'] = nd.Mode.TRAIN.get_mode_num()
        rename = {'train': self._val_col, 'new_train': 'new_val'}
        train_df.rename(columns=rename, inplace=True)

        tram_cols = common_cols + ['tram', 'new_tram']
        tram_df = pd_utils.reindex_cols(tram_train_df, tram_cols)
        tram_df['m'] = nd.Mode.TRAM.get_mode_num()
        rename = {'tram': self._val_col, 'new_tram': 'new_val'}
        tram_df.rename(columns=rename, inplace=True)

        # ## ADD EVERYTHING TO ONE DF READY FOR MODE SHARE ADJUSTMENT ## #
        # Get everything other than tram and train
        other_df = df[~(train_mask | tram_mask)].copy()
        other_df['new_val'] = other_df[self._val_col].copy()

        # Stick back together
        df_list = [other_df, train_df, tram_df]
        new_df = pd.concat(df_list, ignore_index=True)

        # ## ADJUST COMPETITOR MODES BACK TO ORIGINAL SHARES ##
        # Split into competitor and non-competitor dfs
        compet_mode_vals = [x.get_mode_num() for x in tram_competitors]
        compet_mask = new_df[mode_col].isin(compet_mode_vals)
        compet_df = new_df[compet_mask].copy()
        non_compet_df = new_df[~compet_mask].copy()

        # Calculate the old mode shares
        temp = compet_df.groupby(common_cols)
        compet_df['val_sum'] = temp[self._val_col].transform('sum')
        compet_df['new_val_sum'] = temp['new_val'].transform('sum')
        compet_df['old_mode_shares'] = compet_df[self._val_col] / compet_df['val_sum']

        # Adjust new_vals to reflect old mode shares
        compet_df['new_val'] = compet_df['old_mode_shares'] * compet_df['new_val_sum']

        # ## TIDY UP DF FOR RETURN ## #
        compet_df = pd_utils.reindex_cols(compet_df, list(non_compet_df))
        new_df = pd.concat([compet_df, non_compet_df], ignore_index=True)

        # Check we haven't dropped anything
        expected_total = trip_end[self._val_col].values.sum()
        final_total = new_df['new_val'].values.sum()
        if not math_utils.is_almost_equal(expected_total, final_total):
            raise ValueError(
                "Some demand seems to have gone missing while infilling "
                "tram!\n"
                "Starting demand: %s\n"
                "Ending demand: %s"
                % (expected_total, final_total)
            )

        new_df = new_df.drop(columns=[self._val_col])
        new_df = new_df.rename(columns={'new_val': self._val_col})
        new_df = new_df.sort_values(non_val_cols).reset_index(drop=True)

        return new_df, more_tram_report
