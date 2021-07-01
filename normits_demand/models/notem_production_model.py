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

# Third party imports
import pandas as pd

# local imports
import normits_demand as nd

from normits_demand import core
from normits_demand import efs_constants as consts

from normits_demand.utils import general as du
from normits_demand.utils import file_ops as ops
from normits_demand.utils import timing


class HBProductionModel:
    _trip_origin = 'hb'
    _zoning_system = 'msoa'
    _pure_demand = 'pure_demand'
    _fully_segmented = 'fully_segmented'
    _aggregated = 'aggregated'

    # Define wanted columns
    _target_cols = {
        'land_use': ['msoa_zone_id', 'area_type', 'tfn_traveller_type', 'people'],
        'trip_rate': ['tfn_tt', 'tfn_at', 'p', 'trip_rate'],
        'm_tp': ['p', 'tfn_tt', 'tfn_at', 'm', 'tp', 'split'],
    }

    # Define segment renames needed
    seg_rename = {
        'tfn_traveller_type': 'tfn_tt',
        'area_type': 'tfn_at',
    }

    def __init__(self,
                 land_use_paths: Dict[int, nd.PathLike],
                 trip_rates_path: str,
                 mode_time_splits_path: str,
                 constraint_paths: Dict[int, nd.PathLike],
                 export_path: str,
                 process_count: int = consts.PROCESS_COUNT
                 ):
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
        paths = self.create_output_paths(self.export_path, self.years)
        self.pure_demand_out, self.fully_segmented_out, self.aggregated_out = paths

        self.pure_demand_totals_out, self.pure_demand_sec_totals_out, \
            self.pure_demand_ie_totals_out = self.create_pure_dem_report_paths(self.report_path, self.years)

        self.fully_seg_totals_out, self.fully_seg_sec_totals_out, \
            self.fully_seg_ie_totals_out = self.create_fully_seg_report_paths(self.report_path, self.years)

    def run(self,
            export_pure_demand: bool = False,
            reports: bool = True,
            output_fully_segmented: bool = False,
            output_aggregated: bool = False,
            verbose: bool = True,
            ):
        """
        Runs the HB Production model

        Parameters
        ----------
        export_pure_demand:
            Whether to output the pure demand

        reports:
            Whether to output reports while running.

        output_fully_segmented:
            Whether to output the fully segmented hb productions before aggregating to
            the required segmentation and mode.

        output_aggregated:
            Whether to output the aggregated hb productions

        verbose:
            Whether to print progress bars during processing or not.

        Returns
        -------
        HB_Productions:
            HB productions for the mode and segmentation needed
        """

        # Initialise timing
        # TODO(BT): Properly integrate logging
        start_time = timing.current_milli_time()
        du.print_w_toggle("Starting HB Production Model at: %s" % timing.get_datetime(),
                          verbose=verbose)

        for year in self.years:
            du.print_w_toggle("Loading the population data...", verbose=verbose)
            pop_dvec = self._read_land_use_data(year, verbose=verbose)

            du.print_w_toggle("Population generated. Converting to productions...", verbose=verbose)
            pure_demand = self.generate_productions(
                pop_dvec=pop_dvec,
                verbose=verbose,
            )

            if export_pure_demand:
                du.print_w_toggle("Writing pure demand productions to disk...", verbose=verbose)
                pure_demand.to_pickle(self.pure_demand_out[year])

            # Reporting pure demand
            if reports:
                print('\n', '-' * 15, 'Writing reports at pure demand', '-' * 15)
                print("Total Productions for year %d: %.4f" % (year, pure_demand.sum()))
                # msoa level output
                tfn_agg_at_seg = nd.get_segmentation_level('pure_demand_reporting')

                pure_demand_vec = pure_demand.aggregate(tfn_agg_at_seg)
                # pure_demand_vec = pure_demand_vec.sum_zoning()
                pure_demand_vec_df = pure_demand_vec.to_df()
                pure_demand_vec_df.to_csv(self.pure_demand_totals_out[year], index=False)
                # sector level output
                tfn_ca_sectors = nd.get_zoning_system('ca_sector_2020')
                pure_demand_ca = pure_demand_vec.translate_zoning(tfn_ca_sectors)
                pure_demand_ca = pure_demand_ca.to_df()
                pure_demand_ca.to_csv(self.pure_demand_sec_totals_out[year], index=False)
                # ie level output
                ie_sectors = nd.get_zoning_system('ie_sector')
                pure_demand_ie = pure_demand_vec.translate_zoning(ie_sectors)
                pure_demand_ie = pure_demand_ie.to_df()
                pure_demand_ie.to_csv(self.pure_demand_ie_totals_out[year], index=False)

            # SPLIT PRODUCTIONS BY MODE AND TIME ## #
            du.print_w_toggle("Splitting HB productions by mode and time...", verbose=verbose)
            hb_prods = self._split_by_tp_and_mode(pure_demand, verbose=verbose)

            # Output productions before any aggregation
            if output_fully_segmented:
                du.print_w_toggle("Writing fully segmented productions to disk...", verbose=verbose)
                hb_prods.to_pickle(self.fully_segmented_out[year])

            # Reporting fully segmented productions
            if reports:
                print('\n', '-' * 15, 'Writing reports after full segmentation', '-' * 15)
                # msoa level output
                notem_full_tfn = nd.get_segmentation_level('hb_notem_full_tfn')

                fully_seg_vec = hb_prods.aggregate(notem_full_tfn, split_tfntt_segmentation=True)
                print("Total Productions for year %d: %.4f" % (year, fully_seg_vec.sum()))
                fully_seg_vec_df = fully_seg_vec.to_df()
                fully_seg_vec_df.to_csv(self.fully_seg_totals_out[year], index=False)
                # sector level output
                tfn_ca_sectors = nd.get_zoning_system('ca_sector_2020')
                fully_seg_ca = fully_seg_vec.translate_zoning(tfn_ca_sectors)
                fully_seg_ca = fully_seg_ca.to_df()
                fully_seg_ca.to_csv(self.fully_seg_sec_totals_out[year], index=False)
                # ie level output
                ie_sectors = nd.get_zoning_system('ie_sector')
                fully_seg_ie = fully_seg_vec.translate_zoning(ie_sectors)
                fully_seg_ie = fully_seg_ie.to_df()
                fully_seg_ie.to_csv(self.fully_seg_ie_totals_out[year], index=False)

            if output_aggregated:
                # TODO: Aggregate segments
                agg_hb_prods = hb_prods  # aggregate(optional_segmentation)
                du.print_w_toggle("Writing aggregated productions to file...", verbose=verbose)
                agg_hb_prods.to_pickle(self.aggregated_out[year])

            # TODO: Bring in constraints (Validation)
            #  Output some audits of what demand was before and after control
            #  By segment.

            # End timing
            end_time = timing.current_milli_time()
            du.print_w_toggle("Finished HB Production Model at: %s" % timing.get_datetime(),
                              verbose=verbose)
            du.print_w_toggle("HB Production Model took: %s"
                              % timing.time_taken(start_time, end_time), verbose=verbose)

    def _read_land_use_data(self, year: int,
                            verbose: bool = True
                            ) -> nd.DVector:
        """
        Reads land use data and creates population Dvector

        Parameters
        ----------
        year:
            The year for which the population data has to be read.

        verbose:
            Whether to print a progress bar while applying the splits or not

        Returns
        -------
        pop_dvec:
            Returns the population Dvector
        """

        # Read the land use data corresponding to the year
        pop = du.safe_read_csv(self.land_use_paths[year], usecols=self._target_cols['land_use'])

        # Define the zoning and segmentations we want to use
        msoa_zoning = nd.get_zoning_system('msoa')
        pop_seg = nd.get_segmentation_level('lu_pop')

        # Instantiate
        pop_dvec = nd.DVector(
            zoning_system=msoa_zoning,
            segmentation=pop_seg,
            import_data=pop.rename(columns=self.seg_rename),
            zone_col="msoa_zone_id",
            val_col="people",
            verbose=verbose,
        )
        return pop_dvec

    def generate_productions(self,
                             pop_dvec: nd.DVector,
                             verbose: bool = True,
                             ) -> nd.DVector:
        """
        Applies trip rate split on the given HB productions

        Parameters
        ----------
        pop_dvec:
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
            import_data=trip_rates.rename(columns=self.seg_rename),
            val_col="trip_rate",
            verbose=verbose,
        )
        # ## MULTIPLY TOGETHER ## #
        return pop_dvec * trip_rates_dvec

    def _split_by_tp_and_mode(self,
                              pure_demand,
                              verbose: bool = True
                              ) -> nd.DVector:
        """
        Applies time period and mode splits on the given HB productions

        Parameters
        ----------
        pure_demand:
            Dvector containing the HB productions to split.

        verbose:
            Whether to print a progress bar while applying the splits or not

        Returns
        -------
        full_seg_demand:
            The given hb_prods additionally split by tp and mode
        """

        # Define the segmentation we want to use
        m_tp_pure_demand_seg = nd.get_segmentation_level('notem_tfnat')
        notem_seg = nd.get_segmentation_level('notem')

        # Read in mode time splits
        mode_time_splits = pd.read_csv(self.mode_time_splits_path, usecols=self._target_cols['m_tp'])

        # ## CREATE MODE_TIME SPLITS DVEC ## #
        du.print_w_toggle("Creating mode time splits DVec...", verbose=verbose)

        # Instantiate
        mode_time_splits_dvec = nd.DVector(
            zoning_system=None,
            segmentation=m_tp_pure_demand_seg,
            import_data=mode_time_splits,
            val_col="split",
            verbose=verbose,
        )

        du.print_w_toggle("Multiplying...", verbose=verbose)
        full_seg_demand = core.multiply_and_aggregate_dvectors(
            pure_demand,
            mode_time_splits_dvec,
            notem_seg,
        )

        return full_seg_demand

    def create_output_paths(self,
                            export_path: nd.PathLike,
                            years: List[int],
                            ):
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
        pure_demand_out:
            Dictionary containing file names for pure demand outputs with year as key

        fully_segmented_out:
            Dictionary containing file names for fully segmented outputs with year as key

        aggregated_out:
            Dictionary containing file names for aggregated outputs with year as key
        """
        pure_demand_out = dict()
        fully_segmented_out = dict()
        aggregated_out = dict()
        for year in years:
            pure_demand_out[year] = os.path.join(export_path, "%s_%s_%s_%d_dvec.pkl" % (
                HBProductionModel._trip_origin, HBProductionModel._zoning_system, HBProductionModel._pure_demand, year))
            fully_segmented_out[year] = os.path.join(export_path, "%s_%s_%s_%d_dvec.pkl" % (
                HBProductionModel._trip_origin, HBProductionModel._zoning_system, HBProductionModel._fully_segmented,
                year))
            aggregated_out[year] = os.path.join(export_path, "%s_%s_%s_%d_dvec.pkl" % (
                HBProductionModel._trip_origin, HBProductionModel._zoning_system, HBProductionModel._aggregated, year))

        return pure_demand_out, fully_segmented_out, aggregated_out

    def create_pure_dem_report_paths(self,
                                     report_path: nd.PathLike,
                                     years: List[int],
                                     ):
        """
        Creates output file names for pure demand
        HB production reports for the list of years

        Parameters
        ----------
        report_path:
            Location where the pure demand report files are to be created.

        years:
            Contains the list of years for which the production model is run.

        Returns
        -------
        pure_demand_totals_out:
            Dictionary containing file names for pure demand msoa level outputs with year as key

        pure_demand_sec_totals_out:
            Dictionary containing file names for pure demand sector level outputs with year as key

        pure_demand_ie_totals_out:
            Dictionary containing file names for pure demand IE level outputs with year as key
        """

        pure_demand_totals_out = dict()
        pure_demand_sec_totals_out = dict()
        pure_demand_ie_totals_out = dict()

        for year in years:
            pure_demand_totals_out[year] = os.path.join(report_path, "%s_%d_%s.csv" % (
                HBProductionModel._pure_demand, year, "totals"))
            pure_demand_sec_totals_out[year] = os.path.join(report_path, "%s_%d_%s.csv" % (
                HBProductionModel._pure_demand, year, "sector_totals"))
            pure_demand_ie_totals_out[year] = os.path.join(report_path, "%s_%d_%s.csv" % (
                HBProductionModel._pure_demand, year, "ie_totals"))

        return pure_demand_totals_out, pure_demand_sec_totals_out, pure_demand_ie_totals_out

    def create_fully_seg_report_paths(self,
                                      report_path: nd.PathLike,
                                      years: List[int],
                                      ):
        """
        Creates output file names for fully segmented
        HB production reports for the list of years

        Parameters
        ----------
        report_path:
            Location where the fully segmented report files are to be created.

        years:
            Contains the list of years for which the production model is run.

        Returns
        -------
        fully_seg_totals_out:
            Dictionary containing file names for fully segmented msoa level outputs with year as key

        fully_seg_sec_totals_out:
            Dictionary containing file names for fully segmented sector level outputs with year as key

        fully_seg_ie_totals_out:
            Dictionary containing file names for fully segmented IE level outputs with year as key
        """

        fully_seg_totals_out = dict()
        fully_seg_sec_totals_out = dict()
        fully_seg_ie_totals_out = dict()

        for year in years:
            fully_seg_totals_out[year] = os.path.join(report_path, "%s_%d_%s.csv" % (
                HBProductionModel._fully_segmented, year, "totals"))
            fully_seg_sec_totals_out[year] = os.path.join(report_path, "%s_%d_%s.csv" % (
                HBProductionModel._fully_segmented, year, "sector_totals"))
            fully_seg_ie_totals_out[year] = os.path.join(report_path, "%s_%d_%s.csv" % (
                HBProductionModel._fully_segmented, year, "ie_totals"))

        return fully_seg_totals_out, fully_seg_sec_totals_out, fully_seg_ie_totals_out
