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
        'trip_rate': ['tfn_traveller_type', 'area_type', 'p', 'trip_rate'],
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
        self.process_count = process_count
        self.years = list(self.land_use_paths.keys())

        # Initialise Output paths
        self.pure_demand_out, self.fully_segmented_out, self.aggregated_out = self.create_output_paths(
            self.export_path, self._trip_origin, self._zoning_system, self._pure_demand, self._fully_segmented,
            self._aggregated, self.years)

    def run(self,
            export_pure_demand: bool = True,
            audits: bool = True,
            output_fully_segmented: bool = True,
            output_aggregated: bool = True,
            verbose: bool = True,
            ):
        """
        Runs the HB Production model

        Parameters
        ----------
        export_pure_demand:
            Whether to output the pure demand

        audits:
            Whether to output print_audits to the terminal during running. This can
            be used to monitor the population and production numbers being
            generated and constrained.

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
                verbose=verbose)

            if export_pure_demand:
                path = pure_demand.compress_out(self.pure_demand_out[year])

            # TODO:Check with BT
            #
            # # Population Audit
            # if audits:
            #     print('\n', '-' * 15, 'HB Production Audit before constraining', '-' * 15)
            #     print('. Total population for year %s is: %.4f'
            #               %)
            #     print('\n')

            # TODO: Write out audits of pure_demand
            #  Need audit output path
            #  Output some audits of what demand was before and after control
            #  By segment.

            # ## SPLIT PRODUCTIONS BY MODE AND TIME ## #
            du.print_w_toggle("Splitting HB productions by mode and time...", verbose=verbose)
            hb_prods = self._split_by_tp_and_mode(pure_demand, verbose=verbose)

            # Output productions before any aggregation
            if output_fully_segmented:  # output_fully_segmented
                du.print_w_toggle("Writing raw HB Productions to disk...", verbose=verbose)
                hb_prods.compress_out(self.fully_segmented_out[year])

            if output_aggregated:
                # TODO: Aggregate segments
                agg_hb_prods = hb_prods  # aggregate(optional_segmentation)
                du.print_w_toggle("Writing productions to file...", verbose=verbose)
                agg_hb_prods.compress_out(self.aggregated_out[year])

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
                            trip_origin: str,
                            zoning_system: str,
                            pure_demand: str,
                            fully_segmented: str,
                            aggregated: str,
                            years: List[int],
                            ):

        pure_demand_out = fully_segmented_out = aggregated_out = dict()
        for year in years:
            pure_demand_out[year] = os.path.join(export_path, "%s_%s_%s_%d_dvec.pbz2" % (
                trip_origin, zoning_system, pure_demand, year))
            fully_segmented_out[year] = os.path.join(export_path, "%s_%s_%s_%d_dvec.pbz2" % (
                trip_origin, zoning_system, fully_segmented, year))
            aggregated_out[year] = os.path.join(export_path, "%s_%s_%s_%d_dvec.pbz2" % (
                trip_origin, zoning_system, aggregated, year))

        return pure_demand_out, fully_segmented_out, aggregated_out
