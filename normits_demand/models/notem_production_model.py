"""
Created on: 19/06/2021

File purpose: Production Model for NoTEM

"""
# Allow class self type hinting
from __future__ import annotations

# Builtins
import os

from typing import Dict

# Third party imports
import pandas as pd

# local imports
import normits_demand as nd

from normits_demand import core
from normits_demand import efs_constants as consts

from normits_demand.utils import compress
from normits_demand.utils import general as du
from normits_demand.utils import timing


class NoTEM_HBProductionModel:
    _trip_origin = 'hb'
    _zoning_system = 'msoa'
    _pure_demand = 'pure_demand_hb'
    # Define wanted columns
    _target_cols = {
        'land_use': ['msoa_zone_id', 'area_type', 'tfn_traveller_type', 'people'],
        'trip_rate': ['tfn_traveller_type', 'area_type', 'p', 'trip_rate'],
        'm_tp': ['p', 'tfn_tt', 'tfn_at', 'm', 'tp', 'split'],
    }

    def _init_(self,
               land_use_paths: Dict[int, nd.PathLike],
               trip_rates_path: str,
               mode_time_splits_path: str,
               constraint_paths: Dict[int, nd.PathLike],
               export_path: str,
               process_count: int = consts.PROCESS_COUNT
               ):
        # Validate inputs
        [du.check_csv_exists(x) for x in land_use_paths.values()]
        [du.check_csv_exists(x) for x in constraint_paths.values()]
        du.check_csv_exists(trip_rates_path)
        du.check_csv_exists(mode_time_splits_path)
        du.check_csv_exists(export_path)

        # Assign
        self.land_use_paths = land_use_paths
        self.trip_rates_path = trip_rates_path
        self.mode_time_splits_path = mode_time_splits_path
        self.constraint_paths = constraint_paths
        self.export_path = export_path
        self.process_count = process_count

        self.years = list(self.land_use_paths.keys())

    def run(self,
            recreate_productions: bool = True,
            export_pure_demand: bool = True,
            output_raw: bool = True,
            verbose: bool = True,
            ):
        """
        Runs the HB Production model

        Parameters
        ----------
        recreate_productions:
            Whether to recreate the hb productions or not. If False, it will
            look in export_path for previously produced productions and return
            them. If none can be found, they will be generated.

        export_pure_demand:
            Whether to output the pure demand

        output_raw:
            Whether to output the raw hb productions before aggregating to
            the required segmentation and mode.

        verbose:
            Whether to print progress bars during processing or not.

        Returns
        -------
        HB_Productions:
            HB productions for the mode and segmentation needed
        """
        # Return previously created productions if we can
        fname = consts.PRODS_FNAME % (self._zoning_system, self._trip_origin)
        final_output_path = os.path.join(self.export_path, fname)

        if not recreate_productions and os.path.isfile(final_output_path):
            print("Found some already produced productions. Using them!")
            return pd.read_csv(final_output_path)

        # Initialise timing
        # TODO(BT): Properly integrate logging
        start_time = timing.current_milli_time()
        du.print_w_toggle("Starting HB Production Model at: %s" % timing.get_datetime(),
                          verbose=verbose)

        for year in self.years:
            print("Loading the population data...")
            population = self._read_land_use_data(year)

            print("Population generated. Converting to productions...")
            pure_demand = self.generate_productions(
                population=population,
                verbose=verbose
            )

            if export_pure_demand:
                fname = consts.PRODS_FNAME % (self._zoning_system, self._pure_demand)
                output_path = os.path.join(self.export_path, fname)
                path = pure_demand.compress_out(output_path)

            # ## SPLIT PRODUCTIONS BY MODE AND TIME ## #
            print("Splitting HB productions by mode and time...")
            hb_prods = self._split_by_tp_and_mode(pure_demand, verbose=verbose)

            # TODO: Bring in constraints

            # Output productions before any aggregation
            if output_raw:
                print("Writing raw HB Productions to disk...")
                fname = consts.PRODS_FNAME_YEAR % (self._zoning_system, 'raw_hb', year)
                path = os.path.join(self.export_path, fname)
                hb_prods.compress_out(path)

            # TODO: Aggregate segments

            # Write productions to file
            print("Writing productions to file...")
            fname = consts.PRODS_FNAME_YEAR % (self._zoning_system, self._trip_origin, year)
            path = os.path.join(self.export_path, fname)
            hb_prods.compress_out(path)

            # End timing
            end_time = timing.current_milli_time()
            du.print_w_toggle("Finished HB Production Model at: %s" % timing.get_datetime(),
                              verbose=verbose)
            du.print_w_toggle("HB Production Model took: %s"
                              % timing.time_taken(start_time, end_time), verbose=verbose)

    def _read_land_use_data(self, year: int):
        # Read the land use data corresponding to the year
        population = du.safe_read_csv(self.land_use_paths[year], usecols=self._target_cols['land_use'])
        fname = consts.POP_FNAME % (str(year))
        compress.write_out(fname, self.export_path)
        return population

    def generate_productions(self,
                             population: pd.DataFrame,
                             verbose: bool = True,
                             ):

        # Define the zoning and segmentations we want to use
        msoa_zoning = nd.get_zoning_system('msoa')
        pop_seg = nd.get_segmentation_level('lu_pop')
        pure_demand_seg = nd.get_segmentation_level('pure_demand')

        # Define segment renames needed
        seg_rename = {
            'tfn_traveller_type': 'tfn_tt',
            'area_type': 'tfn_at',
        }
        # Read in pop and trip rates
        print("Reading in files...")
        pop = population
        trip_rates = du.safe_read_csv(self.trip_rates_path, usecols=self._target_cols['trip_rate'])

        # ## CREATE THE POP DVEC ## #
        print("Creating pop DVec...")

        # Instantiate
        pop_dvec = nd.DVector(
            zoning_system=msoa_zoning,
            segmentation=pop_seg,
            import_data=pop.rename(columns=seg_rename),
            zone_col="msoa_zone_id",
            val_col="people",
            verbose=verbose,
        )

        # ## CREATE THE TRIP RATES DVEC ## #
        print("Creating trip rates DVec...")

        # Instantiate
        trip_rates_dvec = nd.DVector(
            zoning_system=None,
            segmentation=pure_demand_seg,
            import_data=trip_rates.rename(columns=seg_rename),
            val_col="trip_rate",
            verbose=verbose,
        )
        # ## MULTIPLY TOGETHER ## #
        return pop_dvec * trip_rates_dvec

    def _split_by_tp_and_mode(self,
                              pure_demand,
                              verbose: bool = True):

        """
        Applies time period and mode splits on the given HB productions

        Parameters
        ----------
        pure_demand:
            Dataframe containing the HB productions to split.

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
        print("Creating mode time splits DVec...")

        # Instantiate
        mode_time_splits_dvec = nd.DVector(
            zoning_system=None,
            segmentation=m_tp_pure_demand_seg,
            import_data=mode_time_splits,
            val_col="split",
            verbose=verbose,
        )

        print("Multiplying...")
        full_seg_demand = core.multiply_and_aggregate_dvectors(
            pure_demand,
            mode_time_splits_dvec,
            notem_seg,
        )

        return full_seg_demand
