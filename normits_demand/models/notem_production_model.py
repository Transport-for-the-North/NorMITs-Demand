"""
Created on: 19/06/2021

File purpose: Production Model for NoTEM




"""
# Third party imports
import os
import warnings
from typing import List

import numpy as np
import pandas as pd

# local imports
import normits_demand as nd
from normits_demand import efs_constants as consts

from normits_demand.utils import general as du
from normits_demand.utils import timing


from normits_demand.concurrency import multiprocessing



class NoTEM_HBProductionModel:
    def _init_(self,
              land_use_path:dict,
              trip_rates_path:str,
              mode_time_splits_path: str,
              constraints_path:dict,
              export_path:str
              ):
        #Validate inputs

        #Assign
        self.land_use_path=land_use_path
        self.trip_rates_path=trip_rates_path
        self.mode_time_splits_path=mode_time_splits_path
        self.constraints_path=constraints_path
        self.export_path=export_path
        self.process_count=consts.PROCESS_COUNT

    def run(self,
            output_raw: bool = True,
            recreate_productions: bool = True,
            verbose: bool = True,
            ) -> pd.DataFrame:
        """
        Runs the HB Production model

        Parameters
        ----------
        output_raw:
            Whether to output the raw hb productions before aggregating to
            the required segmentation and mode.

        recreate_productions:
            Whether to recreate the hb productions or not. If False, it will
            look in export_path for previously produced productions and return
            them. If none can be found, they will be generated.

        verbose:
            Whether to print progress bars during processing or not.

        Returns
        -------
        HB_Productions:
            HB productions for the mode and segmentation needed
        """
        # Return previously created productions if we can
        # TODO(NK): Need to way to figure out how self.zoning_system can be replaced
        fname = consts.PRODS_FNAME % (self.zoning_system, 'hb')
        final_output_path = os.path.join(self.export_path, fname)

        if not recreate_productions and os.path.isfile(final_output_path):
            print("Found some already produced productions. Using them!")
            return pd.read_csv(final_output_path)

        # Initialise timing
        start_time = timing.current_milli_time()
        du.print_w_toggle("Starting HB Production Model at: %s" % timing.get_datetime(),
                          verbose=verbose)

        target_cols = ['msoa_zone_id', 'area_type', 'tfn_traveller_type', 'people']
        # # ## READ IN POPULATION DATA ## #
        print("Loading the population data...")
        population = _read_land_use_data(year)

        # ## CREATE PRODUCTIONS ## #
        print("Population generated. Converting to productions...")
        hb_prods = generate_productions(
            population=population,
            trip_rates_path=self.trip_rates_path,
            verbose=verbose
        )
        # ## SPLIT PRODUCTIONS BY MODE AND TIME ## #
        print("Splitting HB productions by mode and time...")
        hb_prods = self._split_by_tp_and_mode(hb_prods, verbose=verbose)

        
        # Write productions to file
        print("Writing productions to file...")
        fname = consts.PRODS_FNAME % (self.zoning_system, 'hb')
        path = os.path.join(self.export_path, fname)
        hb_prods.to_csv(path, index=False)


    def _read_land_use_data(self,year:int):
    #Read the land use data corresponding to the year
        population=du.safe_read_csv(self.land_use_path[str(year)],usecols=target_cols)
        return population

    def generate_productions(self,
                         population: pd.DataFrame,
                         verbose=True,
                         
                         ):

    # Define the zoning and segmentations we want to use
        import_drive = "I:/"
        msoa_zoning = nd.get_zoning_system(name='msoa', import_drive=import_drive)
        pop_seg = nd.get_segmentation_level(name='lu_pop', import_drive=import_drive)
        pure_demand_seg = nd.get_segmentation_level(name='pure_demand', import_drive=import_drive)

        # Define wanted columns
        target_cols = {
            'land_use': ['msoa_zone_id', 'area_type', 'tfn_traveller_type', 'people'],
            'trip_rate': ['tfn_traveller_type', 'area_type', 'p', 'trip_rate']
        }

        # Read in pop and trip rates
        print("Reading in files...")
        pop = population
        trip_rates = du.safe_read_csv(trip_rates_path, usecols=target_cols['trip_rate'])

        # ## CREATE THE POP DVEC ## #
        print("Creating pop DVec...")

        # Add a segment column
        naming_conversion = {
            'tfn_tt': 'tfn_traveller_type',
            'tfn_at': 'area_type',
        }
        pop['segment'] = pop_seg.create_segment_col(pop, naming_conversion)

        # Filter pop down ready for import into Dvec
        pop = pd_utils.reindex_and_groupby(
            df=pop,
            index_cols=['segment', 'msoa_zone_id', 'people'],
            value_cols=['people'],
        )

        # Instantiate
        pop_dvec = nd.DVector(
            zoning_system=msoa_zoning,
            segmentation=pop_seg,
            import_data=pop,
            zone_col="msoa_zone_id",
            segment_col="segment",
            val_col="people",
            verbose=True,
        )

        # ## CREATE THE TRIP RATES DVEC ## #
        print("Creating trip rates DVec...")

        # Add a segment column
        # Create inside DVec!!!
        naming_conversion = {
            'p': 'p',
            'tfn_tt': 'tfn_traveller_type',
            'tfn_at': 'area_type',
        }
        trip_rates['segment'] = pure_demand_seg.create_segment_col(trip_rates, naming_conversion)

        # Filter pop down ready for import into Dvec
        trip_rates = pd_utils.reindex_and_groupby(
            df=trip_rates,
            index_cols=['segment', 'trip_rate'],
            value_cols=['trip_rate'],
        )

        # Instantiate
        trip_rates_dvec = nd.DVector(
            zoning_system=None,
            segmentation=pure_demand_seg,
            import_data=trip_rates,
            zone_col=None,
            segment_col="segment",
            val_col="trip_rate",
            verbose=True,
        )

        # ## MULTIPLY TOGETHER ## #
        # TODO(BT): Need to implement this - example calling structure?
        #  Need to think about how this will work in more detail
        #  Specifically, how to generalise properly!
        pure_demand = ds.__mul__(pop_dvec, trip_rates_dvec)
    
    def _split_by_tp_and_mode(self,
                              hb_prods,
                              verbose: bool = True):

        """
        Applies time period and mode splits on the given HB productions

        Parameters
        ----------
        hb_prods:
            Dataframe containing the HB productions to split.

        verbose:
            Whether to print a progress bar while applying the splits or not

        Returns
        -------
        tp_mode_split_hb_prods:
            The given hb_prods additionally split by tp and mode
        """

