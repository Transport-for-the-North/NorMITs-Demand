# -*- coding: utf-8 -*-
"""
Temporary file for testing DVec - should be moved over to proper tests when there is time!
"""

# BACKLOG: Move run script for DVector into pytest
#  labels: core, testing

# Third party imports
import pandas as pd

import tqdm

# local imports
import normits_demand as nd

# GLOBAL VARIABLES
# I Drive Path locations
POPULATION_PATH = r"I:\NorMITs Land Use\base_land_use\iter3b\outputs\land_use_output_tfn_msoa1.csv"
TRIP_RATES_PATH = r"I:\Data\NTS\outputs\hb\hb_trip_rates\hb_trip_rates_normalised.csv"

# # Nirmal C Drive locations
# POPULATION_PATH = r"C:\Data\NorMITS\land_use_output_tfn_msoa1.csv"
# TRIP_RATES_PATH = r"C:\Data\NorMITS\hb_trip_rates_normalised.csv"

def main():

    # Define wanted columns
    target_cols = {
        'land_use': ['msoa_zone_id', 'area_type', 'tfn_traveller_type', 'people'],
        'trip_rate': ['tfn_traveller_type', 'area_type', 'p', 'trip_rate']
    }

    # Read in pop and trip rates
    print("Reading in files...")
    pop = pd.read_csv(POPULATION_PATH, usecols=target_cols['land_use'])
    trip_rates = pd.read_csv(TRIP_RATES_PATH, usecols=target_cols['trip_rate'])

    # ## CONVERT POP TO DVECTOR ## #
    # Define new combo columns
    pop['segment'] = pop['tfn_traveller_type'].astype(str) + "_" + pop['area_type'].astype(str)
    pop['zone_at'] = pop['msoa_zone_id'].astype(str) + "_" + pop['area_type'].astype(str)

    # Get unique columns names
    # TODO(BT): This should come from a ModelZone object once they are implemented!
    unq_zoning = pop['msoa_zone_id'].unique()

    # Filter down to just the columns we need for this
    group_cols = ['segment', 'msoa_zone_id']
    index_cols = group_cols.copy() + ['people']
    temp_pop = pop.reindex(columns=index_cols)
    temp_pop = temp_pop.groupby(group_cols).sum().reset_index()

    # Get the pop data for each segment
    dvec_pop = dict()
    desc = "To Dvec"
    for segment in tqdm.tqdm(pop['segment'].unique(), desc=desc):
        # Get all available pop for this segment
        seg_pop = temp_pop[temp_pop['segment'] == segment].copy()

        # Filter down to just pop as values, and zoning system as the index
        seg_pop = seg_pop.reindex(columns=['msoa_zone_id', 'people'])
        seg_pop = seg_pop.set_index('msoa_zone_id')

        # Infill any missing zones as 0
        seg_pop = seg_pop.reindex(unq_zoning, fill_value=0)

        # Assign to dict for storage
        dvec_pop[segment] = seg_pop.values


def dvec_obj_main():

    # Define wanted columns
    target_cols = {
        'land_use': ['msoa_zone_id', 'area_type', 'tfn_traveller_type', 'people'],
        'trip_rate': ['tfn_traveller_type', 'area_type', 'p', 'trip_rate']
    }

    # Read in pop and trip rates
    print("Reading in files...")
    pop = pd.read_csv(POPULATION_PATH, usecols=target_cols['land_use'])
    trip_rates = pd.read_csv(TRIP_RATES_PATH, usecols=target_cols['trip_rate'])

    # Add a segment column
    # TODO(BT): Once the Segmentation object is properly implemented, that
    #  should have a function to add this new col automatically?
    pop['segment'] = pop['tfn_traveller_type'].astype(str) + "_" + pop['area_type'].astype(str)

    # Filter pop down ready for import into Dvec
    group_cols = ['segment', 'msoa_zone_id']
    index_cols = group_cols.copy() + ['people']

    pop = pop.reindex(columns=index_cols)
    pop = pop.groupby(group_cols).sum().reset_index()

    pop_dvec = nd.DVector(
        zoning_system=nd.get_zoning_system(name='msoa', import_drive="I:/"),
        segmentation=None,  # Not implemented yet
        import_data=pop,
        zone_col="msoa_zone_id",
        segment_col="segment",
        val_col="people",
        verbose=True,
    )

    print("hi")


if __name__ == '__main__':
    # main()
    dvec_obj_main()
