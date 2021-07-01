# -*- coding: utf-8 -*-
"""
Temporary file for testing DVec - should be moved over to proper tests when there is time!
"""

# BACKLOG: Move run script for DVector into pytest
#  labels: core, testing

# Third party imports
import numpy as np
import pandas as pd

import tqdm

# local imports
import normits_demand as nd

from normits_demand import core
from normits_demand.utils import timing

# GLOBAL VARIABLES
# I Drive Path locations
POPULATION_PATH = r"I:\NorMITs Land Use\base_land_use\iter3b\outputs\land_use_output_tfn_msoa1.csv"
TRIP_RATES_PATH = r"I:\NorMITs Demand\import\NoTEM\HB_Productions\hb_trip_rates_v1.9.csv"
MODE_TIME_SPLITS_PATH = r"I:\NorMITs Demand\import\NoTEM\HB_Productions\hb_mode_time_split_v1.9.csv"

# # Nirmal C Drive locations
# POPULATION_PATH = r"C:\Data\NorMITS\land_use_output_tfn_msoa1.csv"
# TRIP_RATES_PATH = r"C:\Data\NorMITS\hb_trip_rates_normalised.csv"


def main():

    # Define the zoning and segmentations we want to use
    msoa_zoning = nd.get_zoning_system('msoa')
    pop_seg = nd.get_segmentation_level('lu_pop')
    pure_demand_seg = nd.get_segmentation_level('pure_demand')
    m_tp_pure_demand_seg = nd.get_segmentation_level('notem_tfnat')
    notem_seg = nd.get_segmentation_level('notem')

    # Define wanted columns
    target_cols = {
        'land_use': ['msoa_zone_id', 'area_type', 'tfn_traveller_type', 'people'],
        'trip_rate': ['tfn_tt', 'tfn_at', 'p', 'trip_rate'],
        'm_tp': ['p', 'tfn_tt', 'tfn_at', 'm', 'tp', 'split'],
    }

    # Define segment renames needed
    seg_rename = {
        'tfn_traveller_type': 'tfn_tt',
        'area_type': 'tfn_at',
    }

    # Read in pop and trip rates
    print("Reading in files...")
    pop = pd.read_csv(POPULATION_PATH, usecols=target_cols['land_use'])
    trip_rates = pd.read_csv(TRIP_RATES_PATH, usecols=target_cols['trip_rate'])
    mode_time_splits = pd.read_csv(MODE_TIME_SPLITS_PATH, usecols=target_cols['m_tp'])

    # ## CREATE THE POP DVEC ## #
    print("Creating pop DVec...")

    # Instantiate
    pop_dvec = nd.DVector(
        zoning_system=msoa_zoning,
        segmentation=pop_seg,
        import_data=pop.rename(columns=seg_rename),
        zone_col="msoa_zone_id",
        val_col="people",
        verbose=True,
    )

    # ## CREATE THE TRIP RATES DVEC ## #
    print("Creating trip rates DVec...")

    # Instantiate
    trip_rates_dvec = nd.DVector(
        zoning_system=None,
        segmentation=pure_demand_seg,
        import_data=trip_rates.rename(columns=seg_rename),
        val_col="trip_rate",
    )

    # ## MULTIPLY TOGETHER ## #
    pure_demand = pop_dvec * trip_rates_dvec

    # COMPRESS OUT HERE
    from normits_demand.utils import timing

    print("Writing out... %s" % timing.get_datetime())
    output_path = "E:/pure_demand_dvec.pkl"
    pure_demand.to_pickle(output_path)

    print("Reading in... %s" % timing.get_datetime())
    thing = nd.from_pickle(output_path)
    print("Done... %s" % timing.get_datetime())

    # ## CREATE MODE_TIME SPLITS DVEC ## #
    print("Creating mode time splits DVec...")

    # Instantiate
    mode_time_splits_dvec = nd.DVector(
        zoning_system=None,
        segmentation=m_tp_pure_demand_seg,
        import_data=mode_time_splits,
        val_col="split",
    )

    print("Multiplying...")
    final = core.multiply_and_aggregate_dvectors(
        pure_demand,
        mode_time_splits_dvec,
        notem_seg,
    )

    path = "E:/final_dvec.pkl"
    final.to_pickle(path)

    # print(final.to_df())


def aggregate_test():

    path = "E:/pure_demand_dvec_dvec.pbz2"
    print("Reading in... %s" % timing.get_datetime())
    pure_demand_vec = nd.read_compressed_dvector(path)
    tfn_agg_at_seg = nd.get_segmentation_level('pure_demand_reporting')

    tfn_ca_sectors = nd.get_zoning_system('ca_sector_2020')
    ie_sectors = nd.get_zoning_system('ie_sector')
    three_sectors = nd.get_zoning_system('3_sector')

    print("Total: ", pure_demand_vec.sum())
    print("Aggregating... %s" % timing.get_datetime())
    pure_demand_vec = pure_demand_vec.aggregate(tfn_agg_at_seg)

    print("Total: ", pure_demand_vec.sum())
    print("Translating... %s" % timing.get_datetime())
    pure_demand_ca = pure_demand_vec.translate_zoning(tfn_ca_sectors)
    print("Total: ", pure_demand_ca.sum())
    pure_demand_ie = pure_demand_vec.translate_zoning(ie_sectors)
    print("Total: ", pure_demand_ie.sum())
    pure_demand_3 = pure_demand_vec.translate_zoning(three_sectors)
    print("Total: ", pure_demand_3.sum())
    print("Done! %s" % timing.get_datetime())


def tfn_tt_translate_test():

    path = 'E:/final_dvec.pkl'

    notem_full_tfn = nd.get_segmentation_level('hb_notem_full_tfn')
    tfn_ca_sectors = nd.get_zoning_system('ca_sector_2020')
    ie_sectors = nd.get_zoning_system('ie_sector')
    three_sectors = nd.get_zoning_system('3_sector')

    print("Reading in... %s" % timing.get_datetime())
    notem_seg_vec = nd.from_pickle(path)
    print("Total: ", notem_seg_vec.sum())

    print("Aggregating to full_tfn... %s" % timing.get_datetime())
    out_vec = notem_seg_vec.aggregate(notem_full_tfn, split_tfntt_segmentation=True)
    print("Total: ", out_vec.sum())

    print("Translating... %s" % timing.get_datetime())
    out_vec_ca = out_vec.translate_zoning(tfn_ca_sectors)
    print("Total: ", out_vec_ca.sum())
    out_vec_ie = out_vec.translate_zoning(ie_sectors)
    print("Total: ", out_vec_ie.sum())
    out_vec_3 = out_vec.translate_zoning(three_sectors)
    print("Total: ", out_vec_3.sum())

    print("Removing Zoning... %s" % timing.get_datetime())
    out_vec_sans_zone = out_vec.remove_zoning(np.sum)
    print("Total: ", out_vec_sans_zone.sum())

    print("Done! %s" % timing.get_datetime())


if __name__ == '__main__':
    main()
    # aggregate_test()
    tfn_tt_translate_test()
