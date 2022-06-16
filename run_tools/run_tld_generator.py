# -*- coding: utf-8 -*-
"""
Created on: 07/12/2021
Updated on:

Original author: Chris Storey
Last update made by: Ben Taylor
Other updates made by:

File purpose:

"""
# Built-Ins
import os
import sys
import shutil

import itertools

# Third Party

# Local Imports
sys.path.append("..")
# pylint: disable=import-error,wrong-import-position
from normits_demand.tools import trip_length_distributions as tlds
# pylint: enable=import-error,wrong-import-position

# GLOBAL
TLB_FOLDER = 'I:/NTS/outputs/tld'
TLB_VERSION = 'nts_tld_data_v3.1.csv'
OUTPUT_FOLDER = r'I:\NorMITs Demand\import\trip_length_distributions\tld_tool_outputs'
TLD_HOME = r'I:\NorMITs Demand\import\trip_length_distributions\config'

BAND_FOLDER = os.path.join(TLD_HOME, 'bands')
SEGMENTATION_FOLDER = os.path.join(TLD_HOME, 'segmentations')


def run_all_combinations():
    """Runs every combination of inputs through the TLD builder"""
    # Get a list of all available options
    band_list = os.listdir(BAND_FOLDER)
    band_list = [x for x in band_list if '.csv' in x]

    seg_list = os.listdir(SEGMENTATION_FOLDER)
    seg_list = [x for x in seg_list if '.csv' in x]

    extract = tlds.TripLengthDistributionBuilder(
        tlb_folder=TLB_FOLDER,
        tlb_version=TLB_VERSION,
        bands_definition_dir=BAND_FOLDER,
        segment_definition_dir=SEGMENTATION_FOLDER,
        output_folder=OUTPUT_FOLDER,
    )

    for area, bands, seg in itertools.product(list(tlds.GeoArea), band_list, seg_list):
        # Built list of unchanging kwargs
        kwargs = {
            "geo_area": area,
            "sample_period": tlds.SampleTimePeriods.FULL_WEEK,
            "cost_units": tlds.CostUnits.KM,
            "bands_name": bands,
            "segmentation_name": seg,
            "sample_threshold": 10,
            "verbose": False,
        }

        extract.tld_generator(trip_filter_type=tlds.TripFilter.TRIP_OD, **kwargs)

        # Include ie movements filter too if not GB
        if area != tlds.GeoArea.GB:
            extract.tld_generator(trip_filter_type=tlds.TripFilter.TRIP_O, **kwargs)


def run_test():
    """Runs a test set of inputs through the TLD builder"""
    # Get a list of all available options
    band_list = os.listdir(BAND_FOLDER)
    band_list = [x for x in band_list if '.csv' in x]

    seg_list = os.listdir(SEGMENTATION_FOLDER)
    seg_list = [x for x in seg_list if '.csv' in x]

    extract = tlds.TripLengthDistributionBuilder(
        tlb_folder=TLB_FOLDER,
        tlb_version=TLB_VERSION,
        bands_definition_dir=BAND_FOLDER,
        segment_definition_dir=SEGMENTATION_FOLDER,
        output_folder=OUTPUT_FOLDER,
    )

    kwargs = {
        "geo_area": tlds.GeoArea.NORTH,
        "sample_period": tlds.SampleTimePeriods.FULL_WEEK,
        "cost_units": tlds.CostUnits.KM,
        "bands_name": band_list[0],
        "segmentation_name": seg_list[0],
        "sample_threshold": 10,
        "verbose": False,
    }

    # North
    extract.tld_generator(trip_filter_type=tlds.TripFilter.TRIP_OD, **kwargs)

    # North inc_ie
    extract.tld_generator(trip_filter_type=tlds.TripFilter.TRIP_O, **kwargs)


def build_new_dimo_tlds():
    """Build a new version of all the TLDs needed for the distribution model"""

    # A full DiMo run requires:
    # ## run at north_and_mids, trip_OD ## #
    #
    #   dm_highway_bands
    #       hb_p_m
    #       nhb_p_m_tp_car
    #       nhb_p_m     (These then need copying over across TPs)

    #   dm_north_rail_bands
    #       hb_p_m_ca
    #       nhb_p_m_ca  (These then need copying across TPs)

    # ## run at gb, trip_OD ## #

    #   dm_highway_bands
    #       hb_p_m
    #       nhb_p_m_tp_car
    #       nhb_p_m     (These then need copying over across TPs)

    #   dm_gb_rail_bands
    #       hb_p_m_ca
    #       nhb_p_m_ca  (These then need copying across TPs)
    pass


def build_new_traveller_segment_tlds():
    """Build a new version of all the TLDs needed for the traveller segments"""
    tlb_version = 'nts_tld_data_v3.0.csv'

    # Init
    extract = tlds.TripLengthDistributionBuilder(
        tlb_folder=TLB_FOLDER,
        tlb_version=tlb_version,
        bands_definition_dir=BAND_FOLDER,
        segment_definition_dir=SEGMENTATION_FOLDER,
        output_folder=OUTPUT_FOLDER,
    )

    path_kwargs = {
        "geo_area": tlds.GeoArea.GB,
        "trip_filter_type": tlds.TripFilter.TRIP_OD,
        "sample_period": tlds.SampleTimePeriods.FULL_WEEK,
        "cost_units": tlds.CostUnits.KM,
    }
    generate_kwargs = path_kwargs.copy()
    generate_kwargs.update({
        "sample_threshold": 10,
        "verbose": False,
    })

    # ## GENERATE RAIL TLDS ## #

    # Generate with CA combined and then split out
    # Generate with HB and NHB combined and then split out
    extract.tld_generator(
        bands_name="dia_gb_rail_bands",
        segmentation_name="uc_m_seg_m6",
        **generate_kwargs,
    )

    # Copy back out!
    out_path = extract.build_output_path(
        bands_name="dia_gb_rail_bands",
        segmentation_name="uc_m_seg_m6",
        **path_kwargs,
    )
    print(out_path)


    # A full traveller segment run needs:
    # ## run at GB, trip_OD ## #
    #
    #   dm_highway_bands
    #       hb_business
    #       hb_commute
    #       hb_other
    #       nhb_business
    #       nhb_other
    #


    pass


def copy_across_tps():
    in_dir = ''
    out_dir = ''

    for fname in os.listdir(in_dir):
        if fname == 'full_export.csv':
            continue

        for tp in [1, 2, 3, 4]:
            out_name = fname.replace('.csv', '_tp%s.csv' % tp)

            shutil.copy(
                src=os.path.join(in_dir, fname),
                dst=os.path.join(out_dir, out_name),
            )


if __name__ == '__main__':
    # run_test()

    # run_all_combinations()
    # build_new_dimo_tlds()
    build_new_traveller_segment_tlds()
