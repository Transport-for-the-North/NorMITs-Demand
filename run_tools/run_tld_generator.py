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

import itertools

# Third Party

# Local Imports
sys.path.append("..")
# pylint: disable=import-error,wrong-import-position
from normits_demand import core as nd_core
from normits_demand import logging as nd_log
from normits_demand.tools import trip_length_distributions as tlds

# pylint: enable=import-error,wrong-import-position

# GLOBAL
TLB_FOLDER = "I:/NTS/outputs/tld"
TLB_VERSION = "nts_tld_data_v3.1.csv"
OUTPUT_FOLDER = r"E:\NorMITs Demand\import\trip_length_distributions\tld_tool_outputs"
TLD_HOME = r"I:\NorMITs Demand\import\trip_length_distributions\config"

BAND_DIR = os.path.join(TLD_HOME, "bands")
SEGMENTATION_DIR = os.path.join(TLD_HOME, "segmentations")
COPY_DEFINITIONS_DIR = os.path.join(SEGMENTATION_DIR, "copy_defs")

LOG_FILE = "TLD_Builder.log"


def run_all_combinations():
    """Runs every combination of inputs through the TLD builder"""
    # Get a list of all available options
    band_list = os.listdir(BAND_DIR)
    band_list = [x for x in band_list if ".csv" in x]

    seg_list = os.listdir(SEGMENTATION_DIR)
    seg_list = [x for x in seg_list if ".csv" in x]

    extract = tlds.TripLengthDistributionBuilder(
        tlb_folder=TLB_FOLDER,
        tlb_version=TLB_VERSION,
        bands_definition_dir=BAND_DIR,
        segment_copy_definition_dir=COPY_DEFINITIONS_DIR,
        output_folder=OUTPUT_FOLDER,
    )

    for area, bands, seg in itertools.product(list(tlds.GeoArea), band_list, seg_list):
        # Built list of unchanging kwargs
        kwargs = {
            "geo_area": area,
            "sample_period": tlds.SampleTimePeriods.FULL_WEEK,
            "cost_units": nd_core.CostUnits.KM,
            "bands_name": bands,
            "segmentation_name": seg,
            "min_sample_size": 40,
            "check_sample_size": 400,
            "inter_smoothing": True,
        }

        extract.tld_generator(trip_filter_type=tlds.TripFilter.TRIP_OD, **kwargs)

        # Include ie movements filter too if not GB
        if area != tlds.GeoArea.GB:
            extract.tld_generator(trip_filter_type=tlds.TripFilter.TRIP_O, **kwargs)


def run_test():
    """Runs a test set of inputs through the TLD builder"""
    # Get a list of all available options
    band_list = os.listdir(BAND_DIR)
    band_list = [x for x in band_list if ".csv" in x]

    seg_list = os.listdir(SEGMENTATION_DIR)
    seg_list = [x for x in seg_list if ".csv" in x]

    extract = tlds.TripLengthDistributionBuilder(
        tlb_folder=TLB_FOLDER,
        tlb_version=TLB_VERSION,
        bands_definition_dir=BAND_DIR,
        segment_copy_definition_dir=COPY_DEFINITIONS_DIR,
        output_folder=OUTPUT_FOLDER,
    )

    kwargs = {
        "geo_area": tlds.GeoArea.NORTH,
        "sample_period": tlds.SampleTimePeriods.FULL_WEEK,
        "cost_units": nd_core.CostUnits.KM,
        "bands_name": band_list[0],
        "segmentation_name": seg_list[0],
        "min_sample_size": 10,
    }

    # North
    extract.tld_generator(trip_filter_type=tlds.TripFilter.TRIP_OD, **kwargs)

    # North inc_ie
    extract.tld_generator(trip_filter_type=tlds.TripFilter.TRIP_O, **kwargs)


def build_new_dimo_tlds():
    """Build a new version of all the TLDs needed for the distribution model"""
    # This has light and heavy rail split out for tram TLDs
    tlb_version = "nts_tld_data_v3.1.csv"

    # Init
    extractor = tlds.TripLengthDistributionBuilder(
        tlb_folder=TLB_FOLDER,
        tlb_version=tlb_version,
        bands_definition_dir=BAND_DIR,
        segment_copy_definition_dir=COPY_DEFINITIONS_DIR,
        output_folder=OUTPUT_FOLDER,
    )

    # Define consistent kwargs
    path_kwargs = {
        "trip_filter_type": tlds.TripFilter.TRIP_OD,
        "sample_period": tlds.SampleTimePeriods.FULL_WEEK,
        "cost_units": nd_core.CostUnits.KM,
    }
    generate_kwargs = path_kwargs.copy()
    generate_kwargs.update({
        "min_sample_size": 40,
        "check_sample_size": 400,
        "inter_smoothing": True,
    })

    for geo_area in [tlds.GeoArea.GB, tlds.GeoArea.NORTH_AND_MIDS]:
        # ## GENERATE HIGHWAY ## #
        # hway_bands = "dm_highway_bands"
        hway_bands = "dynamic"
        hway_kwargs = generate_kwargs.copy()
        hway_kwargs.update({
            "geo_area": geo_area,
            "bands_name": hway_bands,
        })

        # HB TLDs
        segmentation = nd_core.get_segmentation_level("hb_p_m")
        extractor.tld_generator(segmentation=segmentation, **hway_kwargs)

        # NHB TLDs - car has lots of data, can be done at time periods
        segmentation = nd_core.get_segmentation_level("nhb_p_m_tp_car")
        extractor.tld_generator(segmentation=segmentation, **hway_kwargs)

        # NHB TLDs - other modes need generating at 24hr and duplicating
        segmentation = nd_core.get_segmentation_level("nhb_p_m")
        extractor.tld_generator(segmentation=segmentation, **hway_kwargs)
        extractor.copy_across_tps(
            geo_area=geo_area,
            bands_name=hway_bands,
            segmentation=segmentation,
            **path_kwargs,
        )

        # ## GENERATE RAIL ## #
        rail_bands = "dm_north_rail_bands"
        if geo_area == tlds.GeoArea.GB:
            rail_bands = "dm_gb_rail_bands"

        rail_kwargs = generate_kwargs.copy()
        rail_kwargs.update({"geo_area": geo_area, "bands_name": rail_bands})

        # HB TLDs
        segmentation = nd_core.get_segmentation_level("hb_p_m_ca_rail")
        extractor.tld_generator(segmentation=segmentation, **rail_kwargs)

        # NHB TLDs - other modes need generating at 24hr and duplicating
        segmentation = nd_core.get_segmentation_level("nhb_p_m_ca_rail")
        extractor.tld_generator(segmentation=segmentation, **rail_kwargs)
        extractor.copy_across_tps(
            geo_area=geo_area,
            bands_name=rail_bands,
            segmentation=segmentation,
            **path_kwargs,
        )


def build_new_traveller_segment_tlds():
    """Build a new version of all the TLDs needed for the traveller segments"""
    # NOTE: USING 3.0 to keep light rail in with heavy
    tlb_version = "nts_tld_data_v3.0.csv"

    # Init
    extract = tlds.TripLengthDistributionBuilder(
        tlb_folder=TLB_FOLDER,
        tlb_version=tlb_version,
        bands_definition_dir=BAND_DIR,
        segment_copy_definition_dir=COPY_DEFINITIONS_DIR,
        output_folder=OUTPUT_FOLDER,
    )

    path_kwargs = {
        "geo_area": tlds.GeoArea.GB,
        "trip_filter_type": tlds.TripFilter.TRIP_OD,
        "sample_period": tlds.SampleTimePeriods.FULL_WEEK,
        "cost_units": nd_core.CostUnits.KM,
    }
    generate_kwargs = path_kwargs.copy()
    generate_kwargs.update({
        "min_sample_size": 40,
        "check_sample_size": 400,
        "inter_smoothing": True,
    })

    # # ## GENERATE RAIL TLDS ## #
    # bands_name = "dia_gb_rail_bands"
    # iterator = [
    #     ("uc_m_g_m6", "traveller_segment_m6_g", "g"),
    #     ("uc_m_soc_m6", "traveller_segment_m6_soc", "soc"),
    #     ("uc_m_ns_m6", "traveller_segment_m6_ns", "ns"),
    # ]
    #
    # for segmentation_name, copy_def_name, exc_seg in iterator:
    #     # Generate with CA combined and then split out
    #     # Generate with HB and NHB combined and then split out
    #     extract.tld_generator(
    #         bands_name=bands_name,
    #         segmentation=nd_core.get_segmentation_level(segmentation_name),
    #         aggregated_exclude_segments=exc_seg,
    #         **generate_kwargs,
    #     )
    #
    #     # Copy back out!
    #     extract.copy_tlds(
    #         copy_definition_name=copy_def_name,
    #         bands_name=bands_name,
    #         segmentation=nd_core.get_segmentation_level(segmentation_name),
    #         **path_kwargs,
    #     )

    # ## GENERATE CAR and PASSENGER TLDS ## #
    bands_name = "dia_gb_car_and_passenger_bands"
    iterator = [
        ("uc_m_g_m3", "m3_g", "g"),
        ("uc_m_soc_m3", "m3_soc", "soc"),
        ("uc_m_ns_m3", "m3_ns", "ns"),
    ]
    for segmentation_name, copy_def_name, exc_seg in iterator:
        # Generate with HB and NHB combined
        extract.tld_generator(
            bands_name=bands_name,
            segmentation=nd_core.get_segmentation_level(segmentation_name),
            aggregated_exclude_segments=exc_seg,
            **generate_kwargs,
        )

        # Split HB and NHB for DIA
        extract.copy_tlds(
            copy_definition_name=f"traveller_segment_{copy_def_name}",
            bands_name=bands_name,
            segmentation=nd_core.get_segmentation_level(segmentation_name),
            **path_kwargs,
        )

        # Split into NTEM purposes for FTS
        extract.copy_tlds(
            copy_definition_name=f"ntem_purpose_{copy_def_name}",
            bands_name=bands_name,
            segmentation=nd_core.get_segmentation_level(segmentation_name),
            **path_kwargs,
        )


if __name__ == "__main__":
    # LOGGING
    nd_log.get_logger(
        logger_name=nd_log.get_package_logger_name(),
        log_file_path=os.path.join(OUTPUT_FOLDER, LOG_FILE),
        instantiate_msg="Running TLD Builder",
        log_version=True,
    )

    # run_test()

    # run_all_combinations()
    build_new_dimo_tlds()
    # build_new_traveller_segment_tlds()
