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
# Local Imports
from normits_demand.cost import tld_generator


def main(iterator=False):
    _TLB_FOLDER = 'I:/NTS/outputs/tld'
    _TLB_VERSION = 'nts_tld_data_v3.1.csv'
    _OUTPUT_FOLDER = r'I:\NorMITs Demand\import\trip_length_distributions\tld_tool_outputs'
    _TLD_HOME = r'I:\NorMITs Demand\import\trip_length_distributions\config'
    _BAND_FOLDER = os.path.join(_TLD_HOME, 'bands')

    available_bands = os.listdir(_BAND_FOLDER)
    available_bands = [x for x in available_bands if '.csv' in x]

    _SEGMENTATION_FOLDER = os.path.join(_TLD_HOME, 'segmentations')
    available_segmentations = os.listdir(_SEGMENTATION_FOLDER)
    available_segmentations = [x for x in available_segmentations if '.csv' in x]

    bands_path = os.path.join(_BAND_FOLDER, available_bands[0])
    segmentation_path = os.path.join(_SEGMENTATION_FOLDER, available_segmentations[0])

    extract = tld_generator.TripLengthDistributionGenerator(
        tlb_folder=_TLB_FOLDER,
        tlb_version=_TLB_VERSION,
        output_folder=_OUTPUT_FOLDER,
    )

    if iterator:
        for bsp in available_bands:
            for asp in available_segmentations:
                extract.tld_generator(
                    geo_area='north_incl_ie',
                    sample_period='week',
                    trip_filter_type='trip_OD',
                    bands_path=os.path.join(_BAND_FOLDER, bsp),
                    segmentation_path=os.path.join(_SEGMENTATION_FOLDER, asp),
                    cost_units='km',
                    sample_threshold=10,
                    verbose=True
                )

    else:
        extract.tld_generator(
            geo_area='north_incl_ie',
            sample_period='week',
            trip_filter_type='trip_OD',
            bands_path=bands_path,
            segmentation_path=segmentation_path,
            cost_units='km',
            sample_threshold=10,
            verbose=True
        )


if __name__ == '__main__':
    main(iterator=True)
