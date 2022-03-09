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
import sys

# Third Party

# Local Imports
sys.path.append("..")
from normits_demand.cost import tld_builder


def main():
    # TODO(CS): path and smart search should be in constants
    _TLB_FOLDER = 'I:/NorMITs Demand/import/trip_length_distributions'
    _NTS_IMPORT = 'I:/NTS/classified builds/cb_tfn.csv'
    output_home = r'I:\NorMITs Demand\import\trip_length_distributions\tld_tool_outputs'

    run_another = True
    while run_another:
        extract = tld_builder.TripLengthDistributionBuilder(
            tlb_folder=_TLB_FOLDER,
            nts_import=_NTS_IMPORT,
            output_home=output_home,
        )

        extract.run_tlb_lookups(weekday=True)

        extract.run_tour_props(default_to_p = True)

        if input('Run another y/n').lower() == 'n':
            run_another = False


if __name__ == '__main__':
    main()
