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

        extract.run_tlb_lookups()

        if input('Run another y/n').lower() == 'n':
            run_another = False


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
    main()
