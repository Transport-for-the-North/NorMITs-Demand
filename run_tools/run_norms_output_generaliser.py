# -*- coding: utf-8 -*-
"""
Created on: 10/05/2022
Updated on:

Original author: Ben Taylor
Last update made by:
Other updates made by:

File purpose:

"""
# Built-Ins
import pathlib

# Third Party

# Local Imports
from normits_demand.tools.norms import generaliser as norms_generaliser

# RUNNING ARGS
IMPORT_DIR = r"E:\temp\cube\2f ILF 2018\source - test"

def main():

    tp_proportions = norms_generaliser.get_norms_post_me_tp_proportions(
        norms_generaliser.NormsTpProportionFiles(pathlib.Path(IMPORT_DIR))
    )
    print("DONE")


if __name__ == '__main__':
    main()
