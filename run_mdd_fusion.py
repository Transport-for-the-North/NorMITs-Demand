# -*- coding: utf-8 -*-
"""
Created on: Mon September 13 2021
Updated on:

Original author: Peter Morris
Last update made by:
Other updates made by:

File purpose:
Scoping test runs of MDD fusion
"""

import normits_demand as nd
from normits_demand import fusion_constants as consts

# TODO: scope mdd_pre_processing structure
# TODO: locate un-expanded MDD demand


def main():

    # TODO: scope out main function
    # TODO: add required variables

    # Pre-process options
    run_preprocess_boundary = False

    # Placeholder variables
    model_name = consts.MODEL_NAMES

    if run_preprocess_boundary:
        # Pre process MDD data for expansion and fusion
        nd.MDDPreProcessing()


if __name__ == '__main__':
    main()
