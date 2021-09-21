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

# import normits_demand as nd
from normits_demand import fusion_constants as consts

# TODO: scope mdd_pre_processing structure
# TODO: locate un-expanded MDD demand - is this still valid?


def mode_correction():

    """ Process considerations:
    - What is the current assumption regarding modes contained within MDD data?
    - Are we intending to match high level mode splits from NTS/NoHAM models?
    - Need NoHAM LGV & HGV for mode corrections
    - Convert NoHAM LGV to person trips and MDD purposes using the outputs from NTS
    - Use high-level GOR bus splits
    """
    
    # Data preperation steps
    # TODO: import NoHAM car demand at Tf segmentation
    # TODO: import NoHAM LGV demand and convert to person trips and MDD purposes
    # TODO: import NoHAM HGV demand
    # TODO: import GOR bus splits
    #
    print("made it to the end of mode_correction")


def nts_control():

    """ Process considerations
    - Do we control by day of week or average day?
    - What level of time period dissagregation do we control at?
        - If 24hr do we have existing process to take TP data to 24hr?
    - Do we control by distance band?
        - If yes what distance bands?
        - Where do the zone-zone distances come from, EFS?
    - Where is NTS dataset?
    - What format is it in?
    - What zones are used for the square format MDD data?
    - Is controlling to NTS LAD productions and then dissagregating down acceptable?
    - What modules exist to assess the levels of change through the process?
    """
    
    # Data preperation steps
    # TODO: add any required variables/constants passed to the function
    # TODO: import MDD data from square format pickle file
    # TODO: import NTS control dataset
    # TODO: format NTS control dataset to match square MDD dataset
    # TODO: if control is being applied by distance band import distance dataset
    # TODO: import zone lookup for aggregation and matching purposes
    
    # Main alteration steps
    # TODO: aggregate MDD to LAD-LAD
    # TODO: calc LAD to zone dissagregation factor
    # TODO: aggregate NTS to LAD-LAD
    # TODO: control MDD production to NTS productions
    # TODO: dissagregate back down to zones
    
    # Process checking and outputs
    # TODO: add comparison metrics to assess level of change
    # TODO: add comparison metrics to check agreement with NTS per and post control steps
    print("made it to the end of nts_control")


def main():

    # TODO: scope out main function
    # TODO: add required variables

    # Pre-process options
    run_mode_correction = True
    run_nts_control = True

    # Placeholder variables
    # example - model_name = consts.MODEL_NAMES

    if run_mode_correction:
        # Pre process MDD data for expansion and fusion
        mode_correction()
    
    if run_nts_control:
        # Space for comments
        nts_control()

    print("end of main")


if __name__ == '__main__':
    main()
