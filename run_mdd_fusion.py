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

# TODO: tidy imports into functions
import pandas as pd
import numpy as np
import pickle as pk
import os, sys
import multiprocessing as mp
import normits_demand as nd
from normits_demand.mdd_fusion import mdd_fusion_engine
from normits_demand import fusion_constants as consts

distance_folder = r'Y:\NoHAM\17.TUBA_Runs\-TPT\Skims\RefDist_Skims'
inputs_folder = r'Y:\Mobile Data\Processing\Fusion_Inputs'
# TODO: Add looping dictionaries

def mode_correction():
    # TODO: is this still valid?
    # TODO: Tidy and retain as required
    """ Process considerations:
    - What is the current assumption regarding modes contained within MDD data?
    - Are we intending to match high level mode splits from NTS/NoHAM models?
    - Need NoHAM LGV & HGV for mode corrections
    - Convert NoHAM LGV to person trips and MDD purposes using the outputs from NTS
    - Use high-level GOR bus splits
    -
    """
    
    # Data preparation steps
    # TODO: import NoHAM car demand at Tf segmentation
    # TODO: add purpose, mode & tp constants from fusion_constants
    # TODO: setup matching numpy dictionary for NoHAM demand
    # TODO: build out import loop
    # set zone numbers
    unq_zones = list(range(1, 2771))
    # import single od purpose, mode, tp matrix
    noham_car = np.genfromtxt(r'I:\NorMITs Demand\noham\EFS\iter3i\SC01_JAM\Matrices\OD Matrices\hb_od_from_yr2033_p1_m3_tp1.csv',
                              delimiter=',',
                              skip_header=1,
                              usecols=unq_zones)
    # check contents
    print(type(noham_car))
    print(np.info(noham_car))
    print(noham_car[0:3, 0:3])


    # TODO: import NoHAM LGV demand and convert to person trips and MDD purposes
    # TODO: import NoHAM HGV demand
    # TODO: import GOR bus splits

    print("made it to the end of mode_correction")


def nts_control():
    # TODO: How much of this is still valid?
    """ Process considerations
    - Do we control by day of week or average day?
    - What level of time period disaggregation do we control at?
        - If 24hr do we have existing process to take TP data to 24hr?
    - Do we control by distance band?
        - If yes what distance bands?
        - Where do the zone-zone distances come from, EFS?
    - Where is NTS dataset?
    - What format is it in?
    - What zones are used for the square format MDD data?
    - Is controlling to NTS LAD productions and then disaggregating down acceptable?
    - What modules exist to assess the levels of change through the process?
    """
    
    # Data preparation steps
    # TODO: add any required variables/constants passed to the function
    # TODO: import MDD data from square format pickle file
    # TODO: import NTS control dataset
    # TODO: format NTS control dataset to match square MDD dataset
    # TODO: if control is being applied by distance band import distance dataset
    # TODO: import zone lookup for aggregation and matching purposes
    
    # Main alteration steps
    # TODO: aggregate MDD to LAD-LAD
    # TODO: calc LAD to zone disaggregation factor
    # TODO: aggregate NTS to LAD-LAD
    # TODO: control MDD production to NTS productions
    # TODO: disaggregate back down to zones
    
    # Process checking and outputs
    # TODO: add comparison metrics to assess level of change
    # TODO: add comparison metrics to check agreement with NTS per and post control steps
    print("made it to the end of nts_control")


def fusion_factors():
    # Import distances dictionary
    distance_file = inputs_folder + '\\' + 'NoHAM_Distances.pkl'
    with open(distance_file, 'rb') as log:
        dct_distance = pk.load(log)
    # Import matrix type arrays
    # Origin types
    origin_file = inputs_folder + '\\' + 'Origin_Type.csv'
    origin_type_matrix = pd.read_csv(origin_file, delimiter=',', header=0)
    origin_type_matrix = origin_type_matrix.to_numpy()
    # Destination types
    dest_file = inputs_folder + '\\' + 'Dest_Type.csv'
    dest_type_matrix = pd.read_csv(dest_file, delimiter=',', header=0)
    dest_type_matrix = dest_type_matrix.to_numpy()
    # Import mdd demand
    # TODO: check if PCUs
    mdd_car_file = inputs_folder + '\\' + 'MDD_Car.pkl'
    with open(mdd_car_file, 'rb') as log:
        dct_mdd_demand = pk.load(log)
    # Import noham PCUs
    # TODO: Check for PCU file
    noham_car_file = inputs_folder + '\\' + 'MDD_Car.pkl'
    with open(noham_car_file, 'rb') as log:
        dct_noham_demand = pk.load(log)

    # TODO: loop demand dct and create factor dct
    temp_array = np.zeros((2770, 2770))
    input_matrix = temp_array
    distance_matrix = dct_distance['Commute']
    # TODO: package fusion factors into dct
    fusion_matrix = mdd_fusion_engine.build_fusion_factor(input_matrix,
                                                          distance_matrix,
                                                          origin_type_matrix,
                                                          dest_type_matrix,
                                                          chop_head=False,
                                                          chop_tail=False,
                                                          origin_type=False,
                                                          dest_type=False,
                                                          min_dist=0,
                                                          max_dist=9999,
                                                          default_value=1)
    # TODO: combine demand

def main():

    # TODO: Link to pre_processing files
    # Pre-process options
    run_mode_correction = False
    run_nts_control = False
    if run_mode_correction:
        # Pre process MDD data for expansion and fusion
        mode_correction()
    if run_nts_control:
        # Space for comments
        nts_control()

    run_package_fusion_dist = False
    if run_package_fusion_dist:
        # Updates distance skims used within fusion
        print(mdd_fusion_engine.package_fusion_distances(inputs_path=distance_folder,
                                                         output_path=r'Y:\Mobile Data\Processing\Fusion_Inputs',
                                                         version_name='NoHAM_Base_2018_TS2_v106'))
    run_fusion_factors = True
    if run_fusion_factors:

        fusion_factors()


    print("end of main")


if __name__ == '__main__':
    main()
