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
import pathlib as Path
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
    dctmddpurp = {1: ['HBW', 'HBW_fr', 'Commute'], 2: ['HBW', 'HBW_to', 'Commute'], 3: ['HBO', 'HBO_fr', 'Other'],
                  4: ['HBO', 'HBO_to', 'Other'], 5: ['NHB', 'NHB', 'Other']}
    dctmddpurpuc = {1: ['HBW', 'HBW_fr', 'Commute'], 2: ['HBW', 'HBW_to', 'Commute'],
                    31: ['HBO', 'HBO_fr', 'Other'], 33: ['HBO', 'HBO_fr', 'Other'],
                    41: ['HBO', 'HBO_to', 'Other'], 43: ['HBO', 'HBO_to', 'Other'],
                    51: ['NHB', 'NHB', 'Other'], 53: ['NHB', 'NHB', 'Other']}
    dctuc = {1: ['Business'],
             2: ['Commute'],
             3: ['Other']}
    dcttp = {1: ['AM'], 2: ['IP'], 3: ['PM']}
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
    mdd_car_file = inputs_folder + '\\' + 'dct_MDDCar.pkl'
    with open(mdd_car_file, 'rb') as log:
        dct_mdd_demand = pk.load(log)

    mdd_uc_split_file = 'Y:\\Mobile Data\\Processing\\dctmdd_uc_split.pkl'
    with open(mdd_uc_split_file, 'rb') as log:
        dct_mdd_uc_split = pk.load(log)

    dct_mdd_demand_uc = {3: {}}
    dct_mdd_demand_uc[3][1] = {}
    for pp in dctmddpurpuc:
        dct_mdd_demand_uc[3][1][pp] = {}
        purpose = (3 if pp in [31] else
                   3 if pp in [33] else
                   4 if pp in [41] else
                   4 if pp in [43] else
                   5 if pp in [51] else
                   5 if pp in [53] else
                   1 if pp in [1] else
                   2 if pp in [2] else
                   6)
        uc_split = ('1_hb_from' if pp in [31] else
                    '3_hb_from' if pp in [33] else
                    '1_hb_to' if pp in [41] else
                    '3_hb_to' if pp in [43] else
                    '1_nhb' if pp in [51] else
                    '3_nhb' if pp in [53] else
                    'commute' if pp in [1, 2] else
                    'missing')
        for tp in dcttp:
            if pp in [1, 2]:
                # multiply by 1
                dct_mdd_demand_uc[3][1][pp][tp] = dct_mdd_demand[3][1][purpose][tp] * 1
            else:
                dct_mdd_demand_uc[3][1][pp][tp] = dct_mdd_demand[3][1][purpose][tp] * dct_mdd_uc_split[3][1][uc_split][tp]

    temp_array = np.zeros((2770, 2770))
    dct_mdd_demand_uc_comb = {3: {}}
    dct_mdd_demand_uc_comb[3][1] = {}
    for uc in dctuc:
        dct_mdd_demand_uc_comb[3][1][uc] = {}
        for tp in dcttp:
            dct_mdd_demand_uc_comb[3][1][uc][tp] = temp_array

    for pp in dctmddpurpuc:
        userclass = (1 if pp in [31, 41, 51] else
                     2 if pp in [1, 2] else
                     3 if pp in [33, 43, 53] else
                     4)
        for tp in dcttp:
            dct_mdd_demand_uc_comb[3][1][userclass][tp] = (dct_mdd_demand_uc_comb[3][1][userclass][tp] + dct_mdd_demand_uc[3][1][pp][tp])

    # Import noham PCUs
    noham_car_file = inputs_folder + '\\' + 'dctNoHAM_mddpurp_pcu.pkl'
    with open(noham_car_file, 'rb') as log:
        dct_noham_demand = pk.load(log)

    noham_car_file = inputs_folder + '\\' + 'dctNoHAM_compiled_PCU.pkl'
    with open(noham_car_file, 'rb') as log:
        dct_noham_demand = pk.load(log)


    # TODO: loop demand dct and create factor dct
    temp_array = np.zeros((2770, 2770))
    input_matrix = temp_array

    # package mdd fusion factors
    dct_mdd_fusion_factors = {3: {}}
    dct_mdd_fusion_factors[3][1] = {}
    for pp in dctuc:
        dct_mdd_fusion_factors[3][1][pp] = {}
        distance_matrix = dct_distance[dctuc[pp][0]]
        fusion_matrix = mdd_fusion_engine.build_fusion_factor(input_matrix,
                                                              distance_matrix,
                                                              origin_type_matrix,
                                                              dest_type_matrix,
                                                              chop_head=True,
                                                              chop_tail=False,
                                                              origin_type=True,
                                                              dest_type=True,
                                                              invert=False,
                                                              min_dist=0,
                                                              max_dist=9999,
                                                              default_value=1)
        for tp in dcttp:
            dct_mdd_fusion_factors[3][1][pp][tp] = fusion_matrix
    # package noham fusion factors
    dct_noham_fusion_factors = {3: {}}
    dct_noham_fusion_factors[3][1] = {}
    for pp in dctuc:
        dct_noham_fusion_factors[3][1][pp] = {}
        distance_matrix = dct_distance[dctuc[pp][0]]
        fusion_matrix = mdd_fusion_engine.build_fusion_factor(input_matrix,
                                                              distance_matrix,
                                                              origin_type_matrix,
                                                              dest_type_matrix,
                                                              chop_head=True,
                                                              chop_tail=False,
                                                              origin_type=True,
                                                              dest_type=True,
                                                              invert=True,
                                                              min_dist=0,
                                                              max_dist=9999,
                                                              default_value=1)
        for tp in dcttp:
            dct_noham_fusion_factors[3][1][pp][tp] = fusion_matrix
    # TODO: combine demand
    dct_fusion_demand = {3: {}}
    dct_fusion_demand[3][1] = {}
    for uc in dctuc:
        dct_fusion_demand[3][1][uc] = {}
        for tp in dcttp:
            dct_fusion_demand[3][1][uc][tp] = ((dct_mdd_demand_uc_comb[3][1][uc][tp] * dct_mdd_fusion_factors[3][1][uc][tp])
                                               +
                                               (dct_noham_demand[3][1][uc][tp] * dct_noham_fusion_factors[3][1][uc][tp]))

    unq_zones = list(range(1, 2771))
    version = 0
    export_folder = 'Y:/Mobile Data/Processing/3-1_Fusion_Demand'
    md = 3
    wd = 1
    for uc in dctuc:
        for tp in dcttp:
            folder_path = (export_folder + '/v' + str(version) + '/PCUTrips')
            Path(folder_path).mkdir(parents=True, exist_ok=True)
            file_path = (folder_path + '/' +
                         'od_' + str(dctmddpurp[uc][0]) + '_p' + str(uc) +
                         '_yr2018_m' + str(md) +
                         '_tp' + str(tp) + '.csv')
            print(file_path)
            export_array = dct_fusion_demand[md][wd][uc][tp]
            export_df = pd.DataFrame(data=export_array, index=unq_zones, columns=unq_zones)
            export_df.to_csv(file_path, float_format='%.5f', header=False)

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
