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
from pathlib import Path
from normits_demand.mdd_fusion import mdd_fusion_engine


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
    dctuc = {1: ['business', 'Business'],
             2: ['commute', 'Commute'],
             3: ['other', 'Other']}
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
    mdd_car_file = inputs_folder + '\\' + 'dct_MDDCar_UC_pcu.pkl'
    with open(mdd_car_file, 'rb') as log:
        dct_mdd_demand = pk.load(log)
    # Import NoHAM demand
    noham_car_file = inputs_folder + '\\' + 'dct_NoHAM_Synthetic_PCU.pkl'
    with open(noham_car_file, 'rb') as log:
        dct_noham_demand = pk.load(log)
    # Import Min Distance cut offs
    head_cut_dist_file = inputs_folder + '\\' + 'dictcutoff_v0.pkl'
    with open(head_cut_dist_file, 'rb') as log:
        dct_head_cut_dist = pk.load(log)

    """
    # Import noham PCUs
    noham_car_file = inputs_folder + '\\' + 'dctNoHAM_mddpurp_pcu.pkl'
    with open(noham_car_file, 'rb') as log:
        dct_noham_demand = pk.load(log)

    noham_car_file = inputs_folder + '\\' + 'dctNoHAM_compiled_PCU.pkl'
    with open(noham_car_file, 'rb') as log:
        dct_noham_demand = pk.load(log)
    """

    # TODO: loop demand dct and create factor dct
    temp_array = np.zeros((2770, 2770))
    input_matrix = temp_array

    # package mdd fusion factors
    dct_mdd_fusion_factors = {3: {}}
    dct_mdd_fusion_factors[3][1] = {}
    for uc in dctuc:
        pp = (3 if uc in [1] else
              1 if uc in [2] else
              3 if uc in [3] else
              6)
        dct_mdd_fusion_factors[3][1][uc] = {}
        distance_matrix = dct_distance[dctuc[uc][1]]
        for tp in dcttp:
            min_dist_cut = dct_head_cut_dist[3][1][pp][tp]
            print(min_dist_cut)
            fusion_matrix = mdd_fusion_engine.build_fusion_factor(input_matrix,
                                                                  distance_matrix,
                                                                  origin_type_matrix,
                                                                  dest_type_matrix,
                                                                  chop_head=True,
                                                                  chop_tail=False,
                                                                  origin_type=True,
                                                                  dest_type=True,
                                                                  inclusive=True,
                                                                  invert=False,
                                                                  min_dist=min_dist_cut,
                                                                  max_dist=9999,
                                                                  default_value=1)
            dct_mdd_fusion_factors[3][1][uc][tp] = fusion_matrix
    # package noham fusion factors
    dct_noham_fusion_factors = {3: {}}
    dct_noham_fusion_factors[3][1] = {}
    for uc in dctuc:
        pp = (3 if uc in [1] else
              1 if uc in [2] else
              3 if uc in [3] else
              6)
        dct_noham_fusion_factors[3][1][uc] = {}
        distance_matrix = dct_distance[dctuc[uc][1]]
        for tp in dcttp:
            min_dist_cut = dct_head_cut_dist[3][1][pp][tp]
            print(min_dist_cut)
            fusion_matrix = mdd_fusion_engine.build_fusion_factor(input_matrix,
                                                                  distance_matrix,
                                                                  origin_type_matrix,
                                                                  dest_type_matrix,
                                                                  chop_head=True,
                                                                  chop_tail=False,
                                                                  origin_type=True,
                                                                  dest_type=True,
                                                                  inclusive=True,
                                                                  invert=True,
                                                                  min_dist=min_dist_cut,
                                                                  max_dist=9999,
                                                                  default_value=1)
            dct_noham_fusion_factors[3][1][uc][tp] = fusion_matrix
    # Calc noham target
    dct_noham_target = {3: {}}
    dct_noham_target[3][1] = {}
    for uc in dctuc:
        dct_noham_target[3][1][uc] = {}
        for tp in dcttp:
            print(np.sum(dct_noham_demand[3][1][uc][tp] * dct_mdd_fusion_factors[3][1][uc][tp]))
            dct_noham_target[3][1][uc][tp] = (dct_noham_demand[3][1][uc][tp] * dct_mdd_fusion_factors[3][1][uc][tp])
    # Calc MDD Fusion total
    dct_mdd_fusion_total = {3: {}}
    dct_mdd_fusion_total[3][1] = {}
    for uc in dctuc:
        dct_mdd_fusion_total[3][1][uc] = {}
        for tp in dcttp:
            print(np.sum(dct_mdd_demand[3][1][uc][tp] * dct_mdd_fusion_factors[3][1][uc][tp]))
            dct_mdd_fusion_total[3][1][uc][tp] = (dct_mdd_demand[3][1][uc][tp] * dct_mdd_fusion_factors[3][1][uc][tp])
    # Calc expansion factor
    dct_mdd_expand = {3: {}}
    dct_mdd_expand[3][1] = {}
    for uc in dctuc:
        dct_mdd_expand[3][1][uc] = {}
        for tp in dcttp:
            print('3-1-' + str(uc) + '-' + str(tp))
            print((np.sum(dct_noham_target[3][1][uc][tp]) / np.sum(dct_mdd_fusion_total[3][1][uc][tp])))
            dct_mdd_expand[3][1][uc][tp] = (np.sum(dct_noham_target[3][1][uc][tp]) / np.sum(dct_mdd_fusion_total[3][1][uc][tp]))
    # combine demand
    dct_expansion_factor = {3: {1: {1: {1: {}, 2: {}, 3: {}},
                                    2: {1: {}, 2: {}, 3: {}},
                                    3: {1: {}, 2: {}, 3: {}}}}}
    # TP1
    dct_expansion_factor[3][1][1][1] = 1.04
    dct_expansion_factor[3][1][1][2] = 1.37
    dct_expansion_factor[3][1][1][3] = 1.47
    # TP2
    dct_expansion_factor[3][1][2][1] = 6.32
    dct_expansion_factor[3][1][2][2] = 2.65
    dct_expansion_factor[3][1][2][3] = 6.91
    # TP3
    dct_expansion_factor[3][1][3][1] = 2.94
    dct_expansion_factor[3][1][3][2] = 2.98
    dct_expansion_factor[3][1][3][3] = 2.71

    expansion_applied = True
    dct_fusion_demand = {3: {}}
    dct_fusion_demand[3][1] = {}
    for uc in dctuc:
        dct_fusion_demand[3][1][uc] = {}
        for tp in dcttp:
            if expansion_applied:
                dct_fusion_demand[3][1][uc][tp] = ((dct_mdd_demand[3][1][uc][tp] * dct_mdd_fusion_factors[3][1][uc][tp] * dct_expansion_factor[3][1][uc][tp])
                                                   +
                                                   (dct_noham_demand[3][1][uc][tp] * dct_noham_fusion_factors[3][1][uc][tp]))
            else:
                dct_fusion_demand[3][1][uc][tp] = ((dct_mdd_demand[3][1][uc][tp] * dct_mdd_fusion_factors[3][1][uc][tp])
                                                   +
                                                   (dct_noham_demand[3][1][uc][tp] * dct_noham_fusion_factors[3][1][uc][
                                                       tp]))

    unq_zones = list(range(1, 2771))
    version = '3-6'
    export_folder = 'Y:/Mobile Data/Processing/3-1_Fusion_Demand'
    md = 3
    wd = 1
    for uc in dctuc:
        for tp in dcttp:
            folder_path = (export_folder + '/v' + str(version) + '/PCUTrips')
            Path(folder_path).mkdir(parents=True, exist_ok=True)
            file_path = (folder_path + '/' +
                         'od_' + str(dctuc[uc][0]) + '_p' + str(uc) +
                         '_yr2018_m' + str(md) +
                         '_tp' + str(tp) + '.csv')
            print(file_path)
            export_array = dct_fusion_demand[md][wd][uc][tp]
            export_df = pd.DataFrame(data=export_array, index=unq_zones, columns=unq_zones)
            export_df.to_csv(file_path, float_format='%.5f', index=False, header=False)
    # Export demand totals
    totals_check = True
    check_location = 'Y:\\Mobile Data\\Processing\\3-1_Fusion_Demand'
    if totals_check:
        # Build totals dictionary
        dct_total = {3: {1: {}}}
        for uc in dctuc:
            dct_total[3][1][uc] = {}
            for tp in dcttp:
                print(str(3) + '-' + str(1) + '-' + str(uc) + '-' + str(tp))
                dct_total[3][1][uc][tp] = np.sum(dct_fusion_demand[3][1][uc][tp])
        df_totals = pd.DataFrame.from_dict({(i, j, k): dct_total[i][j][k]
                                            for i in dct_total.keys()
                                            for j in dct_total[i].keys()
                                            for k in dct_total[i][j].keys()},
                                           orient='index')
        df_totals.to_csv(check_location + '\\FusionDemand_' + version + '_Totals.csv')
    # Export to pickle file
    folder_path = (export_folder + '/v' + str(version) + '/PCUTrips')
    with open(folder_path + '\\' + 'dct_FusionDemand_' + version + '.pkl', 'wb') as log:
        pk.dump(dct_fusion_demand, log, pk.HIGHEST_PROTOCOL)

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
