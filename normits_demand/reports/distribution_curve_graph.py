# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 12:53:53 2020

@author: cruella
"""

import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def build_dist_curves(file_drive='Y:/',
                      model_name='Noham',
                      iteration='iter3',
                      mode_subset = [3],
                      write = True,
                      echo = False):
    
    """
    Go into a distribution report and build curves from the cols
    """
    # TODO: Segment handling.
    
    # Check and handle mode_subset is a list
    if type(mode_subset)==int:
        print('Please pass mode subset as a list not an integer, ta')
        new_list = []
        new_list.append(mode_subset)
        mode_subset = new_list

    # Set lookup to dist export
    w_d = (file_drive +
           'NorMITs Synthesiser/' +
           model_name +
           '/' +
           iteration +
           '/Distribution Outputs/Trip Length Distributions')

    output_folder = w_d + '/Distribution Plots/'
    # If it doesn't exist, make it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print('Created new folder')
        print(output_folder)

    # Look for distribution lines folder
    tlds = os.listdir(w_d)
    # csv only
    tlds = [x for x in tlds if '.csv' in x]
    tlds = [x for x in tlds if 'bin' in x]

    graph_bin = []
    for tld in tlds:
        if echo:
            print(tld)

        # Define hard mode and pretty mode for graphs
        # Might as well do one at the same as the other
        if 'mode1' in tld or 'm1' in tld:
            c_mode = 'walk'
            i_mode = 1
        elif 'mode2' in tld or 'm2' in tld:
            c_mode = 'cycle'
            i_mode = 2
        elif 'mode3' in tld or 'm3' in tld:
            c_mode = 'car'
            i_mode = 3
        elif 'mode5' in tld or 'm5 'in tld:
            c_mode = 'PT'
            i_mode = 5
        elif 'mode6' in tld or 'm6' in tld:
            c_mode = 'rail'
            i_mode = 6
        else:
            print('Undefined mode')
            c_mode = 'undefined'
            i_mode = None

        # Import is the most memory intensive bit. Only do that if it's in the target modes.
        if i_mode in mode_subset:
            # Import
            dat = pd.read_csv(w_d + '/' + tld)
    
            # Gonna have to define params from file name :(
            # TODO: This is pretty nifty now :) move to tld report in RA & functionalise
            if 'nhb' in tld:
                c_origin = 'nhb'
            else:
                c_origin = 'hb'
        
            if 'car_availability1' in tld or 'ca1' in tld:
                c_ca = 'NCA'
            elif 'car_availability2' in tld or 'ca2' in tld:
                c_ca = 'CA'
            else:
                # Empty but not null
                c_ca = ''

            if 'purpose_1_' in tld or '_p1_' in tld:
                c_purp = 'commute'
            elif 'purpose_2_' in tld or '_p2' in tld or 'purpose_12_' in tld or '_p12' in tld:
                c_purp = 'employer\'s business'
            elif 'purpose_3_' in tld or '_p3' in tld or 'purpose_13_' in tld or '_p13' in tld:
                c_purp = 'education'
            elif 'purpose_4_' in tld or '_p4' in tld or 'purpose_14_' in tld or '_p14' in tld:
                c_purp = 'shopping'
            elif 'purpose_5_' in tld or '_p5' in tld or 'purpose_15_' in tld or '_p15' in tld:
                c_purp = 'personal business'
            elif 'purpose_6_' in tld or '_p6' in tld or 'purpose_16_' in tld or '_p16' in tld:
                c_purp = 'social'
            elif 'purpose_7_' in tld or '_p7' in tld:
                c_purp = 'visiting friends'
            elif 'purpose_8_' in tld or '_p8' in tld or 'purpose_18_' in tld or '_p18' in tld:
                c_purp = 'holiday \ day trip'
            else:
                c_purp = None
    
            if 'soc0' in tld:
                c_soc = 'other soc'
            elif 'soc1' in tld:
                c_soc = 'high soc'
            elif 'soc2' in tld:
                c_soc = 'med soc'
            elif 'soc3' in tld:
                c_soc = 'low soc'
            else:
                c_soc = ''

            if 'ns1' in tld:
                c_ns = 'high ns'
            elif 'ns2' in tld:
                c_ns = 'high-med ns'
            elif 'ns3' in tld:
                c_ns = 'med ns'
            elif 'ns4' in tld:
                c_ns = 'low ns'
            elif 'ns5' in tld:
                c_ns = 'student ns'
            else:
                c_ns = ''

            # Check mode type is in there
            if echo:
                print('Building distribution graph for ' + tld)

            # Define plot name
            plot_name = (c_origin + ' ' +
                         c_mode + ' ' +
                         c_purp + ' ' +
                         c_ca + ' ' +
                         c_soc + ' ' +
                         c_ns)
            export_name = (output_folder + tld + '.png')

            graph_sub = dat.reindex(['distance', 'dt'],axis=1)
            # Make plot
            plt.figure()
            line_plot = sns.set(style='darkgrid')
            line_plot = sns.lineplot(x='distance', y='dt', data=graph_sub)
            line_plot.set_title(plot_name)
            line_plot.set_xlabel('distance (km)')
            line_plot.set_ylabel('trips (count)')
            # Export
            graph = line_plot.get_figure()

            # Write out
            if write:
                graph.savefig(export_name)
                # Append name and plot to bin

            # In any case
            graph_bin.append({plot_name:graph})
            # Get rid of graph object or they stack
            del(graph)
            del(line_plot)

    return(graph_bin)