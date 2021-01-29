# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 15:41:37 2019

@author: cruella
"""

import pandas as pd # most of the heavy lifting
import numpy as np
import os # File ops
_TMS_PATH = ()

import sys # File ops & sys config

sys.path.append('C:/Users/' + os.getlogin() + 
                '/S/NorMITs Travel Market Synthesiser/Python')
import distribution_model as dm

def trip_length_audit(file_drive='Y:/',
                      model_name='Noham',
                      iteration='iter2',
                      distributions = 'OD Matrices',
                      matrix_format = 'wide',
                      trip_length_distribution_report = True,
                      calibration_report = False):

    """
    Function to go through the distributions folder and give average trip
    lengths.
    
    Parameters
    ----------
    file_drive:
        Base drive for matrices. Will look for a 'NorMITs Synthesiser' folder
        on this drive.

    model_name:
        Name of the model to run trip length audits for. Used to define path.
        Case sensitive.

    iteration:
        Iteration of the model to run trip length audits for.

    distributions = '24hr PA Distributions':
        Distribution folder to get distributions from.

    matrix_format = 'long':
        Format of the matrices to be audited. Takes 'long' or 'wide'.
        If wide, matrices will be translated back to long format for audits.

    trip_length_distribution_report = False:
        Boolean to toggle trip length distribution report.
        If true will return DataFrame containing trip length distributions.

    calibration_report = False:
        Boolean to toggle calibration report. Imports initial betas and
        compares the calibrated trip lengths against them.

    Returns
    ----------
    trip_length_report:
        Data frame of trip length audits.

    tld_report:
        Trip length distribution report if requested, else None.

    iz_report:
        Intra zonal report if requested, else None.

    """

    w_d = (file_drive +
           'NorMITs Synthesiser/' +
           model_name +
           '/' +
           iteration +
           '/Distribution Outputs/Logs & Reports')
    os.chdir(w_d)

    dist_dir = (file_drive +
                '/NorMITs Synthesiser/' +
                model_name +
                '/' +
                iteration +
                '/Distribution Outputs/' +
                distributions)
    dists = os.listdir(dist_dir)

    # Filter out non csv (subfolders for instance)
    dists = [x for x in dists if '.csv' in x]

    model_folder = (file_drive +
                    '/NorMITs Synthesiser/' +
                    model_name +
                    '/Model Zone Lookups')

    p_h = []
    tld_ph = []
    for dist in dists:
        print(dist)
        file = dist
        import_file = pd.read_csv(dist_dir + '/' + dist)

        # If wide translate to long
        if matrix_format == 'wide':
            import_file = pd.melt(import_file,
                                  id_vars=['o_zone'],
                                  var_name='d_zone',
                                  value_name='dt',
                                  col_level=0)
            import_file = import_file[import_file['dt']>0]
            # Also rename columns back to p/a
            import_file = import_file.rename(columns={'o_zone':'p_zone',
                                                      'd_zone':'a_zone'})
            
            import_file['p_zone'] = import_file['p_zone'].astype('int16')
            import_file['a_zone'] = import_file['a_zone'].astype('int16')

        # Do as list eventually
        # Try and get the purposes from the file
        if 'nhb' in file:
            origin = 'nhb'
        elif 'hb_' in file:
            origin = 'hb'
        else:
            print('Undefined origin')
            origin = 'undefined'

        try:
            purpose = import_file['purpose'].drop_duplicates().squeeze()
            mode = import_file['mode'].drop_duplicates().squeeze()
        # If that doesn't work, get them from the dist name
        # Note, these are different from the integer ones, need another loop to catch those.
        except:
            if 'commute' in file:
                purpose = 'commute'
            elif 'business' in file:
                purpose = 'business'
            elif 'other' in file:
                purpose = 'other'
            else:
                # Print warning
                print('Undefined purpose')
                purpose = 'undefined'
            if 'mode3' in file:
                mode = 3
            elif 'mode6' in file:
                mode = 6
            else:
                print('Undefined mode')
                mode = 'undefined'

        # Car availability bodge      
        if 'car_availability_1' in file:
            car_availability = 1
        if 'car_availability_2' in file:
            car_availability = 2
        else:
            car_availability = None

        # Safe var config for rows
        if car_availability == 1:
            c_a = False
        elif car_availability == 2:
            c_a = True
        else:
            c_a = None

        if purpose == 1:
            j_p = 'commute'
        elif purpose in [2,12]:
            j_p = 'business'
        elif purpose in [3,4,5,6,7,8,13,14,15,16,18]:
            j_p = 'other'
        else:
            j_p = purpose

        # Get production numbers
        productions = import_file['dt'].sum()

        # Get intra-zonal metrics
        iz_trips = import_file[import_file['p_zone'] == import_file['a_zone']]
        iz_trips = iz_trips['dt'].sum()
        iz_perc = iz_trips/productions

        distance = dm.get_distance_and_costs(model_folder,
                                             request_type='distance',
                                             journey_purpose=j_p,
                                             direction=None,
                                             car_available=c_a,
                                             seed_intrazonal=True)

        atl = dm.get_average_trip_length(model_folder,
                                         import_file,
                                         internal_distance = distance,
                                         join_distance=True)

        row = {'file':file,
               'origin':origin,
               'mode':mode,
               'purpose':purpose,
               'car_availability':car_availability,
               'atl':atl,
               'productions':productions,
               'intra_zonal_trips':iz_trips,
               'intra_zonal_perc':iz_perc}

        if trip_length_distribution_report:
            
            tld_dist = import_file.merge(distance,
                                         how='left',
                                         on=['p_zone', 'a_zone'])

            # Output trips by target trip length distribution
            dist_cols = ['dt','distance']
            dist_bins = tld_dist.reindex(dist_cols,axis=1)
            del(tld_dist)

            dist_bins['distance'] = np.ceil(dist_bins['distance'])
            dist_bins = dist_bins.groupby('distance').sum().reset_index()

            summary_name = file.replace('.csv','')

            dist_bins = dist_bins.rename(columns={'dt':summary_name})
            tld_ph.append(dist_bins)
            del(dist_bins)

        print(row)
        p_h.append(row)

    # Append initial betas
    initial_betas = dm.get_initial_betas(path = model_folder,
                                         distribution_type = 'hb',
                                         model_name = model_name.lower(),
                                         chunk=None)

    del(initial_betas['source'])

    output = pd.DataFrame(p_h)
    
    if trip_length_distribution_report:
        tld = pd.concat(tld_ph, sort=False)
        tld = tld.fillna(0)
        tld = tld.groupby('distance').sum().reset_index()
    else:
        tld = None

    if calibration_report:
        output = initial_betas.merge(output,
                                     how = 'left',
                                     on = ['purpose', 'mode'])

        output['cal'] = 1-abs(1-output['atl']/output['average_trip_length'])

    export_name = distributions.lower().replace(' ','_')

    output.to_csv((export_name +
                   '_actual_trip_lengths.csv'), index=False)
    tld.to_csv((export_name +
                '_trip_distribution_report.csv'), index=False)

    return(output,
           tld)

def distance_audit(distance):

    """
    """

     # Distance audit
    distance['distance'].mean()

    internal_distance = distance[distance['p_zone']<=1095]

    internal_distance = internal_distance[internal_distance['a_zone']<=1095]

    internal_distance['distance'].mean()

    inter_zonal_distance = distance[distance['p_zone'] == distance['a_zone']]

    inter_zonal_distance['distance'].mean()

    return()