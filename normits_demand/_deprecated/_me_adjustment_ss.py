# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 08:37:06 2019
Functions to apply a matrix estimation adjustment
Needs OD matrices in the same format as the distribution model outputs them.

@author: genie
"""

import pandas as pd # most of the heavy lifting
import os # File ops
import sys # File ops & sys config
sys.path.append('C:/Users/' + os.getlogin() + 
                '/S/NorMITs Travel Market Synthesiser/Python')
import distribution_model as dm # For auto path config

_model_name = 'Noham'

# TODO: Needs to go into matrix processing
def matrix_wide_to_long(wide_matrix,
                        merge_cols = ['p_zone', 'a_zone']):
    
    """
    Takes p/o, a/d & trips
    """

    # TODO: Check len matches
    wide_matrix['dt'] = wide_matrix['dt'].fillna(0)

    # Go long to wide
    wide_matrix = wide_matrix.pivot(index = merge_cols[0],
                                    columns = merge_cols[1],
                                    values = 'dt')

    return(wide_matrix)

# Main
if __name__ == '__main__':
    # TODO: Get params to define import params

    # Config path using dist model function
    import_params = dm.path_config(file_drive = dm._default_import_file_drive,
                                   model_name = _model_name,
                                   iteration = 'iter1')
    export_folders = import_params[4]
    print(os.getcwd())

    flip = True
    vehicle_split = True
    missing_tp = 'tp4'

    # get zone lookup path
    zone_lookup_path = (import_params[1] +
                        '/lad_' +
                        _model_name.lower() +
                        '_pop_weighted_lookup.csv')

    # Get zone to district lookup
    zone_to_district = pd.read_csv(zone_lookup_path)
    ztd_cols = list(zone_to_district)
    zone_to_district = zone_to_district.reindex([ztd_cols[0],
                                                 ztd_cols[1],
                                                 ztd_cols[6]],
    axis=1)
    del(ztd_cols)

    # Get trip splits for PA-OD
    time_period_splits = dm.get_time_period_splits(
            aggregate = True)
    # Test aggregating these to od purpose segments
    # Should be temporary
    export_subsets = {'commute':[1],
                      'business':[2],
                      'other':[3,4,5,6,7,8]}
    export_subsets = pd.DataFrame(export_subsets.items())
    new = []
    for index,row in export_subsets.iterrows():
        for item in row[1]:
            print(item)
            new.append(pd.Series({'purpose':row[0], 'purpose_id':item}))
    new = pd.concat(new, axis=1, ignore_index=False).transpose()
    
    # Sort these time period splits to work at aggregate purpose and time period
    time_p_s_adjusted = time_period_splits.merge(
            new,
            how = 'left',
            left_on = 'purpose_to_home',
            right_on = 'purpose_id').drop(
                    'purpose_id',
                    axis=1).rename(
                            columns={'purpose':'agg_purpose_to_home'}).merge(
                                    new,
                                    how = 'left',
                                    left_on = 'purpose_from_home',
                                    right_on = 'purpose_id').drop(
                                            'purpose_id',
                                            axis=1).rename(
                                                    columns = {
                                                            'purpose':
                                                                'agg_purpose_from_home'})

    time_p_s_adjusted = time_p_s_adjusted.groupby(['agg_purpose_to_home',
                                            'agg_purpose_from_home']).mean(
        ).drop(['time_from_home','time_to_home'], axis=1).reset_index()
    time_p_s_sum = time_p_s_adjusted.groupby('agg_purpose_from_home').sum()
    time_p_s_sum = time_p_s_sum.rename(columns={'direction_factor':'total_split'})
    time_p_s_adjusted = time_p_s_adjusted.merge(time_p_s_sum,
                                                how='left',
                                                on='agg_purpose_from_home')
    time_p_s_adjusted['direction_factor'] = (time_p_s_adjusted[
            'direction_factor']/time_p_s_adjusted['total_split'])
    time_p_s_adjusted = time_p_s_adjusted.drop('total_split', axis=1)
    
    # Get od matrices
    od_matrices = os.listdir(os.getcwd() +
                             export_folders[8])

    od_vehicle_matrices = os.listdir(os.getcwd() +
                                 export_folders[8] +
                                 '/vehicle export')

    me_matrices = os.listdir(os.getcwd() + export_folders[9])

    me_people_matrices = os.listdir(os.getcwd() +
                                     export_folders[9] +
                                     '/people export')

    # TODO: If person folder is empty, undo vehicle split

    # Get files in order
    mode = '3'
    # Get ME files, get missing od, where possible
    od_target_mode = [x for x in od_matrices if ('mode' + mode) in x]
    me_target_mode = [x for x in me_people_matrices if ('mode' + mode) in x]
    # Fill in with missing time periods from original
    od_me_target_mode = [x for x in od_target_mode if missing_tp in x]
    # Filter down to hb_only
    hb_od_target_mode = [x for x in od_target_mode if 'nhb' not in x]
    hb_od_me_target_mode = [x for x in od_me_target_mode if 'nhb' not in x]
    hb_me_target_mode = [x for x in me_target_mode if 'nhb' not in x]

    # Import all time periods for each purpose
    purpose_list = ['commute', 'business', 'other']
    od_bin = []
    me_bin = []
    for purpose in purpose_list:
        # Filter vector of csvs to purpose
        hb_od_me_target_segment = [x for x in hb_od_me_target_mode if purpose in x]
        hb_me_target_segment = [x for x in hb_me_target_mode if purpose in x]
        hb_od_target_segment = [x for x in hb_od_target_mode if purpose in x]

        me_purpose_bin = []

        # Get intact matrices
        print(hb_od_me_target_segment)
        for csv in hb_od_me_target_segment:
            print(csv)
            # establish directionality
            if 'from' in csv:
                direction = 'from'
            elif 'to' in csv:
                direction = 'to'
            else:
                print('No directionality, check imports and code')

            # Don't need this anymore since non-aggregate time period is impossible
            # Get time period from csv import, using location
            tp_loc = csv.find('tp')
            # Location should be 2 up from start of tp
            tp = csv[tp_loc+2]

            # Import csv in question
            ph = pd.read_csv(os.getcwd() +
                             export_folders[8] + # This needs to by 9 for ME
                             '/' +
                             csv,
                             index_col = 0)

            # Wide to long
            ph = ph.unstack(
                    level=0).reset_index(
                            ).rename(columns={'level_0':'d_zone',
                            0:'dt'})
            print(sum(ph['dt']))    

            # Filter out zero values
            ph = ph[ph['dt']!=0].copy()

            # Input time period and purpose
            # ph['time'] = tp
            ph['purpose'] = purpose
            ph = ph.groupby(['o_zone',
                             'd_zone',
                             'purpose']).sum().reset_index()

            # Just got to face it, these are coming back with aggregate purpose and time period

            # If 'to' flip direction
            # How do I account for off purposes?
            if flip == True:
                if direction == 'to':
                    # Back to PA
                    ph = ph.merge(time_p_s_adjusted,
                                  how='left',
                                  left_on='purpose',
                                  right_on='agg_purpose_to_home').drop(
                                          ['agg_purpose_to_home',
                                           'purpose'], axis=1).rename(
                                           columns={'agg_purpose_from_home':'purpose'}).reset_index(
                                                   drop=True)
                    ph['dt'] = ph['dt'] * ph['direction_factor']
                    ph = ph.drop('direction_factor', axis=1)
                    ph = ph.groupby(['o_zone',    
                                     'd_zone',
                                     'purpose']).sum().reset_index()
                    ph = ph.rename(columns={'o_zone':'nd_zone',
                                            'd_zone':'o_zone'})
                    ph = ph.rename(columns={'nd_zone':'d_zone'})

            me_purpose_bin.append(ph)

        # Get estimated matrices
        print(hb_me_target_segment)
        for csv in hb_me_target_segment:
            print(csv)
            # establish directionality
            if 'from' in csv:
                direction = 'from'
            elif 'to' in csv:
                direction = 'to'
            else:
                print('No directionality, check imports and code')

            # Don't need this anymore since non-aggregate time period is impossible
            # Get time period from csv import, using location
            tp_loc = csv.find('tp')
            # Location should be 2 up from start of tp
            tp = csv[tp_loc+2]

            # Import csv in question
            ph = pd.read_csv(os.getcwd() +
                             export_folders[9] + # This needs to by 9 for ME
                             '/people export/' +
                             csv,
                             index_col = 0)

            # Wide to long
            ph = ph.unstack(
                    level=0).reset_index(
                            ).rename(columns={'level_0':'d_zone',
                            0:'dt',
                            'O':'o_zone'})
            print(sum(ph['dt']))

            # Filter out zero values
            ph = ph[ph['dt']!=0].copy()

            # Input time period and purpose
            # ph['time'] = tp
            ph['purpose'] = purpose
            ph = ph.groupby(['o_zone',
                             'd_zone',
                             'purpose']).sum().reset_index()

            # Just got to face it, these are coming back with aggregate purpose and time period

            # If 'to' flip direction
            # How do I account for off purposes?
            if flip == True:
                if direction == 'to':
                    # Back to PA
                    ph = ph.merge(time_p_s_adjusted,
                                  how='left',
                                  left_on='purpose',
                                  right_on='agg_purpose_to_home').drop(
                                          ['agg_purpose_to_home',
                                           'purpose'], axis=1).rename(
                                           columns={'agg_purpose_from_home':'purpose'}).reset_index(
                                                   drop=True)
                    ph['dt'] = ph['dt'] * ph['direction_factor']
                    ph = ph.drop('direction_factor', axis=1)
                    ph = ph.groupby(['o_zone',    
                                     'd_zone',
                                     'purpose']).sum().reset_index()
                    ph = ph.rename(columns={'o_zone':'nd_zone',
                                            'd_zone':'o_zone'})
                    ph = ph.rename(columns={'nd_zone':'d_zone'}).copy()

            me_purpose_bin.append(ph)
        me_purpose = pd.concat(me_purpose_bin)
        # Get od purposes
        od_purpose_bin = []

        # Get intact matrices
        print(hb_od_target_segment)
        for csv in hb_od_target_segment:
            print(csv)
            # establish directionality
            if 'from' in csv:
                direction = 'from'
            elif 'to' in csv:
                direction = 'to'
            else:
                print('No directionality, check imports and code')

            # Don't need this anymore since non-aggregate time period is impossible
            # Get time period from csv import, using location
            tp_loc = csv.find('tp')
            # Location should be 2 up from start of tp
            tp = csv[tp_loc+2]

            # Import csv in question
            ph = pd.read_csv(os.getcwd() +
                             export_folders[8] + # This needs to by 9 for ME
                             '/' +
                             csv,
                             index_col = 0)

            # Wide to long
            ph = ph.unstack(
                    level=0).reset_index(
                            ).rename(columns={'level_0':'d_zone',
                            0:'dt'})
            print(sum(ph['dt']))        

            # Filter out zero values
            ph = ph[ph['dt']!=0].copy()

            # Input time period and purpose
            # ph['time'] = tp
            ph['purpose'] = purpose
            ph = ph.groupby(['o_zone',
                             'd_zone',
                             'purpose']).sum().reset_index()

            # Just got to face it, these are coming back with aggregate purpose and time period

            # If 'to' flip direction
            # How do I account for off purposes?
            if flip == True:
                if direction == 'to':
                    # Back to PA
                    ph = ph.merge(time_p_s_adjusted,
                                  how='left',
                                  left_on='purpose',
                                  right_on='agg_purpose_to_home').drop(
                                          ['agg_purpose_to_home',
                                           'purpose'], axis=1).rename(
                                           columns={'agg_purpose_from_home':'purpose'}).reset_index(
                                                   drop=True)
                    ph['dt'] = ph['dt'] * ph['direction_factor']
                    ph = ph.drop('direction_factor', axis=1)
                    ph = ph.groupby(['o_zone',    
                                     'd_zone',
                                     'purpose']).sum().reset_index()
                    ph = ph.rename(columns={'o_zone':'nd_zone',
                                            'd_zone':'o_zone'})
                    ph = ph.rename(columns={'nd_zone':'d_zone'}).copy()

            od_purpose_bin.append(ph)
        od_purpose = pd.concat(od_purpose_bin)
        me_bin.append(me_purpose)
        od_bin.append(od_purpose)

    # Add long from to long to
    print('Import and conversion done')

    # Get equivalent production totals
    me_total = pd.concat(me_bin).reindex(
            ['o_zone', 'purpose', 'dt'],
            axis=1).groupby(['o_zone', 'purpose']).sum(
                    ).reset_index()
    me_total['o_zone'] = me_total['o_zone'].apply(int)

    od_total = pd.concat(od_bin).reindex(
            ['o_zone', 'purpose', 'dt'],
            axis=1).groupby(['o_zone', 'purpose']).sum(
                    ).reset_index()
    od_total['o_zone'] = od_total['o_zone'].apply(int)

    # TODO: Maybe write these?
    me_audit = me_total.reindex(
            ['purpose', 'dt'],
            axis=1).groupby('purpose').sum().reset_index()

    od_audit = od_total.reindex(
            ['purpose', 'dt'],
            axis=1).groupby('purpose').sum().reset_index()

    print(me_audit)
    print(od_audit)

    # Define OD production splits at district level
    print('me len ' + str(len(me_total)))
    print('od len ' + str(len(od_total)))
    print('me prod ' + str(me_total['dt'].sum()))
    print('od prod ' + str(od_total['dt'].sum()))

    zone_split_col = ('overlap_' + _model_name.lower() + '_split_factor')

    me_district = me_total.merge(zone_to_district,
                                 how='left',
                                 left_on='o_zone',
                                 right_on='noham_zone_id').drop(
                                         'noham_zone_id', axis=1).copy()
    me_district['dt'] = me_district['dt']*me_district['overlap_noham_split_factor']

    # Factor to get original figures
    adj_factor = me_total['dt'].sum()/me_district['dt'].sum()
    me_district['dt'] = me_district['dt']*adj_factor
    
    # Group by district
    me_district = me_district.reindex(
            ['lad_zone_id',
             'purpose',
             'dt'], axis=1).groupby(['lad_zone_id',
             'purpose']).sum().reset_index().copy()
    print('me district prod ' + str(me_district['dt'].sum()))

    # Same for OD
    od_district = od_total.merge(zone_to_district,
                                 how='left',
                                 left_on='o_zone',
                                 right_on='noham_zone_id').drop(
                                         'noham_zone_id', axis=1).copy()
    od_district['dt'] = od_district['dt']*od_district['overlap_noham_split_factor']

    # Factor to get original figures
    adj_factor = od_total['dt'].sum()/od_district['dt'].sum()
    od_district['dt'] = od_district['dt']*adj_factor

    # Group by district
    od_district = od_district.reindex(
            ['lad_zone_id',
             'purpose',
             'dt'], axis=1).groupby(['lad_zone_id',
             'purpose']).sum().reset_index().copy()

    print('od district prod ' + str(od_district['dt'].sum()))

    # Combine matrices to get k factors
    me_district = me_district.rename(columns={'dt':'me'})
    od_district = od_district.rename(columns={'dt':'od'})

    k_district = od_district.merge(me_district,
                                   how = 'left',
                                   on = ['lad_zone_id',
                                         'purpose'])

    # Build k factor
    k_district['k_factor'] = k_district['me']/k_district['od']
    # Round >+10% down to +10%.
    # TODO: Needs a similar one for -10% when it comes to it
    k_district['k_factor_10'] = k_district['k_factor'].mask(
            k_district['k_factor'] > 1.10, other=1.10)
    # Round less than up to 1, temporary
    k_district['k_factor_10'] = k_district['k_factor_10'].mask(
            k_district['k_factor_10'] < 1, other=1)
    # Get lost growth from k factors
    k_district['lost_k'] = k_district['k_factor']-k_district['k_factor_10']

    # get total adjustment factor
    tadjf = (k_district['me'].sum()/k_district['od'].sum())-1
    
    # Apply k10 to od demand for benchmark
    k_district['me_10'] = k_district['od'] * k_district['k_factor_10']

    print('final me district prod (used for factors)' + str(k_district['me'].sum()))
    print('final od district prod (used for factors)' + str(k_district['od'].sum()))
    print('controlled me applied to od (new productions)' + str(k_district['me_10'].sum()))
    
    # Return k district and total adjustment factor
    # TODO: Needs to be homed in model zone lookups
    k_district.to_csv('k_factors_iter1.csv', index=False)   
