# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 11:30:39 2020

Translate back OD to PA.

@author: genie
"""

import vehicle_occupancy as vo
import distribution_model as dm
import reports_audits as ra
import utils as nup

import os

import numpy as np
import pandas as pd

from normits_demand.utils import trip_length_distributions as tld_utils

_post_me_path = 'Y:/NorMITs Synthesiser/Noham/Matrix_Checks/Post ME'
_prior_path = 'Y:/NorMITs Synthesiser/Noham/iter7b/PCU Outputs/PCU OD Matrices'
_noham_internal_path = 'Y:/NorMITs Synthesiser/Noham/Model Zone Lookups/noham_internal_area.csv'
_noham_external_path = 'Y:/NorMITs Synthesiser/Noham/Model Zone Lookups/noham_external_area.csv'
_noham_sector_lookup = 'Y:/NorMITs Synthesiser/Noham/Model Zone Lookups/noham_sector_lookup.csv'
_noham_distance_path = 'Y:/NorMITs Synthesiser/Noham/Model Zone Lookups/'

def import_compiled_matrix_folder(path, target_format='.csv',
                                  input_format = 'wide'):
    folder = os.listdir(path)
    folder = [x for x in folder if target_format in x]
    
    output_list = []
    for file in folder:
        print('Importing ' + file)
        file_dict = {}
        # Define types
        if 'TS1' in file or 'tp1' in file:
            tp = 'am'
        elif 'TS2' in file or 'tp2' in file:
            tp = 'ip'
        elif 'TS3' in file or 'tp3' in file:
            tp = 'pm'
        elif 'TS4' in file or 'tp4' in file:
            tp = 'op'
        else:
            tp = 'NOTP'
        
        if 'commute' in file or 'Commute' in file:
            p = 'commute'
        elif 'business' in file or 'Business' in file:
            p = 'business'
        elif 'other' in file or 'Other' in file:
            p = 'other'
        else:
            p = 'NOP'

        if input_format == 'long':
            dat = pd.read_csv((path + '/' + file), header = None)
            dat = dat.rename(columns={0:'o_zone',1:'d_zone', 2:'dt'})
            dat = dat.pivot(index = 'o_zone', columns = 'd_zone', values = 'dt')
        elif input_format == 'wide':
            dat = pd.read_csv((path + '/' + file))
            dat.index = dat['o_zone']
            dat = dat.drop('o_zone', axis=1)

        # Append to file dict
        file_dict.update({'tp':tp})
        file_dict.update({'p':p})
        file_dict.update({'dat':dat})
        
        output_list.append(file_dict)
        del(tp, p, dat, file_dict)

    return(output_list)


def od_to_pa():
    pa = None
    return(pa)



def post_me_comparison(post_me_path = _post_me_path,
                       prior_path = _prior_path,
                       internal_path = _noham_internal_path,
                       external_path = _noham_external_path,
                       sector_lookup = _noham_sector_lookup,
                       distances = _noham_distance_path,
                       export_folder = 'D:/post_me/',
                       do_as_splits = False):

    internal_zones = pd.read_csv(internal_path).values    
    external_zones = pd.read_csv(external_path).values

    long_distances = dm.get_distance(distances,
                                journey_purpose=None,
                                direction=None,
                                seed_intrazonal = True)
    long_distances = long_distances.reindex(['p_zone', 'a_zone', 'distance'],
                                            axis=1).reset_index(drop=True)
    long_distances['p_zone'] = long_distances['p_zone'].astype(int)
    long_distances['a_zone'] = long_distances['a_zone'].astype(int)

    distances = long_distances.pivot(index='p_zone',
                                     columns='a_zone',
                                     values='distance')

    # format for numpy indexing
    internal_zones = internal_zones-1
    internal_zones = internal_zones.flatten()
    external_zones = external_zones-1
    external_zones= external_zones.flatten()

    # Build list to run with analytical sectors
    an_sec = pd.read_csv(sector_lookup)
    an_sec = an_sec.sort_values(['anlytl_sec', 'noham_zone_id'])
    
    as_indices = []
    as_names = []
    for sec in an_sec['anlytl_sec'].drop_duplicates():
        name = sec
        as_names.append(name)
        ind = an_sec[an_sec['anlytl_sec']==sec]['noham_zone_id'].values-1
        as_indices.append(ind)

    prior = import_compiled_matrix_folder(prior_path,
                                          target_format = '.csv',
                                          input_format = 'wide')
    post_me = import_compiled_matrix_folder(post_me_path,
                                            target_format='.csv',
                                            input_format = 'long')

    prior_summary = []
    prior_ie = []
    prior_sec = []
    prior_atl = []
    prior_tlb = []
    for dist in prior:
        dist['tp']
        summ_row = [dist['tp'], dist['p'], dist['dat'].values.sum()]
        prior_summary.append(summ_row)

        # Build i-i, i-e, e-i, e-e totals
        mat = dist['dat'].values
        i_e_splits = nup.n_matrix_split(mat,
                                        indices = [internal_zones, external_zones],
                                        index_names = ['i', 'e'])
        ie_row = {'tp':dist['tp'], 'p':dist['p']}
        for split in i_e_splits:
            ie_row.update({split['name']:split['dat'].sum()})

        atl = tld_utils.get_trip_length(distances, dist['dat'].values)
        atl_row = {'tp':dist['tp'], 'p':dist['p'], 'atl':atl}
        prior_atl.append(atl_row)
        
        long_dist = pd.DataFrame(dist['dat'].reset_index())
        long_dist = pd.melt(long_dist,
                              id_vars=['o_zone'],
                              var_name='a_zone',
                              value_name='dt',
                              col_level=0)
        long_dist = long_dist.rename(columns={'o_zone':'p_zone'})
        long_dist['p_zone'] = long_dist['p_zone'].astype(int)
        long_dist['a_zone'] = long_dist['a_zone'].astype(int)

        tlb = dm.build_distribution_bins(long_distances,
                                         long_dist)
        tlb_row = {'tp':dist['tp'], 'p':dist['p'], 'tlb':tlb}
        prior_tlb.append(tlb_row)

        if do_as_splits:
            as_splits = nup.n_matrix_split(mat,
                                           indices = as_indices,
                                           index_names = as_names)
            as_row = {'tp':dist['tp'], 'p':dist['p']}
            for split in as_splits:
                as_row.update({split['name']:split['dat'].sum()})
                prior_sec.append(as_row)

    post_summary = []
    post_ie = []
    post_sec = []
    post_atl = []
    post_tlb = []
    for dist in post_me:
        dist['tp']
        summ_row = [dist['tp'], dist['p'], dist['dat'].values.sum()]
        post_summary.append(summ_row)

        # Build i-i, i-e, e-i, e-e totals
        mat = dist['dat'].values
        i_e_splits = nup.n_matrix_split(mat,
                                        indices = [internal_zones, external_zones],
                                        index_names = ['i', 'e'])
        ie_row = {'tp':dist['tp'], 'p':dist['p']}

        for split in i_e_splits:
            ie_row.update({split['name']:split['dat'].sum()})
        post_ie.append(ie_row)

        atl = tld_utils.get_trip_length(distances, dist['dat'].values)
        atl_row = {'tp':dist['tp'], 'p':dist['p'], 'atl':atl}
        post_atl.append(atl_row)
        
        long_dist = pd.DataFrame(dist['dat'].reset_index())
        long_dist = pd.melt(long_dist,
                            id_vars=['o_zone'],
                            var_name='a_zone',
                            value_name='dt',
                            col_level=0)
        long_dist = long_dist.rename(columns={'o_zone':'p_zone'})

        tlb = dm.build_distribution_bins(long_distances,
                                         long_dist)
        tlb_row = {'tp':dist['tp'], 'p':dist['p'], 'tlb':tlb}
        post_tlb.append(tlb_row)

        if do_as_splits:
            as_splits = nup.n_matrix_split(mat,
                                           indices = as_indices,
                                           index_names = as_names)
            as_row = {'tp':dist['tp'], 'p':dist['p']}
            for split in as_splits:
                as_row.update({split['name']:split['dat'].sum()})
                post_sec.append(as_row)

    prior_summary = pd.DataFrame(prior_summary)
    prior_summary = prior_summary.rename(columns={0:'tp',
                                                  1:'p',
                                                  2:'prior'})
    prior_ie = pd.DataFrame(prior_ie)
    prior_ie = prior_ie.rename(columns={'i_to_i':'pr_i_to_i',
                                        'i_to_e':'pr_i_to_e',
                                        'e_to_i':'pr_e_to_i',
                                        'e_to_e':'pr_e_to_e'})

    prior_sec = pd.DataFrame(prior_sec)

    # Format and export prior
    prior_longs = []
    for index, row in prior_sec.iterrows():
        name = (row['tp'] + '_' + row['p'])
        dat = row.copy().drop(['tp', 'p'])
        dat = dat.reset_index()
        rows = dat['index'].str.split('_to_', expand=True)
        rows['dt'] = dat.iloc[:,-1]
        rows = rows.rename(columns={0:'o_zone',1:'d_zone'})
        rows = rows[rows['o_zone']!='0']
        rows['o_zone'] = rows['o_zone'].astype(int)
        rows = rows[rows['d_zone']!='0']
        rows['d_zone'] = rows['d_zone'].astype(int)
        
        # Square export
        square = rows.pivot(index='o_zone', columns='d_zone', values='dt')
        square.to_csv((export_folder + '/prior_' + name + '.csv'),
                      index = True)
        
        # Long export
        prior_longs.append({'tp':row['tp'],'p':row['p'],'dat':rows})
        rows.to_csv((export_folder + '/prior_' + name + '_long.csv'),
                    index = False)

        del(rows,square)

    post_summary = pd.DataFrame(post_summary)
    post_summary = post_summary.rename(columns={0:'tp',
                                                1:'p',
                                                2:'post'})
    post_ie = pd.DataFrame(post_ie)
    post_ie = post_ie.rename(columns={'i_to_i':'po_i_to_i',
                                      'i_to_e':'po_i_to_e',
                                      'e_to_i':'po_e_to_i',
                                      'e_to_e':'po_e_to_e'})
    
    post_sec = pd.DataFrame(post_sec)

    # Format and export post
    
    post_longs = []
    for index, row in post_sec.iterrows():
        name = (row['tp'] + '_' + row['p'])
        dat = row.copy().drop(['tp', 'p'])
        dat = dat.reset_index()
        rows = dat['index'].str.split('_to_', expand=True)
        rows['dt'] = dat.iloc[:,-1]
        rows = rows.rename(columns={0:'o_zone',1:'d_zone'})
        rows = rows[rows['o_zone']!='0']
        rows['o_zone'] = rows['o_zone'].astype(int)
        rows = rows[rows['d_zone']!='0']
        rows['d_zone'] = rows['d_zone'].astype(int)

        # Square export
        square = rows.pivot(index='o_zone', columns='d_zone', values='dt')
        square.to_csv((export_folder + '/post_' + name + '.csv'),index = True)

        # Long export
        post_longs.append({'tp':row['tp'],'p':row['p'],'dat':rows})
        rows.to_csv((export_folder + '/post_' + name + '_long.csv'),
                    index = False)

        del(rows,square)

    # Join sec mats in prior and post
    for prior_set in prior_longs:
        for post_set in post_longs:
            if (prior_set['tp'] == post_set['tp'] and
                prior_set['p'] == post_set['p']):
                name = (prior_set['tp'] + '_' + prior_set['p'])
                # Handle the demand here and get it over with
                prior_dat = prior_set['dat'].copy()
                prior_dat = prior_dat.rename(columns={'dt':'prior_dt'})
                
                post_dat = post_set['dat']
                post_dat = post_dat.rename(columns={'dt':'post_dt'})

                combined_dat = prior_dat.merge(post_dat,
                                        how = 'left',
                                        on = ['o_zone', 'd_zone'])
                # Handle the demand here and get it over with
                if prior_set['tp'] == 'am' or prior_set['tp'] == 'pm':
                    combined_dat['post_dt'] = combined_dat['post_dt'] * 3
                elif prior_set['tp'] == 'ip':
                    combined_dat['post_dt'] = combined_dat['post_dt'] * 6
    
                combined_dat.to_csv((export_folder + '/comparison_' +
                                     name + '.csv'), index=False)
                
    # Same as above but for trip length distributions
    for prior_set in prior_tlb:
        for post_set in post_tlb:
            if (prior_set['tp'] == post_set['tp'] and
                prior_set['p'] == post_set['p']):
                name = (prior_set['tp'] + '_' + prior_set['p'])

                print(name)

                prior_dat = prior_set['tlb'].copy()
                prior_dat = prior_dat.rename(columns={'dt':'prior_dt'})

                post_dat = post_set['tlb']
                post_dat = post_dat.rename(columns={'dt':'post_dt'})

                combined_dat = prior_dat.merge(post_dat,
                                        how = 'left',
                                        on = 'distance')
                # Handle the demand here and get it over with
                if prior_set['tp'] == 'am' or prior_set['tp'] == 'pm':
                    combined_dat['post_dt'] = combined_dat['post_dt'] * 3
                elif prior_set['tp'] == 'ip':
                    combined_dat['post_dt'] = combined_dat['post_dt'] * 6
                    
                combined_dat['difference'] = (combined_dat['post_dt']-
                            combined_dat['prior_dt'])

                combined_dat.to_csv((export_folder + '/tlb_comparison_' +
                                     name + '.csv'), index=False)

    comparison = prior_summary.merge(post_summary,
                                     how='left',
                                     on=['p','tp'])
    comparison['diff'] = comparison['post']-comparison['prior']
    
    ie_comparison = prior_ie.merge(post_ie,
                                   how='left',
                                   on=['p','tp'])

    # Compare atls
    prior_atls = pd.DataFrame(prior_atl)
    prior_atls = prior_atls.rename(columns={'atl':'pr_atl'})

    post_atls = pd.DataFrame(post_atl)
    post_atls = post_atls.rename(columns={'atl':'po_atl'})

    atl_comparison = prior_atls.merge(post_atls,
                                      how='left',
                                      on=['p','tp'])

    return(comparison, ie_comparison, atl_comparison)

def main(input_format = 'pcu',
         ):

    if input_mat == 'pcu':
        vo.people_vehicle_conversion(folder = _default_folder,
                                     import_folder = _import_folder,
                                     export_folder = None,
                                     mode = '3',
                                     method = 'to_people',
                                     write = True)

    # Get factors for splitting other purpose.


    # Get from/to proportions
    
    od_to_pa
    
    return()