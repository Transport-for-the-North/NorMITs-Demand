# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 20:11:14 2020

Some functions to pre-process pld matrices to PA for integration into
the external model

@author: genie
"""

import os

import pandas as pd

_default_pld_path = 'Y:/NorMITs Synthesiser/import/pld_od_matrices/'

def pld_od_to_pa(pld_path = _default_pld_path,
                 write=True):

    pld_dir = os.listdir(pld_path)
    
    write_name = 'external_pa_pld.csv'
    arrivals_name = 'e_to_i_arrivals.csv'

    pld_dir = [x for x in pld_dir if write_name not in x]
    pld_dir = [x for x in pld_dir if arrivals_name not in x]
    pld_dir = [x for x in pld_dir if '.csv' in x]

    pld_list = []

    for mat in pld_dir:
        print(mat)
        mat_dict = {}
        mat_dict.update({'file_name':mat})

        # Get direction from filename
        if 'FH' in mat:
            direction = 'from_home'
        elif 'TH' in mat:
            direction = 'to_home'
        else:
            direction = 'none'

        mat_dict.update({'direction':direction})

        # Get purpose from filename
        if 'Com' in mat:
            purpose = 'commute'
        elif 'EB' in mat:
            purpose = 'business'
        else:
            purpose = 'other'

        mat_dict.update({'purpose':purpose})

        # Get CA from filename
        if 'NCA' in mat:
            ca = 'nca'
        else:
            ca = 'ca'
        
        mat_dict.update({'ca':ca})

        # Import
        print(pld_path + mat)
        ph_mat = pd.read_csv(pld_path + mat)

        ph_mat['direction'] = mat_dict['direction']
        ph_mat['purpose'] = mat_dict['purpose']
        ph_mat['ca'] = mat_dict['ca']
        
        # Look at params - look at them!
        print(mat_dict)
        # Total trips
        print(mat + ' ' + str(ph_mat['trips'].sum()))

        pld_list.append(ph_mat)

    external_pld = pd.concat(pld_list)

    external_pld['trips'].sum()

    metrics = external_pld.reindex(['purpose',
                                    'direction',
                                    'ca',
                                    'trips'],
    axis=1).groupby(
            ['purpose',
             'direction', 'ca']).sum().reset_index()
    print(metrics)

    # from_home & none = pa
    target_directions = ['from_home', 'none']
    external_pa_pld = external_pld[
            external_pld['direction'].isin(target_directions)]

    # Half NCA
    external_pa_pld = half_nca(external_pa_pld)

    external_pa_pld = external_pa_pld.rename(columns={'o_zone':'p_zone',
                                                      'd_zone':'a_zone',
                                                      'trips':'dt'})

    external_pa_pld = external_pa_pld.reindex(['p_zone',
                                               'a_zone',
                                               'purpose',
                                               'ca',
                                               'dt'], axis=1).groupby(
    ['p_zone',
     'a_zone',
     'purpose',
     'ca']).sum().reset_index()

    if write:
        external_pa_pld.to_csv((pld_path + write_name), index=False)

    # pld pa by purpose
    pa_metrics = external_pa_pld.reindex(
            ['purpose', 'ca', 'dt'],
            axis=1).groupby(['purpose', 'ca']).sum().reset_index()
    pa_metrics['share'] = pa_metrics['dt']/(pa_metrics['dt'].sum())
    print(pa_metrics)

    print('Total PA trips')
    print(external_pa_pld['dt'].sum())
    
    # Get a zones & return
    a_totals = external_pa_pld.reindex(['a_zone',
                                        'dt'],
    axis=1).groupby('a_zone').sum().reset_index()

    if write:
        a_totals.to_csv((pld_path + arrivals_name), index=False)

    return(external_pa_pld, a_totals)


def half_nca(external_pa_pld):
    """
    Half NCA
    """
    # Check unique direction
    unq_dir = external_pa_pld['direction'].drop_duplicates()
    print(unq_dir)

    before = external_pa_pld['trips'].sum()

    # Build untouched subset
    intact = external_pa_pld[
            external_pa_pld['ca']=='ca'].copy()

    # Get subset for halving
    half = external_pa_pld[
            external_pa_pld['ca']=='nca'].copy()

    del(external_pa_pld)

    # Check all is NCA in half
    unq_ca = half['ca'].drop_duplicates()
    print(unq_ca)

    # Half
    half['trips'] = half['trips']/2

    # Reappend
    external_pa_pld = pd.concat([intact, half], sort=True)

    # Get after total
    after = external_pa_pld['trips'].sum()
    
    print('Before ' + str(before))
    print('After ' + str(after))
    
    return(external_pa_pld)

# Get i to e total and check



