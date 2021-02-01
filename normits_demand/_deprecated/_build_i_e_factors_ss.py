# -*- coding: utf-8 -*-
"""
Created on Wed May 13 11:35:22 2020

@author: genie
"""

# TODO: Functionalise
# TODO: Add to one of the other modules

import pandas as pd

_default_ie = 'Y:/NorMITs Synthesiser/import/i_e_factors/generic_north_i_e_factors.csv'
_north_east_ie = 'Y:/NorMITs Synthesiser/import/i_e_factors/north_east_i_e_factors.csv'
_teesside_ie = 'Y:/NorMITs Synthesiser/import/i_e_factors/teesside_i_e_factors.csv'

# TODO: Vary by edge zones, somehow

def build_i_e_factors(file_drive,
                      model_name,
                      iteration,
                      ie_factors = _default_ie,
                      write_path = None,
                      write = False):

    """
    ie_factors ['north', one for each region]
    """

    # TODO: Flexible internal areas

    # Set default write out path
    report = (file_drive + 'NorMITs Synthesiser/' +
              model_name + '/' + iteration +
              '/Production Outputs/Production Run Logs/' + 
              'all_segment_report_hb.csv')

    if write_path == None:
        write_path = (file_drive + 'NorMITs Synthesiser/' +
                      model_name +
                      '/Model Zone Lookups/' +
                      model_name.lower() + '_ie_factors.csv')

    report = pd.read_csv(report)
    ie = pd.read_csv(ie_factors)

    internal = report[report['internal']==1]
    external = report[report['internal']!=1]

    internal_ie = ie[ie['agg_orig']=='internal']
    external_ie = ie[ie['agg_orig']=='external']

    internal = internal.merge(internal_ie,
                              how='left',
                              on=['p', 'm'])

    internal['i_to_i'] = internal['trips']*internal['agg_d_internal']
    internal['i_to_e'] = internal['trips']*internal['agg_d_external']
    internal['e_to_e'] = 0
    internal['e_to_i'] = 0
    
    external = external.merge(external_ie,
                              how='left',
                              on=['p', 'm'])

    external['i_to_i'] = 0
    external['i_to_e'] = 0
    external['e_to_e'] = external['trips']*external['agg_d_external']
    external['e_to_i'] = external['trips']*external['agg_d_internal']

    int_ext = pd.concat([internal, external], sort=True)

    int_ext['total'] = (int_ext['i_to_i'] + int_ext['i_to_e'] +
           int_ext['e_to_i'] + int_ext['e_to_e'])
    
    move_cols = ['i_to_i', 'i_to_e', 'e_to_i', 'e_to_e']
    for col in move_cols:
        int_ext[col] = int_ext[col] / int_ext['total']

    final_cols = [(model_name.lower() + '_zone_id'),
                  'p', 'm', 'internal', 'i_to_i',
                  'i_to_e', 'e_to_i', 'e_to_e']

    int_ext = int_ext.reindex(final_cols, axis=1).reset_index(drop=True)

    int_ext.to_csv(write_path, index=False)
    
    return(int_ext)
