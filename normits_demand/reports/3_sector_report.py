# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 11:34:00 2020

@author: genie
"""

import os

import pandas as pd
import numpy as np

import utils as nup

lookups = 'Y:/NorMITs Synthesiser/Noham/Model Zone Lookups'
od_vehicles = 'Y:/NorMITs Synthesiser/Noham/iter8c/PCU Outputs/PCU OD Matrices'
export = 'Y:/NorMITs Synthesiser/Noham/iter8c/PCU Outputs/Logs & Reports'

pcu_mats = [x for x in os.listdir(od_vehicles) if 'od' in x]

sector_lookup = pd.read_csv(os.path.join(lookups, '3_sector_lookup.csv'))

north_sectors = sector_lookup[sector_lookup['sector'] == 'north']['unique_id'].values-1
scotland_sectors = sector_lookup[sector_lookup['sector'] == 'scotland']['unique_id'].values-1
south_sectors = sector_lookup[sector_lookup['sector'] == 'south']['unique_id'].values-1

import_format = 'long'
header = None

for mat in pcu_mats:
    print(mat)
    dat = pd.read_csv(os.path.join(od_vehicles,
                                   mat), header=header)
    dat_cols = list(dat)
    
    # TODO: test input format on list length

    if import_format == 'long':
        # Make it wide again
        
        dat = dat.pivot(index=dat_cols[0], columns=dat_cols[1], values=dat_cols[2]).values
        
        audit_in = dat.sum()
        
        report = nup.n_matrix_split(dat,
                                    indices=[north_sectors, scotland_sectors, south_sectors],
                                    index_names=['1', '2', '3'],
                                    summarise=True)

    elif import_format == 'wide':
        dat = dat.drop(list(dat)[0]).values

        report = nup.n_matrix_split(dat,
                                    indices=[north_sectors,
                                             scotland_sectors,
                                             south_sectors],
                                    index_names=['1', '2', '3'],
                                    summarise=True)

    row_frame = pd.DataFrame.from_dict(report)
    
    audit_out = row_frame['dat'].sum()

    if round(audit_in, 3) == round(audit_out, 3):
        print('Audit in same as audit out')
    else:
        raise Warning('Report total different from in values')

    cols = row_frame['name'].str.split('_to_', expand=True)
    cols = cols.rename(columns={0: 'from',
                                1: 'to'})
    row_frame['from'] = cols['from']
    row_frame['to'] = cols['to']
    row_frame=row_frame.drop('name', axis=1).reindex(['from', 'to', 'dat'], axis=1)

    """
    row_frame = row_frame.pivot(index='from', columns='to', values='dat')
    """

    out_name = mat.replace('od', '3_sector_report_od')

    row_frame.to_csv(os.path.join(export, out_name), index=False)