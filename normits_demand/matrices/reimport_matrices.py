# -*- coding: utf-8 -*-
"""
Created on Thu May 21 12:51:07 2020

@author: genie
"""

# Script to calculate % intra zonal as POC
# Adapted from general read in functions

# TODO: Generalise to an all purpose read in function

import os
import pandas as pd
import numpy as np

_target_folder = 'Y:/NorMITs Synthesiser/Noham/iter7b/Distribution Outputs/24hr PA Distributions'

directory = os.listdir(_target_folder)
directory = [x for x in directory if '.csv' in x]

target_cols = ['name', 'origin', 'purpose', 'mode', 'intra', 'intra_perc', 'total']
group_cols = target_cols.copy()
rem_cols = ['total']
for rem in rem_cols:
    group_cols.remove(rem)

m_ph = []
for mat in directory:
    # Define params for compilation
    # Mode
    if '_m1' in mat:
        mode = 1
    elif '_m2' in mat:
        mode = 2
    elif '_m3' in mat:
        mode = 3
    elif '_m4' in mat:
        mode = 4
    elif '_m5' in mat:
        mode = 5
    elif '_m6' in mat:
        mode = 6
    else:
        mode = ''
    
    # Origin
    if 'nhb' in mat:
        origin = 'nhb'
    else:
        origin = 'hb'
    # Purpose
    if '_p1_' in mat:
        purpose = 'commute'
    elif '_p2' in mat or '_p12' in mat:
        purpose = 'business'
    else:
        purpose = 'other'

    # ca
    # if 'nca' in mat:
    #     ca = 'nca'
    # else:
    #     ca = 'ca'
    
    # Direction
    if 'from' in mat:
        direction= 'from'
    elif 'to' in mat:
        direction = 'to'
    else:
        direction = ''
    
    # tp
    if 'tp1' in mat:
        tp = 'tp1'
    elif 'tp2' in mat:
        tp = 'tp2'
    elif 'tp3' in mat:
        tp = 'tp3'
    elif 'tp4' in mat:
        tp = 'tp4'
    else:
        tp = ''

    print('Importing: ' + mat)
    print('Which is:')
    print(origin)
    print(purpose)
    # print(ca)
    print(tp)
    print('I reckon, if that looks wrong please stop and tweak the import params')
    
    ph_mat = pd.read_csv(_target_folder + '/' + mat)
    print('Dropping first col, named:')
    print(list(ph_mat)[0])
    ph_mat = ph_mat.drop(list(ph_mat)[0], axis=1)
    ph_mat = ph_mat.values

    total = ph_mat.sum()
    intra = np.diagonal(ph_mat).sum()
    if intra == 0:
        intra = intra + 0.00001
    intra_perc = intra/total

    print(total)

    total_dict = {'name':mat,
                  'mode':mode,
                  'total':total,
                  'intra':intra,
                  'intra_perc':intra_perc,
                  'origin':origin,
                  'direction':direction,
                  'purpose':purpose,
                  'tp':tp}

    m_ph.append(total_dict)

output_mat = pd.DataFrame(m_ph)

output_mat = output_mat.reindex(
        target_cols, axis=1).groupby(
                group_cols).sum().reset_index()

output_mat.to_csv('D:/noham_intra_audit_i7b.csv', index=False)