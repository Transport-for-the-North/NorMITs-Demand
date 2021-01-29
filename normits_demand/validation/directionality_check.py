# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import os

_default_folder = ('Y:/NorMITs Synthesiser/Noham/' +
                   'iter6/Distribution Outputs/OD Matrices/')

_test = os.listdir(_default_folder)

# TODO: Directionality check w/ threshold
# TODO: Make numeric purpose work properly
# TODO: Get the zone number somewhere useful

def directionality_check(folder = _default_folder,
                         purpose_type = 'numeric',
                         subset_zones = [129],
                         purpose_subset = [4],
                         mode_subset = ['mode3']):

    """
    Folder = path
    purpose_type = 'numeric' or 'string'
    subset_zones = list
    purpose_subset = list
    """

    mats = os.listdir(folder)

    if mode_subset is not None:
        for m in mode_subset:
            mats = [x for x in mats if m in x]

    if purpose_type == 'string':
        purpose = ['commute', 'business', 'other']
    elif purpose_type == 'numeric':
        purpose = [1,2,3,4,5,6,7,8,12,13,14,15,16,18]
    
    if purpose_subset is not None:
        purpose = [x for x in purpose if x in purpose_subset]

    tp = ['tp1','tp2','tp3','tp4']
    direction = ['from', 'to']

    # O = row, D = col

    ph = []
    for z in subset_zones:
        print(z)
        for p in purpose:
            print(p)
            if purpose_type == 'string':
                p_sub = [x for x in mats if p in x]
            elif purpose_type == 'numeric':
                p_sub = [x for x in mats if ('purpose' + str(p)) in x or ('purpose_' + str(p)) in x]
            for t in tp:
                row = {}
                print(t)

                t_sub = [x for x in p_sub if t in x]
                
                for dc in direction:
                    print(dc)
                    d_sub = [x for x in t_sub if dc in x]
                    
                    # Read
                    mat = pd.read_csv(folder + '/' + d_sub[0])
                    
                    # Subset
                    o_sub = mat[mat['o_zone']==z]
                    o_sub = o_sub.drop('o_zone', axis=1)
                    
                    d_sub = mat[str(z)]
                    
                    # Get O 
                    o = o_sub.sum().sum()
                    
                    # Get D
                    d = d_sub.sum()
                    
                    # row update O, row update d
                    row.update({(str(p) + ' ' + t + ' ' + dc + ' o:'):o})
                    row.update({(str(p) + ' ' + t + ' ' + dc + ' d:'):d})

                    # Append to loop out
                    ph.append(row)

    df_ph = []
    for item in ph:
        df = pd.DataFrame(list(item.items()),
                          columns = ['mat',('zone_' +
                                            str(z) + '_demand')])
        df_ph.append(df)

    out = pd.concat(df_ph)
    
    # Process a little bit to give the right output
    out['dir'] = out['mat'].str[-2:-1]
    out['dist'] = out['mat'].str[0:-3]
    
    out = out.reindex(['dist','dir',('zone_' +
                                     str(z) + '_demand')],
    axis=1).reset_index(drop=True)

    # TODO: Why is it giving me them twice!
    out = out.drop_duplicates()
    
    return(out)