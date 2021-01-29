# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 19:34:03 2020

Script to build 24hr costs from a set of Noham 'Time' skims
Weighting based on CTripEnd time split parameters by purpose.

@author: cruella
"""

import os
import pandas as pd

_import_folder = 'Y:/NorMITs Synthesiser/import/'

nts_tours = (_import_folder + 'phi_factors/gb_tour_proportions.csv')

nts_tours = pd.read_csv(nts_tours)

noham_distance_folder = (_import_folder + 'cost7z')
noham_distance = os.listdir(noham_distance_folder)

commute_p = [1]
business_p = [2]
other_p = [3,4,5,6,7,8]

_24hr = True
_tp = False

def factor_out_tp(c,
                  tp_from,
                  tp_to):
    """
    c: vost vector from import
    tp_from: np mat of tp split for purpose
    tp_to: tp_from.T
    """
    
    # Time filter is just the tp -1 for numpy index
    time_filter = c['tp']-1
    print(time_filter)
    from_factor = tp_from[time_filter].sum()
    to_factor = tp_to[time_filter].sum()
            
    print('from factor: ' + str(from_factor))
    print('to factor: ' + str(to_factor))

    out_mat_from = c['dat'].copy()
    out_mat_from['dist'] = out_mat_from['dist'] * from_factor
    out_mat_from = out_mat_from.rename(
            columns = {'dist':'frh_dist'})
    out_mat_to = c['dat'].copy()
    out_mat_to['dist'] = out_mat_to['dist'] * to_factor

    out_mat_to = out_mat_to.rename(
            columns = {'o_zone':'dzone',
                       'd_zone':'ozone',
                       'dist':'toh_dist'})
    out_mat_to = out_mat_to.rename(
            columns = {'ozone':'o_zone',
                       'dzone':'d_zone'})

    out_mat = out_mat_from.merge(out_mat_to,
                                 on = ['o_zone','d_zone'])
    out_mat['dist'] = (out_mat['frh_dist'] + out_mat['toh_dist'])

    out_dict = {'tp':c['tp'],
                'cost':out_mat}

    return(out_dict)

# Main job
if __name__ == '__main__':

    cost_vector = []

    for mat in noham_distance:
        dat = pd.read_csv(noham_distance_folder + '/' + mat,
                          header = None,
                          names=['o_zone', 'd_zone', 'dist'])

        # Check time params
        if '_ts1_' in mat:
            tp = 1
        elif '_ts2_' in mat:
            tp = 2
        elif '_ts3_' in mat:
            tp = 3
        elif '_ts4_' in mat:
            tp = 4
    
        # check uc params and append
        if '_uc1' in mat:
            p = 'business'
        elif 'uc2' in mat:
            p = 'commute'
        elif 'uc3' in mat:
            p = 'other'

        out_dict = {'tp':tp,
                    'p':p,
                    'dat':dat}
    
        cost_vector.append(out_dict)

    # get unique purpose
    unq_p = nts_tours['p'].drop_duplicates().reset_index(drop=True)


    if _24hr == True:
        out_cost = []
        out_pa = []

        for purpose in unq_p:

            splits = nts_tours[nts_tours['p']==purpose]
            tp_from = splits.drop(['m','p','from_ts'], axis=1).values
            tp_from = tp_from/tp_from.sum()
            tp_to = tp_from.T

            # it's 1,2,3,4 either way
            for c in cost_vector:
                if purpose in commute_p:
                    if c['p'] == 'commute':
                        tp_c = factor_out_tp(c,
                                             tp_from,
                                             tp_to)
                        tp_c.update({'purpose':purpose})
                        out_cost.append(tp_c)
                elif purpose in business_p:
                    if c['p'] == 'business':
                        tp_c = factor_out_tp(c,
                                             tp_from,
                                             tp_to)
                        tp_c.update({'purpose':purpose})
                        out_cost.append(tp_c)
                elif purpose in other_p:
                    if c['p'] == 'other':
                        tp_c = factor_out_tp(c,
                                             tp_from,
                                             tp_to)
                        tp_c.update({'purpose':purpose})
                        out_cost.append(tp_c)

            # Build the aggregate cost
            ph = []
            for o in out_cost:
                if o['purpose'] == purpose:
                    ph.append(o['cost'])
            pa = ph[0].copy()
            # Add them all. This is a bit lazy.
            pa['dist'] = pa['dist'] + ph[1]['dist'] + ph[2]['dist'] + ph[3]['dist']
            pa = pa.drop(['frh_dist', 'toh_dist'], axis=1)
            # Divide by 2 (frh + toh was 2 factors)
            pa['dist'] = pa['dist']/2
            # Divide by 1000 to get km dist
            pa['dist'] = pa['dist']/1000
            pa = pa.rename(columns={'dist':('p' + str(purpose) + '_dist')})
            out_pa.append({'p':purpose, 'cost':pa})

            pa.to_csv('D:/p' + str(purpose) + 'dist.csv', index=False)

        ph = []
        # Final output loop
        for cost in out_pa:
            phdf = cost['cost']
            print(list(phdf))
            phdf[list(phdf)[-1]] = round(phdf[list(phdf)[-1]],3)
            ph.append(phdf)

        base = ph[0].copy()

        for x in ph[1:]:
            base = base.merge(x,
                              how = 'left',
                              on = ['o_zone','d_zone'])

        base = base.rename(columns = {'o_zone':'p_zone',
                                      'd_zone':'a_zone'})

        base.to_csv('D://noham_24hr_cost.csv',
                    index=False)

    elif _tp == True:
        out_cost = dat.reindex(['o_zone', 'd_zone'],
                               axis=1).copy()

        for c in cost_vector:
            name = (str(c['p']) + '_tp' + str(c['tp']))

            out_cost = out_cost.merge(c['dat'],
                                      how = 'left',
                                      on = ['o_zone', 'd_zone'])
            out_cost['dist'] = out_cost['dist']/1000
            
            out_cost = out_cost.rename(columns={'dist':name})
        
        out_cost = out_cost.rename(columns = {'o_zone':'p_zone',
                                              'd_zone':'a_zone'})
        out_cost.to_csv('D://noham_tp_cost.csv',
                        index=False)

    # Check kernel still works
    print('Done')
