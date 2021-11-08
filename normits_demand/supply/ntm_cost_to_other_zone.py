# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 14:50:18 2021

@author: genie
"""

import os
import pandas as pd

import matplotlib.pyplot as plt

def cost_to_long(ntm_in,
                 echo=True):
    
    if echo:
        print('Translating NTM cost to long format')
    
    ntm_in = ntm_in.rename(columns={list(ntm_in)[0]:'ntm_zone_id'})
        
    # Pivot to long
    long_ntm = ntm_in.copy()
    long_ntm = long_ntm.melt(id_vars = 'ntm_zone_id',
                             var_name = 'to_zone',
                             value_name = 'distance')
    long_ntm = long_ntm.rename(columns={'ntm_zone_id':'from_zone'})
    
    long_ntm['from_zone'] = long_ntm['from_zone'].astype(int)
    long_ntm['to_zone'] = long_ntm['to_zone'].astype(int)
    
    return long_ntm


def aggregate_long_cost(long_ntm: pd.DataFrame,
                        zone_correspondence: pd.DataFrame,
                        from_zone_col: str = 'ntmv6_zone_id',
                        to_zone_col: str = 'noham_zone_id',
                        weighting_col: str = 'ntmv6_to_noham',
                        filter_max: bool = False,
                        max_threshold: int = 10000,
                        method: str ='weighted_mean',
                        cost_vector_name: str = 'distance'
                        ):
    """
    

    Parameters
    ----------
    long_ntm : pd.DataFrame
        DataFrame of cost data, long format, NTM zoning.
    zone_correspondence : pd.DataFrame
        NTM zoning to target DataFrame correspondence.
    from_zone_col : str, optional
        Name of from and to in long_ntm in correspondence table.
        The default is 'ntmv6_zone_id'.
    to_zone_col : str, optional
        Name of from and to in long_ntm in correspondence table.
        The default is 'noham_zone_id'.
    weighting_col : str, optional
        Correspondence column to weight by.
        The default is 'ntmv6_to_noham'.
    filter_max: bool, optional
        Filter out the max value in whole matrix, assumes a constant, high
        place holder value.
        The default is True
    method : str, optional
        The default is 'weighted_mean'.
    cost_vector_name: str, optional
        Name of the cost column.
        The default is 'distance'.
   

    Returns
    -------
    new_cost : TYPE
        DESCRIPTION.
    """
    
    # TODO: Doesn't seem to do the trick
    
    if filter_max:
        if max_threshold is None:
            max_threshold = long_ntm[cost_vector_name].max()
        print('Filtering out values above %d' % max_threshold)
        long_ntm = long_ntm[long_ntm[cost_vector_name]<max_threshold]
        
    if method == 'weighted_mean':
        # TODO: Should be in discrete function
        
        # Translate 'from' side of the matrix
        long_ntm = long_ntm.rename(columns={'from_zone':from_zone_col})
        
        long_ntm = long_ntm.merge(zone_correspondence,
                                  how='left',
                                  on=from_zone_col)
        
        # Recontrol weights to 1 where needed
        weight_corr = long_ntm.groupby(
            [from_zone_col, 'to_zone'])[weighting_col].sum().reset_index()
        weight_corr['corr'] = 1/weight_corr[weighting_col]
        weight_corr = weight_corr.drop(weighting_col, axis=1)
        
        long_ntm = long_ntm.merge(weight_corr,
                                  how='left',
                                  on=[from_zone_col, 'to_zone'])
        long_ntm[weighting_col] *= long_ntm['corr']
        long_ntm = long_ntm.drop('corr', axis=1)
        
        # Group and sum
        long_ntm[cost_vector_name] *= long_ntm[weighting_col]
        long_ntm = long_ntm.groupby(
            [to_zone_col, 'to_zone'])[cost_vector_name].sum().reset_index()
        
        long_ntm = long_ntm.rename(columns={to_zone_col:'from_zone'})
        
        # Translate 'to' side of the matrix
        long_ntm = long_ntm.rename(columns={'to_zone':from_zone_col})
        
        long_ntm = long_ntm.merge(zone_correspondence,
                                  how='left',
                                  on=from_zone_col)
        
        # Recontrol weights to 1 where needed
        weight_corr = long_ntm.groupby(
            [from_zone_col, 'from_zone'])[weighting_col].sum().reset_index()
        weight_corr['corr'] = 1/weight_corr[weighting_col]
        weight_corr = weight_corr.drop(weighting_col, axis=1)
        
        long_ntm = long_ntm.merge(weight_corr,
                                  how='left',
                                  on=[from_zone_col, 'from_zone'])
        long_ntm[weighting_col] *= long_ntm['corr']
        long_ntm = long_ntm.drop('corr', axis=1)
        
        # Group and sum
        long_ntm[cost_vector_name] *= long_ntm[weighting_col]
        long_ntm = long_ntm.groupby(
            [to_zone_col, 'from_zone'])[cost_vector_name].sum().reset_index()
        
        long_ntm = long_ntm.rename(columns={to_zone_col:'to_zone'})
    
    return long_ntm

def cost_to_wide(long_cost: pd.DataFrame,
                 initial_unq_zones: list,
                 cost_vector_name: str = 'distance',
                 placeholder_value: int = 1000000):
    """

    Parameters
    ----------
    long_cost : pd.DataFrame
        Dataframe of cost data, long format, NTM zoning
    initial_unq_zones : list
        Audited list of initial zones
    cost_vector_name: str
        Name of the cost vector, default is distance
    placeholder_value: int
        Value to default data to avoid dropping matrix size.
        Defaults to: 10000000

    Returns
    -------
    wide_ntm : pd.DataFrame
        Wide format matrix, with initial index numbers

    """
    
    # Build placeholder matrix
    unq_from = pd.DataFrame(initial_unq_zones)
    unq_from = unq_from.rename(columns={list(unq_from)[0]:'from_zone'})
    unq_from ['ph'] = 1
    unq_to = pd.DataFrame(initial_unq_zones)
    unq_to ['ph'] = 1
    unq_to = unq_to.rename(columns={list(unq_to)[0]:'to_zone'})
    ph_mat = unq_from.merge(unq_to,
                            how='left',
                            on='ph')
    ph_mat = ph_mat.drop('ph', axis=1)
    
    infill_mat = ph_mat.merge(long_cost,
                              how='outer',
                              on=['from_zone', 'to_zone'])
    infill_mat[cost_vector_name] = infill_mat[cost_vector_name].fillna(
        placeholder_value)
    
    wide_ntm = long_cost.pivot(index = 'from_zone',
                               columns = 'to_zone',
                               values = cost_vector_name)
    
    return wide_ntm

def main():
    """
    Returns
    -------
    None.

    """
    # Import NTM cost vectors
    cost_dir = r'I:\Data\NTM Rail'
    ntm_costs = os.listdir(cost_dir)
    target_costs = [x for x in ntm_costs if '.csv' in x]
    target_costs = [x for x in target_costs if 'Dist' in x]
    
    # Import ntm lookups
    ntm_msoa_path = 'I:/Data/Zone Translations/ntmv6_to_msoa_correspondence.csv'
    ntm_noham_path = 'I:/Data/Zone Translations/noham_to_ntmv6_correspondence.csv'
    
    ntm_msoa = pd.read_csv(ntm_msoa_path)
    ntm_noham = pd.read_csv(ntm_noham_path)
    
    unq_msoa = list(ntm_msoa['msoa_zone_id'].sort_values())
    unq_noham = list(range(ntm_noham['noham_zone_id'].astype(int).min(),
                           ntm_noham['noham_zone_id'].astype(int).max()+1))
    
    # Define unq zone lists
    unq_ntm_noham = list(ntm_noham['ntmv6_zone_id'].sort_values().drop_duplicates())
    
    # Import
    for tc in target_costs:
        print(tc)
        ntm_in = pd.read_csv(os.path.join(cost_dir, tc))
        
        # Index unq zones
        unq_zones = list(ntm_in[list(ntm_in)[0]].drop_duplicates())
        # Check length
        if len(unq_zones) != len(ntm_in):
            raise ValueError('Dropping unique zones shrinks matrix')
        
        # Check against lookup
        missing = list(set(unq_zones) - set(unq_ntm_noham))
        print('%d missing NTM zones' % len(missing))
        
        # To long
        long_ntm = cost_to_long(ntm_in)
        
        # Translate
        msoa_cost = aggregate_long_cost(long_ntm, ntm_msoa)
        noham_cost = aggregate_long_cost(long_ntm, ntm_noham) 
        
        # To wide
        msoa_cost = cost_to_wide(msoa_cost, unq_msoa)
        noham_cost = cost_to_wide(noham_cost, unq_noham)
        
        # Out
        msoa_cost.to_csv(os.path.join(cost_dir, 'msoa_' + tc.replace(' ', '_')))
        noham_cost.to_csv(os.path.join(cost_dir, 'noham_' + tc.replace(' ', '_')))
    
if __name__ == '__main__':
    main()