# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 16:33:03 2019

@author: cruella
"""

import pandas as pd # most of the heavy lifting
import numpy as np # vector ops for speed
import os # File ops
import sys # File ops & sys config
sys.path.append('C:/Users/' + os.getlogin() + '/S/NorMITs Utilities/Python')
import normits_utils as nup # Folder management, reindexing, optimisation
sys.path.append('C:/Users/' + os.getlogin() + 
                '/S/NorMITs Travel Market Synthesiser/Python')
import production_model as pm # For NHB production model
import matrix_processing as mp # For OD Outputs
import distribution_model as dm

import geopandas as gpd

# File paths
_default_home_dir = 'Y:/NorMITs Synthesiser/Norms'
_default_iter = 'iter1'
_import_file_drive = "Y:/"
_import_folder = 'Y:/NorMITs Synthesiser/import/'
_default_model_write_folder = '/Model PA'

_default_model_folder = (_default_home_dir +
                         '/Model Zone Lookups/')

os.chdir(_default_model_folder)

# Get norms costs

old_norms_path = 'Y:/NorMITs Synthesiser/Old Norms/Model Zone Lookups'
new_norms_path = 'Y:/NorMITs Synthesiser/Norms/Model Zone Lookups'

pld_demand_path = 'Y:/NorMITs Synthesiser/import/PLD Matrices/PLD_Demand_24hr.csv'

pld_demand = pd.read_csv(pld_demand_path, names=['p_zone', 'a_zone', 'purpose', 'demand'])
pld_demand = pld_demand.reindex(
        ['p_zone',
         'a_zone',
         'demand'],
         axis=1).groupby(
                 ['p_zone', 'a_zone']).sum().reset_index()

costs = dm.get_distance_and_costs(old_norms_path,
                                  request_type='cost',
                                  seed_intrazonal = False)

distance_cols = ['origin', 'destination',
                 'business_ca_tr', 'business_ca_non_tr',
                 'business_nca_tr', 'business_nca_non_tr',
                 'commute_ca_tr', 'commute_ca_non_tr',
                 'commute_nca_tr', 'commute_nca_non_tr',
                 'other_ca_tr', 'other_ca_non_tr',
                 'other_nca_tr', 'other_nca_non_tr']

distances = pd.read_csv(old_norms_path + '/AAMAT00A.PRN', names=distance_cols)

# Get norms to old norms lookup

n2on = ('Y:/NorMITs Synthesiser/Zone Translation' +
        '/Export/norms_old_norms_pop_weighted_lookup.csv')

n2on = pd.read_csv(n2on)

# Get best fit
# This is the best version of this best fit expression atm.
n2on_bestfit = n2on.reindex(
        ['norms_zone_id',
         'old_norms_zone_id',
         'overlap_norms_split_factor'],
         axis=1).groupby(
                 ['norms_zone_id']).max(
                  level='overlap_norms_split_factor').reset_index()

# Sidepot cols that carry values
active_cols = list(costs)[2:]
active_cols = list(distances)[2:]
active_cols = list(pld_demand)[-1]

# Rename first one
costs = costs.rename(columns={'p_zone':'old_norms_zone_id'})
print(list(costs))

distances = distances.rename(columns={'origin':'old_norms_zone_id'})
print(list(distances))

pld_demand = pld_demand.rename(columns={'p_zone':'old_norms_zone_id'})

# Should go from len 1552516 to sq(1238) = 1534622
new_costs = costs.merge(n2on_bestfit,
                        how='inner',
                        on = 'old_norms_zone_id')
print(list(new_costs))

new_distances = distances.merge(n2on_bestfit,
                                how='inner',
                                on = 'old_norms_zone_id')
print(list(new_distances))

new_pld_demand = pld_demand.merge(n2on_bestfit,
                                how='inner',
                                on = 'old_norms_zone_id')
print(list(new_pld_demand))

######## Just copying this from here on

new_costs = new_costs.rename(columns={'norms_zone_id':'p_zone'})
print(list(new_costs))
del(new_costs['old_norms_zone_id'], new_costs['overlap_norms_split_factor'])
print(list(new_costs))

# Rename second one
new_costs = new_costs.rename(columns={'a_zone':'old_norms_zone_id'})
print(list(new_costs))

new_costs = new_costs.merge(n2on_bestfit,
                        how='inner',
                        on = 'old_norms_zone_id')
print(list(new_costs))

new_costs = new_costs.rename(columns={'norms_zone_id':'a_zone'})
print(list(new_costs))
del(new_costs['old_norms_zone_id'], new_costs['overlap_norms_split_factor'])
print(list(new_costs))

######### Distances version

new_distances = new_distances.rename(columns={'norms_zone_id':'p_zone'})
print(list(new_distances))
del(new_distances['old_norms_zone_id'], new_distances['overlap_norms_split_factor'])
print(list(new_distances))

# Rename second one
new_distances = new_distances.rename(columns={'destination':'old_norms_zone_id'})
print(list(new_distances))

new_distances = new_distances.merge(n2on_bestfit,
                                    how='inner',
                                    on = 'old_norms_zone_id')
print(list(new_distances))

new_distances = new_distances.rename(columns={'norms_zone_id':'a_zone'})
print(list(new_distances))
del(new_distances['old_norms_zone_id'], new_distances['overlap_norms_split_factor'])
print(list(new_distances))

####### PLD Version

new_pld_demand = new_pld_demand.rename(columns={'norms_zone_id':'p_zone'})
print(list(new_pld_demand))
del(new_pld_demand['old_norms_zone_id'], new_pld_demand['overlap_norms_split_factor'])
print(list(new_pld_demand))

# PLD Rename second one
new_pld_demand = new_pld_demand.rename(columns={'a_zone':'old_norms_zone_id'})
print(list(new_pld_demand))

new_pld_demand = new_pld_demand.merge(n2on_bestfit,
                                      how='inner',
                                      on = 'old_norms_zone_id')
print(list(new_pld_demand))

new_pld_demand = new_pld_demand.rename(columns={'norms_zone_id':'a_zone'})
print(list(new_pld_demand))
del(new_pld_demand['old_norms_zone_id'], new_pld_demand['overlap_norms_split_factor'])
print(list(new_pld_demand))


####

new_cost_cols = ['p_zone',
                 'a_zone',
                 'cost',
                 'business_ca_to',
                 'business_nca',
                 'commute_ca_from',
                 'commute_ca_to',
                 'commute_nca',
                 'other_ca_from',
                 'other_ca_to',
                 'other_nca']

new_costs = new_costs.reindex(new_cost_cols, axis=1).sort_values(by=['p_zone',
                 'a_zone']).reset_index(drop=True)

# Nod in costs where they're 0 for some reason

# TODO: later
####

new_distance_cols = ['p_zone', 'a_zone']
for col in active_cols:
    new_distance_cols.append(col)

new_distances = new_distances.reindex(
        new_distance_cols,
        axis=1).sort_values(
                by=['p_zone',
                    'a_zone']).reset_index(drop=True)  

##
    
# Build new sum matrix for distance

new_distances['business_ca'] = new_distances['business_ca_tr'] + new_distances['business_ca_non_tr']
new_distances['business_nca'] = new_distances['business_nca_tr'] + new_distances['business_nca_non_tr']
new_distances['commute_ca'] = new_distances['commute_ca_tr'] + new_distances['commute_ca_non_tr']
new_distances['commute_nca'] = new_distances['commute_nca_tr'] + new_distances['commute_nca_non_tr']
new_distances['other_ca'] = new_distances['other_ca_tr'] + new_distances['other_ca_non_tr']
new_distances['other_nca'] = new_distances['other_nca_tr'] + new_distances['other_nca_non_tr']

total_distances = new_distances.reindex(['p_zone', 'a_zone',
                                         'commute_ca', 'commute_nca',
                                         'business_ca', 'business_nca',
                                         'other_ca', 'other_nca'], axis=1)
# Fill in zero demand jobbies
# For cost
    
intra_zonals = new_costs[new_costs['p_zone']==new_costs['a_zone']].copy()
inter_zonals = new_costs[new_costs['p_zone']!=new_costs['a_zone']].copy()

inter_zonals = inter_zonals.sort_values(['p_zone', 'a_zone'])
inter_zonals[inter_zonals==0] = np.NaN
inter_zonals = inter_zonals.fillna(method='ffill')

new_costs = pd.concat([intra_zonals, inter_zonals]).sort_values(['p_zone', 'a_zone']).reset_index(drop=True)

for col in list(new_costs):
    test = new_costs[new_costs[col].isna()]
    print(len(test))
    test = new_costs[new_costs[col]==0]
    print(len(test))

# For distance

intra_zonals = total_distances[total_distances['p_zone']==total_distances['a_zone']].copy()
inter_zonals = total_distances[total_distances['p_zone']!=total_distances['a_zone']].copy()

inter_zonals = inter_zonals.sort_values(['p_zone', 'a_zone'])
inter_zonals[inter_zonals==0] = np.NaN
inter_zonals = inter_zonals.fillna(method='ffill')

total_distances = pd.concat([intra_zonals, inter_zonals]).sort_values(['p_zone', 'a_zone']).reset_index(drop=True)

for col in list(total_distances):
    test = total_distances[total_distances[col].isna()]
    print(len(test))
    test = total_distances[total_distances[col]==0]
    print(len(test))

# PLD Demand

new_pld_demand = new_pld_demand.reindex(['p_zone', 'a_zone', 'demand'], axis=1)
    
    
# Join on to all zone2zone - what's missing?

ia_areas = dm.define_internal_external_areas(paths[1])
internal_area = ia_areas[0]
ia_name = list(internal_area)[0]
external_area = ia_areas[1]

movements = dm.define_zone_movements(ia_name,
                                     internal_area,
                                     external_area,
                                     movement_type = 'all',
                                     labels = True)
    

new_norms_zones = 'Y:/NoRMS/Zoning/Norms/Norms_Zoning-2.11/norms_zoning_freeze_2.11.shp'
new_norms_zones = gpd.read_file(new_norms_zones)

nnz_txt = new_norms_zones.copy()
del(nnz_txt['geometry'])
del(nnz_txt['area_nor'])
del(nnz_txt['internal'])

nnz_txt = nnz_txt.rename(columns={'unique_id':'p_zone'})

zone_audit = nnz_txt.merge(new_costs,
                           how = 'outer',
                           on = 'p_zone').reindex(
                                   ['p_zone', 'a_zone'],
                                   axis=1).groupby(
                                           'p_zone').count(
                                                   ).reset_index()

no_moves = zone_audit[zone_audit['a_zone']==0]

zone_audit = nnz_txt.merge(new_distances,
                           how = 'outer',
                           on = 'p_zone').reindex(
                                   ['p_zone', 'a_zone'],
                                   axis=1).groupby(
                                           'p_zone').count(
                                                   ).reset_index()

no_moves = zone_audit[zone_audit['a_zone']==0]


# No moves len = 61. Total costs len should be (1300 - 61)^2
unq_zones_in_base = np.sqrt(len(new_costs))
confirmation = np.power(len(nnz_txt['p_zone'].drop_duplicates())-len(no_moves),2)==len(new_costs)

if confirmation:
    print('This all worked')
else:
    print('This did not work')

export_norms = 'norms_2018_am_costs.csv'

new_costs.to_csv(export_norms, index=False)

export_norms = 'norms_2018_am_distances.csv'

total_distances.to_csv(export_norms, index=False)

pld_name = 'norms_pld_demand.csv'
new_pld_demand.to_csv(pld_name, index=False)


