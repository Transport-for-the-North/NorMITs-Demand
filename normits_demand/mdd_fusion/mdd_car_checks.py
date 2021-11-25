# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 15:32:36 2021

@author: WoodfiP

TODO:
    For both mdd car and NoHAM OD car matrices
    Zone trip ends
    LAD-LAD sector matrix
    LAD trip ends
    Screenline sector-sector matrix
    Screenline sector trip ends
    TLDs
    
    All run for cars by purpose and time period
"""

import pandas as pd
import numpy as np
import pickle as pk
import matplotlib.pyplot as plt

#Locations
mdd_car_loc = 'Y:/Mobile Data/Processing/dct_MDDCar.pkl'
noham_loc = 'Y:/Mobile Data/Processing'
lad_loc = 'Y:/Mobile Data/Processing/MDD_Check/lookups/lads20_to_noham_correspondence.csv'


#Read in MDD
with open(r'Y:/Mobile Data/Processing/dct_MDDCar.pkl', 'rb') as log:
        dct_mdd_car = pk.load(log)
#[md][wd][pp][hr]
#[3][1][1-5][1-3]
        
print(dct_mdd_car.keys())
print(dct_mdd_car[3].keys())
print(dct_mdd_car[3][1].keys())
print(dct_mdd_car[3][1][1].keys())
print(dct_mdd_car[3][1][1][1])


#Read in NoHAM
with open(r'Y:/Mobile Data/Processing/dctNoHAM_uc.pkl', 'rb') as log:
        dct_noham_car = pk.load(log)
        
print(dct_noham_car.keys())
print(dct_noham_car[3].keys())
print(dct_noham_car[3][1].keys())
print(dct_noham_car[3][1][1].keys())
print(dct_noham_car[3][1][1][1])
np.shape(dct_noham_car[3][1][2][3])
print((dct_noham_car[3][1][2][3]))

#Read in NoHAM-LAD lookup
lad_cor = pd.read_csv(lad_loc, 
                      names=['lad', 'zone', 'lads_f', 'zone_f', 'internal'],
                      skiprows=1)
lad_cor = lad_cor.drop('lads_f', axis=1)

#zone list
unq_zones = list(range(1, 2771))

#Dictionary layers
dctmode = {3: ['Car']}
dctday = {1: ['Weekday']}
dctpurp = {1: ['hbw_fr'], 2: ['hbw_to'], 3: ['hbo_fr'], 4: ['hbo_to'], 5: ['nhb']}
dcttp = {1: ['AM'], 2: ['IP'], 3: ['PM']}

lad_list = []
sc_list = []


for md in dctmode:
        for wd in dctday:
            for pp in dctpurp:
                for tp in dcttp:
                    #Convert to pandas
                    wide_mdd = pd.DataFrame(dct_mdd_car[md][wd][pp][tp], 
                                           index = unq_zones, 
                                           columns = unq_zones).reset_index()
                    long_mdd = pd.melt(wide_mdd, 
                                      id_vars = ['index'], 
                                      var_name = 'd_zone', 
                                      value_name = 'mddtrip',
                                      col_level = 0)
                    long_mdd = long_mdd.rename(columns = {'index':'o_zone'})
                    
                    wide_nm = pd.DataFrame(dct_noham_car[md][wd][pp][tp], 
                                           index = unq_zones, 
                                           columns = unq_zones).reset_index()
                    long_nm = pd.melt(wide_nm, 
                                      id_vars = ['index'], 
                                      var_name = 'd_zone', 
                                      value_name = 'nmtrip',
                                      col_level = 0)
                    long_nm = long_nm.rename(columns = {'index':'o_zone'})
                    
                    #Join MDD and NoHAM
                    mat = pd.merge(long_mdd,
                                   long_nm,
                                   on = ['o_zone', 'd_zone'])
                    
                    ######
                    #Zone tripends
                    o_trips = mat.groupby(['o_zone']).agg({'mddtrip':sum, 'nmtrip':sum}).reset_index()
                    o_trips = o_trips.rename(columns = {'mddtrip':'o_mdd', 'nmtrip':'o_nm'})
                    d_trips = mat.groupby(['d_zone']).agg({'mddtrip':sum, 'nmtrip':sum}).reset_index()
                    d_trips = d_trips.rename(columns = {'mddtrip':'d_mdd', 'nmtrip':'d_nm'})
                    #Join
                    zone_te = pd.merge(o_trips,
                                       d_trips,
                                       left_on = ['o_zone'],
                                       right_on = ['d_zone'])
                    zone_te = zone_te[['o_zone',
                                       'o_mdd',
                                       'd_mdd',
                                       'o_nm',
                                       'd_nm']]
                    zone_te = zone_te.rename(columns = {'o_zone':'zone'})
                    #Export as csv
                    zone_te.to_csv(f'Y:/Mobile Data/Processing/MDD_Check/zone_te_od_p{pp}_m{md}_tp{tp}.csv', 
                                   index = False)
                    
                    #add linear regression
                    d = np.polyfit(mat['nmtrip'], mat['mddtrip'], 1)
                    f = np.poly1d(d)
                    mat['linearfit'] = f(mat['nmtrip'])
                    #plot and export
                    ax = mat.plot.scatter(y='mddtrip', x='nmtrip')
                    mat.plot(x='nmtrip', y='linearfit', color='Red', ax=ax)
                    ax.figure.savefig(f'Y:/Mobile Data/Processing/MDD_Check/zone_od_p{pp}_m{md}_tp{tp}.png')
                    
                    ######
                    #LAD sectors, origins then dests for many-many relationsip
                    mat_lad = pd.merge(mat,
                                       lad_cor,
                                       left_on = ['o_zone'],
                                       right_on = ['zone'])
                    mat_lad['mdd_lad'] = mat_lad['mddtrip'] * mat_lad['zone_f']
                    mat_lad['nm_lad'] = mat_lad['nmtrip'] * mat_lad['zone_f']
                    mat_lad = mat_lad.groupby(['lad', 'd_zone']).agg({'mdd_lad':sum, 'nm_lad':sum}).reset_index()
                    mat_lad = mat_lad.rename(columns = {'lad':'o_lad', 'internal':'o_int'})
                    
                    mat_lad = pd.merge(mat_lad,
                                       lad_cor,
                                       left_on = ['d_zone'],
                                       right_on = ['zone'])
                    mat_lad['mdd_lad'] = mat_lad['mdd_lad'] * mat_lad['zone_f']
                    mat_lad['nm_lad'] = mat_lad['nm_lad'] * mat_lad['zone_f']
                    mat_lad = mat_lad.groupby(['o_lad', 'lad']).agg({'mdd_lad':sum, 'nm_lad':sum}).reset_index()
                    mat_lad = mat_lad.rename(columns = {'lad':'d_lad', 'internal':'d_int'})
                    
                    #LAD tripends
                    o_trips = mat_lad.groupby(['o_lad', 'o_int']).agg({'mdd_lad':sum, 'nm_lad':sum}).reset_index()
                    o_trips = o_trips.rename(columns = {'mdd_lad':'o_mdd', 'nm_lad':'o_nm'})
                    d_trips = mat_lad.groupby(['d_lad', 'd_int']).agg({'mdd_lad':sum, 'nm_lad':sum}).reset_index()
                    d_trips = d_trips.rename(columns = {'mdd_lad':'d_mdd', 'nm_lad':'d_nm'})
                    #Join
                    lad_te = pd.merge(o_trips,
                                       d_trips,
                                       left_on = ['o_lad'],
                                       right_on = ['d_lad'])
                    lad_te = lad_te[['o_lad',
                                     'o_int'
                                     'o_mdd',
                                     'd_mdd',
                                     'o_nm',
                                     'd_nm']]
                    lad_te = lad_te.rename(columns = {'o_lad':'lad', 'o_int':'internal'})
                    #Export as csv
                    lad_te.to_csv(f'Y:/Mobile Data/Processing/MDD_Check/lad_te_od_p{pp}_m{md}_tp{tp}.csv', 
                                   index = False)
                    
                    #add linear regression
                    d = np.polyfit(mat_lad['nm_lad'], mat_lad['mdd_lad'], 1)
                    f = np.poly1d(d)
                    mat_lad['linearfit'] = f(mat_lad['nm_lad'])
                    #plot and export
                    ax = mat_lad.plot.scatter(y='mdd_lad', x='nm_lad')
                    mat_lad.plot(x='nm_lad', y='linearfit', color='Red', ax=ax)
                    ax.figure.savefig(f'Y:/Mobile Data/Processing/MDD_Check/lad_od_p{pp}_m{md}_tp{tp}.png')
                    #Close all plots
                    plt.close('all')
                    
                    #Repeat LAD, excluding External-External movements
                    mat_lad2 = mat_lad[(mat_lad['o_int'] == 1) | (mat_lad['d_int'] == 1)]
                    mat_lad2.drop('linearfit', axis=1, inplace=True)
                    #add linear regression
                    d = np.polyfit(mat_lad2['nm_lad'], mat_lad2['mdd_lad'], 1)
                    f = np.poly1d(d)
                    mat_lad2['linearfit'] = f(mat_lad2['nm_lad'])
                    #plot and export
                    ax = mat_lad2.plot.scatter(y='mdd_lad', x='nm_lad')
                    mat_lad2.plot(x='nm_lad', y='linearfit', color='Red', ax=ax)
                    ax.figure.savefig(f'Y:/Mobile Data/Processing/MDD_Check/lad_od_xEE_p{pp}_m{md}_tp{tp}.png')
                    plt.close('all')
                    
                    #Add LAD to master list
                    mat_lad['purp'] = pp
                    mat_lad['mode'] = md
                    mat_lad['tp'] = tp
                    lad_list.append(mat_lad)
                    
                    
                    
                    
#Concat and export LAD-LAD matrix
master_lad = pd.concat(lad_list)
master_lad.to_csv('Y:/Mobile Data/Processing/MDD_Check/lad-lad_matrices.csv', 
                                   index = False)




