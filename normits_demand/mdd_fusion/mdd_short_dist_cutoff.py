# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 15:32:36 2021

@author: WoodfiP

TODO:
    1. Automate sd cutoff
    Use NoHAM as proxy for NTS for now
    Produce TLD table per 5 car UCs (and plot?)
    Difference method, find min diff in 1km bands up to 10km
        Needs some check on consistency of difference
    2. Remove short distance trips in 2km bands and produce tripend plots with regression
    All done without E-E
    Can re-use a lot of MDD check
    
    All run for cars by purpose and time period
"""

import pandas as pd
import numpy as np
import pickle as pk
import matplotlib.pyplot as plt

# Locations
mdd_car_loc = 'Y:/Mobile Data/Processing/dct_MDDCar.pkl'
noham_loc = 'Y:/Mobile Data/Processing'
# lad_loc = 'Y:/Mobile Data/Processing/MDD_Check/lookups/lads20_to_noham_correspondence.csv'
noham_internal_loc = 'Y:/Mobile Data/Processing/MDD_Check/lookups/noham_internal.csv'
# sector_loc = 'Y:/Mobile Data/Processing/MDD_Check/lookups/sector_to_noham_correspondence.csv'
distance_loc = 'Y:/NoHAM/17.TUBA_Runs/-TPT/Skims/RefDist_Skims/NoHAM_Base_2018_TS2_v106_Dist_Other.csv'


# Read in MDD
with open(r'Y:/Mobile Data/Processing/dct_MDDCar.pkl', 'rb') as log:
        dct_mdd_car = pk.load(log)
# [md][wd][pp][hr]
# [3][1][1-5][1-3]

# Read in NoHAM
with open(r'Y:/Mobile Data/Processing/dctNoHAM_mddpurp.pkl', 'rb') as log:
        dct_noham_car = pk.load(log)
# [md][wd][pp][hr]
# [3][1][1-5][1-4]


zone_int = pd.read_csv(noham_internal_loc,
                       names=['zone', 'internal'],
                       skiprows=1)

distance = pd.read_csv(distance_loc,
                       names=['o_zone', 'd_zone', 'dist_km'])

# TLD distance band
tldDist = [0,1,2,3,4,5,6,7,8,9,10,12.5,15,17.5,20,25,30,35,40,50,75,100,150,200,250,300,400,600,999]
tldLabels = list(range(1,29))
tldBand = {'band': [], 'lower': [], 'upper': []}

for row in zip(range(1, len(tldDist)), tldDist[:-1], tldDist[1:]):
    tldBand['band'].append(row[0])
    tldBand['lower'].append(row[1])
    tldBand['upper'].append(row[2])

# Distance - add intrazonals (half minimum distance)
intra = distance.groupby(['o_zone'])['dist_km'].agg('min').reset_index()
intra['d_zone'] = intra['o_zone']
intra['dist_km'] = intra['dist_km'] / 2
intra = intra[['o_zone', 'd_zone', 'dist_km']]
distance = pd.concat([distance, intra], ignore_index=True)

# zone list
unq_zones = list(range(1, 2771))

# Dictionary layers
dctmode = {3: ['Car']}
dctday = {1: ['Weekday']}
dctpurp = {1: ['hbw_fr'], 2: ['hbw_to'], 3: ['hbo_fr'], 4: ['hbo_to'], 5: ['nhb']}
dcttp = {1: ['AM'], 2: ['IP'], 3: ['PM']}

cutoff_dict = {}

for md in dctmode:
    cutoff_dict[md] = {}
    for wd in dctday:
        cutoff_dict[md][wd] = {}
        for pp in dctpurp:
            cutoff_dict[md][wd][pp] = {}
            for tp in dcttp:
                # Convert to pandas
                wide_mdd = pd.DataFrame(dct_mdd_car[md][wd][pp][tp],
                                        index=unq_zones,
                                        columns=unq_zones).reset_index()
                long_mdd = pd.melt(wide_mdd,
                                   id_vars=['index'],
                                   var_name='d_zone',
                                   value_name='mddtrip',
                                   col_level=0)
                long_mdd = long_mdd.rename(columns={'index': 'o_zone'})

                wide_nm = pd.DataFrame(dct_noham_car[md][wd][pp][tp],
                                       index=unq_zones,
                                       columns=unq_zones).reset_index()
                long_nm = pd.melt(wide_nm,
                                  id_vars=['index'],
                                  var_name='d_zone',
                                  value_name='nmtrip',
                                  col_level=0)
                long_nm = long_nm.rename(columns={'index': 'o_zone'})

                # Join MDD and NoHAM
                mat = pd.merge(long_mdd,
                               long_nm,
                               on=['o_zone', 'd_zone'])

                ######
                # Trip length distribution
                # join distance
                mat = pd.merge(mat,
                               distance,
                               on=['o_zone', 'd_zone'])
                # join internal marker
                mat = pd.merge(mat,
                               zone_int,
                               left_on=['o_zone'],
                               right_on=['zone'])
                mat = mat.rename(columns={'internal': 'o_int'})
                mat = pd.merge(mat,
                               zone_int,
                               left_on=['d_zone'],
                               right_on=['zone'])
                mat = mat.rename(columns={'internal': 'd_int'})
                mat.drop(['zone_x', 'zone_y'], axis=1, inplace=True)
                # cut for distance bands and group
                mat['dband'] = pd.cut(mat['dist_km'], bins=tldDist, right=False, labels=tldLabels)
                distbands = mat.groupby(['dband', 'o_int', 'd_int']).agg(
                    {'mddtrip': sum, 'nmtrip': sum}).reset_index()

                # Remove external-external (very few trips at short distances)
                distbands1 = distbands[(distbands["o_int"] == 1) | (distbands["d_int"] == 1)]
                # Aggregate by dband
                distbands1 = distbands1.groupby(['dband']).agg(
                    {'mddtrip': sum, 'nmtrip': sum}).reset_index()
                # Calculate percentage shares by dband
                distbands1['mddprop'] = (distbands1['mddtrip'] / distbands1['mddtrip'].sum())
                distbands1['nmprop'] = (distbands1['nmtrip'] / distbands1['nmtrip'].sum())
                # Absolute differences in proportions
                distbands1['absdiff'] = abs(distbands1['mddprop'] - distbands1['nmprop'])
                distbands1['pctdiff'] = abs((distbands1['mddprop'] / distbands1['nmprop'] -1))

                # Checking plots
                ax = plt.gca()
                distbands1.plot(kind='line', x='dband', y='nmprop', ax=ax)
                distbands1.plot(kind='line', x='dband', y='mddprop', color='red', ax=ax)
                plt.show()
                plt.savefig(f'Y:/Mobile Data/Processing/Cutoff_check/cutoff_p{pp}_m{md}_tp{tp}.png')
                plt.close()

                # Plot, only for bands 1-10 i.e. up to 10km
                db2 = distbands1[(distbands1["dband"] <= 10)]
                # Proportion plot
                ax = plt.gca()
                db2.plot(kind='line', x='dband', y='nmprop', ax=ax)
                db2.plot(kind='line', x='dband', y='mddprop', color='red', ax=ax)
                plt.close()
                # Difference plot
                db2.plot(kind='bar', x='dband', y='absdiff')
                db2.plot(kind='bar', x='dband', y='pctdiff', color='red')
                plt.close('all')

                # Rank to find min percentage difference
                db2['rank'] = db2['pctdiff'].rank(ascending=True)
                min1 = db2['dband'][db2['rank'] == 1].item()
                min2 = db2['dband'][db2['rank'] == 2].item()
                min3 = db2['dband'][db2['rank'] == 3].item()
                dseries = [min1, min2, min3]
                dseries = sorted(dseries)
                ddiff = np.diff(dseries)
                # Assign cutoff value - want 2 of the 3 smallest differences to be consecutive
                if min(ddiff) == 1:
                    cutoff = min1
                else:
                    cutoff = 6

                # Add cutoff value to dictionary
                cutoff_dict[md][wd][pp][tp] = cutoff

# Save cutoff dictionary
with open('Y:/Mobile Data/Processing/dictcutoff_v1.pkl', 'wb') as log:
    pk.dump(cutoff_dict, log, pk.HIGHEST_PROTOCOL)
print("cutoffs packaged")

for k, v in cutoff_dict.items():
    print(k, v)


with open(r'Y:/Mobile Data/Processing/dictcutoff_v1.pkl', 'rb') as log:
    cutoff_dict = pk.load(log)


