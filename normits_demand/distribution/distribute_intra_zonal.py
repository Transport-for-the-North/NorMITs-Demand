# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 17:42:19 2019

Script to fake active mode distributions by allocating all productions to
intra-zonals.
Only to be used with walk and cycle and even then, only in a pinch.
That said, likely to have fairly good calibration due to the size of most
MSOA zones!

Will be fixed by better PT costs, when they arrive.

@author: scar
"""

# Standard imports
import os
import sys
import pandas as pd

sys.path.append('C:/Users/' +
                os.getlogin() +
                '/Documents/GitHub/Travel-Market-Synthesiser')
import distribution_model as dm

# Import folder for distances
m_folder = 'Y:/NorMITs Synthesiser/Norms_2015/Model Zone Lookups'

# Export folder
x_folder = 'Y:/NorMITs Synthesiser/Norms_2015/iter2/Distribution Outputs/24hr PA Distributions/'

# Get internal productions
i24p_path = 'Y:/NorMITs Synthesiser/Norms_2015/iter2/Production Outputs/internal_productions_norms_2015.csv'
internal_24hr_productions = pd.read_csv(i24p_path)

distribution_segments = ['purpose', 'mode', 'car_availability']

group_segments = []
group_segments.append(list(internal_24hr_productions)[0])
for seg in distribution_segments:
    group_segments.append(seg)

index_segments = group_segments.copy()
index_segments.append('trips')

# Get unq dataframe of mode and purpose, for modes 1,2 & 5.
unq_mp = internal_24hr_productions.reindex(
        distribution_segments,
        axis=1).drop_duplicates().reset_index(drop=True)


ip = internal_24hr_productions.reindex(
        index_segments,
        axis=1).groupby(group_segments).sum(
                ).reset_index()

#### HARD FILTERS ON UNQ_MP
unq_mp = unq_mp[unq_mp['mode']==5]
unq_mp = unq_mp[unq_mp['purpose'].isin([5,6,7,8])]
unq_mp = unq_mp[unq_mp['car_availability']==2]

# Filter down to 1,2 modes only
wanted_modes = [5]

ip = ip[ip['mode'].isin(wanted_modes)].reset_index(drop=True)
unq_mp = unq_mp[unq_mp['mode'].isin(wanted_modes)].reset_index(drop=True)

# Iterate over, allocate all trips to intra-zonals.
for index, row in unq_mp.iterrows():

    # If you see a 3,5 or 6 hit the eject button!
    print(row['mode'])

    # Build calib dict
    calib_params = {}
    for ds in distribution_segments:
        calib_params.update({ds:row[ds]})
    print(calib_params)

    # Safe var config for rows
    if row['car_availability'] == 1:
        c_a = False
    elif row['car_availability'] == 2:
        c_a = True
    else:
        c_a = None

    # Maybe get rid
    if row['purpose'] == 1:
        j_p = 'commute'
    elif row['purpose'] in [2,12]:
        j_p = 'business'
    elif row['purpose'] in [3,4,5,6,7,8,13,14,15,16,18]:
        j_p = 'other'

    distance = dm.get_distance_and_costs(m_folder,
                                         request_type='distance',
                                         journey_purpose=j_p,
                                         direction=None,
                                         car_available=c_a,
                                         seed_intrazonal=True)

    print(list(distance))
    print('Total distance ' + str(distance['distance'].sum()))

    intra = distance[distance['p_zone'] == distance['a_zone']].copy().reset_index(drop=True)

    # Filter down
    ph = ip[ip['purpose']==row['purpose']]
    ph = ph[ph['mode']==row['mode']]
    ph = ph[ph['car_availability'] == row['car_availability']]

    # Join
    ph = intra.merge(ph,
                     how='inner',
                     left_on = 'p_zone',
                     right_on = 'norms_2015_zone_id').reset_index(drop=True)

    # get rid of zone id
    del(ph['norms_2015_zone_id'])

    # Rename trips 'dt' for use in other things
    ph = ph.rename(columns={'trips':'dt'})

    reindex_cols = ['p_zone', 'a_zone']
    for ric in distribution_segments:
        if ric != 'car_availability':
            reindex_cols.append(ric)
        else:
            print('nope')

    reindex_cols.append('dt')

    ph = ph.reindex(reindex_cols, axis=1).reset_index(drop=True)

    dist_path = (x_folder +
                 'hb_synthetic')
    for index, cp in calib_params.items():
        cp_ph = ('_' + index + '_' + str(int(cp)))
        dist_path += cp_ph
    dist_path += '.csv'

    print(dist_path)

    ph.to_csv(dist_path, index=False)
