# -*- coding: utf-8 -*-
"""
Build a distance matrix from Norms input data

future versions should use distance over network using station choice model
weighted by demand preferences at various times of the day for various modes

"""

import os
import math

import pandas as pd
import numpy as np

# Get Norms Network Input paths
norms_network = r'R:\05 Other Analysis\Model_Inputs\2018 Latest Inputs\Inputs\Network'
norms_files = os.listdir(norms_network)


def df_to_np(df,
             values,
             unq_internal_zones,
             v_heading,
             h_heading=None,
             echo=True):
    """
    df: A Dataframe

    v_heading: heading to use as row index

    h_heading: heading to use as column index

    unq_internal_zones: unq zones to use as placeholder

    echo = True:
        Indicates whether to print a log of the process to the terminal.
        Useful to set echo=False when using multi-threaded loops
    """
    df = df.copy()

    placeholder = pd.DataFrame(unq_internal_zones).copy()
    col_name = list(placeholder)[0]

    if h_heading is None:
        placeholder = placeholder.rename(columns={
                list(placeholder)[0]:v_heading})
        full_placeholder = placeholder.merge(df,
                                             how='left',
                                             on=[v_heading])
        # Replace NAs with zeroes
        full_placeholder[values] = full_placeholder[values].fillna(0)

        array = full_placeholder[values].copy().values

    else:
        # Build placeholders
        ph_v = placeholder.copy()
        ph_v = ph_v.rename(columns={col_name: v_heading})
        ph_v['ph'] = 0
        ph_h = placeholder.copy()
        ph_h = ph_h.rename(columns={col_name: h_heading})
        ph_h['ph'] = 0

        # Join placeholders
        placeholder = ph_v.merge(ph_h,
                                 how='left',
                                 on='ph')
        placeholder = placeholder.drop(['ph'], axis=1)

        # Merge df onto placeholder
        full_placeholder = placeholder.merge(df,
                                             how='left',
                                             on=[h_heading, v_heading])

        # Replace NAs with zeroes
        full_placeholder[values] = full_placeholder[values].fillna(0)

        # Pivot to array
        array = full_placeholder.sort_values([
            v_heading,
            h_heading
        ]).pivot(
            index=v_heading,
            columns=h_heading,
            values=values
        ).values

    # Array len should be same as length of unq values
    if echo and len(array) == len(unq_internal_zones):
        print('Matrix length=%d. Matches input constraint.' % (len(array)))

    return array

def iz_infill(dat, infill=0.5):

    cols = list(dat)
    dat = dat.copy()
    min_inter_dat = dat[dat[cols[2]]>0]
    # Derive minimum intrazonal
    min_inter_dat = min_inter_dat.groupby(
        cols[0]).min().reset_index().drop(cols[1],axis=1)
    intra_dat = min_inter_dat.copy()
    intra_dat[cols[2]] = intra_dat[cols[2]]*infill
    iz = dat[dat[cols[0]] == dat[cols[1]]]
    non_iz = dat[dat[cols[0]] != dat[cols[1]]]
    iz = iz.drop(cols[2],axis=1)
    # Rejoin
    iz = iz.merge(intra_dat, how='inner', on=cols[0])
    dat = pd.concat([iz, non_iz],axis=0,sort=True).reset_index(drop=True)
        
    return dat

def get_zone_range(zone_vector):
    """
    Zone vector as series - returns contiguous zones
    Needed to avoid placeholder zones being an issue
    """
    min_zone = int(zone_vector.min())
    max_zone = int(zone_vector.max())
    unq_zones = [i for i in range(min_zone, max_zone+1)]

    return unq_zones

centroids = pd.read_csv(
    os.path.join(
        norms_network,
        [x for x in norms_files if 'centroid' in x.lower()][0]))
zone_station = pd.read_csv(
    os.path.join(
        norms_network,
        [x for x in norms_files if 'zone_station' in x.lower()][0]), header=None)
rail_nodes = pd.read_csv(
    os.path.join(
        norms_network,
        [x for x in norms_files if 'rail_nodes' in x.lower()][0]))
rail_links = pd.read_csv(
    os.path.join(
        norms_network,
        [x for x in norms_files if 'rail_links' in x.lower()][0]))

geo_centroids = centroids.rename(columns={'N':'norms_zone_id',
                                              'X':'x',
                                              'Y':'y'})

geo_centroids = geo_centroids.reindex(['norms_zone_id',
                                       'x',
                                       'y'], axis=1)

distance_dict = {}
for i1, row1 in geo_centroids.iterrows():

    print(i1)

    for i2, row2 in geo_centroids.iterrows():
        
        # Smallest first, skip if done
        zone_a = row1['norms_zone_id']
        zone_b = row2['norms_zone_id']

        name = str(zone_a) + '_' + str(zone_b)
                    
        x1 = row1['x']
        y1 = row1['y']
        x2 = row2['x']
        y2 = row2['y']

        distance = math.sqrt((x1-x2)**2 + (y1-y2)**2)

        distance_dict.update({
            name:{'a':zone_a,
                  'b':zone_b,
                  'distance':distance}})


distance_mat = pd.DataFrame(distance_dict).transpose().reset_index(drop=True)
# Get iz
distance_mat = iz_infill(distance_mat)

# To wide format
zone_range = get_zone_range(distance_mat['a'])

square_distance = df_to_np(distance_mat,
                           'distance',
                           unq_internal_zones = zone_range,
                           v_heading ='a' ,
                           h_heading= 'b',
                           echo=True)

access = np.broadcast_to(
    (np.diag(square_distance)*.5),(len(square_distance),len(square_distance)))
egress = access.T

out_distance_m = square_distance + access + egress
out_distance_km = out_distance_m/1000

# Back to long
out_distance_km_df = pd.DataFrame(out_distance_km,
                                  index=zone_range,
                                  columns=zone_range).reset_index()
out_distance_km_df = out_distance_km_df.rename(columns={'index':'o_zone'})

out_distance_km_long = pd.melt(
    out_distance_km_df, id_vars=['o_zone'],
    var_name='d_zone', value_name='distance', col_level=0)

# Make 24hr and tp using the same single column
cols_24 = ['p1_dist', 'p2_dist', 'p3_dist', 'p4_dist', 
           'p5_dist', 'p6_dist', 'p7_dist', 'p8_dist']
cols_tp = ['business_tp1', 'commute_tp1', 'other_tp1', 'business_tp2',
           'commute_tp2', 'other_tp2', 'business_tp3', 'commute_tp3',
           'other_tp3', 'business_tp4', 'commute_tp4', 'other_tp4']

distance_24 = out_distance_km_long.copy()
for col in cols_24:
    distance_24[col] = distance_24['distance']
distance_24 = distance_24.drop('distance', axis=1)
distance_24 = distance_24.rename(columns={'o_zone':'p_zone',
                                          'd_zone':'a_zone'})
print(list(distance_24))

distance_tp = out_distance_km_long.copy()
for col in cols_tp:
    distance_tp[col] = distance_tp['distance']
distance_tp = distance_tp.drop('distance', axis=1)
distance_tp = distance_tp.rename(columns={'o_zone':'p_zone',
                                          'd_zone':'a_zone'})
print(list(distance_tp))

distance_24.to_csv('norms_24hr_cost.csv', index=False)
distance_tp.to_csv('norms_tp_cost.csv', index=False)
