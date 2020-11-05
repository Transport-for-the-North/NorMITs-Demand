# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 12:40:21 2020

@author: genie
"""

import pandas as pd
import numpy as np

def get_trip_length(distance,
                    demand):
    """    
    distance = distance matrix as 2d numpy ndarray
    demand = demand as 2d numpy ndarray
    """

    # TODO: Just copy that bit below
    global_trips = demand.sum(axis=1).sum()
    global_distance = demand * distance

    global_atl = global_distance.sum(axis=1).sum() / global_trips

    return(global_atl)

def get_trip_length_by_band(band_atl,
                            distance,
                            internal_pa):
    """
    Take ttl by band, return atl by band.
    
    """

    # TODO: Drop averages of nothing in trip length band
    # reset index, needed or not
    band_atl = band_atl.reset_index(drop=True)

    # Get global trips
    global_trips = internal_pa.sum(axis=1).sum()

    # Get global atl
    global_atl = get_trip_length(distance,
                                 internal_pa)

    # Get min max for each
    if 'tlb_desc' in list(band_atl):
        # R built
        ph = band_atl['tlb_desc'].str.split('-', n=1, expand=True)
        band_atl['min'] = ph[0].str.replace('(', '')
        band_atl['max'] = ph[1].str.replace('[', '')
        band_atl['min'] = band_atl['min'].str.replace('(', '').values
        band_atl['max'] = band_atl['max'].str.replace(']', '').values
        del(ph)
    elif 'lower' in list(band_atl):
        # Convert bands to km
        band_atl['min'] = band_atl['lower']*1.61
        band_atl['max'] = band_atl['upper']*1.61

    dist_mat = []
    bs_mat = []

    # Loop over rows in band_atl
    for index, row in band_atl.iterrows():

        # Get total distance
        band_mat = np.where((distance >= float(row['min'])) & (distance < float(row['max'])), distance, 0)
        total_distance = (internal_pa * band_mat).sum()

        # Get subset matrix for distance
        distance_bool = np.where(band_mat==0, band_mat, 1)
        band_trips = internal_pa * distance_bool

        # Get output parameters
        total_trips = band_trips.sum()
        band_share = total_trips/global_trips

        # Get average trip length
        if total_trips > 0:
            atl = total_distance / total_trips
        else:
            atl = 0

        dist_mat.append({'tlb_index': index,
                         'atl': atl,
                         'ttl': row['ave_km']})
        bs_mat.append({'tlb_index': index,
                       'bs': band_share,
                       'tbs': row['band_share']})

    # TODO: Handle on output side to avoid error
    dist_mat = pd.DataFrame(dist_mat)
    dist_mat = dist_mat.reindex(['tlb_index', 'ttl', 'atl'], axis=1)

    bs_mat = pd.DataFrame(bs_mat)
    bs_mat = bs_mat.reindex(['tlb_index', 'tbs', 'bs'], axis=1)

    return(dist_mat, bs_mat, global_atl)
    
    