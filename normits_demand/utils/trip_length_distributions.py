# -*- coding: utf-8 -*-
"""
Created on: 29/09/2021
Updated on:

Original author: Ben Taylor
Last update made by: Ben Taylor
Other updates made by: Chris Storey

File purpose:
Collection of utility functions for dealing with trip distributions
"""
# Built-Ins
import os

from typing import Any
from typing import Optional

# Third Party
import numpy as np
import pandas as pd

# Local Imports
import normits_demand as nd


def get_trip_length_distributions(import_dir: nd.PathLike,
                                  segment_params: nd.SegmentParams,
                                  trip_origin: str,
                                  replace_nan: Optional[Any] = None,
                                  ) -> pd.DataFrame:
    """Returns the trip length distribution for segment params in import_folder

    Parameters
    ----------
    import_dir:
        The directory to look for trip length distributions

    segment_params:
        A dictionary where the keys are the names of the segments and
        the values are the segment values to search for. This is used
        to select the correct trip length distribution.

    trip_origin:
        Where the trip originated. Usually either 'hb' or 'nhb'

    replace_nan:
        Whether to replace any NaN values found in the trip length
        distribution with 0 values or not

    Returns
    -------

    """
    # TODO(BT): Tidy up get_trip_length_distributions
    # Append name of tlb area

    # Index folder
    target_files = os.listdir(import_dir)
    # Define file contents, should just be target files - should fix.
    import_files = target_files.copy()

    for key, value in segment_params.items():
        # Don't want empty segments, don't want ca
        if value != 'none' and key != 'mat_type':
            # print_w_toggle(key + str(value), echo=echo)
            import_files = [x for x in import_files if
                            ('_' + key + str(value)) in x]

    if trip_origin == 'hb':
        import_files = [x for x in import_files if 'nhb' not in x]
    elif trip_origin == 'nhb':
        import_files = [x for x in import_files if 'nhb' in x]
    else:
        raise ValueError('Trip length band import failed,' +
                         'provide valid trip origin')

    if len(import_files) <= 0:
        raise IOError(
            "Cannot find any %s trip length bands.\n"
            "import folder: %s"
            % (trip_origin, import_dir)
        )

    for key, value in segment_params.items():
        # Don't want empty segments, don't want ca
        if value != 'none' and key != 'mat_type':
            # print_w_toggle(key + str(value), echo=echo)
            import_files = [x for x in import_files if
                            ('_' + key + str(value)) in x]

    if len(import_files) <= 0:
        raise IOError(
            "Cannot find any import files matching the given criteria.\n"
            'Import folder: %s\n'
            'Search criteria: %s'
            % (import_dir, segment_params)
        )

    if len(import_files) > 1:
        raise Warning(
            'Found multiple viable files. Cannot pick one.\n'
            'Search criteria: %s\n'
            'Import folder: %s\n'
            'Viable files: %s'
            % (segment_params, import_dir, import_files)
        )

    # Import
    tlb = pd.read_csv(os.path.join(import_dir, import_files[0]))

    if replace_nan is not None:
        for col_name in list(tlb):
            tlb[col_name] = tlb[col_name].fillna(replace_nan)

    return tlb


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
        if isinstance(band_trips, pd.DataFrame):
            band_trips = band_trips.values
        total_trips = np.sum(band_trips)
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

    return dist_mat, bs_mat, global_atl


def get_trip_length(distance, demand):
    """
    Take trip length as matrix
    Take pa as matrix
    Trim distance if needed
    Return average trip length
    Return trip length distribution vector

    distance = distance matrix as numpy ndarray
    internal_pa = demand as
    """

    # TODO: Just copy that bit below
    global_trips = demand.sum(axis=1).sum()
    global_distance = demand * distance

    global_atl = global_distance.sum(axis=1).sum() / global_trips

    return global_atl
