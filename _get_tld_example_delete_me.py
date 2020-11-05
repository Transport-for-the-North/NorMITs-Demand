# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 12:57:47 2020

Dummy script to show how to get trip lengths by band from a matrix and a set
of distances.

Please delete once understood/integrated :)

"""
import os

import pandas as pd

from demand_utilities import reports as dr
from demand_utilities import utils as du

_tld_path = 'Y:/NorMITs Demand/import/trip_length_bands/north/standard_segments'
_cost_lookup_path = 'Y:/NorMITs Demand/import/noham'
_sample_matrix_path = 'Y:/NorMITs Synthesiser/Noham/iter8c/Distribution Outputs/PA Matrices'

# Get a matrix
mat_dict = du.parse_mat_output(os.listdir(_sample_matrix_path),
                              sep = '_',
                              mat_type = 'pa',
                              file_format = '.csv',
                              file_name = 'matrix').loc[0].to_dict()

# Build matrix description dict and import matrix
matrix = mat_dict.pop('matrix')
matrix = pd.read_csv(os.path.join(_sample_matrix_path,
                                  matrix))
trip_origin = mat_dict.pop('trip_origin')
mat_dict.pop('tp')

# Change numerics to integers or it gets funny with you
for item, dat in mat_dict.items():
    if dat.isnumeric():
        mat_dict.update({item:int(dat)})

# Get trip length bands
# You can build new folders in the _tld_path
# for future year target trip length distributions
tlb = du.get_trip_length_bands(_tld_path,
                               mat_dict,
                               'ntem', #Ignore this, legacy code
                               trip_origin,
                               replace_nan=False,
                               echo=True)

# Get distance matrix
costs, cost_name = du.get_costs(_cost_lookup_path,
                                mat_dict,
                                tp = '24hr',
                                iz_infill = 0.5)
print('Returned costs ' + cost_name)

# Make 2d mats into numpy arrays
# Square costs - (needs unq zones as list)
unq_zones = list(range(1,(costs[list(costs)[0]].max())+1))
costs = du.df_to_np(costs,
                    v_heading = 'p_zone',
                    h_heading = 'a_zone',
                    values = 'cost',
                    unq_internal_zones=unq_zones)

# Square demand (already square, just needs index dropping)
matrix = matrix.drop(list(matrix)[0],axis=1).values

# Get trip length by band
# ttl = target trip length
# atl = actual trip length
trip_lengths_by_band_km, band_shares_by_band, average_trip_length = dr.get_trip_length_by_band(tlb,
                                                                                               costs,
                                                                                               matrix)