"""
Fusion engine process
"""

# TODO: add required imports
import numpy as np
import pandas as pd
import pickle as pk


def package_fusion_distances(inputs_path,
                             output_path,
                             version_name):
    print('packaging distances')
    dct_uc = {'Business': 1,
              'Commute': 2,
              'Other': 3,
              'LGV': 4,
              'HGV': 5}

    dct_noham_dist = {}
    for uc in dct_uc:
        path = inputs_path + '\\' + version_name + '_Dist_' + str(uc) + '.csv'
        print(path)
        distance = pd.read_csv(path, header=None, names=['From', 'To', 'Dist'])
        distance_array = distance.pivot(index='From',
                                        columns='To',
                                        values='Dist')
        distance_array = distance_array.to_numpy()
        distance_array = np.nan_to_num(distance_array)
        dct_noham_dist[uc] = distance_array
    with open(output_path + '\\NoHAM_Distances.pkl', 'wb') as log:
        pk.dump(dct_noham_dist, log, pk.HIGHEST_PROTOCOL)
    return ('Version ' + str(version_name) + ' distances,\n' +
            'From: ' + str(inputs_path) + '\n' +
            'Packaged to: ' + str(output_path))


def build_fusion_factor(input_matrix,
                        distance_matrix,
                        od_type_matrix,
                        chop_head=False,
                        chop_tail=False,
                        type_filter=False,
                        include_type=None,
                        exclude_type=None,
                        invert=False,
                        min_dist=0,
                        max_dist=9999,
                        default_value=1):
    if include_type is None:
        include_type = ['I-I']
    if exclude_type is None:
        exclude_type = ['I-E', 'I-S', 'E-I', 'S-I', 'E-E', 'E-S', 'S-E', 'S-S']
    unique_types = list(np.unique(od_type_matrix))
    print(str(chop_head) + ', ' + str(chop_tail) + ', ' + str(type_filter))
    fusion_factor = (input_matrix * 0) + default_value
    if chop_head:
        head_factor = np.where(distance_matrix <= min_dist, 0, 1)
    else:
        head_factor = (input_matrix * 0) + 1
    if chop_tail:
        tail_factor = np.where(distance_matrix > max_dist, 0, 1)
    else:
        tail_factor = (input_matrix * 0) + 1
    if type_filter:
        type_factor = od_type_matrix
        for unq in unique_types:
            print(str(unq))
            if unq in include_type:
                type_factor = np.where(type_factor == unq, 1, type_factor)
            elif unq in exclude_type:
                type_factor = np.where(type_factor == unq, 0, type_factor)
            else:
                print('Variable (' + str(unq) + ') not in either include or exclude list')
        type_factor = type_factor.astype(int)
    else:
        type_factor = (input_matrix * 0) + 1
    fusion_factor = (fusion_factor * head_factor * tail_factor * type_factor)
    if invert:
        fusion_factor = np.array(fusion_factor, dtype=bool)
        fusion_factor = ~fusion_factor
        fusion_factor = np.array(fusion_factor, dtype=int)
    print(np.sum(fusion_factor))
    return fusion_factor


def mdd_fusion(observed_matrix,
               synthetic_matrix,
               distance_matrix,
               type_matrix,
               short_infill=True,
               type_infill='None'
               ):
    print(np.sum(observed_matrix))
    print(np.sum(synthetic_matrix))
    print(np.average(distance_matrix))
    print(type_matrix)
    print(short_infill)
    print(type_infill)
    # TODO: build observed fusion factor call
    # TODO: build synthetic fusion factor call
    # TODO: build fusion process
    fusion_matrix = (observed_matrix + synthetic_matrix) / 2
    return fusion_matrix



