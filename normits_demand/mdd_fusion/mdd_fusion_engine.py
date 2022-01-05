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
                        origin_type_matrix,
                        dest_type_matrix,
                        chop_head=False,
                        chop_tail=False,
                        origin_type=False,
                        dest_type=False,
                        min_dist=0,
                        max_dist=9999,
                        default_value=1):
    print(str(chop_head) + ', ' + str(chop_tail) + ', ' + str(origin_type) + ', ' + str(dest_type))
    fusion_factor = (input_matrix * 0) + default_value
    if chop_head:
        head_factor = np.where(distance_matrix <= min_dist, 0, 1)
    else:
        head_factor = (input_matrix * 0) + 1
    if chop_tail:
        tail_factor = np.where(distance_matrix > max_dist, 0, 1)
    else:
        tail_factor = (input_matrix * 0) + 1
    if origin_type and dest_type:
        type_factor = np.minimum(origin_type_matrix, dest_type_matrix)
    elif origin_type:
        type_factor = origin_type_matrix
    elif dest_type:
        type_factor = dest_type_matrix
    else:
        type_factor = (input_matrix * 0) + 1
    fusion_factor = (fusion_factor * head_factor * tail_factor * type_factor)
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



