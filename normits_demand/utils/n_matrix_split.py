"""

"""
import numpy as np


def n_matrix_split(matrix,
                   indices=['scalar_x', 'scalar_y'],
                   index_names=['i', 'e'],
                   summarise=False):
    """
    Function to split matrices by internal/external or north/south etc.
    internal index has already been reduced to start at 0
    split type should take [ie] or [north/south]
    I'll write the north south later
    """
    # Check for missing indices in the given matrix
    sum_len = sum([len(x) for x in indices])
    if sum_len != len(matrix):
        # TODO: build a third category with a name and a list of the missing
        print('Do something')

    # Bundle up indices into a dict
    ind_dict = {}
    n_i = 0
    for i in indices:
        ind_dict.update({index_names[n_i]:i})
        n_i = n_i+1

    mats = []

    # Iterate over each combination.
    # Use take to get relevant matrix section

    for key_a, dat_a in ind_dict.items():
        for key_b, dat_b in ind_dict.items():
            label = (str(key_a) + '_to_' + str(key_b))
            out_mat = np.take(matrix, dat_a, axis=1)
            out_mat = matrix.take(dat_a, axis=0)
            out_mat = out_mat.take(dat_b, axis=1)
            if summarise:
                out_mat = out_mat.sum()
            ret_dict = {'name':label,
                        'dat':out_mat}
            mats.append(ret_dict)
    # TODO: Can you give me in a matrix??

    return mats
