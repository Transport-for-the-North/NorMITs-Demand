"""
Test implementation of matrix compression
from:
https://medium.com/better-programming/load-fast-load-big-with-compressed-pickles-5f311584507e
For further research!
"""

import bz2
import pickle
import _pickle as cPickle

"""
Test use

import os
import pandas as pd

some_mat = 'I:/NorMITs Synthesiser/Noham/iter8c/Distribution Outputs/Compiled OD Matrices/od_m3_business_tp1.csv'
mat_out = pd.read_csv(some_mat)

path = os.path.join(
            os.getcwd(),
            'matrix_name')

dat_out(path,
        mat_out)

in_path = path + '.pbz2'

mat_in = dat_in(in_path)

mat_out == mat_in

Out as 56mb
"""

def dat_out(
        title: str,
        data):
    """
    title: str
    name of file for output. no file extension.
    data:
    Any data to send to pickle
    """
    with bz2.BZ2File(title + '.pbz2', 'w') as f:
        cPickle.dump(data, f)

def dat_in(
        file: str):
    
    """
    Parameters
    ----------
    file: file path

    Returns
    -------
    data read in - as what it wrote out apparently

    """
    data = bz2.BZ2File(file, 'rb')
    data = cPickle.load(data)
    return data