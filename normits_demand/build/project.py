
import os

import pandas as pd


def select_params_file(config_path):
    """
    Crawls a given folder and returns the TMS run params sheet
    """
    params = [x for x in os.listdir(config_path) if 'params' in x]
    if len(params) == 0:
        raise ValueError('no trip length bands in folder')
    for (i, option) in enumerate(params, 0):
        print(i, option)
        selection_c = input('Choose a config file (index): ')
        params_file = os.path.join(config_path,
                                   params[int(selection_c)],
                                   )

    return params_file
