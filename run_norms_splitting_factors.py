# -*- coding: utf-8 -*-
"""
Created on: Mon March 1 16:30:34 2020
Updated on:

Original author: Ben Taylor
Last update made by:
Other updates made by:

File purpose:
WRITE PURPOSE
"""
import os

import pandas as pd

import normits_demand.constants as consts
from normits_demand.matrices import matrix_processing as mat_p

PA_MATRICES_DIR = r'I:\NorMITs Synthesiser\Norms\iter4\24hr PA Matrices'
PA_VDM_MATRICES_DIR = r'I:\NorMITs Synthesiser\Norms\iter4\24hr VDM PA Matrices'
PARAMS_EXPORT = r'I:\NorMITs Synthesiser\Norms\iter4\params'

MODEL_SCHEMA_DIR = r'I:\NorMITs Demand\import\norms\model schema'
BASE_YEAR = '2018'

AVOID_ZERO_SPLITS = True


def main():
    int_path = os.path.join(MODEL_SCHEMA_DIR, consts.INTERNAL_AREA % 'norms')
    ext_path = os.path.join(MODEL_SCHEMA_DIR, consts.EXTERNAL_AREA % 'norms')

    # Default read as int
    dtype = {'norms_zone_id': int}

    internal_zones = pd.read_csv(int_path, dtype=dtype).squeeze().tolist()
    external_zones = pd.read_csv(ext_path, dtype=dtype).squeeze().tolist()

    mat_p.compile_norms_to_vdm(
        mat_import=PA_MATRICES_DIR,
        mat_export=PA_VDM_MATRICES_DIR,
        params_export=PARAMS_EXPORT,
        year=BASE_YEAR,
        internal_zones=internal_zones,
        external_zones=external_zones,
        post_me_import=None,
        matrix_format='pa',
        avoid_zero_splits=AVOID_ZERO_SPLITS
    )


if __name__ == '__main__':
    main()
