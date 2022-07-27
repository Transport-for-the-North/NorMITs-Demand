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
from normits_demand.matrices import decompilation

PA_MATRICES_DIR = r'I:\NorMITs Synthesiser\Norms\iter4\Compilation Outputs'
PA_MATRICES_DIR = None
PA_24_MATRICES_DIR = r'I:\NorMITs Synthesiser\Norms_2015\iter4\24hr PA Matrices'
PA_VDM_MATRICES_DIR = r'I:\NorMITs Synthesiser\Norms_2015\iter4\24hr VDM PA Matrices'
PARAMS_EXPORT = r'I:\NorMITs Synthesiser\Norms_2015\iter4\params'

MODEL_SCHEMA_DIR = r'I:\NorMITs Demand\import\norms\model schema'
BASE_YEAR = '2018'

AVOID_ZERO_SPLITS = True


def generate_splitting_factors():
    int_path = os.path.join(MODEL_SCHEMA_DIR, consts.INTERNAL_AREA % 'norms')
    ext_path = os.path.join(MODEL_SCHEMA_DIR, consts.EXTERNAL_AREA % 'norms')

    # Default read as int
    dtype = {'norms_zone_id': int}

    internal_zones = pd.read_csv(int_path, dtype=dtype).squeeze().tolist()
    external_zones = pd.read_csv(ext_path, dtype=dtype).squeeze().tolist()

    # Make sure we're aggregated up to TMS segmentation
    if PA_MATRICES_DIR is not None and PA_24_MATRICES_DIR != PA_MATRICES_DIR:
        print("Compiling 24hr Matrices...")
        mat_p.build_24hr_mats(
            import_dir=PA_MATRICES_DIR,
            export_dir=PA_24_MATRICES_DIR,
            matrix_format='pa',
            year_needed='2018',
            p_needed=consts.ALL_P,
            m_needed=consts.MODEL_MODES['norms'],
            ca_needed=consts.CA_NEEDED,
        )

    # Build the splitting factors
    mat_p.compile_norms_to_vdm(
        mat_pa_import=PA_24_MATRICES_DIR,
        # TODO(BT): Actually pass in OD here
        mat_od_import=PA_24_MATRICES_DIR,
        mat_export=PA_VDM_MATRICES_DIR,
        params_export=PARAMS_EXPORT,
        year=BASE_YEAR,
        m_needed=consts.MODEL_MODES['norms'],
        internal_zones=internal_zones,
        external_zones=external_zones,
        pa_matrix_format='pa',
        od_to_matrix_format='pa',
        od_from_matrix_format='pa',
        avoid_zero_splits=AVOID_ZERO_SPLITS
    )


def split_post_me():
    post_me_in = r'I:\NorMITs Synthesiser\Norms_2015\iter4\Norms 15 Post ME'
    renamed_pme = r'I:\NorMITs Synthesiser\Norms_2015\iter4\Norms 15 Post ME\renamed'
    tms_post_me_out = r'I:\NorMITs Synthesiser\Norms_2015\iter4\Norms 15 Post ME\tms_seg'
    params_dir = r'I:\NorMITs Synthesiser\Norms_2015\iter4\params'

    decompilation.decompile_norms(
        year=2018,
        post_me_import=post_me_in,
        post_me_renamed_export=renamed_pme,
        post_me_decompiled_export=tms_post_me_out,
        decompile_factors_dir=params_dir,
        from_to_factors_out=None,
        audit_tol=0.3,
    )


if __name__ == '__main__':
    # generate_splitting_factors()

    split_post_me()

