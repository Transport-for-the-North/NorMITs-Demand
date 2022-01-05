# -*- coding: utf-8 -*-
"""
Created on: Tues March 30 14:32:23 2021
Updated on:

Original author: Ben Taylor
Last update made by:
Other updates made by:

File purpose:
Convert NoHAM outputs into a format ready for NoRMS assignments
"""
# Built-ins
import os

# Third party
import pandas as pd

# Local imports
import normits_demand as nd
from normits_demand import constants
from normits_demand import efs_constants

from normits_demand.utils import output_converter as oc
from normits_demand.utils import general as du

from normits_demand.matrices import decompilation
from normits_demand.matrices import matrix_processing as mat_p


def main():

    # Running params
    run_vdm_od2pa = True
    run_nhb_splitting_factors = False
    convert_tour_props = False
    convert_matrices = False

    # Build and EFS instance for the paths
    scenario = constants.SC00_NTEM
    iter_num = '3i'
    import_home = "I:/"
    export_home = "E:/"
    model_name = efs_constants.MODEL_NAME

    efs = nd.ExternalForecastSystem(
        iter_num=iter_num,
        model_name=model_name,
        scenario_name=scenario,
        import_home=import_home,
        export_home=export_home,
    )

    # Set up segmentation vals
    seg_level = 'vdm'
    hb_seg_params = {
        'to_needed': ['hb'],
        'uc_needed': efs_constants.USER_CLASSES,
        'm_needed': efs_constants.MODEL_MODES[model_name],
        'ca_needed': efs.ca_needed,
        'tp_needed': efs_constants.TIME_PERIODS
    }
    nhb_seg_params = hb_seg_params.copy()
    nhb_seg_params['to_needed'] = ['nhb']

    # Generate VDM Tour Props
    if run_vdm_od2pa:
        decompilation.decompile_noham(
            year='2018',
            seg_level=seg_level,
            seg_params=hb_seg_params,
            post_me_import=efs.imports['post_me_matrices'],
            post_me_renamed_export=efs.exports['post_me']['compiled_od'],
            od_export=efs.exports['post_me']['od'],
            pa_export=efs.exports['post_me']['pa'],
            pa_24_export=efs.exports['post_me']['pa_24'],
            zone_translate_dir=efs.imports['zone_translation']['one_to_one'],
            tour_proportions_export=efs.params['tours'],
            decompile_factors_path=efs.imports['post_me_factors'],
            vehicle_occupancy_import=efs.imports['home'],
            overwrite_decompiled_od=True,
            overwrite_tour_proportions=True,
        )

        mat_p.build_24hr_vdm_mats(
            import_dir=efs.exports['post_me']['pa'],
            export_dir=efs.exports['post_me']['vdm_pa_24'],
            matrix_format='pa',
            years_needed=[efs_constants.BASE_YEAR],
            **hb_seg_params
        )

    if run_nhb_splitting_factors:
        mat_p.build_24hr_vdm_mats(
            import_dir=efs.exports['post_me']['od'],
            export_dir=efs.exports['post_me']['od_24'],
            matrix_format='od',
            split_factors_path=efs.params['tours'],
            years_needed=[efs_constants.BASE_YEAR],
            **nhb_seg_params
        )

    # Convert tour props
    if convert_tour_props:
        # Create the output path
        noham_tp_path = os.path.join(efs.params['tours'], 'noham_format')
        du.create_folder(noham_tp_path)

        # Convert tour props
        oc.noham_vdm_tour_proportions_out(
            input_path=efs.params['tours'],
            output_path=noham_tp_path,
            year=efs_constants.BASE_YEAR,
            seg_level=seg_level,
            seg_params=hb_seg_params,
        )

    # Convert matrices to long format
    if convert_matrices:
        out_path = os.path.join(efs.exports['post_me']['home'], 'Long Format')
        du.create_folder(out_path)

        hb = efs.exports['post_me']['vdm_pa_24']
        nhb = efs.exports['post_me']['od_24']

        for dir_name in [hb, nhb]:
            oc.convert_wide_to_long(
                import_dir=dir_name,
                export_dir=out_path,
                matrix_format='pa'
            )


def future_year():
    # Running params
    future_years = [2033, 2040, 2050]

    compile_future_years = False
    convert_matrices = True

    # Build and EFS instance for the paths
    scenario = constants.SC01_JAM
    iter_num = '3i'
    import_home = "I:/"
    export_home = "I:/"
    model_name = 'noham'

    efs = nd.ExternalForecastSystem(
        iter_num=iter_num,
        model_name=model_name,
        scenario_name=scenario,
        import_home=import_home,
        export_home=export_home,
    )

    # Set up segmentation vals
    # seg_level = 'vdm'
    # hb_seg_params = {
    #     'to_needed': ['hb'],
    #     'uc_needed': consts.USER_CLASSES,
    #     'm_needed': consts.MODEL_MODES[model_name],
    #     'ca_needed': efs.ca_needed,
    #     'tp_needed': consts.TIME_PERIODS
    # }
    # nhb_seg_params = hb_seg_params.copy()
    # nhb_seg_params['to_needed'] = ['nhb']

    if compile_future_years:
        compile_params_paths = mat_p.build_compile_params(
            import_dir=efs.exports['pa_24_elast'],
            export_dir=efs.params['compile'],
            matrix_format='pa',
            split_hb_nhb=True,
            years_needed=future_years,
            m_needed=efs_constants.MODEL_MODES[model_name],
            ca_needed=efs.ca_needed,
            tp_needed=None,
        )

        for path in compile_params_paths:
            mat_p.compile_matrices(
                mat_import=efs.exports['pa_24_elast'],
                mat_export=efs.exports['vdm_pa_24'],
                compile_params_path=path,
                round_dp=constants.DEFAULT_ROUNDING,
            )

    # Convert matrices to long format
    if convert_matrices:
        in_path = efs.exports['vdm_pa_24']
        out_path = os.path.join(in_path, 'Long Format')
        du.create_folder(out_path)

        oc.convert_wide_to_long(
            import_dir=in_path,
            export_dir=out_path,
            matrix_format='pa',
        )


if __name__ == '__main__':
    main()
    # future_year()
