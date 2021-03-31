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
from normits_demand import constants as consts

from normits_demand.utils import output_converter as oc
from normits_demand.utils import general as du

from normits_demand.matrices import decompilation


def main():

    # Running params
    run_vdm_od2pa = False
    run_nhb_splitting_factors = True
    convert_tour_props = True
    convert_splitting_factors = True
    convert_matrices = True

    # Build and EFS instance for the paths
    scenario = consts.SC00_NTEM
    iter_num = '3f'
    import_home = "I:/"
    export_home = "E:/"
    model_name = consts.MODEL_NAME

    efs = nd.ExternalForecastSystem(
        iter_num=iter_num,
        model_name=model_name,
        scenario_name=scenario,
        import_home=import_home,
        export_home=export_home,
    )

    # Set up segmentation vals
    seg_level = 'vdm'
    seg_params = {
        'to_needed': ['hb'],
        'uc_needed': consts.USER_CLASSES,
        'm_needed': consts.MODEL_MODES[model_name],
        'ca_needed': efs.ca_needed,
        'tp_needed': consts.TIME_PERIODS
    }

    # Generate VDM Tour Props
    if run_vdm_od2pa:
        decompilation.decompile_noham(
            year='2018',
            seg_level=seg_level,
            seg_params=seg_params,
            post_me_import=efs.imports['post_me_matrices'],
            post_me_renamed_export=efs.exports['post_me']['compiled_od'],
            od_export=efs.exports['post_me']['od'],
            pa_export=efs.exports['post_me']['pa'],
            pa_24_export=efs.exports['post_me']['pa_24'],
            zone_translate_dir=efs.imports['zone_translation']['one_to_one'],
            tour_proportions_export=efs.params['tours'],
            decompile_factors_path=efs.imports['post_me_factors'],
            vehicle_occupancy_import=efs.imports['home'],
            overwrite_decompiled_od=False,
            overwrite_tour_proportions=True,
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
            year=consts.BASE_YEAR,
            seg_level=seg_level,
            seg_params=seg_params,
        )

    # Convert splitting factors

    # Convert matrices (which years)?
    ## Long format
    ## Norms Zones


if __name__ == '__main__':
    main()
