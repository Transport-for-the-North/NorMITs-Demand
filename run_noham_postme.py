# -*- coding: utf-8 -*-
"""
Created on: Thurs October 8 10:48:12 2020
Updated on:

Original author: Ben Taylor
Last update made by:
Other updates made by:

File purpose:
Small test file to help with testing post-ME code for noham while full noham
EFS runs are not working.
"""

import os


# Local imports
from normits_demand import efs_constants as consts

from normits_demand.utils import general as du
from normits_demand.utils import output_converter as oc
from normits_demand.utils import vehicle_occupancy as vo

from normits_demand.matrices import od_to_pa as od2pa
from normits_demand.matrices import pa_to_od as pa2od
from normits_demand.matrices import matrix_processing as mat_p


def tms_segmentation_tests(model_name,
                           m_needed,
                           process_count,
                           p_needed,
                           seg_level='tms',
                           audit_tol=0.001
                           ):
    if model_name == 'norms' or model_name == 'norms_2015':
        ca_needed = consts.CA_NEEDED
        from_pcu = False
    elif model_name == 'noham':
        ca_needed = None
        from_pcu = True
    else:
        raise ValueError("I don't know what model this is? %s"
                         % str(model_name))

    base_path = r'E:\NorMITs Demand\noham\v0.3-EFS_Output\NTEM\iter3b\Matrices\Post-ME Matrices'

    decompile_od_bool = False
    gen_tour_proportions_bool = False
    post_me_compile_pa = False
    pa_back_to_od_check = True

    # Validate inputs
    seg_level = du.validate_seg_level(seg_level)
    if seg_level != 'tms':
        raise ValueError("NOT TMS")

    # Set up seg params
    seg_params = {
        'p_needed': p_needed,
        'm_needed': m_needed,
        'ca_needed': ca_needed,
    }

    if decompile_od_bool:
        od2pa.convert_to_efs_matrices(
            import_path=r'E:\NorMITs Demand\noham\v2_2-EFS_Output\iter0\Matrices\Post-ME Matrices\Compiled OD Matrices\from_noham',
            export_path=r'E:\NorMITs Demand\noham\v2_2-EFS_Output\iter0\Matrices\Post-ME Matrices\Compiled OD Matrices',
            matrix_format='od',
            year=consts.BASE_YEAR,
            m_needed=m_needed,
            user_class=True,
            to_wide=True,
            wide_col_name=model_name + '_zone_id',
            from_pcu=from_pcu,
            vehicle_occupancy_import=r'Y:\NorMITs Demand\import'
        )

        od2pa.decompile_od(
            od_import=r'E:\NorMITs Demand\noham\v2_2-EFS_Output\iter0\Matrices\Post-ME Matrices\Compiled OD Matrices',
            od_export=r'E:\NorMITs Demand\noham\v2_2-EFS_Output\iter0\Matrices\Post-ME Matrices\OD Matrices',
            decompile_factors_path=r'E:\NorMITs Demand\noham\v2_2-EFS_Output\iter0\Params\Compile Params/od_compilation_factors.pickle',
            year=consts.BASE_YEAR,
            audit_tol=audit_tol
        )

    if gen_tour_proportions_bool:
        od_import = r'E:\NorMITs Demand\noham\v2_2-EFS_Output\iter0\Matrices\Post-ME Matrices\OD Matrices'
        pa_export = r'E:\NorMITs Demand\noham\v2_2-EFS_Output\iter0\Matrices\Post-ME Matrices\PA Matrices'

        mat_p.generate_tour_proportions(
            od_import=od_import,
            pa_export=pa_export,
            tour_proportions_export=r'E:\NorMITs Demand\noham\v2_2-EFS_Output\iter0\Params\Tour Proportions',
            zone_translate_dir=r'Y:\NorMITs Demand\import\zone_translation',
            model_name=model_name,
            year=consts.BASE_YEAR,
            seg_level=seg_level,
            seg_params=seg_params,
            process_count=process_count
        )

    if post_me_compile_pa:
        mat_p.build_compile_params(
            import_dir=r'E:\NorMITs Demand\noham\v2_2-EFS_Output\iter0\Matrices\Post-ME Matrices\PA Matrices',
            export_dir=r'E:\NorMITs Demand\noham\v2_2-EFS_Output\iter0\Params\Compile Params',
            matrix_format='pa',
            years_needed=[consts.BASE_YEAR],
            m_needed=m_needed,
            ca_needed=ca_needed,
            split_hb_nhb=True
        )

    if pa_back_to_od_check:
        mat_p.build_24hr_mats(
            import_dir=os.path.join(base_path, 'PA Matrices'),
            export_dir=os.path.join(base_path, '24hr PA Matrices'),
            matrix_format='pa',
            p_needed=consts.ALL_HB_P,
            year_needed=consts.BASE_YEAR,
            m_needed=m_needed,
            ca_needed=ca_needed,
        )

        mat_p.build_24hr_mats(
            import_dir=os.path.join(base_path, 'PA Matrices'),
            export_dir=os.path.join(base_path, '24hr PA Matrices'),
            matrix_format='pa',
            year_needed=consts.BASE_YEAR,
            m_needed=m_needed,
            p_needed=consts.ALL_NHB_P,
            ca_needed=ca_needed,
        )

        pa2od.build_od_from_fh_th_factors(
            pa_import=os.path.join(base_path, '24hr PA Matrices'),
            od_export=os.path.join(base_path, 'Test OD Matrices'),
            tour_proportions_dir=r'I:\NorMITs Demand\import\noham\post_me_tour_proportions',
            zone_translate_dir=r'I:\NorMITs Demand\import\zone_translation\one_to_one',
            model_name=model_name,
            years_needed=[consts.BASE_YEAR],
            seg_level=seg_level,
            seg_params=seg_params,
            process_count=process_count
        )

        # NEW CODE NEED TO FULLY TEST BELOW
        # Need to copy over NHB
        mat_p.copy_nhb_matrices(
            import_dir=os.path.join(base_path, 'OD Matrices'),
            export_dir=os.path.join(base_path, 'Test OD Matrices'),
        )

        mat_p.build_compile_params(
            import_dir=os.path.join(base_path, 'Test OD Matrices'),
            export_dir=os.path.join(base_path, 'Test Compile Params'),
            matrix_format='od',
            years_needed=[consts.BASE_YEAR],
            m_needed=m_needed,
            ca_needed=ca_needed,
            tp_needed=consts.TIME_PERIODS
        )

        compile_params_fname = du.get_compile_params_name('od', str(consts.BASE_YEAR))
        compile_param_path = os.path.join(base_path, 'Test Compile Params', compile_params_fname)

        du.compile_od(
            od_folder=os.path.join(base_path, 'Test OD Matrices'),
            write_folder=os.path.join(base_path, 'Test Compiled OD Matrices'),
            compile_param_path=compile_param_path,
            build_factor_pickle=False
        )

        # Convert the compiled OD into hourly average PCU
        vo.people_vehicle_conversion(
            mat_import=os.path.join(base_path, 'Test Compiled OD Matrices'),
            mat_export=os.path.join(base_path, 'Test PCU Compiled OD Matrices'),
            import_folder=r'I:\NorMITs Demand\import',
            mode=str(m_needed[0]),
            method='to_vehicles',
            out_format='wide',
            hourly_average=True
        )

        # Check against original postme compiled


def vdm_segmentation_tests(model_name,
                           m_needed,
                           process_count,
                           p_needed,
                           seg_level='vdm',
                           audit_tol=0.001):
    if model_name == 'norms' or model_name == 'norms_2015':
        ca_needed = consts.CA_NEEDED
        from_pcu = False
    elif model_name == 'noham':
        ca_needed = None
        from_pcu = True
    else:
        raise ValueError("I don't know what model this is? %s"
                         % str(model_name))

    decompile_od_bool = False
    gen_tour_proportions_bool = False
    create_outputs_for_vdm = True
    pa_back_to_od_check = False

    # Validate inputs
    seg_level = du.validate_seg_level(seg_level)
    if seg_level != 'vdm':
        raise ValueError("NOT VDM")

    seg_params = {
        'to_needed': ['hb'],
        'uc_needed': consts.USER_CLASSES,
        'm_needed': m_needed,
        'ca_needed': ca_needed,
        'tp_needed': consts.TIME_PERIODS
    }

    # TODO: Add VDM OD Matrices, VDM PA Matrices, VDM 24hr PA Matrices into
    #  imports

    if decompile_od_bool:
        od2pa.convert_to_efs_matrices(
            import_path=r'E:\NorMITs Demand\noham\v2_2-EFS_Output\iter0\Matrices\Post-ME Matrices\Compiled OD Matrices\from_noham',
            export_path=r'E:\NorMITs Demand\noham\v2_2-EFS_Output\iter0\Matrices\Post-ME Matrices\Compiled OD Matrices',
            matrix_format='od',
            year=consts.BASE_YEAR,
            m_needed=m_needed,
            user_class=True,
            to_wide=True,
            wide_col_name=model_name + '_zone_id',
            from_pcu=from_pcu,
            vehicle_occupancy_import=r'Y:\NorMITs Demand\import'
        )

        od2pa.decompile_od(
            od_import=r'E:\NorMITs Demand\noham\v2_2-EFS_Output\iter0\Matrices\Post-ME Matrices\Compiled OD Matrices',
            od_export=r'E:\NorMITs Demand\noham\v2_2-EFS_Output\iter0\Matrices\Post-ME Matrices\OD Matrices',
            decompile_factors_path=r'E:\NorMITs Demand\noham\v2_2-EFS_Output\iter0\Params\Compile Params/od_compilation_factors.pickle',
            year=consts.BASE_YEAR,
            audit_tol=audit_tol
        )

        # Build path for compile params
        output_fname = du.get_compile_params_name('vdm_od', consts.BASE_YEAR)
        compile_param_path = os.path.join(r'E:\NorMITs Demand\noham\v2_2-EFS_Output\iter0\Params\Test Compile Params',
                                          output_fname)

        # Compile to VDM
        mat_p.build_compile_params(
            import_dir=r'E:\NorMITs Demand\noham\v2_2-EFS_Output\iter0\Matrices\Post-ME Matrices\OD Matrices',
            export_dir=r'E:\NorMITs Demand\noham\v2_2-EFS_Output\iter0\Params\Test Compile Params',
            matrix_format='od',
            years_needed=[consts.BASE_YEAR],
            m_needed=m_needed,
            ca_needed=ca_needed,
            tp_needed=consts.TIME_PERIODS,
            split_hb_nhb=True,
            split_od_from_to=True,
            output_fname=output_fname
        )

        mat_p.compile_matrices(
            mat_import=r'E:\NorMITs Demand\noham\v2_2-EFS_Output\iter0\Matrices\Post-ME Matrices\OD Matrices',
            mat_export=r'E:\NorMITs Demand\noham\v2_2-EFS_Output\iter0\Matrices\Post-ME Matrices\VDM OD Matrices',
            compile_params_path=compile_param_path,
            factors_fname='test.pkl'
        )

    if gen_tour_proportions_bool:
        od_import = r'E:\NorMITs Demand\noham\v2_2-EFS_Output\iter0\Matrices\Post-ME Matrices\VDM OD Matrices'
        pa_export = r'E:\NorMITs Demand\noham\v2_2-EFS_Output\iter0\Matrices\Post-ME Matrices\VDM PA Matrices'

        mat_p.generate_tour_proportions(
            od_import=od_import,
            pa_export=pa_export,
            tour_proportions_export=r'E:\NorMITs Demand\noham\v2_2-EFS_Output\iter0\Params\Tour Proportions',
            zone_translate_dir=r'Y:\NorMITs Demand\import\zone_translation',
            model_name=model_name,
            year=consts.BASE_YEAR,
            seg_level=seg_level,
            seg_params=seg_params,
            process_count=process_count
        )

    if create_outputs_for_vdm:
        # Create 24hr PA HB
        mat_p.build_24hr_vdm_mats(
            import_dir=r'E:\NorMITs Demand\noham\v2_2-EFS_Output\iter0\Matrices\Post-ME Matrices\VDM PA Matrices',
            export_dir=r'E:\NorMITs Demand\noham\v2_2-EFS_Output\iter0\Matrices\Post-ME Matrices\VDM 24hr PA Matrices',
            matrix_format='pa',
            years_needed=[consts.BASE_YEAR],
            **seg_params
        )

        # Convert tour props to noham vdm format
        oc.noham_vdm_tour_proportions_out(
            input_path=r'E:\NorMITs Demand\noham\v2_2-EFS_Output\iter0\Params\Tour Proportions',
            output_path=r'E:\NorMITs Demand\noham\v2_2-EFS_Output\iter0\Params\Noham Tour Proportions',
            year=consts.BASE_YEAR,
            seg_level=seg_level,
            seg_params=seg_params,
        )

        # Create 24hr OD NHB - with TP splitting factors
        out_seg_params = seg_params.copy()
        out_seg_params['to_needed'] = ['nhb']
        mat_p.build_24hr_vdm_mats(
            import_dir=r'E:\NorMITs Demand\noham\v2_2-EFS_Output\iter0\Matrices\Post-ME Matrices\VDM OD Matrices',
            export_dir=r'E:\NorMITs Demand\noham\v2_2-EFS_Output\iter0\Matrices\Post-ME Matrices\VDM 24hr PA Matrices',
            matrix_format='od',
            years_needed=[consts.BASE_YEAR],
            split_factors_path=r'E:\NorMITs Demand\noham\v2_2-EFS_Output\iter0\Params\Noham Tour Proportions',
            **out_seg_params
        )

    if pa_back_to_od_check:
        # TODO: Consolidate build_24hr_vdm_mats() and build_24hr_mats()
        mat_p.build_24hr_vdm_mats(
            import_dir=r'E:\NorMITs Demand\noham\v2_2-EFS_Output\iter0\Matrices\Post-ME Matrices\VDM PA Matrices',
            export_dir=r'E:\NorMITs Demand\noham\v2_2-EFS_Output\iter0\Matrices\Post-ME Matrices\VDM 24hr PA Matrices',
            matrix_format='pa',
            years_needed=[consts.BASE_YEAR],
            **seg_params
        )

        pa2od.build_od_from_fh_th_factors(
            pa_import=r'E:\NorMITs Demand\noham\v2_2-EFS_Output\iter0\Matrices\Post-ME Matrices\VDM 24hr PA Matrices',
            od_export=r'E:\NorMITs Demand\noham\v2_2-EFS_Output\iter0\Matrices\Post-ME Matrices\VDM Test OD Matrices',
            tour_proportions_dir=r'E:\NorMITs Demand\noham\v2_2-EFS_Output\iter0\Params\Tour Proportions',
            zone_translate_dir=r'Y:\NorMITs Demand\import\zone_translation',
            model_name=model_name,
            years_needed=[consts.BASE_YEAR],
            seg_level=seg_level,
            seg_params=seg_params,
            process_count=process_count
        )

        # Compile back to ME2
        # ME2 seg = VDM hb + nhb, od_from + od_to
        # Cheat compare -> VDM Test OD Matrices Vs VDM OD Matrices

        # Convert the compiled OD into hourly average PCU
        vo.people_vehicle_conversion(
            input_folder=r'E:\NorMITs Demand\noham\v2_2-EFS_Output\iter0\Matrices\Post-ME Matrices\VDM Test Compiled OD Matrices',
            export_folder=r'E:\NorMITs Demand\noham\v2_2-EFS_Output\iter0\Matrices\Post-ME Matrices\VDM Test PCU Compiled OD Matrices',
            import_folder=r'Y:\NorMITs Demand\import',
            mode=str(m_needed[0]),
            method='to_vehicles',
            out_format='wide',
            hourly_average=True
        )


def main():
    model_name = 'noham'
    m_needed = [3]
    process_count = 5
    p_needed = consts.ALL_HB_P

    # Audit as we go
    audit_tol = 0.001

    # Control Flow
    run_tms_seg_tests = True
    run_tms_vdm_tests = False

    # TODO: Check how OD2PA works with new norms/norms_2015

    if run_tms_seg_tests:
        tms_segmentation_tests(
            model_name,
            m_needed,
            process_count,
            p_needed,
            seg_level='tms',
            audit_tol=audit_tol
        )

    if run_tms_vdm_tests:
        vdm_segmentation_tests(
            model_name,
            m_needed,
            process_count,
            p_needed,
            seg_level='vdm',
            audit_tol=audit_tol
        )


def rename():
    m_needed = [3]
    model_name = 'noham'

    # od2pa.convert_to_efs_matrices(
    #     import_path=r'I:\NorMITs Demand\import\noham\post_me',
    #     export_path=r'I:\NorMITs Demand\import\noham\post_me\renamed',
    #     matrix_format='od',
    #     year=efs_consts.BASE_YEAR,
    #     m_needed=m_needed,
    #     user_class=True,
    #     to_wide=True,
    #     wide_col_name=model_name + '_zone_id',
    #     from_pcu=False,
    # )

    od2pa.decompile_od(
        od_import=r'I:\NorMITs Demand\import\noham\post_me\renamed',
        od_export=r'I:\NorMITs Demand\import\noham\post_me\tms_seg',
        decompile_factors_path=r"I:\NorMITs Demand\import\noham\params\post_me_tms_decompile_factors.pkl",
        year=consts.BASE_YEAR,
    )


if __name__ == '__main__':
    # main()
    rename()
