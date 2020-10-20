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

import od_to_pa as od2pa
import pa_to_od as pa2od
import matrix_processing as mat_p

import efs_constants as consts


def main():
    model_name = 'noham'
    if model_name == 'norms':
        ca_needed = consts.CA_NEEDED
        from_pcu = False
    elif model_name == 'noham':
        ca_needed = None
        from_pcu = True
    else:
        raise ValueError("I don't know what model this is? %s"
                         % str(model_name))

    m_needed = [3]
    process_count = 6

    decompile_od_bool = False
    gen_tour_proportions_bool = True
    post_me_compile_pa = False
    pa_back_to_od_check = False

    if decompile_od_bool:
        od2pa.convert_to_efs_matrices(
            import_path=r'E:\NorMITs Demand\noham\v2_2-EFS_Output\iter0\Matrices\Post-ME Matrices\Compiled OD Matrices\from_noham',
            export_path=r'E:\NorMITs Demand\noham\v2_2-EFS_Output\iter0\Matrices\Post-ME Matrices\Compiled OD Matrices',
            matrix_format='od',
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
            year=consts.BASE_YEAR
        )

    if gen_tour_proportions_bool:
        mat_p.generate_tour_proportions(
            od_import=r'E:\NorMITs Demand\noham\v2_2-EFS_Output\iter0\Matrices\Post-ME Matrices\OD Matrices',
            pa_export=r'E:\NorMITs Demand\noham\v2_2-EFS_Output\iter0\Matrices\Post-ME Matrices\PA Matrices',
            tour_proportions_export=r'E:\NorMITs Demand\noham\v2_2-EFS_Output\iter0\Params\Tour Proportions',
            zone_translate_dir=r'Y:\NorMITs Demand\import\zone_translation',
            year=consts.BASE_YEAR,
            m_needed=m_needed,
            ca_needed=ca_needed,
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
        # mat_p.build_24hr_mats(
        #     import_dir=r'E:\NorMITs Demand\noham\v2_2-EFS_Output\iter0\Matrices\Post-ME Matrices\PA Matrices',
        #     export_dir=r'E:\NorMITs Demand\noham\v2_2-EFS_Output\iter0\Matrices\Post-ME Matrices\24hr PA Matrices',
        #     matrix_format='pa',
        #     years_needed=[consts.BASE_YEAR],
        #     m_needed=m_needed,
        #     ca_needed=ca_needed,
        # )

        pa2od.build_od_from_tour_proportions(
            pa_import=r'E:\NorMITs Demand\noham\v2_2-EFS_Output\iter0\Matrices\Post-ME Matrices\24hr PA Matrices',
            od_export=r'E:\NorMITs Demand\noham\v2_2-EFS_Output\iter0\Matrices\Post-ME Matrices\Test OD Matrices',
            tour_proportions_dir=r'E:\NorMITs Demand\noham\v2_2-EFS_Output\iter0\Params\Tour Proportions',
            years_needed=[consts.BASE_YEAR],
            m_needed=m_needed,
            ca_needed=ca_needed,
            process_count=process_count
        )


if __name__ == '__main__':
    main()
