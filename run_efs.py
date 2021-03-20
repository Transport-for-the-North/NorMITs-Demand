# -*- coding: utf-8 -*-
"""
Created on: Wed January 27 12:31:12 2021
Updated on:

Original author: Ben Taylor
Last update made by:
Other updates made by:

File purpose:
Running test runs of EFS
"""

import normits_demand as nd
from normits_demand import efs_constants as consts

from normits_demand.utils import exceptional_growth as eg


def main():
    verbose = False

    # Land Use imports
    land_use_drive = "Y:/"
    land_use_iteration = 'iter3b'

    # Running control
    integrate_dlog = False
    run_pop_emp_comparison = False

    run_base_efs = True
    recreate_productions = False
    recreate_attractions = False
    recreate_nhb_productions = False
    rerun_growth_criteria = True

    run_bespoke_zones = False
    ignore_bespoke_zones = True

    run_pa_to_od = True
    run_compile_od = True
    run_decompile_post_me = False
    run_future_year_compile_od = False

    # Controls I/O
    scenario = consts.SC00_NTEM
    iter_num = '3d'
    import_home = "I:/"
    export_home = "E:/"
    model_name = consts.MODEL_NAME

    # ## RUN START ## #
    efs = nd.ExternalForecastSystem(
        iter_num=iter_num,
        model_name=model_name,
        integrate_dlog=integrate_dlog,
        run_pop_emp_comparison=run_pop_emp_comparison,
        scenario_name=scenario,
        import_home=import_home,
        export_home=export_home,
        land_use_drive=land_use_drive,
        land_use_iteration=land_use_iteration,
        verbose=verbose
    )

    if run_base_efs:
        # Generates HB PA matrices
        efs.run(
            recreate_productions=recreate_productions,
            recreate_attractions=recreate_attractions,
            recreate_nhb_productions=recreate_nhb_productions,
            echo_distribution=verbose,

            apply_growth_criteria=rerun_growth_criteria,
        )

    if run_bespoke_zones:
        # Convert to HB to OD
        efs.old_pa_to_od(
            years_needed=[2018],
            p_needed=consts.ALL_HB_P,
            use_bespoke_pa=False,
            overwrite_hb_tp_pa=True,
            overwrite_hb_tp_od=True,
            verbose=verbose
        )

        eg.adjust_bespoke_zones(
            consts.BESPOKE_ZONES_INPUT_FILE,
            efs.exports,
            efs.model_name,
            base_year=consts.BASE_YEAR_STR,
            recreate_donor=True,
            audit_path=efs.exports["audits"],
        )

    if run_pa_to_od:
        efs.pa_to_od(
            years_needed=[2050],
            use_bespoke_pa=(not ignore_bespoke_zones),
            verbose=verbose
        )

    if run_compile_od:
        # Compiles base year OD matrices
        efs.compile_od_matrices(
            year=2050,
            overwrite_aggregated_od=True,
            overwrite_compiled_od=True,
        )

    if run_decompile_post_me:
        # Decompiles post-me base year matrices
        efs.decompile_post_me(
            overwrite_decompiled_matrices=True,
            overwrite_tour_proportions=True,
        )

    if run_future_year_compile_od:
        # Uses the generated tour proportions to compile Post-ME OD matrices
        # for future years
        efs.compile_future_year_od_matrices(
            overwrite_aggregated_pa=True,
            overwrite_future_year_od=True
        )


if __name__ == '__main__':
    main()
