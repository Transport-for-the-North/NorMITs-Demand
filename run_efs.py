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
    land_use_drive = "I:/"
    by_land_use_iteration = 'iter3b'
    fy_land_use_iteration = 'iter3c'

    # Running control
    integrate_dlog = False
    run_pop_emp_comparison = False
    apply_wfh_adjustments = True

    # Base EFS
    run_base_efs = True
    recreate_productions = False
    recreate_attractions = False
    recreate_nhb_productions = False
    combine_internal_external = False

    # Running options
    run_bespoke_zones = False
    ignore_bespoke_zones = True
    use_elasticity_to_od = True

    # Compiling matrices
    run_pa_to_od = False
    run_compile_mats = False
    run_decompile_post_me = False

    # Controls matrix conversion
    output_years = consts.ALL_YEARS
    # output_years = consts.FUTURE_YEARS

    # Controls I/O
    scenario = consts.SC03_DD
    iter_num = '3i'
    import_home = "I:/"
    export_home = "E:/"
    model_name = consts.MODEL_NAME

    # ## RUN START ## #
    efs = nd.ExternalForecastSystem(
        iter_num=iter_num,
        model_name=model_name,
        integrate_dlog=integrate_dlog,
        run_pop_emp_comparison=run_pop_emp_comparison,
        apply_wfh_adjustments=apply_wfh_adjustments,
        scenario_name=scenario,
        import_home=import_home,
        export_home=export_home,
        land_use_drive=land_use_drive,
        by_land_use_iteration=by_land_use_iteration,
        fy_land_use_iteration=fy_land_use_iteration,
        verbose=verbose
    )

    print("-" * 40, "Running for %s" % model_name, "-" * 40)

    if run_base_efs:
        # Generates HB PA matrices
        efs.run(
            recreate_productions=recreate_productions,
            recreate_attractions=recreate_attractions,
            recreate_nhb_productions=recreate_nhb_productions,
            combine_internal_external=combine_internal_external,
            echo_distribution=verbose,
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
            years_needed=output_years,
            use_bespoke_pa=(not ignore_bespoke_zones),
            use_elasticity_pa=use_elasticity_to_od,
            verbose=verbose
        )

    if run_compile_mats:
        for year in output_years:
            efs.compile_matrices(year=year)

    if run_decompile_post_me:
        # Decompiles post-me base year matrices
        efs.decompile_post_me(
            overwrite_decompiled_matrices=True,
            overwrite_tour_proportions=True,
        )


if __name__ == '__main__':
    main()
