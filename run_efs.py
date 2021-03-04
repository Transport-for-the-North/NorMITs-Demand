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

    # Running control
    integrate_dlog = False

    run_base_efs = False
    recreate_productions = True
    recreate_attractions = True
    recreate_nhb_productions = True

    run_bespoke_zones = False
    run_hb_pa_to_od = False
    run_compile_od = False
    run_decompile_post_me = True
    run_future_year_compile_od = False

    # Controls I/O
    scenario = consts.SC00_NTEM
    iter_num = 0
    import_home = "I:/"
    export_home = "E:/"
    model_name = consts.MODEL_NAME

    # ## RUN START ## #
    efs = nd.ExternalForecastSystem(
        iter_num=iter_num,
        model_name=model_name,
        integrate_dlog=integrate_dlog,
        scenario_name=scenario,
        import_home=import_home,
        export_home=export_home,
        verbose=verbose
    )

    if run_base_efs:
        # Generates HB PA matrices
        efs.run(
            recreate_productions=recreate_productions,
            recreate_attractions=recreate_attractions,
            recreate_nhb_productions=recreate_nhb_productions,
            echo_distribution=verbose,
        )

    if run_bespoke_zones:
        # Convert to HB to OD
        efs.pa_to_od(
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

    if run_hb_pa_to_od:
        efs.pa_to_od(
            years_needed=[2050],
            use_bespoke_pa=True,
            overwrite_hb_tp_pa=True,
            overwrite_hb_tp_od=True,
            verbose=verbose
        )

    if run_compile_od:
        # Compiles base year OD matrices
        efs.pre_me_compile_od_matrices(
            year=2050,
            overwrite_aggregated_od=False,
            overwrite_compiled_od=True,
        )

    # TODO: Check Post ME process works for NOHAM
    if run_decompile_post_me:
        # Decompiles post-me base year OD matrices - generates tour
        # proportions in the process
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
