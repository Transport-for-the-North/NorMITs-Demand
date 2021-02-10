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


def main():
    verbose = False

    # Running control
    integrate_dlog = False

    run_base_efs = True
    recreate_productions = True
    recreate_attractions = True
    recreate_nhb_productions = False

    run_hb_pa_to_od = False
    run_compile_od = False
    run_decompile_od = False
    run_future_year_compile_od = False

    # Controls I/O
    scenario = consts.SC00_NTEM
    iter_num = 0
    import_home = "Y:/"
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

    # BACKLOG: Properly integrate bespoke zones code
    #  labels: demand merge

    if run_base_efs:
        # Generates HB PA matrices
        efs.run(
            recreate_productions=recreate_productions,
            recreate_attractions=recreate_attractions,
            recreate_nhb_productions=recreate_nhb_productions,
            echo_distribution=verbose,
        )

    if run_hb_pa_to_od:
        # Convert to HB to OD
        efs.pa_to_od(
            years_needed=[2050],
            use_bespoke_pa=True,
            overwrite_hb_tp_pa=True,
            overwrite_hb_tp_od=True,
            echo=verbose
        )

    # TODO: Update Integrated OD2PA codebase
    if run_compile_od:
        # Compiles base year OD matrices
        efs.pre_me_compile_od_matrices(
            year=2050,
            overwrite_aggregated_od=False,
            overwrite_compiled_od=True,
        )

    if run_decompile_od:
        # Decompiles post-me base year OD matrices - generates tour
        # proportions in the process
        efs.generate_post_me_tour_proportions(
            model_name=model_name,
            overwrite_decompiled_od=True,
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
