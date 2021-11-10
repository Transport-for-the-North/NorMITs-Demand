# -*- coding: utf-8 -*-
"""
Created on: 09/09/2021
Updated on:

Original author: Ben Taylor
Last update made by:
Other updates made by:

File purpose:

"""
# Built-Ins
import sys

# Third Party

# Local Imports
sys.path.append("..")
import normits_demand as nd
from normits_demand import constants as consts
from normits_demand.models import TravelMarketSynthesiser

from normits_demand.pathing.travel_market_synthesiser import TMSArgumentBuilder
from normits_demand.pathing.travel_market_synthesiser import TMSExportPaths

# Constants
base_year = 2018
scenario = consts.SC01_JAM
tms_iteration_name = '9.1.1'
tms_import_home = r"I:\NorMITs Demand\import"
tms_export_home = r"E:\NorMITs Demand\TMS"
notem_iteration_name = '4.2'
notem_export_home = r"I:\NorMITs Demand\NoTEM"
notem_cache_path = "E:/"


def main():
    mode = nd.Mode.CAR
    # mode = nd.Mode.BUS
    # mode = nd.Mode.TRAIN

    use_tram = False

    if mode == nd.Mode.CAR:
        zoning_system = nd.get_zoning_system('msoa')
        internal_tld_name = 'p_m_standard_bands'
        external_tld_name = 'p_m_large_bands'
        hb_running_seg = nd.get_segmentation_level('hb_p_m_car')
        nhb_running_seg = nd.get_segmentation_level('nhb_p_m_tp_wday_car')
        intrazonal_cost_infill = 0.5
        em_convergence_target = 0.9
        gm_convergence_target = 0.95
        cost_function = nd.BuiltInCostFunction.LOG_NORMAL.get_cost_function()
        hb_init_params_fname = 'hb_init_params_p_m.csv'
        nhb_init_params_fname = 'nhb_init_params_p_m_tp.csv'
        hb_cost_type = '24hr'
        nhb_cost_type = 'tp'

    elif mode == nd.Mode.BUS:
        zoning_system = nd.get_zoning_system('noham')
        internal_tld_name = 'p_m_standard_bands'
        external_tld_name = 'p_m_large_bands'
        hb_running_seg = nd.get_segmentation_level('hb_p_m_bus')
        nhb_running_seg = nd.get_segmentation_level('nhb_p_m_tp_wday_bus')
        intrazonal_cost_infill = 0.4
        em_convergence_target = 0.8
        gm_convergence_target = 0.85
        cost_function = nd.BuiltInCostFunction.LOG_NORMAL.get_cost_function()
        hb_init_params_fname = 'hb_init_params_p_m.csv'
        nhb_init_params_fname = 'nhb_init_params_p_m_tp.csv'
        hb_cost_type = '24hr'
        nhb_cost_type = 'tp'

    elif mode == nd.Mode.TRAIN:
        zoning_system = nd.get_zoning_system('norms')
        internal_tld_name = 'p_m_ca_internal_norms'
        external_tld_name = 'p_m_ca_external_norms'
        hb_running_seg = nd.get_segmentation_level('hb_p_m_ca_rail')
        nhb_running_seg = nd.get_segmentation_level('nhb_p_m_ca_tp_wday_rail')
        intrazonal_cost_infill = 0.5
        em_convergence_target = 0.9
        gm_convergence_target = 0.95
        cost_function = nd.BuiltInCostFunction.LOG_NORMAL.get_cost_function()
        hb_init_params_fname = 'hb_init_params_p_m_ca.csv'
        nhb_init_params_fname = 'nhb_init_params_p_m_ca_tp.csv'
        hb_cost_type = '24hr'
        nhb_cost_type = 'tp'

    else:
        raise ValueError(
            "Don't know what mode %s is!" % mode.value
        )

    # ## DO WE NEED TO RUN EXTERNAL MODEL? ## #
    if zoning_system.name == 'msoa':
        run_external_model = False
        tms_exports = None
    else:
        run_external_model = True
        tms_exports = TMSExportPaths(
            year=base_year,
            iteration_name=tms_iteration_name,
            running_mode=mode,
            export_home=tms_export_home,
        )

    # ## BUILD TRIP END LOCATIONS ## #
    if use_tram:
        raise NotImplementedError()

    else:
        notem = nd.pathing.NoTEMExportPaths(
            path_years=[base_year],
            scenario=scenario,
            iteration_name=notem_iteration_name,
            export_home=notem_export_home,
        )
        hb_productions_path = notem.hb_production.export_paths.notem_segmented[base_year]
        hb_attractions_path = notem.hb_attraction.export_paths.notem_segmented[base_year]
        nhb_productions_path = notem.nhb_production.export_paths.notem_segmented[base_year]
        nhb_attractions_path = notem.nhb_attraction.export_paths.notem_segmented[base_year]

    # ## BUILD MODEL SPECIFIC KWARGS ## #
    external_kwargs = {
        'internal_tld_name': internal_tld_name,
        'external_tld_name': external_tld_name,
        'convergence_target': em_convergence_target,
    }

    gravity_kwargs = {
        'target_tld_name': internal_tld_name,
        'cost_function': cost_function,
        'convergence_target': gm_convergence_target,
        'hb_init_params_fname': hb_init_params_fname,
        'nhb_init_params_fname': nhb_init_params_fname,
    }

    tms_arg_builder = TMSArgumentBuilder(
        import_home=tms_import_home,
        running_mode=mode,
        zoning_system=zoning_system,
        hb_cost_type=hb_cost_type,
        nhb_cost_type=nhb_cost_type,
        hb_productions_path=hb_productions_path,
        hb_attractions_path=hb_attractions_path,
        nhb_productions_path=nhb_productions_path,
        nhb_attractions_path=nhb_attractions_path,
        intrazonal_cost_infill=intrazonal_cost_infill,
        run_external_model=run_external_model,
        tms_exports=tms_exports,
        external_kwargs=external_kwargs,
        gravity_kwargs=gravity_kwargs,
    )

    tms = TravelMarketSynthesiser(
        year=base_year,
        running_mode=mode,
        hb_running_segmentation=hb_running_seg,
        nhb_running_segmentation=nhb_running_seg,
        iteration_name=tms_iteration_name,
        zoning_system=zoning_system,
        external_model_arg_builder=tms_arg_builder.external_model_arg_builder,
        gravity_model_arg_builder=tms_arg_builder.external_model_arg_builder,
        export_home=tms_export_home,
        process_count=-2,
    )

    tms.run(
        run_all=False,
        run_external_model=False,
        run_gravity_model=True,
        run_pa_matrix_reports=False,
        run_pa_to_od=False,
        run_od_matrix_reports=False,
    )

    tms.compile_to_assignment_format()


if __name__ == '__main__':
    main()
