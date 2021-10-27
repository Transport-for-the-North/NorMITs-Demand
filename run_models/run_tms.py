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

from normits_demand.pathing.travel_market_synthesiser import TMSExportPaths
from normits_demand.pathing.travel_market_synthesiser import ExternalModelArgumentBuilder
from normits_demand.pathing.travel_market_synthesiser import GravityModelArgumentBuilder

# Constants
base_year = 2018
scenario = consts.SC01_JAM
tms_iteration_name = '9'
tms_import_home = r"I:\NorMITs Demand\import"
tms_export_home = r"E:\NorMITs Demand\TMS"
notem_iteration_name = '4.2'
notem_export_home = r"I:\NorMITs Demand\NoTEM"
notem_cache_path = "E:/"


def main():
    mode = nd.Mode.CAR
    mode = nd.Mode.BUS
    # mode = nd.Mode.TRAIN

    if mode == nd.Mode.CAR:
        zoning_system = nd.get_zoning_system('noham')
        internal_tld_name = 'p_m_standard_bands'
        external_tld_name = 'p_m_large_bands'
        hb_running_seg = nd.get_segmentation_level('hb_p_m_car')
        nhb_running_seg = nd.get_segmentation_level('nhb_p_m_tp_wday_car')
        intrazonal_cost_infill = 0.5
        em_convergence_target = 0.9
        gm_convergence_target = 0.95
        cost_function = 'ln'
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
        cost_function = 'ln'
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
        cost_function = 'ln'
        hb_init_params_fname = 'hb_init_params_p_m_ca.csv'
        nhb_init_params_fname = 'nhb_init_params_p_m_ca_tp.csv'
        hb_cost_type = '24hr'
        nhb_cost_type = 'tp'
    else:
        raise ValueError(
            "Don't know what mode %s is!" % mode.value
        )

    # Need to direct TMS to it's own outputs
    tms_exports = TMSExportPaths(
        year=base_year,
        iteration_name=tms_iteration_name,
        running_mode=mode,
        export_home=tms_export_home,
    )

    em_arg_builder = ExternalModelArgumentBuilder(
        import_home=tms_import_home,
        base_year=base_year,
        scenario=scenario,
        running_mode=mode,
        zoning_system=zoning_system,
        internal_tld_name=internal_tld_name,
        external_tld_name=external_tld_name,
        intrazonal_cost_infill=intrazonal_cost_infill,
        convergence_target=em_convergence_target,
        hb_cost_type=hb_cost_type,
        nhb_cost_type=nhb_cost_type,
        notem_iteration_name=notem_iteration_name,
        notem_export_home=notem_export_home,
        cache_path=notem_cache_path,
    )

    gm_arg_builder = GravityModelArgumentBuilder(
        import_home=tms_import_home,
        target_tld_name=internal_tld_name,
        cost_function=cost_function,
        running_mode=mode,
        zoning_system=zoning_system,
        hb_cost_type=hb_cost_type,
        nhb_cost_type=nhb_cost_type,
        convergence_target=gm_convergence_target,
        hb_init_params_fname=hb_init_params_fname,
        nhb_init_params_fname=nhb_init_params_fname,
        external_model_exports=tms_exports.external_model,
        intrazonal_cost_infill=intrazonal_cost_infill,
    )

    tms = TravelMarketSynthesiser(
        year=base_year,
        running_mode=mode,
        hb_running_segmentation=hb_running_seg,
        nhb_running_segmentation=nhb_running_seg,
        iteration_name=tms_iteration_name,
        zoning_system=zoning_system,
        external_model_arg_builder=em_arg_builder,
        gravity_model_arg_builder=gm_arg_builder,
        export_home=tms_export_home,
        process_count=-2,
    )

    tms.run(
        run_all=False,
        run_external_model=True,
        run_gravity_model=False,
        run_pa_matrix_reports=False,
        run_pa_to_od=False,
        run_od_matrix_reports=False,
    )

    tms.compile_to_assignment_format()


if __name__ == '__main__':
    main()
