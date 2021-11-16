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
import os
import sys

from typing import Tuple

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
tms_iteration_name = '9.2.3'
tms_import_home = r"I:\NorMITs Demand\import"
tms_export_home = r"E:\NorMITs Demand\TMS"
notem_iteration_name = '9.2'
notem_export_home = r"I:\NorMITs Demand\NoTEM"
cache_path = "E:/tms_cache"


def main():
    mode = nd.Mode.CAR
    # mode = nd.Mode.BUS
    # mode = nd.Mode.TRAIN

    use_tram = False

    if mode == nd.Mode.CAR:
        zoning_system = nd.get_zoning_system('noham')
        internal_tld_name = 'p_m_standard_bands'
        external_tld_name = 'p_m_large_bands'
        hb_agg_seg = nd.get_segmentation_level('hb_p_m')
        nhb_agg_seg = nd.get_segmentation_level('nhb_p_m_tp_wday')
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
        hb_agg_seg = nd.get_segmentation_level('hb_p_m')
        nhb_agg_seg = nd.get_segmentation_level('nhb_p_m_tp_wday')
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
        # zoning_system = nd.get_zoning_system('msoa')
        zoning_system = nd.get_zoning_system('norms')
        internal_tld_name = 'p_m_ca_internal_norms'
        external_tld_name = 'p_m_ca_external_norms'
        hb_agg_seg = nd.get_segmentation_level('hb_p_m_ca')
        nhb_agg_seg = nd.get_segmentation_level('nhb_p_m_ca_tp_wday')
        hb_running_seg = nd.get_segmentation_level('hb_p_m_ca_rail')
        nhb_running_seg = nd.get_segmentation_level('nhb_p_m_ca_tp_wday_rail')
        intrazonal_cost_infill = None
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

    # ## BUILD TRIP ENDS ## #
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

    # TODO(BT): Should we make this a NoTEM output tool?
    base_fname = "%s_%s_%s.pkl"
    hbp_path = os.path.join(cache_path, base_fname % ('hbp', zoning_system.name, mode.value))
    hba_path = os.path.join(cache_path, base_fname % ('hba', zoning_system.name, mode.value))
    nhbp_path = os.path.join(cache_path, base_fname % ('nhbp', zoning_system.name, mode.value))
    nhba_path = os.path.join(cache_path, base_fname % ('nhba', zoning_system.name, mode.value))

    print("Getting the Production/Attraction Vectors...")
    if not os.path.exists(hbp_path) or not os.path.exists(hba_path):
        hb_productions, hb_attractions = import_pa(
            production_import_path=hb_productions_path,
            attraction_import_path=hb_attractions_path,
            agg_segmentation=hb_agg_seg,
            out_segmentation=hb_running_seg,
            zoning_system=zoning_system,
            trip_origin='hb',
        )
        hb_productions.to_pickle(hbp_path)
        hb_attractions.to_pickle(hba_path)
    else:
        hb_productions = nd.read_pickle(hbp_path)
        hb_attractions = nd.read_pickle(hba_path)

    if not os.path.exists(nhbp_path) or not os.path.exists(nhba_path):
        nhb_productions, nhb_attractions = import_pa(
            production_import_path=nhb_productions_path,
            attraction_import_path=nhb_attractions_path,
            agg_segmentation=nhb_agg_seg,
            out_segmentation=nhb_running_seg,
            zoning_system=zoning_system,
            trip_origin='nhb',
        )
        nhb_productions.to_pickle(nhbp_path)
        nhb_attractions.to_pickle(nhba_path)
    else:
        nhb_productions = nd.read_pickle(nhbp_path)
        nhb_attractions = nd.read_pickle(nhba_path)

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
        hb_running_segmentation=hb_running_seg,
        nhb_running_segmentation=nhb_running_seg,
        zoning_system=zoning_system,
        hb_cost_type=hb_cost_type,
        nhb_cost_type=nhb_cost_type,
        hb_productions=hb_productions,
        hb_attractions=hb_attractions,
        nhb_productions=nhb_productions,
        nhb_attractions=nhb_attractions,
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
        tms_arg_builder=tms_arg_builder,
        export_home=tms_export_home,
        process_count=-2,
    )

    tms.run(
        run_all=False,
        run_external_model=True,
        run_gravity_model=False,
        run_pa_matrix_reports=False,
        run_pa_to_od=True,
        run_od_matrix_reports=False,
    )

    tms.compile_to_assignment_format()


def import_pa(production_import_path,
              attraction_import_path,
              agg_segmentation,
              out_segmentation,
              zoning_system,
              trip_origin,
              ) -> Tuple[nd.DVector, nd.DVector]:
    """
    This function imports productions and attractions from given paths.

    Parameters
    ----------
    production_import_path:
        Path to import productions from.

    attraction_import_path:
        Path to import attractions from.

    model_zone:
        Type of model zoning system. norms or noham

    trip_origin:
        Trip origin

    Returns
    ----------
    [0] productions:
        Mainland GB productions.

    [1] attractions:
        Mainland GB attractions.
    """
    # Determine the required segmentation
    if trip_origin == 'hb':
        reduce_seg = None
        subset_seg = nd.get_segmentation_level('notem_hb_output_wday')
    elif trip_origin == 'nhb':
        reduce_seg = nd.get_segmentation_level('notem_nhb_output_reduced')
        subset_seg = nd.get_segmentation_level('notem_nhb_output_reduced_wday')
    else:
        raise ValueError("Invalid trip origin")

    # Reading pickled Dvector
    prod_dvec = nd.read_pickle(production_import_path)

    # Reduce nhb 11 into 12 if needed
    if reduce_seg is not None:
        prod_dvec = prod_dvec.reduce(out_segmentation=reduce_seg)

    # Convert from ave_week to ave_day
    prod_dvec = prod_dvec.subset(out_segmentation=subset_seg)
    prod_dvec = prod_dvec.convert_time_format('avg_week')

    # Convert zoning and segmentation to desired
    prod_dvec = prod_dvec.aggregate(agg_segmentation)
    prod_dvec = prod_dvec.subset(out_segmentation)
    prod_dvec = prod_dvec.translate_zoning(zoning_system, "population")

    # Reading pickled Dvector
    attr_dvec = nd.read_pickle(attraction_import_path)

    # Reduce nhb 11 into 12 if needed
    if reduce_seg is not None:
        attr_dvec = attr_dvec.reduce(out_segmentation=reduce_seg)

    # Convert from ave_week to ave_day
    attr_dvec = attr_dvec.subset(out_segmentation=subset_seg)
    attr_dvec = attr_dvec.convert_time_format('avg_week')

    # Convert zoning and segmentation to desired
    attr_dvec = attr_dvec.aggregate(agg_segmentation)
    attr_dvec = attr_dvec.subset(out_segmentation)
    attr_dvec = attr_dvec.translate_zoning(zoning_system, "employment")

    return prod_dvec, attr_dvec


if __name__ == '__main__':
    main()
