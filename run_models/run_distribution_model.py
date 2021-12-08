# -*- coding: utf-8 -*-
"""
Created on: 07/12/2021
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

from normits_demand.models import DistributionModel


# ## CONSTANTS ## #
# Trip end import args
notem_iteration_name = '9.3'
notem_export_home = r"I:\NorMITs Demand\NoTEM"
tram_export_home = r"I:\NorMITs Demand\Tram"
cache_path = "E:/gb_dist_cache"

# Distribution running args
base_year = 2018
scenario = consts.SC01_JAM
gb_dist_iteration_name = '9.3.2'
gb_dist_import_home = r"I:\NorMITs Demand\import"
gb_dist_export_home = r"E:\NorMITs Demand\GB_Distribution"

# General constants
INIT_PARAMS_BASE = '{trip_origin}_{zoning}_{area}_init_params_{seg}.csv',

def main():
    mode = nd.Mode.CAR
    # mode = nd.Mode.BUS
    # mode = nd.Mode.TRAIN
    # mode = nd.Mode.TRAM

    # Running params
    use_tram = True

    run_hb = True
    run_nhb = False

    run_all = False,
    run_upper_model = True,
    run_lower_model = True,
    run_pa_matrix_reports = False,
    run_pa_to_od = False,
    run_od_matrix_reports = False,

    if mode == nd.Mode.CAR:
        # Define zoning systems
        upper_zoning_name = 'msoa'
        lower_zoning_name = 'noham'

        # Define cost arguments
        intrazonal_cost_infill = 0.5
        hb_cost_type = '24hr'
        nhb_cost_type = 'tp'

        # Define segmentations for trip ends and running
        if use_tram:
            hb_agg_seg = nd.get_segmentation_level('hb_p_m7')
            nhb_agg_seg = nd.get_segmentation_level('tms_nhb_p_m7_tp_wday')
        else:
            hb_agg_seg = nd.get_segmentation_level('hb_p_m')
            nhb_agg_seg = nd.get_segmentation_level('tms_nhb_p_m_tp_wday')
        hb_running_seg = nd.get_segmentation_level('hb_p_m_car')
        nhb_running_seg = nd.get_segmentation_level('tms_nhb_p_m_tp_wday_car')

        # Define kwargs for the distribution tiers
        # TODO(BT): Link segs into segmentation objects.
        #  Need to split out the hb and NHB target TLDs
        upper_calibration_area = 'gb'
        upper_seg = 'p_m'
        upper_method = nd.DistributionMethods.GRAVITY
        upper_convergence_target = 0.9

        lower_calibration_area = 'north'
        lower_seg = 'p_m_tp'
        lower_method = nd.DistributionMethods.GRAVITY
        lower_convergence_target = 0.9

        upper_kwargs = {'zoning': upper_zoning_name, 'area': upper_calibration_area}
        lower_kwargs = {'zoning': lower_zoning_name, 'area': lower_calibration_area}
        hb_kwargs = {'trip_origin': 'hb', 'seg': upper_seg}
        nhb_kwargs = {'trip_origin': 'nhb', 'seg': lower_seg}

        hb_init_params_fname = INIT_PARAMS_BASE.format(**hb_kwargs, **upper_kwargs)
        nhb_init_params_fname = INIT_PARAMS_BASE.format(**nhb_kwargs, *upper_kwargs)
        upper_kwargs = {
            'zoning_system': nd.get_zoning_system(upper_zoning_name),
            'method': upper_method,
            'cost_function': nd.BuiltInCostFunction.LOG_NORMAL.get_cost_function(),
            'convergence_target': upper_convergence_target,
            'hb_init_params_fname': hb_init_params_fname,
            'nhb_init_params_fname': nhb_init_params_fname,
            'hb_target_tld_dir': os.path.join(upper_calibration_area, upper_seg),
            'nhb_target_tld_dir': os.path.join(upper_calibration_area, upper_seg),
        }

        hb_init_params_fname = INIT_PARAMS_BASE.format(**hb_kwargs, **lower_kwargs)
        nhb_init_params_fname = INIT_PARAMS_BASE.format(**nhb_kwargs, *lower_kwargs)
        lower_kwargs = {
            'zoning_system': nd.get_zoning_system(lower_zoning_name),
            'method': lower_method,
            'target_tld_dir': os.path.join(upper_calibration_area, lower_seg),
            'cost_function': nd.BuiltInCostFunction.LOG_NORMAL.get_cost_function(),
            'convergence_target': lower_convergence_target,
            'hb_init_params_fname': hb_init_params_fname,
            'nhb_init_params_fname': nhb_init_params_fname,
            'hb_target_tld_dir': os.path.join(upper_calibration_area, upper_seg),
            'nhb_target_tld_dir': os.path.join(upper_calibration_area, upper_seg),
        }

    else:
        raise ValueError(
            "Don't know what mode %s is!" % mode.value
        )

    # ## GET TRIP ENDS ## #
    hb_productions, hb_attractions, nhb_productions, nhb_attractions = build_trip_ends(
        use_tram=use_tram,
        zoning_system=nd.get_segmentation_level(upper_zoning_name),
        mode=mode,
        hb_agg_seg=hb_agg_seg,
        hb_running_seg=hb_running_seg,
        nhb_agg_seg=nhb_agg_seg,
        nhb_running_seg=nhb_running_seg,
    )

    # ## BUILD MODEL SPECIFIC KWARGS ## #
    arg_builder = DistModelArgBuilder(
        import_home=gb_dist_import_home,
        running_mode=mode,
        hb_running_segmentation=hb_running_seg,
        nhb_running_segmentation=nhb_running_seg,
        hb_cost_type=hb_cost_type,
        nhb_cost_type=nhb_cost_type,
        hb_productions=hb_productions,
        hb_attractions=hb_attractions,
        nhb_productions=nhb_productions,
        nhb_attractions=nhb_attractions,
        intrazonal_cost_infill=intrazonal_cost_infill,
        export_home=gb_dist_export_home,
        upper_kwargs=upper_kwargs,
        lower_kwargs=lower_kwargs,
    )

    if run_hb:
        hb_distributor = DistributionModel(
            year=base_year,
            running_mode=mode,
            running_segmentation=hb_running_seg,
            iteration_name=gb_dist_iteration_name,
            arg_builder=arg_builder,
            export_home=gb_dist_export_home,
            process_count=-2,
        )

        hb_distributor.run(
            run_all=run_all,
            run_upper_model=run_upper_model,
            run_lower_model=run_lower_model,
            run_pa_matrix_reports=run_pa_matrix_reports,
            run_pa_to_od=run_pa_to_od,
            run_od_matrix_reports=run_od_matrix_reports,
        )

    if run_nhb:
        nhb_distributor = DistributionModel(
            year=base_year,
            running_mode=mode,
            running_segmentation=nhb_running_seg,
            iteration_name=gb_dist_iteration_name,
            arg_builder=arg_builder,
            export_home=gb_dist_export_home,
            process_count=-2,
        )

        nhb_distributor.run(
            run_all=run_all,
            run_upper_model=run_upper_model,
            run_lower_model=run_lower_model,
            run_pa_matrix_reports=run_pa_matrix_reports,
            run_pa_to_od=run_pa_to_od,
            run_od_matrix_reports=run_od_matrix_reports,
        )

    # tms.compile_to_assignment_format()

def build_trip_ends(use_tram,
                    zoning_system,
                    mode,
                    hb_agg_seg,
                    hb_running_seg,
                    nhb_agg_seg,
                    nhb_running_seg,
                    ):
    if use_tram:
        tram = nd.pathing.TramExportPaths(
            path_years=[base_year],
            scenario=scenario,
            iteration_name=notem_iteration_name,
            export_home=tram_export_home,
        )
        hb_productions_path = tram.hb_production.export_paths.notem_segmented[base_year]
        hb_attractions_path = tram.hb_attraction.export_paths.notem_segmented[base_year]
        nhb_productions_path = tram.nhb_production.export_paths.notem_segmented[base_year]
        nhb_attractions_path = tram.nhb_attraction.export_paths.notem_segmented[base_year]

        base_fname = "%s_%s_%s.pkl"
        hbp_path = os.path.join(cache_path, base_fname % ('hbp_tram', zoning_system.name, mode.value))
        hba_path = os.path.join(cache_path, base_fname % ('hba_tram', zoning_system.name, mode.value))
        nhbp_path = os.path.join(cache_path, base_fname % ('nhbp_tram', zoning_system.name, mode.value))
        nhba_path = os.path.join(cache_path, base_fname % ('nhba_tram', zoning_system.name, mode.value))

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
            use_tram=use_tram,
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
            use_tram=use_tram,
        )
        nhb_productions.to_pickle(nhbp_path)
        nhb_attractions.to_pickle(nhba_path)
    else:
        nhb_productions = nd.read_pickle(nhbp_path)
        nhb_attractions = nd.read_pickle(nhba_path)

    return (
        hb_productions,
        hb_attractions,
        nhb_productions,
        nhb_attractions,
    )


def import_pa(production_import_path,
              attraction_import_path,
              agg_segmentation,
              out_segmentation,
              zoning_system,
              trip_origin,
              use_tram,
              ) -> Tuple[nd.DVector, nd.DVector]:

    model_name = 'tram' if use_tram else 'notem'

    # Determine the required segmentation
    if trip_origin == 'hb':
        reduce_seg = None
        subset_name = '%s_hb_output_wday'
        subset_seg = nd.get_segmentation_level(subset_name % model_name)
    elif trip_origin == 'nhb':
        reduce_name = '%s_nhb_output_reduced'
        reduce_seg = nd.get_segmentation_level(reduce_name % model_name)
        subset_name = '%s_nhb_output_reduced_wday'
        subset_seg = nd.get_segmentation_level(subset_name % model_name)
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
