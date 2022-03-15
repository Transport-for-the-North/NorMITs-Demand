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
import numpy as np
import pandas as pd

from typing import Tuple

# Third Party

# Local Imports
sys.path.append("..")
import normits_demand as nd
from normits_demand import constants as consts

from normits_demand.models import DistributionModel
from normits_demand.pathing.distribution_model import DistributionModelArgumentBuilder


# ## CONSTANTS ## #
# Trip end import args
notem_iteration_name = '9.3'
tour_props_version = 'v%s' % notem_iteration_name

notem_export_home = r"I:\NorMITs Demand\NoTEM"
tram_export_home = r"I:\NorMITs Demand\Tram"
cache_path = "C:/PW/TfN/dm_cache"

# Distribution running args
base_year = 2018
scenario = consts.SC01_JAM
dm_iteration_name = '9.3.3'
dm_import_home = r"I:\NorMITs Demand\import"
dm_export_home = r"I:\NorMITs Demand\Distribution Model"

# General constants
INIT_PARAMS_BASE = '{trip_origin}_{zoning}_{area}_init_params_{seg}.csv'


def main():
    mode = nd.Mode.CAR
    # mode = nd.Mode.BUS
    # mode = nd.Mode.TRAIN
    # mode = nd.Mode.TRAM

    # Running params
    use_tram = True
    overwrite_cache = False

    calibrate_params = False

    run_hb = True
    run_nhb = False

    run_all = False
    run_upper_model = False
    run_lower_model = False
    run_pa_matrix_reports = True
    run_pa_to_od = False
    run_od_matrix_reports = False
    compile_to_assignment = False

    if mode == nd.Mode.CAR:
        # Define zoning systems
        upper_zoning_system = nd.get_zoning_system('msoa')
        lower_zoning_system = nd.get_zoning_system('noham')
        compile_zoning_system = None

        # Define cost arguments
        intrazonal_cost_infill = 0.5

        # Define segmentations for trip ends and running
        if use_tram:
            hb_agg_seg = nd.get_segmentation_level('hb_p_m7')
            nhb_agg_seg = nd.get_segmentation_level('tms_nhb_p_m7_tp_wday')
        else:
            hb_agg_seg = nd.get_segmentation_level('hb_p_m')
            nhb_agg_seg = nd.get_segmentation_level('tms_nhb_p_m_tp_wday')
        hb_running_seg = nd.get_segmentation_level('hb_p_m_car')
        nhb_running_seg = nd.get_segmentation_level('tms_nhb_p_m_tp_wday_car')

        # Define segments
        hb_seg_name = 'p_m'
        nhb_seg_name = 'p_m_tp'

        # Define target tld dirs
        target_tld_version = 'v1'
        geo_constraint_type = 'trip_OD'

        upper_calibration_area = 'gb'
        upper_calibration_bands = 'dm_highway_bands'
        upper_target_tld_dir = os.path.join(geo_constraint_type, upper_calibration_bands)
        upper_hb_target_tld_dir = os.path.join(upper_target_tld_dir, 'hb_p_m')
        upper_nhb_target_tld_dir = os.path.join(upper_target_tld_dir, 'nhb_p_m_tp')
        upper_model_method = nd.DistributionMethod.GRAVITY
        upper_calibration_zones_fname = None
        upper_calibration_areas = upper_calibration_area
        upper_calibration_naming = None

        # upper_model_method = nd.DistributionMethod.FURNESS3D
        # upper_calibration_zones_fname = 'noham_north_other_rows.pbz2'
        # upper_calibration_areas = {1: 'north', 2: 'gb'}
        # upper_calibration_naming = {1: 'north', 2: 'other'}

        lower_calibration_area = 'north_and_mids'
        lower_calibration_bands = 'dm_highway_bands'
        lower_target_tld_dir = os.path.join(geo_constraint_type, lower_calibration_bands)
        lower_hb_target_tld_dir = os.path.join(lower_target_tld_dir, 'hb_p_m')
        lower_nhb_target_tld_dir = os.path.join(lower_target_tld_dir, 'nhb_p_m_tp')
        lower_model_method = nd.DistributionMethod.GRAVITY
        lower_calibration_zones_fname = None
        lower_calibration_areas = lower_calibration_area
        lower_calibration_naming = None

        gm_cost_function = nd.BuiltInCostFunction.LOG_NORMAL.get_cost_function()

        gravity_kwargs = {
            'cost_function': gm_cost_function,
            'target_convergence': 0.9,
            'grav_max_iters': 100,
            'furness_max_iters': 3000,
            'furness_tol': 0.1,
            'calibrate_params': calibrate_params,
            'estimate_init_params': False,
            'use_perceived_factors': True,
        }

        # Args only work for upper atm!
        furness3d_kwargs = {
            'target_convergence': 0.9,
            'outer_max_iters': 50,
            'furness_max_iters': 3000,
            'furness_tol': 0.1,
            'calibrate': True,
        }

        # Choose the correct kwargs
        gravity = (nd.DistributionMethod.GRAVITY, gravity_kwargs)
        furness3d = (nd.DistributionMethod.FURNESS3D, furness3d_kwargs)
        choice = [gravity, furness3d]

        upper_model_kwargs = [x[1].copy() for x in choice if x[0] == upper_model_method][0]
        lower_model_kwargs = [x[1].copy() for x in choice if x[0] == lower_model_method][0]

    elif mode == nd.Mode.BUS:
        # Define zoning systems
        upper_zoning_system = nd.get_zoning_system('msoa')
        lower_zoning_system = nd.get_zoning_system('noham')
        compile_zoning_system = None

        # Define cost arguments
        intrazonal_cost_infill = 0.5

        # Define segmentations for trip ends and running
        if use_tram:
            hb_agg_seg = nd.get_segmentation_level('hb_p_m7')
            nhb_agg_seg = nd.get_segmentation_level('tms_nhb_p_m7_tp_wday')
        else:
            hb_agg_seg = nd.get_segmentation_level('hb_p_m')
            nhb_agg_seg = nd.get_segmentation_level('tms_nhb_p_m_tp_wday')
        hb_running_seg = nd.get_segmentation_level('hb_p_m_bus')
        nhb_running_seg = nd.get_segmentation_level('tms_nhb_p_m_tp_wday_bus')

        # Define segments
        hb_seg_name = 'p_m'
        nhb_seg_name = 'p_m_tp'

        # Define target tld dirs
        target_tld_version = 'v1'
        geo_constraint_type = 'trip_OD'

        # Define kwargs for the distribution tiers
        upper_calibration_area = 'gb'
        upper_calibration_bands = 'dm_highway_bands'
        upper_target_tld_dir = os.path.join(geo_constraint_type, upper_calibration_bands)
        upper_hb_target_tld_dir = os.path.join(upper_target_tld_dir, 'hb_p_m')
        upper_nhb_target_tld_dir = os.path.join(upper_target_tld_dir, 'nhb_p_m_tp')
        upper_model_method = nd.DistributionMethod.GRAVITY
        upper_calibration_zones_fname = None
        upper_calibration_areas = upper_calibration_area
        upper_calibration_naming = None

        lower_calibration_area = 'north_and_mids'
        lower_calibration_bands = 'dm_highway_bands'
        lower_target_tld_dir = os.path.join(geo_constraint_type, lower_calibration_bands)
        lower_hb_target_tld_dir = os.path.join(lower_target_tld_dir, 'hb_p_m')
        lower_nhb_target_tld_dir = os.path.join(lower_target_tld_dir, 'nhb_p_m_tp')
        lower_model_method = nd.DistributionMethod.GRAVITY
        lower_calibration_zones_fname = None
        lower_calibration_areas = lower_calibration_area
        lower_calibration_naming = None

        gm_cost_function = nd.BuiltInCostFunction.LOG_NORMAL.get_cost_function()

        gravity_kwargs = {
            'cost_function': gm_cost_function,
            'target_convergence': 0.9,
            'grav_max_iters': 100,
            'furness_max_iters': 3000,
            'furness_tol': 0.1,
            'calibrate_params': calibrate_params,
            'estimate_init_params': False,
            'use_perceived_factors': True,
        }

        # Args only work for upper atm!
        furness3d_kwargs = {
            'target_convergence': 0.9,
            'outer_max_iters': 50,
            'furness_max_iters': 3000,
            'furness_tol': 0.1,
            'calibrate': True,
        }

        # Choose the correct kwargs
        gravity = (nd.DistributionMethod.GRAVITY, gravity_kwargs)
        furness3d = (nd.DistributionMethod.FURNESS3D, furness3d_kwargs)
        choice = [gravity, furness3d]

        upper_model_kwargs = [x[1].copy() for x in choice if x[0] == upper_model_method][0]
        lower_model_kwargs = [x[1].copy() for x in choice if x[0] == lower_model_method][0]

    elif mode == nd.Mode.TRAIN:
        # Define zoning systems
        upper_zoning_system = nd.get_zoning_system('msoa')
        lower_zoning_system = nd.get_zoning_system('msoa')
        compile_zoning_system = nd.get_zoning_system('norms')

        # Define cost arguments
        intrazonal_cost_infill = 0.5

        # Define segmentations for trip ends and running
        if use_tram:
            hb_agg_seg = nd.get_segmentation_level('hb_p_m7_ca')
            nhb_agg_seg = nd.get_segmentation_level('tms_nhb_p_m7_ca_tp_wday')
        else:
            hb_agg_seg = nd.get_segmentation_level('hb_p_m_ca')
            nhb_agg_seg = nd.get_segmentation_level('tms_nhb_p_m_ca_tp_wday')
        hb_running_seg = nd.get_segmentation_level('hb_p_m_ca_rail')
        nhb_running_seg = nd.get_segmentation_level('tms_nhb_p_m_ca_tp_wday_rail')

        # Define segments
        hb_seg_name = 'p_m_ca'
        nhb_seg_name = 'p_m_ca_tp'

        # Define target tld dirs
        target_tld_version = 'v1'
        geo_constraint_type = 'trip_OD'

        # Define kwargs for the distribution tiers
        upper_calibration_area = 'gb'
        upper_calibration_bands = 'dm_gb_rail_bands'
        upper_target_tld_dir = os.path.join(geo_constraint_type, upper_calibration_bands)
        upper_hb_target_tld_dir = os.path.join(upper_target_tld_dir, 'hb_p_m_ca')
        upper_nhb_target_tld_dir = os.path.join(upper_target_tld_dir, 'nhb_p_m_ca_tp')
        upper_model_method = nd.DistributionMethod.GRAVITY
        upper_calibration_zones_fname = None
        upper_calibration_areas = upper_calibration_area
        upper_calibration_naming = None

        lower_calibration_area = 'north_and_mids'
        lower_calibration_bands = 'dm_north_rail_bands'
        lower_target_tld_dir = os.path.join(geo_constraint_type, lower_calibration_bands)
        lower_hb_target_tld_dir = os.path.join(lower_target_tld_dir, 'hb_p_m_ca')
        lower_nhb_target_tld_dir = os.path.join(lower_target_tld_dir, 'nhb_p_m_ca_tp')
        lower_model_method = nd.DistributionMethod.GRAVITY
        lower_calibration_zones_fname = None
        lower_calibration_areas = lower_calibration_area
        lower_calibration_naming = None

        gm_cost_function = nd.BuiltInCostFunction.LOG_NORMAL.get_cost_function()

        gravity_kwargs = {
            'cost_function': gm_cost_function,
            'target_convergence': 0.9,
            'grav_max_iters': 100,
            'furness_max_iters': 3000,
            'furness_tol': 0.1,
            'calibrate_params': calibrate_params,
            'estimate_init_params': False,
            'use_perceived_factors': True,
        }

        # Args only work for upper atm!
        furness3d_kwargs = {
            'target_convergence': 0.9,
            'outer_max_iters': 50,
            'furness_max_iters': 3000,
            'furness_tol': 0.1,
            'calibrate': True,
        }

        # Choose the correct kwargs
        gravity = (nd.DistributionMethod.GRAVITY, gravity_kwargs)
        furness3d = (nd.DistributionMethod.FURNESS3D, furness3d_kwargs)
        choice = [gravity, furness3d]

        upper_model_kwargs = [x[1].copy() for x in choice if x[0] == upper_model_method][0]
        lower_model_kwargs = [x[1].copy() for x in choice if x[0] == lower_model_method][0]

    elif mode == nd.Mode.TRAM:
        # Define zoning systems
        upper_zoning_system = nd.get_zoning_system('msoa')
        lower_zoning_system = None
        compile_zoning_system = None

        # Define cost arguments
        intrazonal_cost_infill = 0.5

        # Define segmentations for trip ends and running
        if use_tram:
            hb_agg_seg = nd.get_segmentation_level('hb_p_m7')
            nhb_agg_seg = nd.get_segmentation_level('tms_nhb_p_m7_tp_wday')
        else:
            hb_agg_seg = nd.get_segmentation_level('hb_p_m')
            nhb_agg_seg = nd.get_segmentation_level('tms_nhb_p_m_tp_wday')
        hb_running_seg = nd.get_segmentation_level('hb_p_m_tram')
        nhb_running_seg = nd.get_segmentation_level('tms_nhb_p_m_tp_wday_tram')

        # Define segments
        hb_seg_name = 'p_m'
        nhb_seg_name = 'p_m_tp'

        # Define target tld dirs
        target_tld_version = 'v1'
        geo_constraint_type = 'trip_OD'

        # Define kwargs for the distribution tiers
        upper_calibration_area = 'north_and_mids'
        upper_calibration_bands = 'dm_highway_bands'
        upper_target_tld_dir = os.path.join(geo_constraint_type, upper_calibration_bands)
        upper_hb_target_tld_dir = os.path.join(upper_target_tld_dir, 'hb_p_m')
        upper_nhb_target_tld_dir = os.path.join(upper_target_tld_dir, 'nhb_p_m_tp')
        upper_model_method = nd.DistributionMethod.GRAVITY
        upper_calibration_zones_fname = 'msoa_individual_tram_zones.pbz2'
        upper_calibration_areas = {x: 'north_and_mids' for x in [1, 2, 3]}
        upper_calibration_naming = {1: 'manchester', 2: 'sheffield', 3: 'tyne_and_wear'}

        lower_calibration_area = None
        lower_model_method = None
        lower_calibration_zones_fname = None
        lower_calibration_areas = None
        lower_calibration_naming = None
        lower_hb_target_tld_dir = None
        lower_nhb_target_tld_dir = None

        gm_cost_function = nd.BuiltInCostFunction.LOG_NORMAL.get_cost_function()

        gravity_kwargs = {
            'cost_function': gm_cost_function,
            'target_convergence': 0.9,
            'grav_max_iters': 100,
            'furness_max_iters': 3000,
            'furness_tol': 0.1,
            'calibrate_params': calibrate_params,
            'estimate_init_params': False,
            'use_perceived_factors': True,
        }

        # Args only work for upper atm!
        furness3d_kwargs = {
            'target_convergence': 0.9,
            'outer_max_iters': 50,
            'furness_max_iters': 3000,
            'furness_tol': 0.1,
            'calibrate': True,
        }

        # Choose the correct kwargs
        gravity = (nd.DistributionMethod.GRAVITY, gravity_kwargs)
        furness3d = (nd.DistributionMethod.FURNESS3D, furness3d_kwargs)
        choice = [gravity, furness3d]

        upper_model_kwargs = [x[1].copy() for x in choice if x[0] == upper_model_method][0]
        if lower_model_method is not None:
            lower_model_kwargs = [x[1].copy() for x in choice if x[0] == lower_model_method][0]
        else:
            lower_model_kwargs = None
    else:
        raise ValueError(
            "Don't know what mode %s is!" % mode.value
        )

    # ## GET TRIP ENDS ## #
    hb_productions, hb_attractions, nhb_productions, nhb_attractions = build_trip_ends(
        use_tram=use_tram,
        zoning_system=upper_zoning_system,
        mode=mode,
        hb_agg_seg=hb_agg_seg,
        hb_running_seg=hb_running_seg,
        nhb_agg_seg=nhb_agg_seg,
        nhb_running_seg=nhb_running_seg,
    )

    # ## BUILD ARGUMENTS ## #
    if lower_zoning_system is not None:
        lower_running_zones = lower_zoning_system.internal_zones
    else:
        lower_running_zones = None

    if compile_zoning_system is not None:
        tour_props_zoning_name = compile_zoning_system.name
    elif lower_zoning_system is not None:
        tour_props_zoning_name = lower_zoning_system.name
    else:
        tour_props_zoning_name = upper_zoning_system.name

    # arg builder
    dmab_kwargs = {
        'year': base_year,
        'import_home': dm_import_home,
        'running_mode': mode,
        'target_tld_version': target_tld_version,
        'upper_zoning_system': upper_zoning_system,
        'upper_running_zones': upper_zoning_system.unique_zones,
        'upper_model_method': upper_model_method,
        'upper_model_kwargs': upper_model_kwargs,
        'upper_calibration_zones_fname': upper_calibration_zones_fname,
        'upper_calibration_areas': upper_calibration_areas,
        'upper_calibration_naming': upper_calibration_naming,
        'lower_zoning_system': lower_zoning_system,
        'lower_running_zones': lower_running_zones,
        'lower_model_method': lower_model_method,
        'lower_model_kwargs': lower_model_kwargs,
        'lower_calibration_zones_fname': lower_calibration_zones_fname,
        'lower_calibration_areas': lower_calibration_areas,
        'lower_calibration_naming': lower_calibration_naming,
        'tour_props_version': tour_props_version,
        'tour_props_zoning_name': tour_props_zoning_name,
        'init_params_cols': gm_cost_function.parameter_names,
        'intrazonal_cost_infill': intrazonal_cost_infill,
        'cache_path': cache_path,
        'overwrite_cache': overwrite_cache,
    }

    # Distribution model
    dm_kwargs = {
        'iteration_name': dm_iteration_name,
        'upper_model_method': upper_model_method,
        'upper_distributor_kwargs': None,
        'lower_model_method': lower_model_method,
        'lower_distributor_kwargs': None,
        'export_home': dm_export_home,
        'process_count': -2,
    }

    # Init params fnames
    upper_kwargs = {'zoning': upper_zoning_system.name, 'area': upper_calibration_area}
    hb_kwargs = {'trip_origin': 'hb', 'seg': hb_seg_name}
    nhb_kwargs = {'trip_origin': 'nhb', 'seg': nhb_seg_name}

    hb_upper_init_params_fname = INIT_PARAMS_BASE.format(**hb_kwargs, **upper_kwargs)
    nhb_upper_init_params_fname = INIT_PARAMS_BASE.format(**nhb_kwargs, **upper_kwargs)

    if lower_zoning_system is not None:
        lower_kwargs = {'zoning': lower_zoning_system.name, 'area': lower_calibration_area}
        hb_lower_init_params_fname = INIT_PARAMS_BASE.format(**hb_kwargs, **lower_kwargs)
        nhb_lower_init_params_fname = INIT_PARAMS_BASE.format(**nhb_kwargs, **lower_kwargs)
    else:
        lower_kwargs = None
        hb_lower_init_params_fname = None
        nhb_lower_init_params_fname = None

    # ## RUN THE MODEL ## #
    if run_hb:
        trip_origin = 'hb'

        arg_builder = DistributionModelArgumentBuilder(
            trip_origin=trip_origin,
            productions=hb_productions,
            attractions=hb_attractions,
            running_segmentation=hb_running_seg,
            upper_init_params_fname=hb_upper_init_params_fname,
            lower_init_params_fname=hb_lower_init_params_fname,
            upper_target_tld_dir=upper_hb_target_tld_dir,
            lower_target_tld_dir=lower_hb_target_tld_dir,
            **dmab_kwargs,
        )

        hb_distributor = DistributionModel(
            arg_builder=arg_builder,
            compile_zoning_system=compile_zoning_system,
            **dm_kwargs,
            **arg_builder.build_distribution_model_init_args(),
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
        trip_origin = 'nhb'

        arg_builder = DistributionModelArgumentBuilder(
            trip_origin=trip_origin,
            productions=nhb_productions,
            attractions=nhb_attractions,
            running_segmentation=nhb_running_seg,
            upper_init_params_fname=nhb_upper_init_params_fname,
            lower_init_params_fname=nhb_lower_init_params_fname,
            upper_target_tld_dir=upper_nhb_target_tld_dir,
            lower_target_tld_dir=lower_nhb_target_tld_dir,
            **dmab_kwargs,
        )

        nhb_distributor = DistributionModel(
            arg_builder=arg_builder,
            compile_zoning_system=compile_zoning_system,
            **dm_kwargs,
            **arg_builder.build_distribution_model_init_args(),
        )

        nhb_distributor.run(
            run_all=run_all,
            run_upper_model=run_upper_model,
            run_lower_model=run_lower_model,
            run_pa_matrix_reports=run_pa_matrix_reports,
            run_pa_to_od=run_pa_to_od,
            run_od_matrix_reports=run_od_matrix_reports,
        )

    # TODO(BT): Move this into Matrix tools!
    #  Fudged to get this to work for now. Handle this better!
    if compile_to_assignment:
        if 'hb_distributor' in locals():
            hb_distributor.compile_to_assignment_format()
        elif 'nhb_distributor' in locals():
            nhb_distributor.compile_to_assignment_format()
        else:
            trip_origin = 'hb'
            arg_builder = DistributionModelArgumentBuilder(
                trip_origin=trip_origin,
                productions=hb_productions,
                attractions=hb_attractions,
                running_segmentation=hb_running_seg,
                upper_init_params_fname=hb_upper_init_params_fname,
                lower_init_params_fname=hb_lower_init_params_fname,
                upper_target_tld_dir=upper_hb_target_tld_dir,
                lower_target_tld_dir=lower_hb_target_tld_dir,
                **dmab_kwargs,
            )

            hb_distributor = DistributionModel(
                arg_builder=arg_builder,
                compile_zoning_system=compile_zoning_system,
                **dm_kwargs,
                **arg_builder.build_distribution_model_init_args(),
            )

            hb_distributor.compile_to_assignment_format()


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
        hb_productions.save(hbp_path)
        hb_attractions.save(hba_path)
    else:
        hb_productions = nd.DVector.load(hbp_path)
        hb_attractions = nd.DVector.load(hba_path)

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
        nhb_productions.save(nhbp_path)
        nhb_attractions.save(nhba_path)
    else:
        nhb_productions = nd.DVector.load(nhbp_path)
        nhb_attractions = nd.DVector.load(nhba_path)

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
    prod_dvec = nd.DVector.load(production_import_path)

    # Reduce nhb 11 into 12 if needed
    if reduce_seg is not None:
        prod_dvec = prod_dvec.reduce(out_segmentation=reduce_seg)

    # Convert from ave_week to ave_day
    prod_dvec = prod_dvec.subset(out_segmentation=subset_seg)
    prod_dvec = prod_dvec.convert_time_format('avg_day')

    # Convert zoning and segmentation to desired
    prod_dvec = prod_dvec.aggregate(agg_segmentation)
    prod_dvec = prod_dvec.subset(out_segmentation)
    prod_dvec = prod_dvec.translate_zoning(zoning_system, "population")

    # Reading pickled Dvector
    attr_dvec = nd.DVector.load(attraction_import_path)

    # Reduce nhb 11 into 12 if needed
    if reduce_seg is not None:
        attr_dvec = attr_dvec.reduce(out_segmentation=reduce_seg)

    # Convert from ave_week to ave_day
    attr_dvec = attr_dvec.subset(out_segmentation=subset_seg)
    attr_dvec = attr_dvec.convert_time_format('avg_day')

    # Convert zoning and segmentation to desired
    attr_dvec = attr_dvec.aggregate(agg_segmentation)
    attr_dvec = attr_dvec.subset(out_segmentation)
    attr_dvec = attr_dvec.translate_zoning(zoning_system, "employment")

    return prod_dvec, attr_dvec


def run_pa_reports():
    # PA/OD RUN REPORTS
    # Matrix Trip End totals
    # Sector Reports Dvec style
    # TLD curve
    #   single mile bands - p/m (ca ) segments full matrix

    # Requires CA sectors
    sector_loc = 'Y:/Mobile Data/Processing/MDD_Check/lookups/sector_to_noham_correspondence.csv'
    sector_cor = pd.read_csv(sector_loc,
                             names=['sector', 'zone', 'factor', 'internal'],
                             skiprows=1)

    # Variable names to be replaced
    md = 3
    pp = 1
    yr = 2018
    pa_od = 'pa'
    # TODO: if handling of pa/od
    #tp = 1
    zone_te_list = []
    sec_list = []

    # Import steps
    mat = nd.read_df(path=f"I:/NorMITs Demand/Distribution Model/iter9.3.3/car_and_passenger/Final Outputs/Full PA Matrices/hb_synthetic_{pa_od}_yr{yr}_p{pp}_m{md}.pbz2")
    zones = mat.index

    # Convert to pandas dataframe
    wide_mat = pd.DataFrame(mat,
                            index=zones,
                            columns=zones).reset_index()
    mat = pd.melt(wide_mat,
                  id_vars=['index'],
                  var_name='d_zone',
                  value_name='trip',
                  col_level=0)
    mat = mat.rename(columns={'index': 'o_zone'})

    # Zone tripends
    o_trips = mat.groupby(['o_zone']).agg({'trip': sum}).reset_index()
    d_trips = mat.groupby(['d_zone']).agg({'trip': sum}).reset_index()
    # Join
    zone_te = pd.merge(o_trips,
                       d_trips,
                       left_on=['o_zone'],
                       right_on=['d_zone'])
    zone_te = zone_te[['o_zone',
                       'trip_x',
                       'trip_y']]
    zone_te = zone_te.rename(columns={'o_zone': 'zone', 'trip_x': 'trip_o', 'trip_y': 'trip_d'})

    # Build master tripend list - for dvector
    zone_te['mode'] = md
    zone_te['purp'] = pp
    #zone_te['tp'] = tp
    zone_te_list.append(zone_te)

    # Sector-Sector
    mat_sec = pd.merge(mat,
                       sector_cor,
                       left_on=['o_zone'],
                       right_on=['zone'])
    mat_sec['trip_sec'] = mat_sec['trip'] * mat_sec['factor']
    mat_sec = mat_sec.groupby(['sector', 'd_zone']).agg({'trip_sec': sum}).reset_index()
    mat_sec = mat_sec.rename(columns={'sector': 'o_sec'})

    mat_sec = pd.merge(mat_sec,
                       sector_cor,
                       left_on=['d_zone'],
                       right_on=['zone'])
    mat_sec['mdd_lad'] = mat_sec['trip_sec'] * mat_sec['factor']
    mat_sec = mat_sec.groupby(['o_sec', 'sector']).agg({'trip_sec': sum}).reset_index()
    mat_sec = mat_sec.rename(columns={'sector': 'd_sec'})

    # Build master sector list - for dvector
    mat_sec['mode'] = md
    mat_sec['purp'] = pp
    # zone_te['tp'] = tp
    sec_list.append(mat_sec)

    # Export - openpyxl
    nd.dvector()
    # separate csv outputs + openpyxl


if __name__ == '__main__':
    main()
