# -*- coding: utf-8 -*-
"""
    Script for running the MidMITs distribution model.

    See Also
    --------
    `run_models.run_distribution_model`
"""

##### IMPORTS #####
# Standard imports
import os
import sys
from typing import Tuple

# Third party imports

# Local imports
sys.path.append("..")
# pylint: disable=import-error,wrong-import-position
import run_distribution_model
import run_mitem
import normits_demand as nd
from normits_demand import pathing
from normits_demand import constants as consts
from normits_demand.distribution import parameters as dist_params
from normits_demand.models import DistributionModel
from normits_demand.pathing.distribution_model import DistributionModelArgumentBuilder
# pylint: enable=import-error,wrong-import-position

##### CONSTANTS #####
INIT_PARAMS_BASE = run_distribution_model.INIT_PARAMS_BASE


##### FUNCTIONS #####
def main() -> None:
    """Run MidMITs distribution model."""
    # Distribution running args
    mitem_iteration_name = run_mitem.mitem_iter
    dm_iteration_name = '9.3.3'
    base_year = 2018
    mode = nd.Mode.CAR
    use_tram = False
    # Distribution model paths
    paths = dist_params.DistributionModelPaths(
        import_home=r"I:\NorMITs Demand\import",
        export_home=r"T:\MidMITs Demand\Distribution Model",
        notem_export_home=run_mitem.mitem_export_home,
        tram_export_home=None,
        cache_path="c:/dm_cache",
        overwrite_cache=False,
    )
    # Trip end export paths
    export_paths = pathing.MiTEMExportPaths(
        path_years=[base_year],
        scenario=consts.SC01_JAM,
        iteration_name=mitem_iteration_name,
        export_home=run_mitem.mitem_export_home,
    )

    params = mode_lookup(
        paths,
        export_paths,
        dm_iteration_name,
        base_year,
        mode,
        use_tram,
        tour_props_version=f"v{mitem_iteration_name}",
    )
    model_runs = dist_params.DistributionModelRuns(
        run_hb=True,
        run_nhb=True,
        run_all=True,
        run_upper_model=False,
        run_lower_model=False,
        run_pa_matrix_reports=False,
        run_pa_to_od=False,
        run_od_matrix_reports=False,
        compile_to_assignment=False,
    )

    run_models(params, model_runs)


def mode_lookup(
    paths: dist_params.DistributionModelPaths,
    export_paths: pathing.NoTEMExportPaths,
    iteration_name: str,
    base_year: int,
    mode: nd.Mode,
    use_tram: bool,
    tour_props_version: str,
) -> dist_params.DistributionModelParameters:
    """Lookup distribution model parameters for a single mode.

    Parameters
    ----------
    paths : dist_params.DistributionModelPaths
        The import, export and cache paths for the distribution model.
    export_paths : pathing.NoTEMExportPaths
        The export paths class for the model being ran, NoTEMExportPaths,
        TramExportPaths or MiTEMExportPaths.
    iteration_name : str
        Name of the distribution model iteration.
    base_year : int
        Base year for the distribution model.
    mode : nd.Mode
        Mode being ran.
    use_tram : bool
        Whether to use tram data or not.
    tour_props_version : str
        Version of the tour proportions to use.

    Returns
    -------
    dist_params.DistributionModelParameters
        All the parameters for running the distribution
        model.

    Raises
    ------
    ValueError
        If given `mode` isn't known.
    """
    # Default values across all modes
    intra_infill = 0.5
    gm_cost = nd.BuiltInCostFunction.LOG_NORMAL.get_cost_function()
    dist_kwargs = {
        nd.DistributionMethod.GRAVITY: {
            'cost_function': gm_cost,
            'target_convergence': 0.9,
            'grav_max_iters': 100,
            'furness_max_iters': 3000,
            'furness_tol': 0.1,
            'calibrate_params': True,
            'estimate_init_params': False
        },
        nd.DistributionMethod.FURNESS3D: {
            'target_convergence': 0.9,
            'outer_max_iters': 50,
            'furness_max_iters': 3000,
            'furness_tol': 0.1,
            'calibrate': True,
        }
    }

    # Mode specific values
    if mode == nd.Mode.CAR:
        zone_systems = dist_params.DistributionZoneSystems(
            nd.get_zoning_system("msoa"),
            nd.get_zoning_system("miham"),
        )
        segmentations = dist_params.DistributionSegmentations(
            nd.get_segmentation_level('hb_p_m7' if use_tram else 'hb_p_m'),
            nd.get_segmentation_level(
                'tms_nhb_p_m7_tp_wday' if use_tram else 'tms_nhb_p_m_tp_wday'
            ),
            nd.get_segmentation_level('hb_p_m_car'),
            nd.get_segmentation_level('tms_nhb_p_m_tp_wday_car'),
            'p_m',
            'p_m_tp',
        )
        upper_cal = dist_params.CalibrationParams(
            'gb', nd.DistributionMethod.GRAVITY, 'gb'
        )
        lower_cal = dist_params.CalibrationParams(
            'north', nd.DistributionMethod.GRAVITY, 'north'
        )
    elif mode == nd.Mode.BUS:
        zone_systems = dist_params.DistributionZoneSystems(
            nd.get_zoning_system("msoa"),
            nd.get_zoning_system("miham"),
        )
        segmentations = dist_params.DistributionSegmentations(
            nd.get_segmentation_level('hb_p_m7' if use_tram else 'hb_p_m'),
            nd.get_segmentation_level(
                'tms_nhb_p_m7_tp_wday' if use_tram else 'tms_nhb_p_m_tp_wday'
            ),
            nd.get_segmentation_level('hb_p_m_bus'),
            nd.get_segmentation_level('tms_nhb_p_m_tp_wday_bus'),
            'p_m',
            'p_m_tp',
        )
        upper_cal = dist_params.CalibrationParams(
            'gb', nd.DistributionMethod.GRAVITY, 'gb'
        )
        lower_cal = dist_params.CalibrationParams(
            'north', nd.DistributionMethod.GRAVITY, 'north'
        )
    elif mode == nd.Mode.TRAIN:
        zone_systems = dist_params.DistributionZoneSystems(
            nd.get_zoning_system('msoa'),
            nd.get_zoning_system('msoa'),
            nd.get_zoning_system('norms'),
        )
        segmentations = dist_params.DistributionSegmentations(
            nd.get_segmentation_level('hb_p_m7_ca' if use_tram else 'hb_p_m_ca'),
            nd.get_segmentation_level(
                'tms_nhb_p_m7_ca_tp_wday' if use_tram else 'tms_nhb_p_m_ca_tp_wday'
            ),
            nd.get_segmentation_level('hb_p_m_ca_rail'),
            nd.get_segmentation_level('tms_nhb_p_m_ca_tp_wday_rail'),
            'p_m_ca',
            'p_m_ca_tp',
        )
        upper_cal = dist_params.CalibrationParams(
            'gb', nd.DistributionMethod.GRAVITY, 'gb'
        )
        lower_cal = dist_params.CalibrationParams(
            'north', nd.DistributionMethod.GRAVITY, 'north'
        )
    elif mode == nd.Mode.TRAM:
        zone_systems = dist_params.DistributionZoneSystems(nd.get_zoning_system('msoa'))
        segmentations = dist_params.DistributionSegmentations(
            nd.get_segmentation_level('hb_p_m7' if use_tram else 'hb_p_m'),
            nd.get_segmentation_level(
                'tms_nhb_p_m7_tp_wday' if use_tram else 'tms_nhb_p_m_tp_wday'
            ),
            nd.get_segmentation_level('hb_p_m_tram'),
            nd.get_segmentation_level('tms_nhb_p_m_tp_wday_tram'),
            'p_m',
            'p_m_tp',
        )
        upper_cal = dist_params.CalibrationParams(
            'north', nd.DistributionMethod.GRAVITY, 'north'
        )
        lower_cal = dist_params.CalibrationParams()
    else:
        raise ValueError("Don't know what mode %s is!" % mode.value)

    return dist_params.DistributionModelParameters(
        paths=paths,
        export_paths=export_paths,
        iteration=iteration_name,
        base_year=base_year,
        mode=mode,
        use_tram=use_tram,
        zone_systems=zone_systems,
        intrazonal_infill=intra_infill,
        segmentations=segmentations,
        upper_calibration=upper_cal,
        lower_calibration=lower_cal,
        gm_cost_function=gm_cost,
        upper_distributor_kwargs=dist_kwargs[upper_cal.method],
        lower_distributor_kwargs=dist_kwargs[lower_cal.method],
        tour_proportions_version=tour_props_version,
    )


def run_models(
    params: dist_params.DistributionModelParameters,
    model_runs: dist_params.DistributionModelRuns,
):
    """Run the distribution model methods defined in `model_runs`.

    TODO: This should be moved to `run_distribution_model` once
    the functionality for tram has been added.

    Parameters
    ----------
    params : dist_params.DistributionModelParameters
        Parameters for running the distribution models.
    model_runs : dist_params.DistributionModelRuns
        Options for which functionality to run.

    Raises
    ------
    NotImplementedError
        When attempting to run with tram data.
    """
    hb_prod, hb_attr, nhb_prod, nhb_attr = build_trip_ends(params)
    dmab_kwargs, dm_kwargs = dist_params.build_dm_kwargs(params)
    # Init params fnames
    upper_kwargs = {
        'zoning': params.zone_systems.upper.name,
        'area': params.upper_calibration.area,
    }
    hb_kwargs = {'trip_origin': 'hb', 'seg': params.segmentations.hb_seg_name}
    nhb_kwargs = {'trip_origin': 'nhb', 'seg': params.segmentations.nhb_seg_name}

    hb_upper_init_params_fname = INIT_PARAMS_BASE.format(**hb_kwargs, **upper_kwargs)
    nhb_upper_init_params_fname = INIT_PARAMS_BASE.format(**nhb_kwargs, **upper_kwargs)

    if params.zone_systems.lower is not None:
        lower_kwargs = {
            'zoning': params.zone_systems.lower.name,
            'area': params.lower_calibration.area,
        }
        hb_lower_init_params_fname = INIT_PARAMS_BASE.format(**hb_kwargs, **lower_kwargs)
        nhb_lower_init_params_fname = INIT_PARAMS_BASE.format(
            **nhb_kwargs, **lower_kwargs
        )
    else:
        lower_kwargs = None
        hb_lower_init_params_fname = None
        nhb_lower_init_params_fname = None

    # ## RUN THE MODEL ## #
    if model_runs.run_hb:
        trip_origin = 'hb'

        arg_builder = DistributionModelArgumentBuilder(
            trip_origin=trip_origin,
            productions=hb_prod,
            attractions=hb_attr,
            running_segmentation=params.segmentations.hb_running_seg,
            upper_init_params_fname=hb_upper_init_params_fname,
            lower_init_params_fname=hb_lower_init_params_fname,
            target_tld_dir=params.segmentations.hb_seg_name,
            **dmab_kwargs,
        )

        hb_distributor = DistributionModel(
            arg_builder=arg_builder,
            compile_zoning_system=params.zone_systems.compile_,
            **dm_kwargs,
            **arg_builder.build_distribution_model_init_args(),
        )

        hb_distributor.run(
            run_all=model_runs.run_all,
            run_upper_model=model_runs.run_upper_model,
            run_lower_model=model_runs.run_lower_model,
            run_pa_matrix_reports=model_runs.run_pa_matrix_reports,
            run_pa_to_od=model_runs.run_pa_to_od,
            run_od_matrix_reports=model_runs.run_od_matrix_reports,
        )

    if model_runs.run_nhb:
        trip_origin = 'nhb'

        arg_builder = DistributionModelArgumentBuilder(
            trip_origin=trip_origin,
            productions=nhb_prod,
            attractions=nhb_attr,
            running_segmentation=params.segmentations.nhb_running_seg,
            upper_init_params_fname=nhb_upper_init_params_fname,
            lower_init_params_fname=nhb_lower_init_params_fname,
            target_tld_dir=params.segmentations.nhb_seg_name,
            **dmab_kwargs,
        )

        nhb_distributor = DistributionModel(
            arg_builder=arg_builder,
            compile_zoning_system=params.zone_systems.compile_,
            **dm_kwargs,
            **arg_builder.build_distribution_model_init_args(),
        )

        nhb_distributor.run(
            run_all=model_runs.run_all,
            run_upper_model=model_runs.run_upper_model,
            run_lower_model=model_runs.run_lower_model,
            run_pa_matrix_reports=model_runs.run_pa_matrix_reports,
            run_pa_to_od=model_runs.run_pa_to_od,
            run_od_matrix_reports=model_runs.run_od_matrix_reports,
        )

    # TODO(BT): Move this into Matrix tools!
    #  Fudged to get this to work for now. Handle this better!
    if model_runs.compile_to_assignment:
        if 'hb_distributor' in locals():
            hb_distributor.compile_to_assignment_format()
        elif 'nhb_distributor' in locals():
            nhb_distributor.compile_to_assignment_format()
        else:
            trip_origin = 'hb'
            arg_builder = DistributionModelArgumentBuilder(
                trip_origin=trip_origin,
                productions=hb_prod,
                attractions=hb_attr,
                running_segmentation=params.segmentations.hb_running_seg,
                upper_init_params_fname=hb_upper_init_params_fname,
                lower_init_params_fname=hb_lower_init_params_fname,
                target_tld_dir=os.path.join(
                    params.upper_calibration.area, params.segmentations.hb_seg_name
                ),
                **dmab_kwargs,
            )

            hb_distributor = DistributionModel(
                arg_builder=arg_builder,
                compile_zoning_system=params.zone_systems.compile_,
                **dm_kwargs,
                **arg_builder.build_distribution_model_init_args(),
            )

            hb_distributor.compile_to_assignment_format()


def build_trip_ends(
    params: dist_params.DistributionModelParameters
) -> Tuple[nd.DVector, nd.DVector, nd.DVector, nd.DVector]:
    """Builds the trip end data based on given `params`.

    TODO: This should be moved to `run_distribution_model` once
    the functionality for tram has been added.

    Parameters
    ----------
    params : dist_params.DistributionModelParameters
        Parameters for running the distribution model.

    Returns
    -------
    nd.DVector
        Home-based productions data.
    nd.DVector
        Home-based attractions data.
    nd.DVector
        Non-home-based productions data.
    nd.DVector
        Non-home-based attractions data.

    Raises
    ------
    NotImplementedError
        When attempting to run with tram data.
    """
    zoning_system = params.zone_systems.upper
    if params.use_tram:
        # TODO(MB) Implement functionality for tram so it is flexible
        # and can be used for MidMITs and NorMITs
        raise NotImplementedError("Not implemented for MidMITs")
    else:
        tem = params.export_paths
        hb_productions_path = (
            tem.hb_production.export_paths.notem_segmented[params.base_year]
        )
        hb_attractions_path = (
            tem.hb_attraction.export_paths.notem_segmented[params.base_year]
        )
        nhb_productions_path = (
            tem.nhb_production.export_paths.notem_segmented[params.base_year]
        )
        nhb_attractions_path = (
            tem.nhb_attraction.export_paths.notem_segmented[params.base_year]
        )

        # TODO(BT): Should we make this a NoTEM output tool?
        base_fname = "%s_%s_%s.pkl"
        paths = {}
        for nm in ('hbp', 'hba', 'nhbp', 'nhba'):
            paths[nm] = os.path.join(
                params.paths.cache_path,
                base_fname % (nm, zoning_system.name, params.mode.value),
            )

    print("Getting the Production/Attraction Vectors...")
    if not os.path.exists(paths["hbp"]) or not os.path.exists(paths["hba"]):
        hb_productions, hb_attractions = run_distribution_model.import_pa(
            production_import_path=hb_productions_path,
            attraction_import_path=hb_attractions_path,
            agg_segmentation=params.segmentations.hb_agg_seg,
            out_segmentation=params.segmentations.hb_running_seg,
            zoning_system=zoning_system,
            trip_origin='hb',
            use_tram=params.use_tram,
        )
        hb_productions.to_pickle(paths["hbp"])
        hb_attractions.to_pickle(paths["hba"])
    else:
        hb_productions = nd.read_pickle(paths["hbp"])
        hb_attractions = nd.read_pickle(paths["hba"])

    if not os.path.exists(paths["nhbp"]) or not os.path.exists(paths["nhba"]):
        nhb_productions, nhb_attractions = run_distribution_model.import_pa(
            production_import_path=nhb_productions_path,
            attraction_import_path=nhb_attractions_path,
            agg_segmentation=params.segmentations.nhb_agg_seg,
            out_segmentation=params.segmentations.nhb_running_seg,
            zoning_system=zoning_system,
            trip_origin='nhb',
            use_tram=params.use_tram,
        )
        nhb_productions.to_pickle(paths["nhbp"])
        nhb_attractions.to_pickle(paths["nhba"])
    else:
        nhb_productions = nd.read_pickle(paths["nhbp"])
        nhb_attractions = nd.read_pickle(paths["nhba"])

    return (
        hb_productions,
        hb_attractions,
        nhb_productions,
        nhb_attractions,
    )


##### MAIN #####
if __name__ == '__main__':
    main()
