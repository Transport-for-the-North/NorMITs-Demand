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

# Third party imports

# Local imports
sys.path.append("..")
# pylint: disable=import-error,wrong-import-position
import run_distribution_model
import run_mitem
import normits_demand as nd
from normits_demand import converters
from normits_demand.distribution import parameters as dist_params
from normits_demand.models import DistributionModel
from normits_demand.pathing.distribution_model import DistributionModelArgumentBuilder

# pylint: enable=import-error,wrong-import-position

##### CONSTANTS #####


##### FUNCTIONS #####
def main() -> None:
    """Run MidMITs distribution model."""
    # Distribution running args
    mitem_iteration_name = "9.7-COVID"
    dm_iteration_name = f"{mitem_iteration_name}.1"
    base_year = 2021
    mode = nd.Mode.CAR
    use_tram = False
    scenario = nd.Scenario.NTEM
    # Distribution model paths
    paths = dist_params.DistributionModelPaths(
        import_home=r"I:\NorMITs Demand\import",
        export_home=r"T:\MidMITs Demand\Distribution Model",
        notem_export_home=run_mitem.mitem_export_home,
        tram_export_home=None,
    )

    params = mode_lookup(
        paths,
        dm_iteration_name,
        base_year,
        mode,
        use_tram,
        tour_props_version=f"v{mitem_iteration_name}",
        scenario=scenario,
        trip_end_iteration=mitem_iteration_name,
    )
    model_runs = dist_params.DistributionModelRuns(
        run_hb=True,
        run_nhb=True,
        run_all=True,
        run_upper_model=True,
        run_lower_model=True,
        run_pa_matrix_reports=True,
        run_pa_to_od=True,
        run_od_matrix_reports=True,
        compile_to_assignment=True,
    )

    run_models(params, model_runs)


def mode_lookup(
    paths: dist_params.DistributionModelPaths,
    iteration_name: str,
    base_year: int,
    mode: nd.Mode,
    use_tram: bool,
    tour_props_version: str,
    scenario: nd.Scenario,
    trip_end_iteration: str,
) -> dist_params.DistributionModelParameters:
    """Lookup distribution model parameters for a single mode.

    Parameters
    ----------
    paths : dist_params.DistributionModelPaths
        The import, export and cache paths for the distribution model.
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
    target_tld_version = "v1"
    geo_constraint_type = "trip_OD"
    gm_cost = nd.BuiltInCostFunction.LOG_NORMAL.get_cost_function()
    dist_kwargs = {
        nd.DistributionMethod.GRAVITY: {
            "cost_function": gm_cost,
            "target_convergence": 0.9,
            "grav_max_iters": 100,
            "furness_max_iters": 3000,
            "furness_tol": 0.1,
            "calibrate_params": True,
            "estimate_init_params": False,
        },
        nd.DistributionMethod.FURNESS3D: {
            "target_convergence": 0.9,
            "outer_max_iters": 50,
            "furness_max_iters": 3000,
            "furness_tol": 0.1,
            "calibrate": True,
        },
    }

    # Mode specific values
    if mode == nd.Mode.CAR:
        zone_systems = dist_params.DistributionZoneSystems(
            nd.get_zoning_system("msoa"),
            nd.get_zoning_system("miham"),
        )
        segmentations = dist_params.DistributionSegmentations(
            nd.get_segmentation_level("hb_p_m7" if use_tram else "hb_p_m"),
            nd.get_segmentation_level(
                "tms_nhb_p_m7_tp_wday" if use_tram else "tms_nhb_p_m_tp_wday"
            ),
            nd.get_segmentation_level("hb_p_m_car"),
            nd.get_segmentation_level("tms_nhb_p_m_tp_wday_car"),
            "p_m",
            "p_m_tp",
        )
        target_tld_dir = os.path.join(geo_constraint_type, "dm_highway_bands")
        upper_cal = dist_params.CalibrationParams(
            "gb",
            nd.DistributionMethod.GRAVITY,
            "gb",
            hb_target_tld_dir=os.path.join(target_tld_dir, "hb_p_m"),
            nhb_target_tld_dir=os.path.join(target_tld_dir, "nhb_p_m_tp"),
        )
        lower_cal = dist_params.CalibrationParams(
            "north_and_mids",
            nd.DistributionMethod.GRAVITY,
            "north_and_mids",
            hb_target_tld_dir=os.path.join(target_tld_dir, "hb_p_m"),
            nhb_target_tld_dir=os.path.join(target_tld_dir, "nhb_p_m_tp"),
        )
    elif mode == nd.Mode.BUS:
        zone_systems = dist_params.DistributionZoneSystems(
            nd.get_zoning_system("msoa"),
            nd.get_zoning_system("miham"),
        )
        segmentations = dist_params.DistributionSegmentations(
            nd.get_segmentation_level("hb_p_m7" if use_tram else "hb_p_m"),
            nd.get_segmentation_level(
                "tms_nhb_p_m7_tp_wday" if use_tram else "tms_nhb_p_m_tp_wday"
            ),
            nd.get_segmentation_level("hb_p_m_bus"),
            nd.get_segmentation_level("tms_nhb_p_m_tp_wday_bus"),
            "p_m",
            "p_m_tp",
        )
        target_tld_dir = os.path.join(geo_constraint_type, "dm_highway_bands")
        upper_cal = dist_params.CalibrationParams(
            "gb",
            nd.DistributionMethod.GRAVITY,
            "gb",
            hb_target_tld_dir=os.path.join(target_tld_dir, "hb_p_m"),
            nhb_target_tld_dir=os.path.join(target_tld_dir, "nhb_p_m_tp"),
        )
        lower_cal = dist_params.CalibrationParams(
            "north_and_mids",
            nd.DistributionMethod.GRAVITY,
            "north_and_mids",
            hb_target_tld_dir=os.path.join(target_tld_dir, "hb_p_m"),
            nhb_target_tld_dir=os.path.join(target_tld_dir, "nhb_p_m_tp"),
        )
    elif mode == nd.Mode.TRAIN:
        zone_systems = dist_params.DistributionZoneSystems(
            nd.get_zoning_system("msoa"),
            nd.get_zoning_system("miranda"),
        )
        segmentations = dist_params.DistributionSegmentations(
            nd.get_segmentation_level("hb_p_m7_ca" if use_tram else "hb_p_m_ca"),
            nd.get_segmentation_level(
                "tms_nhb_p_m7_ca_tp_wday" if use_tram else "tms_nhb_p_m_ca_tp_wday"
            ),
            nd.get_segmentation_level("hb_p_m_ca_rail"),
            nd.get_segmentation_level("tms_nhb_p_m_ca_tp_wday_rail"),
            "p_m_ca",
            "p_m_ca_tp",
        )
        target_tld_dir = os.path.join(geo_constraint_type, "dm_gb_rail_bands")
        upper_cal = dist_params.CalibrationParams(
            "gb",
            nd.DistributionMethod.GRAVITY,
            "gb",
            hb_target_tld_dir=os.path.join(target_tld_dir, "hb_p_m_ca"),
            nhb_target_tld_dir=os.path.join(target_tld_dir, "nhb_p_m_ca_tp"),
        )
        target_tld_dir = os.path.join(geo_constraint_type, "dm_north_rail_bands")
        lower_cal = dist_params.CalibrationParams(
            "north_and_mids",
            nd.DistributionMethod.GRAVITY,
            "north_and_mids",
            hb_target_tld_dir=os.path.join(target_tld_dir, "hb_p_m_ca"),
            nhb_target_tld_dir=os.path.join(target_tld_dir, "nhb_p_m_ca_tp"),
        )
    elif mode == nd.Mode.TRAM:
        zone_systems = dist_params.DistributionZoneSystems(nd.get_zoning_system("msoa"))
        segmentations = dist_params.DistributionSegmentations(
            nd.get_segmentation_level("hb_p_m7" if use_tram else "hb_p_m"),
            nd.get_segmentation_level(
                "tms_nhb_p_m7_tp_wday" if use_tram else "tms_nhb_p_m_tp_wday"
            ),
            nd.get_segmentation_level("hb_p_m_tram"),
            nd.get_segmentation_level("tms_nhb_p_m_tp_wday_tram"),
            "p_m",
            "p_m_tp",
        )
        target_tld_dir = os.path.join(geo_constraint_type, "dm_highway_bands")
        upper_cal = dist_params.CalibrationParams(
            "north_and_mids",
            nd.DistributionMethod.GRAVITY,
            hb_target_tld_dir=os.path.join(target_tld_dir, "hb_p_m"),
            nhb_target_tld_dir=os.path.join(target_tld_dir, "nhb_p_m_tp"),
        )
        lower_cal = dist_params.CalibrationParams()
    else:
        raise ValueError("Don't know what mode %s is!" % mode.value)

    return dist_params.DistributionModelParameters(
        paths=paths,
        iteration=iteration_name,
        trip_end_iteration=trip_end_iteration,
        scenario=scenario,
        base_year=base_year,
        mode=mode,
        use_tram=use_tram,
        zone_systems=zone_systems,
        intrazonal_infill=intra_infill,
        segmentations=segmentations,
        upper_calibration=upper_cal,
        lower_calibration=lower_cal,
        gm_cost_function=gm_cost,
        upper_model_kwargs=dist_kwargs[upper_cal.method],
        lower_model_kwargs=dist_kwargs[lower_cal.method],
        tour_proportions_version=tour_props_version,
        target_tld_version=target_tld_version,
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
    dmab_kwargs, dm_kwargs = dist_params.build_dm_kwargs(params)
    # Init params fnames
    upper_kwargs = {
        "zoning": params.zone_systems.upper.name,
        "area": params.upper_calibration.area,
    }
    hb_kwargs = {"trip_origin": "hb", "seg": params.segmentations.hb_seg_name}
    nhb_kwargs = {"trip_origin": "nhb", "seg": params.segmentations.nhb_seg_name}

    hb_upper_init_params_fname = run_distribution_model.INIT_PARAMS_BASE.format(
        **hb_kwargs, **upper_kwargs
    )
    nhb_upper_init_params_fname = run_distribution_model.INIT_PARAMS_BASE.format(
        **nhb_kwargs, **upper_kwargs
    )

    if params.zone_systems.lower is not None:
        lower_kwargs = {
            "zoning": params.zone_systems.lower.name,
            "area": params.lower_calibration.area,
        }
        hb_lower_init_params_fname = run_distribution_model.INIT_PARAMS_BASE.format(
            **hb_kwargs, **lower_kwargs
        )
        nhb_lower_init_params_fname = run_distribution_model.INIT_PARAMS_BASE.format(
            **nhb_kwargs, **lower_kwargs
        )
    else:
        lower_kwargs = None
        hb_lower_init_params_fname = None
        nhb_lower_init_params_fname = None

    # ## SETUP TRIP END ARGS ## #
    kwargs = {
        "output_zoning": params.zone_systems.upper,
        "base_year": params.base_year,
        "scenario": params.scenario,
        "notem_iteration_name": params.trip_end_iteration,
        "time_format": nd.core.TimeFormat.AVG_DAY,
    }
    if params.use_tram:
        trip_end_getter = converters.TramToDistributionModel(
            export_home=params.paths.tram_export_home,
            **kwargs,
        )
        te_model_name = "tram"
    else:
        trip_end_getter = converters.NoTEMToDistributionModel(
            export_home=params.paths.notem_export_home,
            **kwargs,
        )
        te_model_name = "notem"

    # ## RUN THE MODEL ## #
    if model_runs.run_hb:
        trip_origin = nd.TripOrigin.HB

        # Build the trip end kwargs
        subset_name = run_distribution_model.HB_SUBSET_SEG_BASE_NAME.format(
            trip_origin=trip_origin.value,
            te_model_name=te_model_name,
        )
        trip_end_kwargs = {
            "reduce_segmentation": None,
            "subset_segmentation": nd.get_segmentation_level(subset_name),
            "aggregation_segmentation": params.segmentations.hb_agg_seg,
            "modal_segmentation": params.segmentations.hb_running_seg,
        }

        arg_builder = DistributionModelArgumentBuilder(
            trip_origin=trip_origin,
            trip_end_getter=trip_end_getter,
            trip_end_kwargs=trip_end_kwargs,
            running_segmentation=params.segmentations.hb_running_seg,
            upper_init_params_fname=hb_upper_init_params_fname,
            lower_init_params_fname=hb_lower_init_params_fname,
            upper_target_tld_dir=params.upper_calibration.hb_target_tld_dir,
            lower_target_tld_dir=params.lower_calibration.hb_target_tld_dir,
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
        trip_origin = nd.TripOrigin.NHB

        # Build the trip end kwargs
        kwargs = {"trip_origin": trip_origin.value, "te_model_name": te_model_name}
        subset_name = run_distribution_model.NHB_SUBSET_SEG_BASE_NAME.format(**kwargs)
        reduce_name = run_distribution_model.REDUCE_SEG_BASE_NAME.format(**kwargs)
        trip_end_kwargs = {
            "reduce_segmentation": nd.get_segmentation_level(reduce_name),
            "subset_segmentation": nd.get_segmentation_level(subset_name),
            "aggregation_segmentation": params.segmentations.nhb_agg_seg,
            "modal_segmentation": params.segmentations.nhb_running_seg,
        }

        arg_builder = DistributionModelArgumentBuilder(
            trip_origin=trip_origin,
            trip_end_getter=trip_end_getter,
            trip_end_kwargs=trip_end_kwargs,
            running_segmentation=params.segmentations.nhb_running_seg,
            upper_init_params_fname=nhb_upper_init_params_fname,
            lower_init_params_fname=nhb_lower_init_params_fname,
            upper_target_tld_dir=params.upper_calibration.nhb_target_tld_dir,
            lower_target_tld_dir=params.lower_calibration.nhb_target_tld_dir,
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
        if "hb_distributor" in locals():
            hb_distributor.compile_to_assignment_format()
        elif "nhb_distributor" in locals():
            nhb_distributor.compile_to_assignment_format()
        else:
            trip_origin = nd.TripOrigin.HB
            arg_builder = DistributionModelArgumentBuilder(
                trip_origin=trip_origin,
                trip_end_getter=trip_end_getter,
                trip_end_kwargs=dict(),
                running_segmentation=params.segmentations.hb_running_seg,
                upper_init_params_fname=hb_upper_init_params_fname,
                lower_init_params_fname=hb_lower_init_params_fname,
                upper_target_tld_dir=params.upper_calibration.hb_target_tld_dir,
                lower_target_tld_dir=params.lower_calibration.hb_target_tld_dir,
                **dmab_kwargs,
            )

            hb_distributor = DistributionModel(
                arg_builder=arg_builder,
                compile_zoning_system=params.zone_systems.compile_,
                **dm_kwargs,
                **arg_builder.build_distribution_model_init_args(),
            )

            hb_distributor.compile_to_assignment_format()


##### MAIN #####
if __name__ == "__main__":
    main()
