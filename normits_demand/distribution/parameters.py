# -*- coding: utf-8 -*-
"""
    Module containing functionality for storing and managing the
    distribution model parameters.
"""

##### IMPORTS #####
# Standard imports
from pathlib import Path
from typing import NamedTuple, Optional, Union, Dict, Any

# Local imports
import normits_demand as nd
from normits_demand import cost


##### CLASSES #####
class DistributionZoneSystems(NamedTuple):
    """Zone systems for running the distribution model."""
    upper: nd.ZoningSystem
    lower: Optional[nd.ZoningSystem] = None
    compile_: Optional[nd.ZoningSystem] = None


class DistributionSegmentations(NamedTuple):
    """Segmentations for running the distribution model."""
    hb_agg_seg: nd.SegmentationLevel
    nhb_agg_seg: nd.SegmentationLevel
    hb_running_seg: nd.SegmentationLevel
    nhb_running_seg: nd.SegmentationLevel
    hb_seg_name: str
    nhb_seg_name: str


class CalibrationParams(NamedTuple):
    """Upper or lower calibration parameters for running the distribution model."""
    area: Optional[str] = None
    method: Optional[nd.DistributionMethod] = None
    areas: Optional[Union[Dict[int, str], str]] = None
    zones_fname: Optional[str] = None
    naming: Optional[str] = None
    hb_target_tld_dir: str = None
    nhb_target_tld_dir: str = None


class DistributionModelPaths(NamedTuple):
    """The import, export and cache paths for the distribution model."""
    import_home: Union[Path, str]
    export_home: Union[Path, str]
    notem_export_home: Union[Path, str]
    tram_export_home: Union[Path, str]

class DistributionModelParameters(NamedTuple):
    """Stores all parameters required for running the distribution model."""
    paths: DistributionModelPaths
    iteration: str
    trip_end_iteration: str
    scenario: nd.Scenario
    base_year: int
    mode: nd.Mode
    use_tram: bool
    zone_systems: DistributionZoneSystems
    intrazonal_infill: float
    segmentations: DistributionSegmentations
    upper_calibration: CalibrationParams
    lower_calibration: CalibrationParams
    gm_cost_function: cost.CostFunction
    upper_model_kwargs: Dict[str, Any]
    lower_model_kwargs: Dict[str, Any]
    tour_proportions_version: str
    target_tld_version: str


class DistributionModelKwargs(NamedTuple):
    """Keyword arguments for distribution model and argument builder classes."""
    dmab_kwargs: Dict[str, Any]
    dm_kwargs: Dict[str, Any]


class DistributionModelRuns(NamedTuple):
    """Options for which parts of the distribution model to run."""
    run_hb: bool
    run_nhb: bool
    run_all: bool
    run_upper_model: bool
    run_lower_model: bool
    run_pa_matrix_reports: bool
    run_pa_to_od: bool
    run_od_matrix_reports: bool
    compile_to_assignment: bool


##### FUNCTIONS #####
def build_dm_kwargs(params: DistributionModelParameters) -> DistributionModelKwargs:
    """Build the keyword argument dictionaries for the distribution model.

    Parameters
    ----------
    params : DistributionModelParameters
        Parameters for this run of the distribution model.

    Returns
    -------
    DistributionModelKwargs
        Keyword argument dictionaries for the distribution model and
        the distribution model argument builder.
    """
    if params.zone_systems.lower is not None:
        lower_running_zones = params.zone_systems.lower.internal_zones
    else:
        lower_running_zones = None
    if params.zone_systems.compile_ is not None:
        tour_props_zoning_name = params.zone_systems.compile_.name
    else:
        tour_props_zoning_name = params.zone_systems.lower.name
    dmab_kwargs = {
        'year': params.base_year,
        'import_home': params.paths.import_home,
        'running_mode': params.mode,
        'target_tld_version': params.target_tld_version,
        'upper_zoning_system': params.zone_systems.upper,
        'upper_running_zones': params.zone_systems.upper.unique_zones,
        'upper_model_method': params.upper_calibration.method,
        'upper_model_kwargs': params.upper_model_kwargs,
        'upper_calibration_zones_fname': params.upper_calibration.zones_fname,
        'upper_calibration_areas': params.upper_calibration.areas,
        'upper_calibration_naming': params.upper_calibration.naming,
        'lower_zoning_system': params.zone_systems.lower,
        'lower_running_zones': lower_running_zones,
        'lower_model_method': params.lower_calibration.method,
        'lower_model_kwargs': params.lower_model_kwargs,
        'lower_calibration_zones_fname': params.lower_calibration.zones_fname,
        'lower_calibration_areas': params.lower_calibration.areas,
        'lower_calibration_naming': params.lower_calibration.naming,
        'tour_props_version': params.tour_proportions_version,
        'tour_props_zoning_name': tour_props_zoning_name,
        'init_params_cols': params.gm_cost_function.parameter_names,
        'intrazonal_cost_infill': params.intrazonal_infill,
    }

    # Distribution model
    dm_kwargs = {
        'iteration_name': params.iteration,
        'upper_model_method': params.upper_calibration.method,
        'upper_distributor_kwargs': None,
        'lower_model_method': params.lower_calibration.method,
        'lower_distributor_kwargs': None,
        'export_home': params.paths.export_home,
        'process_count': -2,
    }
    return DistributionModelKwargs(dmab_kwargs, dm_kwargs)
