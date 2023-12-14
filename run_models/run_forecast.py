# -*- coding: utf-8 -*-
"""Module for running forecasting processes."""
##### IMPORTS #####
from __future__ import annotations

# Standard imports
import argparse
import dataclasses
import pathlib
import re
import sys
import warnings
from typing import Union

import numpy as np
import pandas as pd

# Third party imports

# Add parent and current folder to path to make normits_demand importable
sys.path.append("..")
sys.path.append(".")
# Local imports
# pylint: disable=import-error,wrong-import-position
import normits_demand as nd
from normits_demand import distribution
from normits_demand import logging as nd_log
from normits_demand.models.forecasting import (
    edge_replicant,
    forecast_cnfg,
    ntem_forecast,
    tem_forecast,
    tempro_trip_ends,
)
from normits_demand.reports import ntem_forecast_checks
from normits_demand.utils import timing

# pylint: enable=import-error,wrong-import-position

##### CONSTANTS #####

LOG_FILE = "Forecast.log"
LOG = nd_log.get_logger(nd_log.get_package_logger_name() + ".run_models.run_forecast")
_MATRIX_SEGMENTATION = {"hb": "hb_p_m_car", "nhb": "nhb_p_m_car"}


##### CLASSES #####
@dataclasses.dataclass
class ForecastingArguments:
    """Command line arguments for running forecasting.

    Attributes
    ----------
    model: forecast_cnfg.ForecastModel
        Forecasting model to run.
    config_path: pathlib.Path
        Path to config file.
    """

    model: forecast_cnfg.ForecastModel
    config_path: pathlib.Path

    @classmethod
    def parse(cls) -> ForecastingArguments:
        """Parse command line arguments."""
        parser = argparse.ArgumentParser(
            description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        parser.add_argument(
            "model",
            type=str.lower,
            help="forecasting model to run",
            choices=forecast_cnfg.ForecastModel.values_to_list(),
        )
        parser.add_argument(
            "config",
            type=pathlib.Path,
            help="path to config file containing parameters",
        )

        parsed_args = parser.parse_args()
        return ForecastingArguments(
            forecast_cnfg.ForecastModel(parsed_args.model), parsed_args.config
        )

    def validate(self) -> None:
        """Raise error if any arguments are invalid."""
        if not self.config_path.is_file():
            raise FileNotFoundError(f"config file doesn't exist: {self.config_path}")


##### FUNCTIONS #####
def main(
    model: forecast_cnfg.ForecastModel,
    config_path: pathlib.Path,
    init_logger: bool = True,
):
    """Main function for running forecasting models.

    Loads config file and runs `model`.

    Parameters
    ----------
    model : forecast_cnfg.ForecastModel
        Forecasting model to run.
    config_path : pathlib.Path
        Config file containing input parameters.
    init_logger : bool, default True
        Whether, or not, to initialise the logger.

    Raises
    ------
    NotImplementedError
        If a `model` other than TRIP_END is given.
    """
    start = timing.current_milli_time()

    if model == forecast_cnfg.ForecastModel.TRIP_END:
        params = forecast_cnfg.TEMForecastParameters.load_yaml(config_path)
    elif model == forecast_cnfg.ForecastModel.NTEM:
        params = forecast_cnfg.NTEMForecastParameters.load_yaml(config_path)
    elif model == forecast_cnfg.ForecastModel.EDGE:
        params = forecast_cnfg.EDGEParameters.load_yaml(config_path)
    elif model == forecast_cnfg.ForecastModel.DLOG:
        params = forecast_cnfg.DLOGForecastParameters.load_yaml(config_path)
    else:
        raise NotImplementedError(f"forecasting not implemented for {model.value}")

    if params.export_path.exists():
        msg = "export folder already exists: %s"
    else:
        params.export_path.mkdir(parents=True)
        msg = "created export folder: %s"
    if init_logger:
        # Add log file output to main package logger
        nd_log.get_logger(
            nd_log.get_package_logger_name(),
            params.export_path / LOG_FILE,
            f"Running {model.value.upper()} forecast",
        )
    LOG.info(msg, params.export_path)
    LOG.info("Input parameters:\n%s", params.to_yaml())

    output_params_file = params.export_path / "forecasting_parameters.yml"
    params.save_yaml(output_params_file)
    LOG.info("Saved input parameters to %s", output_params_file)

    if model in (forecast_cnfg.ForecastModel.TRIP_END, forecast_cnfg.ForecastModel.NTEM):
        tem_forecasting(params, model)
    elif model == forecast_cnfg.ForecastModel.EDGE:
        edge_replicant.run_edge_growth(params)
    elif model == forecast_cnfg.ForecastModel.DLOG:
        dlog_forecasting(params)
    else:
        raise ValueError(f"unknown forecast model {model}")

    LOG.info(
        "%s forecasting completed in %s",
        model.value.upper(),
        timing.time_taken(start, timing.current_milli_time()),
    )


# TODO(MB) Create normits_demand\models\forecasting\forecasting.py module
# and move all functions below there, to simplfy this front-end script
def model_mode_subset(
    trip_ends: tempro_trip_ends.TEMProTripEnds,
    assignment_model: nd.AssignmentModel,
) -> tempro_trip_ends.TEMProTripEnds:
    """Get subset of `trip_ends` segmentation for specific `model_name`.

    Parameters
    ----------
    trip_ends : tempro_trip_ends.TEMProTripEnds
        Trip end data, which has segmentation split by mode.
    assignment_model : nd.AssignmentModel
        Assignment model being ran, currently only works for
        NoHAM or MiHAM.

    Returns
    -------
    tempro_trip_ends.TEMProTripEnds
        Trip end data at new segmentation.

    Raises
    ------
    NotImplementedError
        If any `assignment_model` other than NoHAM or MiHAM is given.
    """
    car_modes = list(filter(lambda x: x.get_mode() == nd.Mode.CAR, nd.AssignmentModel))
    if assignment_model in car_modes:
        segmentation = {
            "hb_attractions": "hb_p_m_car",
            "hb_productions": "hb_p_m_car",
            "nhb_attractions": "nhb_p_m_car",
            "nhb_productions": "nhb_p_m_car",
        }
    else:
        raise NotImplementedError(
            "Forecasting only implemented for "
            + ", ".join(i.value for i in car_modes)
            + f" not {assignment_model.value}"
        )

    return trip_ends.subset(segmentation)


def _normal_spaces(text: str) -> str:
    """Remove multiple spaces between words and strip ends."""
    return re.sub(r"\s+", " ", text).strip()


def _get_trip_end_data(
    forecast_model: forecast_cnfg.ForecastModel,
    data_params: Union[forecast_cnfg.TEMDataParameters, forecast_cnfg.NTEMDataParameters],
    params: forecast_cnfg.ForecastParameters,
    name_suffix: str = "",
) -> tuple[tempro_trip_ends.TEMProTripEnds, str]:
    """Load the forecast trip end data."""
    if forecast_model == forecast_cnfg.ForecastModel.NTEM:
        if not isinstance(data_params, forecast_cnfg.NTEMDataParameters):
            raise TypeError(
                "expected NTEMForecastParameters for "
                f"{forecast_model.value} forecasting not {type(data_params)}"
            )

        trip_end_name = _normal_spaces(f"TEMPro {name_suffix}")
        tripend_data = ntem_forecast.get_tempro_data(
            data_params.data_path,
            [params.base_year, *params.future_years],
            ntem_version=data_params.version,
            ntem_scenario=data_params.scenario,
        )

    elif forecast_model == forecast_cnfg.ForecastModel.TRIP_END:
        if not isinstance(data_params, forecast_cnfg.TEMDataParameters):
            raise TypeError(
                "expected TEMForecastParameters for "
                f"{forecast_model.value} forecasting not {type(data_params)}"
            )

        trip_end_name = _normal_spaces(
            f"{params.assignment_model.get_name()} {name_suffix} Trip End"
        )
        tripend_data = tem_forecast.read_tripends(
            params.base_year,
            params.future_years,
            data_params.tripend_path,
            nd.get_zoning_system(data_params.tem_input_zoning),
        )

    else:
        raise ValueError(f"forecasting for trip end or NTEM only not {forecast_model}")

    if params.output_trip_end_data:
        tripend_data.save(params.export_path / f"{trip_end_name} Data")

    return tripend_data, trip_end_name


def _matrix_conversions(
    ntem_inputs: ntem_forecast.NTEMImportMatrices,
    pa_output_folder: pathlib.Path,
    tripend_data: tempro_trip_ends.TEMProTripEnds,
    params: forecast_cnfg.ForecastParameters,
) -> None:
    """Convert PA matrices to OD and to compiled highway and rail."""
    try:
        ntem_forecast_checks.pa_matrix_comparison(
            ntem_inputs,
            pa_output_folder,
            tripend_data,
            params.assignment_model.get_mode(),
            params.comparison_zone_systems,
            params.base_year,
        )
    except Exception:
        LOG.error("Error performing PA matrix comparison", exc_info=True)

    od_folder = pa_output_folder.with_name("OD")
    ntem_forecast.convert_to_od(
        pa_output_folder,
        od_folder,
        params.base_year,
        params.future_years,
        params.assignment_model.get_mode().get_mode_values(),
        {"hb": params.hb_purposes_needed, "nhb": params.nhb_purposes_needed},
        params.pa_to_od_factors,
        params.export_path,
        post_me_tours=False,
    )

    # Compile to output formats
    ntem_forecast.compile_highway_for_rail(
        pa_output_folder,
        params.future_years,
        params.assignment_model.get_mode().get_mode_values(),
    )
    compiled_od_path = ntem_forecast.compile_highway(
        od_folder,
        params.future_years,
        params.car_occupancies_path,
    )

    try:
        ntem_forecast_checks.od_matrix_comparison(
            ntem_inputs.od_matrix_folder,
            compiled_od_path / "PCU",
            params.assignment_model.get_zoning_system().name,
            params.comparison_zone_systems["matrix 1"],
            params.user_classes,
            params.time_periods,
            params.future_years,
        )
    except Exception:
        LOG.error("Error performing OD matrix comparison", exc_info=True)


def _trip_end_grow_matrices(
    params: forecast_cnfg.ForecastParameters,
    tripend_data: tempro_trip_ends.TEMProTripEnds,
    growth_name: str,
    pa_output_folder: pathlib.Path,
) -> ntem_forecast.NTEMImportMatrices:
    """Grow matrices based on trip end growth."""
    tripend_data = model_mode_subset(tripend_data, params.assignment_model)
    tripend_growth = ntem_forecast.tempro_growth(
        tripend_data, params.assignment_model.get_zoning_system(), params.base_year
    )
    if params.output_trip_end_growth:
        tripend_growth.save(params.export_path / f"{growth_name} Growth Factors")

    matrix_paths = ntem_forecast.NTEMImportMatrices(
        params.matrix_import_path, params.base_year, params.assignment_model
    )
    ntem_forecast.grow_all_matrices(matrix_paths, tripend_growth, pa_output_folder)

    return matrix_paths


def tem_forecasting(
    params: Union[forecast_cnfg.TEMForecastParameters, forecast_cnfg.NTEMForecastParameters],
    forecast_model: forecast_cnfg.ForecastModel,
) -> None:
    """Run the NTEM or trip end forecasting.

    Parameters
    ----------
    params : NTEMForecastParameters | TEMForecastParameters
        Parameters for running NTEM forecasting.
    forecast_model : ForecastModel
        Forecasting model being ran.

    See Also
    --------
    normits_demand.models.ntem_forecast
    """
    tripend_data, trip_end_name = _get_trip_end_data(
        forecast_model, params.data_parameters, params
    )

    pa_output_folder = params.export_path / "Matrices" / "PA"
    matrix_paths = _trip_end_grow_matrices(
        params, tripend_data, trip_end_name, pa_output_folder
    )

    _matrix_conversions(
        ntem_inputs=matrix_paths,
        pa_output_folder=pa_output_folder,
        tripend_data=tripend_data,
        params=params,
    )


def _development_gravity_distribute(
    params: forecast_cnfg.ForecastParameters,
    year: int,
    export_folder: pathlib.Path,
    trip_ends: tempro_trip_ends.TEMProTripEnds,
    matrix_segmentations: dict[str, nd.SegmentationLevel],
    cost_matrices: dict[str, pd.DataFrame],
    target_cost_distributions: dict[str, pd.DataFrame],
) -> None:
    """Calibrate gravity model to target distribution and output development matrices."""
    LOG.info(
        "Producing development forecast matrices for %s, saved to '%s'", year, export_folder
    )
    zoning = params.assignment_model.get_zoning_system()

    for trip_origin in ("hb", "nhb"):
        distributor = distribution.DistributionMethod.GRAVITY.get_distributor(
            year=year,
            running_mode=params.assignment_model.get_mode(),
            trip_origin=trip_origin,
            zoning_system=zoning,
            running_zones=zoning.unique_zones,
            export_home=export_folder,
        )

        dvecs: dict[str, nd.DVector] = {
            i: getattr(trip_ends, f"{trip_origin}_{i}") for i in ("productions", "attractions")
        }

        distributor.distribute(
            **{k: v.to_df() for k, v in dvecs.items()},
            running_segmentation=matrix_segmentations[trip_origin],
            # Use the same costs and TLD for each segment
            cost_matrices={1: cost_matrices},
            calibration_matrix=np.ones_like(list(cost_matrices.values())[0]),
            target_cost_distributions={1: target_cost_distributions},
            calibration_naming={},
        )


def _read_matrix(path: pathlib.Path, zoning: nd.ZoningSystem) -> pd.DataFrame:
    """Read matrix file, assumed to be in square format."""
    data = pd.read_csv(path, index_col=0)
    data.columns = pd.to_numeric(data.columns, downcast="unsigned")
    data.index = pd.to_numeric(data.index, downcast="unsigned")

    if not data.columns.equals(data.index):
        raise ValueError(f"matrix ({path.name}) columns and index aren't equal")

    missing = zoning.unique_zones[~np.isin(zoning.unique_zones, data.columns)]
    if len(missing) > 0:
        raise ValueError(
            f"{len(missing)} zones missing from matrix ({path.name}): "
            + ", ".join(str(i) for i in missing)
        )

    return data


def _load_cost_matrices(
    segmentation: dict[str, nd.SegmentationLevel],
    zoning: nd.ZoningSystem,
    folder: pathlib.Path,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Load cost matrices for each segment.

    Warning: Currently the same cost is used for all segments.
    """
    LOG.info("Loading cost matrices from '%s'")
    warnings.warn("currently a single cost matrix is used for all segments")
    # TODO(MB) Load different costs for each segment if available
    data = _read_matrix(folder, zoning)

    costs = {}
    for trip_origin, seg_level in segmentation.items():
        costs[trip_origin] = {}

        for seg_params in seg_level:
            costs[trip_origin][seg_level.get_segment_name(seg_params)] = data.copy()

    return costs


def _load_cost_distributions(
    segmentation: dict[str, nd.SegmentationLevel], folder: pathlib.Path
) -> dict[str, dict[str, pd.DataFrame]]:
    """Load cost distributions by trip origin and segment."""
    LOG.info("Loading cost distributions from '%s'", folder)
    distributions = {}
    for trip_origin, seg_level in segmentation.items():
        LOG.info(
            "%s cost distributions at segmentation '%s'", trip_origin.upper(), seg_level.name
        )
        distributions[trip_origin] = {}

        for seg_params in seg_level:
            path = folder / seg_level.generate_file_name_from_template(
                f"{trip_origin}_tlb_{{segment_params}}.csv", seg_params
            )

            data = pd.read_csv(
                path, usecols=["lower", "upper", "ave_km", "trips", "band_share"], dtype=float
            )

            distributions[trip_origin][seg_level.get_segment_name(seg_params)] = data

    return distributions


def _combine_development_matrices(
    background_folder: pathlib.Path,
    developments_folder: pathlib.Path,
    years: list[int],
    matrix_segmentation: dict[str, nd.SegmentationLevel],
    zoning: nd.ZoningSystem,
    export_folder: pathlib.Path,
) -> None:
    """Add development matrices onto background and output."""
    LOG.info(
        "Combining background growth matrices from '%s' "
        "with development matrices from '%s'",
        background_folder.name,
        developments_folder.name,
    )

    for year in years:
        for trip_origin, seg_level in matrix_segmentation.items():
            for seg_params in seg_level:
                filename = seg_level.generate_file_name(
                    seg_params, trip_origin=trip_origin, year=year, csv=True
                )

                background = _read_matrix(background_folder / filename, zoning)
                developments = _read_matrix(developments_folder / filename, zoning)

                combined = background + developments

                out_path = export_folder / filename
                combined.to_csv(out_path)
                LOG.info("Written: '%s'", out_path)


def dlog_forecasting(params: forecast_cnfg.DLOGForecastParameters) -> None:
    # TODO Test and fix any issues with dlog forecasting
    warnings.warn("dlog forecasting function is WIP and needs testing")
    if isinstance(params.background_trip_end_parameters, forecast_cnfg.TEMDataParameters):
        background_model = forecast_cnfg.ForecastModel.TRIP_END
    elif isinstance(params.background_trip_end_parameters, forecast_cnfg.NTEMDataParameters):
        background_model = forecast_cnfg.ForecastModel.NTEM
    else:
        raise TypeError(
            "unexpected background_trip_end_parameters type: "
            f"{type(params.background_trip_end_parameters)}"
        )

    # Create background growth matrices
    background_trip_ends, background_name = _get_trip_end_data(
        background_model,
        params.background_trip_end_parameters,
        params,
        name_suffix="Background",
    )
    background_output_folder = (
        params.export_path / "Matrices" / "PA" / f"Background {background_name} Growth"
    )
    base_matrix_paths = _trip_end_grow_matrices(
        params,
        background_trip_ends,
        f"{background_name} Background",
        background_output_folder,
    )

    # Developments growth matrix
    dlog_trip_ends, _ = _get_trip_end_data(
        forecast_cnfg.ForecastModel.NTEM,
        params.dlog_trip_end_parameters,
        params,
        name_suffix="Developments",
    )

    cost_matrices = _load_cost_matrices(
        _MATRIX_SEGMENTATION,
        params.assignment_model.get_zoning_system(),
        params.cost_matrix_path,
    )
    cost_distributions = _load_cost_distributions(
        _MATRIX_SEGMENTATION, params.cost_distribution_folder
    )

    pa_matrix_folder = params.export_path / r"Matrices\PA"
    dev_matrix_folder = params.export_path / "Development Forecast Matrices"

    for year in params.future_years:
        _development_gravity_distribute(
            params=params,
            year=year,
            export_folder=dev_matrix_folder,
            trip_ends=dlog_trip_ends,
            matrix_segmentations=_MATRIX_SEGMENTATION,
            cost_matrices=cost_matrices,
            target_cost_distributions=cost_distributions,
        )

    _combine_development_matrices(
        background_folder=background_output_folder,
        developments_folder=dev_matrix_folder,
        years=params.future_years,
        matrix_segmentation=_MATRIX_SEGMENTATION,
        zoning=params.assignment_model.get_zoning_system(),
        export_folder=pa_matrix_folder,
    )

    _matrix_conversions(
        ntem_inputs=base_matrix_paths,
        pa_output_folder=pa_matrix_folder,
        tripend_data=background_trip_ends,
        params=params,
    )


##### MAIN #####
if __name__ == "__main__":
    args = ForecastingArguments.parse()
    args.validate()

    try:
        main(args.model, args.config_path)
    except Exception:
        LOG.critical("Forecasting error:", exc_info=True)
        raise
