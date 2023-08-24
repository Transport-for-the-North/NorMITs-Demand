# -*- coding: utf-8 -*-
"""Module for running forecasting processes."""
##### IMPORTS #####
from __future__ import annotations

# Standard imports
import argparse
import dataclasses
import pathlib
import sys
from typing import Union

# Third party imports

# Add parent and current folder to path to make normits_demand importable
sys.path.append("..")
sys.path.append(".")
# Local imports
# pylint: disable=import-error,wrong-import-position
import normits_demand as nd
from normits_demand.models.forecasting import (
    edge_replicant,
    tem_forecast,
    ntem_forecast,
    tempro_trip_ends,
    forecast_cnfg,
)
from normits_demand import logging as nd_log
from normits_demand.reports import ntem_forecast_checks
from normits_demand.utils import timing

# pylint: enable=import-error,wrong-import-position

##### CONSTANTS #####
LOG_FILE = "Forecast.log"
LOG = nd_log.get_logger(nd_log.get_package_logger_name() + ".run_models.run_forecast")


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
    model: forecast_cnfg.ForecastModel, config_path: pathlib.Path, init_logger: bool = True
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

    LOG.info(
        "%s forecasting completed in %s",
        model.value.upper(),
        timing.time_taken(start, timing.current_milli_time()),
    )


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
    if assignment_model in (nd.AssignmentModel.NOHAM, nd.AssignmentModel.MIHAM):
        segmentation = {
            "hb_attractions": "hb_p_m_car",
            "hb_productions": "hb_p_m_car",
            "nhb_attractions": "nhb_p_m_car",
            "nhb_productions": "nhb_p_m_car",
        }
    else:
        raise NotImplementedError(
            "Forecasting only implemented for NoHAM and MiHAM"
            f"not {assignment_model.get_name()}"
        )

    return trip_ends.subset(segmentation)


def tem_forecasting(
    params: Union[forecast_cnfg.TEMForecastParameters, forecast_cnfg.NTEMForecastParameters],
    forecast_model: forecast_cnfg.ForecastModel,
) -> None:
    """Run the NTEM or trip end forecasting.

    Parameters
    ----------
    params : NTEMForecastParameters | TEMForecastParameters
        Parameters for running NTEM forecasting.

    See Also
    --------
    normits_demand.models.ntem_forecast
    """
    if forecast_model == forecast_cnfg.ForecastModel.NTEM:
        if not isinstance(params, forecast_cnfg.NTEMForecastParameters):
            raise TypeError(
                "expected NTEMForecastParameters for "
                f"{forecast_model.value} forecasting not {type(params)}"
            )

        trip_end_name = "TEMPro"
        tripend_data = ntem_forecast.get_tempro_data(
            params.ntem_parameters.data_path,
            [params.base_year, *params.future_years],
            ntem_version=params.ntem_parameters.version,
            ntem_scenario=params.ntem_parameters.scenario,
        )

    elif forecast_model == forecast_cnfg.ForecastModel.TRIP_END:
        if not isinstance(params, forecast_cnfg.TEMForecastParameters):
            raise TypeError(
                "expected TEMForecastParameters for "
                f"{forecast_model.value} forecasting not {type(params)}"
            )

        trip_end_name = f"{params.assignment_model.get_name()} Trip End"
        tripend_data = tem_forecast.read_tripends(
            params.base_year,
            params.future_years,
            params.tripend_path,
            nd.get_zoning_system(params.tem_input_zoning),
        )

    else:
        raise ValueError(f"forecasting for trip end or NTEM only not {forecast_model}")

    if params.output_trip_end_data:
        tripend_data.save(params.export_path / f"{trip_end_name} Data")

    # TODO This bit might break with non-msoa zone system
    tripend_data = model_mode_subset(tripend_data, params.assignment_model)
    tripend_growth = ntem_forecast.tempro_growth(
        tripend_data, params.assignment_model.get_zoning_system(), params.base_year
    )
    if params.output_trip_end_growth:
        tripend_growth.save(params.export_path / f"{trip_end_name} Growth Factors")

    ntem_inputs = ntem_forecast.NTEMImportMatrices(
        params.matrix_import_path, params.base_year, params.assignment_model
    )
    pa_output_folder = params.export_path / "Matrices" / "PA"
    ntem_forecast.grow_all_matrices(ntem_inputs, tripend_growth, pa_output_folder)

    ntem_forecast_checks.pa_matrix_comparison(
        ntem_inputs,
        pa_output_folder,
        tripend_data,
        params.assignment_model.get_mode(),
        params.comparison_zone_systems,
        params.base_year,
    )
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

    ntem_forecast_checks.od_matrix_comparison(
        ntem_inputs.od_matrix_folder,
        compiled_od_path / "PCU",
        params.assignment_model.get_zoning_system().name,
        params.comparison_zone_systems["matrix 1"],
        params.user_classes,
        params.time_periods,
        params.future_years,
    )


##### MAIN #####
if __name__ == "__main__":
    args = ForecastingArguments.parse()
    args.validate()

    try:
        main(args.model, args.config_path)
    except Exception as err:
        LOG.critical("Forecasting error:", exc_info=True)
        raise
