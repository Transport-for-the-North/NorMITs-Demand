# -*- coding: utf-8 -*-
"""Module for running forecasting processes."""
##### IMPORTS #####
from __future__ import annotations

# Standard imports
import argparse
import dataclasses
import os
import pathlib
import sys
from pathlib import Path

# Third party imports

# Add parent and current folder to path to make normits_demand importable
sys.path.append("..")
sys.path.append(".")
# Local imports
# pylint: disable=import-error,wrong-import-position
import normits_demand as nd
from normits_demand.models import mitem_forecast, ntem_forecast, tempro_trip_ends
from normits_demand.models.forecasting import forecast_cnfg
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
def model_mode_subset(
    trip_ends: tempro_trip_ends.TEMProTripEnds,
    model_name: str,
) -> tempro_trip_ends.TEMProTripEnds:
    """Get subset of `trip_ends` segmentation for specific `model_name`.

    Parameters
    ----------
    trip_ends : tempro_trip_ends.TEMProTripEnds
        Trip end data, which has segmentation split by
        mode.
    model_name : str
        Name of the model being ran, currently only
        works for 'noham'.

    Returns
    -------
    tempro_trip_ends.TEMProTripEnds
        Trip end data at new segmentation.

    Raises
    ------
    NotImplementedError
        If any `model_name` other than 'noham' is
        given.
    """
    model_name = model_name.lower().strip()
    if model_name == "noham" or model_name == "miham":
        segmentation = {
            "hb_attractions": "hb_p_m_car",
            "hb_productions": "hb_p_m_car",
            "nhb_attractions": "nhb_p_m_car",
            "nhb_productions": "nhb_p_m_car",
        }
    else:
        raise NotImplementedError(
            f"MiTEM forecasting only not implemented for model {model_name!r}"
        )
    return trip_ends.subset(segmentation)


def read_tripends(
    base_year: int, forecast_years: list[int], tripend_path: Path
) -> tempro_trip_ends.TEMProTripEnds:
    """
    Reads in trip-end dvectors from picklefiles
    Args:
        base_year (int): The base year for the forecast
        forecast_years (list[int]): A list of forecast years
    Returns:
        tempro_trip_ends.TEMProTripEnds: the same trip-ends read in
    """
    SEGMENTATION = {"hb": "hb_p_m", "nhb": "nhb_p_m"}
    dvectors = {
        "hb_attractions": {},
        "hb_productions": {},
        "nhb_attractions": {},
        "nhb_productions": {},
    }
    for i in ["hb", "nhb"]:
        for j in ["productions", "attractions"]:
            years = {}
            key = f"{i}_{j}"
            for year in [base_year] + forecast_years:
                dvec = nd.DVector.load(
                    os.path.join(
                        tripend_path,
                        key,
                        f"{i}_msoa_notem_segmented_{year}_dvec.pkl",
                    )
                )
                if i == "nhb":
                    dvec = dvec.reduce(nd.get_segmentation_level("notem_nhb_output_reduced"))
                years[year] = dvec.aggregate(nd.get_segmentation_level(SEGMENTATION[i]))
            dvectors[key] = years
    return tempro_trip_ends.TEMProTripEnds(**dvectors)


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

    if model == forecast_cnfg.ForecastModel.TRIP_END or forecast_cnfg.ForecastModel.NTEM:
        params = forecast_cnfg.ForecastParameters.load_yaml(config_path)
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
    LOG.info("Input parameters: %r", params)

    if model == forecast_cnfg.ForecastModel.TRIP_END:
        tem_forecasting(params)
    elif model == forecast_cnfg.ForecastModel.NTEM:
        raise NotImplementedError("forecasting not yet implemented for NTEM")

    LOG.info(
        "%s forecasting completed in %s",
        model.value.upper(),
        timing.time_taken(start, timing.current_milli_time()),
    )


def tem_forecasting(params: forecast_cnfg.ForecastParameters) -> None:
    """Main function for running the TEM forecasting.

    Parameters
    ----------
    params : ForecastParameters
        Parameters for running TEM forecasting.

    See Also
    --------
    normits_demand.models.mitem_forecast
    """
    tripend_data = read_tripends(params.base_year, params.future_years, params.tripend_path)
    tripend_data = model_mode_subset(tripend_data, params.model_name)
    tempro_growth = ntem_forecast.tempro_growth(
        tripend_data, params.model_name, params.base_year
    )
    tempro_growth.save(params.export_path / "TEMPro Growth Factors")
    mitem_inputs = mitem_forecast.MiTEMImportMatrices(
        params.matrix_import_path,
        params.base_year,
        params.model_name,
    )
    pa_output_folder = params.export_path / "Matrices" / "PA"
    ntem_forecast.grow_all_matrices(mitem_inputs, tempro_growth, pa_output_folder)
    ntem_forecast_checks.pa_matrix_comparison(
        mitem_inputs,
        pa_output_folder,
        tripend_data,
        list(params.mode.keys())[0],
        params.comparison_zone_systems,
        params.base_year,
    )
    od_folder = pa_output_folder.with_name("OD")
    ntem_forecast.convert_to_od(
        pa_output_folder,
        od_folder,
        params.base_year,
        params.future_years,
        [mitem_inputs.mode],
        {"hb": params.hb_purposes_needed, "nhb": params.nhb_purposes_needed},
        params.pa_to_od_factors,
        params.iteration,
        params.time_periods,
        params.matrix_import_path,
        params.export_path,
    )

    # Compile to output formats
    # ntem_forecast.compile_highway_for_rail(pa_output_folder, params.future_years, params.mode)
    compiled_od_path = ntem_forecast.compile_highway(
        od_folder,
        params.future_years,
        params.car_occupancies_path,
    )
    ntem_forecast_checks.od_matrix_comparison(
        mitem_inputs.od_matrix_folder,
        compiled_od_path / "PCU",
        params.model_name,
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
