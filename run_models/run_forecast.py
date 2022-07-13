# -*- coding: utf-8 -*-
"""
    Module for running the NTEM forecast.
"""
# todo
##### IMPORTS #####
from __future__ import annotations
from ast import For, Str

# Standard imports
import configparser
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional


# Third party imports

# Add parent folder to path
sys.path.append("..")
# Local imports
# pylint: disable=import-error,wrong-import-position
import forecast_cnfg
import normits_demand as nd
from normits_demand.models import mitem_forecast, ntem_forecast, tempro_trip_ends
from normits_demand import logging as nd_log
from normits_demand.reports import ntem_forecast_checks
from normits_demand.utils import timing




# pylint: enable=import-error,wrong-import-position

##### CONSTANTS #####
LOG_FILE = "Forecast.log"
LOG = nd_log.get_logger(nd_log.get_package_logger_name() + ".run_models.run_forecast")
PARAMS = forecast_cnfg.ForecastParameters.load_yaml("config/run_forecast.yml")


##### FUNCTIONS #####
def model_mode_subset(
    trip_ends: tempro_trip_ends.TEMProTripEnds, model_name: str,
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
    base_year: int, forecast_years: list[int]
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
                        PARAMS.tripend_path, key, f"{i}_msoa_notem_segmented_{year}_dvec.pkl",
                    )
                )
                if i == "nhb":
                    dvec = dvec.reduce(nd.get_segmentation_level("notem_nhb_output_reduced"))
                years[year] = dvec.aggregate(nd.get_segmentation_level(SEGMENTATION[i]))
            dvectors[key] = years
    return tempro_trip_ends.TEMProTripEnds(**dvectors)


def main(params: forecast_cnfg.ForecastParameters, init_logger: bool = True):
    """Main function for running the MiTEM forecasting.

    Parameters
    ----------
    params : ForecastParameters
        Parameters for running MiTEM forecasting.
    init_logger : bool, default True
        If True initialises logger with log file
        in the export folder.

    See Also
    --------
    normits_demand.models.mitem_forecast
    """

    start = timing.current_milli_time()
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
            "Running MiTEM forecast",
        )
    LOG.info(msg, params.export_path)
    LOG.info("Input parameters: %r", params)

    tripend_data = read_tripends(
        base_year=params.base_year, forecast_years=params.future_years
    )
    tripend_data = model_mode_subset(tripend_data, params.model_name)
    tempro_growth = ntem_forecast.tempro_growth(
        tripend_data, params.model_name, params.base_year
    )
    tempro_growth.save(params.export_path / "TEMPro Growth Factors")
    mitem_inputs = mitem_forecast.MiTEMImportMatrices(
        params.matrix_import_path, params.base_year, params.model_name,
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
        od_folder, params.future_years, params.car_occupancies_path,
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

    LOG.info(
        "NTEM forecasting completed in %s",
        timing.time_taken(start, timing.current_milli_time()),
    )


##### MAIN #####
if __name__ == "__main__":
    try:
        main(PARAMS)
    except Exception as err:
        LOG.critical("MiTEM forecasting error:", exc_info=True)
        raise
