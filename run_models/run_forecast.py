# -*- coding: utf-8 -*-
"""
    Module for running the NTEM forecast.
"""
# todo
##### IMPORTS #####
from __future__ import annotations

# Standard imports
import configparser
import dataclasses
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional


# Third party imports

# Add parent folder to path
sys.path.append("..")
# Local imports
# pylint: disable=import-error,wrong-import-position
from normits_demand.models import mitem_forecast, ntem_forecast, tempro_trip_ends
from normits_demand import efs_constants as efs_consts
from normits_demand import logging as nd_log
from normits_demand.utils import timing, config_base
from normits_demand.reports import ntem_forecast_checks
import normits_demand as nd

# pylint: enable=import-error,wrong-import-position

##### CONSTANTS #####
LOG_FILE = "Forecast.log"
LOG = nd_log.get_logger(nd_log.get_package_logger_name() + ".run_models.run_forecast")


##### CLASSES #####
@dataclasses.dataclass(repr=False)
class ForecastParameters:  # TODO Rewrite class as BaseConfig subclass
    """Class for storing the parameters for running forecasting.

    Attributes
    ----------
    import_path : Path
        Path to the NorMITs demand imports.
    tempro_data_path : Path
        Path to the CSV containing the TEMPro data.
    model_name : str
        Name of the model.
    iteration : str
        Iteration number.
    export_path_fmt: str
        Format for the export path, used for building
        `export_path` property.
    export_path_params : Dict[str, Any]
        Dictionary containing any additional parameters
        for building the `export_path`.
    export_path : Path
        Read-only path to export folder, this is built from
        the `export_path_fmt` with variables filled in from
        the class attributes, and additional optional values
        from `export_path_params`.
    tripend_path: Path
        Base path for the process to read trip-end data from for 
        growth factors
    matrix_import_path: Path
        Path to directory the matrix files are saved in
    """

    iteration: str = "9.7-COVID"
    import_path: Path = Path(r"I:/NorMITs Demand/import")
    model_name: str = "miham"
    base_year: int = 2021
    future_years = [2030, 2040]
    export_path_fmt: str = rf"T:/MidMITs Demand/Forecasting/{model_name}/tripend/{iteration}"
    export_path_params: Optional[Dict[str, Any]] = None
    _export_path: Optional[Path] = dataclasses.field(default=None, init=False, repr=False)
    tripend_path: Path = Path(rf'T:\MidMITs Demand\MiTEM\iter{iteration}\NTEM')
    matrix_import_path: Path = Path(rf'T:\MidMITs Demand\Distribution Model\iter{iteration}.1\car_and_passenger\Final Outputs\Full PA Matrices')

    @property
    def export_path(self) -> Path:
        """
        Read-only path to export folder, this is built from
        the `export_path_fmt` with variables filled in from
        the class attributes, and additional optional values
        from `export_path_params`.
        """
        if self._export_path is None:
            fmt_params = dataclasses.asdict(self)
            if self.export_path_params is not None:
                fmt_params.update(self.export_path_params)
        return Path(self.export_path_fmt.format(**fmt_params))

    def __repr__(self) -> str:
        params = (
            "import_path",
            "export_path",
            "model_name",
            "iteration",
        )
        msg = f"{self.__class__.__name__}("
        for p in params:
            msg += f"\n\t{p}={getattr(self, p)!s}"
        for p, val in self.pa_to_od_factors.items():
            msg += f"\n\t{p}={val!s}"
        msg += "\n)"
        return msg

    @property
    def pa_to_od_factors(self) -> Dict[str, Path]:
        """Dict[str, Path]
        Paths to the PA to OD tour proportions, has
        keys `post_me_tours` and `post_me_fh_th_factors`.
        """
        tour_prop_home = Path(
            self.import_path / self.model_name / "synthetic_tour_proportions"
        )
        paths = {
            "post_me_tours": tour_prop_home,
            "post_me_fh_th_factors": tour_prop_home / "fh_th_factors",
        }
        for nm, p in paths.items():
            if not p.is_dir():
                raise NotADirectoryError(f"cannot find {nm} folder: {p}")
        return paths

    @property
    def car_occupancies_path(self) -> Path:
        """Path
        Path to the vehicle occupancies CSV file.
        """
        path = self.import_path / "vehicle_occupancies/car_vehicle_occupancies.csv"
        if not path.exists():
            raise FileNotFoundError(f"cannot find vehicle occupancies CSV: {path}")
        return path

    def save(self, output_path: Optional[Path] = None):
        """Save current parameters to config file.

        Parameters
        ----------
        output_path : Path, optional
            Path to the file to save, if not given saves to
            'NTEM_forecast_parameters.txt' in `export_folder`.
        """
        if output_path is None:
            output_path = self.export_path / "NTEM_forecast_parameters.txt"
        config = configparser.ConfigParser()
        config["parameters"] = {
            "import_path": self.import_path,
            "model_name": self.model_name,
            "iteration": self.iteration,
            "export_path_fmt": self.export_path_fmt,
        }
        if self.export_path_params is not None:
            config["export_path_params"] = self.export_path_params
        with open(output_path, "wt") as f:
            config.write(f)
        LOG.info("Written input parameters to: %s", output_path)

    @staticmethod
    def load(path: Path) -> ForecastParameters:
        """Load parameters from config file.

        Parameters
        ----------
        path : Path
            Path to config file.

        Returns
        -------
        ForecastParameters
            New instance of the class with all parameters
            filled in from the config file.

        Raises
        ------
        FileNotFoundError
            If `path` isn't a file.
        tempro_trip_ends.NTEMForecastError
            If the config file doesn't contain a
            'parameters' section.
        """
        if not path.is_file():
            raise FileNotFoundError(f"cannot find config file: {path}")
        config = configparser.ConfigParser()
        config.read(path)
        param_sec = "parameters"
        if not config.has_section(param_sec):
            raise tempro_trip_ends.NTEMForecastError(
                f"config file doesn't contain '{param_sec}' section"
            )
        params = dict(config[param_sec])
        for p in ("import_path", "tempro_data_path"):
            if p in params:
                params[p] = Path(params[p])
        export_params = "export_path_params"
        if config.has_section(export_params):
            params[export_params] = dict(config[export_params])
        return ForecastParameters(**params)


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
    # TODO Add docstring
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
                        ForecastParameters.tripend_path,
                        key,
                        f"{i}_msoa_notem_segmented_{year}_dvec.pkl",
                    )
                )
                if i == "nhb":
                    dvec = dvec.reduce(nd.get_segmentation_level("notem_nhb_output_reduced"))
                years[year] = dvec.aggregate(nd.get_segmentation_level(SEGMENTATION[i]))
            dvectors[key] = years
    return tempro_trip_ends.TEMProTripEnds(**dvectors)


def main(params: ForecastParameters, init_logger: bool = True):
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
    params.save()

    tripend_data = read_tripends(
        base_year=efs_consts.BASE_YEAR, forecast_years=params.future_years
    )
    tripend_data = model_mode_subset(tripend_data, params.model_name)
    tempro_growth = ntem_forecast.tempro_growth(tripend_data, params.model_name)
    tempro_growth.save(params.export_path / "TEMPro Growth Factors")
    # The following lines are only to save time when testing the process and should be commented out for a real run
    # dvecs = {}
    # for purp in ["hb", "nhb"]:
    #     for pa in ["productions", "attractions"]:
    #         years = {}
    #         for year in params.future_years:
    #             dvec = nd.DVector.load(
    #                 params.export_path / "TEMPro Growth Factors" / f"{purp}_{pa}-{year}.pkl"
    #             )
    #             years[year] = dvec
    #         dvecs[f"{purp}_{pa}"] = years
    # tempro_growth = tempro_trip_ends.TEMProTripEnds(**dvecs)
    mitem_inputs = mitem_forecast.MiTEMImportMatrices(
        params.matrix_import_path,
        params.base_year,
        params.model_name,
    )
    pa_output_folder = params.export_path / "Matrices" / "PA"
    ntem_forecast.grow_all_matrices(mitem_inputs, tempro_growth, pa_output_folder)
    ntem_forecast_checks.pa_matrix_comparison(mitem_inputs, pa_output_folder, tripend_data)
    od_folder = pa_output_folder.with_name("OD")
    ntem_forecast.convert_to_od(
        pa_output_folder,
        od_folder,
        params.base_year,
        params.future_years,
        [mitem_inputs.mode],
        {"hb": efs_consts.HB_PURPOSES_NEEDED, "nhb": efs_consts.NHB_PURPOSES_NEEDED,},
        params.model_name,
        params.pa_to_od_factors,
    )

    # Compile to output formats
    # ntem_forecast.compile_highway_for_rail(pa_output_folder, params.future_years)
    compiled_od_path = ntem_forecast.compile_highway(
        od_folder, params.future_years, params.car_occupancies_path,
    )
    ntem_forecast_checks.od_matrix_comparison(
        mitem_inputs.od_matrix_folder,
        compiled_od_path / "PCU",
        params.model_name,
        ntem_forecast_checks.COMPARISON_ZONE_SYSTEMS["matrix 1"],
    )

    LOG.info(
        "NTEM forecasting completed in %s",
        timing.time_taken(start, timing.current_milli_time()),
    )


##### MAIN #####
if __name__ == "__main__":
    try:
        main(ForecastParameters())
    except Exception as err:
        LOG.critical("MiTEM forecasting error:", exc_info=True)
        raise
