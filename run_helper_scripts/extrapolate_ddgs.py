# -*- coding: utf-8 -*-
"""Script to extrapolate various DDGs into the future."""

##### IMPORTS #####
import logging
import os
import pathlib
import re
import sys

import pandas as pd

sys.path.append("..")
sys.path.append(".")
# pylint: disable=import-error, wrong-import-position
from normits_demand import logging as nd_log
from normits_demand.utils import config_base

# pylint: enable=import-error, wrong-import-position

##### CONSTANTS #####
LOG = logging.getLogger(nd_log.get_package_logger_name() + ".extrapolate_ddgs")
LOG_FILE = "Extrapolate_DDGs.log"
CONFIG_PATH = pathlib.Path("config/helper/Extrapolate_DDGs_config.yml")

##### CLASSES #####
# pylint: disable=too-few-public-methods
class ExtrapolateDDGsParameters(config_base.BaseConfig):
    """Class for handling config file and parameters."""

    input_folder: pathlib.Path
    output_folder: pathlib.Path
    final_year: int
    number_calculation_years: int


# pylint: enable=too-few-public-methods

##### FUNCTIONS #####
def _check_columns(ddg: pd.DataFrame) -> tuple[pd.DataFrame, list[int]]:
    ddg.columns = ddg.columns.str.strip()

    year_columns = []
    index_columns = []
    for column in ddg.columns:
        if re.match(r"^\d+$", column):
            year_columns.append(int(column))
        else:
            index_columns.append(column)

    year_columns = sorted(year_columns)
    LOG.debug("Found following years in DDG: %s", ", ".join(str(i) for i in year_columns))
    ddg = ddg.set_index(index_columns)
    ddg.columns = pd.to_numeric(ddg.columns, downcast="unsigned")

    # Rename any unnamed index columns
    ddg.index.names = ["" if n.lower().startswith("unnamed") else n for n in ddg.index.names]

    return ddg, year_columns


def _check_calculation_years(ddg_years: list[int], calculation_years: int) -> tuple[int, int]:
    if calculation_years > len(ddg_years):
        raise ValueError(
            f"{len(ddg_years)} years found in DDG so cannot use "
            f"{calculation_years} years for extrapolation calculation"
        )
    if calculation_years < -1 or calculation_years == 0:
        raise ValueError(
            "number of calculation years should be a positive integer "
            f"(or -1 to select all years) not {calculation_years}"
        )

    if calculation_years == -1:
        calc_range = (ddg_years[0], ddg_years[-1])
    else:
        calc_range = (ddg_years[-1 * calculation_years], ddg_years[-1])
    LOG.info("Using %s - %s for calculating extrapolation factor", *calc_range)
    return calc_range


def extrapolate_ddg(
    input_file: pathlib.Path,
    output_file: pathlib.Path,
    reports_folder: pathlib.Path,
    extrapolation_year: int,
    calculation_years: int,
) -> None:
    """Extrapolate give `input_file` to `extrapolation_year`.

    Won't perform extrapolation if the `input_file` already contains
    a year column >= `extrapolation_year`. Adds all years between
    the maximum current year in the file and `extrapolation_year`.

    Parameters
    ----------
    input_file : pathlib.Path
        Path to the DDG file.
    output_file : pathlib.Path
        Path to save extrapolated file to.
    reports_folder : pathlib.Path
        Folder to save the reports / summaries to.
    extrapolation_year : int
        Year to extrapolate to, inclusive.
    calculation_years : int
        Number of years to use when calculating yearly average increase.
    """
    LOG.info("Reading %s", input_file.name)
    ddg = pd.read_csv(input_file)
    ddg, year_columns = _check_columns(ddg)

    if extrapolation_year <= year_columns[-1]:
        LOG.info(
            "Extrapolation aborted because DDG %s max year "
            "is %s which is larger than the extrapolation year %s",
            input_file.stem,
            year_columns[-1],
            extrapolation_year,
        )
        return

    calc_range = _check_calculation_years(year_columns, calculation_years)

    yearly_increase = (ddg[calc_range[1]] - ddg[calc_range[0]]) / (
        calc_range[1] - calc_range[0]
    )
    yearly_increase.name = "Average Yearly Increase"
    increase_path = reports_folder / (output_file.stem + "-average_yearly_increase.csv")
    yearly_increase.to_csv(increase_path)
    LOG.info("Written average yearly increase to %s", increase_path)

    extrapolated = []
    for i, year in enumerate(range(calc_range[1] + 1, extrapolation_year + 1), 1):
        extrap = (i * yearly_increase) + ddg[calc_range[1]]
        extrap.name = year
        extrapolated.append(extrap)

    extrapolated_ddg = pd.concat([ddg] + extrapolated, axis=1)
    # Convert columns back to string to avoid float format issues
    extrapolated_ddg.columns = [f"{c:.0f}" for c in extrapolated_ddg.columns]
    extrapolated_ddg.to_csv(output_file)
    LOG.info(
        "Added additional years %s - %s and saved to %s",
        calc_range[1] + 1,
        extrapolation_year,
        output_file,
    )


def main(params: ExtrapolateDDGsParameters, init_logger: bool = True) -> None:
    """Run extrapolate DDGs.

    Parameters
    ----------
    params : ExtrapolateDDGsParameters
        Config parameters for running.
    init_logger : bool, default True
        Whether or not to initlise a log file.
    """
    params.output_folder.mkdir(exist_ok=True)
    reports_folder = params.output_folder / "reports"
    reports_folder.mkdir(exist_ok=True)

    if init_logger:
        nd_log.get_logger(
            nd_log.get_package_logger_name(),
            params.output_folder / LOG_FILE,
            "Running DDG Extrapolation",
            log_version=True,
        )
        nd_log.capture_warnings(
            file_handler_args=dict(log_file=params.output_folder / LOG_FILE)
        )

    param_path = params.output_folder / CONFIG_PATH.name
    params.save_yaml(param_path)
    LOG.info("Input parameters saved to %s", param_path)

    LOG.info("Extrapolating DDGs to %s", params.final_year)
    ddg_files = [p for p in params.input_folder.iterdir() if p.is_file()]
    LOG.info("Found %s DDG files for extrapolation in %s", len(ddg_files), params.input_folder)

    for path in ddg_files:
        extrapolate_ddg(
            path,
            params.output_folder / path.name,
            reports_folder,
            params.final_year,
            params.number_calculation_years,
        )

    os.startfile(params.output_folder)


def _run() -> None:
    parameters = ExtrapolateDDGsParameters.load_yaml(CONFIG_PATH)
    main(parameters)


##### MAIN #####
if __name__ == "__main__":
    _run()
