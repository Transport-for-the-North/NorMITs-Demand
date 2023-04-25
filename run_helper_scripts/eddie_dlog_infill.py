# -*- coding: utf-8 -*-
"""Script to infill EDDIE land use data using DLIT outputs."""

##### IMPORTS #####
import datetime as dt
import pathlib
import sys
from typing import Any

import pandas as pd
import pydantic
from pydantic import dataclasses

sys.path.extend([".", "./run_helper_scripts"])
# pylint: disable=wrong-import-position
from eddie_npier_infill import (
    EDDIE_ZONE_SYSTEM,
    OUTPUT_YEARS,
    EDDIELandUseData,
    EDDIEWorkbookParams,
    NPIEREDDIEFormatLandUse,
    _calculate_yearly_quarters,
    _normalise_region_names,
    load_disaggregated_eddie,
    load_landuse,
    load_landuse_eddie,
    merge_with_eddie,
    npier_eddie_comparison_heatmaps,
    output_eddie_format,
)

import normits_demand as nd
from normits_demand import logging as nd_log
from normits_demand.core.enumerations import LandUseType
from normits_demand.utils import config_base, plots

# pylint: enable=wrong-import-position


##### CONSTANTS #####
LOG = nd_log.get_logger(nd_log.get_package_logger_name() + ".eddie_dlog_infill")
LOG_FILE = "EDDIE_DLOG_infill.log"
CONFIG_FILE = pathlib.Path(r"config\helper\EDDIE_dlog_infill_parameters.yml")
DLOG_ZONE_SYSTEM = "msoa"


##### CLASSES #####
@dataclasses.dataclass
class DLogLandUsePaths:
    """Folder and filenames for DLIT land use files."""

    folder: pydantic.DirectoryPath  # pylint: disable=no-member
    employment_filename: str = "employment_msoa_build_out.csv"
    residential_filename: str = "residential_msoa_build_out.csv"

    # Makes a classmethod not recognised by pylint, hence disabling self check
    @pydantic.validator("employment_filename", "residential_filename")
    def _check_files(  # pylint: disable=no-self-argument
        cls, value: str, values: dict[str, Any]
    ) -> str:
        """Check files exist in given `folder`."""
        folder: pathlib.Path = values["folder"]

        if not isinstance(folder, pathlib.Path):
            raise ValueError("no folder path given")

        if not (folder / value).is_file():
            raise ValueError(f"'{value}' is not a file found in '{folder}'")

        return value


class EDDIEDLogParameters(config_base.BaseConfig):
    """Parameters for running EDDIE D-Log infill."""

    output_folder: pydantic.DirectoryPath  # pylint: disable=no-member
    eddie_file: pydantic.FilePath  # pylint: disable=no-member
    workbook_parameters: EDDIEWorkbookParams
    dlit_landuse: DLogLandUsePaths
    map_years: list[int]
    lad_geospatial_file: plots.GeoSpatialFile | None = None
    regions_geospatial_file: plots.GeoSpatialFile | None = None

    # Makes a classmethod not recognised by pylint, hence disabling self check
    @pydantic.validator("lad_geospatial_file", "regions_geospatial_file", pre=True)
    def _dict_to_tuple(cls, value: dict) -> tuple:  # pylint: disable=no-self-argument
        try:
            return value["path"], value["id_column"]
        except KeyError as err:
            raise TypeError(f"missing {err} value") from err


##### FUNCTIONS #####
def load_dlog_landuse(paths: DLogLandUsePaths) -> dict[LandUseType, pd.DataFrame]:
    """Load D-Log land use and convert to EDDIE zone system."""
    landuse: dict[LandUseType, pd.DataFrame] = {}

    from_zone = nd.get_zoning_system(DLOG_ZONE_SYSTEM)
    to_zone = nd.get_zoning_system(EDDIE_ZONE_SYSTEM)

    file_paths = (
        (LandUseType.EMPLOYMENT, paths.employment_filename),
        (LandUseType.POPULATION, paths.residential_filename),
    )

    for lu_type, filename in file_paths:
        LOG.info("Loading %s landuse from %s", lu_type.value, filename)
        df = load_landuse(paths.folder / filename, from_zone, to_zone, [])

        landuse[lu_type] = df.cumsum(axis=1)

    return landuse


def _update_columns(
    data: pd.DataFrame,
    columns: list[str],
    eddie_suffix: str = "_eddie",
    dlog_suffix: str = "_north",
) -> pd.DataFrame:
    new_columns = [data["_merge"]]

    for col_nm in columns:
        eddie_col = col_nm + eddie_suffix
        dlog_col = col_nm + dlog_suffix

        try:
            col = data[eddie_col].add(data[dlog_col], fill_value=0)
        except KeyError:
            continue

        col.name = col_nm
        new_columns.append(col)

    return pd.concat(new_columns, axis=1)


def write_dlog_infilled_eddie(
    eddie: EDDIELandUseData,
    dlog: dict[LandUseType, pd.DataFrame],
    output_file: pathlib.Path,
) -> NPIEREDDIEFormatLandUse:
    """Add `dlog` to `eddie` and save to spreadsheet.

    Parameters
    ----------
    eddie : EDDIELandUseData
        EDDIE data.
    dlog : dict[LandUseType, pd.DataFrame]
        D-Log data (from DLIT).
    output_file : pathlib.Path
        Excel file to create.

    Returns
    -------
    NPIEREDDIEFormatLandUse
        D-Log adjusted EDDIE land use data.
    """
    LOG.info("Creating D-Log updated EDDIE format")
    kwargs = dict(
        how="left",
        validate="1:1",
        indicator=True,
    )
    columns = [f"{i}_q{j}" for i in OUTPUT_YEARS for j in "1234"]

    merged_landuse: dict[LandUseType, pd.DataFrame] = {}
    for lu_type, eddie_data in eddie.data.items():
        dlog_data = dlog[lu_type].copy()
        dlog_data.index.name = "dlog_lad"

        merged = merge_with_eddie(
            eddie_data,
            dlog_data,
            eddie_cols=["lad_2017_zone_id"],
            north_cols=["dlog_lad"],
            **kwargs,
        )
        merged_landuse[lu_type] = _update_columns(merged, columns)

    # TODO Do we need to calculate WOR / WAP for EDDIE infilling?
    # Do we have enough data from D-Log to calculate WOR / WAP?

    landuse_data = NPIEREDDIEFormatLandUse(
        population=merged_landuse[LandUseType.POPULATION],
        employment=merged_landuse[LandUseType.EMPLOYMENT],
    )
    output_eddie_format(landuse_data, output_file)
    return landuse_data


def main(parameters: EDDIEDLogParameters, init_logger: bool = True) -> None:
    """Run EDDIE infilling process with D-Log data."""
    output_folder: pathlib.Path = (
        parameters.output_folder / f"D-Log Adjusted EDDIE - {dt.date.today():%Y-%m-%d}"
    )
    if not output_folder.is_dir():
        output_folder.mkdir(parents=True)

    if init_logger:
        nd_log.get_logger(
            nd_log.get_package_logger_name(),
            output_folder / LOG_FILE,
            "Running EDDIE D-Log Infill",
            log_version=True,
        )
        nd_log.capture_warnings(file_handler_args=dict(log_file=output_folder / LOG_FILE))

    LOG.debug("Input parameters:\n%s", parameters.to_yaml())
    params_out_file = output_folder / CONFIG_FILE.name
    LOG.info("Written input parameters to %s", params_out_file)
    parameters.save_yaml(params_out_file)

    eddie = load_landuse_eddie(
        parameters.eddie_file,
        parameters.workbook_parameters.landuse,
        parameters.workbook_parameters.lad_lookup,
    )
    disaggregated_eddie = load_disaggregated_eddie(
        parameters.eddie_file, parameters.workbook_parameters
    )

    dlog = load_dlog_landuse(parameters.dlit_landuse)

    reformatted_dlog = {k: _calculate_yearly_quarters(v) for k, v in dlog.items()}

    if parameters.regions_geospatial_file is None:
        LOG.warning("Regions geospatial file not given")
        regions = None
    else:
        regions = plots.get_geo_data(parameters.regions_geospatial_file)
        regions.index = _normalise_region_names(regions.index)

    if parameters.lad_geospatial_file is None:
        LOG.warning("LADs geospatial file not given")
        lads = None
    else:
        lads = plots.get_geo_data(parameters.lad_geospatial_file)

    dlog_adjusted_eddie = write_dlog_infilled_eddie(
        eddie,
        reformatted_dlog,
        output_folder / "DLOG_EDDIE_format.xlsx",
    )
    npier_eddie_comparison_heatmaps(
        eddie,
        disaggregated_eddie,
        dlog_adjusted_eddie,
        output_folder / "D-Log Adjusted EDDIE Comparison",
        plot_years=parameters.map_years,
        regions=regions,
        lads=lads,
        npier_name="DLOG",
    )


def _run() -> None:
    # TODO(MB) Add argument for config file path
    parameters = EDDIEDLogParameters.load_yaml(CONFIG_FILE)

    main(parameters)


if __name__ == "__main__":
    _run()
