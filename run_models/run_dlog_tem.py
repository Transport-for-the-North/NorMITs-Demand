# -*- coding: utf-8 -*-
"""Run module for D-Log forecast trip ends."""

##### IMPORTS #####
import pathlib
import re
import sys
from typing import Literal, Mapping
import pandas as pd

import pydantic
from pydantic import dataclasses

sys.path.extend(("..", "."))
# pylint: disable=import-error,wrong-import-position
import normits_demand as nd
from normits_demand import core, logging as nd_log
from normits_demand.utils import config_base, file_ops
from normits_demand.models import NoTEM
from normits_demand.pathing import NoTEMImportPaths

# pylint: enable=import-error,wrong-import-position

##### CONSTANTS #####
LOG = nd_log.get_logger(nd_log.get_package_logger_name() + ".run_dlog_tem")
LOG_FILE = "DLog_TEM.log"
CONFIG_FILE = pathlib.Path(r"config\models\DLog_tem_config.yaml")
# TODO Update landuse zoning, in future this parameter should be moved to config
LANDUSE_ZONING = "msoa"
RAW_LANDUSE_INDEX_COLUMNS = {
    "population": [f"{LANDUSE_ZONING}_zone_id", "tfn_traveller_type"],
    "employment": [f"{LANDUSE_ZONING}_zone_id", "sic_code"],
}


##### CLASSES #####
@dataclasses.dataclass
class _LandUseIter:
    base: str
    future: str


@dataclasses.dataclass
class _ImportVersion:
    hb_production: str
    hb_attraction: str
    nhb_production: str


@dataclasses.dataclass
class _LandUsePaths:
    population: pydantic.FilePath  # pylint: disable=no-member
    employment: pydantic.FilePath  # pylint: disable=no-member


class DLogTEMParameters(config_base.BaseConfig):
    """Load and manage parameters for running D-Log TEM."""

    raw_dlog_landuse: _LandUsePaths
    notem_import_home: pydantic.DirectoryPath  # pylint: disable=no-member
    base_year: int
    years: list[int]
    land_use_import_home: pydantic.DirectoryPath  # pylint: disable=no-member
    land_use_iteration: _LandUseIter
    import_version: _ImportVersion
    trip_end_iteration: str
    export_folder: pydantic.DirectoryPath  # pylint: disable=no-member
    scenario: core.Scenario = core.Scenario.DLOG
    overwrite_intermediary_landuse: bool = False

    @pydantic.validator("years", each_item=True)
    def _validate_years(  # pylint: disable=no-self-argument
        cls, value: int, values: dict
    ) -> int:
        base: int = values.get("base_year")  # type: ignore

        if value <= base:
            raise ValueError(f"years cannot be <= base year ({base})")

        return value


##### FUNCTIONS #####
def _find_any_landuse_file(
    scenario_folder: pathlib.Path, landuse: Literal["pop", "emp"]
) -> pathlib.Path:
    """Find files called 'land_use_{year}_{landuse}' for any year."""
    for path in scenario_folder.with_name(nd.Scenario.SC01_JAM.value).iterdir():
        if not path.is_file():
            continue

        pattern = rf"^land_use_\d{{4}}_{landuse}\.(pbz2|csv|csv\.bz2)$"
        match = re.match(pattern, path.name, re.IGNORECASE)
        if match:
            return path

    raise FileNotFoundError(f"cannot find a {landuse} land use file in {scenario_folder}")


def _infill_zones(landuse_data: pd.DataFrame) -> pd.DataFrame:
    """Infill missing zones with 0."""
    zoning = nd.get_zoning_system(LANDUSE_ZONING)

    unique_indices = []
    for column in landuse_data.index.names:
        if column == zoning.col_name:
            continue

        unique_indices.append(landuse_data.index.get_level_values(column).unique())

    full_index = pd.MultiIndex.from_product(
        [zoning.unique_zones] + unique_indices, names=landuse_data.index.names
    )

    return landuse_data.reindex(index=full_index, fill_value=0.0)


def _infill_area_type(
    scenario_folder: pathlib.Path, landuse_data: pd.DataFrame
) -> pd.DataFrame:
    """Infill area type in `landuse_data` using any existing land use files."""
    index_columns = RAW_LANDUSE_INDEX_COLUMNS["population"].copy()
    area_col = "area_type"

    try:
        other_landuse = _find_any_landuse_file(scenario_folder, "pop")
    except FileNotFoundError as error:
        raise FileNotFoundError("cannot find file to infill population area type") from error

    columns = [index_columns[0], area_col]
    area_type_lookup = file_ops.read_df(other_landuse, usecols=columns)[columns]
    # Area type is tied to MSOA but need to group rows because there
    # is more disaggregated data in the other land use file
    area_type_lookup = area_type_lookup.groupby(index_columns[0]).first()

    # TODO Convert area type lookup to LANDUSE_ZONE_SYSTEM, this lookup should already exist
    msoa_zoning = nd.get_zoning_system("msoa")
    landuse_zoning = nd.get_zoning_system(LANDUSE_ZONING)
    # MSOA to landuse zoning lookup
    # msoa_zone_id, miham_zone_id, msoa_to_miham
    translation = msoa_zoning._get_translation_definition(landuse_zoning)

    LOG.debug("Infilling area type column using %s", other_landuse)
    landuse_data = landuse_data.merge(
        area_type_lookup, on=index_columns[0], how="left", validate="m:1", indicator=True
    )
    missing = landuse_data["_merge"] != "both"
    if missing.sum() > 0:
        raise ValueError(
            f"{missing.sum()} rows not found when infilling area type from {other_landuse}"
        )
    landuse_data = landuse_data.drop(columns="_merge")

    index_columns.insert(1, area_col)
    return landuse_data.set_index(index_columns, verify_integrity=True)


def split_raw_landuse(
    raw_file: pathlib.Path,
    landuse: Literal["population", "employment"],
    output_files: Mapping[int, nd.PathLike],
    base_year: int,
    overwrite: bool,
) -> None:
    """Split a raw landuse file into separate year files.

    Output land use files will contain the cummulative land
    use change for a single year, from the `base_year` to the
    future year.

    Parameters
    ----------
    raw_file : pathlib.Path
        Path to the raw landuse data file.
    landuse : {"population", "employment"}
        Type of landuse data.
    output_files : Mapping[int, nd.PathLike]
        Dictionary where the keys are the years and the values are
        the paths to the corresponding output files to be created.
    base_year : int
        Model base year from which to calculate cummulative total.
    overwrite : bool, optional
        Whether, or not, to overwrite any existing split out land
        use files in `output_files`.
    """

    def column_filter(column: str) -> bool:
        if column in RAW_LANDUSE_INDEX_COLUMNS[landuse]:
            return True

        try:
            year = int(column)
        except ValueError:
            return False

        return base_year < year <= max_year

    scenario_folder = pathlib.Path(next(iter(output_files.values()))).parent

    if not overwrite:
        new_files = {}
        existing = []
        for year, path in output_files.items():
            if pathlib.Path(path).is_file():
                existing.append(pathlib.Path(path).name)
            else:
                new_files[year] = path

        if len(existing) > 0:
            LOG.info(
                "%s %s land use files already exist in "
                '"%s" and aren\'t being overwritten: %s',
                len(existing),
                landuse,
                scenario_folder,
                ", ".join(existing),
            )

        if len(new_files) == 0:
            LOG.info("No new files to create")
            return
        output_files = new_files

    max_year = max(output_files)
    years = [str(i) for i in output_files]
    LOG.info("Splitting raw %s landuse to separate years files from '%s'", landuse, raw_file)
    landuse_data = file_ops.read_df(raw_file, usecols=column_filter)

    # Aggregate rows because raw population is disaggregated by unneeded dwelling type column
    landuse_data = landuse_data.groupby(RAW_LANDUSE_INDEX_COLUMNS[landuse]).sum()

    # Infill missing zones with 0
    landuse_data = _infill_zones(landuse_data)

    if landuse == "population":
        landuse_data = _infill_area_type(scenario_folder, landuse_data.reset_index())

    LOG.debug(
        "Calculated cummulative total %s using years: %s",
        landuse,
        ", ".join(str(i) for i in landuse_data.columns),
    )
    # Raw landuse is change in landuse for each year separately,
    # trip end model requires total change in landuse by the
    # given year so getting cummulative totals
    landuse_data = landuse_data.cumsum(axis=1).loc[:, years]
    LOG.info("Calculated total land use change for years: %s", ", ".join(years))

    for year, path in output_files.items():
        path = pathlib.Path(path)
        path.parent.mkdir(exist_ok=True, parents=True)

        data = landuse_data.loc[:, str(year)]
        if landuse == "employment":
            data.name = "people"

        data.to_csv(path)
        LOG.info("Written: %s", path)


def main(params: DLogTEMParameters, init_logger: bool = True) -> None:
    """Run NoTEM with D-Log data."""
    if init_logger:
        # Add log file output to main package logger
        log_file = params.export_folder / LOG_FILE
        nd_log.get_logger(
            nd_log.get_package_logger_name(),
            log_file,
            "Running D-Log Trip End Model",
            log_version=True,
        )
        nd_log.capture_warnings(file_handler_args=dict(log_file=log_file))
        LOG.info("Log file saved to %s", log_file)

    LOG.debug("Input parameters:\n%s", params.to_yaml())

    # TODO Double check these paths but I think all the trip rates inputs can stay the same
    import_builder = NoTEMImportPaths(
        import_home=params.notem_import_home,
        scenario=params.scenario,
        years=params.years,
        land_use_import_home=params.land_use_import_home,
        by_land_use_iter=params.land_use_iteration.base,
        fy_land_use_iter=params.land_use_iteration.future,
        hb_production_import_version=params.import_version.hb_production,
        hb_attraction_import_version=params.import_version.hb_attraction,
        nhb_production_import_version=params.import_version.nhb_production,
    )

    split_landuse_log = (
        pathlib.Path(next(iter(import_builder.population_paths.values()))).parent / LOG_FILE
    )
    split_landuse_log = split_landuse_log.with_name(
        split_landuse_log.stem + "-split_landuse.log"
    )
    split_landuse_log.parent.mkdir(exist_ok=True, parents=True)

    with nd_log.TemporaryLogFile(LOG, split_landuse_log):
        LOG.debug(
            'Split landuse outputs created by DLog trip end model, main log file: "%s"',
            log_file,
        )

        # TODO(MB) Determine a better location to save the split land use for separate iterations
        # Split required years into separate files and load into land use folder
        split_raw_landuse(
            params.raw_dlog_landuse.population,
            "population",
            import_builder.population_paths,
            params.base_year,
            params.overwrite_intermediary_landuse,
        )
        split_raw_landuse(
            params.raw_dlog_landuse.employment,
            "employment",
            import_builder.employment_paths,
            params.base_year,
            params.overwrite_intermediary_landuse,
        )

    tem = NoTEM(
        years=params.years,
        scenario=params.scenario,
        iteration_name=params.trip_end_iteration,
        import_builder=import_builder,
        export_home=params.export_folder,
        hb_attraction_balance_zoning=False,
        nhb_attraction_balance_zoning=False,
        # TODO Pass LANDUSE_ZONING as parameter
    )
    tem.run(generate_all=True, non_resi_path=False)


def _run() -> None:
    # TODO(MB) Add command line argument to pass a different config file
    print(f"Loading config: {CONFIG_FILE}")
    parameters = DLogTEMParameters.load_yaml(CONFIG_FILE)

    try:
        main(parameters)
    except Exception:
        LOG.critical("Critical error occurred", exc_info=True)
        raise


##### MAIN #####
if __name__ == "__main__":
    _run()
