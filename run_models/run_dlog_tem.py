# -*- coding: utf-8 -*-
"""Run module for D-Log forecast trip ends."""

##### IMPORTS #####
import pathlib
import sys
from typing import Literal, Mapping

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
RAW_LANDUSE_INDEX_COLUMNS = {
    "population": ["msoa_zone_id", "dwelling_type", "tfn_traveller_type"],
    "employment": ["msoa_zone_id", "sic_code"],
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

    @pydantic.validator("years", each_item=True)
    def _validate_years(  # pylint: disable=no-self-argument
        cls, value: int, values: dict
    ) -> int:
        base: int = values.get("base_year")  # type: ignore

        if value <= base:
            raise ValueError(f"years cannot be <= base year ({base})")

        return value


##### FUNCTIONS #####
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
                pathlib.Path(next(iter(output_files.values()))).parent,
                ", ".join(existing),
            )

        if len(new_files) == 0:
            LOG.info("No new files to create")
            return
        output_files = new_files

    max_year = max(output_files)
    years = [str(i) for i in output_files]
    LOG.info("Splitting raw %s landuse to separate years files", landuse)
    landuse_data = file_ops.read_df(
        raw_file,
        index_col=RAW_LANDUSE_INDEX_COLUMNS[landuse],  # type: ignore
        usecols=column_filter,
    )

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

        landuse_data.loc[:, str(year)].to_csv(path)
        LOG.info("Written: %s", path)


def main(params: DLogTEMParameters, init_logger: bool = True) -> None:
    if init_logger:
        # Add log file output to main package logger
        nd_log.get_logger(
            nd_log.get_package_logger_name(),
            params.export_folder / LOG_FILE,
            "Running D-Log Trip End Model",
            log_version=True,
        )
        nd_log.capture_warnings(
            file_handler_args=dict(log_file=params.export_folder / LOG_FILE)
        )
        LOG.info("Log file saved to %s", params.export_folder / LOG_FILE)

    LOG.debug("Input parameters:\n%s", params.to_yaml())

    # Define different balancing zones for each mode
    mode_balancing_zones = {5: nd.get_zoning_system("ca_sector_2020")}
    hb_attraction_balance_zoning = nd.BalancingZones.build_single_segment_group(
        nd.get_segmentation_level("notem_hb_output"),
        nd.get_zoning_system("gor"),
        "m",
        mode_balancing_zones,
    )
    nhb_attraction_balance_zoning = nd.BalancingZones.build_single_segment_group(
        nd.get_segmentation_level("notem_nhb_output"),
        nd.get_zoning_system("gor"),
        "m",
        mode_balancing_zones,
    )

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

    # Split required years into separate files and load into land use folder
    split_raw_landuse(
        params.raw_dlog_landuse.population,
        "population",
        import_builder.population_paths,
        params.base_year,
        False,
    )
    split_raw_landuse(
        params.raw_dlog_landuse.employment,
        "employment",
        import_builder.employment_paths,
        params.base_year,
        False,
    )

    raise NotImplementedError()
    tem = NoTEM(
        years=params.years,
        scenario=params.scenario,
        iteration_name=params.trip_end_iteration,
        import_builder=import_builder,
        export_home=params.export_folder,
        hb_attraction_balance_zoning=hb_attraction_balance_zoning,
        nhb_attraction_balance_zoning=nhb_attraction_balance_zoning,
    )
    tem.run(generate_all=True)

    # TODO(MB) Run distribution model


def _run() -> None:
    print(f"Loading config: {CONFIG_FILE}")
    parameters = DLogTEMParameters.load_yaml(CONFIG_FILE)

    main(parameters)


##### MAIN #####
if __name__ == "__main__":
    _run()
