# -*- coding: utf-8 -*-
"""Compare the D-Log trip ends to the base year trip ends."""

##### IMPORTS #####
# Standard imports
from dataclasses import fields
import functools
import pathlib
import sys
from typing import Optional

# Third party imports
import geopandas as gpd
import numpy as np
import pandas as pd
import pydantic
from pydantic import dataclasses

# Local imports
sys.path.extend([".", ".."])
# pylint: disable=import-error,wrong-import-position
import normits_demand as nd
from normits_demand import logging as nd_log
from normits_demand.utils import config_base, plots

# pylint: enable=import-error,wrong-import-position

##### CONSTANTS #####
LOG = nd_log.get_logger(nd_log.get_package_logger_name() + ".dlog_trip_end_comparison")
LOG_FILE = "DLog_TE_comparison.log"
TRIP_END_PATH_FMT = (
    "{folder}/iter{iter}/{scenario}/{trip_end_type}"
    "/{trip_origin}_msoa_notem_segmented_{year}_dvec.pkl"
)
ANNUAL_GROWTH = 1.01
COMPARISON_ZONING = "lad_2020"
COMPARISON_SEGMENTATIONS = {nd.TripOrigin.HB: "hb_p_m", nd.TripOrigin.NHB: "nhb_p_m"}
KEY_LOOKUP = {"p": "purpose", "m": "mode"}


##### CLASSES #####
@dataclasses.dataclass
class _TEMParameters:
    """Stores TEM parameters."""

    base_folder: pydantic.DirectoryPath  # pylint: disable=no-member
    scenario: nd.Scenario
    iteration: str


class DLogTEComparisonParameters(config_base.BaseConfig):
    """Parameters for running D-Log trip end comparison."""
    base_tem: _TEMParameters
    base_year: int
    dlog_tem: _TEMParameters
    dlog_years: list[int]
    output_folder: pydantic.DirectoryPath  # pylint: disable=no-member


class _Config:
    arbitrary_types_allowed = True


@dataclasses.dataclass(config=_Config)
class _DVecComparison:
    """Stores four DataFrames for the D-Log comparison."""
    dlog: pd.DataFrame
    grown_base: pd.DataFrame
    difference: pd.DataFrame
    percentage: pd.DataFrame


##### FUNCTIONS #####
def _load_dvector(
    parameters: _TEMParameters, trip_end_type: nd.TripEndType, year: int
) -> nd.DVector:
    """Load DVector from file.

    Parameters
    ----------
    parameters : _TEMParameters
        Parameters for the TEM outputs location.
    trip_end_type : nd.TripEndType
        Type of trip end to load.
    year : int
        Year of data to load.

    Returns
    -------
    nd.DVector
        DVector data.

    Raises
    ------
    FileNotFoundError
        If a file can't be found with given parameters.
    """
    path = pathlib.Path(
        TRIP_END_PATH_FMT.format(
            folder=parameters.base_folder,
            iter=parameters.iteration,
            scenario=parameters.scenario.value,
            trip_end_type=trip_end_type.value,
            trip_origin=trip_end_type.trip_origin.value,
            year=year,
        )
    )

    if not path.is_file():
        raise FileNotFoundError(f"cannot find trip ends: {path}")

    return nd.DVector.load(path)


def _compound_growth(arr: np.ndarray, growth: float, year: int, base_year: int) -> np.ndarray:
    """Applies growth percentage to `arr` for the number of years since `base_year`."""
    return arr * growth ** (year - base_year)


def _compare_dvectors(
    dlog: nd.DVector,
    other: nd.DVector,
    segmentation: nd.SegmentationLevel,
    zoning: nd.ZoningSystem,
    weighting: Optional[str] = None,
) -> dict[str, _DVecComparison]:
    """Compare two DVectors at different segmentations.

    Comparisons are done for the given segmentation, purpose, mode
    and zone totals.

    Parameters
    ----------
    dlog : nd.DVector
        D-Log trip ends.
    other : nd.DVector
        Grown base trip ends.
    segmentation : nd.SegmentationLevel
        Segmentation level both DVectors are aggregated to before comparison.
    zoning : nd.ZoningSystem
        Zoning system for both DVectors to be translated to.
    weighting : Optional[str], optional
        Weighting to use for zone translation.

    Returns
    -------
    dict[str, _DVecComparison]
        Dictionary of the different comparisons done with keys
        "{segmentation name}", "purpose", "mode" and "total".
    """
    raw_dlog = dlog.translate_zoning(zoning, weighting).aggregate(segmentation).to_df()
    raw_other = other.translate_zoning(zoning, weighting).aggregate(segmentation).to_df()

    comparisons = {}
    for agg in (None, "p", "m", "total"):
        if agg is None:
            key = segmentation.name
            dlog_data = raw_dlog.set_index([zoning.col_name, "p", "m"])
            other_data = raw_other.set_index([zoning.col_name, "p", "m"])

        elif agg == "total":
            key = agg
            dlog_data = raw_dlog.groupby(zoning.col_name)["val"].sum().to_frame()
            other_data = raw_other.groupby(zoning.col_name)["val"].sum().to_frame()

            rename = {"val": "Trip Ends"}
            dlog_data.rename(columns=rename, inplace=True)
            other_data.rename(columns=rename, inplace=True)

        else:
            key = KEY_LOOKUP[agg]
            dlog_data = (
                raw_dlog.groupby([zoning.col_name, agg], as_index=False)
                .sum()
                .pivot(index=zoning.col_name, columns=agg, values="val")
            )
            other_data = (
                raw_other.groupby([zoning.col_name, agg], as_index=False)
                .sum()
                .pivot(index=zoning.col_name, columns=agg, values="val")
            )

        # Remove any zones / segmentations with no D-Log data
        dlog_index = dlog_data.index[(dlog_data != 0).any(axis=1).values]
        dlog_data = dlog_data.loc[dlog_index]
        other_data = other_data.loc[dlog_index]

        diff = dlog_data - other_data
        perc_diff = dlog_data / other_data

        comparisons[key] = _DVecComparison(
            dlog=dlog_data, grown_base=other_data, difference=diff, percentage=perc_diff
        )

    return comparisons


def _write_comparisons(
    comparisons: dict[str, _DVecComparison], excel_path: pathlib.Path, zoning: nd.ZoningSystem
) -> None:
    """Write all `comparisons` to separate sheets in the same Excel workbook."""
    with pd.ExcelWriter(excel_path) as excel:
        for name, comp in comparisons.items():
            for field in fields(comp):
                data: pd.DataFrame = getattr(comp, field.name).copy()

                # Add zone descriptions
                name_column = f"{zoning.name}_name"
                data.loc[:, name_column] = (
                    data.index.get_level_values(zoning.col_name)
                    .to_series()
                    .replace(zoning.zone_to_description_dict)
                    .values
                )
                data = data.set_index(name_column, append=True)

                if data.index.nlevels > 2:
                    # Order index so the first 2 columns are zone ID and zone name
                    index = list(data.index.names)
                    index.remove(name_column)
                    data = data.reorder_levels([index[0], name_column, *index[1:]])

                data.to_excel(excel, sheet_name=f"{name} - {field.name}")

    LOG.info("Written: %s", excel_path)


def _comparison_heatmaps(
    geodata: gpd.GeoDataFrame,
    comparison: _DVecComparison,
    zoning: nd.ZoningSystem,
    output_folder: pathlib.Path,
    data_name: str,
    title: str,
) -> None:
    """Produce heatmaps of the percetanage difference for each column in `comparison`."""
    plot_data = geodata.merge(
        comparison.percentage.reset_index(), on=zoning.col_name, how="left", validate="1:1"
    )

    for column in comparison.percentage.columns:
        # TODO Need to create a new branch for trip end comparison and merge
        # eddie_inputs branch to get the updated version of plots module
        fig = plots._heatmap_figure(
            plot_data,
            column,
            title,
            n_bins=5,
            positive_negative_colormaps=True,
            legend_label_fmt="{:.0%}",
            legend_title=f"{data_name.title()} {column}",
            # bins=np.concatenate([neg_bins, pos_bins]),
            zoomed_bounds=plots.Bounds(290000, 340000, 550000, 670000),
            missing_kwds=dict(color=(0.8, 0.8, 0.8, 1), label="No Developments"),
        )

        file = output_folder / f"{title} - {column}.png"
        fig.savefig(file)
        LOG.info("Written: %s", file)


def main(params: DLogTEComparisonParameters, init_logger: bool = True) -> None:
    """Produce comparisons between D-Log and grown base trip ends.

    Base trip ends are grown by 1% annually.

    Parameters
    ----------
    params : DLogTEComparisonParameters
        Parameters for the inputs.
    init_logger : bool, default True
        Initialise a logger with a log file in the output folder.
    """
    output_folder = params.output_folder / (
        f"{params.dlog_tem.scenario.value}_iter{params.dlog_tem.iteration}-"
        f"{params.base_tem.scenario.value}_iter{params.base_tem.iteration}"
    )
    output_folder.mkdir(exist_ok=True, parents=True)
    LOG.info("Outputs saved to %s", output_folder)

    if init_logger:
        # Add log file output to main package logger
        log_file = output_folder / LOG_FILE
        nd_log.get_logger(
            nd_log.get_package_logger_name(),
            log_file,
            "Running D-Log Trip End Comparisons",
            log_version=True,
        )
        nd_log.capture_warnings(file_handler_args=dict(log_file=log_file))
        LOG.info("Log file saved to %s", log_file)

    LOG.debug("Input parameters:\n%s", params.to_yaml())

    comparison_zone_system = nd.get_zoning_system(COMPARISON_ZONING)
    zone_meta = comparison_zone_system.get_metadata()
    geodata = gpd.read_file(zone_meta.shapefile_path).loc[
        :, [zone_meta.shapefile_id_col, "geometry"]
    ]
    geodata = geodata.rename(
        columns={zone_meta.shapefile_id_col: comparison_zone_system.col_name}
    )

    for te_type in nd.TripEndType:
        LOG.info("Loading base %s %s", params.base_year, te_type)
        base = _load_dvector(params.base_tem, te_type, params.base_year)

        for year in params.dlog_years:
            LOG.info("Loading D-Log %s %s", year, te_type)
            dlog = _load_dvector(params.dlog_tem, te_type, year)

            factored_base = base.segment_apply(
                functools.partial(
                    _compound_growth,
                    growth=ANNUAL_GROWTH,
                    year=year,
                    base_year=params.base_year,
                )
            )

            comparisons = _compare_dvectors(
                dlog,
                factored_base,
                nd.get_segmentation_level(COMPARISON_SEGMENTATIONS[te_type.trip_origin]),
                comparison_zone_system,
            )
            _write_comparisons(
                comparisons,
                output_folder / f"{te_type.formatted()} - {year}.xlsx",
                comparison_zone_system,
            )

            for name in ("total", "purpose", "mode"):
                comparison = comparisons[name]
                plot_folder = output_folder / f"{te_type.formatted()} {year} - heatmaps"
                plot_folder.mkdir(exist_ok=True)
                _comparison_heatmaps(
                    geodata,
                    comparison,
                    comparison_zone_system,
                    plot_folder,
                    name,
                    f"{te_type.formatted()} {year} {name.title()}",
                )


def _run() -> None:
    # TODO(MB) Load parameters from config file
    parameters = DLogTEComparisonParameters(
        base_tem=_TEMParameters(
            base_folder=r"I:\NorMITs Demand\NoTEM",
            scenario=nd.Scenario.SC01_JAM,
            iteration="9.10",
        ),
        base_year=2018,
        dlog_tem=_TEMParameters(
            base_folder=r"C:\Users\ukmjb018\OneDrive - WSP O365\WSP_Projects\TfN NorMITs Demand Partner 2022\DLog Matrix\Outputs\DLogTEM",
            scenario=nd.Scenario.DLOG,
            iteration="1.0",
        ),
        dlog_years=[2025, 2030],
        output_folder=r"C:\Users\ukmjb018\OneDrive - WSP O365\WSP_Projects\TfN NorMITs Demand Partner 2022\DLog Matrix\Outputs\DLogTEM\comparisons",
    )

    try:
        main(parameters)
    except Exception:
        LOG.critical("Critical error occurred", exc_info=True)
        raise


##### MAIN #####
if __name__ == "__main__":
    _run()
