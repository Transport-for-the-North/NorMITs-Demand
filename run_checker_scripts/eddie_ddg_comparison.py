# -*- coding: utf-8 -*-
"""Comparisons between the EDDIE DDGs from NPIER or original."""

##### IMPORTS #####
from __future__ import annotations
import dataclasses

# Standard imports
import enum
import pathlib
import re
import sys
from typing import Optional

# Third party imports
import geopandas as gpd
import numpy as np
import pandas as pd
import pydantic

# Local imports
sys.path.append("..")
sys.path.append(".")
# pylint: disable=import-error, wrong-import-position
import normits_demand as nd
from normits_demand import logging as nd_log
from normits_demand.utils import config_base, file_ops, pandas_utils, plots

# pylint: enable=import-error, wrong-import-position


##### CONSTANTS #####
LOG = nd_log.get_logger(nd_log.get_package_logger_name() + ".ddg_comparison")
LOG_FILE = "EDDIE_DDG_comparison.log"
EDDIE_ZONE_SYSTEM = "lad_2017"
CONFIG_FILE = pathlib.Path(r"config\checker\EDDIE_DDG_comparison_parameters.yml")
DDG_FILENAME_FORMAT = r"^DD_(?P<date>\w+\d+)_(?P<scenario>\w+)_(?P<data>[\w\s{}]+)_LA$"
NA_VALUES = ["#DIV/0!"]


##### CLASSES #####
class DDGType(enum.StrEnum):
    """Distinguish between the pop emp index and normal DDGs."""

    EMP = "Emp"
    POP = "Pop"
    WORWAP = r"frac{WOR}{WAP}"
    EMPINDEX = "EmpIndex"
    POPLONGINDEX = "PopLongIndex"
    POPSHORTINDEX = "PopShortIndex"

    def get_ddg_index(self) -> list[str]:
        lookup = {
            **dict.fromkeys(
                [DDGType.EMP, DDGType.POP, DDGType.WORWAP], ["cebr_lad", "lad13cd"]
            ),
            **dict.fromkeys(
                [DDGType.EMPINDEX, DDGType.POPLONGINDEX, DDGType.POPSHORTINDEX],
                ["o_lad13cd", "d_lad13cd"],
            ),
        }
        return lookup[self]

    @classmethod
    def get(cls, name: str) -> DDGType:
        """Case insensitive way to get DDGType based on given name."""
        values_lookup = {d.value.lower().strip(): d for d in cls}
        try:
            return values_lookup[name.strip().lower()]
        except KeyError as error:
            raise KeyError(f"{name} is not a valid DDGType") from error


class RegionsDataParams(pydantic.BaseModel):
    lad_lookup: pathlib.Path
    shapefile: pathlib.Path
    id_column: str

    @pydantic.validator("lad_lookup", "shapefile")
    def _file_exists(  # pylint: disable=no-self-argument
        cls, value: pathlib.Path
    ) -> pathlib.Path:
        if not value.is_file():
            raise ValueError(f"file doesn't exist: {value}")
        return value


@dataclasses.dataclass
class RegionsData:
    lad_lookup: pd.DataFrame
    geodata: gpd.GeoDataFrame
    join_column: str


class DDGComparisonParameters(config_base.BaseConfig):
    """Parameters for running EDDIE DDG comparison script."""

    output_folder: pathlib.Path
    npier_ddg_folder: pathlib.Path
    npier_scenario: str
    eddie_ddg_folder: pathlib.Path
    eddie_scenario: str
    date: str
    heatmap_years: list[int]
    comparison_ddgs: set[DDGType]
    base_year: int
    regions_data: RegionsDataParams


@dataclasses.dataclass
class ComparisonData:
    base: pd.DataFrame
    regions: Optional[pd.DataFrame] = None


##### FUNCTIONS #####
def find_ddg_files(
    folder: pathlib.Path, date: str, scenario: str, ddg_filter: set[str]
) -> dict[DDGType, pathlib.Path]:
    """Find DDG CSV files which match `DDG_FILENAME_FORMAT`.

    Parameters
    ----------
    folder : pathlib.Path
        Folder to look for CSVs in.
    date : str
        Date of DDGs e.g. 'Nov21'.
    scenario : str
        Name of DDG scenario e.g. 'central'.
    ddg_filter : set[str]
        DDG data to keep, any other DDGs are ignored.

    Returns
    -------
    dict[DDGType, pathlib.Path]
        Dictionary containing paths to the CSVs for the
        given `date` and `scenario` with the key being the
        type of DDG.

    Raises
    ------
    ValueError
        If multiple of the same DDGs are found.
    """
    invalid = {"filename": [], "date": [], "scenario": []}
    files = {}

    for file in folder.glob("*.csv"):
        match = re.match(DDG_FILENAME_FORMAT, file.stem, re.I)
        if match is None:
            invalid["filename"].append(file)
            continue

        data = match.groupdict()

        if data["date"].lower() != date.lower().strip():
            invalid["date"].append(file)
            continue

        if data["scenario"].lower() != scenario.lower():
            invalid["scenario"].append(file)
            continue

        if data["data"] in files:
            raise ValueError(f"found duplicate files for {date} {scenario} {data['data']}")

        if data["data"].lower().strip() in ddg_filter:
            ddg_type = DDGType.get(data["data"].lower().strip())
            files[ddg_type] = file

    LOG.info('Found %s DDGs in "%s"', len(files), folder)

    for nm, ls in invalid.items():
        if ls:
            LOG.debug("Found %s files with invalid %s", len(ls), nm)

    return files


def check_data_files(
    npier: dict[DDGType, pathlib.Path],
    eddie: dict[DDGType, pathlib.Path],
    ddg_filter: set[str],
) -> set[DDGType]:
    """Check that `npier` and `eddie` have the same keys.

    Log a warning message for any keys present in only one
    of the dictionaries.

    Parameters
    ----------
    npier, eddie : dict[DDGType, pathlib.Path]
        Dictionary containing NPIER / EDDIE files, respectively,
        with DDG type as the keys.
    ddg_filter : set[DDGType]
        DDG data to check is given for NPIER and EDDIE.

    Returns
    -------
    set[DDGType]
        Unique set of keys present in both dictionaries.
    """
    npier_set = set(npier)
    eddie_set = set(eddie)
    missing = {"NPIER": ddg_filter - npier_set, "EDDIE": ddg_filter - eddie_set}

    for data_name, ddg_type in missing.items():
        if ddg_type:
            LOG.warning(
                "%s data type files found not found for %s: %s",
                len(ddg_type),
                data_name,
                ", ".join(ddg_type),
            )

    return eddie_set & npier_set


def _read_ddg(file: pathlib.Path, model_name: str, ddg_type: DDGType) -> pd.DataFrame:
    """Read DDG CSV file.

    Parameters
    ----------
    file : pathlib.Path
        Path to the CSV.
    model_name : str
        Name of the model this DDG is from ('EDDIE', 'NPIER'),
        used for naming the columns.

    Returns
    -------
    pd.DataFrame
        DDG DataFrame with index columns `DDG_INDEX` and
        multilevel columns where the first level is `model_name`
        and the second is the year.
    """
    df = file_ops.read_df(file, na_values=NA_VALUES)
    df = pandas_utils.tidy_dataframe(df)

    # Fix incorrect LAD column name in some DDGs
    df.rename(columns={"la13cd": "lad13cd"}, inplace=True)

    df = df.set_index(ddg_type.get_ddg_index())
    df.columns = pd.MultiIndex.from_product(([model_name], df.columns))

    return df


def _multilevel_dataframe_to_excel(df: pd.DataFrame, file: pathlib.Path, **kwargs) -> None:
    """Write dataframe with MultiLevel columns to separate sheets in Excel `file`.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing data with MultiLevel columns.
    file : pathlib.Path
        Excel file to write to.
    """
    with pd.ExcelWriter(file) as excel:
        for c in df.columns.get_level_values(0).unique():
            df.loc[:, c].to_excel(excel, sheet_name=c, **kwargs)

    LOG.info('Written "%s"', file)


def compare_ddg(
    npier_file: pathlib.Path,
    eddie_file: pathlib.Path,
    ddg_type: DDGType,
    output_file: pathlib.Path,
    npier_scenario: str,
    regions: Optional[RegionsData] = None,
) -> ComparisonData:
    """Compare NPIER and EDDIE DDG files.

    Parameters
    ----------
    npier_file, eddie_file : pathlib.Path
        Path to the DDG files.
    ddg_type : DDGType
        Type of the DDG being compared, for log messages.
    output_file : pathlib.Path
        Excel workbook to save the comparisons to, will
        be overwritten if exists.
    npier_scenario : str
        Name of the NPIER scenario, used for naming outputs.
    regions : RegionsData, optional
        Lookup to perform regions grouping, not done if not
        given.

    Returns
    -------
    ComparisonData
        Comparisons DataFrames containing the following 4 column groups:
        `npier_scenario`, EDDIE, {npier_scenario} - EDDIE,
        % {npier_scenario} - EDDIE. Each column group contains
        different columns for the different years.
    """
    LOG.info("Comparing %s DDG", ddg_type)

    npier = _read_ddg(npier_file, npier_scenario, ddg_type)
    eddie = _read_ddg(eddie_file, "EDDIE", ddg_type)
    joined = {"base": npier.merge(eddie, how="outer", on=ddg_type.get_ddg_index())}

    if regions:
        df = joined["base"].copy()
        df.index = df.index.droplevel(0)
        merged = df.merge(
            regions.lad_lookup[["lad17cd", regions.join_column]],
            left_index=True,
            right_on="lad17cd",
            validate="1:1",
        )
        merged = merged.drop(columns="lad17cd")

        merged = merged.groupby(regions.join_column, as_index=True).sum()
        merged.columns = pd.MultiIndex.from_tuples(merged.columns)
        joined["regions"] = merged

    diff_name = f"{npier_scenario} - EDDIE"
    perc_name = f"% {npier_scenario} - EDDIE"

    comparisons: dict[str, pd.DataFrame] = {}
    for data_name, comparison in joined.items():
        differences: dict[str, pd.DataFrame] = {
            diff_name: comparison.loc[:, npier_scenario] - comparison.loc[:, "EDDIE"],
            perc_name: comparison.loc[:, npier_scenario].divide(comparison.loc[:, "EDDIE"])
            - 1,
        }

        for nm, df in differences.items():
            df = df.dropna(axis=1, how="all")
            df.columns = pd.MultiIndex.from_product(((nm,), df.columns))
            differences[nm] = df
        differences[perc_name].fillna(0, inplace=True)
        differences[perc_name].replace([np.inf, -np.inf], np.nan, inplace=True)

        comparison = pd.concat([comparison] + list(differences.values()), axis=1)

        if data_name == "regions":
            _multilevel_dataframe_to_excel(
                comparison,
                output_file.with_name(output_file.stem + "-regions" + output_file.suffix),
            )
        else:
            _multilevel_dataframe_to_excel(comparison, output_file)

        comparisons[data_name] = comparison

    return ComparisonData(**comparisons)


def comparison_heatmap(
    data: pd.DataFrame,
    geodata: gpd.GeoDataFrame,
    geom_id_column: str,
    years: list[str],
    title: str,
    output_file_base: pathlib.Path,
    plot_data_column: str,
    legend_label_fmt: str,
) -> None:
    """Produce heatmaps for the comparison `data`.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the columns for comparison, should have
        2 levels of column where the first contains the value
        `plot_data_column` and the second contains the years.
    geodata : gpd.GeoDataFrame
        Geospatial data for the relevant area type.
    geom_id_column : str
        Name of the column which links `data` to `geodata`.
    years : list[str]
        Years to plot heatmaps for.
    title : str
        Title for the plots.
    output_file_base : pathlib.Path
        Base file path for writing the heatmap images.
    plot_data_column : str
        Column group in `data` to plot.
    legend_label_fmt : str
        Format for the numbers in the legend.
    """
    LOG.info("Creating heatmaps for %s", title)

    data = data.loc[:, plot_data_column].reset_index()
    data = data.loc[:, [geom_id_column] + years]
    data = data.loc[data[geom_id_column] != "N/A"]

    geodata = geodata[[geom_id_column, "geometry"]].merge(
        data, on=geom_id_column, how="left", validate="1:1"
    )

    for yr in years:
        # pos_bins = np.linspace(0, np.max(np.abs(geodata[yr])), 5)
        # neg_bins = np.sort(-1 * pos_bins)

        fig = plots._heatmap_figure(
            geodata,
            yr,
            f"{title} - {yr}",
            n_bins=5,
            positive_negative_colormaps=True,
            legend_label_fmt=legend_label_fmt,
            legend_title=f"{plot_data_column} {yr}",
            # bins=np.concatenate([neg_bins, pos_bins]),
            zoomed_bounds=plots.Bounds(290000, 340000, 550000, 670000),
        )

        file = output_file_base.with_name(output_file_base.stem + f"-{yr}.png")
        fig.savefig(file)
        LOG.info("Written: %s", file)


def get_zoning(zone_system: str) -> tuple[nd.ZoningSystem, gpd.GeoDataFrame]:
    """Load `zone_system` data and geospatial data.

    Parameters
    ----------
    zone_system : str
        Name of zoning system.

    Returns
    -------
    nd.ZoningSystem
        Zoning system with name `zone_system`.
    gpd.GeoDataFrame
        Geospatial data for `zone_system`.
    """
    zones = nd.get_zoning_system(zone_system)
    zones_meta = zones.get_metadata()

    geom = gpd.read_file(zones_meta.shapefile_path)
    geom = geom.loc[:, [zones_meta.shapefile_id_col, "geometry"]]
    geom = geom.rename(columns={zones_meta.shapefile_id_col: zones.col_name})

    return zones, geom


def get_regions_data(regions: RegionsDataParams) -> RegionsData:
    """Load regions shapefile and LAD lookup."""
    lookup = pd.read_csv(regions.lad_lookup)
    geodata = gpd.read_file(regions.shapefile)[[regions.id_column, "geometry"]]

    return RegionsData(lad_lookup=lookup, geodata=geodata, join_column=regions.id_column)


def growth_comparison(
    data: pd.DataFrame, base_year: str, output_file: pathlib.Path, ddg_name: str, npier_scenario: str
) -> pd.DataFrame:
    """Calculate DDG growth and compare between NPIER and EDDIE.

    Parameters
    ----------
    data : pd.DataFrame
        NPIER and EDDIE DDG values.
    base_year : str
        Year to calculate growth from.
    output_file : pathlib.Path
        Path to Excel file to save growth comparisons to.
    ddg_name : str
        Name of DDG, used for logging.
    npier_scenario : str
        Name of NPIER scenario, used for labelling columns

    Returns
    -------
    pd.DataFrame
        Comparisons DataFrame containing the following 3 column groups:
        `npier_scenario`, EDDIE, {npier_scenario} - EDDIE. Each column group
        contains different columns for the different years.
    """
    LOG.info("Comparing growth from %s for %s", base_year, ddg_name)

    growth: dict[str, pd.DataFrame] = {}
    for i in (npier_scenario, "EDDIE"):
        df = data.loc[:, i].divide(data[(i, base_year)], axis=0)
        growth[i] = df

    growth[f"{npier_scenario} - EDDIE"] = growth[npier_scenario] - growth["EDDIE"]

    for nm, df in growth.items():
        df = df.dropna(axis=1, how="all")
        df.columns = pd.MultiIndex.from_product(((nm,), df.columns))
        growth[nm] = df

    growth = pd.concat(growth.values(), axis=1)

    _multilevel_dataframe_to_excel(growth, output_file)

    return growth


def main(params: DDGComparisonParameters, init_logger: bool = True):
    """Compare EDDIE and NPIER DDGs and output summaries and heatmaps.

    Parameters
    ----------
    params : DDGComparisonParameters
        Parameters for the comparison process.
    init_logger : bool, default True
        Whether to initialise a logger.
    """
    if not params.output_folder.is_dir():
        params.output_folder.mkdir(parents=True)

    if init_logger:
        nd_log.get_logger(
            nd_log.get_package_logger_name(),
            params.output_folder / LOG_FILE,
            "Running EDDIE DDG Comparison",
            log_version=True,
        )
        nd_log.capture_warnings(
            file_handler_args=dict(log_file=params.output_folder / LOG_FILE)
        )

    ddg_filter = {str(s).strip().lower() for s in params.comparison_ddgs}
    npier_files = find_ddg_files(
        params.npier_ddg_folder, params.date, params.npier_scenario, ddg_filter
    )
    eddie_files = find_ddg_files(
        params.eddie_ddg_folder, params.date, params.eddie_scenario, ddg_filter
    )
    ddg_types = check_data_files(npier_files, eddie_files, ddg_filter)

    eddie_zone, eddie_geom = get_zoning(EDDIE_ZONE_SYSTEM)
    regions = get_regions_data(params.regions_data)

    folders: dict[str, pathlib.Path] = {}
    for f in ("comparison", "growth"):
        folders[f] = params.output_folder / f"DDG {f.title()}"
        folders[f].mkdir(exist_ok=True)

    # Cannot map other DDG types because they're matrices
    MAP_TYPES = [DDGType.EMP, DDGType.POP, DDGType.WORWAP]

    for ddg_type in ddg_types:
        fname = f"DD_{params.date}_{params.npier_scenario}_{params.eddie_scenario}_{ddg_type}"
        out_file = folders["comparison"] / (fname + "_comparison.xlsx")

        region_data = regions if ddg_type in MAP_TYPES else None

        comparison = compare_ddg(
            npier_files[ddg_type], eddie_files[ddg_type], ddg_type, out_file, params.npier_scenario, region_data
        )
        comparison.base.index.set_names(eddie_zone.col_name, level=1, inplace=True)

        if ddg_type in MAP_TYPES:
            for plt_data in (f"{params.npier_scenario} - EDDIE", f"% {params.npier_scenario} - EDDIE"):
                plt_nm = plt_data.replace("%", "Percentage")
                fname = (
                    out_file.stem + "_" + plt_nm.lower().replace(" - ", "-").replace(" ", "_")
                )
                out_file = folders["comparison"] / f"{plt_nm} Heatmaps" / fname
                out_file.parent.mkdir(exist_ok=True)

                comparison_heatmap(
                    comparison.base,
                    eddie_geom,
                    eddie_zone.col_name,
                    [str(i) for i in params.heatmap_years],
                    f"EDDIE vs {params.npier_scenario} {ddg_type} Comparison",
                    out_file,
                    plt_data,
                    "{:.1%}" if "%" in plt_data else "{:.3g}",
                )

                if comparison.regions is not None and region_data is not None:
                    comparison_heatmap(
                        comparison.regions,
                        region_data.geodata,
                        region_data.join_column,
                        [str(i) for i in params.heatmap_years],
                        f"EDDIE vs {params.npier_scenario} {ddg_type} Region Comparison",
                        out_file.with_name(out_file.stem + "-regions" + out_file.suffix),
                        plt_data,
                        "{:.1%}" if "%" in plt_data else "{:.3g}",
                    )

        out_file = folders["growth"] / (fname + f"_{params.base_year}_growth_comparison.xlsx")
        growth = growth_comparison(
            comparison.base.loc[:, [params.npier_scenario, "EDDIE"]],
            str(params.base_year),
            out_file,
            ddg_type,
            params.npier_scenario,
        )

        if ddg_type in MAP_TYPES:
            for plt_data in growth.columns.get_level_values(0).unique():
                fname = (
                    out_file.stem
                    + "_"
                    + plt_data.lower().replace(" - ", "-").replace(" ", "_")
                )
                out_file = folders["growth"] / f"{plt_data} Heatmaps" / fname
                out_file.parent.mkdir(exist_ok=True)

                comparison_heatmap(
                    growth,
                    eddie_geom,
                    eddie_zone.col_name,
                    [str(i) for i in params.heatmap_years],
                    f"{plt_data} Growth",
                    out_file,
                    plt_data,
                    "{:.1%}",
                )


if __name__ == "__main__":
    # TODO(MB) Add argument for config file path
    parameters = DDGComparisonParameters.load_yaml(CONFIG_FILE)

    main(parameters)
