# -*- coding: utf-8 -*-
"""Comparisons between the EDDIE DDGs from NPIER or original."""

##### IMPORTS #####
# Standard imports
import pathlib
import re
import sys

# Third party imports
import geopandas as gpd
import numpy as np
import pandas as pd

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
DDG_INDEX = ["cebr_lad", "lad13cd"]


##### CLASSES #####
class DDGComparisonParameters(config_base.BaseConfig):
    """Parameters for running EDDIE DDG comparison script."""

    output_folder: pathlib.Path
    npier_ddg_folder: pathlib.Path
    npier_scenario: str
    eddie_ddg_folder: pathlib.Path
    eddie_scenario: str
    date: str
    heatmap_years: list[int]
    comparison_ddgs: set[str]
    base_year: int


##### FUNCTIONS #####
def find_ddg_files(
    folder: pathlib.Path, date: str, scenario: str, ddg_filter: set[str]
) -> dict[str, pathlib.Path]:
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
    dict[str, pathlib.Path]
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
            files[data["data"]] = file

    LOG.info('Found %s DDGs in "%s"', len(files), folder)

    for nm, ls in invalid.items():
        if ls:
            LOG.debug("Found %s files with invalid %s", len(ls), nm)

    return files


def check_data_files(
    npier: dict[str, pathlib.Path], eddie: dict[str, pathlib.Path], ddg_filter: set[str]
) -> set:
    """Check that `npier` and `eddie` have the same keys.

    Log a warning message for any keys present in only one
    of the dictionaries.

    Parameters
    ----------
    npier, eddie : dict[str, pathlib.Path]
        Dictionary containing NPIER / EDDIE files, respectively,
        with DDG name as the keys.
    ddg_filter : set[str]
        DDG data to check is given for NPIER and EDDIE.

    Returns
    -------
    set
        Unique set of keys present in both dictionaries.
    """
    npier = {s.lower().strip() for s in npier}
    eddie = {s.lower().strip() for s in eddie}
    missing = {"NPIER": ddg_filter - npier, "EDDIE": ddg_filter - eddie}

    for nm, dt in missing.items():
        if dt:
            LOG.warning(
                "%s data type files found not found for %s: %s", len(dt), nm, ", ".join(dt)
            )

    return eddie & npier


def _read_ddg(file: pathlib.Path, model_name: str) -> pd.DataFrame:
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

    df = df.set_index(DDG_INDEX)
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
    data_name: str,
    output_file: pathlib.Path,
) -> pd.DataFrame:
    """Compare NPIER and EDDIE DDG files.

    Parameters
    ----------
    npier_file, eddie_file : pathlib.Path
        Path to the DDG files.
    data_name : str
        Name of the DDG being compared, for log messages.
    output_file : pathlib.Path
        Excel workbook to save the comparisons to, will
        be overwritten if exists.

    Returns
    -------
    pd.DataFrame
        Comparisons DataFrame containing the following 4 column groups:
        NPIER, EDDIE, NPIER - EDDIE, % NPIER - EDDIE. Each column group
        contains different columns for the different years.
    """
    LOG.info("Comparing %s DDG", data_name)

    npier = _read_ddg(npier_file, "NPIER")
    eddie = _read_ddg(eddie_file, "EDDIE")
    comparison = npier.merge(eddie, how="outer", on=DDG_INDEX)

    differences: dict[str, pd.DataFrame] = {
        "NPIER - EDDIE": comparison.loc[:, "NPIER"] - comparison.loc[:, "EDDIE"],
        "% NPIER - EDDIE": comparison.loc[:, "NPIER"].divide(comparison.loc[:, "EDDIE"]) - 1,
    }

    for nm, df in differences.items():
        df = df.dropna(axis=1, how="all")
        df.columns = pd.MultiIndex.from_product(((nm,), df.columns))
        differences[nm] = df
    differences["% NPIER - EDDIE"].fillna(0, inplace=True)
    differences["% NPIER - EDDIE"].replace([np.inf, -np.inf], np.nan, inplace=True)

    comparison = pd.concat([comparison] + list(differences.values()), axis=1)

    _multilevel_dataframe_to_excel(comparison, output_file)

    return comparison


def comparison_heatmap(
    data: pd.DataFrame,
    geodata: gpd.GeoDataFrame,
    geom_id_column: str,
    years: list[str],
    data_name: str,
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
    data_name : str
        Name of DDG data being used.
    output_file_base : pathlib.Path
        Base file path for writing the heatmap images.
    plot_data_column : str
        Column group in `data` to plot.
    legend_label_fmt : str
        Format for the numbers in the legend.
    """
    LOG.info("Creating heatmaps for %s", data_name)

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
            f"EDDIE vs NPIER {data_name} Comparison - {yr}",
            n_bins=5,
            positive_negative_colormaps=True,
            legend_label_fmt=legend_label_fmt,
            legend_title=f"{plot_data_column} {yr}",
            # bins=np.concatenate([neg_bins, pos_bins]),
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


def growth_comparison(
    data: pd.DataFrame, base_year: str, output_file: pathlib.Path, ddg_name: str
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

    Returns
    -------
    pd.DataFrame
        Comparisons DataFrame containing the following 3 column groups:
        NPIER, EDDIE, NPIER - EDDIE. Each column group contains different
        columns for the different years.
    """
    LOG.info("Comparing growth from %s for %s", base_year, ddg_name)

    growth: dict[str, pd.DataFrame] = {}
    for i in ("NPIER", "EDDIE"):
        df = data.loc[:, i].divide(data[(i, base_year)], axis=0)
        growth[i] = df

    growth["NPIER - EDDIE"] = growth["NPIER"] - growth["EDDIE"]

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
            "Running EDDIE & NPIER DDG Comparison",
            log_version=True,
        )
        nd_log.capture_warnings(
            file_handler_args=dict(log_file=params.output_folder / LOG_FILE)
        )

    ddg_filter = {str(s).strip().lower(): s for s in params.comparison_ddgs}
    npier_files = find_ddg_files(
        params.npier_ddg_folder, params.date, params.npier_scenario, set(ddg_filter)
    )
    eddie_files = find_ddg_files(
        params.eddie_ddg_folder, params.date, params.eddie_scenario, set(ddg_filter)
    )
    data_types = check_data_files(npier_files, eddie_files, set(ddg_filter))

    eddie_zone, eddie_geom = get_zoning(EDDIE_ZONE_SYSTEM)

    folders: dict[str, pathlib.Path] = {}
    for f in ("comparison", "growth"):
        folders[f] = params.output_folder / f"DDG {f.title()}"
        folders[f].mkdir(exist_ok=True)

    for dt in data_types:
        ddg_name = ddg_filter[dt]
        fname = f"DD_{params.date}_{params.npier_scenario}_{params.eddie_scenario}_{ddg_name}"
        out_file = folders["comparison"] / (fname + "_comparison.xlsx")

        comparison = compare_ddg(
            npier_files[ddg_name], eddie_files[ddg_name], ddg_name, out_file
        )
        comparison.index.set_names(eddie_zone.col_name, level=1, inplace=True)

        for plt_data in ("NPIER - EDDIE", "% NPIER - EDDIE"):
            plt_nm = plt_data.replace("%", "Percentage")
            out_file = (
                folders["comparison"]
                / f"{plt_nm} Heatmaps"
                / (out_file.stem + "_" + plt_nm.lower().replace(" ", "_"))
            )
            out_file.parent.mkdir(exist_ok=True)

            comparison_heatmap(
                comparison,
                eddie_geom,
                eddie_zone.col_name,
                [str(i) for i in params.heatmap_years],
                ddg_name,
                out_file,
                plt_data,
                "{:.1%}" if "%" in plt_data else "{:.3g}",
            )

        out_file = folders["growth"] / (fname + f"_{params.base_year}_growth_comparison.xlsx")
        growth = growth_comparison(
            comparison.loc[:, ["NPIER", "EDDIE"]], str(params.base_year), out_file, ddg_name
        )

        # TODO Growth heatmaps for NPIER, EDDIE and NPIER - EDDIE


if __name__ == "__main__":
    # TODO(MB) Add argument for config file path
    parameters = DDGComparisonParameters.load_yaml(CONFIG_FILE)

    main(parameters)
