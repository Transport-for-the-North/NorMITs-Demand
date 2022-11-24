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


##### FUNCTIONS #####
def find_ddg_files(folder: pathlib.Path, date: str, scenario: str) -> dict[str, pathlib.Path]:
    """Find DDG CSV files which match `DDG_FILENAME_FORMAT`.

    Parameters
    ----------
    folder : pathlib.Path
        Folder to look for CSVs in.
    date : str
        Date of DDGs e.g. 'Nov21'.
    scenario : str
        Name of DDG scenario e.g. 'central'.

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

        files[data["data"]] = file

    LOG.info('Found %s DDGs in "%s"', len(files), folder)

    for nm, ls in invalid.items():
        if ls:
            LOG.debug("Found %s files with invalid %s", len(ls), nm)

    return files


def check_data_files(npier: dict[str, pathlib.Path], eddie: dict[str, pathlib.Path]) -> set:
    """Check that `npier` and `eddie` have the same keys.

    Log a warning message for any keys present in only one
    of the dictionaries.

    Parameters
    ----------
    npier, eddie : dict[str, pathlib.Path]
        Dictionary containing NPIER / EDDIE files, respectively,
        with DDG name as the keys.

    Returns
    -------
    set
        Unique set of keys present in both dictionaries.
    """
    npier = set(npier)
    eddie = set(eddie)
    missing = {"NPIER": eddie - npier, "EDDIE": npier - eddie}

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
        NPIER, EDDIE, Difference, % Difference. Each column group
        contains different columns for the different years.
    """
    LOG.info("Comparing %s DDG", data_name)

    npier = _read_ddg(npier_file, "NPIER")
    eddie = _read_ddg(eddie_file, "EDDIE")
    comparison = npier.merge(eddie, how="outer", on=DDG_INDEX)

    differences = []
    perc_differences = []
    for _, yr in eddie.columns:
        try:
            npier_col = npier[("NPIER", yr)]
        except KeyError:
            continue

        eddie_col = eddie[("EDDIE", yr)]
        differences.append(pd.Series(npier_col - eddie_col, name=("Difference", yr)))
        perc_diff = (
            np.divide(
                npier_col,
                eddie_col,
                out=np.ones_like(npier_col.values),
                where=eddie_col > 0,
            )
            - 1
        )
        perc_differences.append(
            pd.Series(
                np.where(np.isnan(npier_col) | np.isnan(eddie_col), np.nan, perc_diff),
                name=("% Difference", yr),
                index=perc_diff.index,
            )
        )

    comparison = pd.concat([comparison] + differences + perc_differences, axis=1)

    with pd.ExcelWriter(output_file) as excel:
        for c in comparison.columns.get_level_values(0).unique():
            comparison.loc[:, c].to_excel(excel, sheet_name=c)

    LOG.info('Written "%s"', output_file)

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
            # bins=np.concatenate([neg_bins, pos_bins]),
        )

        file = output_file_base.with_name(output_file_base.stem + f"-{yr}.png")
        fig.savefig(file)
        LOG.info("Written: %s", file)


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

    npier_files = find_ddg_files(params.npier_ddg_folder, params.date, params.npier_scenario)
    eddie_files = find_ddg_files(params.eddie_ddg_folder, params.date, params.eddie_scenario)
    data_types = check_data_files(npier_files, eddie_files)

    eddie_zone = nd.get_zoning_system(EDDIE_ZONE_SYSTEM)
    eddie_zone_meta = eddie_zone.get_metadata()
    eddie_geom = gpd.read_file(eddie_zone_meta.shapefile_path)
    eddie_geom = eddie_geom.loc[:, [eddie_zone_meta.shapefile_id_col, "geometry"]]
    eddie_geom = eddie_geom.rename(
        columns={eddie_zone_meta.shapefile_id_col: eddie_zone.col_name}
    )

    for dt in data_types:
        out_file = (
            params.output_folder / f"DD_{params.date}_{params.npier_scenario}"
            f"_{params.eddie_scenario}_{dt}_comparison.xlsx"
        )
        comparison = compare_ddg(npier_files[dt], eddie_files[dt], dt, out_file)
        comparison.index.set_names(eddie_zone.col_name, level=1, inplace=True)

        for plt_data in ("Difference", "% Difference"):
            plt_nm = plt_data.replace("%", "Percentage")
            out_file = (
                params.output_folder
                / f"{plt_nm} Heatmaps"
                / (out_file.stem + "_" + plt_nm.lower().replace(" ", "_"))
            )
            out_file.parent.mkdir(exist_ok=True)

            comparison_heatmap(
                comparison,
                eddie_geom,
                eddie_zone.col_name,
                [str(i) for i in params.heatmap_years],
                dt,
                out_file,
                plt_data,
                "{:.1%}" if "%" in plt_data else "{:.3g}",
            )


if __name__ == "__main__":
    # TODO(MB) Add argument for config file path
    parameters = DDGComparisonParameters.load_yaml(CONFIG_FILE)

    main(parameters)
