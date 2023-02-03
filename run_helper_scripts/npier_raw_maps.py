# -*- coding: utf-8 -*-
"""Create heatmaps for NPIER raw population and employment."""

##### IMPORTS #####
# Standard imports
import pathlib
import sys


# Third party imports
import geopandas as gpd
import pandas as pd

# Local imports
sys.path.append("..")
sys.path.append(".")
sys.path.append("./run_checker_scripts")
# pylint: disable=import-error,wrong-import-position
import eddie_npier_infill
import normits_demand as nd
from normits_demand import logging as nd_log
from normits_demand.utils import config_base, pandas_utils, plots

# pylint: enable=import-error,wrong-import-position

##### CONSTANTS #####
LOG = nd_log.get_logger(nd_log.get_package_logger_name() + ".NPIER_mapping")
LOG_FILE = "NPIER_mapping.log"
CA_SECTOR_NAME = "ca_sector_2020"
NPIER_ZONING = "lad_2020"
NPIER_LAD_COLUMN = "npier_lad"
LAD_NAME_COLUMN = "lad_name"


##### CLASSES #####
class NPIERMappingParams(config_base.BaseConfig):
    npier_file: eddie_npier_infill.RawTransformationalParameters
    output_folder: pathlib.Path
    plot_years: list[str]
    base_year: str


##### FUNCTIONS #####
def load_npier(
    npier_file: eddie_npier_infill.RawTransformationalParameters,
) -> dict[str, pd.DataFrame]:
    npier_data, _ = eddie_npier_infill.load_raw_transformational(npier_file)

    npier_lads = {}
    for nm in ("population", "employment"):
        df: pd.DataFrame = getattr(npier_data, nm).reset_index()
        df.loc[:, NPIER_LAD_COLUMN] = _fix_lad_names(df[NPIER_LAD_COLUMN])
        npier_lads[nm] = df.set_index(NPIER_LAD_COLUMN)

    return npier_lads


def translate_npier(
    npier: pd.DataFrame,
    translation: pd.DataFrame,
    translation_join_column: str,
    new_zone_column: str,
    factor_column: str,
    output_file: pathlib.Path,
) -> pd.DataFrame:
    data_columns = npier.columns.tolist()

    # Inner join because we expect only Northern values in NPIER but
    # translation will have all GB LADs
    merged = pandas_utils.fuzzy_merge(
        translation,
        npier.reset_index(),
        translation_join_column,
        NPIER_LAD_COLUMN,
        validate="m:1",
        indicator=True,
    )

    merged.loc[:, data_columns] = merged[data_columns].mul(merged[factor_column], axis=0)
    grouped = merged.groupby(new_zone_column)[data_columns].sum()

    # Comparing merged totals to NPIER North value
    # (ignoring the non-LADs which are present in NPIER)
    npier.loc["North"]
    grouped.sum(axis=0)

    grouped.to_csv(output_file)
    LOG.info("Saved: %s", output_file)
    return grouped


def _fix_lad_names(lads: pd.Series) -> pd.Series:
    return lads.str.replace(
        r"^\s*kingston\s+upon\s+hull(?:,?\s+city\s+of)?\s*$",
        "kingston upon hull",
        regex=True,
        case=True,
    )


def plot_heatmaps(
    data: pd.DataFrame,
    geodata: gpd.GeoDataFrame,
    data_join_column: str,
    plot_columns: list[str],
    output_folder: pathlib.Path,
    title: str,
    legend_label_fmt: str,
) -> None:
    npier_geodata = data.merge(
        geodata, left_on=data_join_column, right_index=True, validate="1:1"
    )
    npier_geodata = gpd.GeoDataFrame(npier_geodata, crs=geodata.crs, geometry="geometry")

    for column in plot_columns:
        fig = plots._heatmap_figure(
            npier_geodata,
            column,
            f"{title} - {column}",
            n_bins=5,
            positive_negative_colormaps=True,
            zoomed_bounds=None,
            legend_label_fmt=legend_label_fmt,
        )
        path = output_folder / (title + f"-{column}.png")
        fig.savefig(path)
        LOG.info("Saved: %s", path)


def calculate_growth(
    data: pd.DataFrame, base_year: str, output_file: pathlib.Path
) -> pd.DataFrame:
    sector_growth = data.div(data[base_year], axis=0) - 1
    sector_growth.to_csv(output_file)
    LOG.info("Saved: %s", output_file)

    return sector_growth


def main(params: NPIERMappingParams, init_logger: bool = True) -> None:
    params.output_folder.mkdir(exist_ok=True, parents=True)

    if init_logger:
        nd_log.get_logger(
            nd_log.get_package_logger_name(),
            params.output_folder / LOG_FILE,
            "Running NPIER Mapping",
            log_version=True,
        )
        nd_log.capture_warnings(
            file_handler_args=dict(log_file=params.output_folder / LOG_FILE)
        )

    npier_lads = load_npier(params.npier_file)

    lads = nd.get_zoning_system(NPIER_ZONING)
    sectors = nd.get_zoning_system(CA_SECTOR_NAME)
    sectors_meta = sectors.get_metadata()
    sectors_geodata = plots.get_geo_data(
        plots.GeoSpatialFile(
            path=sectors_meta.shapefile_path, id_column=sectors_meta.shapefile_id_col
        )
    )

    translation = lads._get_translation_definition(sectors)
    translation.loc[:, LAD_NAME_COLUMN] = translation[lads.col_name].replace(
        lads.zone_to_description_dict
    )
    translation.loc[:, LAD_NAME_COLUMN] = _fix_lad_names(translation[LAD_NAME_COLUMN])

    for landuse, npier_data in npier_lads.items():
        LOG.info("Plotting %s sectors", landuse)
        output_folder = params.output_folder / landuse
        output_folder.mkdir(exist_ok=True, parents=True)

        npier_sectors = translate_npier(
            npier_data,
            translation,
            LAD_NAME_COLUMN,
            sectors.col_name,
            f"{lads.name}_to_{sectors.name}",
            output_folder / f"{landuse}_sector_totals.csv",
        )

        plot_heatmaps(
            npier_sectors,
            sectors_geodata,
            sectors.col_name,
            params.plot_years,
            output_folder,
            f"NPIER {landuse.title()}",
            legend_label_fmt="{:.3g}",
        )

        LOG.info("Plotting %s sector growth", landuse)
        output_folder = params.output_folder / f"{landuse} growth"
        output_folder.mkdir(exist_ok=True)
        sector_growth = calculate_growth(
            npier_sectors, params.base_year, output_folder / f"{landuse}_sector_growth.csv"
        )

        plot_heatmaps(
            sector_growth,
            sectors_geodata,
            sectors.col_name,
            params.plot_years,
            output_folder,
            f"NPIER {landuse.title()} Growth",
            legend_label_fmt="{:.0%}",
        )


##### MAIN #####
if __name__ == "__main__":
    parameters = NPIERMappingParams(
        npier_file=eddie_npier_infill.RawTransformationalParameters(
            npier_data_workbook=r"I:\Data\EDDIE Inputs\NPIER Transformational\NPIER Transformational Raw Inputs\NPIER Technical Update database (November 2019).xlsx",
            npier_regions_workbook=r"I:\Data\EDDIE Inputs\NPIER Transformational\NPIER Transformational Raw Inputs\NPIER 2019 Rest of UK regions .xlsx",
        ),
        output_folder=".temp/NPIER mapping",
        plot_years=[2018, 2038, 2040, 2050],
        base_year=2018,
    )
    main(parameters)
