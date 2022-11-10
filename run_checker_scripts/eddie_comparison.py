# -*- coding: utf-8 -*-
"""Module for creating inputs for EDDIE from TfN land use data."""

##### IMPORTS #####
# Standard imports
import pathlib
import re
import sys
from typing import NamedTuple

# Third party imports
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd

# Local imports
sys.path.append("..")
sys.path.append(".")
# pylint: disable=import-error, wrong-import-position
import normits_demand as nd
from normits_demand import logging as nd_log
from normits_demand.utils import config_base, file_ops, plots, pandas_utils
from normits_demand.core.enumerations import LandUseType

# pylint: enable=import-error, wrong-import-position

##### CONSTANTS #####
LOG = nd_log.get_logger(nd_log.get_package_logger_name() + ".run_traveller_segmentation")
TFN_ZONE_SYSTEM = "msoa"
EDDIE_ZONE_SYSTEM = "lad_2017"
EDDIE_LAD_LOOKUP_SHEET = "I; Map Full LA-reduced LA list"

##### CLASSES #####
class EDDIEComparisonParameters(config_base.BaseConfig):
    eddie_file: pathlib.Path = pathlib.Path(
        r"C:\WSP_Projects\TfN Secondment\NorMITs-Demand"
        r"\Codebase\EDDIE\EDDIE_Proper_April22_DN.xlsm"
    )
    eddie_sheet: str = "I; CEBR_LA_EMP_POP_GDP_Central"
    eddie_header_row: int = 17
    tfn_base_folder: pathlib.Path = pathlib.Path(
        r"C:\WSP_Projects\TfN Secondment\NorMITs-Demand"
        r"\Inputs\NorMITs Land Use\import\scenarios"
    )
    scenario: nd.Scenario = nd.Scenario.SC04_UZC
    output_folder: pathlib.Path = pathlib.Path(
        r"C:\WSP_Projects\TfN Secondment\NorMITs-Demand"
        r"\Codebase\EDDIE\TfN Land Use Comparison"
    )
    map_years: list[int] = [2018, 2038]


class EDDIELandUseData(NamedTuple):
    sheet_name: str
    data: dict[LandUseType, pd.DataFrame]
    excel_row_lookup: pd.DataFrame = NotImplemented


class LandUseData(NamedTuple):
    scenario: nd.Scenario
    data: dict[LandUseType, pd.DataFrame]


##### FUNCTIONS #####
def load_eddie_lad_lookup(path: pathlib.Path) -> pd.DataFrame:
    LOG.info("Loading EDDIE LAD lookup from '%s' in '%s'", EDDIE_LAD_LOOKUP_SHEET, path)
    lookup = pd.read_excel(path, sheet_name=EDDIE_LAD_LOOKUP_SHEET, usecols="H:K", skiprows=14)
    lookup = lookup.dropna(how="all")
    lookup = pandas_utils.column_name_tidy(lookup)

    correct_codes = lookup["lad13cd"].str.match(r"^[EWS]\d+$")
    LOG.warning(
        "Dropping %s EDDIE LADs with incorrect codes:\n%s",
        np.sum(~correct_codes),
        lookup.loc[~correct_codes],
    )
    return lookup.loc[correct_codes]


def load_landuse_eddie(
    path: pathlib.Path, sheet_name: str, header_row: int
) -> EDDIELandUseData:
    lad_lookup = load_eddie_lad_lookup(path)

    LOG.info("Loading EDDIE data from '%s' in '%s'", sheet_name, path)
    data = pd.read_excel(path, sheet_name=sheet_name, skiprows=header_row - 1, na_values="-")
    data = pandas_utils.column_name_tidy(data)

    unnamed = [c for c in data.columns if c.startswith("unnamed:")]
    data = data.drop(columns=unnamed)
    data = data.dropna(axis=1, how="all")
    data = data.dropna(axis=0, how="all")

    fill_cols = ["variable", "region"]
    data.loc[:, fill_cols] = data[fill_cols].fillna(method="ffill")
    data = data.dropna(subset="local_authority")

    # Calculate yearly average
    years: dict[str, list[str]] = {}
    for col in data.columns:
        year = re.match(r"(\d{4})_q\d", col, re.I)
        if year is None:
            continue

        if year.group(1) in years:
            years[year.group(1)].append(year.group(0))
        else:
            years[year.group(1)] = [year.group(0)]

    for yr, columns in years.items():
        data.loc[:, yr] = data[columns].mean(axis=1)
        data = data.drop(columns=columns)

    # Add zone code column to index
    lad_replace = lad_lookup.set_index("cebr_lad")["lad13cd"].to_dict()
    zone_col_name = f"{EDDIE_ZONE_SYSTEM}_zone_id"
    data.loc[:, zone_col_name] = data["local_authority"].replace(lad_replace)

    # TODO(MB) Keep track of original Excel row numbers for replacing data
    data = data.set_index(["variable", "region", "local_authority", "units", zone_col_name])

    return EDDIELandUseData(
        sheet_name,
        {LandUseType.EMPLOYMENT: data.loc["Emp"], LandUseType.POPULATION: data.loc["Pop"]},
    )


def translate_tfn_landuse(
    data: pd.DataFrame,
    from_zone: nd.ZoningSystem,
    to_zone: nd.ZoningSystem,
    index_columns: list[str],
    data_columns: list[str],
) -> pd.DataFrame:
    LOG.info("Translating land use from '%s' to '%s' zoning", from_zone.name, to_zone.name)

    translation = from_zone._get_translation_definition(to_zone)

    translated = data.merge(translation, on=from_zone.col_name, validate="m:1")
    factor_column = f"{from_zone.name}_to_{to_zone.name}"
    for col in data_columns:
        translated.loc[:, col] = translated[col] * translated[factor_column]

    translated = translated.drop(columns=[from_zone.col_name, factor_column])
    return translated.groupby([to_zone.col_name] + index_columns).sum()


def load_landuse_tfn(folder: pathlib.Path, scenario: nd.Scenario):
    landuse: dict[LandUseType, pd.DataFrame] = {}
    from_zone = nd.get_zoning_system(TFN_ZONE_SYSTEM)
    to_zone = nd.get_zoning_system(EDDIE_ZONE_SYSTEM)

    # landuse_indices = {LandUseType.POPULATION: ["soc", "ns"], LandUseType.EMPLOYMENT: ["soc"]}
    # For now just aggregating to population and employment totals
    landuse_indices = {LandUseType.POPULATION: [], LandUseType.EMPLOYMENT: []}

    for pop_emp, index_columns in landuse_indices.items():
        file = folder / pop_emp.value / "future_growth_values.csv"
        LOG.info("Loading %s land use data from '%s'", pop_emp.value, file)
        data = file_ops.read_df(file)

        # Convert to DVectors and translate to LAD
        years = []
        for col in data.columns:
            year = re.match(r"\d{2,4}", col)
            if year is None:
                continue
            years.append(year.group(0))

        landuse[pop_emp] = translate_tfn_landuse(
            data.loc[:, [from_zone.col_name] + index_columns + years],
            from_zone,
            to_zone,
            index_columns,
            years,
        )

    return LandUseData(scenario, landuse)


def compare_landuse(
    eddie: EDDIELandUseData, tfn: LandUseData, output_file_base: pathlib.Path
) -> dict[LandUseType, pd.DataFrame]:
    """Compare EDDIE to TfN land use data in Excel workbook.

    Outputs a spreadsheet for each land use type containing
    4 sheets: EDDIE, TfN, Difference and % Difference.

    Parameters
    ----------
    eddie : EDDIELandUseData
        Land use data taken from the EDDIE spreadsheet.
    tfn : LandUseData
        TfN's land use data.
    output_file_base : pathlib.Path
        Base filepath used for naming Excel workbooks..

    Returns
    -------
    dict[LandUseType, pd.DataFrame]
        Land use data with 4 index columns ('region', 'local_authority',
        'units' and 'lad_2020_zone_id') with column groups 'EDDIE', 'TfN',
        'Difference' and '% Difference', each containing a column for
        each year.
    """
    output_file_base.parent.mkdir(exist_ok=True, parents=True)

    LOG.info("Comparing EDDIE to TfN land use")
    comparison: dict[LandUseType, pd.DataFrame] = {}
    for pop_emp in LandUseType.to_list():
        eddie_data = eddie.data[pop_emp].copy()
        eddie_data.columns = pd.MultiIndex.from_product((("EDDIE",), eddie_data.columns))
        tfn_data = tfn.data[pop_emp].copy()
        tfn_data.columns = pd.MultiIndex.from_product((("TfN",), tfn_data.columns))

        index_columns = eddie_data.index.names
        merged = eddie_data.reset_index().merge(
            tfn_data.reset_index(),
            how="outer",
            on=(nd.get_zoning_system(EDDIE_ZONE_SYSTEM).col_name,),
            validate="1:1",
            indicator=True,
        )

        merged = merged.set_index([(c,) for c in index_columns])
        merged.index.names = index_columns

        join_check = merged["_merge"].to_frame(name="Data Found")
        join_check.loc[:, "Data Found"] = join_check["Data Found"].replace(
            {"both": "Both", "left_only": "EDDIE only", "right_only": "TfN only"}
        )
        merged = merged.drop(columns="_merge")

        merge_stats = dict(zip(*np.unique(join_check["Data Found"], return_counts=True)))
        if merge_stats.get("EDDIE only", 0) > 0 or merge_stats.get("TfN only", 0) > 0:
            LOG.warning(
                "EDDIE & TfN do not share all LADs, counts of LADs found in %s", merge_stats
            )

        # Calculate region totals
        units = merged.index.get_level_values("units")[0]
        index_slice = pd.IndexSlice["Great Britain", "Total", units, "N/A"]
        merged.loc[index_slice, :] = merged.sum(axis=0)
        for region in merged.index.get_level_values("region").unique():
            if region == "Great Britain":
                continue
            index_slice = pd.IndexSlice[region, "Total", units, "N/A"]

            if isinstance(region, float) and np.isnan(region):
                # Append total to merged because cannot insert using loc with NaN
                total = (
                    merged.loc[region]
                    .sum(axis=0)
                    .to_frame(name=(region, "Total", units, "N/A"))
                    .T
                )
                merged = pd.concat([merged, total])
            else:
                merged.loc[index_slice, :] = merged.loc[region].sum(axis=0)

        for _, yr in eddie_data.columns:
            try:
                tfn_col = merged[("TfN", yr)]
            except KeyError:
                continue

            eddie_col = merged[("EDDIE", yr)]
            merged.loc[:, ("Difference", yr)] = tfn_col - eddie_col

            perc_diff = (
                np.divide(
                    tfn_col,
                    eddie_col,
                    out=np.ones_like(tfn_col.values),
                    where=eddie_col > 0,
                )
                - 1
            )
            merged.loc[:, ("% Difference", yr)] = np.where(
                np.isnan(tfn_col) | np.isnan(eddie_col), np.nan, perc_diff
            )

        comparison[pop_emp] = merged

        file = output_file_base.with_name(output_file_base.stem + f"-{pop_emp.value}.xlsx")
        LOG.info("Writing: %s", file)
        with pd.ExcelWriter(file) as excel:
            for name in merged.columns.get_level_values(0).unique():
                merged.loc[:, name].to_excel(excel, sheet_name=name)

            join_check.to_excel(excel, sheet_name="LAD Join")

    return comparison


def comparison_heatmaps(
    comparisons: dict[LandUseType, pd.DataFrame],
    geodata: gpd.GeoDataFrame,
    geom_id_column: str,
    years: list[int],
    output_file_base: pathlib.Path,
) -> None:
    years = [str(y) for y in years]
    LOG.info("Plotting EDDIE vs TfN comparisons for %s", ", ".join(years))

    plt.rcParams["figure.facecolor"] = "w"

    for pop_emp, data in comparisons.items():
        data = data.loc[:, "% Difference"].reset_index()
        data = data.loc[:, [geom_id_column] + years]
        data = data.loc[data[geom_id_column] != "N/A"]

        geodata = geodata[[geom_id_column, "geometry"]].merge(
            data, on=geom_id_column, how="left", validate="1:1"
        )

        for yr in years:
            fig = plots._heatmap_figure(
                geodata,
                yr,
                f"EDDIE vs TfN {pop_emp.value.title()} Comparison - {yr}",
                n_bins=5,
                positive_negative_colormaps=True,
            )
            file = output_file_base.with_name(
                output_file_base.stem + f"-{pop_emp.value}_{yr}.png"
            )
            fig.savefig(file)
            LOG.info("Written: %s", file)


def write_eddie_format(
    comparisons: dict[LandUseType, pd.DataFrame], output_file: pathlib.Path
) -> None:
    LOG.info("Writing TfN land use to EDDIE format")
    with pd.ExcelWriter(output_file) as excel:
        for pop_emp, data in comparisons.items():
            data = data.loc[
                data.index.get_level_values(f"{EDDIE_ZONE_SYSTEM}_zone_id") != "N/A", "TfN"
            ]

            quarters = []
            for yr in data.columns:
                for q in range(1, 5):
                    series = data[yr].copy()
                    series.name = f"{yr} Q{q}"
                    quarters.append(series)

            data = pd.concat(quarters, axis=1)
            data.to_excel(excel, sheet_name=pop_emp.value.title())

    LOG.info("Written: %s", output_file)


def main(params: EDDIEComparisonParameters):
    eddie = load_landuse_eddie(params.eddie_file, params.eddie_sheet, params.eddie_header_row)
    tfn_folder = params.tfn_base_folder / params.scenario.value
    tfn = load_landuse_tfn(tfn_folder, params.scenario)

    output_file_base = params.output_folder / "EDDIE_TfN_landuse_comparison"
    comparisons = compare_landuse(eddie, tfn, output_file_base)
    write_eddie_format(comparisons, params.output_folder / "TfN_landuse-EDDIE_format.xlsx")

    eddie_zone = nd.get_zoning_system(EDDIE_ZONE_SYSTEM)
    eddie_zone_meta = eddie_zone.get_metadata()
    eddie_geom = gpd.read_file(eddie_zone_meta.shapefile_path)
    eddie_geom = eddie_geom.loc[:, [eddie_zone_meta.shapefile_id_col, "geometry"]]
    eddie_geom = eddie_geom.rename(
        columns={eddie_zone_meta.shapefile_id_col: eddie_zone.col_name}
    )

    comparison_heatmaps(
        comparisons, eddie_geom, eddie_zone.col_name, params.map_years, output_file_base
    )


if __name__ == "__main__":
    main(EDDIEComparisonParameters())
