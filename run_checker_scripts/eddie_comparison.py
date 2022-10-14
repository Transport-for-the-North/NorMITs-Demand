# -*- coding: utf-8 -*-
"""Module for creating inputs for EDDIE from TfN land use data."""

##### IMPORTS #####
# Standard imports
import pathlib
import re
import sys
from typing import NamedTuple

# Third party imports
import pandas as pd

# Local imports
sys.path.append("..")
sys.path.append(".")
# pylint: disable=import-error, wrong-import-position
import normits_demand as nd
from normits_demand import logging as nd_log
from normits_demand.utils import config_base, file_ops
from normits_demand.core.enumerations import LandUseType

# pylint: enable=import-error, wrong-import-position

##### CONSTANTS #####
LOG = nd_log.get_logger(nd_log.get_package_logger_name() + ".run_traveller_segmentation")
TFN_ZONE_SYSTEM = "msoa"
EDDIE_ZONE_SYSTEM = "lad_2020"

##### CLASSES #####
class EDDIEComparisonParameters(config_base.BaseConfig):
    eddie_file: pathlib.Path = pathlib.Path(
        r"C:\WSP_Projects\TfN Secondment\NorMITs-Demand\Codebase\EDDIE\EDDIE_Proper_April22_DN.xlsm"
    )
    eddie_sheet: str = "I; CEBR_LA_EMP_POP_GDP_Central"
    eddie_header_row: int = 17
    tfn_base_folder: pathlib.Path = pathlib.Path(
        r"C:\WSP_Projects\TfN Secondment\NorMITs-Demand\Inputs\NorMITs Land Use\import\scenarios"
    )
    scenario: nd.Scenario = nd.Scenario.SC04_UZC
    output_folder: pathlib.Path = pathlib.Path(
        r"C:\WSP_Projects\TfN Secondment\NorMITs-Demand\Codebase\EDDIE\TfN Land Use Comparison"
    )


class EDDIELandUseData(NamedTuple):
    sheet_name: str
    data: dict[LandUseType, pd.DataFrame]
    excel_row_lookup: pd.DataFrame = NotImplemented


class LandUseData(NamedTuple):
    scenario: nd.Scenario
    data: dict[LandUseType, pd.DataFrame]


##### FUNCTIONS #####
def load_landuse_eddie(
    path: pathlib.Path, sheet_name: str, header_row: int
) -> EDDIELandUseData:
    LOG.info("Loading EDDIE data from '%s' in '%s'", sheet_name, path)
    data = pd.read_excel(path, sheet_name=sheet_name, skiprows=header_row - 1, na_values="-")
    data.columns = [re.sub(r"\s+", "_", c.lower().strip()) for c in data.columns]

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
    zones = nd.get_zoning_system(EDDIE_ZONE_SYSTEM)
    desc_to_zone = {v: k for k, v in zones.zone_to_description_dict.items()}
    data.loc[:, zones.col_name] = data["local_authority"].replace(desc_to_zone)

    # TODO(MB) Keep track of original Excel row numbers for replacing data
    data = data.set_index(["variable", "region", "local_authority", "units", zones.col_name])

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


def compare_landuse(eddie: EDDIELandUseData, tfn: LandUseData, output_folder: pathlib.Path):
    output_folder.mkdir(exist_ok=True, parents=True)

    LOG.info("Comparing EDDIE to TfN land use")
    for pop_emp in LandUseType.to_list():
        eddie_data = eddie.data[pop_emp].copy()
        eddie_data.columns = pd.MultiIndex.from_product((("EDDIE",), eddie_data.columns))
        tfn_data = tfn.data[pop_emp].copy()
        tfn_data.columns = pd.MultiIndex.from_product((("TfN",), tfn_data.columns))

        index_columns = eddie_data.index.names
        merged = eddie_data.reset_index().merge(
            tfn_data.reset_index(),
            on=(nd.get_zoning_system(EDDIE_ZONE_SYSTEM).col_name,),
            validate="1:1",
        )
        merged = merged.set_index([(c,) for c in index_columns])
        merged.index.names = index_columns

        # Calculate region totals
        units = merged.index.get_level_values("units")[0]
        index_slice = pd.IndexSlice["Great Britain", "Total", units, "N/A"]
        merged.loc[index_slice, :] = merged.sum(axis=0)
        for region in merged.index.get_level_values("region").unique():
            if region == "Great Britain":
                continue
            index_slice = pd.IndexSlice[region, "Total", units, "N/A"]
            merged.loc[index_slice, :] = merged.loc[region].sum(axis=0)

        for _, yr in eddie_data.columns:
            try:
                tfn_col = merged[("TfN", yr)]
            except KeyError:
                continue

            merged.loc[:, ("Difference", yr)] = tfn_col - merged[("EDDIE", yr)]
            merged.loc[:, ("% Difference", yr)] = (tfn_col / merged[("EDDIE", yr)]) - 1

        file = output_folder / f"EDDIE_TfN_landuse_comparison-{pop_emp.value}.xlsx"
        LOG.info("Writing: %s", file)
        with pd.ExcelWriter(file) as excel:
            for name in merged.columns.get_level_values(0).unique():
                merged.loc[:, name].to_excel(excel, sheet_name=name)


def main(params: EDDIEComparisonParameters):
    eddie = load_landuse_eddie(params.eddie_file, params.eddie_sheet, params.eddie_header_row)
    tfn_folder = params.tfn_base_folder / params.scenario.value
    tfn = load_landuse_tfn(tfn_folder, params.scenario)

    compare_landuse(eddie, tfn, params.output_folder)


if __name__ == "__main__":
    main(EDDIEComparisonParameters())
