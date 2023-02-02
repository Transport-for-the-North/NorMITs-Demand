# -*- coding: utf-8 -*-
"""
Module for creating inputs for EDDIE from TfN land use data.

Flowchart detailing the methodolgy in the module is given here:
`docs\op_modelser\Misc\EDDIE_inputs.drawio`.
"""

##### IMPORTS #####
# Standard imports
from __future__ import annotations
import enum
import os
import pathlib
import re
import string
import sys
from typing import Any, NamedTuple, Optional, TypeVar

# Third party imports
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pydantic
from pydantic import fields
import geopandas as gpd

# Local imports
sys.path.append("..")
sys.path.append(".")
# pylint: disable=import-error, wrong-import-position
import normits_demand as nd
from normits_demand import logging as nd_log
from normits_demand.utils import config_base, file_ops, plots, pandas_utils
from normits_demand.core import enumerations
from normits_demand.core.enumerations import LandUseType

# pylint: enable=import-error, wrong-import-position

##### CONSTANTS #####
LOG = nd_log.get_logger(nd_log.get_package_logger_name() + ".eddie_comparison")
LOG_FILE = "EDDIE_comparison.log"
TFN_ZONE_SYSTEM = "msoa"
EDDIE_ZONE_SYSTEM = "lad_2017"
CONFIG_FILE = pathlib.Path(r"config\checker\EDDIE_comparison_parameters.yml")
EDDIE_EMPLOYMENT_HEADER_SKIP = 9
"""Number of header rows to skip in the occupation and industry sheets."""
CEBR_INDUSTRY_LOOKUP = {
    "Agriculture, forestry & fishing": "Agriculture & fishing",
    "Mining & Quarrying": "Other services",
    "Manufacturing": "Manufacturing",
    "Electricity, gas, steam and air ": "Energy & water",
    "Water supply": "Energy & water",
    "Construction": "Construction",
    "Wholesale and retail trade": "Distribution, hotels & restaurants",
    "Transportation and storage": "Transport & communication",
    "Accommodation and food service": "Distribution, hotels & restaurants",
    "Information and communication": "Transport & communication",
    "Financial and insurance ": "Banking, finance & insurance",
    "Real estate activities": "Other services",
    "Professional, scientific and tech": "Other services",
    "Administrative and support ": "Public admin, education & health",
    "Public administration and defence": "Public admin, education & health",
    "Education": "Public admin, education & health",
    "Human health and social work ": "Public admin, education & health",
    "Arts, entertainment and rec": "Other services",
    "Other service activities": "Other services",
}
"Lookup from NPIER industries to EDDIE industries."
MIN_YEAR = 2011
MAX_YEAR = 2043
OUTPUT_YEARS = range(MIN_YEAR, MAX_YEAR + 1)
YEARLY_COLUMNS = tuple(str(s) for s in OUTPUT_YEARS)
# Adds 4 quarters for every year except the final year which only adds Q1
QUARTERLY_COLUMNS = tuple(
    f"{y}_q{q}" for y in OUTPUT_YEARS for q in "1234" if y != MAX_YEAR or q not in "234"
)
NORTH_REGIONS = ("north east", "north west", "yorkshire & the humber")


##### CLASSES #####
class WorksheetParams(pydantic.BaseModel):
    """Parameters for an individual worksheet."""

    sheet_name: str
    header_row: int = 0
    index_columns: Optional[list[int]] = None
    column_letters: Optional[str] = None


class EDDIEWorkbookParams(pydantic.BaseModel):
    """Parameters for the worksheets in the EDDIE workbook."""

    landuse: WorksheetParams
    lad_lookup: WorksheetParams
    wor_wap: WorksheetParams
    employment_industry: WorksheetParams


class NPIERScenarioLandUseParameters(pydantic.BaseModel):
    """Parameters for the land use data from a NPIER scenario."""

    base_folder: pathlib.Path
    scenario: nd.Scenario

    _land_use_files: dict[LandUseType, pathlib.Path] = pydantic.PrivateAttr(None)

    @pydantic.validator("base_folder")
    def _check_folder(  # pylint: disable=no-self-argument
        cls, value: pathlib.Path
    ) -> pathlib.Path:
        """Raise ValueError if `value` isn't a folder."""
        if not value.is_dir():
            raise ValueError(f"folder doesn't exist: {value}")
        return value

    @staticmethod
    def _build_landuse_paths(
        base_folder: pathlib.Path, scenario: nd.Scenario
    ) -> dict[LandUseType, pathlib.Path]:
        folder = base_folder / scenario.value
        return {
            pop_emp: folder / pop_emp.value / "future_growth_values.csv"
            for pop_emp in LandUseType
        }

    @pydantic.root_validator(skip_on_failure=True)
    def _check_files(  # pylint: disable=no-self-argument
        cls, values: dict[str, Any]
    ) -> dict[str, Any]:
        missing = []
        for path in cls._build_landuse_paths(
            values.get("base_folder"), values.get("scenario")
        ).values():
            if not path.is_file:
                missing.append(f'"{path}"')

        if missing:
            raise ValueError("cannot find land use files:{}".format("\n\t".join(missing)))

        return values

    @property
    def land_use_files(self) -> dict[LandUseType, pathlib.Path]:
        """Paths to the NPIER land use files."""
        if self._land_use_files is None:
            self._land_use_files = self._build_landuse_paths(self.base_folder, self.scenario)
        return self._land_use_files.copy()


class RawTransformationalParameters(pydantic.BaseModel):
    """Parameters for the NPIER raw land use data from the Oxford economics workbooks."""

    npier_data_workbook: pathlib.Path
    npier_regions_workbook: pathlib.Path
    local_demographics_sheet: WorksheetParams = WorksheetParams(
        sheet_name="TRA - Local demog",
        header_row=5,
        index_columns=[0],
        column_letters="B,M:BA",
    )
    local_economics_sheet: WorksheetParams = WorksheetParams(
        sheet_name="TRA - Local econ head",
        header_row=5,
        index_columns=[0],
        column_letters="B, M:BA",
    )
    industry_employment_sheet: WorksheetParams = WorksheetParams(
        sheet_name="TRA - Industry detail, emp",
        header_row=5,
        index_columns=[0],
        column_letters="B, M:BA",
    )
    north_demographics_sheet: WorksheetParams = WorksheetParams(
        sheet_name="TRA - North demog",
        header_row=5,
        index_columns=[0],
        column_letters="B, M:BA",
    )
    regions_population_sheet: WorksheetParams = WorksheetParams(
        sheet_name="Population", header_row=5, index_columns=[0], column_letters="B, M:BA"
    )
    regions_employment_sheet: WorksheetParams = WorksheetParams(
        sheet_name="Employment", header_row=5, index_columns=[0], column_letters="B, M:BA"
    )

    @pydantic.validator("npier_data_workbook", "npier_regions_workbook")
    def _check_workbook_paths(  # pylint: disable=no-self-argument
        cls, value: pathlib.Path
    ) -> pathlib.Path:
        if not value.is_file():
            raise ValueError(f"file doesn't exist: {value}")
        if value.suffix.lower() != ".xlsx":
            raise ValueError(f"file isn't an existing Excel Workbook: {value}")

        return value


class NPIERInputType(enumerations.IsValidEnumWithAutoNameLower):
    NPIER_SCENARIO_LANDUSE = enum.auto()
    NPIER_RAW_TRANSFORMATIONAL = enum.auto()


class EDDIEComparisonParameters(config_base.BaseConfig):
    """Parameters for running EDDIE comparison script."""

    eddie_file: pathlib.Path
    workbook_parameters: EDDIEWorkbookParams
    npier_input: NPIERInputType
    npier_scenario_landuse: Optional[NPIERScenarioLandUseParameters] = None
    npier_raw_transformational: Optional[RawTransformationalParameters] = None
    output_folder: pathlib.Path
    map_years: list[int]
    lad_geospatial_file: Optional[plots.GeoSpatialFile] = None
    regions_geospatial_file: Optional[plots.GeoSpatialFile] = None

    @pydantic.validator(
        "npier_scenario_landuse", "npier_raw_transformational", pre=True, always=True
    )
    def _validate_npier_input(  # pylint: disable=no-self-argument
        cls, value: Any, field: fields.ModelField, values: dict[str, Any]
    ) -> Any:
        """Check if the required parameters are given for the NPIER input type."""
        # Ignore any data given if not using that input type
        npier_input: NPIERInputType = values.get("npier_input")
        if npier_input is None:
            raise ValueError("missing value npier_input")

        if field.name != npier_input.value:
            if value is not None:
                LOG.debug("ignoring %s value", field.name)
            return None

        if value is None:
            raise ValueError(f"required when {npier_input.value=}")

        return value

    # Makes a classmethod not recognised by pylint, hence disabling self check
    @pydantic.validator("eddie_file", "output_folder")
    def _expand_path(  # pylint: disable=no-self-argument
        cls, value: pathlib.Path
    ) -> pathlib.Path:
        """Expand environment variables in path and resolve."""
        try:
            expanded = string.Template(str(value)).substitute(os.environ)
        except KeyError as err:
            raise ValueError(f"missing environment variable {err}") from err

        return pathlib.Path(expanded).resolve()

    # Makes a classmethod not recognised by pylint, hence disabling self check
    @pydantic.validator("lad_geospatial_file", "regions_geospatial_file", pre=True)
    def _dict_to_tuple(cls, value: dict) -> tuple:  # pylint: disable=no-self-argument
        try:
            return value["path"], value["id_column"]
        except KeyError as err:
            raise TypeError(f"missing {err} value") from err


class EDDIELandUseData(NamedTuple):
    """Land use data loaded from the EDDIE workbook."""

    sheet_name: str
    data: dict[LandUseType, pd.DataFrame]
    excel_row_lookup: pd.DataFrame = NotImplemented


class LandUseData(NamedTuple):
    """Land use data from NPIER."""

    scenario: nd.Scenario
    data: dict[LandUseType, pd.DataFrame]


class DisaggregatedLandUse(NamedTuple):
    """Disaggregated land use data."""

    wap: pd.DataFrame
    wor: pd.DataFrame
    employment_industry: pd.DataFrame


class NPIERNorthernLandUse(NamedTuple):
    """Land use data from NPIER Oxford economics forecasts in the North."""

    population: pd.DataFrame
    employment: pd.DataFrame
    wap: pd.Series
    wor: pd.Series
    industry_employment: pd.DataFrame


class NPIERRegionsLandUse(NamedTuple):
    """Land use data from NPIER Oxford economics region forecasts."""

    population: pd.DataFrame
    employment: pd.DataFrame
    industry_employment: pd.DataFrame


NPIERLandUseType = TypeVar("NPIERLandUseType", NPIERRegionsLandUse, NPIERNorthernLandUse)


class NPIEREDDIEFormatLandUse(NamedTuple):
    population: pd.DataFrame
    employment: pd.DataFrame
    industry_employment: pd.DataFrame
    wap: Optional[pd.DataFrame] = None
    wor: Optional[pd.DataFrame] = None

    def copy(self) -> NPIEREDDIEFormatLandUse:
        return NPIEREDDIEFormatLandUse(
            population=self.population.copy(),
            employment=self.employment.copy(),
            wap=self.wap.copy(),
            wor=self.wor.copy(),
            industry_employment=self.industry_employment.copy(),
        )


class InfillMethod(enum.StrEnum):
    """Method to use when finding which Na columns require infilling."""

    ANY = enum.auto()
    ALL = enum.auto()
    NONE = enum.auto()


##### FUNCTIONS #####
def load_eddie_lad_lookup(path: pathlib.Path, sheet_params: WorksheetParams) -> pd.DataFrame:
    """Load LAD lookup data from EDDIE workbook.

    Parameters
    ----------
    path : pathlib.Path
        Path to EDDIE Workbook.
    sheet_params : WorksheetParams
        Parameters for the worksheet containing the LAD lookup.

    Returns
    -------
    pd.DataFrame
        LAD lookup with columns: "cebr_lad", "region", "lad13cd",
        "industry_and_occupation_local_authority_group".
    """
    LOG.info("Loading EDDIE LAD lookup from '%s' in '%s'", sheet_params.sheet_name, path)
    lookup = pd.read_excel(
        path,
        sheet_name=sheet_params.sheet_name,
        usecols="H:K",
        skiprows=sheet_params.header_row - 1,
    )
    lookup = pandas_utils.tidy_dataframe(lookup)

    correct_codes = lookup["lad13cd"].str.match(r"^[EWS]\d+$")
    LOG.warning(
        "Dropping %s EDDIE LADs with incorrect codes:\n%s",
        np.sum(~correct_codes),
        lookup.loc[~correct_codes],
    )
    return lookup.loc[correct_codes]


def load_landuse_eddie(
    path: pathlib.Path, sheet_params: WorksheetParams, lad_sheet_params: WorksheetParams
) -> EDDIELandUseData:
    """Read population and employment data from EDDIE workbook.

    Parameters
    ----------
    path : pathlib.Path
        Path to EDDIE workbook file.
    sheet_params : WorksheetParams
        Parameters for the worksheet containing the population
        and employment data.
    lad_sheet_params : WorksheetParams
        Parameters to the worksheet containing the LAD lookup.

    Returns
    -------
    EDDIELandUseData
        Land use data from EDDIE workbook.
    """
    lad_lookup = load_eddie_lad_lookup(path, lad_sheet_params)

    LOG.info("Loading EDDIE data from '%s' in '%s'", sheet_params.sheet_name, path)
    data = pd.read_excel(
        path,
        sheet_name=sheet_params.sheet_name,
        skiprows=sheet_params.header_row - 1,
        na_values="-",
    )
    data = pandas_utils.tidy_dataframe(data)

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
        # data = data.drop(columns=columns)

    # Add zone code column to index
    lad_replace = lad_lookup.set_index("cebr_lad")["lad13cd"].to_dict()
    zone_col_name = f"{EDDIE_ZONE_SYSTEM}_zone_id"
    data.loc[:, zone_col_name] = data["local_authority"].replace(lad_replace)

    # Update regions to use names from LAD lookup sheet, as
    # these are consistent across other worksheets
    region_replace = lad_lookup.set_index("lad13cd")["region"].to_dict()
    data.loc[:, "region"] = data[zone_col_name].replace(region_replace)

    # TODO(MB) Keep track of original Excel row numbers for replacing data
    data["region"] = _normalise_region_names(data["region"])
    data = data.set_index(["variable", "region", "local_authority", "units", zone_col_name])

    return EDDIELandUseData(
        sheet_params.sheet_name,
        {LandUseType.EMPLOYMENT: data.loc["Emp"], LandUseType.POPULATION: data.loc["Pop"]},
    )


def translate_tfn_landuse(
    data: pd.DataFrame,
    from_zone: nd.ZoningSystem,
    to_zone: nd.ZoningSystem,
    index_columns: list[str],
    data_columns: list[str],
) -> pd.DataFrame:
    """Perform zone translation on NPIER land use data.

    Parameters
    ----------
    data : pd.DataFrame
        Land use data at `from_zone` zone system to be translated.
    from_zone : nd.ZoningSystem
        Zone system `data` is given at.
    to_zone : nd.ZoningSystem
        Zone system to convert to.
    index_columns : list[str]
        Columns in `data` to be considered indices (no
        translation factor will be applied).
    data_columns : list[str]
        Columns containing data (translation factor will be applied).

    Returns
    -------
    pd.DataFrame
        `data` in the `to_zone` zone system.
    """
    LOG.info("Translating land use from '%s' to '%s' zoning", from_zone.name, to_zone.name)

    translation = from_zone._get_translation_definition(to_zone)

    translated = data.merge(translation, on=from_zone.col_name, validate="m:1")
    factor_column = f"{from_zone.name}_to_{to_zone.name}"
    for col in data_columns:
        translated.loc[:, col] = translated[col] * translated[factor_column]

    translated = translated.drop(columns=[from_zone.col_name, factor_column])
    return translated.groupby([to_zone.col_name] + index_columns).sum()


def load_landuse_tfn(npier_parameters: NPIERScenarioLandUseParameters) -> LandUseData:
    """Load NPIER landuse data.

    Parameters
    ----------
    npier_parameters : NPIERScenarioLandUseParameters
        Parameters for the NPIER land use files.

    Returns
    -------
    LandUseData
        NPIER population and employment land use data.
    """
    landuse: dict[LandUseType, pd.DataFrame] = {}
    from_zone = nd.get_zoning_system(TFN_ZONE_SYSTEM)
    to_zone = nd.get_zoning_system(EDDIE_ZONE_SYSTEM)

    # landuse_indices = {LandUseType.POPULATION: ["soc", "ns"], LandUseType.EMPLOYMENT: ["soc"]}
    # For now just aggregating to population and employment totals
    landuse_indices = {LandUseType.POPULATION: [], LandUseType.EMPLOYMENT: []}

    for pop_emp, index_columns in landuse_indices.items():
        file = npier_parameters.land_use_files[pop_emp]
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

    return LandUseData(npier_parameters.scenario, landuse)


def read_excel_sheet(
    file: pathlib.Path, parameters: WorksheetParams, **kwargs
) -> pd.DataFrame:
    tidy_args = ["rename", "drop_unnamed", "nan_columns", "nan_rows", "nan_index"]
    tidy_argument_values = {}
    for arg in tidy_args:
        if arg in kwargs:
            tidy_argument_values[arg] = kwargs.pop(arg)

    df = pd.read_excel(
        file,
        sheet_name=parameters.sheet_name,
        header=parameters.header_row,
        index_col=parameters.index_columns,
        usecols=parameters.column_letters,
        **kwargs,
    )

    return pandas_utils.tidy_dataframe(df, **tidy_argument_values)


def _read_raw_demographics(
    file: pathlib.Path, sheet_params: WorksheetParams
) -> tuple[pd.DataFrame, pd.DataFrame]:
    demographics = read_excel_sheet(file, sheet_params, nan_rows=False)
    # All values in demographics have units thousands
    demographics *= 1000
    demographics.index.names = ["npier_lad"]

    population_index = "Local area total population (thousands)"
    labour_index = "Local area labour force (thousands)"

    population = demographics.loc[population_index:labour_index].dropna(axis=0, how="all")
    labour = demographics.loc[labour_index:].dropna(axis=0, how="all")

    return population, labour


def _read_raw_economics(
    file: pathlib.Path, sheet_params: WorksheetParams
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    economics = read_excel_sheet(file, sheet_params, nan_rows=False)
    economics.index.names = ["npier_lad"]
    gva_index = "GVA (£ million, constant 2016 prices)"
    employment_index = "Employment (thousands, jobs)"
    productivity_index = "Productivity (£ thousands constant 2016 prices, GVA per job)"

    gva = economics.loc[gva_index:employment_index].dropna(axis=0, how="all")
    gva *= 1e6
    employment = economics.loc[employment_index:productivity_index].dropna(axis=0, how="all")
    employment *= 1000
    productivity = economics.loc[productivity_index:].dropna(axis=0, how="all")
    productivity *= 1000

    return gva, employment, productivity


def _read_north_wor_wap(
    file: pathlib.Path, sheet_params: WorksheetParams
) -> tuple[pd.Series, pd.Series]:
    wor_wap = read_excel_sheet(file, sheet_params, nan_rows=False)
    wor_wap *= 1000
    wap = wor_wap.loc["16-64 population"]
    wap.name = "NPIER"

    wor = wor_wap.loc[
        "Labour force by age and gender":"Participation rate by age and gender"
    ].dropna(axis=0, how="all")
    working_ages = ["16-19", "20-24", "25-34", "35-49", "50-64"]
    working_ages = [f"All persons - {i}" for i in working_ages]
    wor = wor.loc[working_ages].sum(axis=0)
    wor.name = "NPIER"
    return wor, wap


def _read_employment_industries(
    file: pathlib.Path, sheet_params: WorksheetParams, zone_names: list[str]
) -> pd.DataFrame:
    industries = read_excel_sheet(file, sheet_params, nan_rows=False)
    industries *= 1000

    indices: dict[int, str] = {}
    for zone in zone_names:
        i = industries.index.get_loc(zone)
        indices[i] = zone

    all_industries = []
    int_indices = sorted(indices)
    for i, j in zip(int_indices, int_indices[1:] + [np.inf]):
        zone = indices[i]

        if np.isinf(j):
            zone_industries = industries.iloc[i:]
        else:
            zone_industries = industries.iloc[i:j]
        zone_industries = zone_industries.loc["Industry section":].dropna(axis=0, how="all")
        zone_industries.index = pd.MultiIndex.from_product(((zone,), zone_industries.index))
        all_industries.append(zone_industries)

    industries = pd.concat(all_industries, axis=0)
    industries.index.names = ["npier_lad", "cebr_industry"]
    return industries


def _normalise_region_names(names: pd.Index | pd.Series) -> pd.Index:
    """Normalise names of UK regions in an index."""
    names = (
        names.str.lower()
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.replace(r"""[!"#$%'()*+,-./:;<=>?@[\\\]^_`{|}~]""", "", regex=True)
    )

    rename = {
        r"^united kindgom.*$": "uk",
        r"\smids$": " midlands",
        r"yorks\s+&?\s+hum": "yorkshire & the humber",
        r"^\s*east\s*$": "east of england",
    }
    for pattern, repl in rename.items():
        names = names.str.replace(pattern, repl, regex=True, case=False)
    return names


def _read_npier_regions(
    file: pathlib.Path, population_sheet: WorksheetParams, employment_sheet: WorksheetParams
) -> NPIERRegionsLandUse:
    population = read_excel_sheet(file, population_sheet)
    population.index.names = ["region"]
    population *= 1000

    region_names = population.index.tolist()
    # UK is named differently in the employment workbook
    region_names.remove("UK")
    region_names.append("United Kingdom (mainland)")

    industry_employment = _read_employment_industries(file, employment_sheet, region_names)
    industry_employment.index.names = ["region", "cebr_industry"]
    employment = industry_employment.groupby(level="region").sum()

    population.index = _normalise_region_names(population.index)
    employment.index = _normalise_region_names(employment.index)

    industry_employment.index = pd.MultiIndex.from_arrays(
        [
            _normalise_region_names(industry_employment.index.get_level_values("region")),
            industry_employment.index.get_level_values("cebr_industry"),
        ]
    )

    return NPIERRegionsLandUse(population, employment, industry_employment)


def load_raw_transformational(
    params: RawTransformationalParameters,
) -> tuple[NPIERNorthernLandUse, NPIERRegionsLandUse]:
    """Load raw NPIER land use data from Oxford economics workbook.

    Parameters
    ----------
    params : RawTransformationalParameters
        Parameters for the workbook.

    Returns
    -------
    NPIERNorthernLandUse
        Land use data at LAD level for the North.
    NPIERRegionsLandUse
        Land use data for all other regions.
    """
    population, _ = _read_raw_demographics(
        params.npier_data_workbook, params.local_demographics_sheet
    )
    _, employment, _ = _read_raw_economics(
        params.npier_data_workbook, params.local_economics_sheet
    )
    wor, wap = _read_north_wor_wap(params.npier_data_workbook, params.north_demographics_sheet)
    industries = _read_employment_industries(
        params.npier_data_workbook, params.industry_employment_sheet, employment.index.tolist()
    )

    northern_land_use = NPIERNorthernLandUse(population, employment, wap, wor, industries)
    regions_land_use = _read_npier_regions(
        params.npier_regions_workbook,
        params.regions_population_sheet,
        params.regions_employment_sheet,
    )

    return northern_land_use, regions_land_use


def _load_wor_wap(path: pathlib.Path, sheet_params: WorksheetParams) -> pd.DataFrame:
    """Load WOR and WAP data from EDDIE workbook.

    Parameters
    ----------
    path : pathlib.Path
        Path to EDDIE workbook file.
    sheet_params : WorksheetParams
        Parameters for the sheet containing the WOR / WAP data.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the WOR and WAP data, with 3 index columns
        ("description", "region" and "units") and columns for each year
        quarter e.g. "2011_q1".
    """
    wor_wap = pd.read_excel(
        path, sheet_name=sheet_params.sheet_name, skiprows=sheet_params.header_row - 1
    )

    wor_wap = pandas_utils.tidy_dataframe(wor_wap)
    wor_wap = wor_wap.drop(columns="variable")
    wor_wap.loc[:, "description"] = wor_wap["description"].fillna(method="ffill")

    index_columns = ["description", "region", "units"]
    wor_wap = wor_wap.dropna(axis=0, subset=index_columns, how="any")
    wor_wap = wor_wap.dropna(axis=1, how="all")
    wor_wap.loc[:, "region"] = _normalise_region_names(wor_wap["region"])
    wor_wap = wor_wap.set_index(index_columns)

    numeric_wor_wap = []
    for c in wor_wap.columns:
        numeric_wor_wap.append(pd.to_numeric(wor_wap[c]))

    return pd.concat(numeric_wor_wap, axis=1)


def load_disaggregated_eddie(
    path: pathlib.Path, workbook_params: EDDIEWorkbookParams
) -> DisaggregatedLandUse:
    """Load WOR, WAP and employment by industry and occupancy from EDDIE.

    Parameters
    ----------
    path : pathlib.Path
        Path to EDDIE workbook file.
    workbook_params : EDDIEWorkbookParams
        Parameters for the various worksheets in the file.

    Returns
    -------
    DisaggregatedLandUse
        DataFrames for the various disaggregated land use data.
    """
    wor_wap = _load_wor_wap(path, workbook_params.wor_wap)

    emp_params = {
        "industry": workbook_params.employment_industry,
    }
    emp_data = {}
    drop_columns = [
        "nts_/_rdfe_sector",
        "la+occupation_lookup",
        "base_year_contains_data?",
        "first_year_contains_data?",
        "units",
        "calendar_year",
    ]
    for name, sheet_params in emp_params.items():
        df = pd.read_excel(
            path, sheet_name=sheet_params.sheet_name, skiprows=sheet_params.header_row - 1
        )
        df = df.iloc[EDDIE_EMPLOYMENT_HEADER_SKIP:]
        rename = {}
        for nm, col in df.items():
            if str(nm).lower().startswith("unnamed") and isinstance(col.iloc[0], str):
                rename[nm] = col.iloc[0]
        df.rename(columns=rename, inplace=True)
        df = df.iloc[1:]

        df = pandas_utils.tidy_dataframe(df)
        df = df.drop(columns=drop_columns, errors="ignore")
        index_columns = ["region", "local_authority", f"cebr_{name}"]
        df = df.dropna(axis=0, how="any", subset=index_columns)

        df.loc[:, "region"] = _normalise_region_names(df["region"])
        df = df.set_index(index_columns)
        for c in df.columns:
            df.loc[:, c] = pd.to_numeric(df[c], downcast="unsigned", errors="coerce")

        emp_data[name] = df

    return DisaggregatedLandUse(
        wap=wor_wap.loc["Working-Age Population"],
        wor=wor_wap.loc["Employment (Residence-based)"] * 1000,
        employment_industry=emp_data["industry"],
    )


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
        # Only keep average year columns
        columns = [c for c in eddie_data.columns if re.match("^\d+$", c) is not None]
        eddie_data = eddie_data.loc[:, columns]

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


def _calculate_yearly_quarters(data: pd.DataFrame) -> pd.DataFrame:
    """Convert single year columns into quarters.

    All quarters for a single year are identical to the
    original year column.
    """
    quarters = []
    for yr in data.columns:
        previous = str(int(yr) - 1)
        if previous in data.columns:
            # Use previous year value for Q1 as we're dealing with financial years
            col = data[previous].copy()
            col.name = f"{yr}_q1"
            quarters.append(col)

        for q in range(2, 5):
            col = data[yr].copy()
            col.name = f"{yr}_q{q}"
            quarters.append(col)

    return pd.concat(quarters, axis=1)


def _disaggregate_tfn_wor_wap(
    eddie: pd.DataFrame, tfn: pd.DataFrame, eddie_wor_wap: pd.DataFrame
) -> pd.DataFrame:
    """Disaggregate NPIER land use data using EDDIE proportions.

    Disaggregates to either WOR or WAP.

    Parameters
    ----------
    eddie : pd.DataFrame
        EDDIE population or employment data, should be the
        same land use type as `tfn`.
    tfn : pd.DataFrame
        TfN (NPIER) population or employment data, should be the
        same land use type as `eddie`.
    eddie_wor_wap : pd.DataFrame
        EDDIE WAP or WOR data, WAP corresponds to population data
        and WOR to employment.

    Returns
    -------
    pd.DataFrame
        NPIER land use disaggregated to WOR / WAP.
    """
    eddie_wor_wap = eddie_wor_wap / eddie.groupby(level="region").sum()
    # Drop NaN columns, which will be average years which aren't
    # given in wor_regions, left with year quarters
    eddie_wor_wap = eddie_wor_wap.dropna(how="all", axis=1)

    tfn_pop_regions = tfn.groupby(level="region").sum()
    tfn_pop_regions = _calculate_yearly_quarters(tfn_pop_regions)

    tfn_wor_wap = pd.DataFrame(index=eddie_wor_wap.index)
    for c in eddie_wor_wap.columns:
        if c in tfn_pop_regions.columns:
            tfn_wor_wap.loc[:, c] = eddie_wor_wap[c] * tfn_pop_regions[c]
        else:
            tfn_wor_wap.loc[:, c] = np.nan

    return tfn_wor_wap


def _disaggregate_tfn_employment(
    eddie_emp: pd.DataFrame,
    tfn_emp: pd.DataFrame,
    disagg_emp: pd.DataFrame,
    disagg_column: str,
) -> pd.DataFrame:
    """Disaggregate NPIER employment using EDDIE proportions.

    Parameters
    ----------
    eddie_emp : pd.DataFrame
        EDDIE employment data at LAD.
    tfn_emp : pd.DataFrame
        NPIER employment data and LAD.
    disagg_emp : pd.DataFrame
        EDDIE disaggregated employment data.
    disagg_column : str
        Column in `disagg_emp` which contains the disaggregation
        names.

    Returns
    -------
    pd.DataFrame
        NPIER employement disaggregated based on `disagg_emp`.
    """
    geo_cols = ["region", "local_authority"]
    columns = disagg_emp.columns.tolist()
    disagg_emp = disagg_emp.reset_index().merge(
        eddie_emp, how="left", on=geo_cols, suffixes=("_disagg", "_tot")
    )
    disagg_emp = disagg_emp.set_index(geo_cols + [disagg_column])

    # Calculate proportion of employment for each disaggregation
    for c in columns:
        disagg_emp.loc[:, c] = disagg_emp[f"{c}_disagg"] / disagg_emp[f"{c}_tot"]
    disagg_emp = disagg_emp.loc[:, columns]

    disagg_emp = disagg_emp.reset_index().merge(
        tfn_emp, how="left", on=geo_cols, suffixes=("_prop", "_tfn")
    )
    disagg_emp = disagg_emp.set_index(geo_cols + [disagg_column])

    # Apply proportion of employment to TfN totals
    for c in columns:
        if f"{c}_tfn" in disagg_emp.columns:
            disagg_emp.loc[:, c] = disagg_emp[f"{c}_prop"] * disagg_emp[f"{c}_tfn"]
        else:
            disagg_emp.loc[:, c] = np.nan

    return disagg_emp.loc[:, columns]


def disaggregate_tfn_landuse(
    eddie_data: dict[LandUseType, pd.DataFrame],
    tfn_data: dict[LandUseType, pd.DataFrame],
    eddie_disaggregated: DisaggregatedLandUse,
) -> DisaggregatedLandUse:
    """Disaggregate the NPIER land use using EDDIE proportions.

    Parameters
    ----------
    eddie_data : dict[LandUseType, pd.DataFrame]
        Population and employment dand use data for EDDIE at LAD.
    tfn_data : dict[LandUseType, pd.DataFrame]
        Population and employment dand use data for NPIER at LAD.
    eddie_disaggregated : DisaggregatedLandUse
        Disaggregated EDDIE land use.

    Returns
    -------
    DisaggregatedLandUse
        Disaggregated NPIER land use.
    """
    tfn_wor = _disaggregate_tfn_wor_wap(
        eddie_data[LandUseType.POPULATION],
        tfn_data[LandUseType.POPULATION],
        eddie_disaggregated.wor.droplevel(level="units", axis=0),
    )
    tfn_wap = _disaggregate_tfn_wor_wap(
        eddie_data[LandUseType.EMPLOYMENT],
        tfn_data[LandUseType.EMPLOYMENT],
        eddie_disaggregated.wap.droplevel(level="units", axis=0),
    )
    tfn_emp_industry = _disaggregate_tfn_employment(
        eddie_data[LandUseType.EMPLOYMENT],
        tfn_data[LandUseType.EMPLOYMENT],
        eddie_disaggregated.employment_industry,
        "cebr_industry",
    )

    return DisaggregatedLandUse(
        wap=tfn_wap,
        wor=tfn_wor,
        employment_industry=tfn_emp_industry,
    )


def comparison_heatmaps(
    comparisons: dict[LandUseType, pd.DataFrame],
    geodata: gpd.GeoDataFrame,
    geom_id_column: str,
    years: list[int],
    output_file_base: pathlib.Path,
) -> None:
    """Plot heatmaps of the percentrage difference between EDDIE and NPIER.

    Parameters
    ----------
    comparisons : dict[LandUseType, pd.DataFrame]
        DataFrames containing the EDDIE and TfN land use data.
    geodata : gpd.GeoDataFrame
        Geometries for the LADs to create the heatmap.
    geom_id_column : str
        Name of the column in `geodata` containing the geometry
        ID to link to `comparisons`.
    years : list[int]
        List of years to produce comparisons for.
    output_file_base : pathlib.Path
        Base file name for saving the plots to, additional
        metadata about the plot will be appended to the filename.
    """
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
    comparisons: dict[LandUseType, pd.DataFrame],
    disaggregated_tfn: DisaggregatedLandUse,
    output_file: pathlib.Path,
) -> None:
    """Write NPIER data to workbook in EDDIE format.


    Parameters
    ----------
    comparisons : dict[LandUseType, pd.DataFrame]
        DataFrames containing TfN and EDDIE land use.
    disaggregated_tfn : DisaggregatedLandUse
        DataFrames containing the disaggregated NPIER land use.
    output_file : pathlib.Path
        Excel workbook file to write to.
    """
    LOG.info("Writing TfN land use to EDDIE format")
    with pd.ExcelWriter(output_file) as excel:
        for pop_emp, data in comparisons.items():
            eddie_years = data.loc[:, "EDDIE"].columns.tolist()
            data = data.loc[
                data.index.get_level_values(f"{EDDIE_ZONE_SYSTEM}_zone_id") != "N/A", "TfN"
            ]

            # Use original columns for EDDIE even if they are empty
            quarters = []
            for yr in eddie_years:
                for q in range(1, 5):
                    if yr in data.columns:
                        series = data[yr].copy()
                    else:
                        series = pd.Series(index=data.index)

                    series.name = f"{yr} Q{q}"
                    quarters.append(series)

            data = pd.concat(quarters, axis=1)
            data.to_excel(excel, sheet_name=pop_emp.value.title())

        for nm, data in disaggregated_tfn._asdict().items():
            data.to_excel(excel, sheet_name=nm)

    LOG.info("Written: %s", output_file)


def convert_to_eddie_years(landuse: NPIERLandUseType) -> NPIERLandUseType:
    """Convert some data to year quarters to match EDDIE format.

    Population, employment, WAP and WOR are converted to year quarters
    (the values are identical to the full year but the column is
    duplicated for the 4 quarters). Employment by industry and
    employment by occupation are left as full years.

    Parameters
    ----------
    landuse : NPIERLandUseType
        Land use data to be converted.

    Returns
    -------
    NPIERLandUseType
        Converted land use data.
    """
    quarterly_frames = ("population", "employment")
    quarterly_series = ("wap", "wor")

    reformatted = {}
    for nm, df in landuse._asdict().items():
        if nm in quarterly_frames:
            df = _calculate_yearly_quarters(df)
        elif nm in quarterly_series:
            df = _calculate_yearly_quarters(df.to_frame().T)
            df = df.T.iloc[:, 0]
        reformatted[nm] = df

    return type(landuse)(**reformatted)


def _get_index_names(df: pd.DataFrame) -> list[str]:
    names = df.index.names
    if len(names) == 1 and names[0] is None:
        return ["index"]
    return list(names)


def merge_with_eddie(
    eddie: pd.DataFrame,
    north: pd.DataFrame,
    eddie_cols: list[str],
    north_cols: list[str],
    **kwargs,
) -> pd.DataFrame:
    lad_rename = {"kingston upon hull, city of": "kingston upon hull"}

    eddie = eddie.copy()
    eddie.columns = [f"{c}_eddie" for c in eddie.columns]
    north = north.copy()
    north.columns = [f"{c}_north" for c in north.columns]

    index = _get_index_names(eddie) + _get_index_names(north)
    eddie = eddie.reset_index()
    eddie_merge_columns = []
    for col in eddie_cols:
        eddie_merge_col = col + "_merge_col"
        eddie.loc[:, eddie_merge_col] = eddie[col].str.lower().replace(lad_rename)
        eddie_merge_columns.append(eddie_merge_col)
    north = north.reset_index()

    merged = pandas_utils.fuzzy_merge(eddie, north, eddie_merge_columns, north_cols, **kwargs)
    if "_merge" in merged.columns:
        merged.loc[:, "_merge"] = merged["_merge"].replace(
            {"left_only": "eddie_only", "right_only": "npier_only"}
        )

    merged.drop(columns=eddie_merge_col, inplace=True)
    return merged.set_index(index)


def _infill_columns(
    data: pd.DataFrame,
    columns: list[str],
    eddie_suffix: str = "_eddie",
    npier_suffix: str = "_north",
) -> pd.DataFrame:
    eddie_mask = data["_merge"] == "eddie_only"
    new_columns = [data["_merge"]]

    for col_nm in columns:
        eddie_col = col_nm + eddie_suffix
        npier_col = col_nm + npier_suffix

        if npier_col in data.columns and eddie_col in data.columns:
            col = data[npier_col].copy()
            col.loc[eddie_mask] = data.loc[eddie_mask, eddie_col]

        elif npier_col in data.columns:
            col = data[npier_col].copy()

        else:
            col = pd.Series(np.nan, index=data.index)

        col.name = col_nm
        new_columns.append(col)

    return pd.concat(new_columns, axis=1)


def output_eddie_format(
    land_use_dataframes: NPIEREDDIEFormatLandUse, output_file: pathlib.Path
) -> None:
    with pd.ExcelWriter(output_file) as excel:
        for nm, df in land_use_dataframes._asdict().items():
            if df is None:
                continue
            # Output spreadsheet is in 000s
            df = df.copy()
            numeric_columns = df.select_dtypes((int, float)).columns
            if nm == "wor":
                df.loc[:, numeric_columns] = df[numeric_columns] / 1000
            df.to_excel(excel, sheet_name=nm)

    LOG.info("Written %s", output_file)


def split_column_names(
    name: str, eddie_suffix: str = "eddie", npier_suffix: str = "north"
) -> tuple[str, str]:
    pattern = rf"^(.*?)(?:_({eddie_suffix}|{npier_suffix}))?$"
    match = re.match(pattern, name)
    if match is None:
        raise ValueError(f"something's wrong with '{name}'")

    if match.group(2) is None:
        return ("", match.group(1))

    return (match.group(2), match.group(1))


def write_north_only_eddie_format(
    eddie: EDDIELandUseData,
    disaggregated_eddie: DisaggregatedLandUse,
    npier_north: NPIERNorthernLandUse,
    output_file: pathlib.Path,
) -> NPIEREDDIEFormatLandUse:
    LOG.info("Creating NPIER North only EDDIE format")
    # Add in Northern values from NPIER raw but don't change the external areas
    kwargs = dict(
        how="left",
        validate="1:1",
        indicator=True,
    )

    merged_population = merge_with_eddie(
        eddie.data[LandUseType.POPULATION],
        npier_north.population,
        eddie_cols=["local_authority"],
        north_cols=["npier_lad"],
        **kwargs,
    )
    columns = [f"{i}_q{j}" for i in OUTPUT_YEARS for j in "1234"]
    merged_population = _infill_columns(merged_population, columns)

    merged_employment = merge_with_eddie(
        eddie.data[LandUseType.EMPLOYMENT],
        npier_north.employment,
        eddie_cols=["local_authority"],
        north_cols=["npier_lad"],
        **kwargs,
    )
    merged_employment = _infill_columns(merged_employment, columns)

    merged_wor_wap: dict[str, pd.DataFrame] = {}
    for w in ("wor", "wap"):
        eddie_wor_wap = getattr(disaggregated_eddie, w)
        eddie_factors = (
            eddie_wor_wap.loc[list(NORTH_REGIONS)]
            / eddie_wor_wap.loc[list(NORTH_REGIONS)].sum()
        )
        # Add additional columns using split from final column
        eddie_factors = eddie_factors.reindex(columns=columns, method="ffill")

        npier_wor_wap = eddie_factors.multiply(getattr(npier_north, w), axis=1).loc[:, columns]
        merged_wor_wap[w] = eddie_wor_wap.reindex(columns=columns)
        merged_wor_wap[w].loc[list(NORTH_REGIONS)] = npier_wor_wap

    # Industry doesn't have the same LADs as pop / emp in EDDIE for some reason
    north_industry = npier_north.industry_employment.copy()
    north_industry_index = ["npier_lad", "cebr_industry_npier"]
    north_industry.index.names = north_industry_index
    # Update NPIER industries to EDDIE ones as they're different
    north_industry.index = pd.MultiIndex.from_arrays(
        [
            north_industry.index.get_level_values("npier_lad"),
            north_industry.index.get_level_values("cebr_industry_npier")
            .to_series()
            .replace(CEBR_INDUSTRY_LOOKUP),
        ]
    )
    north_industry = north_industry.groupby(level=[0, 1]).sum()

    merged_industry = merge_with_eddie(
        disaggregated_eddie.employment_industry,
        north_industry,
        eddie_cols=["local_authority", "cebr_industry"],
        north_cols=north_industry_index,
        **kwargs,
    )
    merged_industry = _infill_columns(merged_industry, [str(i) for i in OUTPUT_YEARS])

    landuse_data = NPIEREDDIEFormatLandUse(
        population=merged_population,
        employment=merged_employment,
        wap=merged_wor_wap["wap"],
        wor=merged_wor_wap["wor"],
        industry_employment=merged_industry,
    )
    output_eddie_format(landuse_data, output_file)
    return landuse_data


def _infill_external_regions_single(
    data: pd.DataFrame, index_column: str | list[str], method: InfillMethod
) -> pd.DataFrame:
    external_regions = ~data.index.get_level_values("region").isin(NORTH_REGIONS)

    if method == InfillMethod.ANY:
        infill_columns = data.loc[external_regions].isna().any(axis=0)
    elif method == InfillMethod.ALL:
        infill_columns = data.loc[external_regions].isna().all(axis=0)
    else:
        raise ValueError(f"invalid infill method {method}")

    infill_columns = infill_columns.index[infill_columns]

    if len(infill_columns) == 0:
        return data

    original_index = data.index.names
    data = data.reset_index().set_index(index_column)

    data.loc[external_regions, infill_columns] = data.loc[
        external_regions, infill_columns
    ].fillna(axis=1, method="ffill")
    return data.reset_index().set_index(original_index)


def _split_quarterly_columns(
    columns: list[str], min_year: int, max_year: int
) -> tuple[list[str], list[str]]:
    quarterly_columns = []
    yearly_columns = []
    for yr in range(min_year, max_year + 1):
        yr = str(yr)
        if yr in columns:
            yearly_columns.append(yr)

        for q in "1234":
            quarter = f"{yr}_q{q}"
            if quarter in columns:
                quarterly_columns.append(quarter)

    return quarterly_columns, yearly_columns


def _infill_external_regions(
    eddie_format_npier_data: NPIEREDDIEFormatLandUse, method: InfillMethod
) -> NPIEREDDIEFormatLandUse:
    LAD_INDEX = "lad_2017_zone_id"
    REGION_INDEX = "region"

    infilled_population = _infill_external_regions_single(
        eddie_format_npier_data.population.drop(columns="_merge", errors="ignore"),
        LAD_INDEX,
        method,
    )
    infilled_employment = _infill_external_regions_single(
        eddie_format_npier_data.employment.drop(columns="_merge", errors="ignore"),
        LAD_INDEX,
        method,
    )
    infilled_wap = _infill_external_regions_single(
        eddie_format_npier_data.wap, REGION_INDEX, method
    )
    infilled_wor = _infill_external_regions_single(
        eddie_format_npier_data.wor, REGION_INDEX, method
    )
    infilled_industry = _infill_external_regions_single(
        eddie_format_npier_data.industry_employment.drop(columns="_merge", errors="ignore"),
        ["local_authority", "cebr_industry"],
        method,
    )

    return NPIEREDDIEFormatLandUse(
        population=infilled_population,
        employment=infilled_employment,
        wap=infilled_wap,
        wor=infilled_wor,
        industry_employment=infilled_industry,
    )


def _add_year_columns(
    data: pd.DataFrame,
    min_year: int,
    max_year: int,
    yearly: bool = True,
    quarterly: bool = True,
) -> pd.DataFrame:
    columns = []
    if yearly:
        columns += [str(i) for i in range(min_year, max_year + 1)]
    if quarterly:
        columns += [f"{i}_q{j}" for i in range(min_year, max_year + 1) for j in "1234"]
    if columns == []:
        raise ValueError("no columns to add")

    return data.reindex(columns=columns)


def _factor_eddie_regions(
    base: pd.DataFrame, target: pd.DataFrame, columns: list[str], external: bool = True
) -> pd.DataFrame:
    if external:
        region_filter = lambda r: r not in NORTH_REGIONS
    else:
        region_filter = lambda r: r in NORTH_REGIONS

    regions = [r for r in base.index.get_level_values("region").unique() if region_filter(r)]

    base = base.loc[:, columns].copy()
    base.loc[regions] = (
        base.loc[regions]
        .div(base.groupby("region").sum())
        .mul(target.loc[:, columns], level="region", axis=1)
    )
    return base


def write_factored_external_eddie_format(
    npier_regions: NPIERRegionsLandUse,
    eddie_format_npier_north: NPIEREDDIEFormatLandUse,
    output_file: pathlib.Path,
    infill: InfillMethod = InfillMethod.NONE,
) -> NPIEREDDIEFormatLandUse:
    # Add in Northern values from NPIER raw and factor external areas to match NPIER external regions
    if infill in (InfillMethod.ANY, InfillMethod.ALL):
        infilled_npier_north = _infill_external_regions(eddie_format_npier_north, infill)
    else:
        infilled_npier_north = eddie_format_npier_north.copy()

    # Normalise region index column for all dataframes
    npier_dict: dict[str, pd.DataFrame] = infilled_npier_north._asdict()
    for df in npier_dict.values():
        index_level = df.index.names.index("region")
        df.index.set_levels(
            _normalise_region_names(df.index.levels[index_level]), level="region", inplace=True
        )
    infilled_npier_north = NPIEREDDIEFormatLandUse(**npier_dict)

    factored: dict[str, pd.DataFrame] = {}

    factored["population"] = _factor_eddie_regions(
        infilled_npier_north.population, npier_regions.population, QUARTERLY_COLUMNS
    )
    factored["employment"] = _factor_eddie_regions(
        infilled_npier_north.employment, npier_regions.employment, QUARTERLY_COLUMNS
    )
    factored["wap"] = _factor_eddie_regions(
        infilled_npier_north.wap, npier_regions.population, QUARTERLY_COLUMNS
    )
    factored["wor"] = _factor_eddie_regions(
        infilled_npier_north.wor, npier_regions.employment, QUARTERLY_COLUMNS
    )

    regions_yearly_employment = npier_regions.employment.rename(
        columns={f"{y}_q2": y for y in YEARLY_COLUMNS}
    ).loc[:, YEARLY_COLUMNS]
    factored["industry_employment"] = _factor_eddie_regions(
        infilled_npier_north.industry_employment, regions_yearly_employment, YEARLY_COLUMNS
    )

    landuse_data = NPIEREDDIEFormatLandUse(**factored)
    output_eddie_format(landuse_data, output_file)
    return landuse_data


def write_factored_north_eddie_format(
    eddie: EDDIELandUseData,
    disaggregated_eddie: DisaggregatedLandUse,
    eddie_format_npier_north: NPIEREDDIEFormatLandUse,
    output_file: pathlib.Path,
    infill: InfillMethod = InfillMethod.ALL,
) -> NPIEREDDIEFormatLandUse:
    # Add in Northern values from NPIER factored to EDDIE region totals and leave external area alone
    if infill in (InfillMethod.ALL, InfillMethod.ANY):
        infilled_npier_north = _infill_external_regions(eddie_format_npier_north, infill)
        infilled_eddie = _infill_external_regions(
            NPIEREDDIEFormatLandUse(
                population=_add_year_columns(
                    eddie.data[LandUseType.POPULATION], MIN_YEAR, MAX_YEAR, yearly=False
                ),
                employment=_add_year_columns(
                    eddie.data[LandUseType.EMPLOYMENT], MIN_YEAR, MAX_YEAR, yearly=False
                ),
                wap=_add_year_columns(
                    disaggregated_eddie.wap, MIN_YEAR, MAX_YEAR, yearly=False
                ),
                wor=_add_year_columns(
                    disaggregated_eddie.wor, MIN_YEAR, MAX_YEAR, yearly=False
                ),
                industry_employment=_add_year_columns(
                    disaggregated_eddie.employment_industry,
                    MIN_YEAR,
                    MAX_YEAR,
                    quarterly=False,
                ),
            ),
            infill,
        )
    else:
        infilled_npier_north = eddie_format_npier_north.copy()
        infilled_eddie = NPIEREDDIEFormatLandUse(
            population=eddie.data[LandUseType.POPULATION],
            employment=eddie.data[LandUseType.EMPLOYMENT],
            wap=disaggregated_eddie.wap,
            wor=disaggregated_eddie.wor,
            industry_employment=disaggregated_eddie.employment_industry,
        )

    factored: dict[str, pd.DataFrame] = {}

    factored["population"] = _factor_eddie_regions(
        infilled_npier_north.population,
        infilled_eddie.population.groupby("region").sum(),
        QUARTERLY_COLUMNS,
        external=False,
    )
    factored["employment"] = _factor_eddie_regions(
        infilled_npier_north.employment,
        infilled_eddie.employment.loc[:, QUARTERLY_COLUMNS].groupby("region").sum(),
        QUARTERLY_COLUMNS,
        external=False,
    )

    factored["industry_employment"] = _factor_eddie_regions(
        infilled_npier_north.industry_employment,
        infilled_eddie.industry_employment.groupby("region").sum(),
        YEARLY_COLUMNS,
        external=False,
    )

    landuse_data = NPIEREDDIEFormatLandUse(**factored)
    output_eddie_format(landuse_data, output_file)
    return landuse_data


def _npier_eddie_comparison(
    eddie: pd.DataFrame, npier: pd.DataFrame, join_columns: list[str]
) -> pd.DataFrame:
    for col in join_columns:
        if col not in eddie.index.names:
            raise KeyError(f"column '{col}' missing from EDDIE index")
        if col not in npier.index.names:
            raise KeyError(f"column '{col}' missing from NPIER index")

    eddie = eddie.copy()
    npier = npier.copy()

    indices = [c for c in eddie.index.names if c not in join_columns]
    if indices:
        eddie.index = eddie.index.droplevel(indices)

    columns = [c for c in npier.columns if c in eddie.columns]
    eddie = eddie.loc[:, columns]

    npier.columns = pd.MultiIndex.from_product((("NPIER",), npier.columns))
    eddie.columns = pd.MultiIndex.from_product((("EDDIE",), eddie.columns))

    merged = npier.merge(
        eddie, how="outer", left_index=True, right_index=True, validate="1:1", indicator=True
    )
    missing = merged["_merge"] != "both"
    if missing.any():
        raise ValueError(f"merging NPIER to EDDIE found {missing.sum()} missing LAs")
    merged = merged.drop(columns="_merge")

    diff = merged.loc[:, "NPIER"].subtract(merged.loc[:, "EDDIE"])
    diff = diff.dropna(axis=1, how="all")
    diff.columns = pd.MultiIndex.from_product((("NPIER - EDDIE",), diff.columns))

    perc_diff = merged.loc[:, "NPIER"].divide(merged.loc[:, "EDDIE"]) - 1
    perc_diff = perc_diff.dropna(axis=1, how="all")
    perc_diff.columns = pd.MultiIndex.from_product((("NPIER / EDDIE (%)",), perc_diff.columns))

    return pd.concat([merged, diff, perc_diff], axis=1)


def _npier_eddie_heatmap(
    geospatial: gpd.GeoSeries,
    data: pd.DataFrame,
    join_column: str,
    plot_column: str,
    title: str,
    output_file: pathlib.Path,
    legend_label_fmt: str,
):
    geodata = data.merge(
        geospatial, left_on=join_column, right_index=True, how="right", validate="1:1"
    )
    geodata = gpd.GeoDataFrame(geodata, geometry=geospatial.geometry.name, crs=geospatial.crs)
    fig = plots._heatmap_figure(
        geodata.reset_index(drop=True),
        plot_column,
        title,
        n_bins=5,
        positive_negative_colormaps=True,
        legend_label_fmt=legend_label_fmt,
    )

    fig.savefig(output_file)
    LOG.info("Written: %s", output_file)


def _npier_eddie_comparisons_heatmaps_plot(
    comparisons: dict[str, pd.DataFrame],
    plot_years: list[int],
    output_folder: pathlib.Path,
    regions: Optional[gpd.GeoSeries] = None,
    lads: Optional[gpd.GeoSeries] = None,
) -> None:
    YEAR_KEYS = (
        "industry_employment",
        "industry_employment_regions",
    )

    def get_grouped_data(key: str) -> tuple[pd.DataFrame, str, list[str]]:
        if key in YEAR_KEYS:
            years = [str(y) for y in plot_years]
        else:
            # Use Q1 data for heatmaps
            years = [f"{y}_q1" for y in plot_years]

        data = comparisons[key].loc[:, (plot_type, years)]
        data = data.groupby(level=join_column).sum()
        data.columns = data.columns.droplevel(0)
        return data.reset_index(), key.replace("_", " ").title(), years

    for plot_type in ("NPIER - EDDIE", "NPIER / EDDIE (%)"):
        fname = plot_type.replace("/", "div")
        plot_folder = output_folder / fname
        plot_folder.mkdir(exist_ok=True)
        legend_fmt = "{:.1%}" if "%" in plot_type else "{:.0f}"

        if lads is not None:
            join_column = "lad_2017_zone_id"
            for data_key in (
                "population",
                "employment",
                "industry_employment",
            ):
                data, data_name, data_years = get_grouped_data(data_key)

                for yr in data_years:
                    _npier_eddie_heatmap(
                        lads,
                        data,
                        join_column,
                        str(yr),
                        f"{plot_type} {data_name} - {yr}",
                        plot_folder / (fname + f" {data_name} {yr}.png"),
                        legend_label_fmt=legend_fmt,
                    )

        if regions is not None:
            join_column = "region"
            # Use Q1 data for heatmaps
            data_years = [f"{y}_q1" for y in plot_years]

            for data_key in (
                "population_regions",
                "employment_regions",
                "industry_employment_regions",
                "wap",
                "wor",
            ):
                try:
                    data, data_name, data_years = get_grouped_data(data_key)
                except KeyError:
                    continue

                for yr in data_years:
                    _npier_eddie_heatmap(
                        regions,
                        data,
                        join_column,
                        str(yr),
                        f"{plot_type} {data_name} - {yr}",
                        plot_folder / (fname + f" {data_name} {yr}.png"),
                        legend_label_fmt=legend_fmt,
                    )


def _simplify_strings(data: pd.Series) -> pd.Series:
    return data.str.lower().str.replace("\s+", "", regex=True)


def npier_eddie_comparison_heatmaps(
    eddie: EDDIELandUseData,
    disaggregated_eddie: DisaggregatedLandUse,
    npier: NPIEREDDIEFormatLandUse,
    output_folder: pathlib.Path,
    plot_years: list[int],
    regions: Optional[gpd.GeoSeries] = None,
    lads: Optional[gpd.GeoSeries] = None,
):
    LOG.info("Creating %s", output_folder.stem)
    output_folder.mkdir(exist_ok=True)

    comparisons: dict[str, pd.DataFrame] = {}

    for nm in ("population", "employment"):
        eddie_data = eddie.data[LandUseType(nm)]
        npier_data: pd.DataFrame = getattr(npier, nm)
        comparisons[nm] = _npier_eddie_comparison(eddie_data, npier_data, ["local_authority"])
        comparisons[f"{nm}_regions"] = _npier_eddie_comparison(
            eddie_data.groupby("region").sum(), npier_data.groupby("region").sum(), ["region"]
        )

    for nm in ("industry",):
        eddie_data: pd.DataFrame = getattr(disaggregated_eddie, f"employment_{nm}")
        npier_data: pd.DataFrame = getattr(npier, f"{nm}_employment")
        comparisons[f"{nm}_employment"] = _npier_eddie_comparison(
            eddie_data, npier_data, ["local_authority", f"cebr_{nm}"]
        )
        comparisons[f"{nm}_employment_regions_split"] = _npier_eddie_comparison(
            eddie_data.groupby(["region", f"cebr_{nm}"]).sum(),
            npier_data.groupby(["region", f"cebr_{nm}"]).sum(),
            ["region", f"cebr_{nm}"],
        )
        comparisons[f"{nm}_employment_regions"] = _npier_eddie_comparison(
            eddie_data.groupby("region").sum(),
            npier_data.groupby("region").sum(),
            ["region"]
        )

    if npier.wap is not None:
        comparisons["wap"] = _npier_eddie_comparison(
            disaggregated_eddie.wap, npier.wap, ["region"]
        )
    if npier.wor is not None:
        comparisons["wor"] = _npier_eddie_comparison(
            disaggregated_eddie.wor, npier.wor, ["region"]
        )

    # Add LAD IDs to disaggregated employment
    lad_id_column = "lad_2017_zone_id"
    lad_id_lookup = dict(
        zip(
            _simplify_strings(
                comparisons["employment"].index.get_level_values("local_authority")
            ),
            comparisons["employment"].index.get_level_values(lad_id_column),
        )
    )
    for nm in ("industry_employment",):
        lad_ids = _simplify_strings(
            comparisons[nm].index.get_level_values("local_authority").to_series()
        ).replace(lad_id_lookup)
        comparisons[nm][lad_id_column] = lad_ids.values
        comparisons[nm].set_index(lad_id_column, append=True, inplace=True)

    LOG.info("Writing comparison spreadsheets & heatmaps")
    for nm, df in comparisons.items():
        excel_file = output_folder / f"NPIER_EDDIE_inputs_comparison-{nm}.xlsx"
        with pd.ExcelWriter(excel_file, engine="openpyxl") as excel:
            for sheet in df.columns.get_level_values(0).unique():
                df.loc[:, sheet].to_excel(excel, sheet_name=sheet.replace("/", "div"))

        LOG.info("Written: %s", excel_file)

    _npier_eddie_comparisons_heatmaps_plot(
        comparisons, plot_years, output_folder, regions, lads
    )


def main(params: EDDIEComparisonParameters, init_logger: bool = True):
    """Compare EDDIE land use inputs to NPIER.

    Parameters
    ----------
    params : EDDIEComparisonParameters
        Parameters for running comparison.
    init_logger : bool, default True
        Whether to initialise a logger.
    """
    output_folder = params.output_folder / params.npier_input.value
    if not output_folder.is_dir():
        output_folder.mkdir(parents=True)

    if init_logger:
        nd_log.get_logger(
            nd_log.get_package_logger_name(),
            output_folder / LOG_FILE,
            "Running EDDIE Inputs Comparison",
            log_version=True,
        )
        nd_log.capture_warnings(file_handler_args=dict(log_file=output_folder / LOG_FILE))

    LOG.debug("Input parameters:\n%s", params.to_yaml())
    params_out_file = output_folder / "EDDIE_comparison_parameters.yml"
    LOG.info("Written input parameters to %s", params_out_file)
    params.save_yaml(params_out_file)

    eddie = load_landuse_eddie(
        params.eddie_file,
        params.workbook_parameters.landuse,
        params.workbook_parameters.lad_lookup,
    )
    disaggregated_eddie = load_disaggregated_eddie(
        params.eddie_file, params.workbook_parameters
    )

    if params.npier_input == NPIERInputType.NPIER_SCENARIO_LANDUSE:
        # Used when using UZC data, probably no longer user
        tfn = load_landuse_tfn(params.npier_scenario_landuse)

        output_file_base = output_folder / "EDDIE_TfN_landuse_comparison"
        comparisons = compare_landuse(eddie, tfn, output_file_base)

        # Load other land use data and use it to split the TfN totals
        disaggregated = disaggregate_tfn_landuse(
            eddie.data,
            {k: v.loc[:, "TfN"] for k, v in comparisons.items()},
            disaggregated_eddie,
        )

        write_eddie_format(
            comparisons, disaggregated, output_folder / "TfN_landuse-EDDIE_format.xlsx"
        )

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

    else:
        # NPIER transformational raw data
        npier_raw_north, npier_raw_regions = load_raw_transformational(
            params.npier_raw_transformational
        )

        reformatted_north = convert_to_eddie_years(npier_raw_north)
        reformatted_regions = convert_to_eddie_years(npier_raw_regions)

        heatmap_kwargs = dict(plot_years=parameters.map_years)
        if parameters.regions_geospatial_file is None:
            LOG.warning("Regions geospatial file not given")
            heatmap_kwargs["regions"] = None
        else:
            heatmap_kwargs["regions"] = plots.get_geo_data(parameters.regions_geospatial_file)
            heatmap_kwargs["regions"].index = _normalise_region_names(
                heatmap_kwargs["regions"].index
            )

        if parameters.lad_geospatial_file is None:
            LOG.warning("LADs geospatial file not given")
            heatmap_kwargs["lads"] = None
        else:
            heatmap_kwargs["lads"] = plots.get_geo_data(parameters.lad_geospatial_file)

        npier_north_only_eddie = write_north_only_eddie_format(
            eddie,
            disaggregated_eddie,
            reformatted_north,
            output_folder / "NPIER_Raw_EDDIE_format-North_only.xlsx",
        )
        npier_eddie_comparison_heatmaps(
            eddie,
            disaggregated_eddie,
            npier_north_only_eddie,
            output_folder / "NPIER North Only Comparison",
            **heatmap_kwargs,
        )
        npier_factored_external = write_factored_external_eddie_format(
            reformatted_regions,
            npier_north_only_eddie.copy(),
            output_folder / "NPIER_Raw_EDDIE_format-North_with_factored_external.xlsx",
        )
        npier_eddie_comparison_heatmaps(
            eddie,
            disaggregated_eddie,
            npier_factored_external,
            output_folder / "NPIER North with Factored External Comparison",
            **heatmap_kwargs,
        )
        npier_factored_north = write_factored_north_eddie_format(
            eddie,
            disaggregated_eddie,
            npier_north_only_eddie,
            output_folder / "NPIER_Raw_EDDIE_format-factored_North.xlsx",
        )
        npier_eddie_comparison_heatmaps(
            eddie,
            disaggregated_eddie,
            npier_factored_north,
            output_folder / "NPIER Factored North Comparison",
            **heatmap_kwargs,
        )


if __name__ == "__main__":
    # TODO(MB) Add argument for config file path
    parameters = EDDIEComparisonParameters.load_yaml(CONFIG_FILE)

    main(parameters)
