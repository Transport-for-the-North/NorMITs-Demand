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
import pydantic
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
LOG = nd_log.get_logger(nd_log.get_package_logger_name() + ".eddie_comparison")
LOG_FILE = "EDDIE_comparison.log"
TFN_ZONE_SYSTEM = "msoa"
EDDIE_ZONE_SYSTEM = "lad_2017"
CONFIG_FILE = pathlib.Path(r"config\checker\EDDIE_comparison_parameters.yml")
EDDIE_EMPLOYMENT_HEADER_SKIP = 9
"""Number of header rows to skip in the occupation and industry sheets."""


##### CLASSES #####
class WorksheetParams(pydantic.BaseModel):
    """Parameters for an individual worksheet."""

    sheet_name: str
    header_row: int


class EDDIEWorkbookParams(pydantic.BaseModel):
    """Parameters for the worksheets in the EDDIE workbook."""

    landuse: WorksheetParams
    lad_lookup: WorksheetParams
    wor_wap: WorksheetParams
    employment_industry: WorksheetParams
    employment_occupation: WorksheetParams


class EDDIEComparisonParameters(config_base.BaseConfig):
    """Parameters for running EDDIE comparison script."""

    eddie_file: pathlib.Path
    workbook_parameters: EDDIEWorkbookParams
    tfn_base_folder: pathlib.Path
    scenario: nd.Scenario
    output_folder: pathlib.Path
    map_years: list[int]


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
    employment_occupation: pd.DataFrame


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


def load_landuse_tfn(folder: pathlib.Path, scenario: nd.Scenario) -> LandUseData:
    """Load NPIER landuse data.

    Parameters
    ----------
    folder : pathlib.Path
        Folder to find land use CSVs in, files are expected to be in population
        / employment sub-folders and named 'future_growth_values.csv'.
    scenario : nd.Scenario
        Scenario of land use data being used.

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
    wor_wap = wor_wap.set_index(index_columns)

    return wor_wap


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
        "occupation": workbook_params.employment_occupation,
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

        df = df.set_index(index_columns)
        for c in df.columns:
            df.loc[:, c] = pd.to_numeric(df[c], downcast="unsigned", errors="coerce")

        emp_data[name] = df

    return DisaggregatedLandUse(
        wap=wor_wap.loc["Working-Age Population"],
        wor=wor_wap.loc["Employment (Residence-based)"],
        employment_industry=emp_data["industry"],
        employment_occupation=emp_data["occupation"],
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
    drop = []
    for yr in data.columns:
        for q in range(1, 5):
            data.loc[:, f"{yr}_q{q}"] = data[yr]
        drop.append(yr)

    return data.drop(columns=drop)


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
    tfn_emp_occupation = _disaggregate_tfn_employment(
        eddie_data[LandUseType.EMPLOYMENT],
        tfn_data[LandUseType.EMPLOYMENT],
        eddie_disaggregated.employment_occupation,
        "cebr_occupation",
    )

    return DisaggregatedLandUse(
        wap=tfn_wap,
        wor=tfn_wor,
        employment_industry=tfn_emp_industry,
        employment_occupation=tfn_emp_occupation,
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


def main(params: EDDIEComparisonParameters, init_logger: bool = True):
    """Compare EDDIE land use inputs to NPIER.

    Parameters
    ----------
    params : EDDIEComparisonParameters
        Parameters for running comparison.
    init_logger : bool, default True
        Whether to initialise a logger.
    """
    if not params.output_folder.is_dir():
        params.output_folder.mkdir(parents=True)

    if init_logger:
        nd_log.get_logger(
            nd_log.get_package_logger_name(),
            params.output_folder / LOG_FILE,
            "Running EDDIE Inputs Comparison",
            log_version=True,
        )
        nd_log.capture_warnings(
            file_handler_args=dict(log_file=params.output_folder / LOG_FILE)
        )

    LOG.debug("Input parameters:\n%s", params.to_yaml())
    params_out_file = params.output_folder / "EDDIE_comparison_parameters.yml"
    LOG.info("Written input parameters to %s", params_out_file)
    params.save_yaml(params_out_file)

    eddie = load_landuse_eddie(
        params.eddie_file,
        params.workbook_parameters.landuse,
        params.workbook_parameters.lad_lookup,
    )
    tfn_folder = params.tfn_base_folder / params.scenario.value
    tfn = load_landuse_tfn(tfn_folder, params.scenario)

    output_file_base = params.output_folder / "EDDIE_TfN_landuse_comparison"
    comparisons = compare_landuse(eddie, tfn, output_file_base)

    # Load other land use data and use it to split the TfN totals
    disaggregated = load_disaggregated_eddie(params.eddie_file, params.workbook_parameters)
    disaggregated = disaggregate_tfn_landuse(
        eddie.data, {k: v.loc[:, "TfN"] for k, v in comparisons.items()}, disaggregated
    )

    write_eddie_format(
        comparisons, disaggregated, params.output_folder / "TfN_landuse-EDDIE_format.xlsx"
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


if __name__ == "__main__":
    # TODO(MB) Add argument for config file path
    parameters = EDDIEComparisonParameters.load_yaml(CONFIG_FILE)

    main(parameters)
