import pathlib
from dataclasses import dataclass
from functools import reduce
from os import path
from typing import Optional

import pandas as pd
import pyodbc

# CONSTANTS
EMPCOLS = ["E" + str(item).zfill(2) for item in range(1, 16)]
POPCOLS = ["S" + str(item).zfill(3) for item in range(1, 89)]
KCOLS = ["K" + str(item).zfill(2) for item in range(1, 16)]
NTEMFILES = {
    "2011": "04",
    "2016": "13",
    "2021": "14",
    "2026": "15",
    "2031": "16",
    "2036": "17",
    "2041": "18",
    "2046": "19",
    "2051": "20",
}
KEEPCOLS = {"emp": EMPCOLS, "pop": POPCOLS}
DROPCOLS = {"emp": POPCOLS, "pop": EMPCOLS}
ACCESSDRIVER = "{Microsoft Access Driver (*.mdb, *.accdb)}"


@dataclass
class params:
    """
    All of these constants should work as of 25/04/2022.  Updates may be necessary with newer versions of NTEM
    base_year:This is the year which you have TfN landuse data for and want to scale up
    base_year_lower and base_year_higher are the years NTEM data is available for sandwiching your base year.  If NTEM is available for your base year, set these equal to None
    target_year_lower and target_year_higher are the same but for your target year
    NTEM years are contained in the keys of the dictionary 'ntem_files'
    You will also need to set data_source to your base folder in which all of the files you need are located (being the NTEM databases and your TfN landuse data)
    Here the NTEM databases are stored in a second folder called 'NTEM' and the landuse data is in the base folder
    The landuse growth lookup excel file is also in the base folder
    The lookups to go from the NTEM zone system to the TfN one for Scotland are csvs stored in SHP/emp and SHP/pop respectively
    """

    years = {
        "base year": 2018,
        "base year lower": 2016,
        "base year higher": 2021,
        "target year": 2019,
        "target year lower": 2016,
        "target year higher": 2021,
    }
    data_source = pathlib.Path(r"C:\Projects\MidMITS\NTEM")
    lookup_dir = pathlib.Path(r"SHP/NTEM_land_use_growth_lookup.xlsx")
    tfn_base_pop_dir = pathlib.Path(path.join("SHP", f"land_use_{years['base year']}_pop.csv"))
    tfn_base_emp_dir = pathlib.Path(path.join("SHP", f"land_use_{years['base year']}_emp.csv"))
    NTEM_output_dir = pathlib.Path(r"Temp storage")
    NTEM_input_dir = pathlib.Path(r"NTEM")
    emp_corr_dir = pathlib.Path(r"SHP/emp/ntem_to_int_zone_correspondence.csv")
    pop_corr_dir = pathlib.Path(r"SHP/pop/ntem_to_int_zone_correspondence.csv")
    corr_dict = {"emp": emp_corr_dir, "pop": pop_corr_dir}


def read_tfn() -> dict:
    """
    Read in TfN landuse data for your base year for population and employment

    Returns:
        _dict_: A dictionary with keys "emp" and "pop" accessing necessary data on 
        employment and population landuse
    """
    p = params
    lookup_pop = pd.read_excel(
        path.join(p.data_source, p.lookup_dir),
        sheet_name="TfN Traveller Types",
        skiprows=11,
        usecols=["tfn_traveller_type", "ntem_traveller_type"],
    ).set_index("tfn_traveller_type")
    lookup_emp = pd.read_excel(
        path.join(p.data_source, p.lookup_dir),
        sheet_name="TfN Employment Types",
        skiprows=11,
        usecols=["employment_cat", "NTEM_cat"],
    ).set_index("employment_cat")
    emp_base = (
        pd.read_csv(path.join(p.data_source, p.tfn_base_emp_dir))
        .set_index(["msoa_zone_id", "employment_cat"])
        .join(lookup_emp, how="inner")
        .reset_index()
        .set_index(["msoa_zone_id", "employment_cat", "NTEM_cat"])
    )
    pop_base = (
        pd.read_csv(path.join(p.data_source, p.tfn_base_pop_dir))
        .set_index(["msoa_zone_id", "tfn_traveller_type"])
        .join(lookup_pop, how="inner")
        .reset_index()
        .set_index(["msoa_zone_id", "tfn_traveller_type", "ntem_traveller_type"])
    )
    emp_base.index.rename(["msoa_zone_id", "employment_cat", "emp code"])
    pop_base.index.rename(["msoa_zone_id", "tfn_traveller_type", "pop code"])
    emp_base.columns = ["soc", f"{p.years['base year']}"]
    pop_base.columns = ["area type", f"{p.years['base year']}"]
    dict = {
        "emp": [emp_base, ["soc", f"{p.years['base year']}"]],
        "pop": [pop_base, ["area type", f"{p.years['base year']}"]],
    }
    return dict


def read_NTEM(year: int) -> pd.DataFrame:
    """_summary_
    Reads NTEM data from an access database and outputs it as a DataFrame
    Args:
        year (int): The year you want NTEM data for

    Returns:
        pd.DataFrame: The NTEM data for that year with "ZoneID" as index, and employment/population categories as columns
    """
    p = params
    file_name = f"CTripEnd7_{str(year)}_run_{NTEMFILES[str(year)]}.accdb"
    conn = pyodbc.connect(
        f"Driver={ACCESSDRIVER};DBQ={path.join(p.data_source,p.NTEM_input_dir,file_name)};"
    )
    df = (
        pd.read_sql("SELECT * FROM ZoneData", conn)
        .drop(KCOLS + ["I", "R", "B", "Borough", "ZoneName"], axis=1)
        .set_index("ZoneID")
    )
    return df


def int_year(
    target_year: int, year_1: Optional[int] = None, year_2: Optional[int] = None
) -> pd.DataFrame:
    """_summary_
    Returns NTEM dataframes for any year, a crude calculation is done if NTEM data is not available directly for that year
    Args:
        target_year (int): The year you want NTEM data for.
        year_1 (Optional[int], optional): Year closest to your target year below if target year has no NTEM data
        year_2 (Optional[int], optional): Year closest to your target year above if target year has no NTEM data

    Raises:
        ValueError: Raises error if your target year isn't in NTEM and you haven't provided year_1 and/or year_2

    Returns:
        pd.DataFrame: NTEM data for target year
    """
    p = params
    if str(target_year) in NTEMFILES.keys():
        return pd.read_csv(
            path.join(p.data_source, p.NTEM_output_dir, f"{target_year}.csv")
        ).set_index("ZoneID")
    else:
        if year_1 is None or year_2 is None:
            raise ValueError("Please provide two NTEM years sandwiching your target year")
        df_1 = pd.read_csv(
            path.join(p.data_source, p.NTEM_output_dir, f"{year_1}.csv")
        ).set_index("ZoneID")
        df_2 = pd.read_csv(
            path.join(p.data_source, p.NTEM_output_dir, f"{year_2}.csv")
        ).set_index("ZoneID")
        diff = df_2 - df_1
        target_df = df_1 + ((target_year - year_1) / (year_2 - year_1)) * diff
        return target_df


def rezone(corr: pd.DataFrame, df: pd.DataFrame, sector: str) -> pd.DataFrame:
    """_summary_
        Converts NTEM's Scotland zoning system to that used by TfN (intermediate datazones)
    Args:
        corr (pd.DataFrame): A loook-up between the two zone systems, generateed using TfN's zone translation tool
        df (pd.DataFrame): The NTEM data in the NTEM zone system
        sector (str): "emp" or "pop"

    Returns:
        pd.DataFrame: A dataframe of Scottish zones in the TfN zone system
    """
    scotland = df[df.index.str.startswith("S")]
    new = pd.merge(scotland, corr, left_index=True, right_index=True, validate="1:m").drop(
        DROPCOLS[sector], axis=1
    )
    for column in new.loc[:, KEEPCOLS[sector]]:
        new[column] = new[column] * new["ntem_to_int_zone"]
    final = new.groupby("int_zone_zone_id").sum().drop("ntem_to_int_zone", axis=1)
    return final


def func(
    sector: str, target_year: int, year_1: Optional[int] = None, year_2: Optional[int] = None
) -> pd.DataFrame:
    """_summary_
        Reads in an NTEM csv and converts the zone system to TfN's
    Args:
        sector (str): "pop" or "emp"
        target_year (int): The year you want NTEM data for
        year_1 (Optional[int], optional): See int_year. Defaults to None.
        year_2 (Optional[int], optional): See int_year. Defaults to None.

    Returns:
        pd.DataFrame: NTEM dataframe in TfN zone system
    """
    p = params
    corr = (
        pd.read_csv(path.join(p.data_source, p.corr_dict[sector]))
        .set_index("ntem_zone_id")
        .drop("int_zone_to_ntem", axis=1)
    )
    df = int_year(target_year, year_1, year_2)
    scot_df = rezone(corr, df, sector)
    engwal_df = df[df.index.str.startswith("S") == False].drop(DROPCOLS[sector], axis=1)
    full_df = pd.concat([scot_df, engwal_df], axis=0).stack().to_frame()
    full_df.columns = [str(target_year)]
    return full_df


def apply_growth(
    df: pd.DataFrame,
    base=params.years["base year"],
    target=params.years["target year"],
    low=params.years["base year lower"],
    high=params.years["target year higher"],
) -> pd.DataFrame:
    """_summary_
        Reads a dataframe with data for base and target years and produces a 'factor' column used to apply growth to TfN landuse.
        Checks for zeros in years used to calculate factors to remove arbitrary factors (set to the average factor for that employment
        or population type, or to 1 in the case of zeros in both base and target year)
    Args:
        df (pd.DataFrame): The dataframe with columns for base and target year, and 'pct' which is the mean growth percentage
        for a given employment or population category
        base (_type_, optional): _description_. Defaults to params.years["base year"].
        target (_type_, optional): _description_. Defaults to params.years["target year"].
        low (_type_, optional): _description_. Defaults to params.years["base year lower"].
        high (_type_, optional): _description_. Defaults to params.years["target year higher"].

    Returns:
        pd.DataFrame: The same dataframe as read in with an extra 'factor' column
    """
    df["factor"] = df["pct"]
    if high is None and low is None:
        df.loc[(df[str(base)] != 0) & (df[str(target)] != 0), "factor"] = df["growth"]
    elif high is None:
        df.loc[(df[str(low)] != 0) & (df[str(target)] != 0), "factor"] = df["growth"]
    else:
        df.loc[(df[str(low)] != 0) & (df[str(high)] != 0), "factor"] = df["growth"]
    df.loc[(df[str(base)] == 0) & (df[str(target)] == 0), "factor"] = 1
    return df


def process_df(sector: str) -> pd.DataFrame:
    """_summary_
        Function which calls other functions, and exports the TfN landuse data scaled up from base year to target year
    Args:
        sector (str): 'pop' or 'emp'
    """
    p = params

    df_base = func(
        sector, p.years["base year"], p.years["base year lower"], p.years["base year higher"]
    )
    df_target = func(
        sector,
        p.years["target year"],
        p.years["target year lower"],
        p.years["target year higher"],
    )
    frames = [df_base, df_target]
    if p.years["base year lower"] is not None:
        df_lower = func(sector, p.years["base year lower"])
        frames.append(df_lower)
    if p.years["target year higher"] is not None:
        df_higher = func(sector, p.years["target year higher"])
        frames.append(df_higher)
    df = reduce(
        lambda left, right: pd.merge(
            left, right, left_index=True, right_index=True, how="inner"
        ),
        frames,
    )
    df.index.rename(["msoa_zone_id", f"{sector} code"], inplace=True)
    df_trav = df.groupby(f"{sector} code").sum()
    df_trav["pct"] = df_trav[str(p.years["target year"])] / df_trav[str(p.years["base year"])]
    df_final = df.join(df_trav["pct"], how="inner")
    df_final["growth"] = (
        df_final[str(p.years["target year"])] / df_final[str(p.years["base year"])]
    )
    growth_df = apply_growth(df_final)
    return growth_df


def final(sector: str):
    p = params
    growth_df = process_df(sector).drop(["pct", "growth"], axis=1)
    tfn_dict = read_tfn()
    tfn_data = tfn_dict[sector][0]
    tfn_data.index.rename(
        ["msoa_zone_id", "tfn_traveller_type", f"{sector} code"], inplace=True
    )
    int = tfn_data.join(growth_df["factor"], how="inner")
    int[f"{p.years['target year']}"] = int[f"{p.years['base year']}"] * int["factor"]
    export = int.drop([f"{p.years['base year']}", "factor"], axis=1).droplevel(
        f"{sector} code"
    )
    export.to_csv(path.join(p.data_source, "SHP", sector, f"landuse 2021_{sector}.csv"))
    int.to_csv(path.join(p.data_source, "SHP", sector, f"landuse 2021 complete_{sector}.csv"))


def apply_abs(sector: str) -> pd.DataFrame:
    """_summary_
    Applies growth to TfN landuse using absolute increases from NTEM instead of percentage
    Args:
        sector (str): 'pop' or 'emp'

    Returns:
        dataframe: returns a dataframe with more columns than necessary scaled to target year
    """
    p = params
    tfn_dict = read_tfn()
    tfn_data = tfn_dict[sector][0]
    tfn_data.index.rename(
        ["msoa_zone_id", "tfn_traveller_type", f"{sector} code"], inplace=True
    )
    NTEM = func(
        sector, p.years["base year"], p.years["base year lower"], p.years["base year higher"]
    ).join(
        func(
            sector,
            p.years["target year"],
            p.years["target year lower"],
            p.years["target year higher"],
        ),
        how="inner",
    )
    NTEM.index.rename(["msoa_zone_id", f"{sector} code"], inplace=True)
    NTEM["diff"] = NTEM[f"{p.years['target year']}"] - NTEM[f"{p.years['base year']}"]
    grouped = tfn_data.groupby(["msoa_zone_id", f"{sector} code"]).sum()[
        f"{p.years['base year']}"
    ]
    joined = tfn_data.join(grouped, how="inner", rsuffix="_grouped")
    joined["prop"] = (
        joined[f"{p.years['base year']}"] / joined[f"{p.years['base year']}_grouped"]
    )
    output = joined.join(NTEM["diff"], how="inner")
    output[f"{p.years['target year']}"] = (
        output[f"{p.years['base year']}"] + output["diff"] * output["prop"]
    )
    output.loc[output[f"{p.years['target year']}"] < 0, f"{p.years['target year']}"] = 0
    return output


def main():
    p = params
    for sector in ["emp", "pop"]:
        dic = read_tfn()
        cols = ["msoa_zone_id", "tfn_traveller_type", "2019"]
        cols.append(dic[sector][1][0])
        df = apply_abs(sector).reset_index()
        output = df[cols]
        output.set_index(
            ["msoa_zone_id", "tfn_traveller_type", dic[sector][1][0]], inplace=True
        )
        output.to_csv(
            path.join(
                p.data_source, "SHP", sector, f"landuse_{p.years['target year']}_{sector}.csv"
            )
        )


if __name__ == "__main__":
    main()
