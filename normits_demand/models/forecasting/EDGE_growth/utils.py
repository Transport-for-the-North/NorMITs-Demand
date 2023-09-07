# -*- coding: utf-8 -*-
"""
Utils for edge growth, such as conversions between matrix formats.
"""
# Built-Ins
import itertools
import pathlib
from typing import Tuple

# Third Party
import numpy as np
import pandas as pd
from caf.toolkit import pandas_utils
# Local Imports
# pylint: disable=import-error,wrong-import-position
# Local imports here
from normits_demand.matrices.cube_mat_converter import CUBEMatConverter
# pylint: enable=import-error,wrong-import-position

# # # CONSTANTS # # #

# # # CLASSES # # #

# # # FUNCTIONS # # #
def transpose_matrix(
    mx_df: pd.DataFrame, stations: bool = False
) -> pd.DataFrame:
    """Transpose a matrix O<>D/P<>A.

    Parameters
    ----------
    mx : pd.DataFrame
        input matrix to transpose
    stations : bool
        whether it's a stations matrix or not

    Returns
    -------
    mx : pd.DataFrame
        transposed matrix
    """
    # o/d columns
    from_col = "from_model_zone_id"
    to_col = "to_model_zone_id"
    # if stations matrix, update the od columns
    if stations:
        from_col = "from_stn_zone_id"
        to_col = "to_stn_zone_id"
    # transpose to-home PA to OD by renaming from <> to model zone id
    mx_df = mx_df.rename(
        columns={
            from_col: to_col,
            to_col: from_col,
        }
    )

    return mx_df


def expand_matrix(
    mx_df: pd.DataFrame, zones: int = 1300, stations: bool = False
) -> pd.DataFrame:
    """Expand matrix to all possible movements (zones x zones).

    Parameters
    ----------
    mx_df : pd.DataFrame
        matrix
    zones: int
        number of model zones, default = 1300
    stations : bool
        whether it's a stations matrix or not'
    Returns
    -------
    expanded_mx : pd.DataFrame
        expanded matrix
    """
    # o/d columns
    od_cols = ["from_model_zone_id", "to_model_zone_id"]
    # if stations matrix, update the od columns
    if stations:
        od_cols = ["from_stn_zone_id", "to_stn_zone_id"]
    # create empty dataframe
    expanded_mx = pd.DataFrame(
        list(
            itertools.product(range(1, zones + 1), range(1, zones + 1))
        ),
        columns=od_cols,
    )
    # get first matrix
    expanded_mx = expanded_mx.merge(
        mx_df, how="outer", on=od_cols
    ).fillna(0)
    return expanded_mx


def filter_stations(stations_lookup, df):
    """
    I think this is redundant as the df is left merged to stations lookup on
    both filtered columns.
    """
    used_stations = stations_lookup["STATIONCODE"].to_list()
    df = df.loc[
        (df["ZoneCodeFrom"].isin(used_stations))
        & (df["ZoneCodeTo"].isin(used_stations))
    ]
    return df


def merge_to_stations(stations_lookup: pd.DataFrame, df: pd.DataFrame, left_from: str, left_to: str, right: str = 'STATIONCODE'):
    """
    Merge dataframe to stations lookup with the processing that goes with this.

    Parameters
    ----------

    stations_lookup: pd.DataFrame
        The stations lookup.
    df: pd.DataFrame
        The dataframe being merged to stations.
    left_from: str
        The name of the 'from' column df
    left_to: str
        The name of the 'to' column in df
    
    Returns
    -------

    DF merged to stations_lookup and renamed.
    """
    factors_df = df.merge(
        stations_lookup,
        how="inner",
        left_on=[left_from],
        right_on=[right],
    )
    # rename
    factors_df = factors_df.rename(
        columns={"stn_zone_id": "from_stn_zone_id"}
    )
    # merge on destination/attraction
    factors_df = factors_df.merge(
        stations_lookup,
        how="inner",
        left_on=[left_to],
        right_on=[right],
    )
    factors_df.dropna(axis=0, inplace=True)
    # rename
    factors_df = factors_df.rename(
        columns={
            "stn_zone_id": "to_stn_zone_id",
            "Demand_rate": "Demand",
        }
    )
    return factors_df


def convert_csv_2_mat(
    norms_segments: list,
    cube_exe: pathlib.Path,
    fcast_year: int,
    output_folder: pathlib.Path,
) -> None:
    """Convert CSV output matrices to Cube .MAT.
    Function converts output CSV matrices into a single Cube .MAT matrix
    in NoRMS input demand matrix format
    Parameters
    ----------
    norms_segments : list
        list of NoRMS input demand segments
    cube_exe : Path
        path to Cube Voyager executable
    fcast_year : int
        forecast year
    output_folder : Path
        path to folder where CSV matrices are saved. this is where the .MAT
        will also be saved to
    """
    # empty dictionary
    mats_dict = {}
    # create a dictionary of matrices and their paths
    for segment in norms_segments:
        mats_dict[segment] = pathlib.Path(
            output_folder, f"{fcast_year}_24Hr_{segment}.csv"
        )

    # call CUBE convertor class
    c_m = CUBEMatConverter(cube_exe)
    c_m.csv_to_mat(
        1300,
        mats_dict,
        pathlib.Path(output_folder, f"PT_24hr_Demand_{fcast_year}.MAT"),
        1,
    )


def zonal_from_to_stations_demand(
    demand_mx: pd.DataFrame,
    irsj_props: pd.DataFrame,
    stations_count: int,
    userclass: int,
    time_period
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Create a stn2stn matrix and produce a two way conversion lookup (zonal <> stations).

    Parameters
    ----------
    demand_mx : pd.DataFrame
        demand matrix dataframe
    irsj_props : pd.DataFrame
        iRSj split probabilities dataframe
    stations_count : int
        number of stations
    userclass : int
        segments userclass

    Returns
    -------
    np_mx: np.ndarray
        numpy wide matrix
    zonal_from_to_stns: pd.dataframe
        zonal <> stations conversion lookup

    """
    # expand matrix
    demand_mx = expand_matrix(demand_mx)
    # add userclass info
    demand_mx.loc[:, "userclass"] = userclass
    # merge demand matrix to iRSj probabilities
    mx_df = demand_mx.merge(
        irsj_props,
        how="left",
        on=["from_model_zone_id", "to_model_zone_id", "userclass"],
    )
    # find zone to zone movements with demand but no proportion
    missing_props = mx_df.loc[
        (mx_df["proportion"].isna()) & (mx_df["Demand"] > 0),
        ["from_model_zone_id", "to_model_zone_id", "Demand"],
    ]
    # rename column
    mx_df = mx_df.rename(columns={"proportion": "stn_from_zone"})
    # calculate movement demand proportion
    mx_df.loc[:, "Demand"] = mx_df["Demand"] * mx_df["stn_from_zone"]

    # sum stn2stn demand
    stn2stn_mx = (
        mx_df.groupby(["from_stn_zone_id", "to_stn_zone_id"])["Demand"]
        .sum()
        .reset_index()
    )
    # rename column
    stn2stn_mx = stn2stn_mx.rename(
        columns={"Demand": "stn2stn_total_demand"}
    )

    # join the stn2stn demand total to the main matrix
    mx_df = mx_df.merge(
        stn2stn_mx,
        how="left",
        on=["from_stn_zone_id", "to_stn_zone_id"],
    )
    # calculate stn 2 zone proportions
    mx_df.loc[:, "stn_to_zone"] = (
        mx_df["Demand"] / mx_df["stn2stn_total_demand"]
    )

    # create lookup dataframe
    zonal_from_to_stns = mx_df[
        [
            "from_model_zone_id",
            "from_stn_zone_id",
            "to_stn_zone_id",
            "to_model_zone_id",
            "stn_from_zone",
            "stn_to_zone",
        ]
    ]
    # create stn2stn matrix dataframe
    mx_df = mx_df[["from_stn_zone_id", "to_stn_zone_id", "Demand"]]
    # expand matrix
    mx_df = expand_matrix(mx_df, zones=stations_count, stations=True)
    # group by stations
    mx_df = (
        mx_df.groupby(["from_stn_zone_id", "to_stn_zone_id"])["Demand"]
        .sum()
        .reset_index()
    )
    # remove no-record stations
    mx_df = mx_df.loc[
        (mx_df["from_stn_zone_id"] != 0)
        & (mx_df["to_stn_zone_id"] != 0)
    ].reset_index(drop=True)

    # fill na
    zonal_from_to_stns = zonal_from_to_stns.fillna(0)
    mx_df = mx_df.fillna(0)

    # convert to long matrix
    cols = mx_df.columns
    np_mx = pandas_utils.long_to_wide_infill(
                    df=mx_df,
                    index_col=cols[0],
                    columns_col=cols[1],
                    values_col=cols[2],

                    infill=0
                ).values
    missing_props_demand = missing_props.Demand.sum() / mx_df.Demand.sum()
    if missing_props_demand > 0.01:
        raise ValueError(f"{missing_props_demand * 100}% of demand does not have irsj prop values for UC{userclass}, tp{time_period}. Stopping process.")
    else:
        return np_mx, zonal_from_to_stns

def convert_stns_to_zonal_demand(
    np_stns_mx: np.ndarray,
    zones_2_stns_lookup: pd.DataFrame,
    time_period: str,
    to_home: bool = False,
) -> np.ndarray:
    """Convert numpy stn2stn matrix to pandas zonal level matrix.

    Parameters
    ----------
    np_stns_mx : np.ndarray
        station 2 station level matrix
    zones_2_stns_lookup : pd.DataFrame
        zonal to/from stations conversion proportions dataframe
    time_period : str
        time period being processed
    to_home : bool
        whether it's ToHome demand segment

    Returns
    -------
    zonal_mx : np.array
        zonal matrix dataframe
    """
    # convert wide stns matrix to long stns matrix

    stns_mx = wide_to_long_np(np_stns_mx, cols=["from_stn_zone_id", "to_stn_zone_id", "Demand"])
    # join stns matrix to conversion lookup
    if to_home:
        zonal_mx = zones_2_stns_lookup.merge(
            stns_mx,
            how="left",
            left_on=["from_stn_zone_id", "to_stn_zone_id"],
            right_on=["from_stn_zone_id", "to_stn_zone_id"],
        )
    else:
        zonal_mx = zones_2_stns_lookup.merge(
            stns_mx,
            how="left",
            on=["from_stn_zone_id", "to_stn_zone_id"],
        )
    # calculate zonal demand
    zonal_mx["ZonalDemand"] = (
        zonal_mx["Demand"] * zonal_mx["stn_to_zone"]
    )
    # group to zonal level
    zonal_mx = (
        zonal_mx.groupby(["from_model_zone_id", "to_model_zone_id"])[
            "ZonalDemand"
        ]
        .sum()
        .reset_index()
    )
    # rename
    zonal_mx = zonal_mx.rename(
        columns={"ZonalDemand": f"{time_period}_Demand"}
    )
    cols = zonal_mx.columns
    # convert back to wide numpy matrix
    zonal_mx = pandas_utils.long_to_wide_infill(zonal_mx, cols[0], cols[1], cols[2], infill=0).values

    return zonal_mx

def split_irsj(irsj_dir: pathlib.Path, split_col: str, tp: str):
    """
    Splits an irsj prop files into separate files by userclass. Cycles through tps.
    Args:
        irsj_dir (pathlib.Path): Dir the prop files are saved in
        split_col (str): The column to split by (designed to be userclass)
        tps (list[str]): List of tps to read in for.
    """
    hdf_path = irsj_dir / f"{tp}_iRSj_probabilities.h5"
    df = pd.read_hdf(hdf_path)
    group = df.groupby(split_col)
    dfs = [group.get_group(i) for i in group.groups]
    write_path = irsj_dir / f"{tp}_iRSj_probabilities_split.h5"
    with pd.HDFStore(write_path, mode='w') as store:
        for df in dfs:
            # Save each split DataFrame as a separate key in the HDF5 store
            key = f'{split_col}_{df[split_col].unique()[0]}'
            store[key] = df

def wide_to_long_np(mx: np.array, cols: list[str] = ["from_model_zone_id","to_model_zone_id","Demand"]):
    """
    Wrapper around toolkit wide_to_long function to work on numpy array. Will add functionality to the base function
    soon.
    Parameters
    ----------
        mx (np.array): The matrix to be converted
        cols (list[str], optional): The columns for the long matrix. Defaults to ["from_model_zone_id","to_model_zone_id","Demand"].

    Returns
    -------
        pd.DataFrame: The requested long matrix
    """
    shp = mx.shape
    ind = range(1, shp[0] + 1)
    df = pd.DataFrame(mx, ind, ind)
    return pandas_utils.wide_to_long_infill(df, cols[0], cols[1], cols[2])