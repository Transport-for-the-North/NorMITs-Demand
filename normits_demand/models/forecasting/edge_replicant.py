# -*- coding: utf-8 -*-
"""EDGE Replicant process to grow demand."""
# ## IMPORTS ## #
# Standard imports
import logging
import itertools
import pathlib

# Third party imports
from typing import Tuple
import pandas as pd
import numpy as np
from tqdm import tqdm

# Local imports
from normits_demand.utils import timing
from normits_demand.matrices import omx_file
from normits_demand.utils import file_ops
from normits_demand.models.forecasting import forecast_cnfg
from normits_demand.matrices.cube_mat_converter import CUBEMatConverter

# ## CONSTANTS ## #
LOG = logging.getLogger(__name__)

# ## CLASSES ## #

# ## FUNCTIONS ## #


def long_mx_2_wide_mx(
    mx_df: pd.DataFrame,
    row: str = "no_entry",
    col: str = "no_entry",
    value: str = "no_entry",
) -> np.ndarray:
    """Convert pandas long matrix to numpy wide matrix.

    Function assumes default entry of a pandas dataframe of three columns:
        [origin/from/production, destination,to/attraction, demand]
    Can be specified if the matrix is of a different length or order.

    Parameters
    ----------
    mx_df : pd.DataFrame
        pandas long matrix dataframe to convert
    row : str, optional
        rows vector in the matrix dataframe
    col : str, optional
        columns vector in the matrix dataframe
    value : str, optional
        demand vector in the matrix dataframe

    Returns
    -------
    np.ndarray
        numpy wide matrix
    """
    # if user specified entries
    if row == "no_entry":
        row = mx_df.columns[0]
    if col == "no_entry":
        col = mx_df.columns[1]
    if value == "no_entry":
        value = mx_df.columns[2]

    # reshape to wide numpy matrix
    wide_mx = mx_df.pivot_table(index=row, columns=col, values=value).values

    return wide_mx


def wide_mx_2_long_mx(
    mx_np: np.ndarray,
    rows: str = "from_stn_zone_id",
    cols: str = "to_stn_zone_id",
    values: str = "Demand",
) -> pd.DataFrame:
    """Convert numpy wide matrix to pandas long matrix.

    Function assumes conversion is happening to a stn2stn matrix hence the headers
    for the output dataframe are named to station level by default. Optional entries
    can be given through rows, cols and values

    Parameters
    ----------
    mx_np : np.ndarray
        numpy wide matrix dataframe to convert
    row : str, optional
        rows vector in the matrix dataframe
    col : str, optional
        columns vector in the matrix dataframe
    value : str, optional
        demand vector in the matrix dataframe

    Returns
    -------
    mx_df : pd.DataFrame
        pandas long matrix
    """
    # get omx array to pandas dataframe and reset productions
    mx_df = pd.DataFrame(mx_np).reset_index().rename(columns={"index": rows})
    # melt DF to get attractions vector
    mx_df = mx_df.melt(id_vars=[rows], var_name=cols, value_name=values)
    # adjust zone number
    mx_df[rows] = mx_df[rows] + 1
    mx_df[cols] = mx_df[cols] + 1

    return mx_df


def transpose_matrix(mx_df: pd.DataFrame, stations: bool = False) -> pd.DataFrame:
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
        list(itertools.product(range(1, zones + 1), range(1, zones + 1))),
        columns=od_cols,
    )
    # get first matrix
    expanded_mx = expanded_mx.merge(mx_df, how="outer", on=od_cols).fillna(0)
    return expanded_mx


def add_distance_band_tag_flow(mx_df: pd.DataFrame) -> pd.DataFrame:
    """Add TAGs distance band based on distance.

    Parameters
    ----------
    mx_df : pd.DataFrame
        prepared matrix with flows

    Returns
    -------
    mx_df : pd.DataFrame
        dataframe with added new TAG flow
    """
    # set new flow to match the non-distance flow to begin with
    mx_df.loc[:, "TAG_Flow"] = mx_df["TAG_NonDist"]
    # Outside South East
    mx_df.loc[
        (mx_df["TAG_NonDist"] == "Outside South East".lower()) & (mx_df["Distance"] < 25),
        "TAG_Flow",
    ] = "Outside South East <25 miles".lower()
    mx_df.loc[
        (mx_df["TAG_NonDist"] == "Outside South East".lower())
        & (mx_df["Distance"] >= 25)
        & (mx_df["Distance"] < 100),
        "TAG_Flow",
    ] = "Outside South East 25 to 100 miles".lower()
    mx_df.loc[
        (mx_df["TAG_NonDist"] == "Outside South East".lower()) & (mx_df["Distance"] >= 100),
        "TAG_Flow",
    ] = "Outside South East  100 + miles - adjusted".lower()
    # Outside South East to/from London
    mx_df.loc[
        (mx_df["TAG_NonDist"] == "Outside South East to/from London".lower())
        & (mx_df["Distance"] < 100),
        "TAG_Flow",
    ] = "Outside South East to/from London < 100 miles".lower()
    mx_df.loc[
        (mx_df["TAG_NonDist"] == "Outside South East to/from London".lower())
        & (mx_df["Distance"] >= 100),
        "TAG_Flow",
    ] = "Outside South East to/from London 100 + miles".lower()

    return mx_df


def prepare_growth_matrices(
    demand_segments_df: pd.DataFrame,
    factors_df: pd.DataFrame,
    stations_lookup: pd.DataFrame,
) -> dict:
    """Prepare Growth Factor matrices into numpy matrices.

    Function creates growth numpy matrix for all purposes and ticket types combination
    on a station to station level.

    Parameters
    ----------
    demand_segments_df : pd.DataFrame
        models' demand segments information dataframe
    factors_df : pd.DataFrame
        EDGE growth factors dataframe
    stations_lookup : pd.DataFrame
        lookup for all model used stations zones and TLCs

    Returns
    -------
    growth_matrices : dict
        numpy growth matrices for all purposes and ticket types
    """
    # get list of purposes
    purposes = demand_segments_df["Purpose"].drop_duplicates().to_list()
    # get list of ticket types
    ticket_types = factors_df["TicketType"].drop_duplicates().to_list()
    # create a list of model used stations
    used_stations = stations_lookup["STATIONCODE"].to_list()
    # filter factors file to keep only used stations
    factors_df = factors_df.loc[
        (factors_df["ZoneCodeFrom"].isin(used_stations))
        & (factors_df["ZoneCodeTo"].isin(used_stations))
    ]

    # add stns zones
    # merge on origin/production
    factors_df = factors_df.merge(
        stations_lookup, how="left", left_on=["ZoneCodeFrom"], right_on=["STATIONCODE"]
    )
    # rename
    factors_df = factors_df.rename(columns={"stn_zone_id": "from_stn_zone_id"})
    # merge on destination/attraction
    factors_df = factors_df.merge(
        stations_lookup, how="left", left_on=["ZoneCodeTo"], right_on=["STATIONCODE"]
    )
    # rename
    factors_df = factors_df.rename(
        columns={"stn_zone_id": "to_stn_zone_id", "Demand_rate": "Demand"}
    )
    # keep needed columns
    factors_df = factors_df[
        [
            "from_stn_zone_id",
            "to_stn_zone_id",
            "ZoneCodeFrom",
            "ZoneCodeTo",
            "purpose",
            "TicketType",
            "Demand",
        ]
    ]

    # create matrices dictionary
    growth_matrices = {}

    # get growth matrices for each purpose/ticket type
    for purpose in purposes:
        for ticket_type in ticket_types:
            mx_df = factors_df[["from_stn_zone_id", "to_stn_zone_id", "Demand"]].loc[
                (factors_df["purpose"] == purpose) & (factors_df["TicketType"] == ticket_type)
            ]
            # expand matrix
            mx_df = expand_matrix(mx_df, zones=len(stations_lookup), stations=True)
            growth_matrices[f"{purpose}_{ticket_type}"] = long_mx_2_wide_mx(mx_df)

    return growth_matrices


def zonal_from_to_stations_demand(
    demand_mx: pd.DataFrame,
    irsj_props: pd.DataFrame,
    stations_count: int,
    userclass: int,
    to_home: bool = False,
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
    to_home : bool
        True if the demand is a ToHome demand

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
    # if ToHome demand then transpose matrix
    if to_home:
        demand_mx = transpose_matrix(demand_mx)
    # merge demand matrix to iRSj probabilities
    mx_df = demand_mx.merge(
        irsj_props, how="left", on=["from_model_zone_id", "to_model_zone_id", "userclass"]
    )
    # rename column
    mx_df = mx_df.rename(columns={"proportion": "stn_from_zone"})
    # calculate movement demand proportion
    mx_df.loc[:, "Demand"] = mx_df["Demand"] * mx_df["stn_from_zone"]

    # sum stn2stn demand
    stn2stn_mx = (
        mx_df.groupby(["from_stn_zone_id", "to_stn_zone_id"])["Demand"].sum().reset_index()
    )
    # rename column
    stn2stn_mx = stn2stn_mx.rename(columns={"Demand": "stn2stn_total_demand"})

    # join the stn2stn demand total to the main matrix
    mx_df = mx_df.merge(stn2stn_mx, how="left", on=["from_stn_zone_id", "to_stn_zone_id"])
    # calculate stn 2 zone proportions
    mx_df.loc[:, "stn_to_zone"] = mx_df["Demand"] / mx_df["stn2stn_total_demand"]

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
    mx_df = mx_df.groupby(["from_stn_zone_id", "to_stn_zone_id"])["Demand"].sum().reset_index()
    # remove no-record stations
    mx_df = mx_df.loc[
        (mx_df["from_stn_zone_id"] != 0) & (mx_df["to_stn_zone_id"] != 0)
    ].reset_index(drop=True)

    # fill na
    zonal_from_to_stns = zonal_from_to_stns.fillna(0)
    mx_df = mx_df.fillna(0)

    # convert to long matrix
    np_mx = long_mx_2_wide_mx(mx_df)

    return np_mx, zonal_from_to_stns


def produce_ticketype_splitting_matrices(
    edge_flows: pd.DataFrame,
    stations_lookup: pd.DataFrame,
    distance_mx: pd.DataFrame,
    flows_lookup: pd.DataFrame,
    ticket_split_proportions: pd.DataFrame,
) -> dict:
    """Produce numpy matrices to split stn2stn demand into the different ticket types.

    Parameters
    ----------
    edge_flows : pd.DataFrame
        all eDGE flows per station pair dataframe
    stations_lookup : pd.DataFrame
        lookup for all model used stations zones and TLCs
    distance_mx : pd.DataFrame
        stn2stn distance matrix
    flows_lookup : pd.DataFrame
        lookup for flow category names between EDGE and TAG
    ticket_split_proportions : pd.DataFrame
        ticket type splitting proportions dataframe

    Returns
    -------
    splitting_matrices : dict
        numpy ticket type splitting matrices for all purposes and ticket types
    """
    # normalize flows
    edge_flows.loc[:, "FlowCatName"] = edge_flows["FlowCatName"].str.lower()
    flows_lookup.loc[:, "FlowCatName"] = flows_lookup["FlowCatName"].str.lower()
    flows_lookup.loc[:, "TAG_NonDist"] = flows_lookup["TAG_NonDist"].str.lower()
    ticket_split_proportions.loc[:, "TAG_Flow"] = ticket_split_proportions[
        "TAG_Flow"
    ].str.lower()

    # create a list of model used stations
    used_stations = stations_lookup["STATIONCODE"].to_list()
    # filter flows to keep only used stations
    edge_flows = edge_flows.loc[
        (edge_flows["FromCaseZoneID"].isin(used_stations))
        & (edge_flows["ToCaseZoneID"].isin(used_stations))
    ]
    # add flows TAG category
    edge_flows = edge_flows.merge(flows_lookup, how="left", on=["FlowCatName"])
    # add station zones
    # origin/production
    edge_flows = edge_flows.merge(
        stations_lookup, how="left", left_on=["FromCaseZoneID"], right_on=["STATIONCODE"]
    )
    edge_flows = edge_flows.rename(columns={"stn_zone_id": "from_stn_zone_id"})
    # destination/attraction
    edge_flows = edge_flows.merge(
        stations_lookup, how="left", left_on=["ToCaseZoneID"], right_on=["STATIONCODE"]
    )
    edge_flows = edge_flows.rename(columns={"stn_zone_id": "to_stn_zone_id"})
    # keep needed columns
    edge_flows = edge_flows[
        [
            "from_stn_zone_id",
            "to_stn_zone_id",
            "FromCaseZoneID",
            "ToCaseZoneID",
            "FlowCatName",
            "TAG_NonDist",
        ]
    ]
    # merge distance to flows
    edge_flows = edge_flows.merge(
        distance_mx, how="left", on=["from_stn_zone_id", "to_stn_zone_id"]
    )
    # rename
    edge_flows = edge_flows.rename(columns={"tran_distance": "Distance"})
    # fill na
    edge_flows = edge_flows.fillna(0)
    # allocate distance bands
    edge_flows = add_distance_band_tag_flow(edge_flows)
    # keep needed columns
    edge_flows = edge_flows[
        ["from_stn_zone_id", "to_stn_zone_id", "FromCaseZoneID", "ToCaseZoneID", "TAG_Flow"]
    ]
    # merge ticket split factors
    edge_flows = edge_flows.merge(ticket_split_proportions, how="left", on=["TAG_Flow"])
    # get list of purposes
    purposes = edge_flows["Purpose"].drop_duplicates().to_list()
    # create matrices dictionary
    splitting_matrices = {}
    # create numpy splitting matrices
    for purpose in purposes:
        for ticketype in ["F", "R", "S"]:
            # get current purpose
            mx_df = edge_flows.loc[edge_flows["Purpose"] == purpose].reset_index(drop=True)
            # keep needed columns
            mx_df = mx_df[["from_stn_zone_id", "to_stn_zone_id", ticketype]]
            # rename
            mx_df = mx_df.rename(columns={ticketype: "Demand"})
            # expand matrix
            mx_df = expand_matrix(mx_df, zones=len(stations_lookup), stations=True)
            # convert to numpy and add to matrices dictionary
            splitting_matrices[f"{purpose}_{ticketype}"] = long_mx_2_wide_mx(mx_df)

    return splitting_matrices


def fill_missing_factors(
    purposes: list,
    growth_matrices: dict,
) -> dict:
    """Fills missing factors for specific ticket type with an available one.

    The filling process looks for an available factor for the same station pair
    and journey purpose and follows the below hierarchy in filling missing factors:

        Full tickets: First look for Reduced and then look for Season
        Reduced tickets: First look for Full and then look for Season
        Season tickets: First look for Reduced and then look for Full

    function also adds a growth factor of 1 (i.e. no growth) where no factor is available

    Parameters
    ----------
    purposes : list
        list of journey purposes
    growth_matrices : dict
        numpy growth matrices for all journey purposes and ticket types

    Returns
    -------
    filled_growth_matrices : dict
        filled growth factor matrices
    """
    # order S: R, F
    filled_growth_matrices = {}
    for purpose in purposes:
        # get current matrices
        f_mx = growth_matrices[f"{purpose}_F"]
        r_mx = growth_matrices[f"{purpose}_R"]
        s_mx = growth_matrices[f"{purpose}_S"]
        # create a new growth factors matrix and fill from other ticket types
        # full - order F > R > S
        filled_f_mx = np.where(f_mx == 0, r_mx, f_mx)
        filled_f_mx = np.where(filled_f_mx == 0, s_mx, filled_f_mx)
        filled_f_mx = np.where(filled_f_mx == 0, 1, filled_f_mx)
        # reduced - order R > F > S
        filled_r_mx = np.where(r_mx == 0, f_mx, r_mx)
        filled_r_mx = np.where(filled_r_mx == 0, s_mx, filled_r_mx)
        filled_r_mx = np.where(filled_r_mx == 0, 1, filled_r_mx)
        # season - order S > R > F
        filled_s_mx = np.where(s_mx == 0, r_mx, s_mx)
        filled_s_mx = np.where(filled_s_mx == 0, f_mx, filled_s_mx)
        filled_s_mx = np.where(filled_s_mx == 0, 1, filled_s_mx)

        # append to filled matrices
        filled_growth_matrices[f"{purpose}_F"] = filled_f_mx
        filled_growth_matrices[f"{purpose}_R"] = filled_r_mx
        filled_growth_matrices[f"{purpose}_S"] = filled_s_mx

    return filled_growth_matrices


def apply_demand_growth(
    stn2stn_base_mx: np.ndarray,
    splitting_matrices: dict,
    filled_growth_matrices: dict,
    matrix_zones: int,
    purpose: str,
    growth_method: int,
    to_home: bool = False,
) -> np.ndarray:
    """Apply growth factors to base matrix on ticketype level based on growth method.

    growth_method = 1; apply the growth on PA level
    growth_method = 2; average of the two directions

    Parameters
    ----------
    stn2stn_base_mx : np.array
        station 2 station level base matrix
    splitting_matrices : dict
        dictionary of ticketype/purpose splitting matrices
    filled_growth_matrices : dict
        dictionary of growth factor matrices by purpose nad ticketype
    matrix_zones : int
        number of zones on the matrix
    purpose : str
        current matrix's journey purpose
    growth_method : int
        growth method to be applied, 1> PA, 2> Average
    to_home : bool
        whether or not this is a ToHome demand segment

    Returns
    -------
    stn2stn_forecast_mx : np.ndarray
        grown stn2stn demand matrix
    """
    # create a total stn2stn demand array to regroup the grown ticket demand into
    stn2stn_forecast_mx = np.empty(shape=[matrix_zones, matrix_zones])
    # split matrix to ticket types and apply growth
    for ticketype in ["F", "R", "S"]:
        ticketype_np_matrix = stn2stn_base_mx * splitting_matrices[f"{purpose}_{ticketype}"]
        # transpose ToHome demand
        if to_home:
            ticketype_np_matrix = ticketype_np_matrix.transpose()
        # get growth matrix
        if growth_method == 1:
            growth_mx = filled_growth_matrices[f"{purpose}_{ticketype}"]
        else:
            growth_mx = (
                filled_growth_matrices[f"{purpose}_{ticketype}"]
                + filled_growth_matrices[f"{purpose}_{ticketype}"].transpose()
            ) / 2
        # apply growth
        ticketype_np_matrix = ticketype_np_matrix * growth_mx
        # sum grown demand
        stn2stn_forecast_mx = stn2stn_forecast_mx + ticketype_np_matrix

    # transpose ToHome demand
    if to_home:
        stn2stn_forecast_mx = stn2stn_forecast_mx.transpose()

    return stn2stn_forecast_mx


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
    stns_mx = wide_mx_2_long_mx(np_stns_mx)
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
            stns_mx, how="left", on=["from_stn_zone_id", "to_stn_zone_id"]
        )
    # calculate zonal demand
    zonal_mx["ZonalDemand"] = zonal_mx["Demand"] * zonal_mx["stn_to_zone"]
    # fill na
    zonal_mx = zonal_mx.fillna(0)
    # group to zonal level
    zonal_mx = (
        zonal_mx.groupby(["from_model_zone_id", "to_model_zone_id"])["ZonalDemand"]
        .sum()
        .reset_index()
    )
    # rename
    zonal_mx = zonal_mx.rename(columns={"ZonalDemand": f"{time_period}_Demand"})
    # convert back to wide numpy matrix
    zonal_mx = long_mx_2_wide_mx(zonal_mx)

    return zonal_mx


def fromto_2_from_by_averaging(
    matrices_dict: dict, norms_segments: list, all_segments: list
) -> dict:
    """Produce the FromHome demand by averaging FromHome and ToHome.


    Function combines From/To by averaging the two directions to produce the 19
    segments needed by NoRMS

    Parameters
    ----------
    matrices_dict : dictionary
        24Hr demand matrices dictionary
    norms_segments : list
        list of NoRMS demand segments
    all_segments: list
        all demand segments in a From/To format

    Returns
    -------
    matrices : dictionary
        dictionary of matrices
    """
    # empty dictionary
    matrices = {}

    # loop over all norms segments
    for segment in norms_segments:
        # check if the segment has a ToHome component or if it's a non-home based
        if (segment + "_T" in all_segments) and (segment[:3].lower() != "NHB".lower()):
            # average the FromHome and the transposition of the toHome
            matrices[segment] = (
                matrices_dict[segment] + matrices_dict[segment + "_T"].transpose()
            ) / 2

        else:
            # Keep as it is
            matrices[segment] = matrices_dict[segment]

    return matrices


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
        mats_dict[segment] = pathlib.Path(output_folder, f"{fcast_year}_24Hr_{segment}.csv")

    # call CUBE convertor class
    c_m = CUBEMatConverter(cube_exe)
    c_m.csv_to_mat(
        1300, mats_dict, pathlib.Path(output_folder, f"PT_24hr_Demand_{fcast_year}.MAT"), 1
    )


# TODO (MI): Possibly modularize this into 2 or 3 modules. e.g. growth matrices, splitting matrices, growth application
def run_edge_growth(params: forecast_cnfg.EDGEParameters) -> None:
    """Run Growth Process."""
    LOG.info("#" * 80)
    LOG.info("Started Process @ %s", timing.get_datetime())
    LOG.info("#" * 80)

    # fixed objects
    time_periods = ["AM", "IP", "PM", "OP"]

    # read global input files
    demand_segments = file_ops.read_df(params.demand_segments)
    demand_segments.loc[:, "ToHome"] = demand_segments["ToHome"].astype(bool)
    model_stations_tlcs = file_ops.read_df(params.norms_to_edge_stns_path)
    ticket_splits_df = file_ops.read_df(params.ticket_type_splits_path)
    flow_cats = file_ops.read_df(params.flow_cat_path)
    edge_flows = file_ops.read_df(params.edge_flows_path, usecols=[0, 2, 5])
    # declare journey purposes
    purposes = demand_segments["Purpose"].drop_duplicates().to_list()

    # demand segment list groups
    # NoRMS demand segments
    norms_segments = (
        demand_segments.loc[demand_segments["ModelSegment"] == 1][["Segment"]]
        .drop_duplicates()
        .values.tolist()
    )
    norms_segments = [segment for sublist in norms_segments for segment in sublist]
    # all segments
    all_segments = demand_segments["Segment"].to_list()

    # loop over forecast years
    for forecast_year in params.forecast_years:
        LOG.info("**** Applying growth for %s @ %s", forecast_year, timing.get_datetime())
        # read input files
        growth_factors = file_ops.read_df(
            params.edge_growth_dir / params.forecast_years[forecast_year]
        )
        # produce growth matrices
        growth_matrices = prepare_growth_matrices(
            demand_segments, growth_factors, model_stations_tlcs
        )
        # fill growth matrices
        filled_growth_matrices = fill_missing_factors(
            purposes,
            growth_matrices,
        )
        # TODO (MI): Record where no factor or a different factor is being used and log
        # create empty dictionary to store matrices
        factored_matrices = {}
        factored_24hr_matrices = {}
        # empty dataframe for growth summary
        growth_summary = pd.DataFrame(
            {
                "Time_Period": [],
                "Demand_Segment": [],
                "Base_Demand": [],
                f"{forecast_year}_Demand": [],
            }
        )
        # loop over time periods
        for time_period in tqdm(
            time_periods,
            desc="Time Periods Loop ",
            unit=" Period",
            colour="cyan",
            total=len(time_periods),
        ):
            LOG.info(
                "-- Processing Time Period %s @ %s",
                time_period,
                timing.get_datetime(),
            )
            # create dictionary
            factored_matrices[time_period] = {}
            # read time period specific files
            irsj_props = pd.read_hdf(
                params.matrices_to_grow_dir / f"{time_period}_iRSj_probabilities.h5",
                key="iRSj",
            )
            dist_mx = pd.read_csv(
                params.matrices_to_grow_dir / f"{time_period}_stn2stn_costs.csv",
                usecols=[0, 1, 4],
            )
            # produce ticket type splitting matrices
            splitting_matrices = produce_ticketype_splitting_matrices(
                edge_flows, model_stations_tlcs, dist_mx, flow_cats, ticket_splits_df
            )
            LOG.debug(
                f"{'Time_Period':>12}{'Demand_Segment':>15}"
                f"{'Base_Demand':>12}{f'{forecast_year}_Demand':>12}"
            )
            # TODO (MI): Potentially multiprocessing
            # loop over demand segments
            for i, row in tqdm(
                demand_segments.iterrows(),
                desc="    Demand Segments Loop ",
                unit=" Segment",
                colour="cyan",
                total=len(demand_segments),
            ):
                # store current segment's details
                segment = row["Segment"]
                to_home = row["ToHome"]
                growth_method = row["Growth_Method"]
                userclass = row["Userclass"]
                purpose = row["Purpose"]
                # read demand matrix
                with omx_file.OMXFile(
                    pathlib.Path(params.matrices_to_grow_dir, f"PT_{time_period}.omx")
                ) as omx_mat:
                    # read segment matrix into a dataframe
                    zonal_base_demand_mx = wide_mx_2_long_mx(
                        omx_mat.get_matrix_level(segment),
                        rows="from_model_zone_id",
                        cols="to_model_zone_id",
                    )
                # check if matrix has no demand then continue
                if zonal_base_demand_mx["Demand"].sum() == 0:
                    # keep matrix as it is, i.e. = 0
                    factored_matrices[time_period][segment] = long_mx_2_wide_mx(
                        zonal_base_demand_mx
                    )
                    LOG.debug(f"{time_period:>12}{segment:>15}" f"{0:>12}{0:>12}")
                    continue
                # reduce probabilities to current userclass
                irsj_probs_segment = irsj_props.loc[
                    irsj_props["userclass"] == userclass
                ].reset_index(drop=True)
                # convert matrix to numpy stn2stn and produce a conversion lookup
                np_stn2stn_base_demand_mx, zonal_from_to_stns = zonal_from_to_stations_demand(
                    zonal_base_demand_mx,
                    irsj_probs_segment,
                    len(model_stations_tlcs),
                    userclass,
                    to_home,
                )
                # store matrix total demand
                tot_input_demand = round(np_stn2stn_base_demand_mx.sum())
                # apply growth
                np_stn2stn_grown_demand_mx = apply_demand_growth(
                    np_stn2stn_base_demand_mx,
                    splitting_matrices,
                    filled_growth_matrices,
                    len(np_stn2stn_base_demand_mx),
                    purpose,
                    growth_method,
                    to_home,
                )
                # TODO (MI): Check the proportion of the demand that is not being grown, if > 1% then terminate
                # store matrix total demand
                tot_output_demand = round(np_stn2stn_grown_demand_mx.sum())
                LOG.debug(
                    f"{time_period:>12}{segment:>15}"
                    f"{tot_input_demand:>12}{tot_output_demand:>12}"
                )
                # append to growth summary df
                # empty dataframe for growth summary
                segment_growth_summary = pd.DataFrame(
                    {
                        "Time_Period": [time_period],
                        "Demand_Segment": [segment],
                        "Base_Demand": [tot_input_demand],
                        f"{forecast_year}_Demand": [tot_output_demand],
                    }
                )
                growth_summary = pd.concat([growth_summary, segment_growth_summary], axis=0)
                # calculate applied growth factor
                # np_stn2stn_growth_mx = np_stn2stn_grown_demand_mx / np_stn2stn_base_demand_mx
                # convert back to zonal level demand
                zonal_grown_demand_mx = convert_stns_to_zonal_demand(
                    np_stn2stn_grown_demand_mx,
                    zonal_from_to_stns,
                    time_period,
                    to_home,
                )
                # add to grown matrices dictionary
                factored_matrices[time_period][segment] = zonal_grown_demand_mx

        # prepare 24Hr level demand matrices
        for i, row in demand_segments.iterrows():
            # declare current segment's details
            segment = row["Segment"]
            # get 24Hr demand matrix
            forecast_matrix_24hr = (
                factored_matrices["AM"][segment]
                + factored_matrices["IP"][segment]
                + factored_matrices["PM"][segment]
                + factored_matrices["OP"][segment]
            )
            # add to 24Hr matrices dict
            factored_24hr_matrices[segment] = forecast_matrix_24hr

        # Combine matrices into NoRMS segments
        norms_matrices = fromto_2_from_by_averaging(
            factored_24hr_matrices, norms_segments, all_segments
        )

        # export files
        # TODO (MI): Export to .OMX and then convert .OMX to .MAT instead of the .CSVs
        for segment in norms_segments:
            # write out demand matrix
            file_ops.write_df(
                wide_mx_2_long_mx(
                    norms_matrices[segment], rows="from_model_zone_id", cols="to_model_zone_id"
                ).sort_values(by=["from_model_zone_id", "to_model_zone_id"]),
                params.export_path / f"{forecast_year}_24Hr_{segment}.csv",
                index=False,
            )
        # convert to Cube .MAT
        convert_csv_2_mat(norms_segments, params.cube_exe, forecast_year, params.export_path)
        # export growth summary
        file_ops.write_df(
            growth_summary,
            params.export_path / f"{forecast_year}_Growth_Summary.csv",
            index=False,
        )
