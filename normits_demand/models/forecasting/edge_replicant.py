# -*- coding: utf-8 -*-
"""EDGE Replicant process to grow demand."""
# ## IMPORTS ## #
# Standard imports
import logging
import itertools
import pathlib

# Third party imports
import pandas as pd
from tqdm import tqdm

# Local imports
# pylint: disable=import-error,wrong-import-position
from normits_demand.utils import timing
from normits_demand.utils import file_ops
from normits_demand.models.forecasting import forecast_cnfg
from normits_demand.matrices.cube_mat_converter import CUBEMatConverter
# pylint: enable=import-error,wrong-import-position

# ## CONSTANTS ## #
LOG = logging.getLogger(__name__)

# ## CLASSES ## #

# ## FUNCTIONS ## #


def add_tlc2stations_matrix(mx_df: pd.DataFrame, stn_tlc: pd.DataFrame) -> pd.DataFrame:
    """Add station TLCs to the stations matrix.

    Parameters
    ----------
    mx_df : pd.DataFrame
        stn 2 stn matrix dataframe.
    stn_tlc : pd.DataFrame
        station zone ID to TLC lookup.

    Returns
    -------
    mx_df : pd.DataFrame
        matrix with TLCs.

    """
    # add TLCs
    mx_df = mx_df.merge(
        stn_tlc, how="left", left_on=["from_stn_zone_id"], right_on=["stn_zone_id"]
    )
    # keep needed columns
    mx_df = mx_df[
        [
            "from_model_zone_id",
            "to_model_zone_id",
            "from_stn_zone_id",
            "to_stn_zone_id",
            "userclass",
            "Distance",
            "Demand",
            "STATIONCODE",
            "STATIONNAME",
        ]
    ]
    # rename column
    mx_df = mx_df.rename(columns={"STATIONCODE": "O_TLC", "STATIONNAME": "O_StnName"})
    # add TLCs
    mx_df = mx_df.merge(
        stn_tlc, how="left", left_on=["to_stn_zone_id"], right_on=["stn_zone_id"]
    )
    # keep needed columns
    mx_df = mx_df[
        [
            "from_model_zone_id",
            "to_model_zone_id",
            "from_stn_zone_id",
            "O_TLC",
            "O_StnName",
            "to_stn_zone_id",
            "STATIONCODE",
            "STATIONNAME",
            "userclass",
            "Distance",
            "Demand",
        ]
    ]
    # rename column
    mx_df = mx_df.rename(columns={"STATIONCODE": "D_TLC", "STATIONNAME": "D_StnName"})

    return mx_df


def prepare_stn2stn_matrix(
    demand_mx: pd.DataFrame,
    irsj_props: pd.DataFrame,
    dist_mx: pd.DataFrame,
    stn_tlc: pd.DataFrame,
    to_home: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare stn 2 stn matrix with TLCs and distances from ij matrix.

    Parameters
    ----------
    demand_mx : pd.DataFrame
        demand matrix dataframe
    irsj_props : pd.DataFrame
        iRSj split probabilities dataframe
    dist_mx : pd.DataFrame
        stn2stn distance matrix
    stn_tlc : pd.DataFrame
        station zone id to TLC lookup dataframe
    to_home : bool
        True if the demand is a ToHome demand

    Returns
    -------
    mx_df : pd.DataFrame
        demand matrix with added attributes of Distances and TLCs
    unassigned_demand : pd.DataFrame
        demand movements that are not assigned

    """
    # if ToHome demand then transpose matrix
    if to_home:
        demand_mx = transpose_matrix(demand_mx)
    # merge demand matrix to iRSj probabilities
    mx_df = demand_mx.merge(
        irsj_props,
        how="outer",
        on=["from_model_zone_id", "to_model_zone_id", "userclass"],
        indicator=True,
    )
    # get unassigned demand
    unassigned_demand = mx_df.loc[mx_df["_merge"] == "left_only"]
    # keep needed columns
    unassigned_demand = unassigned_demand[["from_model_zone_id", "to_model_zone_id", "Demand"]]
    # keep only assigned demand in the matrix to carry forward to growth
    mx_df = mx_df.loc[mx_df["_merge"] == "both"]
    # drop _merge column
    mx_df = mx_df.drop(["_merge"], axis=1)
    # calculate movement demand proportion
    mx_df.loc[:, "Demand"] = mx_df["Demand"] * mx_df["proportion"]
    # group by stn2stn
    mx_df = (
        mx_df.groupby(
            [
                "from_model_zone_id",
                "to_model_zone_id",
                "from_stn_zone_id",
                "to_stn_zone_id",
                "userclass",
            ]
        )["Demand"]
        .sum()
        .reset_index()
    )
    # remove records of zero stations
    mx_df = mx_df.loc[mx_df["from_stn_zone_id"] != 0].reset_index()
    # add distance matrix to get stn2stn distance
    mx_df = mx_df.merge(dist_mx, how="left", on=["from_stn_zone_id", "to_stn_zone_id"])
    # keep needed columns
    mx_df = mx_df[
        [
            "from_model_zone_id",
            "to_model_zone_id",
            "from_stn_zone_id",
            "to_stn_zone_id",
            "userclass",
            "tran_distance",
            "Demand",
        ]
    ]
    # rename column
    mx_df = mx_df.rename(columns={"tran_distance": "Distance"})
    # add TLCs
    mx_df = add_tlc2stations_matrix(mx_df, stn_tlc)

    return mx_df, unassigned_demand


def assign_edge_flow(
    flows_file: pd.DataFrame, flows_lookup: pd.DataFrame, mx_df: pd.DataFrame
) -> pd.DataFrame:
    """Assign EDGE flow to each stn2stn movement.

    Parameters
    ----------
    flows_file : pd.DataFrame
        EDGE flows lookup dataframe
    flows_lookup: pd.DataFrame
        lookup dataframe between EDGE flows and TAG non-distance flows
    mx_df : pd.DataFrame
        stn2stn matrix to assign flows to

    Returns
    -------
    mx_df : pd.DataFrame
        demand matrix with added EDGE flows
    """
    # rename column
    edge_flows = flows_file.rename(
        columns={"FromCaseZoneID": "O_TLC", "ToCaseZoneID": "D_TLC"}
    )
    # keep needed columns
    edge_flows = edge_flows[["O_TLC", "D_TLC", "FlowCatID", "FlowCatName"]]
    # merge to matrix
    mx_df = mx_df.merge(edge_flows, how="left", on=["O_TLC", "D_TLC"])
    # add TAG flows
    mx_df = mx_df.merge(flows_lookup, how="left", on=["FlowCatName"])

    return mx_df


def assign_purposes(mx_df: pd.DataFrame) -> pd.DataFrame:
    """Assign Journey Purpose based on userclass.

    Add purpose category to a dataframe based on userclass as below:
        1-3: EB
        4-6: Com
        7-9: Oth

    Parameters
    ----------
    mx_df : pd.DataFrame
        dataframe with userclass info to add purpose info to

    Returns
    -------
    mx_df : pd.DataFrame
        dataframe with added userclass info
    """
    # assign journey purpose to userclasses
    userclass_lookup = {
        **dict.fromkeys((1, 2, 3), "Business"),
        **dict.fromkeys((4, 5, 6), "Commuting"),
        **dict.fromkeys((7, 8, 9), "Leisure"),
    }
    mx_df.loc[:, "Purpose"] = mx_df["userclass"].replace(userclass_lookup)

    return mx_df


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
        (mx_df["TAG_NonDist"] == "Outside South East") & (mx_df["Distance"] < 25), "TAG_Flow"
    ] = "Outside South East <25 miles"
    mx_df.loc[
        (mx_df["TAG_NonDist"] == "Outside South East")
        & (mx_df["Distance"] >= 25)
        & (mx_df["Distance"] < 100),
        "TAG_Flow",
    ] = "Outside South East 25 to 100 miles"
    mx_df.loc[
        (mx_df["TAG_NonDist"] == "Outside South East") & (mx_df["Distance"] >= 100), "TAG_Flow"
    ] = "Outside South East  100 + miles - adjusted"
    # Outside South East to/from London
    mx_df.loc[
        (mx_df["TAG_NonDist"] == "Outside South East to/from London")
        & (mx_df["Distance"] < 100),
        "TAG_Flow",
    ] = "Outside South East to/from London < 100 miles"
    mx_df.loc[
        (mx_df["TAG_NonDist"] == "Outside South East to/from London")
        & (mx_df["Distance"] >= 100),
        "TAG_Flow",
    ] = "Outside South East to/from London 100 + miles"

    return mx_df


def apply_ticket_splits(mx_df: pd.DataFrame) -> pd.DataFrame:
    """Split demand by ticket type.

    Parameters
    ----------
    mx_df : pd.DataFrame
        prepared demand matrix with ticket split proportions

    Returns
    -------
    mx_df : pd.DataFrame
        demand matrix by flow, ticket type and purpose
    """
    # re-balance proportion to adjust any possible loss due to precision
    for c in ("F", "R", "S"):
        mx_df.loc[:, c + "_Adj"] = mx_df[c] / (mx_df["F"] + mx_df["R"] + mx_df["S"])

    # apply split proportions by ticket type
    for c in ("F", "R", "S"):
        mx_df.loc[:, c] = mx_df[c + "_Adj"] * mx_df["Demand"]

    # keep needed columns
    mx_df = mx_df[
        [
            "from_model_zone_id",
            "to_model_zone_id",
            "from_stn_zone_id",
            "O_TLC",
            "to_stn_zone_id",
            "D_TLC",
            "userclass",
            "Purpose",
            "F",
            "R",
            "S",
        ]
    ]

    return mx_df


def create_factors_for_missing_moira_movements(
    to_home: bool,
    mx_df: pd.DataFrame,
    edge_factors: pd.DataFrame,
    other_tickets_df: pd.DataFrame,
    no_factors_df: pd.DataFrame,
    internal_zone_limit: int = 1157,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Produce Factors for missing movements/purposes/ticket types from other movements.


    some movements don't have factors for specific ticket types or purposes given these do not
    appear in MOIRA, hence this function populate growth factor records for missing
    movements/ticket types with available ticket types/purposes for the same movement

    Parameters
    ----------
    to_home : bool
        bool to flag whether or not the segment is a ToHome segment
    mx_df : pd.DataFrame
        prepared stn2stn matrix by flow, ticket type and purpose
    edge_factors : pd.DataFrame
        EDGE growth factors by flow, ticket type and purpose
    other_tickets_df : pd.DataFrame
        dataframe to record movements where other ticket types where used to fill
        for missing ticket types
    no_factors_df : pd.DataFrame
        dataframe to record movements where no factor was found at all
    internal_zone_limit : int
        last zone in the internal range of zones
    Returns
    -------
    upd_edge_factors : pd.DataFrame
        updated edge factors dataframe with added records for missing movements/tickets
    other_tickets_df: pd.DataFrame
        movements where factor of other tickets was used
    no_factors_df: pd.DataFrame
        movements with no factor at all
    """
    # keep needed columns
    edge_factors = edge_factors[
        ["ZoneCodeFrom", "ZoneCodeTo", "purpose", "TicketType", "Demand_rate", "Flag"]
    ]
    # remove records with growth of nan
    edge_factors = edge_factors[~edge_factors["Demand_rate"].isna()]
    # copy of factors dataframe
    upd_edge_factors = edge_factors.copy()
    # melt Matrix
    mx_df = pd.melt(
        mx_df,
        id_vars=[
            "from_model_zone_id",
            "to_model_zone_id",
            "from_stn_zone_id",
            "O_TLC",
            "to_stn_zone_id",
            "D_TLC",
            "userclass",
            "Purpose",
        ],
        value_vars=["F", "R", "S"],
    )
    # rename column
    mx_df = mx_df.rename(
        columns={
            "value": "T_Demand",
            "variable": "TicketType",
            "O_TLC": "ZoneCodeFrom",
            "D_TLC": "ZoneCodeTo",
            "Purpose": "purpose",
        }
    )
    # for ToHome segments, transpose the stations directionality
    if to_home:
        # rename column
        mx_df = mx_df.rename(
            columns={
                "ZoneCodeFrom": "ZoneCodeTo",
                "ZoneCodeTo": "ZoneCodeFrom",
            }
        )
    # merge factors to matrix
    mx_df = mx_df.merge(
        edge_factors, how="left", on=["ZoneCodeFrom", "ZoneCodeTo", "TicketType", "purpose"]
    )

    # get records of movements missing in MOIRA
    missing_moira = mx_df[mx_df["Demand_rate"].isna()].reset_index(drop=True)
    # copy with zonal info
    missing_moira_zonal = missing_moira.copy()
    # add internal flag
    missing_moira_zonal["Internal"] = 0
    missing_moira_zonal.loc[
        (missing_moira_zonal["from_model_zone_id"] <= internal_zone_limit)
        & (missing_moira_zonal["to_model_zone_id"] <= internal_zone_limit),
        "Internal",
    ] = 1

    # group to stn2stn
    missing_moira = (
        missing_moira.groupby(["ZoneCodeFrom", "ZoneCodeTo", "TicketType", "purpose"])[
            "T_Demand"
        ]
        .sum()
        .reset_index()
    )

    # merge factors
    missing_moira = missing_moira.merge(
        edge_factors, how="left", on=["ZoneCodeFrom", "ZoneCodeTo", "purpose"]
    )
    missing_moira_zonal = missing_moira_zonal.merge(
        edge_factors, how="left", on=["ZoneCodeFrom", "ZoneCodeTo", "purpose"]
    )
    # rename column
    missing_moira = missing_moira.rename(
        columns={"TicketType_y": "Available_TicketType", "TicketType_x": "Missing_TicketType"}
    )
    missing_moira_zonal = missing_moira_zonal.rename(
        columns={"TicketType_y": "Available_TicketType", "TicketType_x": "Missing_TicketType"}
    )
    # keep those that have available records
    available_ticket = missing_moira[~missing_moira["Available_TicketType"].isna()]
    # keep one ticket type
    available_ticket = available_ticket.drop_duplicates(
        subset=["ZoneCodeFrom", "ZoneCodeTo", "Missing_TicketType", "purpose"]
    )
    # keep needed columns
    available_ticket = available_ticket[
        [
            "ZoneCodeFrom",
            "ZoneCodeTo",
            "purpose",
            "Missing_TicketType",
            "Available_TicketType",
            "Demand_rate",
        ]
    ]
    # rename columns
    available_ticket = available_ticket.rename(columns={"Missing_TicketType": "TicketType"})
    # create flag with '0' indicating it's a populated factor
    available_ticket["Flag"] = 0
    # add to EDGE factors
    upd_edge_factors = pd.concat([upd_edge_factors, available_ticket], axis=0)

    # logging DFs
    # keep those that have available records
    available_ticket_log = missing_moira_zonal[
        ~missing_moira_zonal["Available_TicketType"].isna()
    ]
    available_ticket_log = (
        available_ticket_log.groupby(
            [
                "ZoneCodeFrom",
                "ZoneCodeTo",
                "purpose",
                "Missing_TicketType",
                "Available_TicketType",
                "Internal",
            ]
        )["T_Demand"]
        .sum()
        .reset_index()
    )
    # log these movements to main dataframe
    other_tickets_df = pd.concat([other_tickets_df, available_ticket_log], axis=0)

    # check missing tickets for current purpose in different purposes
    missing_ticket = missing_moira_zonal[missing_moira_zonal["Available_TicketType"].isna()]
    # keep needed columns
    missing_ticket = (
        missing_ticket.groupby(["ZoneCodeFrom", "ZoneCodeTo", "Internal"])["T_Demand"]
        .sum()
        .reset_index()
    )
    # log these movements to main dataframe
    no_factors_df = pd.concat([no_factors_df, missing_ticket], axis=0)

    return upd_edge_factors, other_tickets_df, no_factors_df


def apply_edge_growth_method1(
    mx_df: pd.DataFrame, edge_factors: pd.DataFrame, to_home: bool = False
) -> pd.DataFrame:
    """Grow demand by factoring it by EDGE factors using method 1.

    using method 1 where the factors are applied on P=O and A=D level

    Parameters
    ----------
    mx_df : pd.DataFrame
        prepared stn2stn matrix by flow, ticket type and purpose
    edge_factors : pd.DataFrame
        EDGE growth factors by flow, ticket type and purpose
    to_home : bool
        True if the matrix is a ToHome matrix

    Returns
    -------
    mx_df : pd.DataFrame
        grown stn2stn demand matrix
    reporting_df : pd.DataFrame
        stn2stn level grouped demand or reporting purposes
    """
    # melt Matrix
    mx_df = pd.melt(
        mx_df,
        id_vars=[
            "from_model_zone_id",
            "to_model_zone_id",
            "from_stn_zone_id",
            "O_TLC",
            "to_stn_zone_id",
            "D_TLC",
            "userclass",
            "Purpose",
        ],
        value_vars=["F", "R", "S"],
    )
    # rename column
    mx_df = mx_df.rename(
        columns={
            "value": "T_Demand",
            "variable": "TicketType",
            "O_TLC": "ZoneCodeFrom",
            "D_TLC": "ZoneCodeTo",
            "Purpose": "purpose",
        }
    )
    # merge new factors file to matrix
    if to_home:
        mx_df = mx_df.merge(
            edge_factors,
            how="left",
            left_on=["ZoneCodeFrom", "ZoneCodeTo", "TicketType", "purpose"],
            right_on=["ZoneCodeTo", "ZoneCodeFrom", "TicketType", "purpose"],
        )

        mx_df = mx_df.rename(
            columns={
                "ZoneCodeFrom_x": "ZoneCodeFrom",
                "ZoneCodeTo_x": "ZoneCodeTo",
            }
        )

    else:
        mx_df = mx_df.merge(
            edge_factors,
            how="left",
            on=["ZoneCodeFrom", "ZoneCodeTo", "TicketType", "purpose"],
        )
    # Records with nan means no factor was found hence no growth therefore fill nan with 1
    mx_df.loc[:, "Demand_rate"] = mx_df["Demand_rate"].fillna(1)
    # fill nan flag with zero as it doesn't exist in the input EDGE factors
    mx_df.loc[:, "Flag"] = mx_df["Flag"].fillna(0)
    # apply growth
    mx_df.loc[:, "N_Demand"] = mx_df["T_Demand"] * mx_df["Demand_rate"]
    # create reporting dataframe by grouping the demand to stn2stn level
    reporting_df = (
        mx_df.groupby(
            ["from_stn_zone_id", "to_stn_zone_id", "userclass", "purpose", "TicketType"]
        )[["T_Demand", "N_Demand"]]
        .sum()
        .reset_index()
    )
    # keep needed columns
    mx_df = mx_df[
        [
            "from_model_zone_id",
            "to_model_zone_id",
            "from_stn_zone_id",
            "ZoneCodeFrom",
            "to_stn_zone_id",
            "ZoneCodeTo",
            "userclass",
            "purpose",
            "TicketType",
            "T_Demand",
            "N_Demand",
        ]
    ]

    return mx_df, reporting_df


def apply_edge_growth_method2(mx_df: pd.DataFrame, edge_factors: pd.DataFrame) -> pd.DataFrame:
    """Grow demand by factoring it by EDGE factors using method 2.

    using method 2 where an average factor of the two directions is applied.

    Parameters
    ----------
    mx_df : pd.DataFrame
        prepared stn2stn matrix by flow, ticket type and purpose
    edge_factors : pd.DataFrame
        EDGE growth factors by flow, ticket type and purpose

    Returns
    -------
    mx_df : pd.DataFrame
        grown stn2stn demand matrix
    reporting_df : pd.DataFrame
        stn2stn level grouped demand or reporting purposes
    """
    # melt Matrix
    mx_df = pd.melt(
        mx_df,
        id_vars=[
            "from_model_zone_id",
            "to_model_zone_id",
            "from_stn_zone_id",
            "O_TLC",
            "to_stn_zone_id",
            "D_TLC",
            "userclass",
            "Purpose",
        ],
        value_vars=["F", "R", "S"],
    )
    # rename column
    mx_df = mx_df.rename(
        columns={
            "value": "T_Demand",
            "variable": "TicketType",
            "O_TLC": "ZoneCodeFrom",
            "D_TLC": "ZoneCodeTo",
            "Purpose": "purpose",
        }
    )
    # merge new factors file to matrix on first direction O>D
    mx_df = mx_df.merge(
        edge_factors, how="left", on=["ZoneCodeFrom", "ZoneCodeTo", "TicketType", "purpose"]
    )
    # rename growth column to indicate first merge
    # rename column
    mx_df = mx_df.rename(columns={"Demand_rate": "1st_Dir_Growth"})
    # merge new factors file to matrix on second direction O>D
    mx_df = mx_df.merge(
        edge_factors,
        how="left",
        left_on=["ZoneCodeFrom", "ZoneCodeTo", "TicketType", "purpose"],
        right_on=["ZoneCodeTo", "ZoneCodeFrom", "TicketType", "purpose"],
    )
    # rename growth column to indicate second merge
    # rename column
    mx_df = mx_df.rename(
        columns={
            "ZoneCodeFrom_x": "ZoneCodeFrom",
            "ZoneCodeTo_x": "ZoneCodeTo",
            "Demand_rate": "2nd_Dir_Growth",
        }
    )
    # get average growth for both directions
    mx_df.loc[:, "Demand_rate"] = mx_df[["1st_Dir_Growth", "2nd_Dir_Growth"]].mean(axis=1)
    # Records with nan means no factor was found hence no growth therefore fill nan with 1
    mx_df.loc[:, "Demand_rate"] = mx_df[["Demand_rate"]].fillna(1)
    # fill nan flag with zero as it doesn't exist in the input EDGE factors
    mx_df.loc[:, "Flag_x"] = mx_df[["Flag_x"]].fillna(0)
    mx_df.loc[:, "Flag_y"] = mx_df[["Flag_y"]].fillna(0)
    mx_df["Flag"] = 0
    mx_df.loc[(mx_df["Flag_x"] == 1) & (mx_df["Flag_y"] == 1), "Flag"] = 1
    # apply growth
    mx_df.loc[:, "N_Demand"] = mx_df["T_Demand"] * mx_df["Demand_rate"]
    # create reporting dataframe by grouping the demand to stn2stn level
    reporting_df = (
        mx_df.groupby(
            ["from_stn_zone_id", "to_stn_zone_id", "userclass", "purpose", "TicketType"]
        )[["T_Demand", "N_Demand"]]
        .sum()
        .reset_index()
    )
    # keep needed columns
    mx_df = mx_df[
        [
            "from_model_zone_id",
            "to_model_zone_id",
            "from_stn_zone_id",
            "ZoneCodeFrom",
            "to_stn_zone_id",
            "ZoneCodeTo",
            "userclass",
            "purpose",
            "TicketType",
            "T_Demand",
            "N_Demand",
        ]
    ]

    return mx_df, reporting_df


def prepare_logging_info(
    other_tickets_df: pd.DataFrame, no_factors_df: pd.DataFrame, demand_total: float
) -> tuple[pd.DataFrame, pd.DataFrame, float, float, float, float]:
    """Calculate logging stats and prepare to write to logfile.


    Function calculates logging stats of the proportion of the demand each category
    has and the proportion of that demand that is internal
    the function also prepare the dataframe in a format ready to print to the logfile

    Parameters
    ----------
    other_tickets_df : pd.DataFrame
        dataframe with movements that has used factors for other ticket types
    no_factors_df : pd.DataFrame
        dataframe with movements that has no factor at all
    demand_total : float
        sum of all input demand

    Returns
    -------
    other_tickets_df : pd.DataFrame
        dataframe with movements that has used factors for other ticket types
    no_factors_df : pd.DataFrame
        dataframe with movements that has no factor at all
    no_factor_demand_prop: float
        proportion of total demand with no factors proportion
    tickets_demand_prop: float
        proportion of total demand where factors for other ticket types were used
    tickets_internal_prop: float
        internal demand proportion out of tickets_demand_prop
    no_factor_demand_prop: float
        internal demand proportion out of no_factor_demand_prop
    """
    # log warning info
    # get demand total by movement/ticket
    other_tickets_df = (
        other_tickets_df.groupby(
            [
                "ZoneCodeFrom",
                "ZoneCodeTo",
                "purpose",
                "Missing_TicketType",
                "Available_TicketType",
                "Internal",
            ]
        )["T_Demand"]
        .sum()
        .reset_index()
    )
    # get demand totals for movements where different ticket type was used
    demand_total_ticket = other_tickets_df["T_Demand"].sum()
    # demand total for internals
    demand_total_ticket_internal = other_tickets_df["T_Demand"][
        other_tickets_df["Internal"] == 1
    ].sum()
    # movements with no factor at all
    no_factors_df = (
        no_factors_df.groupby(["ZoneCodeFrom", "ZoneCodeTo", "Internal"])["T_Demand"]
        .sum()
        .reset_index()
    )
    # get demand totals for movements where different ticket type was used
    demand_total_factors = no_factors_df["T_Demand"].sum()
    # demand total for internals
    demand_total_factors_internal = no_factors_df["T_Demand"][
        no_factors_df["Internal"] == 1
    ].sum()
    # check proportion of un-factored demand to total demand and other tickets demand to total demand
    #   as well as internal proportion of that demand
    tickets_internal_prop = round(demand_total_ticket_internal / demand_total_ticket * 100, 3)
    factors_internal_prop = round(
        demand_total_factors_internal / demand_total_factors * 100, 3
    )
    # total proportions
    no_factor_demand_prop = round(demand_total_factors / demand_total * 100, 3)
    tickets_demand_prop = round(demand_total_ticket / demand_total * 100, 3)
    # regroup dataframes for logging
    other_tickets_df = (
        other_tickets_df.groupby(
            [
                "ZoneCodeFrom",
                "ZoneCodeTo",
                "purpose",
                "Missing_TicketType",
                "Available_TicketType",
            ]
        )["T_Demand"]
        .sum()
        .reset_index()
    )
    no_factors_df = (
        no_factors_df.groupby(["ZoneCodeFrom", "ZoneCodeTo"])["T_Demand"].sum().reset_index()
    )

    return (
        other_tickets_df,
        no_factors_df,
        no_factor_demand_prop,
        tickets_demand_prop,
        tickets_internal_prop,
        factors_internal_prop,
    )


def sum_periods_demand(
    am_df: pd.DataFrame, ip_df: pd.DataFrame, pm_df: pd.DataFrame, op_df: pd.DataFrame
) -> pd.DataFrame:
    """Sum Periods demand to 24Hr demand.

    Parameters
    ----------
    am_df : pd.DataFrame
        demand matrix for the AM period
    ip_df : pd.DataFrame
        demand matrix for the IP period
    pm_df : pd.DataFrame
        demand matrix for the PM period
    op_df : pd.DataFrame
        demand matrix for the OP period

    Returns
    -------
    comb_df : pd.DataFrame
        24Hr demand matrix
    """
    comb_df = am_df.merge(ip_df, how="outer", on=["from_model_zone_id", "to_model_zone_id"])
    comb_df = comb_df.merge(pm_df, how="outer", on=["from_model_zone_id", "to_model_zone_id"])
    comb_df = comb_df.merge(op_df, how="outer", on=["from_model_zone_id", "to_model_zone_id"])
    # fill nans with zeros
    comb_df = comb_df.fillna(0)
    # sum 24Hr demand
    comb_df.loc[:, "Demand"] = (
        comb_df["AM_Demand"]
        + comb_df["IP_Demand"]
        + comb_df["PM_Demand"]
        + comb_df["OP_Demand"]
    )
    # keep needed columns
    comb_df = comb_df[["from_model_zone_id", "to_model_zone_id", "Demand"]]

    return comb_df


def average_two_matrices(
    mx1_df: pd.DataFrame, mx2_df: pd.DataFrame, zones: int = 1300
) -> pd.DataFrame:
    """Calculate the average of two input matrices.

    Parameters
    ----------
    mx1_df : pd.DataFrame
        first matrix
    mx2_df : pd.DataFrame
        second matrix
    zones: int
        number of model zones, default = 1300

    Returns
    -------
    avg_mx : pd.DataFrame
        averaged matrix
    """
    # create empty dataframe
    avg_mx = pd.DataFrame(
        list(itertools.product(range(1, zones + 1), range(1, zones + 1))),
        columns=["from_model_zone_id", "to_model_zone_id"],
    )
    # get first matrix
    avg_mx = avg_mx.merge(
        mx1_df, how="outer", on=["from_model_zone_id", "to_model_zone_id"]
    ).fillna(0)
    # get second matrix
    avg_mx = avg_mx.merge(
        mx2_df, how="outer", on=["from_model_zone_id", "to_model_zone_id"]
    ).fillna(0)
    # sum demand
    avg_mx.loc[:, "Demand"] = (avg_mx["Demand_x"] + avg_mx["Demand_y"]) / 2
    # keep needed columns
    avg_mx = avg_mx[["from_model_zone_id", "to_model_zone_id", "Demand"]].fillna(0)
    return avg_mx


def expand_matrix(mx_df: pd.DataFrame, zones: int = 1300) -> pd.DataFrame:
    """Expand matrix to all possible movements (zones x zones).

    Parameters
    ----------
    mx_df : pd.DataFrame
        matrix
    zones: int
        number of model zones, default = 1300

    Returns
    -------
    expanded_mx : pd.DataFrame
        expanded matrix
    """
    # create empty dataframe
    expanded_mx = pd.DataFrame(
        list(itertools.product(range(1, zones + 1), range(1, zones + 1))),
        columns=["from_model_zone_id", "to_model_zone_id"],
    )
    # get first matrix
    expanded_mx = expanded_mx.merge(
        mx_df, how="outer", on=["from_model_zone_id", "to_model_zone_id"]
    ).fillna(0)
    return expanded_mx


def fromto_2_from_by_averaging(
    matrices_dict: dict, norms_segments: list, all_segments: list
) -> dict:
    """Get the FromHome demand by averaging FromHome and ToHome.


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
            matrices[segment] = average_two_matrices(
                matrices_dict[segment], transpose_matrix(matrices_dict[segment + "_T"])
            )
        else:
            # Expand the matrix and add to the matrices dict
            matrices[segment] = expand_matrix(matrices_dict[segment])

    return matrices


def fromto_2_from_by_from(matrices_dict: dict, norms_segments: list) -> dict:
    """Get the FromHome demand by using the FromHome only.


    Function keeps the From home only when moving back to NoRMS segments for the
    pd.DataFrame From/To

    Parameters
    ----------
    matrices_dict : dictionary
        24Hr demand matrices dictionary
    norms_segments : list
        list of NoRMS demand segments

    Returns
    -------
    matrices : dictionary
        dictionary of matrices
    """
    # empty dictionary
    matrices = {}

    # loop over all norms segments
    for segment in norms_segments:
        # expand the FromHome matrix and add to the matrices dict
        matrices[segment] = expand_matrix(matrices_dict[segment])

    return matrices


def transpose_matrix(mx_df: pd.DataFrame) -> pd.DataFrame:
    """Transpose a matrix O<>D/P<>A.

    Parameters
    ----------
    mx : pd.DataFrame
        input matrix to transpose

    Returns
    -------
    mx : pd.DataFrame
        transposed matrix
    """
    # transpose to-home PA to OD by renaming from <> to model zone id
    mx_df = mx_df.rename(
        columns={
            "from_model_zone_id": "to_model_zone_id",
            "to_model_zone_id": "from_model_zone_id",
        }
    )

    return mx_df


def convert_csv_2_mat(
    norms_segments: list,
    cube_exe: pathlib.Path,
    forecast_year: int,
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
    forecast_year : int
        forecast year
    output_folder : Path
        path to folder where CSV matrices are saved. this is where the .MAT
        will also be saved to

    """
    # empty dictionary
    mats_dict = {}
    # create a dictionary of matrices and their paths
    for segment in norms_segments:
        mats_dict[segment] = pathlib.Path(output_folder, f"{forecast_year}_24Hr_{segment}.csv")

    # call CUBE convertor class
    c_m = CUBEMatConverter(cube_exe)
    c_m.csv_to_mat(
        1300, mats_dict, pathlib.Path(output_folder, f"PT_24hr_Demand_{forecast_year}.MAT"), 1
    )


def reintsate_unassigned_demand(
    unassigned_demand_mx: pd.DataFrame,
    grown_mx: pd.DataFrame,
) -> pd.DataFrame:
    """Reinstate unassigned demand to the grown matrix from the base matrix.

    Parameters
    ----------
    unassigned_demand_mx : pd.DataFrame
        input base matrix movements that are not assigned
    grown_mx : pd.DataFrame
        forecasted matrix

    Returns
    -------
    filled_mx : pd.DataFrame
        grown matrix with reinstated unassigned demand
    """
    # join base matrix to grown matrix
    filled_mx = unassigned_demand_mx.merge(
        grown_mx, how="outer", on=["from_model_zone_id", "to_model_zone_id"], indicator=True
    )
    # update Demand for un-grown movements
    filled_mx.loc[filled_mx["_merge"] == "left_only", "N_Demand"] = filled_mx["Demand"]
    # keep needed columns
    filled_mx = filled_mx[["from_model_zone_id", "to_model_zone_id", "N_Demand"]]

    return filled_mx


def run_edge_growth(params: forecast_cnfg.EDGEParameters) -> None:
    """Run Growth Process."""
    LOG.info("#" * 80)
    LOG.info("Started Process @ %s", timing.get_datetime())
    LOG.info("#" * 80)

    # Process Fixed objects
    periods = ["AM", "IP", "PM", "OP"]

    # ## READ INPUT FILES ## #
    # Custom input files
    demand_segments = file_ops.read_df(params.demand_segments)
    ticket_type_splits = file_ops.read_df(params.ticket_type_splits_path)
    flow_cats = file_ops.read_df(params.flow_cat_path)
    norms_to_edge_stns = file_ops.read_df(params.norms_to_edge_stns_path)

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

    # lop over forecast years
    for forecast_year in params.forecast_years:
        LOG.info("**** Applying growth for %s @ %s", forecast_year, timing.get_datetime())
        # EDGE files
        edge_flows_file = file_ops.read_df(params.edge_flows_path)
        edge_growth_factors = file_ops.read_df(
            params.edge_growth_dir / params.forecast_years[forecast_year]
        )
        # create list of used stations
        used_stations = norms_to_edge_stns["STATIONCODE"].to_list()
        # filter factors file to keep only used stations
        edge_growth_factors = edge_growth_factors.loc[
            (edge_growth_factors["ZoneCodeFrom"].isin(used_stations))
            & (edge_growth_factors["ZoneCodeTo"].isin(used_stations))
        ]
        # Add Flag = 1 for all input factors in EDGE
        #    i.e. Flag = 1 if the factor comes directly from EDGE
        edge_growth_factors.loc[:, "Flag"] = 1

        # factored matrices dictionary
        factored_matrices = {}
        factored_24hr_matrices = {}
        # empty stn2stn demand reporting dataframe
        stn2stn_reporting_df = pd.DataFrame()

        # empty DFs for recording missing factors
        other_tickets_df = pd.DataFrame()
        no_factors_df = pd.DataFrame()
        # empty dataframe for growth summary
        growth_summary = pd.DataFrame(
            {
                "Time_Period": [],
                "Demand_Segment": [],
                "Base_Demand": [],
                f"{forecast_year}_Demand": [],
            }
        )
        # set demand total to 0
        demand_total = 0

        # loop over periods
        for period in tqdm(periods, desc="Time Periods Loop ", unit=" Period", colour="cyan"):
            LOG.info(
                "-- Processing Time Period %s @ %s",
                period,
                timing.get_datetime(),
            )
            # read distance matrix
            dist_mx = file_ops.read_df(
                params.matrices_to_grow_dir / f"{period}_stn2stn_costs.csv"
            )
            # read iRSj props
            irsj_props = pd.read_hdf(
                params.matrices_to_grow_dir / f"{period}_iRSj_probabilities.h5", key="iRSj"
            )
            # period dictionary
            factored_matrices[period] = {}
            LOG.debug(
                f"{'Time_Period':>12}{'Demand_Segment':>15}"
                f"{'Base_Demand':>12}{f'{forecast_year}_Demand':>12}"
            )
            # loop over demand segments
            for i, row in tqdm(
                demand_segments.iterrows(),
                desc="    Demand Segments Loop ",
                unit=" Segment",
                colour="cyan",
                total=len(demand_segments),
            ):
                # get current segment's details
                segment = row["Segment"]
                to_home = row["ToHome"]
                growth_method = row["Growth_Method"]
                userclass = row["Userclass"]
                purpose = row["Purpose"]

                # filter probabilities dataframe to the current userclass
                irsj_props_segment = irsj_props.loc[
                    irsj_props["userclass"] == userclass
                ].reset_index(drop=True)
                # demand matrices
                demand_mx = file_ops.read_df(
                    params.matrices_to_grow_dir / f"{period}_{segment}.csv"
                )
                # sum total demand
                tot_input_demand = round(demand_mx["Demand"].sum())
                # if demand matrix is zero, move on
                if tot_input_demand == 0:
                    demand_mx = demand_mx.rename(columns={"Demand": f"{period}_Demand"})
                    factored_matrices[period][segment] = demand_mx
                    LOG.debug(f"{period:>12}{segment:>15}" f"{0:>12}{0:>12}")
                    continue
                # sum total demand
                demand_total = demand_total + tot_input_demand
                # add UCs to demand based on demand segment
                demand_mx.loc[:, "userclass"] = userclass
                # keep needed columns
                demand_mx = demand_mx[
                    ["from_model_zone_id", "to_model_zone_id", "userclass", "Demand"]
                ]
                # keep non-zero demand records
                demand_mx = demand_mx.loc[demand_mx["Demand"] > 0].reset_index(drop=True)
                # prepare demand matrix
                demand_mx, unassigned_demand_mx = prepare_stn2stn_matrix(
                    demand_mx, irsj_props_segment, dist_mx, norms_to_edge_stns, to_home
                )
                # assign EDGE flows
                demand_mx = assign_edge_flow(edge_flows_file, flow_cats, demand_mx)
                # add TAG flows
                demand_mx = add_distance_band_tag_flow(demand_mx)
                # add purposes to matrix
                demand_mx.loc[:, "Purpose"] = purpose
                # add ticket splits props
                demand_mx = demand_mx.merge(
                    ticket_type_splits, how="left", on=["TAG_Flow", "Purpose"]
                )
                # apply Ticket Splits
                demand_mx = apply_ticket_splits(demand_mx)
                # Get factors for missing movements if any
                (
                    edge_growth_factors,
                    other_tickets_df,
                    no_factors_df,
                ) = create_factors_for_missing_moira_movements(
                    to_home, demand_mx, edge_growth_factors, other_tickets_df, no_factors_df
                )
                # apply factoring based on demand segment
                if growth_method == 1:
                    demand_mx, stn2stn_rep_df = apply_edge_growth_method1(
                        demand_mx, edge_growth_factors, to_home
                    )
                else:
                    demand_mx, stn2stn_rep_df = apply_edge_growth_method2(
                        demand_mx, edge_growth_factors
                    )

                # add id columns to reporting df
                stn2stn_rep_df["Period"] = period
                stn2stn_rep_df["Segment"] = segment
                # append to main reporting dataframe
                stn2stn_reporting_df = pd.concat(
                    [stn2stn_reporting_df, stn2stn_rep_df], axis=0
                )
                # move back to zone2zone matrix
                demand_mx = (
                    demand_mx.groupby(["from_model_zone_id", "to_model_zone_id"])[
                        ["T_Demand", "N_Demand"]
                    ]
                    .sum()
                    .reset_index()
                )
                tot_output_demand = round(demand_mx["N_Demand"].sum())
                # create logging line
                LOG.debug(
                    f"{period:>12}{segment:>15}"
                    f"{tot_input_demand:>12}{tot_output_demand:>12}"
                )

                # empty dataframe for growth summary
                temp_growth_summary = pd.DataFrame(
                    {
                        "Time_Period": [period],
                        "Demand_Segment": [segment],
                        "Base_Demand": [tot_input_demand],
                        f"{forecast_year}_Demand": [tot_output_demand],
                    }
                )
                # add growth stats to growth summary df
                growth_summary = pd.concat([growth_summary, temp_growth_summary], axis=0)

                # reinstate unassigned demand if needed
                if len(unassigned_demand_mx > 0):
                    demand_mx = reintsate_unassigned_demand(
                        unassigned_demand_mx,
                        demand_mx,
                    )
                # rename demand column
                demand_mx = demand_mx.rename(columns={"N_Demand": f"{period}_Demand"})
                # store in grown matrices dictionary
                factored_matrices[period][segment] = demand_mx

        # get logging stats
        (
            other_tickets_df,
            no_factors_df,
            no_factor_demand_prop,
            tickets_demand_prop,
            tickets_internal_prop,
            factors_internal_prop,
        ) = prepare_logging_info(other_tickets_df, no_factors_df, demand_total)

        # write filled factors file
        file_ops.write_df(
            edge_growth_factors,
            params.export_path / f"Filled_Factors_{forecast_year}.csv",
            index=False,
        )
        # write growth summary dataframe
        file_ops.write_df(
            growth_summary,
            params.export_path / f"Growth_Summary_{forecast_year}.csv",
            index=False,
        )
        # write stn2stn reporting dataframe
        file_ops.write_df(
            stn2stn_reporting_df,
            params.export_path / f"stn2stn_growth_report_{forecast_year}.csv",
            index=False,
        )

        # if the proportion of the demand that has no factor at all in EDGE exceeds 1%
        #        then report these movements and quit the program
        #        user MUST look into these movements and check why these have no factor
        #        and act accordingly
        if no_factor_demand_prop > 1:
            LOG.warning(
                f"Demand with no factors = {no_factor_demand_prop}% "
                "exceeding the 1% threshold of the total demand hence the process terminated"
            )
            LOG.warning("Table Below lists all movements with no factors:")
            LOG.warning("%s", no_factors_df.to_string(index=False))
            LOG.info("Process was interrupted @ %s", timing.get_datetime())
            print("Process was interrupted - Check the logfile for more details!")
            # quit
            raise ValueError(
                "Process interrupted due to high proportion of demand"
                " having no Growth factor - see Logfile for more details!"
            )

        LOG.info(
            "Records below have missing factors for -Missing_TicketType- "
            "and therefore growth factors for"
        )
        LOG.info("Tickets from Available_TicketType- have been used")
        LOG.info(
            "Total demand proportion for these movements = %s %% "
            "of which %s %% is Internal",
            tickets_demand_prop,
            tickets_internal_prop,
        )
        LOG.info("-----------------------------------")
        LOG.info("%s", other_tickets_df.to_string(index=False))

        # log info
        path = params.export_path / f"no_factors_log_{forecast_year}.csv"
        LOG.warning(
            "Some records have no factors at all for some movements. "
            "Therefore, no growth has been applied. See %s for a full summary "
            "of these movements." % path
        )
        no_factors_df.to_csv(path, index=False)

        # write out matrices
        for i, row in demand_segments.iterrows():
            # get current segment's details
            segment = row["Segment"]
            # get demand for each period
            am_mx = factored_matrices["AM"][segment]
            ip_mx = factored_matrices["IP"][segment]
            pm_mx = factored_matrices["PM"][segment]
            op_mx = factored_matrices["OP"][segment]
            # get 24Hr demand matrix
            demand_mx = sum_periods_demand(am_mx, ip_mx, pm_mx, op_mx)
            # add to 24Hr matrices dict
            factored_24hr_matrices[segment] = demand_mx

        # Combine matrices into NoRMS segments
        norms_matrices1 = fromto_2_from_by_averaging(
            factored_24hr_matrices, norms_segments, all_segments
        )
        # norms_matrices2 = fromto_2_from_by_from(factored_24Hr_matrices, norms_segments)
        # plot matrices
        for segment in norms_segments:
            # write out demand matrix
            file_ops.write_df(
                norms_matrices1[segment],
                params.export_path / f"{forecast_year}_24Hr_{segment}.csv",
                index=False,
            )
            # file_ops.write_df(
            #    norms_matrices2[segment],
            #    params.export_path / f"{forecast_year}_24Hr_{segment}.csv",
            #    index=False,
            # )
        # convert to NoRMS format .MAT
        convert_csv_2_mat(norms_segments, params.cube_exe, forecast_year, params.export_path)
    print("Process finished successfully!")
    LOG.info("Process finished successfully @ %s", timing.get_datetime())
