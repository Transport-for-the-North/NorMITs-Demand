# -*- coding: utf-8 -*-
"""
Module to process ticket type splitting factors.

Splits by tag flow and purpose exist, which need to be converted to splits by station pair.
"""
# Built-Ins
from pathlib import Path
import pickle
# Third Party
import pandas as pd
import numpy as np

# Local Imports
# pylint: disable=import-error,wrong-import-position
# Local imports here
import utils

# pylint: enable=import-error,wrong-import-position

# # # CONSTANTS # # #

# # # CLASSES # # #

# # # FUNCTIONS # # #
def append_dist(row):
    """
    Convert TAG_NonDist to TAG_Flow.

    This function is designed to applied to a dataframe.
    Parameters
    ----------

    row: The row this is applied to.
    Returns
    -------

    An altered row.
    """
    if row["Distance"] < 25:
        return row["TAG_NonDist"].lower() + " <25 miles"
    elif row["Distance"] < 100:
        return row["TAG_NonDist"].lower() + " 25 to 100 miles"
    else:
        return row["TAG_NonDist"].lower() + " 100 + miles"


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
    edge_flows.loc[:, "FlowCatName"] = edge_flows[
        "FlowCatName"
    ].str.lower()
    flows_lookup.loc[:, "FlowCatName"] = flows_lookup[
        "FlowCatName"
    ].str.lower()
    flows_lookup.loc[:, "TAG_NonDist"] = flows_lookup[
        "TAG_NonDist"
    ].str.lower()
    ticket_split_proportions.loc[
        :, "TAG_Flow"
    ] = ticket_split_proportions["TAG_Flow"].str.lower()
    # add flows TAG category
    edge_flows = edge_flows.merge(
        flows_lookup, how="left", on=["FlowCatName"]
    )
    edge_flows = utils.merge_to_stations(
        stations_lookup, edge_flows, "FromCaseZoneID", "ToCaseZoneID"
    )
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
        distance_mx,
        how="left",
        on=["from_stn_zone_id", "to_stn_zone_id"],
    )
    # rename
    edge_flows = edge_flows.rename(
        columns={"tran_distance": "Distance"}
    )
    # fill na
    edge_flows = edge_flows.fillna(0)
    # allocate distance bands
    edge_flows["TAG_Flow"] = edge_flows[
        edge_flows["TAG_NonDist"].str.startswith("outside south east")
    ].apply(append_dist, axis=1)
    edge_flows["TAG_Flow"] = edge_flows["TAG_Flow"].fillna(
        edge_flows["TAG_NonDist"]
    )
    # edge_flows.rename(columns={"TAG_NonDist": "TAG_Flow"})
    # keep needed columns
    edge_flows = edge_flows[
        [
            "from_stn_zone_id",
            "to_stn_zone_id",
            "FromCaseZoneID",
            "ToCaseZoneID",
            "TAG_Flow",
        ]
    ]
    # merge ticket split factors
    edge_flows = edge_flows.merge(
        ticket_split_proportions, how="left", on=["TAG_Flow"]
    )
    # get list of purposes
    purposes = edge_flows["Purpose"].drop_duplicates().to_list()
    # create matrices dictionary
    splitting_matrices = {i: {} for i in purposes}
    # create numpy splitting matrices
    for purpose in purposes:
        for ticketype in ["F", "R", "S"]:
            # get current purpose
            mx_df = edge_flows.loc[
                edge_flows["Purpose"] == purpose
            ].reset_index(drop=True)
            # keep needed columns
            mx_df = mx_df[
                ["from_stn_zone_id", "to_stn_zone_id", ticketype]
            ]
            # rename
            mx_df = mx_df.rename(columns={ticketype: "Demand"})
            # expand matrix
            mx_df = utils.expand_matrix(
                mx_df, zones=len(stations_lookup), stations=True
            )
            # convert to numpy and add to matrices dictionary
            splitting_matrices[purpose][
                ticketype
            ] = utils.long_mx_2_wide_mx(mx_df)

    return splitting_matrices

def splits_loop(
    edge_flows: pd.DataFrame,
    stations_lookup: pd.DataFrame,
    flows_lookup: pd.DataFrame,
    ticket_split_proportions: pd.DataFrame,
    dist_dir: Path
):
    """
    Generates splitting ticket splitting factors for a given set of inputs.
    Automatically dumps the dict to a pickle file in the same dir as the distance
    matrices used.
    """
    split_dict = {}
    for tp in ['AM','IP','PM','OP']:
        dist_mx = pd.read_csv(dist_dir / f"{tp}_stn2stn_costs.csv", usecols=[0, 1, 4])
        splitting_matrices = (
            produce_ticketype_splitting_matrices(
                edge_flows,
                stations_lookup,
                dist_mx,
                flows_lookup,
                ticket_split_proportions,
            )
        )
        split_dict[tp] = splitting_matrices
    with open(dist_dir / "splitting_matrices.pkl", 'wb') as file:
        pickle.dump(split_dict, file)
    return split_dict
