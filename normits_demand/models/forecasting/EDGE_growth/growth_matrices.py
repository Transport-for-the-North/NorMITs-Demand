# -*- coding: utf-8 -*-
"""
Module to get growth matrices ready for application to base matrices.

Formats matrices, including converting to numpy arrays, and infills missing growth factors.
"""
# Built-Ins

# Third Party
import pandas as pd
import numpy as np
import caf.toolkit as ctk
# Local Imports
# pylint: disable=import-error,wrong-import-position
# Local imports here
from normits_demand.models.forecasting.edge_growth import utils

# pylint: enable=import-error,wrong-import-position

# # # CONSTANTS # # #

# # # CLASSES # # #

# # # FUNCTIONS # # #


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
    # Init
    purposes = demand_segments_df["Purpose"].unique()
    ticket_types = factors_df["TicketType"].unique()
    factors_df = utils.merge_to_stations(
        stations_lookup, factors_df, "ZoneCodeFrom", "ZoneCodeTo",
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
    growth_matrices = {i: {} for i in purposes}

    # get growth matrices for each purpose/ticket type
    for purpose in purposes:
        for ticket_type in ticket_types:
            mx_df = factors_df[
                ["from_stn_zone_id", "to_stn_zone_id", "Demand"]
            ].loc[
                (factors_df["purpose"] == purpose)
                & (factors_df["TicketType"] == ticket_type)
            ]
            # expand matrix
            index = range(1, len(stations_lookup)+1)
            growth_matrices[purpose][
                ticket_type
            ] = ctk.pandas_utils.long_to_wide_infill(mx_df, mx_df.columns[0], mx_df.columns[1], mx_df.columns[2], index, index, infill=0).values

    return growth_matrices


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
    missing_factors: dict
        dict of missing factors. Gives indices of missing factors.
    """
    # order S: R, F
    filled_growth_matrices = dict.fromkeys(purposes, {})
    for purpose in purposes:
        # get current matrices
        f_mx = growth_matrices[purpose]["F"]
        r_mx = growth_matrices[purpose]["R"]
        s_mx = growth_matrices[purpose]["S"]
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
        filled_growth_matrices[purpose]["F"] = filled_f_mx
        filled_growth_matrices[purpose]["R"] = filled_r_mx
        filled_growth_matrices[purpose]["S"] = filled_s_mx

    return filled_growth_matrices
