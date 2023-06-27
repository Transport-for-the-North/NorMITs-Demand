import numpy as np
import utils
import pandas as pd
import caf.toolkit as ctk


def apply_demand_growth(
    stn2stn_base_mx: np.ndarray,
    splitting_matrices: dict,
    filled_growth_matrices: dict,
    matrix_zones: int,
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
    for ticketype, mx in splitting_matrices.items():
        ticketype_np_matrix = (
            stn2stn_base_mx * mx
        )
        # transpose ToHome demand
        if to_home:
            ticketype_np_matrix = ticketype_np_matrix.transpose()
        # get growth matrix
        if growth_method == 1:
            growth_mx = filled_growth_matrices[ticketype]
        else:
            growth_mx = (
                filled_growth_matrices[ticketype]
                + filled_growth_matrices[ticketype].transpose()
            ) / 2
        # apply growth
        ticketype_np_matrix *=  growth_mx
        # sum grown demand
        stn2stn_forecast_mx += ticketype_np_matrix

    # transpose ToHome demand
    if to_home:
        stn2stn_forecast_mx = stn2stn_forecast_mx.transpose()

    return stn2stn_forecast_mx


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
        if (segment + "_T" in all_segments) and (
            segment[:3].lower() != "NHB".lower()
        ):
            # average the FromHome and the transposition of the toHome
            matrices[segment] = (
                matrices_dict[segment]
                + matrices_dict[segment + "_T"].transpose()
            ) / 2

        else:
            # Keep as it is
            matrices[segment] = matrices_dict[segment]

    return matrices
