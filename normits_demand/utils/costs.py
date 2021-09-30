# -*- coding: utf-8 -*-
"""
Created on: 29/09/2021
Updated on:

Original author: Ben Taylor
Last update made by: Ben Taylor
Other updates made by: Chris Storey

File purpose:

"""
# Built-Ins
import os

# Third Party
import pandas as pd

# Local Imports
import normits_demand as nd


def get_costs(import_path: nd.PathLike,
              segment_params: nd.SegmentParams,
              iz_infill: float = 0.5,
              replace_nhb_with_hb: bool  = False,
              ):
    # units takes different parameters
    # TODO(BT): Stop calling this function. Replace with NorMITs Supply
    # TODO: Needs a config guide for the costs somewhere
    """
    This function imports distances or costs from a given path.

    Parameters
    ----------
    import_path:
        Model folder to look in for distances/costs. Should be in call or global.

    segment_params:
        Calibration parameters dictionary'

    iz_infill:
        whether to add a value half the minimum
        interzonal value to the intrazonal cells. Currently needed for distance
        but not cost.

    Returns
    -------
    dat:
        DataFrame containing required cost or distance values.
    """
    # TODO: Adapt model input costs to take time periods
    # TODO: The name cost_cols is misleading
    dat = pd.read_csv(import_path)
    cols = list(dat)

    # Get purpose and direction from calib_params
    ca = None
    purpose = None
    time_period = None

    for index, param in segment_params.items():
        # Need a purpose, if a ca is not picked up returns none
        if index == 'p':
            purpose = param
        if index == 'ca':
            if param == 1:
                ca = 'nca'
            elif param == 2:
                ca = 'ca'
        if index == 'tp':
            time_period = param

    # Purpose to string
    commute = [1]
    business = [2, 12]
    other = [3, 4, 5, 6, 7, 8, 13, 14, 15, 16, 18]
    if purpose in commute:
        str_purpose = 'commute'
    elif purpose in business:
        str_purpose = 'business'
    elif purpose in other:
        str_purpose = 'other'
    else:
        raise ValueError("Cannot convert purpose to string. " +
                         "Got %s." % str(purpose))

    # Filter down on purpose
    cost_cols = [x for x in cols if str_purpose in x]
    # Handle if we have numeric purpose costs, hope so, they're better!
    if len(cost_cols) == 0:
        if replace_nhb_with_hb:
            if purpose >= 10:
                purpose -= 10
        cost_cols = [x for x in cols if ('p' + str(purpose)) in x]

    # Filter down on car availability
    if ca is not None:
        # Have to be fussy as ca is in nca...
        if ca == 'ca':
            cost_cols = [x for x in cost_cols if 'nca' not in x]
        elif ca == 'nca':
            cost_cols = [x for x in cost_cols if 'nca' in x]

    if time_period is not None:
        cost_cols = [x for x in cost_cols if str(time_period) in x]

    target_cols = ['p_zone', 'a_zone']
    for col in cost_cols:
        target_cols.append(col)
    cost_return_name = cost_cols[0]

    dat = dat.reindex(target_cols, axis=1)
    dat = dat.rename(columns={cost_cols[0]: 'cost'})

    # Redefine cols
    cols = list(dat)

    if iz_infill is not None:
        dat = dat.copy()
        min_inter_dat = dat[dat[cols[2]] > 0]
        # Derive minimum intrazonal
        min_inter_dat = min_inter_dat.groupby(cols[0]).min().reset_index().drop(cols[1], axis=1)
        intra_dat = min_inter_dat.copy()
        intra_dat[cols[2]] = intra_dat[cols[2]] * iz_infill
        iz = dat[dat[cols[0]] == dat[cols[1]]]
        non_iz = dat[dat[cols[0]] != dat[cols[1]]]
        iz = iz.drop(cols[2], axis=1)
        # Rejoin
        iz = iz.merge(intra_dat, how='inner', on=cols[0])
        dat = pd.concat([iz, non_iz], axis=0, sort=True).reset_index(drop=True)

    return dat, cost_return_name
