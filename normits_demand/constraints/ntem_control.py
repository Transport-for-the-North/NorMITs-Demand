# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 17:36:43 2021

@author: genie
"""
# builtins
from typing import Any
from typing import List
from typing import Dict

# Third party
import numpy as np
import pandas as pd

# Local imports
from normits_demand import constants as consts
from normits_demand.utils import general as du

from normits_demand.validation import checks


# TODO: Allow control_to_ntem() to take flexible col names
def control_to_ntem(control_df: pd.DataFrame,
                    ntem_totals: pd.DataFrame,
                    zone_to_lad: pd.DataFrame,
                    constraint_cols: List[str] = None,
                    base_value_name: str = 'attractions',
                    ntem_value_name: str = 'attractions',
                    base_zone_name: str = 'msoa_zone_id',
                    trip_origin: str = 'hb',
                    group_cols: List[str] = None,
                    constraint_dtypes: Dict[str, Any] = None,
                    verbose=True,
                    ) -> pd.DataFrame:
    """
    Control to a vector of NTEM constraints using single factor.
    Return productions controlled to NTEM.

    Parameters:
    ----------
    control_df:
        DF of productions, flexible segments.

    ntem_totals:
        DF of NTEM totals to control to. Needs to be in LADs.

    zone_to_lad:
        DF of translation between control_df zone system and LAD

    constraint_cols = ['p', 'm']:
        Segments to include in control. Will usually be ['p','m'], or
        ['p','m','ca'] for rail

    base_value_name = 'attractions':
        Name of the value column in control_df - ie. productions or
        attractions. Might also be trips or dt, origin, destination

    ntem_value_name = 'attractions':
        Name of the value column in the NTEM dataset. ie. productions or
        attractions. Might also be trips or dt, origin, destination
    
    base_zone_name = 'msoa_zone_id':
        name of zoning column used in control_df. Same column name should be
        used in zone_to_lad for translation.

    trip_origin = 'hb':
        trip_origin set to aggregate on. Can be 'hb' or 'nhb'

    Returns:
    ----------
    adjusted_control_df:
        control_df with same zoning as input but controlled to NTEM.
    """
    # set defaults
    if constraint_cols is None:
        constraint_cols = ['p', 'm']

    if constraint_dtypes is None:
        constraint_dtypes = {c: str for c in constraint_cols}

    if group_cols is None:
        group_cols = du.list_safe_remove(list(control_df), [base_value_name])

    index_cols = group_cols.copy() + [base_value_name]

    # ## VALIDATE INPUTS ## #
    for col in constraint_cols:
        for df, df_name in zip([control_df, ntem_totals], ['control_df', 'NTEM']):
            if col not in list(df):
                raise ValueError(
                    'Column %s not in %s dataframe' % (str(col), df_name)
                )

    trip_origin = checks.validate_trip_origin(trip_origin)

    # Init
    control_df = control_df.copy()
    zone_to_lad = zone_to_lad.copy()
    ntem_totals = ntem_totals.copy()
    ntem_value_name = ntem_value_name.strip().lower()

    index_df = control_df.reindex(columns=group_cols)
    purposes = du.trip_origin_to_purposes(trip_origin)

    # See if our control_df is a subset
    constraint_subset = dict()
    for col in constraint_cols:
        constraint_subset[col] = control_df[col].unique()

    # TODO: Not necessarily True for EFS
    # Establish all non trip segments
    segments = list()
    for col in list(control_df):
        if col not in [base_zone_name, base_value_name]:
            segments.append(col)

    # ## BEGIN CONTROLLING ## #
    # Print starting value
    before = control_df[base_value_name].sum()
    du.print_w_toggle('Before: ' + str(before), echo=verbose)

    # Build factors
    ntem_k_factors = ntem_totals[ntem_totals['p'].isin(purposes)].copy()

    kf_groups = ['lad_zone_id']
    for col in constraint_cols:
        kf_groups.append(col)
    kf_sums = kf_groups.copy()
    kf_sums.append(ntem_value_name)

    # Sum down to drop non attraction segments
    ntem_k_factors = ntem_k_factors.reindex(kf_sums, axis=1)
    ntem_k_factors = ntem_k_factors.groupby(kf_groups).sum().reset_index()

    # Calculate the total we are aiming for
    ntem_subset = ntem_k_factors.copy()
    for col, vals in constraint_subset.items():
        subset_mask = ntem_subset[col].isin(vals)
        ntem_subset = ntem_subset[subset_mask]

    target = ntem_subset[ntem_value_name].sum()
    du.print_w_toggle('NTEM: ' + str(target), echo=verbose)

    # Assumes vectors are called productions or attractions
    for col in constraint_cols:
        ntem_k_factors.loc[:, col] = ntem_k_factors[col].astype(constraint_dtypes[col])
    ntem_k_factors['lad_zone_id'] = ntem_k_factors[
            'lad_zone_id'].astype(float).astype(int)

    zone_to_lad = zone_to_lad.reindex(['lad_zone_id', base_zone_name], axis=1)

    control_df = control_df.merge(zone_to_lad,
                                  how='left',
                                  on=base_zone_name)

    # No lad match == likely an island - have to drop
    control_df = control_df[~control_df['lad_zone_id'].isna()]

    for col in constraint_cols:
        control_df.loc[:, col] = control_df[col].astype(constraint_dtypes[col])
    control_df['lad_zone_id'] = control_df['lad_zone_id'].astype(float).astype(int)

    # Seed zero infill
    control_df[base_value_name] = control_df[base_value_name].replace(0, 0.001)

    # Build LA adjustment
    # Note tp not in the picture
    af_groups = ['lad_zone_id']
    for col in constraint_cols:
        af_groups.append(col)
    af_sums = af_groups.copy()
    af_sums.append(base_value_name)

    adj_fac = control_df.reindex(af_sums, axis=1).groupby(
            af_groups).sum().reset_index().copy()
    # Just have to do this manually
    adj_fac['lad_zone_id'] = adj_fac['lad_zone_id'].astype(float).astype(int)

    # Merge NTEM values
    adj_fac = adj_fac.merge(ntem_k_factors,
                            how='left',
                            on=af_groups)
    # Get adjustment factors
    adj_fac['adj_fac'] = adj_fac[ntem_value_name]/adj_fac[base_value_name]
    af_only = af_groups.copy()
    af_only.append('adj_fac')
    adj_fac = adj_fac.reindex(af_only, axis=1)
    adj_fac['adj_fac'] = adj_fac['adj_fac'].replace(np.nan, 1)

    for col in constraint_cols:
        adj_fac.loc[:, col] = adj_fac[col].astype(constraint_dtypes[col])

    adjustments = adj_fac['adj_fac']

    # TODO: Report adj factors here
    adj_control_df = control_df.merge(adj_fac,
                                  how='left',
                                  on=kf_groups)

    adj_control_df[base_value_name] *= adj_control_df['adj_fac']

    # Output segmented lad totals
    lad_groups = ['lad_zone_id']
    for col in segments:
        lad_groups.append(col)
    lad_index = lad_groups.copy()
    lad_index.append(base_value_name)

    lad_totals = adj_control_df.reindex(lad_index, axis=1)
    lad_totals = lad_totals.groupby(lad_groups).sum().reset_index()

    # Reindex outputs
    adj_control_df = adj_control_df.drop(['lad_zone_id', 'adj_fac'], axis=1)

    after = adj_control_df[base_value_name].sum()
    du.print_w_toggle('After: ' + str(after), echo=verbose)

    audit = {
        'before': before,
        'target': target,
        'after': after
    }

    # Tidy up the return
    adj_control_df = adj_control_df.reindex(columns=index_cols)
    adj_control_df = adj_control_df.groupby(group_cols).sum().reset_index()

    # If we have dropped zones, we need to add them back in
    if len(adj_control_df) != len(index_df):
        adj_control_df = pd.merge(
            index_df,
            adj_control_df,
            on=group_cols,
            how='left'
        ).fillna(0)

    return adj_control_df, audit, adjustments, lad_totals
