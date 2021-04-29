# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 17:36:43 2021

@author: genie
"""
# builtins
import os

from typing import Any
from typing import List
from typing import Dict
from typing import Callable

# Third party
import numpy as np
import pandas as pd

# Local imports
import normits_demand as nd
from normits_demand import constants as consts
from normits_demand.utils import general as du

from normits_demand.validation import checks

# BACKLOG: Optimise control_to_ntem()
#  labels: optimisation

# TODO: Allow control_to_ntem() to take flexible col names
def new_control_to_ntem(control_df: pd.DataFrame,
                        ntem_totals: pd.DataFrame,
                        zone_to_lad: pd.DataFrame,
                        constraint_cols: List[str] = None,
                        base_value_name: str = 'attractions',
                        ntem_value_name: str = 'attractions',
                        base_zone_name: str = 'msoa_zone_id',
                        trip_origin: str = 'hb',
                        input_group_cols: List[str] = None,
                        constraint_dtypes: Dict[str, Any] = None,
                        replace_drops: bool = False,
                        verbose: bool = True,
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

    group_cols:
        A list of column names in control_df that contain segmentation data,
        and can be used in a group_by call, incase the final dataframe is
        not in the correct shape (due to zone_to_lad splits).
        If left as None, all columns other than base_value_name are assumed
        to be group_cols.

    constraint_dtypes:
        A dictionary of constraint col_names to constraint col dtypes.
        If left as None, defaults to all dtypes being str.
        e.g. {'p': str, 'm': str}.

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

    if not all([c in constraint_dtypes.keys() for c in constraint_cols]):
        raise ValueError(
            "I haven't been given a datatype for all of the constraint_cols "
            "given."
        )

    if input_group_cols is None:
        input_group_cols = du.list_safe_remove(list(control_df), [base_value_name])

    input_index_cols = input_group_cols.copy() + [base_value_name]

    # ## VALIDATE INPUTS ## #
    for col in constraint_cols:
        for df, df_name in zip([control_df, ntem_totals], ['control_df', 'NTEM']):
            if col not in list(df):
                raise ValueError(
                    'Column %s not in %s dataframe' % (str(col), df_name)
                )

    # Make sure value columns exist
    if base_value_name not in list(control_df):
        raise ValueError(
            'Column %s not in control_df dataframe' % str(base_value_name)
        )

    if ntem_value_name not in list(ntem_totals):
        raise ValueError(
            'Column %s not in ntem_totals dataframe' % str(ntem_value_name)
        )

    trip_origin = checks.validate_trip_origin(trip_origin)

    # Init
    control_df = control_df.copy()
    zone_to_lad = zone_to_lad.copy()
    ntem_value_name = ntem_value_name.strip().lower()

    index_df = control_df.reindex(columns=input_group_cols)
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

    # Build Control DF
    group_cols = ['lad_zone_id'] + constraint_cols
    index_cols = group_cols.copy() + [ntem_value_name]

    ntem_control = ntem_totals[ntem_totals['p'].isin(purposes)].copy()
    ntem_control = ntem_control.reindex(columns=index_cols)
    ntem_control = ntem_control.groupby(group_cols).sum().reset_index()

    # Calculate the total we are aiming for
    ntem_subset = ntem_control.copy()
    for col, vals in constraint_subset.items():
        subset_mask = ntem_subset[col].isin(vals)
        ntem_subset = ntem_subset[subset_mask]

    target = ntem_subset[ntem_value_name].sum()
    du.print_w_toggle('NTEM: ' + str(target), echo=verbose)

    # Match the dtypes to the given ones
    for col in constraint_cols:
        ntem_control[col] = ntem_control[col].astype(constraint_dtypes[col])
    ntem_control['lad_zone_id'] = ntem_control['lad_zone_id'].astype(float).astype(int)

    # ## ADD LAD INTO CONTROL ## #
    zone_to_lad = zone_to_lad.reindex(columns=['lad_zone_id', base_zone_name])
    control_df = pd.merge(
        control_df,
        zone_to_lad,
        how='left',
        on=base_zone_name
    )

    # No lad match == likely an island - have to drop
    control_df = control_df[~control_df['lad_zone_id'].isna()]

    # Seed zero infill
    control_df[base_value_name] = control_df[base_value_name].replace(0, 0.001)

    # ## CALCULATE ADJUSTMENT FACTORS ## #
    # Keep just the needed stuff
    group_cols = ['lad_zone_id'] + constraint_cols
    index_cols = group_cols.copy() + [base_value_name]

    adj_factors = control_df.reindex(columns=index_cols)
    adj_factors = adj_factors.groupby(group_cols).sum().reset_index()

    # Match the adj factor zones
    for col in constraint_cols:
        adj_factors[col] = adj_factors[col].astype(constraint_dtypes[col])
    adj_factors['lad_zone_id'] = adj_factors['lad_zone_id'].astype(float).astype(int)

    # Stick the current to the control
    merge_cols = ['lad_zone_id'] + constraint_cols
    adj_factors = pd.merge(
        adj_factors,
        ntem_control,
        how='left',
        on=merge_cols,
    )

    # Get adjustment factors
    adj_factors['adj_fac'] = adj_factors[ntem_value_name]/adj_factors[base_value_name]
    adj_factors['adj_fac'] = adj_factors['adj_fac'].replace(np.nan, 1)

    # TODO(BT): Does this still need returning?
    index_cols = ['lad_zone_id'] + constraint_cols + ['adj_fac']
    adj_factors = adj_factors.reindex(columns=index_cols)
    adjustments = adj_factors['adj_fac']

    # TODO: Report adj factors here
    merge_cols = ['lad_zone_id'] + constraint_cols
    adj_control_df = pd.merge(
        control_df,
        adj_factors,
        how='left',
        on=merge_cols,
    )
    adj_control_df[base_value_name] *= adj_control_df['adj_fac']

    after = adj_control_df[base_value_name].sum()
    du.print_w_toggle('After: ' + str(after), echo=verbose)

    audit = {
        'before': before,
        'target': target,
        'after': after
    }

    # Output segmented lad totals
    lad_totals = adj_control_df.drop(columns=['msoa_zone_id', 'adj_fac'])

    # Tidy up the return
    adj_control_df = adj_control_df.drop(columns=['lad_zone_id', 'adj_fac'])

    if replace_drops:
        # If we have dropped zones, we need to add them back in
        if len(adj_control_df) != len(index_df):
            adj_control_df = pd.merge(
                index_df,
                adj_control_df,
                on=group_cols,
                how='left'
            ).fillna(0)

        # return and starting df aren't the same still, something really bad
        # has happened
        if len(adj_control_df) != len(index_df):
            raise nd.NormitsDemandError(
                "Tried to correct the missing zones after doing the translation, "
                "but something has gone wrong.\nLength of the starting df: %d\n"
                "Length of the ending df: %d"
                % (len(index_df), len(adj_control_df))
            )

    return adj_control_df, audit, adjustments, lad_totals


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
                    replace_drops: bool = False,
                    verbose: bool = True,
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

    group_cols:
        A list of column names in control_df that contain segmentation data,
        and can be used in a group_by call, incase the final dataframe is
        not in the correct shape (due to zone_to_lad splits).
        If left as None, all columns other than base_value_name are assumed
        to be group_cols.

    constraint_dtypes:
        A dictionary of constraint col_names to constraint col dtypes.
        If left as None, defaults to all dtypes being str.
        e.g. {'p': str, 'm': str}.

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

    if not all([c in constraint_dtypes.keys() for c in constraint_cols]):
        raise ValueError(
            "I haven't been given a datatype for all of the constraint_cols "
            "given."
        )

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

    # Make sure value columns exist
    if base_value_name not in list(control_df):
        raise ValueError(
            'Column %s not in control_df dataframe' % str(base_value_name)
        )

    if ntem_value_name not in list(ntem_totals):
        raise ValueError(
            'Column %s not in ntem_totals dataframe' % str(ntem_value_name)
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
    du.print_w_toggle('Before: ' + str(before), verbose=verbose)

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
    du.print_w_toggle('NTEM: ' + str(target), verbose=verbose)

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
    adj_fac['adj_fac'] = adj_fac[ntem_value_name] / adj_fac[base_value_name]
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
    du.print_w_toggle('After: ' + str(after), verbose=verbose)

    audit = {
        'before': before,
        'target': target,
        'after': after
    }

    # Tidy up the return
    adj_control_df = adj_control_df.reindex(columns=index_cols)
    adj_control_df = adj_control_df.groupby(group_cols).sum().reset_index()

    if replace_drops:
        # If we have dropped zones, we need to add them back in
        if len(adj_control_df) != len(index_df):
            adj_control_df = pd.merge(
                index_df,
                adj_control_df,
                on=group_cols,
                how='left'
            ).fillna(0)

        # return and starting df aren't the same still, something really bad
        # has happened
        if len(adj_control_df) != len(index_df):
            raise nd.NormitsDemandError(
                "Tried to correct the missing zones after doing the translation, "
                "but something has gone wrong.\nLength of the starting df: %d\n"
                "Length of the ending df: %d"
                % (len(index_df), len(adj_control_df))
            )

    return adj_control_df, audit, adjustments, lad_totals


def control_vector_to_ntem(vector: pd.DataFrame,
                           vector_type: str,
                           trip_origin: str,
                           ntem_dir: str,
                           lad_lookup_path: str,
                           base_year: str,
                           future_years: List[str] = None,
                           control_base_year: bool = True,
                           control_future_years: bool = False,
                           ntem_control_cols: List[str] = None,
                           ntem_control_dtypes: List[Callable] = None,
                           vector_zone_sys: str = 'msoa',
                           reports_dir: str = None
                           ) -> pd.DataFrame:
    """
    Controls the given vector to NTEM in base and/or future years

    Parameters
    ----------
    vector:
        A dataframe of the vector to control

    vector_type:
        The type of the vector to control, either 'productions' or 'attractions'

    trip_origin:
        THe trip origin of the vector to control. Either 'hb' or 'nhb'

    ntem_dir:
        The path to the directory containing the NTEM control files

    lad_lookup_path:
        The path to the file containing the lookup from vector_zone_sys to
        lad.

    base_year:
        The column in vector relating to the base year values.

    future_years:
        The columns in vector relating to the future year values.

    control_base_year:
        Whether to control the base year production to NTEM or not.

    control_future_years:
        Whether to control the future year vector to NTEM or not. If this is
        False, and control_base_year is True, multiplicative growth is applied
        to the future years to bring it into line with the controlled base
        year.

    ntem_control_cols:
        The name of the columns in vector to control to NTEM at LAD level.

    ntem_control_dtypes:
        THe datatypes of ntem_control_cols. Should be a list of same length
        of ntem_control_cols.

    vector_zone_sys:
        The name of the zoning system being used by vector

    reports_dir:
        Path to a directory to output reports from the control to. If
        growth factors are generated, they will also be output here.

    Returns
    -------
    controlled_vector:
        Vector controlled to ntem in the base and/or future years
    """
    # Set up default args
    if ntem_control_cols is None:
        ntem_control_cols = ['p', 'm']

    if ntem_control_dtypes is None:
        ntem_control_dtypes = [int, int]

    valid_vector_types = ['productions', 'attractions']
    vector_type = vector_type.strip().lower()
    if vector_type not in valid_vector_types:
        raise ValueError(
            "Unexpected vector type given. Got %s\nExpected "
            "one of: %s" % (vector_type, valid_vector_types)
        )

    # Init
    future_years = list() if future_years is None else future_years
    all_years = [base_year] + future_years
    init_index_cols = list(vector)
    init_group_cols = du.list_safe_remove(list(vector), all_years)
    constraint_dtypes = {k: v for k, v in zip(ntem_control_cols, ntem_control_dtypes)}

    # Determine report filenames
    if vector_type == 'productions':
        gf_report_fname = consts.PRODS_MG_FNAME % (vector_zone_sys, trip_origin)
        control_report_fname = consts.PRODS_FNAME % (vector_zone_sys, trip_origin)
    elif vector_type == 'attractions':
        gf_report_fname = consts.ATTRS_MG_FNAME % (vector_zone_sys, trip_origin)
        control_report_fname = consts.ATTRS_FNAME % (vector_zone_sys, trip_origin)
    else:
        raise ValueError(
            "Somehow my vector type is valid, but I don't know how to "
            "deal with it!"
        )

    # Use sorting to avoid merge.
    all_years = [base_year] + future_years
    sort_cols = du.list_safe_remove(list(vector), all_years)
    vector = vector.sort_values(sort_cols)

    # Do we need to grow on top of a controlled base year? (multiplicative)
    grow_over_base = control_base_year and not control_future_years

    # Get growth values over base
    growth_factors = None
    if grow_over_base:
        growth_factors = vector.copy()
        for year in future_years:
            growth_factors[year] /= growth_factors[base_year]
        growth_factors.drop(columns=[base_year], inplace=True)

        # Output an audit of the growth factors calculated
        if reports_dir is not None:
            path = os.path.join(reports_dir, gf_report_fname)
            pd.DataFrame(growth_factors).to_csv(path, index=False)

    # ## NTEM CONTROL YEARS ## #
    # Figure out which years to control
    control_years = list()
    if control_base_year:
        control_years.append(base_year)
    if control_future_years:
        control_years += future_years

    audits = list()
    for year in control_years:
        # Init audit
        year_audit = {'year': year}

        # Setup paths
        ntem_fname = consts.NTEM_CONTROL_FNAME % ('pa', year)
        ntem_path = os.path.join(ntem_dir, ntem_fname)

        # Read in control files
        ntem_totals = pd.read_csv(ntem_path)
        ntem_lad_lookup = pd.read_csv(lad_lookup_path)

        print("\nPerforming NTEM constraint for %s..." % year)
        vector, audit, *_, = new_control_to_ntem(
            control_df=vector,
            ntem_totals=ntem_totals,
            zone_to_lad=ntem_lad_lookup,
            constraint_cols=ntem_control_cols,
            constraint_dtypes=constraint_dtypes,
            base_value_name=year,
            ntem_value_name=vector_type,
            trip_origin=trip_origin
        )

        # Update Audits for output
        year_audit.update(audit)
        audits.append(year_audit)

    # Controlling to NTEM seems to change some of the column dtypes
    dtypes = {c: d for c, d in zip(ntem_control_cols, ntem_control_dtypes)}
    vector = vector.astype(dtypes)

    # Write the audit to disk
    if len(audits) > 0 and reports_dir is not None:
        path = os.path.join(reports_dir, control_report_fname)
        pd.DataFrame(audits).to_csv(path, index=False)

    if growth_factors is None:
        return vector

    # ## ADD PRE CONTROL GROWTH BACK ON ## #
    # Merge on all possible columns
    merge_cols = du.list_safe_remove(list(growth_factors), all_years)
    vector = pd.merge(
        vector,
        growth_factors,
        how='left',
        on=merge_cols,
        suffixes=['_orig', '_gf'],
    ).fillna(1)

    # Add growth back on
    for year in future_years:
        vector[year] = vector[base_year] * vector["%s_gf" % year].values

    # make sure we only have the columns we started with
    vector = vector.reindex(columns=init_index_cols)
    vector = vector.groupby(init_group_cols).sum().reset_index()

    return vector
