# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 17:36:43 2021

@author: genie
"""

import numpy as np


# TODO: Allow control_to_ntem() to take flexible col names
def control_to_ntem(msoa_output,
                    ntem_totals,
                    lad_lookup,
                    group_cols = ['p', 'm'],
                    base_value_name = 'attractions',
                    ntem_value_name = 'attractions',
                    base_zone_name = 'msoa_zone_id',
                    purpose = 'hb'):
    """
    Control to a vector of NTEM constraints using single factor.
    Return productions controlled to NTEM.

    Parameters:
    ----------
    msoa_output:
        DF of productions, flexible segments, should be in MSOA zones

    ntem_totals:
        DF of NTEM totals to control to. Will need all group cols.

    lad_lookup:
        DF of translation between MSOA and LAD - should be in globals

    group_cols = ['p', 'm']:
        Segments to include in control. Will usually be ['p','m'], or
        ['p','m','ca'] for rail

    msoa_value_name = 'attractions':
        Name of the value column in the MSOA dataset - ie. productions or
        attractions. Might also be trips or dt.

    ntem_value_name = 'Attractions':
        Name of the value column in the NTEM dataset. Usually 'Productions'
        or 'Attractions' but could be used for ca variable or growth.
    
    base_zone_name = 'msoa_zone_id':
        name of base zoning system. Will be dictated by the lad lookup.
        Should be msoa in hb production model and attraction model but will
        be target zoning system in nhb production model.

    purpose = 'hb':
        Purpose set to aggregate on. Can be 'hb' or 'nhb'

    Returns:
    ----------
        adjusted_output:
            DF with same msoa zoning as input but controlled to NTEM.
    """
    # Init
    lad_lookup = lad_lookup.copy()
    ntem_totals = ntem_totals.copy()

    ntem_value_name = ntem_value_name.strip().lower()

    # Copy output
    output = msoa_output.copy()

    # Check params
    # Groups
    for col in group_cols:
        if col not in list(output):
            raise ValueError('Column ' + col + ' not in MSOA data')
        if col not in list(ntem_totals):
            raise ValueError('Column ' + col + ' not in NTEM data')

    # Establish all non trip segments
    segments = []
    for col in list(output):
        if col not in [base_zone_name,
                       base_value_name]:
            segments.append(col)

    # Purposes
    hb_purpose = [1,2,3,4,5,6,7,8]
    nhb_purpose = [12,13,14,15,16,18]
    if purpose not in ['hb', 'nhb']:
        raise ValueError('Invalid purpose type')
    else:
        if purpose == 'hb':
            p_vector = hb_purpose
        else:
            p_vector = nhb_purpose

    # Print target value
    before = output[base_value_name].sum()
    print('Before: ' + str(before))

    # Build factors
    ntem_k_factors = ntem_totals[ntem_totals['p'].isin(p_vector)].copy()

    kf_groups = ['lad_zone_id']
    for col in group_cols:
        kf_groups.append(col)
    kf_sums = kf_groups.copy()
    kf_sums.append(ntem_value_name)

    # Sum down to drop non attraction segments
    ntem_k_factors = ntem_k_factors.reindex(kf_sums,
                                            axis=1).groupby(
                                                    kf_groups).sum().reset_index()

    target = ntem_k_factors[ntem_value_name].sum()
    print('NTEM: ' + str(target))

    # Assumes vectors are called productions or attractions
    for col in group_cols:
        ntem_k_factors.loc[:,col] = ntem_k_factors[
                col].astype(int).astype(str)
    ntem_k_factors['lad_zone_id'] = ntem_k_factors[
            'lad_zone_id'].astype(float).astype(int)

    lad_lookup = lad_lookup.reindex(['lad_zone_id', base_zone_name], axis=1)

    output = output.merge(lad_lookup,
                          how = 'left',
                          on = base_zone_name)

    # No lad match == likely an island - have to drop
    output = output[~output['lad_zone_id'].isna()]

    for col in group_cols:
        output.loc[:,col] = output[col].astype(int).astype(str)
    output['lad_zone_id'] = output['lad_zone_id'].astype(float).astype(int)

    # Seed zero infill
    output[base_value_name] = output[base_value_name].replace(0,0.001)

    # Build LA adjustment
    # Note tp not in the picture
    af_groups = ['lad_zone_id']
    for col in group_cols:
        af_groups.append(col)
    af_sums = af_groups.copy()
    af_sums.append(base_value_name)

    adj_fac = output.reindex(af_sums, axis=1).groupby(
            af_groups).sum().reset_index().copy()
    # Just have to do this manually
    adj_fac['lad_zone_id'] = adj_fac['lad_zone_id'].astype(float).astype(int)

    # Merge NTEM values
    adj_fac = adj_fac.merge(ntem_k_factors,
                            how = 'left',
                            on = af_groups)
    # Get adjustment factors
    adj_fac['adj_fac'] = adj_fac[ntem_value_name]/adj_fac[base_value_name]
    af_only = af_groups.copy()
    af_only.append('adj_fac')
    adj_fac = adj_fac.reindex(af_only, axis=1)
    adj_fac['adj_fac'] = adj_fac['adj_fac'].replace(np.nan, 1)

    for col in group_cols:
        adj_fac.loc[:,col] = adj_fac[col].astype(int).astype(str)

    adjustments = adj_fac['adj_fac']

    # TODO: Report adj factors here
    output = output.merge(adj_fac,
                          how = 'left',
                          on = kf_groups)

    output[base_value_name] = output[base_value_name] * output['adj_fac']

    # Output segmented lad totals
    lad_groups = ['lad_zone_id']
    for col in segments:
        lad_groups.append(col)
    lad_index = lad_groups.copy()
    lad_index.append(base_value_name)

    lad_totals = output.reindex(lad_index,
                                axis=1).groupby(
                                        lad_groups).sum().reset_index()

    # Reindex outputs
    output = output.drop(['lad_zone_id', 'adj_fac'], axis=1)

    after = output[base_value_name].sum()
    print('After: ' + str(after))

    audit = {
        'before': before,
        'target': target,
        'after': after
    }

    return output, audit, adjustments, lad_totals
