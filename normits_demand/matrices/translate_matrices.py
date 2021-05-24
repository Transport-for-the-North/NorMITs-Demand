# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 12:11:52 2020

@author: genie
"""

import os

import pandas as pd
import numpy as np

import normits_demand.utils.utils as nup
import normits_demand.utils.n_matrix_split as ms

import_folder = (r'I:\NorMITs Demand\noham\v0.3-EFS_Output\NTEM\iter3f\Matrices\OD Matrices')
export_folder = (r'I:\NorMITs Synthesiser\Nelum\iter2\Outputs\From Home Matrices')

translation_lookup_folder = ('I:/NorMITs Synthesiser/Zone Translation/Export')

start_zone_model_folder = (r'I:\NorMITs Synthesiser\Noham\Model Zone Lookups')
end_zone_model_folder = (r'I:\NorMITs Synthesiser\Nelum\Model Zone Lookups')

start_zoning_system = 'Noham'
end_zoning_system = 'Nelum'


def convert_correspondence_to_wide(long_correspondence: pd.DataFrame,
                                   primary_key: str = None,
                                   secondary_key: str = None,
                                   weight_value: str = None,
                                   placeholder_value: int = 9999):

    """
    Convert long format zone correspondences into wide format for use
    with wide format matrix reporter.

    long_correspondence:
        Standard format translation, weighted or otherwise.
    primary_key:
        Unique identifier ie. side of correspondence with most zones
    secondary_key:
        Secondary identifier ie. side with fewest zones
    placeholder_value:
        Integer for infilling non corresponded unique zones
    """

    # Init
    lc = long_correspondence.copy()
    # If no weight - add simple weight, assuming many:1
    if weight_value is None:
        lc['weight'] = 1
        weight_value = 'weight'

    # Simplify correspondence
    lc = lc.reindex([primary_key, secondary_key, weight_value], axis=1)
    lc = lc.reset_index(drop=True)

    # Fill in any consecutive primary key zones dropped in translation
    min_pk, max_pk = min(lc[primary_key]), max(lc[primary_key])
    trans_ready = pd.DataFrame({primary_key: range(min_pk, max_pk+1)})
    trans_ready = trans_ready.merge(lc, how='left', on=primary_key)

    # Check if that's gone many: many

    # Infill nans on right hand side with a default placeholder
    trans_ready[secondary_key] = trans_ready[secondary_key].fillna(
        placeholder_value)
    trans_ready[weight_value] = trans_ready[weight_value].fillna(
        1)

    # Build audit zone totals
    pk_len = len(trans_ready[primary_key])

    # Pivot to wide - infill 0s
    wide_c = trans_ready.pivot(index=primary_key,
                               columns=secondary_key,
                               values=weight_value)
    wide_c = wide_c.fillna(0)

    # Audit axis sum
    pk_tot = sum(wide_c.values.sum(axis=0))
    sk_tot = sum(wide_c.values.sum(axis=1))
    if pk_len != pk_tot or pk_len != sk_tot:
        raise ValueError('unique zones %d and translation sum %d differ' % (
            pk_len, pk_tot))

    return wide_c


def _get_correspondence(translation_folder,
                        start_zoning_system,
                        end_zoning_system,
                        translation_type='pop'):
    """
    translation_type = 'pop', 'emp', 'pop_emp', 'spatial'
    """
    # Index folder
    lookup_folder = os.listdir(translation_folder)
    # Sub on type
    if translation_type in ['pop', 'emp']:
        lookup_folder = [x for x in lookup_folder if translation_type in x]
    elif translation_type == 'pop_emp':
        lookup_folder = [x for x in lookup_folder if 'pop' in x or 'emp' in x]
    elif translation_type == 'spatial':
        lookup_folder = [x for x in lookup_folder if
                         'pop' not in x and 'emp' not in x]
    
    # Sub on zone names
    lookup_folder = [x for x in lookup_folder if start_zoning_system.lower()
    in x and end_zoning_system.lower() in x]

    # Import
    lookups = {}
    for l in lookup_folder:
        lookup = pd.read_csv(
                os.path.join(translation_lookup_folder,
                             l))

        for col in list(lookup):
            if 'zone_id' in col:
                lookup[col] = lookup[col].astype(int)

        lookups.update({l:lookup})

    return lookups


def _define_mat_headings(mat_format):
    """
    
    """

    if mat_format == 'pa':
        left_col = 'p_zone'
        right_col = 'a_zone'
    elif mat_format == 'od':
        left_col = 'o_zone'
        right_col = 'd_zone'
    
    return left_col, right_col



def translate_demand(mat,
                     translation,
                     target_col,
                     other_col,
                     named_cols):
    """
    named_cols = [base col name, target col name, overlap name]
    """
    trans_mat = mat.copy().rename(columns={
                target_col:named_cols[0]})
    trans_mat = trans_mat.merge(translation,
                                how='left',
                                on=named_cols[0])

    # Translate demand
    trans_mat['dt'] = trans_mat['dt'] * trans_mat[named_cols[2]]
    # rename target col to col
    trans_mat = trans_mat.rename(
            columns={named_cols[1]:target_col})
    # Group and sum
    trans_mat = trans_mat.reindex(
            [target_col, other_col, 'dt'], axis=1).groupby(
                    [target_col, other_col]).sum().reset_index()

    return trans_mat

def translate_matrices(start_zoning_system,
                       end_zoning_system,
                       translation_lookup_folder,
                       import_folder,
                       export_folder,
                       start_zone_model_folder=None,
                       end_zone_model_folder=None,
                       translation_type='pop',  # default to pop
                       mat_format='od',
                       before_tld=True,
                       after_tld=False,
                       export=True):
    """
    OLD VERSION WITH LONG JOIN METHOD
    translation_type = 'pop', 'emp', 'pop_emp', 'spatial'
    
    """

    # TODO: make this work for a pop emp weight at PA
    # TODO: Multithread

    # Define mat format variables
    left_col, right_col = _define_mat_headings(mat_format)
        
    # Import lookup
    lookups = _get_correspondence(translation_lookup_folder,
                                  start_zoning_system,
                                  end_zoning_system,
                                  translation_type)

    # Get contents of input folder
    input_mats = nup.parse_mat_output(os.listdir(import_folder),
                                      sep='_',
                                      mat_type=mat_format,
                                      file_format='.csv')
    
    translation_report = list()
    before_tld_report = list()
    after_tld_report = list()

    # TODO: Multiprocess
    # TODO: Unit test w/ generated output table

    for index, row in input_mats.iterrows():
        print(row['file'])
        mat = pd.read_csv(os.path.join(
                import_folder,
                row['file']))

        if before_tld:
            # TODO: Should be functions - at least 2
            # TODO: Hacking for weird filenames here - should just get it
            if row['p'] == 'commute':
                p = 1
            elif row['p'] == 'business':
                p = 2
            elif row['p'] == 'other':
                p = 3
            else:
                p = row['p']

            calib_params = {'p': p,
                            'm': row['m']}

            if row['trip_origin'] == 'hb' or 'nhb' not in row['file']:
                target_tp = '24hr'
            elif row['trip_origin'] == 'nhb' or 'nhb' in row['file']:
                target_tp = 'tp'
                calib_params.update({'tp': 1})  # Default to tp 1 if nhb

            costs = nup.get_costs(
                start_zone_model_folder,
                calib_params,
                tp=target_tp,
                iz_infill=0.5)[0]

            costs['cost'] = costs['cost'].round(0)
            costs['p_zone'] = costs['p_zone'].astype(int)
            costs['a_zone'] = costs['a_zone'].astype(int)

            # Pivot mat to long
            long_mat = pd.melt(
                mat,
                id_vars=list(mat)[0],
                var_name='a_zone',
                value_name='dt',
                col_level=0)
            long_mat = long_mat.rename(columns={list(mat)[0]: 'p_zone'})
            long_mat['p_zone'] = long_mat['p_zone'].astype(int)
            long_mat['a_zone'] = long_mat['a_zone'].astype(int)

            tld = long_mat.merge(costs,
                                 how='left',
                                 on=['p_zone', 'a_zone'])
            del long_mat

            tld = tld.reindex(
                ['cost', 'dt'], axis=1).groupby('cost').sum().reset_index()
            tld['cost'] = tld['cost'].astype(int)

            before_tld_report.append({row['file']: tld})
            if export:
                report_name = row['file'].replace(
                    '.csv', '_tld_report.csv')
                tld.to_csv(
                    os.path.join(
                        export_folder,
                        report_name
                    ), index=False
                )

        mat = pd.melt(mat,
                      id_vars=[left_col],
                      var_name=right_col,
                      value_name='dt',
                      col_level=0)

        # Make int, wide or not
        mat[left_col] = mat[left_col].astype(int)
        mat[right_col] = mat[right_col].astype(int)

        # Get before total
        before = mat['dt'].sum()

        if translation_type == 'pop':
            for item, dat in lookups.items():
                if 'pop' in item:
                    translation = dat

        # Reindex cols
        b_col = (start_zoning_system.lower() + '_zone_id')
        t_col = (end_zoning_system.lower() + '_zone_id')
        b_ov = [x for x in list(translation) if
                start_zoning_system.lower() in x and 'overlap' in x][0]
        named_cols = [b_col, t_col, b_ov]

        translation = translation.reindex(named_cols,
                                          axis=1)

        # Translate
        for target_col in [left_col,right_col]:
            if target_col == left_col:
                other_col = right_col
            else:
                other_col = left_col

            mat = translate_demand(mat,
                                   translation,
                                   target_col,
                                   other_col,
                                   named_cols)

            mat = mat.reindex([left_col, right_col, 'dt'], axis=1)

        after = mat['dt'].sum()
        print('Total before ' + str(before))
        print('Total after ' + str(after))

        # TODO: Back to square format if you want

        # Build translation report
        translation_report.append({'matrix': row['file'],
                                   'before': before,
                                   'after': after})

        if export:
            mat.to_csv(os.path.join(export_folder, row['file']), index=False)

    translation_report = pd.DataFrame(translation_report)

    if export:
        translation_report.to_csv(
            os.path.join(
                export_folder,
                'translation_report.csv'
            ), index=False
        )

    return translation_report, before_tld_report, after_tld_report


def matrix_zone_translation(mat: np.ndarray,
                            sector_trans_mat: np.ndarray) -> np.ndarray:

    """
    Much needed function to translate demand from a given matrix shape
    to another - using a weighted translation.

    mat: ndarray of wide matrix of travel demand, function assumes consecutive
    zero based indices - ie. if you want indexers you'll have to retain them yourself.

    sector_trans_mat: ndarray of wide matrix in standard many:1 many:many format
    could be built from convert_correspondence_to_wide.
    """

    n_mat, n_sec = sector_trans_mat.shape

    # Translate rows
    t_shape = (n_mat, n_mat, n_sec)
    a = np.broadcast_to(np.expand_dims(mat, axis=2), t_shape)
    trans_a = np.broadcast_to(
        np.expand_dims(sector_trans_mat, axis=1), t_shape)
    temp = a * trans_a

    # mat is transposed, but we need it this way
    col_mat = temp.sum(axis=0)

    # Translate cols
    t_shape = (n_mat, n_sec, n_sec)
    b = np.broadcast_to(np.expand_dims(col_mat, axis=2), t_shape)
    trans_b = np.broadcast_to(
        np.expand_dims(sector_trans_mat, axis=1), t_shape)
    temp = b * trans_b
    out_mat = temp.sum(axis=0)

    return out_mat
