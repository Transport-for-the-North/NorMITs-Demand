# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 12:11:52 2020

@author: genie
"""

import os

import pandas as pd

import utils as nup

import_folder = (r'Y:\NorMITs Synthesiser\Nelum\Outputs\From Home Matrices')
export_folder = (r'Y:\NorMITs Synthesiser\Nelum\Outputs\From Home Matrices NELUM')

translation_lookup_folder = ('Y:/NorMITs Synthesiser/Zone Translation/Export')

start_zoning_system = 'Noham'
end_zoning_system = 'Nelum'

def _get_lookups(translation_lookup_folder,
                 start_zoning_system,
                 end_zoning_system,
                 translation_type = 'pop'):
    """
    translation_type = 'pop', 'emp', 'pop_emp', 'spatial'
    """
    # Index folder
    lookup_folder = os.listdir(translation_lookup_folder)
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
                       translation_type = 'pop', # default to pop
                       mat_format = 'od',
                       import_format = 'wide',
                       import_headers = True,
                       export_format = 'wide',
                       export_headers = True,
                       to_pcu = False,
                       to_ave_hour = False,
                       export = True):
    """
    translation_type = 'pop', 'emp', 'pop_emp', 'spatial'
    
    """
    # TODO: make this work for a pop emp weight at PA
    # TODO: Multithread

    # Define mat format variables
    left_col, right_col = _define_mat_headings(mat_format)
        
    # Import lookup
    lookups = _get_lookups(translation_lookup_folder,
                           start_zoning_system,
                           end_zoning_system,
                           translation_type)

    # Get contents of input folder
    input_mats = nup.parse_mat_output(os.listdir(import_folder),
                                      sep = '_',
                                      mat_type = mat_format,
                                      file_format = '.csv')
    
    translation_report = []

    # TODO: Multiprocess
    # TODO: Unit test w/ generated output table

    for index, row in input_mats.iterrows():
        print(row['file'])
        mat = pd.read_csv(os.path.join(
                import_folder,
                row['file']))
        # TODO: smart reading of mats
        mat = pd.melt(mat,
                      id_vars = [left_col],
                      var_name = right_col,
                      value_name = 'dt',
                      col_level = 0)

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
        translation_report.append({'matrix':row['file'],
                                   'before':before,
                                   'after':after})

        if export:
            mat.to_csv(os.path.join(export_folder,row['file']), index=False)

    translation_report = pd.DataFrame(translation_report)

    return translation_report