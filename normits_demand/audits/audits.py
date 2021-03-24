# -*- coding: utf-8 -*-
"""
Created on: Wed November 4 15:20:32 2020
Updated on:

Original author: Ben Taylor
Last update made by:
Other updates made by:

File purpose:
Module of audit functions to carry out during runs to ensure the values
being returned make sense
"""
import os

from typing import List

import numpy as np
import pandas as pd

from collections import defaultdict

# Local imports
from normits_demand.utils import general as du


# BACKLOG: furness audits are not audits. They should be moved over to
#  reporting.
#  labels: EFS, QoL Updates


class AuditError(du.NormitsDemandError):
    """
    Exception raised for errors when auditing values
    """

    def __init__(self, message=None):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return self.message


def audit_furness(row_targets: pd.DataFrame,
                  col_targets: pd.DataFrame,
                  furness_out: pd.DataFrame,
                  output_path: str,
                  idx_col: str = 'model_zone_id',
                  unique_col: str = 'trips',
                  row_prefix: str = 'row',
                  col_prefix: str = 'col',
                  index_name: str = 'zone'
                  ) -> None:
    """
    Writes a report of the furness performance to disk

    Parameters
    ----------
    row_targets:
        The target values for the sum of each row. In production/attraction
        furnessing, this would be the productions. The idx_col must match
        the idx_col of col_targets.

    col_targets:
        The target values for the sum of each column. In production/attraction
        furnessing, this would be the attractions. The idx_col must match
        the idx_col of row_targets.

    furness_out:
        The actual achieved output of the furness. The index and columns must
        match the idx_col of row_targets and col_targets.

    output_path:
        Path to the directory to output this audit

    idx_col:
        Name of the columns in row_targets and col_targets that contain the
        index data that matches seed_values index/columns

    unique_col:
        Name of the columns in row_targets and col_targets that contain the
        values to target during the furness

    row_prefix:
        The prefix to attach to the row analysis in the output file

    col_prefix:
        The prefix to attach to the column analysis in the output file

    index_name:
        The name to give to the index column in the output file

    Returns
    -------
    None
    """
    # Init
    row_targets = row_targets.copy()
    col_targets = col_targets.copy()
    index = row_targets[idx_col].unique()

    # Make sure we only have what we need
    row_targets = row_targets.reindex([idx_col, unique_col], axis='columns')
    col_targets = col_targets.reindex([idx_col, unique_col], axis='columns')
    row_targets = row_targets.set_index(idx_col)
    col_targets = col_targets.set_index(idx_col)

    # Get everything into the same format
    row_achieved = furness_out.sum(axis='columns')
    col_achieved = furness_out.sum(axis='rows')
    row_targets = row_targets.sum(axis='columns')
    col_targets = col_targets.sum(axis='columns')

    # Set the output column headers
    row_target_col = '%s_target' % row_prefix
    row_ach_col = '%s_achieved' % row_prefix
    row_diff_col = '%s_difference' % row_prefix
    col_target_col = '%s_target' % col_prefix
    col_ach_col = '%s_achieved' % col_prefix
    col_diff_col = '%s_difference' % col_prefix

    # Calculate the difference for every zone
    audit_dict = defaultdict(list)
    for idx in index:
        # Get the differences
        row_diff = np.absolute(row_targets.loc[idx] - row_achieved.loc[idx])
        col_diff = np.absolute(col_targets.loc[idx] - col_achieved.loc[idx])

        # Add all to the outputs
        audit_dict[index_name].append(idx)
        audit_dict[row_target_col].append(row_targets.loc[idx])
        audit_dict[row_ach_col].append(row_achieved.loc[idx])
        audit_dict[row_diff_col].append(row_diff)
        audit_dict[col_target_col].append(col_targets.loc[idx])
        audit_dict[col_ach_col].append(col_achieved.loc[idx])
        audit_dict[col_diff_col].append(col_diff)

    # Convert to table and write out
    pd.DataFrame(audit_dict).to_csv(output_path, index=False)


def summarise_audit_furness(audit_furness_output_path: str,
                            trip_origin: str,
                            format_name: str,
                            year: str,
                            p_needed: List[int],
                            m_needed: List[int],
                            soc_needed: List[int] = None,
                            ns_needed: List[int] = None,
                            ca_needed: List[int] = None,
                            tp_needed: List[int] = None,
                            fname_suffix: str = None,
                            p_diff_col_name: str = 'p_difference',
                            a_diff_col_name: str = 'a_difference',
                            ) -> None:
    # TODO: Write summarise_audit_furness() docs
    # Create iterator
    segment_generator = du.cp_segmentation_loop_generator(p_list=p_needed,
                                                          m_list=m_needed,
                                                          ns_list=ns_needed,
                                                          soc_list=soc_needed,
                                                          ca_list=ca_needed,
                                                          tp_list=tp_needed)

    # Generate a summary for each segmentation in this year
    summary_ph = list()
    index_cols = set()
    for segment_dict in segment_generator:
        # Keep a record of columns for indexing df later
        index_cols = index_cols.union(segment_dict.keys())

        # Need to add year for filename
        name_dict = segment_dict.copy()
        name_dict['yr'] = year

        # Read in the audit for this segmentation
        dist_name = du.calib_params_to_dist_name(
            trip_origin=trip_origin,
            matrix_format=format_name,
            calib_params=name_dict,
            suffix=fname_suffix,
            csv=True
        )
        df = pd.read_csv(os.path.join(audit_furness_output_path, dist_name))

        # Get summaries of the productions and attractions
        p_mean = np.mean(df[p_diff_col_name].values)
        p_sum = np.sum(df[p_diff_col_name].values)

        a_mean = np.mean(df[a_diff_col_name].values)
        a_sum = np.sum(df[a_diff_col_name].values)

        # Build into a dictionary to record values
        segment_summary = segment_dict.copy()
        segment_summary.update({
            'p_mean': p_mean,
            'p_sum': p_sum,
            'a_mean': a_mean,
            'a_sum': a_sum,
        })
        summary_ph.append(segment_summary)

    # Build the index columns for the df
    index_cols = du.segmentation_order(list(index_cols))
    index_cols += ['p_mean', 'p_sum', 'a_mean', 'a_sum']

    # Build all summaries into a dataframe and format
    out_df = pd.DataFrame(summary_ph).reindex(columns=index_cols)

    # Write to disk
    fname = "%s_%s_furness_summary.csv" % (trip_origin, year)
    out_path = os.path.join(audit_furness_output_path, fname)
    out_df.to_csv(out_path, index=False)
