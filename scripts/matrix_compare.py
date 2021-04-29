# -*- coding: utf-8 -*-
"""
Created on: Fri October 16 08:26:11 2020
Updated on:

Original author: Ben Taylor
Last update made by:
Other updates made by:

File purpose:
Used to compare two sets of matrices from different directories and output
a report on their similarity
"""

import os
import re

from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

# Post-ME output
# base_path = r'E:\NorMITs Demand\noham\v0.3-EFS_Output\NTEM\iter3b\Matrices\Post-ME Matrices'
# OUTPUT_DIR = base_path

# Compare compiled matrices
# ORIGINAL_DIR = os.path.join(base_path, r'Compiled OD Matrices\from_pcu')
# COMPARE_DIR = os.path.join(base_path, 'Test PCU Compiled OD Matrices')
# TRIP_ORIGIN = None
# REPORT_FNAME = 'comparison_report_compiled.csv'

# Compare time period split OD matrices
# ORIGINAL_DIR = os.path.join(base_path, 'OD Matrices')
# COMPARE_DIR = os.path.join(base_path, 'Test OD Matrices')
# TRIP_ORIGIN = 'hb'
# REPORT_FNAME = 'comparison_report_tp_od.csv'

# ## COMPARE POST_ME INPUT TO 2018 OUTPUT ## #
TRIP_ORIGIN = None
OUTPUT_DIR = r'I:\NorMITs Demand\noham\EFS\iter3g\NTEM\Reports\EFS Reporter'

# Compare EFS output to post-ME PA
# ORIGINAL_DIR = r'I:\NorMITs Demand\import\noham\decompiled_post_me'
# COMPARE_DIR = r'I:\NorMITs Demand\noham\EFS\iter3g\NTEM\Matrices\24hr PA Matrices'
# REPORT_FNAME = 'output_to_post_me_pa_report.csv'

# Compare EFS output to post-ME OD
# ORIGINAL_DIR = r'I:\NorMITs Demand\import\noham\post_me\tms_seg_od'
# COMPARE_DIR = r'E:\NorMITs Demand\noham\EFS\NTEM\iter3g\Matrices\OD Matrices'
# REPORT_FNAME = 'output_to_post_me_od_report.csv'

# Compare EFS output to post-ME compiled OD
# ORIGINAL_DIR = r'I:\NorMITs Demand\import\noham\post_me\renamed'
# COMPARE_DIR = r'I:\NorMITs Demand\noham\EFS\iter3g\NTEM\Matrices\Compiled OD Matrices PCU'
# REPORT_FNAME = 'output_to_post_me_compiled_od_report.csv'

# Compare NoRMS compiled to post-ME
ORIGINAL_DIR = r'I:\NorMITs Demand\import\norms\post_me\renamed'
COMPARE_DIR = r'E:\NorMITs Demand\norms\EFS\iter3a\SC02_PP\Matrices\Compiled PA Matrices'

# Compare TMS PA to PA
# ORIGINAL_DIR = r'I:\NorMITs Demand\import\norms\decompiled_post_me'
# COMPARE_DIR = r'E:\NorMITs Demand\norms\EFS\iter3a\SC02_PP\Matrices\24hr PA Matrices'

# Compare EFS named PA
# ORIGINAL_DIR = r'I:\NorMITs Demand\import\norms\post_me\efs named'
# COMPARE_DIR = r'E:\NorMITs Demand\norms\EFS\iter3a\SC02_PP\Matrices\Compiled PA Matrices\compiled_non_split'

OUTPUT_DIR = r'E:/'
REPORT_FNAME = 'report.csv'


def list_files(path):
    fnames = os.listdir(path)
    return [x for x in fnames if os.path.isfile(os.path.join(path, x))]


def starts_with(s, x):
    search_string = '^' + x
    return re.search(search_string, s) is not None


def main():
    # Check the given directories exist
    if not os.path.isdir(ORIGINAL_DIR):
        raise ValueError("Cannot find directory %s" % ORIGINAL_DIR)

    if not os.path.isdir(COMPARE_DIR):
        raise ValueError("Cannot find directory %s" % COMPARE_DIR)

    # Get the files to compare
    original_mats = [x for x in list_files(ORIGINAL_DIR) if '.csv' in x]
    compare_mats = [x for x in list_files(COMPARE_DIR) if '.csv' in x]

    # Filter if needed
    if TRIP_ORIGIN is not None:
        original_mats = [x for x in original_mats if starts_with(x, TRIP_ORIGIN)]
        compare_mats = [x for x in compare_mats if starts_with(x, TRIP_ORIGIN)]
    compare_mats = [x for x in compare_mats if x in original_mats]

    # Check that all the matrices we need exist
    for mat_name in original_mats:
        if mat_name not in compare_mats:
            raise ValueError(
                "Cannot find all of the original matrices in the compare directory. "
                "unable to find %s in the compare dir."
                % mat_name
            )

    # Check the names match
    if len(original_mats) != len(compare_mats):
        raise ValueError(
            "Cannot find all of the original matrices in the compare directory. "
            "Found %d original, and %d matching in the compare dir."
            % (len(original_mats), len(compare_mats))
        )

    for mat in original_mats:
        if mat not in compare_mats:
            raise ValueError("Cannot find matrix in compare directory."
                             % mat)

    if not original_mats == compare_mats:
        raise ValueError("After all the checks, the list of matrices does not "
                         "match. Not sure what's gone wrong here.")

    print("Checks complete. Comparing matrices...")
    report = defaultdict(list)
    for mat_fname in tqdm(original_mats):
        orig = pd.read_csv(os.path.join(ORIGINAL_DIR, mat_fname), index_col=0)
        comp = pd.read_csv(os.path.join(COMPARE_DIR, mat_fname), index_col=0)

        # Check the dimensions
        # noinspection PyTypeChecker
        if all(orig.columns != comp.columns):
            raise ValueError(
                "The column names of matrix %s do not match in the original "
                "and compare directories. Please check manually."
                % mat_fname
            )

        # noinspection PyTypeChecker
        if all(orig.index != comp.index):
            raise ValueError(
                "The index names of matrix %s do not match in the original "
                "and compare directories. Please check manually."
                % mat_fname
            )

        # extract just the values
        orig = orig.values
        comp = comp.values

        # Get the absolute difference
        diff = np.absolute(orig - comp)

        # Store the comparison into a report
        report['matrix_name'].append(mat_fname)
        report['mean_diff'].append(diff.mean())
        report['max_diff'].append(diff.max())
        report['absolute_diff'].append(diff.sum())
        report['actual_diff'].append(comp.sum() - orig.sum())
        report['% actual_diff'].append((comp.sum() - orig.sum()) / orig.sum() * 100)

        # # ## ERROR CHECKING ## #
        # max_idx = np.where(diff == diff.max())
        # max_row, max_col = max_idx[0][0], max_idx[1][0]
        # report['spacer'].append('')
        # report['max_OD'].append((max_row+1, max_col+1))
        # report['original_value'].append(orig[max_row, max_col])
        # report['test_value'].append(comp[max_row, max_col])

    # Write the report to disk
    pd.DataFrame(report).to_csv(os.path.join(OUTPUT_DIR, REPORT_FNAME),
                                index=False)


if __name__ == '__main__':
    main()
