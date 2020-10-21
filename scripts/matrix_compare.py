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

ORIGINAL_DIR = r'E:\NorMITs Demand\noham\v2_2-EFS_Output\iter1\Matrices\Post-ME Matrices\OD Matrices'
COMPARE_DIR = r'E:\NorMITs Demand\noham\v2_2-EFS_Output\iter1\Matrices\Post-ME Matrices\Test OD Matrices'

OUTPUT_DIR = COMPARE_DIR
REPORT_FNAME = 'comparison_report.csv'

TRIP_ORIGIN = 'hb'


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
    original_mats = list_files(ORIGINAL_DIR)
    compare_mats = list_files(COMPARE_DIR)

    # Filter if needed
    if TRIP_ORIGIN is not None:
        original_mats = [x for x in original_mats if starts_with(x, TRIP_ORIGIN)]
        compare_mats = [x for x in compare_mats if starts_with(x, TRIP_ORIGIN)]
    compare_mats = [x for x in compare_mats if x in original_mats]

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
        report['total_diff'].append(diff.sum())

    # Write the report to disk
    pd.DataFrame(report).to_csv(os.path.join(OUTPUT_DIR, REPORT_FNAME),
                                index=False)


if __name__ == '__main__':
    main()
