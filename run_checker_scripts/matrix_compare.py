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
import sys
import operator

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append('..')
from normits_demand.concurrency import multiprocessing
from normits_demand import constants as consts

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
# OUTPUT_DIR = r'I:\NorMITs Demand\noham\EFS\iter3g\NTEM\Reports\EFS Reporter'

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

# Compare TMS PA to PA
# ORIGINAL_DIR = r'I:\NorMITs Demand\import\norms\decompiled_post_me'
# COMPARE_DIR = r'I:\NorMITs Demand\norms\EFS\iter3i\NTEM\Matrices\24hr PA Matrices'
# OUTPUT_DIR = r'F:/'
# REPORT_FNAME = 'tfn_report.csv'

# Compare EFS named PA
# ORIGINAL_DIR = r'I:\NorMITs Demand\import\norms\post_me\efs named'
# COMPARE_DIR = r'F:\NorMITs Demand\norms\EFS\iter3g\NTEM\Matrices\Compiled PA Matrices\compiled_non_split'
# OUTPUT_DIR = r'F:/'
# REPORT_FNAME = 'efs_named_report.csv'

# Compare NoRMS compiled to post-ME
# ORIGINAL_DIR = r'E:\NorMITs Demand\noham\EFS\iter3i\SC04_UZC\Matrices\OD Matrices'
# COMPARE_DIR = r'E:\NorMITs Demand\noham\EFS\iter3j\SC04_UZC\Matrices\OD Matrices'
# OUTPUT_DIR = r'E:/'
# REPORT_FNAME = 'compiled_report.csv'

ORIGINAL_DIR = r'I:\NorMITs Demand\TMS\iter9.3.1\car_and_passenger\Final Outputs\Compiled OD Matrices\PCU'
COMPARE_DIR = r'I:\NorMITs Demand\Distribution Model\iter9.3.2\car_and_passenger\Final Outputs\Compiled OD Matrices\PCU'
OUTPUT_DIR = r'E:\tms_dm_reports'
REPORT_FNAME = 'ext_compiled_report.csv'


def list_files(path):
    fnames = os.listdir(path)
    return [x for x in fnames if os.path.isfile(os.path.join(path, x))]


def starts_with(s, x):
    search_string = '^' + x
    return re.search(search_string, s) is not None


def compare_mats_fn(mat_fname, original_dir, compare_dir):
    report = dict()

    orig = pd.read_csv(os.path.join(original_dir, mat_fname), index_col=0)
    comp = pd.read_csv(os.path.join(compare_dir, mat_fname), index_col=0)

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

    # # Get specific area
    # split = 2516        # noham
    # total = 2770        # noham
    # internal = list(range(1, split+1))
    # external = list(range(split+1, total+1))
    #
    # area = external
    # join_fn = operator.or_
    #
    # # Create square masks for the rows and cols
    # orig.columns = orig.columns.astype(int)
    # col_mask = np.broadcast_to(orig.columns.isin(area), orig.shape)
    # index_mask = np.broadcast_to(orig.index.isin(area), orig.shape).T
    #
    # # Combine together to get the full mask
    # mask = join_fn(col_mask, index_mask)
    #
    # orig *= mask
    # comp *= mask

    # extract just the values
    orig = orig.values
    comp = comp.values

    # Get the absolute difference
    diff = np.absolute(orig - comp)

    # Store the comparison into a report
    report['matrix_name'] = mat_fname
    report['orig_sum'] = orig.sum()
    report['comp_sum'] = comp.sum()
    report['mean_diff'] = diff.mean()
    report['max_diff'] = diff.max()
    report['absolute_diff'] = diff.sum()
    report['actual_diff'] = comp.sum() - orig.sum()
    report['% actual_diff'] = (comp.sum() - orig.sum()) / orig.sum() * 100

    # # ## ERROR CHECKING ## #
    # max_idx = np.where(diff == diff.max())
    # max_row, max_col = max_idx[0][0], max_idx[1][0]
    # report['spacer'].append('')
    # report['max_OD'].append((max_row+1, max_col+1))
    # report['original_value'].append(orig[max_row, max_col])
    # report['test_value'].append(comp[max_row, max_col])

    return report


def main(original_dir, compare_dir, output_dir, report_fname):
    # Check the given directories exist
    if not os.path.isdir(original_dir):
        raise ValueError("Cannot find directory %s" % original_dir)

    if not os.path.isdir(compare_dir):
        raise ValueError("Cannot find directory %s" % compare_dir)

    # Get the files to compare
    original_mats = [x for x in list_files(original_dir) if '.csv' in x]
    compare_mats = [x for x in list_files(compare_dir) if '.csv' in x]

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

    # ## MULTIPROCESS MATRIX COMPARISON ## #
    print("Checks complete. Comparing matrices...")
    kwarg_dict =  {'original_dir': original_dir, 'compare_dir': compare_dir}
    kwarg_list = [{'mat_fname': x, **kwarg_dict} for x in original_mats]
    pbar_kwargs = {'desc': 'Comparing Matrices'}

    reports = multiprocessing.multiprocess(
        fn=compare_mats_fn,
        kwargs=kwarg_list,
        pbar_kwargs=pbar_kwargs,
        process_count=consts.PROCESS_COUNT,

    )

    # Write the report to disk
    df = pd.DataFrame(reports)
    df.to_csv(os.path.join(output_dir, report_fname), index=False)


if __name__ == '__main__':
    main(ORIGINAL_DIR, COMPARE_DIR, OUTPUT_DIR, REPORT_FNAME)
