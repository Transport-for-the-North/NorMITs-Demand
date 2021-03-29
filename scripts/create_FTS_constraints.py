# -*- coding: utf-8 -*-
"""
Created on: Mon March 29 09:54:12 2021
Updated on:

Original author: Ben Taylor
Last update made by:
Other updates made by:

File purpose:
Generates the FTS constraints files similar to NTEM constraints. Files will be used
to compare FTS outputs to FTS to see if outputs are similar.
"""

import os

import pandas as pd
import numpy as np

from tqdm import tqdm

# CONSTANTS
POP_SUBPATH = os.path.join('population', 'future_growth_values.csv')
EMP_SUBPATH = os.path.join('employment', 'future_growth_values.csv')

MSOA_ZONE_COL = 'msoa_zone_id'
LAD_ZONE_COL = 'lad_zone_id'
LAD_TO_MSOA_PATH = r"I:\NorMITs Demand\import\zone_translation\no_overlap\lad_to_msoa.csv"

# scenario, year
OUT_BASE_FNAME = '%s_pop_emp_%s.csv'

# RUNNING PARAMS
IN_DIR = r'I:\NorMITs Demand\import\scenarios'
OUT_DIR = r'I:\NorMITs Demand\import\fts_constraints'

FTS_NAMES = [
    'SC01_JAM',
    'SC02_PP',
    'SC03_DD',
    'SC04_UZC',
]
YEARS = [2018, 2033, 2035, 2050]


def main():
    years_str = [str(x) for x in YEARS]

    lad_to_msoa = pd.read_csv(LAD_TO_MSOA_PATH)
    lad_to_msoa = lad_to_msoa.reindex(columns=[MSOA_ZONE_COL, LAD_ZONE_COL])

    msoa_group_cols = [MSOA_ZONE_COL]
    msoa_index_cols = msoa_group_cols.copy() + years_str

    lad_group_cols = [LAD_ZONE_COL]
    lad_index_cols = lad_group_cols.copy() + years_str

    pbar = tqdm(
        desc="Generating FTS constraints",
        total=len(FTS_NAMES) * len(YEARS),
    )

    for fts_name in FTS_NAMES:
        # Get the pop and emp data
        pop_emp = list()
        for subpath in [POP_SUBPATH, EMP_SUBPATH]:
            # Read in file
            path = os.path.join(IN_DIR, fts_name, subpath)
            df = pd.read_csv(path)

            # Filter down to just what we need
            df = df.reindex(columns=msoa_index_cols)
            df = df.groupby(msoa_group_cols).sum().reset_index()

            # Translate to LAD
            df = pd.merge(
                df,
                lad_to_msoa,
                on=MSOA_ZONE_COL,
            )
            df = df.reindex(columns=lad_index_cols)
            df = df.groupby(lad_group_cols).sum().reset_index()

            pop_emp.append(df)

        # Combine the pop and emp
        pop, emp = pop_emp
        pop_emp_df = pd.merge(
            pop,
            emp,
            suffixes=['_pop', '_emp'],
            on=LAD_ZONE_COL,
        )

        # Create an output file for each year
        for year in years_str:
            # Build the out col names
            pop_col = '%s_pop' % year
            emp_col = '%s_emp' % year

            col_rename = {
                pop_col: 'population',
                emp_col: 'employment',
            }
            index_cols = [LAD_ZONE_COL, pop_col, emp_col]

            # Structure the output
            out_df = pop_emp_df[index_cols].copy()
            out_df = out_df.rename(columns=col_rename)

            # Write to disk
            out_fname = OUT_BASE_FNAME % (fts_name, year)
            out_path = os.path.join(OUT_DIR, out_fname)
            out_df.to_csv(out_path, index=False)

            pbar.update(1)


if __name__ == '__main__':
    main()
