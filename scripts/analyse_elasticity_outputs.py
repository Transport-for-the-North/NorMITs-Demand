# -*- coding: utf-8 -*-
"""
Created on: Tues May 25 09:56:23 2021
Updated on:

Original author: Ben Taylor
Last update made by:
Other updates made by:

File purpose:
Analyse the outputs of an Elasticity run
"""
import os
import sys
import itertools
import functools
import operator

from collections import defaultdict

from multiprocessing import Pool

# Third party
import tqdm
import pandas as pd

# Local imports
sys.path.append("..")
from normits_demand.utils import general as du
from normits_demand.utils import file_ops

from normits_demand.concurrency import multiprocessing

NOHAM_INPUT = r'I:\NorMITs Demand\noham\EFS\iter3i\SC02_PP\Matrices\24hr PA Matrices WFH\internal'
NORMS_INPUT = r'I:\NorMITs Demand\norms\EFS\iter3i\SC02_PP\Matrices\24hr PA Matrices WFH\internal'

NOHAM_OUTPUT = r'I:\NorMITs Demand\noham\EFS\iter3i\SC02_PP\Matrices\24hr PA Matrices - Elasticity\internal'
NORMS_OUTPUT = r'I:\NorMITs Demand\norms\EFS\iter3i\SC02_PP\Matrices\24hr PA Matrices - Elasticity\internal'

NOHAM_EXTERNAL = r'I:\NorMITs Demand\noham\EFS\iter3i\SC02_PP\Matrices\24hr PA Matrices WFH\external'
NORMS_EXTERNAL = r'I:\NorMITs Demand\norms\EFS\iter3i\SC02_PP\Matrices\24hr PA Matrices WFH\external'

OUTPUT = r'F:\elasticity.csv'

YEARS = [2018, 2033, 2040, 2050]
YEARS = [2050]

UC_TO_P = {
    'commute': [1],
    'business': [2, 12],
    'other': [3, 4, 5, 6, 7, 8, 13, 14, 15, 16, 18],
}


def find_uc_mats(path):
    # Get the csv / .pbz files
    all_mats = file_ops.list_files(path, ftypes=['.csv', '.pbz2'])

    # Split all_mats into dictionaries
    uc_dict = defaultdict(dict)
    mats = all_mats.copy()
    for year in YEARS:
        # Filter down to the mats for this year
        yr_str = "_yr%s_" % year
        yr_mats = [x for x in mats if yr_str in x]
        init_yr_mats = yr_mats.copy()

        # Grab the mats for the purposes
        for uc, ps in UC_TO_P.items():
            # Get the relevant mats
            ps_str = ['_p%s_' % p for p in ps]
            uc_dict[year][uc] = [x for x in yr_mats if du.is_in_string(ps_str, x)]

            # Remove from yearly as we have them - faster!
            for m in uc_dict[year][uc]:
                yr_mats.remove(m)

        # Remove yearly mats as we have them
        for m in init_yr_mats:
            mats.remove(m)

    return uc_dict


def get_mat_totals(path, fnames):
    mats = list()
    for fname in fnames:
        full_path = os.path.join(path, fname)
        mats.append(file_ops.read_df(full_path, index_col=0, find_similar=True))

    mats_sum = functools.reduce(operator.add, mats)

    return mats_sum.sum().sum()


def mp_func(year, purpose, model, norms, noham):
    # Determine paths
    if model == 'noham':
        in_path = NOHAM_INPUT
        out_path = NOHAM_OUTPUT
        ext_path = NOHAM_EXTERNAL
        mat_dict = noham
    elif model == 'norms':
        in_path = NORMS_INPUT
        out_path = NORMS_OUTPUT
        ext_path = NORMS_EXTERNAL
        mat_dict = norms
    else:
        raise ValueError("WHAT?!")

    # Build a list of args
    paths = [out_path, in_path, ext_path]
    dict_keys = ['output', 'input', 'external']

    # We don't actually output in 2018!
    if year == 2018:
        paths = [in_path, in_path, ext_path]
        dict_keys = ['input', 'input', 'external']

    args = [(p, mat_dict[k][year][purpose]) for p, k in zip(paths, dict_keys)]

    # Multiprocess
    res = [get_mat_totals(*a) for a in args]

    # Store results
    return {
        'year': year,
        'model': model,
        'purpose': purpose,
        'input': res[1],
        'output': res[0],
        'external': res[2],
    }


def main():

    # Find the matrices that belong to each UC
    print("Reading in matrices...")
    paths = [
        NORMS_INPUT,
        NORMS_OUTPUT,
        NORMS_EXTERNAL,
        NOHAM_INPUT,
        NOHAM_OUTPUT,
        NOHAM_EXTERNAL,
    ]

    with Pool(processes=6) as pool:
        procs = [pool.apply_async(find_uc_mats, [p]) for p in paths]
        res = [p.get(timeout=100) for p in procs]

        norms = dict()
        norms['input'] = res[0]
        norms['output'] = res[1]
        norms['external'] = res[2]

        noham = dict()
        noham['input'] = res[3]
        noham['output'] = res[4]
        noham['external'] = res[5]

    # ## Analyse - create df ## #
    pbar_kwargs = ({
        'total': len(YEARS) * len(UC_TO_P.keys()) * 2,
        'desc': "Analysing outputs...",
        'colour': '#0d0f3d',
    })

    kwarg_list = list()
    purposes = reversed(UC_TO_P.keys())
    for purpose, year, model in itertools.product(purposes, YEARS, ['noham', 'norms']):
        kwarg_list.append({
            'year': year,
            'purpose': purpose,
            'model': model,
            'norms': norms,
            'noham': noham,
        })

    ph = multiprocessing.multiprocess(
        fn=mp_func,
        kwargs=kwarg_list,
        process_count=os.cpu_count()-2,
        pbar_kwargs=pbar_kwargs,
    )

    pd.DataFrame(ph).to_csv(OUTPUT, index=False)


if __name__ == '__main__':
    main()
