# -*- coding: utf-8 -*-
"""
Created on: 23/08/2021
Updated on:

Original author: Ben Taylor
Last update made by:
Other updates made by:

File purpose:
Front-End script for converting Tour Props into From Home and To Home factors.
New format is needed for optimised PA to OD conversion.
"""
# Built-Ins
import os
import sys
import operator
import functools

from typing import Any
from typing import List
from typing import Dict


# Third Party
import numpy as np
import pandas as pd

from tqdm import tqdm

# Local Imports
sys.path.append('..')
import normits_demand as nd

from normits_demand import constants as consts
from normits_demand.utils import general as du
from normits_demand.utils import file_ops

from normits_demand.matrices import utils as mat_utils

# ## GLOBALS ## #
# Special format for NoRMS conversion
MODEL_NAME = 'norms'
MODE = consts.MODEL_MODES[MODEL_NAME][0]

NORMS_ZONES = list(range(1, 1300 + 1))
NORMS_INTERNAL_ZONES = list(range(1, 1156 + 1))
NORMS_EXTERNAL_ZONES = list(range(1157, 1300 + 1))

BASE_YEAR = 2018

NORMS_TOUR_PROPS_INPUT_FNAME = r'NoRMS_Tour_Prop.pkl'

CA_CONVERSION = {
    1: 'nca',
    2: 'ca',
}

HB_PURPOSES = [1, 2, 3, 4, 5, 6, 7, 8]
NHB_PURPOSES = [12, 13, 14, 15, 16, 18]
TP_NEEDED = [1, 2, 3, 4]

PURPOSE_TO_TOUR_PROP_KEYS = {
    1: {'internal': ['hbw'],  'external': ['ex_hbw']},
    2: {'internal': ['hbeb'], 'external': ['ex_eb']},
    3: {'internal': ['hbo'],  'external': ['ex_oth']},
    4: {'internal': ['hbo'],  'external': ['ex_oth']},
    5: {'internal': ['hbo'],  'external': ['ex_oth']},
    6: {'internal': ['hbo'],  'external': ['ex_oth']},
    7: {'internal': ['hbo'],  'external': ['ex_oth']},
    8: {'internal': ['hbo'],  'external': ['ex_oth']},
    12: {'internal': ['nhbeb'], 'external': ['ex_eb']},
    13: {'internal': ['nhbo'],  'external': ['ex_oth']},
    14: {'internal': ['nhbo'],  'external': ['ex_oth']},
    15: {'internal': ['nhbo'],  'external': ['ex_oth']},
    16: {'internal': ['nhbo'],  'external': ['ex_oth']},
    18: {'internal': ['nhbo'],  'external': ['ex_oth']},
}

FH_FACTORS_TO_KEYS = {
    1: [11, 12, 13, 14],
    2: [21, 22, 23, 24],
    3: [31, 32, 33, 34],
    4: [41, 42, 43, 44],
}

TH_FACTORS_TO_KEYS = {
    1: [11, 21, 31, 41],
    2: [12, 22, 32, 42],
    3: [13, 23, 33, 43],
    4: [14, 24, 34, 44],
}


def sum_dict_list(dict_list: List[Dict[Any, Any]]) -> Dict[Any, Any]:
    """
    Sums all dictionaries in dict_list together.

    Parameters
    ----------
    dict_list:
        A list of dictionaries to sum together.

    Returns
    -------
    summed_dict:
        A single dictionary of all the dicts in dict_list summed together.
    """

    # Define the accumulator function to call in functools.reduce
    def reducer(accumulator, item):
        for key, value in item.items():
            accumulator[key] = accumulator.get(key, 0) + value
        return accumulator

    return functools.reduce(reducer, dict_list)


def get_input_tour_props(efs_imports):
    tour_prop_dir = efs_imports['post_me_tours']
    file_path = os.path.join(tour_prop_dir, NORMS_TOUR_PROPS_INPUT_FNAME)
    return nd.from_pickle(file_path)


def get_hb_output_paths(efs_imports, purpose, mode, ca):
    trip_origin = 'hb'
    tour_factor_dir = efs_imports['post_me_fh_th_factors']

    # Generate the output factor paths
    fh_factor_fname = du.get_dist_name(
        trip_origin=trip_origin,
        matrix_format='fh_factors',
        year=str(BASE_YEAR),
        purpose=str(purpose),
        mode=str(mode),
        car_availability=str(ca),
        suffix='.pkl'
    )
    fh_factor_path = os.path.join(tour_factor_dir, fh_factor_fname)

    th_factor_fname = fh_factor_fname.replace('fh_factors', 'th_factors')
    th_factor_path = os.path.join(tour_factor_dir, th_factor_fname)

    return fh_factor_path, th_factor_path


def infill_all_zeros(factor_dict):
    fh_zero_dict = dict.fromkeys(TP_NEEDED)
    for tp in fh_zero_dict:
        fh_factors = factor_dict[tp]
        fh_zero_dict[tp] = np.where(fh_factors == 0, True, False)
    fh_zeros = functools.reduce(operator.and_, fh_zero_dict.values())

    for tp in factor_dict:
        factor_dict[tp] = np.where(fh_zeros, .25, factor_dict[tp])

    return factor_dict


def hb_conversion(input_tour_props, efs_imports, mode):
    # Validate globals that are needed
    if not all([p in PURPOSE_TO_TOUR_PROP_KEYS for p in HB_PURPOSES]):
        raise ValueError(
            "Not all of the defined HB_PURPOSES can be found in "
            "PURPOSE_TO_TOUR_PROP_KEYS. Double check that all the "
            "keys match!\n"
            "Missing keys: %s"
            % (set(HB_PURPOSES) - set(PURPOSE_TO_TOUR_PROP_KEYS))
        )

    if not all([tp in FH_FACTORS_TO_KEYS for tp in TP_NEEDED]):
        raise ValueError(
            "Not all of the defined TP_NEEDED can be found in "
            "FH_FACTORS_TO_KEYS. Double check that all the "
            "keys match!\n"
            "Missing keys: %s"
            % (set(TP_NEEDED) - set(FH_FACTORS_TO_KEYS))
        )

    if not all([tp in TH_FACTORS_TO_KEYS for tp in TP_NEEDED]):
        raise ValueError(
            "Not all of the defined TP_NEEDED can be found in "
            "TH_FACTORS_TO_KEYS. Double check that all the "
            "keys match!\n"
            "Missing keys: %s"
            % (set(TP_NEEDED) - set(TH_FACTORS_TO_KEYS))
        )

    # Use masks to make sure we're only getting internal and external numbers
    norms_zones = max(NORMS_ZONES)
    blank_norms_df = pd.DataFrame(
        index=NORMS_ZONES,
        columns=NORMS_ZONES,
        data=np.zeros((norms_zones, norms_zones)),
    )
    internal_mask = mat_utils.get_internal_mask(blank_norms_df, NORMS_INTERNAL_ZONES)
    external_mask = mat_utils.get_external_mask(blank_norms_df, NORMS_EXTERNAL_ZONES)

    # ## CONVERT PURPOSE BY PURPOSE ## #
    for p in tqdm(HB_PURPOSES, desc='Converting hb purposes to factors'):
        # Whittle down the input tour props
        int_keys = PURPOSE_TO_TOUR_PROP_KEYS[p]['internal']
        ext_keys = PURPOSE_TO_TOUR_PROP_KEYS[p]['external']
        int_p_tour_props = [input_tour_props[x] for x in int_keys]
        ext_p_tour_props = [input_tour_props[x] for x in ext_keys]

        # ## GRAB ALL INPUT MATRICES ## #
        for ca in CA_CONVERSION.keys():
            # Get the input tour props that relate to this ca segment
            tour_props_list = [x[CA_CONVERSION[ca]] for x in int_p_tour_props]

            # Initialise lists to collect all tour props
            fh_factors_list = list()
            th_factors_list = list()

            # ## CONVERT INTERNAL INPUT TOUR PROPS ## #
            for tour_props in tour_props_list:
                # From home conversion
                fh_factors_dict = dict.fromkeys(TP_NEEDED)
                for tp in fh_factors_dict:
                    tp_tour_props = [tour_props[x] * internal_mask for x in FH_FACTORS_TO_KEYS[tp]]
                    fh_factors_dict[tp] = functools.reduce(operator.add, tp_tour_props)
                fh_factors_list.append(fh_factors_dict)

                # To home conversion
                th_factors_dict = dict.fromkeys(TP_NEEDED)
                for tp in th_factors_dict:
                    tp_tour_props = [tour_props[x] * internal_mask for x in TH_FACTORS_TO_KEYS[tp]]
                    th_factors_dict[tp] = functools.reduce(operator.add, tp_tour_props)
                th_factors_list.append(th_factors_dict)

            # ## CONVERT EXTERNAL INPUT TOUR PROPS ## #
            # We fudge this slightly and use ca no matter what
            for tour_props in ext_p_tour_props:
                # Grab from home factors
                fh_factors_dict = dict.fromkeys(TP_NEEDED)
                for tp in fh_factors_dict:
                    fh_factors_dict[tp] = tour_props['ca_fh'][tp] * external_mask
                fh_factors_list.append(fh_factors_dict)

                # Grab to home factors
                th_factors_dict = dict.fromkeys(TP_NEEDED)
                for tp in th_factors_dict:
                    th_factors_dict[tp] = tour_props['ca_th'][tp] * external_mask
                th_factors_list.append(th_factors_dict)

            # Add all the dictionaries in the lists together
            fh_factors_dict = sum_dict_list(fh_factors_list)
            th_factors_dict = sum_dict_list(th_factors_list)

            # ## EQUALLY SPLIT WHERE ALL FACTORS ARE 0 ## #
            fh_factors_dict = infill_all_zeros(fh_factors_dict)
            th_factors_dict = infill_all_zeros(th_factors_dict)

            fh_factor_path, th_factor_path = get_hb_output_paths(
                efs_imports=efs_imports,
                purpose=p,
                mode=mode,
                ca=ca,
            )

            file_ops.to_pickle(fh_factors_dict, fh_factor_path)
            file_ops.to_pickle(th_factors_dict, th_factor_path)

            # ## VALIDATE THE GENERATED TOUR PROPS ## #
            target = norms_zones ** 2
            tol = 1e-3

            # Get the achieved values
            fh_total = 0
            for v in fh_factors_dict.values():
                fh_total += v

            th_total = 0
            for v in th_factors_dict.values():
                th_total += v

            # Percentage diff
            fh_perc_diff = np.abs(fh_total - target) / target
            th_perc_diff = np.abs(th_total - target) / target

            if fh_perc_diff > tol:
                raise ValueError(
                    "Produced From-home tour proportions are not within %s%% "
                    "of what they should be! (%s)"
                    "Check the inputs and outputs to find out where demand will "
                    "go missing."
                    % (tol, target)
                )

            if th_perc_diff > tol:
                raise ValueError(
                    "Produced To-home tour proportions are not within %s%% "
                    "of what they should be! (%s)"
                    "Check the inputs and outputs to find out where demand will "
                    "go missing."
                    % (tol, target)
                )


def nhb_conversion():
    pass


def main():
    # Build and instance of EFS to et imports
    iter_num = '3j'
    scenario_name = consts.SC04_UZC
    import_home = 'I:/'
    export_home = 'E:/'

    # Need an instance of EFS to get IO paths
    efs = nd.ExternalForecastSystem(
        model_name=MODEL_NAME,
        iter_num=iter_num,
        scenario_name=scenario_name,
        import_home=import_home,
        export_home=export_home,
        verbose=False,
    )

    input_tour_props = get_input_tour_props(efs.imports)

    # HB and NHB need converting differently
    hb_conversion(input_tour_props, efs.imports, MODE)
    exit()
    nhb_conversion()


if __name__ == '__main__':
    main()
