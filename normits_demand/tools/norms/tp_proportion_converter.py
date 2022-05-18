# -*- coding: utf-8 -*-
"""
Created on: 11/05/2022
Updated on:

Original author: Ben Taylor
Last update made by:
Other updates made by:

File purpose:
Takes the outputs of tp_proportion_extractor and converts into a useable
format.
DEPENDS HEAVILY ON THE FORMAT OUTPUT IN tp_proportion_extractor
"""
# Built-Ins
import operator
import functools
import logging

from typing import Any
from typing import List
from typing import Dict
from typing import Tuple
from typing import Callable

# Third Party
import numpy as np
import pandas as pd

# Local Imports
from normits_demand import logging as nd_log
from normits_demand import core as nd_core
from normits_demand.utils import pandas_utils as pd_utils

SegmentFactorDict = Dict[Any, Dict[Any, Dict[int, np.ndarray]]]
LOG = logging.getLogger(__name__)


def infill_missing_values(
    infill_dict: Dict[int, np.ndarray],
    all_zones: List[Any],
    relevant_zones_fn: Callable = None,
    relevant_zones_fn_kwargs: Dict[str, Any] = None,
) -> Dict[int, np.ndarray]:
    # init
    return_dict = dict.fromkeys(infill_dict.keys())
    if relevant_zones_fn_kwargs is None:
        relevant_zones_fn_kwargs = dict()

    # Infill any missing values
    factor_sum = functools.reduce(operator.add, infill_dict.values())
    missing_idx = (factor_sum < 0.05).nonzero()

    for tp, mat in infill_dict.items():
        new_mat = mat.copy()
        new_mat[missing_idx] = 1 / len(infill_dict)

        # Filter to relevant area only
        df = pd.DataFrame(
            data=new_mat,
            index=all_zones,
            columns=all_zones,
        )
        relevant_df = relevant_zones_fn(df, **relevant_zones_fn_kwargs)
        return_dict[tp] = relevant_df.values

    return return_dict


def extract_fh_th_factors(
    tp_to_keys: Dict[int, List[int]],
    data: Dict[int, np.ndarray],
    zoning_system: nd_core.ZoningSystem,
) -> Dict[int, np.ndarray]:
    # init
    return_dict = dict.fromkeys(tp_to_keys.keys())

    # Extract the factors
    for tp, tp_keys in tp_to_keys.items():
        # Generate the factors
        matrices = [data[k] for k in tp_keys]
        factors = functools.reduce(operator.add, matrices)
        return_dict[tp] = factors

    return_dict = infill_missing_values(
        infill_dict=return_dict,
        all_zones=zoning_system.unique_zones,
        relevant_zones_fn=pd_utils.get_internal_values,
        relevant_zones_fn_kwargs={"zones": zoning_system.internal_zones},
    )

    return return_dict


def convert_tour_proportions(
    tour_prop_pickle, zoning_system: nd_core.ZoningSystem
) -> Tuple[SegmentFactorDict, SegmentFactorDict]:
    """Convert the dictionary into from-home and to-home factor dicts"""
    purpose_to_keys = {
        "commute": "hbw",
        "business": "hbeb",
        "other": "hbo",
    }
    ca_to_keys = {1: "nca", 2: "ca"}
    fh_factors_to_keys = {
        1: [11, 12, 13, 14],
        2: [21, 22, 23, 24],
        3: [31, 32, 33, 34],
        4: [41, 42, 43, 44],
    }

    th_factors_to_keys = {
        1: [11, 21, 31, 41],
        2: [12, 22, 32, 42],
        3: [13, 23, 33, 43],
        4: [14, 24, 34, 44],
    }

    # Build the dictionaries
    fh_factors: SegmentFactorDict = dict.fromkeys(purpose_to_keys.keys())
    th_factors: SegmentFactorDict = dict.fromkeys(purpose_to_keys.keys())
    for purpose, p_key in purpose_to_keys.items():
        # Create purpose specific stuff
        fh_factors[purpose] = dict.fromkeys(ca_to_keys)
        th_factors[purpose] = dict.fromkeys(ca_to_keys)
        purpose_dict = tour_prop_pickle[p_key]

        for ca, ca_key in ca_to_keys.items():
            # Create ca stuff
            fh_factors[purpose][ca] = dict.fromkeys(fh_factors_to_keys)
            th_factors[purpose][ca] = dict.fromkeys(th_factors_to_keys)
            ca_dict = purpose_dict[ca_key]

            fh_factors[purpose][ca] = extract_fh_th_factors(
                tp_to_keys=fh_factors_to_keys,
                data=ca_dict,
                zoning_system=zoning_system,
            )

            th_factors[purpose][ca] = extract_fh_th_factors(
                tp_to_keys=th_factors_to_keys,
                data=ca_dict,
                zoning_system=zoning_system,
            )

            # Check for where things don't sum to 1
            lower_bound = 0.9
            upper_bound = 1.1
            n_zones = len(zoning_system.internal_zones)

            sum_mat = functools.reduce(operator.add, fh_factors[purpose][ca].values())
            sum_mat = sum_mat[:n_zones, :n_zones]
            idx = ((upper_bound > 1.1) | (sum_mat < lower_bound)).nonzero()
            n_bad_values = idx[0].shape[0]
            if n_bad_values > 0:
                LOG.warning(
                    "In %s, ca%s, from-home factors. Found %s values that are "
                    "not within the range %s-%s.",
                    purpose,
                    ca,
                    n_bad_values,
                    lower_bound,
                    upper_bound,
                )

            sum_mat = functools.reduce(operator.add, th_factors[purpose][ca].values())
            sum_mat = sum_mat[:n_zones, :n_zones]
            idx = ((sum_mat > upper_bound) | (sum_mat < lower_bound)).nonzero()
            n_bad_values = idx[0].shape[0]
            if n_bad_values > 0:
                LOG.warning(
                    "In %s, ca%s, to-home factors. Found %s values that are "
                    "not within the range %s-%s.",
                    purpose,
                    ca,
                    n_bad_values,
                    lower_bound,
                    upper_bound,
                )

    return fh_factors, th_factors


def convert_internal_tp_split_factors(tour_prop_pickle, zoning_system):
    purpose_to_keys = {
        "business": "nhbeb",
        "other": "nhbo",
    }
    ca_to_keys = {1: "nca", 2: "ca"}

    # Build the NHB splitting factor dictionary
    splitting_factors = dict.fromkeys(purpose_to_keys.keys())
    for purpose, p_key in purpose_to_keys.items():
        # Create purpose specific stuff
        splitting_factors[purpose] = dict.fromkeys(ca_to_keys)
        purpose_dict = tour_prop_pickle[p_key]

        for ca, ca_key in ca_to_keys.items():
            # Create ca stuff
            ca_dict = purpose_dict[ca_key]

            # Infill any 0 values
            splitting_factors[purpose][ca] = infill_missing_values(
                infill_dict=ca_dict,
                all_zones=zoning_system.unique_zones,
                relevant_zones_fn=pd_utils.get_internal_values,
                relevant_zones_fn_kwargs={"zones": zoning_system.internal_zones},
            )

            # Check for where things don't sum to 1
            lower_bound = 0.9
            upper_bound = 1.1
            n_zones = len(zoning_system.internal_zones)

            sum_mat = functools.reduce(operator.add, splitting_factors[purpose][ca].values())
            sum_mat = sum_mat[:n_zones, :n_zones]
            idx = ((sum_mat > upper_bound) | (sum_mat < lower_bound)).nonzero()
            n_bad_values = idx[0].shape[0]
            if n_bad_values > 0:
                LOG.warning(
                    "In %s, ca%s, internal nhb tp split factors. Found %s "
                    "values that are not within the range %s-%s.",
                    purpose,
                    ca,
                    n_bad_values,
                    lower_bound,
                    upper_bound,
                )

    return splitting_factors


def convert_external_tp_split_factors(tour_prop_pickle, zoning_system):
    purpose_to_keys = {
        "commute": "ex_hbw",
        "business": "ex_eb",
        "other": "ex_oth",
    }
    ca_to_keys = {"nca": "nca", "ca_fh": "ca_fh", "ca_th": "ca_th"}

    # Build the external splitting factor dictionary
    splitting_factors = dict.fromkeys(purpose_to_keys.keys())
    for purpose, p_key in purpose_to_keys.items():
        # Create purpose specific stuff
        splitting_factors[purpose] = dict.fromkeys(ca_to_keys)
        purpose_dict = tour_prop_pickle[p_key]

        for ca, ca_key in ca_to_keys.items():
            # Create ca stuff
            ca_dict = purpose_dict[ca_key]

            # Infill any 0 values
            splitting_factors[purpose][ca] = infill_missing_values(
                infill_dict=ca_dict,
                all_zones=zoning_system.unique_zones,
                relevant_zones_fn=pd_utils.get_external_values,
                relevant_zones_fn_kwargs={"zones": zoning_system.external_zones},
            )

            # Check for where things don't sum to 1
            lower_bound = 0.9
            upper_bound = 1.1
            n_zones = len(zoning_system.internal_zones)

            sum_mat = functools.reduce(operator.add, splitting_factors[purpose][ca].values())
            sum_mat[:n_zones, :n_zones] = np.inf
            idx = ((sum_mat > upper_bound) | (sum_mat < lower_bound))
            idx = (idx & (sum_mat != np.inf)).nonzero()
            n_bad_values = idx[0].shape[0]
            if n_bad_values > 0:
                LOG.warning(
                    "In %s, %s, external tp split factors. Found %s "
                    "values that are not within the range %s-%s.",
                    purpose,
                    ca,
                    n_bad_values,
                    lower_bound,
                    upper_bound,
                )

    return splitting_factors
