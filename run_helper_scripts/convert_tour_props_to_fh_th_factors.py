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
import warnings

from typing import Dict, Any, Tuple

from itertools import product


# Third Party
import numpy as np

from tqdm import tqdm

# Local Imports
sys.path.append('..')
import normits_demand as nd

from normits_demand.utils import general as du
from normits_demand.utils import math_utils
from normits_demand.utils import file_ops

# ## GLOBALS ## #
# Running variables
MODEL_NAME = 'miham'
NOTEM_ITER = '9.7-COVID'
MODE = nd.Mode.CAR

TOUR_PROP_DIR = (
    f'I:/NorMITs Demand/import/modal/{MODE.get_name()}'
    f'/pre_me_tour_proportions/v{NOTEM_ITER}/{MODEL_NAME}'
)
TOUR_FACTOR_DIR = os.path.join(TOUR_PROP_DIR, 'fh_th_factors')

# CONSTANTS
BASE_YEAR = 2021
TRIP_ORIGIN = 'hb'
PURPOSES = [1, 2, 3, 4, 5, 6, 7, 8]
TP_NEEDED = [1, 2, 3, 4]
ZONE_TRANSLATION_DIR = r'I:\NorMITs Demand\import\zone_translation\one_to_one'
TOUR_PROPS_SEGMENTATIONS = {
    nd.Mode.CAR: nd.get_segmentation_level("hb_p_m_car"),
    nd.Mode.TRAIN: nd.get_segmentation_level("hb_p_m_ca_rail"),
}


def get_io(
        segmentation: nd.SegmentationLevel, segment_params: Dict[str, Any]
    ) -> Tuple[Dict, Dict, Dict, str, str]:

    # Generate the output factor paths
    fh_factor_fname = segmentation.generate_file_name(
        segment_params,
        file_desc="fh_factors",
        trip_origin=TRIP_ORIGIN,
        year=BASE_YEAR
    ) + ".pkl"
    fh_factor_path = os.path.join(TOUR_FACTOR_DIR, fh_factor_fname)

    th_factor_fname = fh_factor_fname.replace('fh_factors', 'th_factors')
    th_factor_path = os.path.join(TOUR_FACTOR_DIR, th_factor_fname)

    # ## Load the tour proportions - always generated on base year ## #
    # Load the model zone tour proportions
    tour_prop_fname = segmentation.generate_file_name(
        segment_params,
        file_desc="tour_proportions",
        trip_origin=TRIP_ORIGIN,
        year=BASE_YEAR
    ) + ".pkl"
    model_tour_props = nd.read_pickle(os.path.join(TOUR_PROP_DIR, tour_prop_fname))

    # Load the aggregated tour props
    lad_fname = tour_prop_fname.replace('tour_proportions', 'lad_tour_proportions')
    try:
        lad_tour_props = nd.read_pickle(os.path.join(TOUR_PROP_DIR, lad_fname))
    except FileNotFoundError:
        lad_tour_props = None

    tfn_fname = tour_prop_fname.replace('tour_proportions', 'tfn_tour_proportions')
    try:
        tfn_tour_props = nd.read_pickle(os.path.join(TOUR_PROP_DIR, tfn_fname))
    except FileNotFoundError:
        tfn_tour_props = None

    return (
        model_tour_props,
        lad_tour_props,
        tfn_tour_props,
        fh_factor_path,
        th_factor_path,
    )


def maybe_get_aggregated_tour_proportions(orig: int,
                                          dest: int,
                                          model_tour_props: Dict[int, Dict[int, np.array]],
                                          lad_tour_props: Dict[int, Dict[int, np.array]],
                                          tfn_tour_props: Dict[int, Dict[int, np.array]],
                                          model2lad: Dict[int, int],
                                          model2tfn: Dict[int, int],
                                          ) -> np.array:
    # Translate to the aggregated zones
    lad_orig = model2lad.get(orig, -1)
    lad_dest = model2lad.get(dest, -1)
    tfn_orig = model2tfn.get(orig, -1)
    tfn_dest = model2tfn.get(dest, -1)

    # If the model zone tour proportions are zero, fall back on the
    # aggregated tour proportions
    bad_key = False
    if model_tour_props[orig][dest].sum() != 0:
        od_tour_props = model_tour_props[orig][dest]

    elif lad_tour_props[lad_orig][lad_dest].sum() != 0:
        # We have a problem if this used a negative key
        bad_key = lad_orig < 0 or lad_dest < 0

        if not bad_key:
            # First - fall back to LAD aggregation
            od_tour_props = lad_tour_props[lad_orig][lad_dest]

    elif tfn_tour_props[tfn_orig][tfn_dest].sum() != 0:
        # We have a problem if this used a negative key
        bad_key = tfn_orig < 0 or tfn_dest < 0

        if not bad_key:
            # Second - Try fall back to TfN Sector aggregation
            od_tour_props = tfn_tour_props[tfn_orig][tfn_dest]

    else:
        # If all aggregations are zero, evenly split
        od_tour_props = np.array([1 / 16]*16).reshape((4, 4))

    if bad_key:
        # If no good aggregations, evenly split
        od_tour_props = np.array([1 / 16] * 16).reshape((4, 4))

    return od_tour_props


def main():
    tour_props_seg = TOUR_PROPS_SEGMENTATIONS[MODE]

    # ## CONVERT PURPOSE BY PURPOSE ## #
    for seg_params in tour_props_seg:
        # Get the tour props and export path
        io = get_io(tour_props_seg, seg_params)
        model_tour_props, lad_tour_props, tfn_tour_props = io[:3]
        fh_factor_path, th_factor_path = io[3:5]

        # Make sure output paths exist
        out_dir, _ = os.path.split(fh_factor_path)
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        # ## VALIDATE INPUTS ## #
        max_o = max(model_tour_props.keys())
        max_d = max(model_tour_props[max_o].keys())
        if max_o != max_d:
            raise ValueError(
                "Not Gonna output a square mat. Something bad is happening"
            )

        # This assumes that the keys are sequential from 1 to max_o, which is not the case for MiHAM
        # if((len(model_tour_props.keys()) != max_o)
        #    or (len(model_tour_props[max_o].keys()) != max_d)):
        #     # Throw Error
        #     raise ValueError(
        #         "Tour Props seem to be missing some keys!"
        #     )

        if model_tour_props[max_o][max_d].shape != (len(TP_NEEDED), len(TP_NEEDED)):
            raise ValueError(
                "Loaded in tour props are not the right size!\n"
                "Expected: (%s, %s)\n"
                "Got: %s\n"
                % (len(TP_NEEDED), len(TP_NEEDED), model_tour_props[max_o][max_d].shape)
            )

        # ## GET TRANSLATIONS ## #
        # Load the zone aggregation dictionaries for this model
        try:
            model2lad = du.get_zone_translation(
                import_dir=ZONE_TRANSLATION_DIR,
                from_zone=MODEL_NAME,
                to_zone='lad'
            )
        except FileNotFoundError:
            warnings.warn("cannot find LAD translation", RuntimeWarning)
            model2lad = {}
        try:
            model2tfn = du.get_zone_translation(
                import_dir=ZONE_TRANSLATION_DIR,
                from_zone=MODEL_NAME,
                to_zone='tfn_sectors'
            )
        except FileNotFoundError:
            warnings.warn("cannot find TfN translation", RuntimeWarning)
            model2tfn = {}

        # ## INITIALISE OUTPUT DICTIONARIES ## #
        idx = list(model_tour_props.keys())
        # Lookup from zone name to index
        idx_lookup = dict(zip(idx, range(len(idx))))

        # Create empty from_home factor matrices
        fh_factors_dict = dict.fromkeys(TP_NEEDED)
        for tp in fh_factors_dict.keys():
            fh_factors_dict[tp] = np.zeros((len(idx), len(idx)))

        # Create empty to_home factor matrices
        th_factors_dict = dict.fromkeys(TP_NEEDED)
        for tp in th_factors_dict.keys():
            th_factors_dict[tp] = np.zeros((len(idx), len(idx)))

        # ## CREATE FACTORS FOR EACH OD ## #
        total = len(idx) ** 2
        desc = "Converting %s to factors" % tour_props_seg.get_segment_name(seg_params)
        for orig, dest in tqdm(product(idx, idx), total=total, desc=desc):

            # Will get the aggregated tour props if needed
            od_tour_props = maybe_get_aggregated_tour_proportions(
                orig=orig,
                dest=dest,
                model_tour_props=model_tour_props,
                lad_tour_props=lad_tour_props,
                tfn_tour_props=tfn_tour_props,
                model2lad=model2lad,
                model2tfn=model2tfn,
            )

            # Np.array is 0 based, o/d is labelled
            orig_loc = idx_lookup[orig]
            dest_loc = idx_lookup[dest]

            # Convert to factors and assign
            fh_factors = np.sum(od_tour_props, axis=1)
            for tp, factor in enumerate(fh_factors, start=1):
                fh_factors_dict[tp][orig_loc, dest_loc] = factor

            th_factors = np.sum(od_tour_props, axis=0)
            for tp, factor in enumerate(th_factors, start=1):
                th_factors_dict[tp][orig_loc, dest_loc] = factor

        # ## OUTPUT FACTORS ## #
        file_ops.write_pickle(fh_factors_dict, fh_factor_path)
        file_ops.write_pickle(th_factors_dict, th_factor_path)

        # ## VALIDATE FACTORS ## #
        for factor_dict in [fh_factors_dict, th_factors_dict]:
            total = 0
            for v in factor_dict.values():
                total += v.sum()

            expected = len(idx) ** 2
            if not math_utils.is_almost_equal(total, expected):
                raise ValueError(
                    "Factor dictionary is not similar enough to the expected "
                    "value!\n"
                    "Got: %s\n"
                    "Expected: %s\n"
                    "Please validate the factors written out to '%s' and '%s'."
                    % (total, expected, fh_factor_path, th_factor_path)
                )


if __name__ == '__main__':
    main()
