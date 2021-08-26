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

from typing import Dict

from itertools import product


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

# ## GLOBALS ## #
# Currently only have tour proportions for Noham
MODEL_NAME = 'noham'
MODE = consts.MODEL_MODES[MODEL_NAME][0]

BASE_YEAR = 2018
TRIP_ORIGIN = 'hb'
PURPOSES = [1, 2, 3, 4, 5, 6, 7, 8]
TP_NEEDED = [1, 2, 3, 4]


def get_io(model_name,
           iter_num,
           scenario_name,
           import_home,
           export_home,
           purpose,
           mode,
           ):

    # Build the import paths
    efs = nd.ExternalForecastSystem(
        model_name=model_name,
        iter_num=iter_num,
        scenario_name=scenario_name,
        import_home=import_home,
        export_home=export_home,
        verbose=False,
    )
    tour_prop_dir = efs.imports['post_me_tours']
    tour_factor_dir = efs.imports['post_me_fh_th_factors']

    # Generate the output factor paths
    fh_factor_fname = du.get_dist_name(
        trip_origin=TRIP_ORIGIN,
        matrix_format='fh_factors',
        year=str(BASE_YEAR),
        purpose=str(purpose),
        mode=str(mode),
        suffix='.pkl'
    )
    fh_factor_path = os.path.join(tour_factor_dir, fh_factor_fname)

    th_factor_fname = fh_factor_fname.replace('fh_factors', 'th_factors')
    th_factor_path = os.path.join(tour_factor_dir, th_factor_fname)

    # ## Load the tour proportions - always generated on base year ## #
    # Load the model zone tour proportions
    tour_prop_fname = du.get_dist_name(
        trip_origin=TRIP_ORIGIN,
        matrix_format='tour_proportions',
        year=str(BASE_YEAR),
        purpose=str(purpose),
        mode=str(mode),
        suffix='.pkl'
    )
    model_tour_props = nd.from_pickle(os.path.join(tour_prop_dir, tour_prop_fname))

    # Load the aggregated tour props
    lad_fname = tour_prop_fname.replace('tour_proportions', 'lad_tour_proportions')
    lad_tour_props = nd.from_pickle(os.path.join(tour_prop_dir, lad_fname))

    tfn_fname = tour_prop_fname.replace('tour_proportions', 'tfn_tour_proportions')
    tfn_tour_props = nd.from_pickle(os.path.join(tour_prop_dir, tfn_fname))

    return (
        model_tour_props,
        lad_tour_props,
        tfn_tour_props,
        fh_factor_path,
        th_factor_path,
        efs.imports,
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
        # First - fall back to LAD aggregation
        od_tour_props = lad_tour_props[lad_orig][lad_dest]

        # We have a problem if this used a negative key
        bad_key = lad_orig < 0 or lad_dest < 0

    elif tfn_tour_props[tfn_orig][tfn_dest].sum() != 0:
        # Second - Try fall back to TfN Sector aggregation
        od_tour_props = tfn_tour_props[tfn_orig][tfn_dest]

        # We have a problem if this used a negative key
        bad_key = tfn_orig < 0 or tfn_dest < 0

    else:
        # If all aggregations are zero, evenly split
        od_tour_props = np.array([1 / 16]*16).reshape((4, 4))

    if bad_key:
        # If no good aggregations, evenly split
        od_tour_props = np.array([1 / 16] * 16).reshape((4, 4))

    return od_tour_props


def main():
    # Build and instance of EFS to et imports
    iter_num = '3j'
    scenario_name = consts.SC04_UZC
    import_home = 'I:/'
    export_home = 'E:/'

    # ## CONVERT PURPOSE BY PURPOSE ## #
    for p in PURPOSES:
        # Get the tour props and export path
        io = get_io(
            model_name=MODEL_NAME,
            iter_num=iter_num,
            scenario_name=scenario_name,
            import_home=import_home,
            export_home=export_home,
            purpose=p,
            mode=MODE,
        )

        model_tour_props, lad_tour_props, tfn_tour_props = io[:3]
        fh_factor_path, th_factor_path = io[3:5]
        efs_imports = io[5]

        # Make sure output paths exist
        out_dir, _ = os.path.split(fh_factor_path)
        if not os.path.exists(out_dir):
            raise IOError("Directory %s does not exist" % out_dir)

        # ## VALIDATE INPUTS ## #
        max_o = max(model_tour_props.keys())
        max_d = max(model_tour_props[max_o].keys())
        if max_o != max_d:
            raise ValueError(
                "Not Gonna output a square mat. Something bad is happening"
            )

        if((len(model_tour_props.keys()) != max_o)
           or (len(model_tour_props[max_o].keys()) != max_d)):
            # Throw Error
            raise ValueError(
                "Tour Props seem to be missing some keys!"
            )

        if model_tour_props[max_o][max_d].shape != (len(TP_NEEDED), len(TP_NEEDED)):
            raise ValueError(
                "Loaded in tour props are not the right size!\n"
                "Expected: (%s, %s)\n"
                "Got: %s\n"
                % (len(TP_NEEDED), len(TP_NEEDED), model_tour_props[max_o][max_d].shape)
            )

        # ## GET TRANSLATIONS ## #
        # Load the zone aggregation dictionaries for this model
        model2lad = du.get_zone_translation(
            import_dir=efs_imports['zone_translation']['one_to_one'],
            from_zone=MODEL_NAME,
            to_zone='lad'
        )
        model2tfn = du.get_zone_translation(
            import_dir=efs_imports['zone_translation']['one_to_one'],
            from_zone=MODEL_NAME,
            to_zone='tfn_sectors'
        )

        # ## INITIALISE OUTPUT DICTIONARIES ## #
        idx = range(1, max_o+1)

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
        desc = "Converting p%s to factors" % p
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

            # Np.array is 0 based, o/d is 1 based
            orig_loc = orig - 1
            dest_loc = dest - 1

            # Convert to factors and assign
            fh_factors = np.sum(od_tour_props, axis=1)
            for tp, factor in enumerate(fh_factors, start=1):
                fh_factors_dict[tp][orig_loc, dest_loc] = factor

            th_factors = np.sum(od_tour_props, axis=0)
            for tp, factor in enumerate(th_factors, start=1):
                th_factors_dict[tp][orig_loc, dest_loc] = factor

        # ## OUTPUT FACTORS ## #
        file_ops.to_pickle(fh_factors_dict, fh_factor_path)
        file_ops.to_pickle(th_factors_dict, th_factor_path)

        # ## VALIDATE FACTORS ## #
        for factor_dict in [fh_factors_dict, th_factors_dict]:
            total = 0
            for v in factor_dict.values():
                total += v.sum()

            expected = len(idx) ** 2
            if not du.is_almost_equal(total, expected, significant=5):
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
