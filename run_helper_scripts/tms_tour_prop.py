# -*- coding: utf-8 -*-
"""
Created on: Wednesday October 20th 2021
Updated on:

Original author: Ben Taylor
Last update made by: Nirmal Kumar
Other updates made by: Isaac Scott

File purpose:
Pre ME tour proportions Generation
"""

# Built-Ins
import os
import sys
import collections

# Third party imports
import numpy as np
import pandas as pd

from tqdm import tqdm

# Local Imports
sys.path.append("..")
import normits_demand as nd
from normits_demand.utils import general as du

# ## GLOBALS ## #
MODES = [nd.MODE.TRAIN.get_mode_values()]
YEARS = [2018]
TPS = [1, 2, 3, 4]
ZONING_SYSTEM = "miham"
NOTEM_ITER = '9.6'
out_folder = r"C:\WSP_Projects\MidMITs\01 Pre-processing\Tour Proportions"

phi_import_folder = r"I:\NorMITs Demand\import\phi_factors"
notem_import_folder = r"T:\MidMITs Demand\MiTEM\iter%s\SC01_JAM\hb_productions" % NOTEM_ITER
phi_fname = "mode_%d_fhp_tp_pa_to_od.csv"
prod_vec_fname = "hb_msoa_notem_segmented_%d_dvec.pkl"
zone_translate_dir = r"I:\NorMITs Demand\import\zone_translation\one_to_one"

MODEL_FNAME = "hb_tour_proportions_yr%d_p%d_m%d.pkl"
LAD_FNAME = "hb_lad_tour_proportions_yr%d_p%d_m%d.pkl"
TFN_FNAME = "hb_tfn_tour_proportions_yr%d_p%d_m%d.pkl"
# out_folder = r"I:\NorMITs Demand\import\noham\pre_me_tour_proportions\example_new"


def tms_tour_prop():

    for year in YEARS:
        for mode in MODES:

            # ## GRAB PHI FACTORS BY MODE ## #
            print("Reading in Phi factors...")
            phi_file = phi_fname % mode
            phi_df = pd.read_csv(os.path.join(phi_import_folder, phi_file))

            # Rename cols to match notem
            rename = {
                'purpose_from_home': 'p',
                'time_from_home': 'tp',
                'time_to_home': 'phi',
                'direction_factor': 'factor',
            }
            phi_df = phi_df.rename(columns=rename)

            # Drop unneeded tps
            mask = (
                phi_df['tp'].isin(TPS)
                & phi_df['phi'].isin(TPS)
            )
            phi_df = phi_df[mask].copy()

            # Pivot
            phi_df = phi_df.pivot(
                index=['p', 'tp'],
                columns='phi',
                values='factor',
            ).reset_index()

            # ## GRAB TP SPLITS BY PURPOSE AND ZONE ## #  READS IN DATA 
            print("Reading in NoTEM time period split factors...")
            notem_file = prod_vec_fname % year
            notem_dvec = nd.DVector.load(os.path.join(notem_import_folder, notem_file))

            # Convert to needed segments and translate
            if MODES==[nd.MODE.TRAIN.get_mode_values()]:
                week_seg = nd.get_segmentation_level("hb_p_ca_tp_week")
                wday_seg = nd.get_segmentation_level("hb_p_ca_tp_wday")
            else:
                week_seg = nd.get_segmentation_level("hb_p_tp_week")
                wday_seg = nd.get_segmentation_level("hb_p_tp_wday")
            zoning = nd.get_zoning_system(ZONING_SYSTEM)

            notem_dvec = notem_dvec.aggregate(week_seg)
            notem_dvec = notem_dvec.subset(wday_seg)
            notem_df = notem_dvec.translate_zoning(zoning, weighting="population").to_df()

            # Adjust factors back to 1
            notem_df['sum'] = notem_df.groupby([zoning.col_name, 'p'])['val'].transform('sum')
            notem_df['val'] /= notem_df['sum']

            # If val is NaN, assume even split
            notem_df['val'] = notem_df['val'].fillna(0.25)
            notem_df = notem_df.drop(columns='sum')

            # ## CALCULATE TOUR PROPS PER ZONE ## #
            print("Beginning tour props calculation...")
            full_df = pd.merge(
                left=notem_df,
                right=phi_df,
                how='left',
                on=['p', 'tp']
            )

            # Split tps by phis
            for phi_col in TPS:
                full_df[phi_col] *= full_df['val']
            full_df = full_df.drop(columns='val')

            # ## STICK INTO O/D NESTED DICT ## #
            zones = zoning.unique_zones
            purposes = full_df['p'].unique()

            # Load the zone aggregation dictionaries for this zoning
            try:
                model2lad = du.get_zone_translation(
                    import_dir=zone_translate_dir,
                    from_zone=ZONING_SYSTEM,
                    to_zone='lad'
                )
            except FileNotFoundError:
                model2lad = None
            try:
                model2tfn = du.get_zone_translation(
                    import_dir=zone_translate_dir,
                    from_zone=ZONING_SYSTEM,
                    to_zone='tfn_sectors'
                )
            except FileNotFoundError:
                model2tfn = None

            # Define the default value for the nested defaultdict
            def empty_tour_prop():
                return np.zeros((len(TPS), len(TPS)))

            # Do by purpose
            desc = 'Generating tour props per purpose'
            for purpose in tqdm(purposes, desc=desc):
                print("Start purpose %s..." % purpose)
                p_df = full_df[full_df['p'] == purpose].reset_index(drop=True)
                p_df = p_df.drop(columns='p')

                # Loop through zones
                model_tour_props = dict.fromkeys(zones)
                if model2lad:
                    lad_tour_props = collections.defaultdict(empty_tour_prop)
                else:
                    lad_tour_props = None
                if model2tfn:
                    tfn_tour_props = collections.defaultdict(empty_tour_prop)
                else:
                    tfn_tour_props = None
                for orig in zones:
                    # Extract values from the DF
                    vals = p_df[p_df[zoning.col_name] == orig].copy()
                    vals = vals.drop(columns=zoning.col_name)
                    vals = vals.set_index('tp')
                    props = vals.values

                    # Make the nested dict - these will all be the same
                    # due to the phi factors not being zonally split
                    dest_dict = dict.fromkeys(zones)
                    dest_dict = {k: props.copy() for k in dest_dict.keys()}
                    model_tour_props[orig] = dest_dict

                    # Aggregate the tour props
                    if model2lad:
                        lad_orig = model2lad.get(orig, -1)
                        lad_tour_props[lad_orig] += props

                    if model2tfn:
                        tfn_orig = model2tfn.get(orig, -1)
                        tfn_tour_props[tfn_orig] += props

                # Expand aggregated dicts  - these will all be the same
                # due to the phi factors not being zonally split
                if lad_tour_props:
                    dest_dict = dict.fromkeys(lad_tour_props.keys())
                    for orig, props in lad_tour_props.items():
                        dest_dict = {k: props.copy() for k in dest_dict.keys()}
                        lad_tour_props[orig] = dest_dict

                if tfn_tour_props:
                    dest_dict = dict.fromkeys(tfn_tour_props.keys())
                    for orig, props in tfn_tour_props.items():
                        dest_dict = {k: props.copy() for k in dest_dict.keys()}
                        tfn_tour_props[orig] = dest_dict

                # Normalise all of the tour proportion matrices to 1
                print("Normalising to 1...")
                for agg_tour_props in [model_tour_props, lad_tour_props, tfn_tour_props]:
                    if agg_tour_props is None:
                        continue
                    for key1, inner_dict in agg_tour_props.items():
                        for key2, mat in inner_dict.items():
                            # Avoid warning if 0
                            if mat.sum() == 0:
                                continue
                            agg_tour_props[key1][key2] = mat / mat.sum()

                # Write files out
                if lad_tour_props:
                    lad_tour_props = du.defaultdict_to_regular(lad_tour_props)
                if tfn_tour_props:
                    tfn_tour_props = du.defaultdict_to_regular(tfn_tour_props)

                print("Writing files out...")
                out_file = MODEL_FNAME % (year, purpose, mode)
                out_path = os.path.join(out_folder, out_file)
                nd.write_pickle(model_tour_props, out_path)

                if lad_tour_props:
                    out_file = LAD_FNAME % (year, purpose, mode)
                    out_path = os.path.join(out_folder, out_file)
                    nd.write_pickle(lad_tour_props, out_path)

                if tfn_tour_props:
                    out_file = TFN_FNAME % (year, purpose, mode)
                    out_path = os.path.join(out_folder, out_file)
                    nd.write_pickle(tfn_tour_props, out_path)


if __name__ == '__main__':
    tms_tour_prop()
