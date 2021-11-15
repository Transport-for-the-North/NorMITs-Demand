# -*- coding: utf-8 -*-
"""
Created on: Wednesday October 20th 2021
Updated on:

Original author: Nirmal Kumar


File purpose:
TMS tour proportions Generation
"""

# Built-Ins
import os
import sys

# Third party imports
import pandas as pd

# Local Imports
sys.path.append("..")
import normits_demand as nd
from normits_demand.utils import pandas_utils as pd_utils
from normits_demand.utils import general as du
from normits_demand.core.data_structures import DVector

# ## GLOBALS ## #
modes = [3]
years = [2018]
zoning_system = "noham"

phi_import_folder = r"I:\NorMITs Demand\import\phi_factors"
notem_import_folder = r"I:\NorMITs Demand\NoTEM\iter4.2\SC01_JAM\hb_productions"
phi_fname = "mode_%d_fhp_tp_pa_to_od.csv"
prod_vec_fname = "hb_msoa_notem_segmented_%d_dvec.pkl"
zone_translate_dir = r"I:\NorMITs Demand\import\zone_translation\one_to_one"

out_fname = "hb_tour_proportions_yr%d_p%d_m%d.pkl"
# out_folder = r"I:\NorMITs Demand\import\noham\pre_me_tour_proportions\example_new"
out_folder = r'E:\test'


def tms_tour_prop():

    for year in years:
        for mode in modes:

            # ## GRAB PHI FACTORS BY MODE ## #
            phi_file = phi_fname % mode
            phi_df = pd.read_csv(os.path.join(phi_import_folder, phi_file))

            # Rename cols to match notem
            rename = {
                'purpose_from_home': 'p',
                'time_from_home': 'tp',
                'time_to_home': 'phi',
            }
            phi_df = phi_df.rename(columns=rename)

            # Drop unneeded tps
            mask = (
                phi_df['tp'].isin([1, 2, 3, 4])
                & phi_df['phi'].isin([1, 2, 3, 4])
            )
            phi_df = phi_df[mask].copy()

            # Pivot
            phi_df = phi_df.pivot(
                index=['p', 'tp'],
                columns=['phi'],
            ).reset_index()

            # ## GRAB TP SPLITS BY PURPOSE AND ZONE ## #
            notem_file = prod_vec_fname % year
            notem_dvec = nd.from_pickle(os.path.join(notem_import_folder, notem_file))

            # Convert to needed segments and translate
            week_seg = nd.get_segmentation_level("hb_p_tp_week")
            wday_seg = nd.get_segmentation_level("hb_p_tp_wday")
            zoning = nd.get_zoning_system(zoning_system)

            notem_dvec = notem_dvec.aggregate(week_seg)
            notem_dvec = notem_dvec.subset(wday_seg)
            notem_df = notem_dvec.translate_zoning(zoning, weighting="population").to_df()

            # Adjust factors back to 1
            notem_df['sum'] = notem_df.groupby([zoning.col_name, 'p'])['val'].transform('sum')
            notem_df['val'] /= notem_df['sum']
            notem_df = notem_df.drop(columns='sum')

            print(phi_df)
            print(notem_df)

            full_df = pd.merge(
                left=notem_df,
                right=phi_df,
                how='left',
                on=['p', 'tp']
            )

            print(full_df)



            exit()


            uniq = notem_df['uniq_id'].drop_duplicates().reset_index(drop=True)
            tp_split_dict = {}
            for q in uniq:
                p = int(q.split('_')[1])
                tp_sub_splits = pd_utils.filter_df(notem_df, {'uniq_id': q})
                # Remove tp5 and tp6
                tp_sub_splits = tp_sub_splits.head(-2)
                # Recalculate tp split
                tp_sub_splits['value1'] = (tp_sub_splits['val'] / tp_sub_splits['val'].sum())
                # Phi_factor * tp_split
                tp_split_dict[q] = (tp_sub_splits[['value1']].to_numpy()) * phi_dict[p]

            d = {}
            for p in purpose:
                p1 = '_' + str(p)
                d = {key: tp_split_dict[key] for key in tp_split_dict.keys() if p1 in key}
                print(d)

                out_file = out_fname % (year, p, mode)
                DVector.to_pickle(d, os.path.join(out_folder,out_file))


if __name__ == '__main__':
    tms_tour_prop()
