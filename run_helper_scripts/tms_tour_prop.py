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

# Third party imports
import pandas as pd

# Local Imports
import normits_demand as nd
from normits_demand.utils import pandas_utils as pd_utils
from normits_demand.utils import general as du
from normits_demand.core.data_structures import DVector

## GLOBALS ##
modes = [3]
years = [2018]
phi_import_folder = r"I:\NorMITs Demand\import\phi_factors"
notem_import_folder = r"I:\NorMITs Demand\NoTEM\iter4.2\SC01_JAM\hb_productions"
phi_fname = "mode_%d_fhp_tp_pa_to_od.csv"
prod_vec_fname = "hb_msoa_notem_segmented_%d_dvec.pkl"
zoning_system = "noham"
zone_translate_dir = r"I:\NorMITs Demand\import\zone_translation\one_to_one"


out_fname = "hb_tour_proportions_yr%d_p%d_m%d.pkl"
out_folder = r"I:\NorMITs Demand\import\noham\pre_me_tour_proportions\example_new"


def tms_tour_prop():
    for year in years:
        for mode in modes:
            # Read phi factors for every mode
            phi_file = phi_fname % mode
            phi_df = pd.read_csv(os.path.join(phi_import_folder, phi_file))

            purpose = phi_df['purpose_from_home'].drop_duplicates().reset_index(drop=True)
            phi_dict = {}
            # Convert the phi factors to the required format for each purpose
            for p in purpose:
                phi_sub_df = pd_utils.filter_df(phi_df, {'purpose_from_home': p})
                phi_pivot = pd.pivot_table(phi_sub_df, values='direction_factor', index='time_from_home',
                                           columns='time_to_home', aggfunc='sum')
                phi_pivot = phi_pivot.head(-2)
                phi_pivot = phi_pivot.drop(phi_pivot.columns[[4, 5]], axis=1)
                phi_dict[p] = phi_pivot.to_numpy()

            print(phi_dict)
            ## TP Splits##

            notem_file = prod_vec_fname % year
            # Convert it to the required segmentation
            return_seg = nd.get_segmentation_level("hb_p_6tp")
            # Read the production vector
            notem_dvec = nd.from_pickle(os.path.join(notem_import_folder,notem_file))
            # Aggregate to the required segmentation
            notem_dvec = notem_dvec.aggregate(return_seg)

            zon_sys = nd.get_zoning_system(zoning_system)
            notem_df = notem_dvec.translate_zoning(new_zoning=zon_sys, weighting="population").to_df()
            zon_col = "%s_zone_id" % zoning_system
            notem_df['uniq_id'] = pd_utils.str_join_cols(notem_df, [zon_col, 'p'])
            print(notem_df)
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
