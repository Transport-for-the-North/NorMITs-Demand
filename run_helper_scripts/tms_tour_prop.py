# -*- coding: utf-8 -*-
"""
Created on: Wednesday October 20th 2021
Updated on:

Original author: Nirmal Kumar


File purpose:
TMS tour proportions Generation
"""

# Third party imports
import os.path

import pandas as pd

import normits_demand as nd
from normits_demand.utils import pandas_utils as pd_utils
from normits_demand.utils import file_ops
from normits_demand.core.data_structures import DVector

modes = [3, 5, 6]
years = [2018]
phi_import_folder = r"I:\NorMITs Demand\import\phi_factors"
notem_import_folder = r"I:\NorMITs Demand\NoTEM\iter4.2\SC01_JAM\hb_productions"
phi_fname = "mode_%d_fhp_tp_pa_to_od"
prod_vec_fname = "hb_msoa_notem_segmented_%d_dvec.pkl"

phi_factors = {
    'noham': r"I:\NorMITs Demand\import\phi_factors\mode_3_fhp_tp_pa_to_od.csv",
    'norms': r"I:\NorMITs Demand\import\phi_factors\mode_6_fhp_tp_pa_to_od.csv"
}
out_fname = "hb_tour_proportions_yr2018%s_m%d.pkl"
out_folder = r"I:\NorMITs Demand\import\noham\pre_me_tour_proportions\example"


def tms_tour_prop():
    for year in years:
        for mode in modes:
            phi_file = phi_fname % mode
            # phi_path = os.path.join(phi_import_folder, phi_file)
            phi_df = pd.read_csv(os.path.join(phi_import_folder, phi_file))

            purpose = phi_df['purpose_from_home'].drop_duplicates().reset_index(drop=True)
            phi_dict = {}

            for p in purpose:
                phi_sub_df = pd_utils.filter_df(phi_df, {'purpose_from_home': p})
                phi_pivot = pd.pivot_table(phi_sub_df, values='direction_factor', index='time_from_home',
                                           columns='time_to_home', aggfunc='sum')
                phi_pivot = phi_pivot.head(-2)
                phi_pivot = phi_pivot.drop(phi_pivot.columns[[4, 5]], axis=1)
                phi_dict[p] = phi_pivot.to_numpy()

            print(phi_dict)

    # notem = r"I:\NorMITs Demand\NoTEM\iter4.2\SC01_JAM\hb_productions\hb_msoa_notem_segmented_2018_dvec.pkl"
            notem_file = prod_vec_fname % year
            return_seg = nd.get_segmentation_level("hb_p_6tp")
            notem_dvec = nd.from_pickle(os.path.join(notem_import_folder,notem_file))
            notem_dvec = notem_dvec.aggregate(return_seg)
            zon_sys = nd.get_zoning_system('noham')
            notem_df = notem_dvec.translate_zoning(new_zoning=zon_sys, weighting="population").to_df()
            notem_df['uniq_id'] = pd_utils.str_join_cols(notem_df, ['noham_zone_id', 'p'])
            print(notem_df)
            uniq = notem_df['uniq_id'].drop_duplicates().reset_index(drop=True)
            tp_split_dict = {}
            for q in uniq:
                p = int(q.split('_')[1])
                tp_sub_splits = pd_utils.filter_df(notem_df, {'uniq_id': q})
                tp_sub_splits = tp_sub_splits.head(-2)
                tp_sub_splits['value1'] = (tp_sub_splits['val'] / tp_sub_splits['val'].sum())
                tp_split_dict[q] = (tp_sub_splits[['value1']].to_numpy()) * phi_dict[p]

            for p in purpose:
                p1 = '_' + str(p)
                d = {key: tp_split_dict[key] for key in tp_split_dict.keys() if p1 in key}
                out_file = out_fname % (p1, year)
                DVector.to_pickle(d, os.path.join(out_folder,out_file))


if __name__ == '__main__':
    tms_tour_prop()
