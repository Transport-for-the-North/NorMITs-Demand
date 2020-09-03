# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 11:50:10 2020

@author: tyu2
"""

# import os
# import sys

import numpy as np
import pandas as pd

from demand_utilities import tms_utils as dut

# distribution files location
home_path = 'Y:/EFS/'
lookup_path = home_path + 'inputs/default/import/'
import_path = home_path + 'inputs/default/tp_pa/'
pa_export_path = "C:/Users/Sneezy/Desktop/NorMITs_Demand/"
nhb_production_location = home_path + 'inputs/distributions/'

model_name = 'norms'


# needed lists
purposes_needed = [
            1,
            2,
            3,
            4,
            5,
            6,
            # 7,
            8,
            ]
modes_needed = [
            1,
            2,
            3,
            5,
            6,
            ]
soc_needed = [
            0,
            1,
            2,
            3,
            ]
ns_needed = [
            1,
            2,
            3,
            4,
            5,                   
            ]
nhb_purpose_needed = [
            12,
            13,
            14,
            15,
            16,
            18,
            ]
nhb_mode_needed = [
            # 1,
            # 2,
            # 3,
            # 4,
            # 5,
            6,
            ]
car_availabilities_needed = [
            1,
            2,
            ]
year_string_list = [
            # 2018,
            2033,
            ]

# def get_init_params(path,
#                     distribution_type = 'hb',
#                     model_name = None,
#                     mode_subset = None,
#                     purpose_subset = None):
#     """
#     This function imports beta values for deriving distributions from
#     a given path. Chunk exists as a filename in the target folder.

#     Parameters
#     ----------
#     path:
#         Path to folder containing containing required beta values.

#     distribution_type = hb:
#         Distribution type. Takes hb or nhb.

#     model_name:
#         Name of model. For pathing to new lookups.

#     chunk:
#         Number of chunk to take if importing by chunks. This is designed to
#         make it easy to multi-process in future, and can be used to run
#         distributions in parralell IDEs now.

#     Returns:
#     ----------
#     initial_betas:
#         DataFrame containing target betas for distibution.
#     """

#     if model_name is None:
#         path = (path +
#                 # '/' +
#                 'init_params_' +
#                 distribution_type +
#                 '.csv')
#     else:
#         path = (path +
#                 # '/' +
#                 model_name.lower() +
#                 '_init_params_' +
#                 distribution_type +
#                 '.csv')

#     init_params = pd.read_csv(path)

#     if mode_subset:
#         init_params = init_params[
#                 init_params['m'].isin(mode_subset)]
#     if purpose_subset:
#         init_params = init_params[
#                 init_params['p'].isin(purpose_subset)]

#     return(init_params)

def build_tp_pa(
        # init_params,
                model_name,
                # productions,
                pa_import,
                pa_export,                
                required_purposes,
                required_modes,
                required_soc,
                required_ns,
                required_car_availabilities,
                year_string_list,
                # distribution_segments,
                # internal_import,
                # external_import,
                write_modes = [1,2,3,5,6],
                # arrivals = True,
                # arrival_export = None,
                write = True):
    
    matrix_totals_dictionary = {}
    
    for year in year_string_list:
        for purpose in required_purposes:
            # TODO: How to allocate tp to NHB            
            if purpose in (12, 13, 14, 15, 16, 18):
                trip_origin = 'nhb'
                print('NHB run')
                model_zone = 'o_zone'
                tp_split = pd.read_csv(
                    pa_import + 'export_nhb_productions_norms.csv'
                ).rename({
                    'norms_zone_id': 'p_zone',
                    'p': 'purpose_id',
                    'm': 'mode_id',
                    'soc': 'soc_id',
                    'ns': 'ns_id',
                    'ca': 'car_availability_id'
                    })
            elif purpose in (1,2):
                required_segments = soc_needed
                segment_text = "_soc"
                
                trip_origin = 'hb'
                print('HB run')
                tp_split = pd.read_csv(
                            pa_import + 'export_productions_norms.csv'
                            ).rename(columns = {
                                    'norms_zone_id':'p_zone',
                                    'p': 'purpose_id',
                                    'm': 'mode_id',
                                    'soc': 'soc_id',
                                    'ns': 'ns_id',
                                    'ca': 'car_availability_id'
                                        }
                                    )
                model_zone = 'p_zone'                
            else:                                
                required_segments = ns_needed
                segment_text = "_ns"
                trip_origin = 'hb'
                print('HB run')
                tp_split = pd.read_csv(
                    pa_import + 'export_productions_norms.csv'
                ).rename(
                    columns={
                        'purpose': 'purpose_id',
                        'mode': 'mode_id',
                        'employment_type': 'soc_id',
                        'age': 'ns_id',
                        'car_availability': 'car_availability_id'}
                )
                model_zone = 'p_zone'                

            for mode in required_modes:                
                for segment in required_segments:                
                    for car_availability in required_car_availabilities:
                        productions = pd.read_csv(pa_export_path + "24hr PA Matrices/"
                                                  + 
                                                  "hb_pa"
                                                  +
                                                  "_yr"
                                                  +
                                                  str(year)
                                                  +
                                                  "_p"
                                                  +
                                                  str(purpose)
                                                  +
                                                  "_m"
                                                  +
                                                  str(mode)
                                                  +
                                                  segment_text
                                                  +
                                                  str(segment)
                                                  +
                                                  "_ca"
                                                  +
                                                  str(car_availability)
                                                  +
                                                  ".csv"
                                                  ) 
   
                        tp_pa_path = (pa_export + 'PA Matrices/' + trip_origin + '_pa_yr' + str(year))
                    
                        matrix_totals = []
                        compile_params = {}
                        #     for ds in distribution_segments:
                        #         calib_params.update({ds:init_params[ds][tp_pa]})
                        #         print(calib_params)
                        tp_split['p_zone'] = tp_split['p_zone'].astype(int)
                        productions['purpose_id'] = purpose
                        productions['mode_id'] = mode
                        if purpose in [1, 2]:
                            productions['soc_id'] = segment
                            productions['ns_id'] = 'none'
                        else:
                            productions['soc_id'] = 'none'
                            productions['ns_id'] = segment
                        
                        print(tp_split)
                        print()
                        print(productions)
                        print()
                        p_subset = tp_split.copy()
                        p_subset = productions.merge(
                            p_subset,
                            on=['p_zone',
                                'purpose_id',
                                'mode_id',
                                'soc_id',
                                'ns_id',
                                'car_availability_id']
                        )

                        # This won't work if there are duplicates
                        p_totals = p_subset.reindex(
                            [model_zone, 'trips'], axis=1).groupby(
                                model_zone).sum().reset_index()
                        p_totals = p_totals.rename(columns={'trips':'p_totals'})
                        tp_totals = p_subset.reindex(
                            [model_zone,
                             'tp',
                             'trips'], axis=1).groupby(
                                 [model_zone, 'tp']).sum().reset_index()
                        time_splits = tp_totals.merge(p_totals,
                                                      how='left',
                                                      on=[model_zone])
                        time_splits['time_split'] = (time_splits['trips']/
                                    time_splits['p_totals'])
                        time_splits = time_splits.drop(['p_totals'], axis=1)
                        
                        # Apply time period
                        unq_time = time_splits['tp'].drop_duplicates()           
                        for time in unq_time:
                            print('tp' + str(time))
                            time_factors = time_splits.loc[time_splits['tp'] == time]           
                            gb_tp = productions.merge(time_factors,
                                                            # how='left',
                                                      on=[model_zone])
                            gb_tp['dt'] = gb_tp['dt'] * gb_tp['time_split']
                            gb_tp = gb_tp.groupby([
                                "p_zone",
                                "a_zone",
                                "purpose_id",
                                "car_availability_id",
                                "mode_id",
                                "soc_id",
                                "ns_id",
                                "tp"
                                ])["dt"].sum().reset_index()
                                # productions = productions.drop([model_zone,'trips','time_split'], axis=1)
                                
                                # gb_tp = all_zone_ph * time_factors
                            # compile_params.update({'gb_tp':gb_tp.sum()})    
                    
                                # if arrivals:
                                #     arrivals_np = gb_tp.sum(axis=0)
                                #     arrivals_mat = pd.DataFrame(all_zone_ph[model_zone])
                                #     arrivals_mat['arrivals'] = arrivals_np
                                    
                                #     arrivals_write_path = nup.build_path(arrivals_path,
                                #                                          calib_params,
                                #                                          tp=time)
                    
                            # Build write paths
                            out_tp_pa_path = tp_pa_path + (
                                                '_p' + str(purpose)
                                                +
                                                '_m' + str(mode)
                                                +
                                                str(segment_text) + str(segment) 
                                                + 
                                                '_ca' + str(car_availability)
                                                )
                            out_tp_pa = out_tp_pa_path
                            if time:
                                out_tp_pa_path += ('_tp' + str(time))
                            out_tp_pa_path += '.csv'
                    
                            print(out_tp_pa_path)
                    
                            compile_params.update({'hb_export_path':out_tp_pa_path})
                    
                            if write:
                                # Define write path
                                if mode in write_modes:
                                        # productions = pd.DataFrame(productions,
                                        #                      index=all_zone_ph['p_zone'],
                                        #                      columns=all_zone_ph[
                                        #                              'p_zone']).reset_index()
                    
                                    gb_tp.to_csv(out_tp_pa_path,
                                                      index=False)
                                    
                                    # if arrivals:
                                    #     # Write arrivals anyway
                                    #     arrivals_mat.to_csv(arrivals_write_path,
                                    #                         index=False)
                            matrix_totals.append(gb_tp)                           
                            gb_tp = gb_tp.iloc[0:0]
                            
                            matrix_totals_dictionary[out_tp_pa] = matrix_totals                           
           
    # matrix_totals_dict = matrix_totals
    return(matrix_totals_dictionary)

def build_od(pa_matrix_dictionary,
             lookup_folder,
             od_export,
             aggregate_to_wday = True
                           ):
    """
    This function imports time period split factors from a given path.

    Parameters
    ----------
    mode:
        Target mode as single integer

    phi_type:
        Takes one of ['fhp_tp', 'fhp_24hr' 'p_tp']. From home purpose & time period
        or from home and to home purpose & time period

    aggregate_to_wday:
        Boolean to aggregate to weekday or not.

    Returns:
    ----------
    period_time_splits:
        DataFrame containing time split factors for pa to od.
    """
    period_time_splits = pd.read_csv(lookup_folder + "IphiHDHD_Final.csv")
    # Audit new totals
    if aggregate_to_wday:
        # TODO: This could be a lot easier

        # Define target times
        target_times = [1,2,3,4]

        # Filter time from home to target
        period_time_splits = period_time_splits[
                period_time_splits[
                        'time_from_home'].isin(target_times)]
        period_time_splits = period_time_splits[
                period_time_splits[
                        'time_to_home'].isin(target_times)]

        # Different methods depending on the phi type
        # If it's to home purpose only, split on from home col only
        unq_combo = period_time_splits.reindex(
                ['purpose_from_home'], axis=1).drop_duplicates(
                        ).reset_index(drop=True)

        # Define period time split placeholder
        pts_ph = []

        # Loop to do the factor
        # Loop over purpose
        for c_index, combo in unq_combo.iterrows():    
            purpose_frame = period_time_splits.copy()
            for param in combo.index:
                # Subset down
                purpose_frame = purpose_frame[
                        purpose_frame[param] == combo[param]]

            # Get unq times from home
            unq_tfh = purpose_frame[
                    'time_from_home'].drop_duplicates(
                    ).reset_index(drop=True)

            # Placeholder for consolidated time
            ctph = []
            # Loop to get new factors
            for tfh in unq_tfh:
                time_sub = purpose_frame.copy()
                time_sub = time_sub[
                        time_sub[
                                'time_from_home']==tfh]
                time_sub = time_sub[
                        time_sub['time_to_home'].isin(target_times)]
                
                new_total = time_sub['direction_factor'].sum()
                
                time_sub['direction_factor'] = time_sub[
                        'direction_factor']/new_total
                
                ctph.append(time_sub)

            purpose_frame = pd.concat(ctph, sort=True)
            pts_ph.append(purpose_frame)
        
        # Compile
        period_time_splits = pd.concat(pts_ph, sort=True)

    # Audit new totals
    from_cols = ['purpose_from_home', 'time_from_home', 'direction_factor']
    wday_from_totals = period_time_splits.reindex(from_cols,axis=1)
    from_cols.remove('direction_factor')
    wday_from_totals = wday_from_totals.groupby(
            from_cols).sum().reset_index()

    # TODO: Proper error handle
    print('From-To split factors - should return 1s or conversion will' +
          ' drop trips')
    print(wday_from_totals['direction_factor'].drop_duplicates())
    
    #read in pa matrix
    for key in pa_matrix_dictionary:
        pa_matrix_list = pa_matrix_dictionary[key]
        for pa_matrix in pa_matrix_list:
            
            od_matrix = pa_matrix.merge(period_time_splits, 
                                        left_on=["tp", "purpose_id"],
                                        right_on=["time_from_home", "purpose_from_home"])
            od_matrix["demand_to_home"] = od_matrix["dt"] * od_matrix["direction_factor"]
            od_matrix = od_matrix.groupby([
                                "p_zone",
                                "a_zone",
                                "purpose_id",
                                "car_availability_id",
                                "mode_id",
                                "soc_id",
                                "ns_id",
                                "purpose_from_home",
                                "time_from_home",
                                "purpose_to_home",
                                "time_to_home"                                
                                ])["dt","demand_to_home"].sum().reset_index().rename(columns={                                    
                                    "dt": "demand_from_home"
                                    })
            out_od_matrix = key[-21:]                        
            od_matrix.to_csv(od_export + "hb_od_" + out_od_matrix, index=False)
            print("HB OD for " + out_od_matrix + " complete!")
                                    
                                
    return()

def nhb_production_dataframe(
                         required_purposes,
                         required_soc,
                         required_ns,
                         required_car_availabilities,
                         year_string_list,
                         nhb_production_file_location,
                         ):
    """
    This function builds NHB productions by
    aggregates HB distribution from EFS output to destination 

    Parameters
    ----------
    required lists:
        to loop over TfN segments

    Returns:
    ----------
    nhb_production_dictionary:
        Dictionary containing NHB productions by year
    """
    
    required_segments = []
    nhb_production_dataframe_list = []
    nhb_production_dictionary = {}
    #read in nhb trip rates
    nhb_trip_rate_dataframe = pd.read_csv(nhb_production_file_location + "IgammaNMHM.csv").rename(
                    columns = {
                            "p": "purpose_id",
                            "m": "mode_id"
                            }
                    )
    # loop over by year, purpose, soc/ns, car availability
    for year in year_string_list:
        for purpose in required_purposes:                
            if purpose in (1,2):
                required_segments = required_soc
                segment_text = "_soc"
            else:
                required_segments = required_ns
                segment_text = "_ns"
            for segment in required_segments:                
                for car_availability in required_car_availabilities:
                    nhb_production_dist = (
                    "hb_pa_yr" 
                    + 
                    str(year) 
                    + 
                    "_p" 
                    + 
                    str(purpose) 
                    + 
                    segment_text 
                    + 
                    str(segment)
                    +
                    "_ca"
                    +
                    str(car_availability) 
                    + 
                    ".csv"
                    )
                    #read in HB distributions
                    nhb_production_dataframe = pd.read_csv(nhb_production_file_location 
                                                             + 
                                                             nhb_production_dist                                                              
                                                             )
                    # Aggregates to destinations                                                            
                    nhb_production_dataframe = nhb_production_dataframe.groupby([
                        "a_zone",
                        "purpose_id",
                        "mode_id",
                        "car_availability_id",
                        "soc_id",
                        "ns_id"
                        ])["dt"].sum().reset_index()
                    # join nhb trip rates
                    nhb_production_dataframe = nhb_trip_rate_dataframe.merge(
                        nhb_production_dataframe,                    
                    # nhb_production_dataframe = nhb_production_dataframe.merge(
                    #     nhb_trip_rate_dataframe, 
                        on = [
                            "purpose_id",
                            "mode_id"                         
                            ]
                        )
                    # Calculate NHB productions
                    nhb_production_dataframe["nhb_dt"] = nhb_production_dataframe["dt"] * nhb_production_dataframe["nhb_trip_rate"]

                    # aggregate nhb_p 11_12    
                    nhb_production_dataframe.loc[nhb_production_dataframe["nhb_p"]==11, "nhb_p"] = 12
                    # change = nhb_production_dataframe[nhb_production_dataframe['nhb_p']==11].copy()
                    # change['nhb_p'] = 12
                    # no_change = nhb_production_dataframe[nhb_production_dataframe['nhb_p']!=11].copy()
                    # nhb_production_dataframe = pd.concat([change, no_change], sort=True)
                    nhb_production_dataframe = nhb_production_dataframe.groupby([
                        "a_zone",
                        "nhb_p",
                        "nhb_m",
                        "car_availability_id",
                        "soc_id",
                        "ns_id"
                        ])["nhb_dt"].sum().reset_index()

                    nhb_production_dataframe_list.append(nhb_production_dataframe)
                    print("NHB Productions for " + nhb_production_dist + " complete!")
        # concatenate purpose, soc/ns, car availability 
        nhb_productions_all = pd.concat(nhb_production_dataframe_list)
        nhb_production_dataframe_list.clear()
        # aggregate to mode and purpose
        nhb_productions_all = nhb_productions_all.groupby(["a_zone", "nhb_p", "nhb_m"])["nhb_dt"].sum().reset_index()
        # save to dictionary by year
        nhb_production_dictionary[year] = nhb_productions_all
        nhb_productions_all = nhb_productions_all.iloc[0:0]
   
    return(nhb_production_dictionary)
                        
def nhb_furness(
            production_dictionary,                    
            required_nhb_purposes,
            required_nhb_modes,
            required_car_availabilities,
            required_soc,
            required_ns, 
            year_string_list,
            nhb_distribution_file_location,
            nhb_distribution_output_location,
            replace_zero_values,
            zero_replacement_value,
            ):
    
    """
    Provides a one-iteration Furness constrained on production
    with options whether to replace zero values on the seed

    Parameters
    ----------
    nhb_production_dictionary:
        Dictionary containing NHB productions by year
    
    required lists:
        to loop over TfN segments

    Output:
    ----------
    final_nhb_distribution with the columns
    "o_zone", "d_zone", "nhb_p", "nhb_m",
    "car_availability_id", "soc_id", "ns_id", "dt"

    """
      
    for year in year_string_list:
        for purpose in required_nhb_purposes:
            for mode in required_nhb_modes:
                # select needed nhb_production
                nhb_production = production_dictionary[year]
                nhb_production = nhb_production.loc[nhb_production["nhb_p"] == purpose]
                nhb_production = nhb_production.loc[nhb_production["nhb_m"] == mode]
                
                # read in nhb_dist
                nhb_dist = pd.read_csv(
                    nhb_distribution_file_location
                    +
                    ("24hr PA Matrices/nhb_pa_p%s_m%s.csv" %
                     (str(purpose), str(mode)))
                )

                # convert from wide to long format
                nhb_dist = nhb_dist.melt(
                    id_vars=['o_zone'],
                    var_name='d_zone', value_name='seed_values'
                )
    
                nhb_dist['d_zone'] = nhb_dist['d_zone'].astype(int)
                 
                # TODO
                #  @@MSP / TY - NEED TO REMOVE FROM FINAL VERSION!!
                nhb_dist = nhb_dist[nhb_dist['o_zone'].isin([259, 267, 268, 270, 275, 1171, 1173])]
                nhb_dist = nhb_dist[nhb_dist['d_zone'].isin([259, 267, 268, 270, 275, 1171, 1173])]
                 
                # set distribution zones for checks             
                nhb_distribution_zones = set(nhb_dist["o_zone"].tolist())

                # set production zones for checks                          
                nhb_production_zones = set(nhb_production["a_zone"].tolist())
               
                if replace_zero_values:
                    # fill zero values
                    nhb_dist.loc[nhb_dist["seed_values"] == 0, "seed_values"] = \
                        zero_replacement_value
                
                # divide seed values by total on p_zone to get percentages
                nhb_dist_total = nhb_dist.groupby(
                    "o_zone"
                )["seed_values"].sum().reset_index().rename(
                    columns={"seed_values": "seed_total"}
                )
                nhb_dist = nhb_dist.merge(nhb_dist_total, on="o_zone")
                nhb_dist["seed_values"] = nhb_dist["seed_values"] / nhb_dist["seed_total"]
                     
                # for zone in nhb_distribution_zones:
                #     # divide seed values by total on p_zone to get
                #     # percentages
                #     zone_mask = (nhb_dist["o_zone"] == zone)
                #     nhb_dist.loc[
                #             zone_mask,
                #             "seed_values"
                #         ] = (
                #             nhb_dist[
                #                     zone_mask
                #                     ]["seed_values"].values
                #             /
                #             nhb_dist[
                #                     zone_mask
                #                     ]["seed_values"].sum()
                #             )
                 # grouped = nhb_dist.groupby("o_zone")["seed_values"].sum().reset_index()
                            
                nhb_furnessed_frame = nhb_dist.merge(
                    nhb_production, left_on="o_zone", right_on="a_zone")
                # calculate NHB distribution  
                nhb_furnessed_frame["dt"] = nhb_furnessed_frame["seed_values"] * nhb_furnessed_frame["nhb_dt"]
                
                # output by year, purpose, mode, ca, soc/ns
                for mode in required_nhb_modes:
                    for car_availability in required_car_availabilities:
                        # loop over socs                                                 
                        for soc in required_soc:
                            final_nhb_distribution = nhb_furnessed_frame[
                                       (nhb_furnessed_frame["nhb_m"] == mode)
                                       &
                                       (nhb_furnessed_frame["car_availability_id"] == car_availability)
                                       &
                                       (nhb_furnessed_frame["soc_id"] == soc)
                                       ][
                                            [
                                                "o_zone",
                                                "d_zone",
                                                "nhb_p",
                                                "nhb_m",
                                                "soc_id",
                                                "ns_id",
                                                "car_availability_id",
                                                "dt"
                                                ]
                                            ]
                            final_nhb_distribution_dict = (
                                        "nhb_od"
                                         +
                                         "_yr"
                                         +
                                         str(year)
                                         +
                                         "_p"
                                         +
                                         str(purpose)                                    
                                         +
                                         "_m"
                                         +
                                         str(mode)
                                         +                                                                     
                                         "_soc"
                                         +
                                         str(soc)
                                         +                                    
                                         "_ca"
                                         +
                                         str(car_availability)
                                         +
                                         ".csv"
                                         ) 
                            final_nhb_distribution.to_csv(nhb_distribution_output_location + final_nhb_distribution_dict, index=False) 
                            print(("NHB Distribution " + final_nhb_distribution_dict + " complete!"))
                         # loop over ns    
                        for ns in required_ns:
                            final_nhb_distribution = nhb_furnessed_frame[
                                             (nhb_furnessed_frame["nhb_m"] == mode)
                                             &
                                             (nhb_furnessed_frame["car_availability_id"] == car_availability)
                                             &
                                             (nhb_furnessed_frame["ns_id"] == ns)
                                         ][
                                              [
                                                  "o_zone",
                                                  "d_zone",
                                                  "nhb_p",
                                                  "nhb_m",
                                                  "car_availability_id",
                                                  "soc_id",
                                                  "ns_id",
                                                  "dt"
                                                  ]
                                            ]
                            final_nhb_distribution_dict = (
                                         "nhb_od"
                                          +
                                          "_yr"
                                          +
                                          str(year)
                                          +
                                          "_p"
                                          +
                                          str(purpose)                                    
                                          +
                                          "_m"
                                          +
                                          str(mode)
                                          +                                                                     
                                          "_ns"
                                          +
                                          str(ns)
                                          +                                    
                                          "_ca"
                                          +
                                          str(car_availability)
                                          +
                                          ".csv"
                                          )                           
                            final_nhb_distribution.to_csv(nhb_distribution_output_location + final_nhb_distribution_dict, index=False)
                            print(("NHB Distribution " + final_nhb_distribution_dict + " complete!"))


def main():
    # init_params = get_init_params(
    #     import_path,
    #     model_name = model_name,
    #     distribution_type = 'hb',
    #     mode_subset = None,
    #     purpose_subset = None)

    matrix_totals = build_tp_pa(
        # init_params,
        model_name,
        # productions = pa_distribution,
        required_purposes = purposes_needed,
        required_modes = modes_needed,
        required_soc = soc_needed,
        required_ns = ns_needed,
        required_car_availabilities = car_availabilities_needed,
        year_string_list = year_string_list,
        # distribution_segments = ['p', 'm'],
        # internal_import = hb_export_path + 'summaries',
        # external_import = hb_export_path + 'external',
        pa_import = import_path,
        pa_export = pa_export_path,
        write_modes = [1,2,3,5,6],
        # arrivals = True,
        # arrival_export = (hb_export_path + 'arrival_export' + '/arrivals'),
        write = True)
    print('Transposed HB PA to tp PA')

    phi_factors = build_od(
        pa_matrix_dictionary = matrix_totals,
        lookup_folder = lookup_path,
        od_export =pa_export_path + "OD Matrices/",
        aggregate_to_wday = True
        )

    nhb_production_dictionary = nhb_production_dataframe(
        # nhb_trip_rate_dataframe = nhb_trip_rate_file,
        required_purposes = purposes_needed,
        required_soc = soc_needed,
        required_ns = ns_needed,
        required_car_availabilities = car_availabilities_needed,
        year_string_list = year_string_list,
        # distribution_dataframe_dict = distributions,
        nhb_production_file_location =pa_export_path + "forArrivals/",
        )

    final_nhb_distribution = nhb_furness(
        production_dictionary =  nhb_production_dictionary,
        required_nhb_purposes = nhb_purpose_needed,
        required_nhb_modes = nhb_mode_needed,
        required_car_availabilities = car_availabilities_needed,
        required_soc = soc_needed,
        required_ns = ns_needed,
        year_string_list = year_string_list,
        nhb_distribution_file_location = nhb_production_location,
        nhb_distribution_output_location =pa_export_path + "OD Matrices 24hr/",
        zero_replacement_value = 0.01,
        replace_zero_values = True
        )


if __name__ == '__main__':
    main()
