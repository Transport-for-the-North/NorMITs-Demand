# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 11:50:10 2020

@author: tyu2
"""

# import sys

import os

from typing import List

import numpy as np
import pandas as pd

import pa_to_od as pa2od
from demand_utilities import tms_utils as dut

# distribution files location
home_path = 'Y:/EFS/'
lookup_path = os.path.join(home_path, 'import')
import_path = os.path.join(home_path, 'inputs/default/tp_pa')
pa_export_path = "C:/Users/Sneezy/Desktop/NorMITs_Demand/nhb_dev"
nhb_production_location = os.path.join(home_path, 'inputs/distributions')

model_name = 'norms'


# needed lists
purposes_needed = [
            1,
            # 2,
            # 3,
            # 4,
            # 5,
            # 6,
            # # 7,
            # 8,
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
years_needed = [
            # 2018,
            2033,
            ]


def _build_tp_pa_internal(trip_origin,
                          year,
                          purpose,
                          mode,
                          segment,
                          car_availability,
                          model_zone,
                          tp_split,
                          output_dir):
    """
    The internals of build_tp_pa(). Useful for making the code more
    readable due to the number of nested loops needed

    Returns
    -------

    """
    # Init return value
    matrix_totals_dictionary = dict()

    # ## Read in productions ## #
    productions_fname = get_dist_name(
        'hb',
        'pa',
        str(year),
        str(purpose),
        str(mode),
        str(segment),
        str(car_availability),
        csv=True
    )
    productions = pd.read_csv(os.path.join(
        pa_export_path,
        "24hr PA Matrices",
        productions_fname
    ))

    y_zone = 'a_zone' if model_zone == 'p_zone' else 'd_zone'
    productions = productions.melt(
        id_vars=[model_zone],
        var_name=y_zone,
        value_name='trips'
    )

    # ## Add in production columns ready for merge ## #
    productions['purpose_id'] = purpose
    productions['mode_id'] = mode
    productions['car_availability_id'] = car_availability
    if purpose in [1, 2]:
        productions['soc_id'] = str(segment)
        productions['ns_id'] = 'none'
    else:
        productions['soc_id'] = 'none'
        productions['ns_id'] = str(segment)

    # ## Narrow tp_split down to just the segment here ## #
    segment_id = 'soc_id' if purpose in [1, 2] else 'ns_id'
    segmentation_mask = (
        (tp_split['purpose_id'] == purpose)
        & (tp_split['mode_id'] == mode)
        & (tp_split[segment_id] == str(segment))
        & (tp_split['car_availability_id'] == car_availability)
    )
    tp_split = tp_split.loc[segmentation_mask]

    # ## Calculate the time split factors for each zone ## #
    # Total tp-split productions in each zone
    tp_totals = tp_split.reindex(
        [model_zone, 'tp', 'trips'],
        axis=1
    ).groupby([model_zone, 'tp']).sum().reset_index()

    # Calculate tp-split factors
    unq_zone = tp_totals[model_zone].drop_duplicates()
    for zone in unq_zone:
        zone_mask = (tp_totals[model_zone] == zone)
        tp_totals.loc[zone_mask, 'time_split'] = (
            tp_totals[zone_mask]['trips'].values
            /
            tp_totals[zone_mask]['trips'].sum()
        )
    time_splits = tp_totals.reindex(
        [model_zone, 'tp', 'time_split'],
        axis=1
    )

    # ## Apply tp-split factors to total productions ## #
    unq_time = time_splits['tp'].drop_duplicates()
    for time in unq_time:
        time_factors = time_splits.loc[time_splits['tp'] == time]
        gb_tp = pd.merge(
            productions,
            time_factors,
            on=[model_zone]
        ).rename(columns={'trips': 'dt'})

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

        # Build write path
        tp_pa_name = get_dist_name(
            str(trip_origin),
            'pa',
            str(year),
            str(purpose),
            str(mode),
            str(segment),
            str(car_availability),
            tp=str(time)
        )
        tp_pa_fname = tp_pa_name + '.csv'
        out_tp_pa_path = os.path.join(
            output_dir,
            tp_pa_fname
        )

        # Convert table from long to wide format
        # And save
        gb_tp.pivot_table(
            index='p_zone',
            columns='a_zone',
            values='dt'
        ).to_csv(out_tp_pa_path)

        matrix_totals_dictionary[tp_pa_name] = gb_tp

    return matrix_totals_dictionary


def build_tp_pa(pa_import,
                pa_export,                
                required_purposes,
                required_modes,
                required_soc,
                required_ns,
                required_car_availabilities,
                year_string_list):

    # loop Init
    matrix_totals_dictionary = {}
    out_tp_pa_dir = os.path.join(pa_export, 'PA Matrices')
    dut.create_folder(out_tp_pa_dir, chDir=False)

    # For every: Year, purpose, mode, segment, ca
    for year in year_string_list:
        print("\nYear: %s" % str(year))
        for purpose in required_purposes:

            # Purpose specific set-up
            # Do it here to avoid repeats in inner loops
            if purpose in (12, 13, 14, 15, 16, 18):
                # TODO: How to allocate tp to NHB
                print('NHB run')
                trip_origin = 'nhb'
                required_segments = list()
                model_zone = 'o_zone'
                tp_split_fname = 'export_nhb_productions_norms.csv'
                tp_split_path = os.path.join(pa_import, tp_split_fname)

            elif purpose in (1, 2, 3, 4, 5, 6, 7, 8):
                print('HB run')
                trip_origin = 'hb'
                model_zone = 'p_zone'
                tp_split_fname = 'export_productions_norms.csv'
                tp_split_path = os.path.join(pa_import, tp_split_fname)
                if purpose in [1, 2]:
                    required_segments = required_soc
                else:
                    required_segments = required_ns

            else:
                raise ValueError("%s is not a valid purpose."
                                 % str(purpose))

            # TODO: @Chris: is this the correct time split file to use?
            #  Should use TMS base year tp PA as seed?
            #  For example - TMS pa_to_od uses:
            #  Y:\NorMITs Synthesiser\Noham\iter8a\Production Outputs/hb_productions_noham.csv

            # Read in the seed values for tp splits
            tp_split = pd.read_csv(tp_split_path).rename(
                columns={
                    'norms_zone_id': 'p_zone',
                    'p': 'purpose_id',
                    'm': 'mode_id',
                    'soc': 'soc_id',
                    'ns': 'ns_id',
                    'ca': 'car_availability_id'
                }
            )
            tp_split['p_zone'] = tp_split['p_zone'].astype(int)

            matrix_totals_dictionary = dict()
            for mode in required_modes:
                print("\tMode: %s" % str(mode))
                for segment in required_segments:
                    print("\t\tSegment: %s" % str(segment))
                    for car_availability in required_car_availabilities:
                        matrix_totals = _build_tp_pa_internal(
                            trip_origin,
                            year,
                            purpose,
                            mode,
                            segment,
                            car_availability,
                            model_zone,
                            tp_split,
                            out_tp_pa_dir
                        )

                        matrix_totals_dictionary.update(matrix_totals)

    return matrix_totals_dictionary


def build_od(pa_matrix_dictionary,
             lookup_folder,
             od_export,
             aggregate_to_wday=True):
    """
    This function imports time period split factors from a given path.W
    """
    period_time_splits = pd.read_csv(os.path.join(lookup_folder,
                                                  "IphiHDHD_Final.csv"))

    # TODO:
    #  6_fhp_tp
    #  drop weekend tps (5/6)

    # Audit new totals
    if aggregate_to_wday:
        # TODO: This could be a lot easier

        # Define target times
        target_times = [1, 2, 3, 4]

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
                time_sub = time_sub[time_sub['time_from_home'] == tfh]
                time_sub = time_sub[time_sub['time_to_home'].isin(target_times)]
                
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
    wday_from_totals = period_time_splits.reindex(from_cols, axis=1)
    from_cols.remove('direction_factor')
    wday_from_totals = wday_from_totals.groupby(
            from_cols).sum().reset_index()

    # TODO: Proper error handle
    print('From-To split factors - should return 1s or conversion will' +
          ' drop trips')
    print(wday_from_totals['direction_factor'].drop_duplicates())
    
    # read in pa matrix
    for key, pa_matrix in pa_matrix_dictionary.items():
            
        od_matrix = pd.merge(
            pa_matrix,
            period_time_splits,
            left_on=["tp", "purpose_id"],
            right_on=["time_from_home", "purpose_from_home"]
        )
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
        ])["dt", "demand_to_home"].sum().reset_index().rename(
            columns={"dt": "demand_from_home"}
        )

        # Convert the name from PA to OD
        name_parts = get_dist_name_parts(key)
        name_parts[1] = 'od'
        tp_od_name = get_dist_name(*name_parts, csv=True)

        od_matrix.to_csv(os.path.join(od_export, tp_od_name), index=False)
        print("HB OD for " + tp_od_name + " complete!")


def nhb_production_dataframe(required_purposes,
                             required_soc,
                             required_ns,
                             required_car_availabilities,
                             year_string_list,
                             nhb_production_file_location,
                             lookup_folder,
                             ):
    """
    This function builds NHB productions by
    aggregates HB distribution from EFS output to destination 

    Parameters
    ----------
    required lists:
        to loop over TfN segments

    Returns
    ----------
    nhb_production_dictionary:
        Dictionary containing NHB productions by year
    """
    
    required_segments = []
    nhb_production_dataframe_list = []
    nhb_production_dictionary = {}
    #read in nhb trip rates
    nhb_trip_rate_dataframe = pd.read_csv(
        os.path.join(lookup_folder, "IgammaNMHM.csv")
    ).rename(
        columns={
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


# TODO: Import from original efs
def get_dist_name(trip_origin: str,
                  matrix_format: str,
                  year: str,
                  purpose: str,
                  mode: str,
                  segment: str,
                  car_availability: str,
                  tp: str = None,
                  csv: bool = False
                  ) -> str:
    """
    Generates the distribution name
    """
    seg_name = "soc" if purpose in ['1', '2'] else "ns"

    name_parts = [
        trip_origin,
        matrix_format,
        "yr" + year,
        "p" + purpose,
        "m" + mode,
        seg_name + segment,
        "ca" + car_availability
    ]

    if tp is not None:
        name_parts += ["tp" + tp]

    final_name = '_'.join(name_parts)
    if csv:
        final_name += '.csv'

    return final_name


# TODO: copy over to original EFS
def get_dist_name_parts(dist_name: str) -> List[str]:
    """
    Splits a full dist name into its individual components


    Parameters
    ----------
    dist_name:
        The dist name to parse

    Returns
    -------
    name_parts:
        dist_name split into parts. Returns in the following order:
        [trip_origin, matrix_format, year, purpose, mode, segment, ca, tp]
    """
    if dist_name[-4:] == '.csv':
        dist_name = dist_name[:-4]

    name_parts = dist_name.split('_')

    # TODO: Can this be done smarter
    return [
        name_parts[0],
        name_parts[1],
        name_parts[2][-4:],
        name_parts[3][-1:],
        name_parts[4][-1:],
        name_parts[5][-1:],
        name_parts[6][-1:],
        name_parts[7][-1:],
    ]


def main():
    # init_params = get_init_params(
    #     import_path,
    #     model_name = model_name,
    #     distribution_type = 'hb',
    #     mode_subset = None,
    #     purpose_subset = None)

    # TODO: Integrate into TMS
    # output = pa2od.build_tp_pa(
    #     file_drive='Y:/',
    #     model_name='norms',
    #     iteration='iter1',
    #     distribution_segments=['p', 'm'],
    #     internal_input='efs',
    #     external_input='efs',
    #     normits_tool='demand',
    #     write_modes=[1, 2, 3, 5, 6],
    #     arrivals=False,
    #     export_24hr=False,
    #     arrival_export=None,
    #     write=True
    # )

    matrix_totals = build_tp_pa(
        required_purposes=purposes_needed,
        required_modes=modes_needed,
        required_soc=soc_needed,
        required_ns=ns_needed,
        required_car_availabilities=car_availabilities_needed,
        year_string_list=years_needed,
        pa_import=import_path,
        pa_export=pa_export_path
    )
    print('Transposed HB PA to tp PA')

    build_od(
        pa_matrix_dictionary=matrix_totals,
        lookup_folder=lookup_path,
        od_export=os.path.join(pa_export_path, "OD Matrices"),
        aggregate_to_wday=True
    )
    print('Transposed HB tp PA to OD')

    nhb_production_dictionary = nhb_production_dataframe(
        # nhb_trip_rate_dataframe = nhb_trip_rate_file,
        required_purposes=purposes_needed,
        required_soc=soc_needed,
        required_ns=ns_needed,
        required_car_availabilities=car_availabilities_needed,
        year_string_list=years_needed,
        # distribution_dataframe_dict = distributions,
        nhb_production_file_location=os.path.join(pa_export_path, "forArrivals"),
        lookup_folder=lookup_path
    )
    print('Generated NHB productions')

    nhb_furness(
        production_dictionary=nhb_production_dictionary,
        required_nhb_purposes=nhb_purpose_needed,
        required_nhb_modes=nhb_mode_needed,
        required_car_availabilities=car_availabilities_needed,
        required_soc=soc_needed,
        required_ns=ns_needed,
        year_string_list=years_needed,
        nhb_distribution_file_location=nhb_production_location,
        nhb_distribution_output_location=os.path.join(pa_export_path, " 24hr OD Matrices"),
        zero_replacement_value=0.01,
        replace_zero_values=True
    )
    print("Furnessed NHB Productions")


if __name__ == '__main__':
    main()
