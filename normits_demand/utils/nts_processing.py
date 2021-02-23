# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 16:47:54 2020

@author: genie
"""
import os

import pandas as pd

import normits_demand.constants as con

# BACKLOG: This is useful enough to deserve a proper wrapper that indexes the various params as options.

trip_length_bands = pd.read_csv('I:/NorMITs Synthesiser/import/trip_length_bands/default_bands_miles.csv')

_nts_import = pd.read_csv('Y:/NTS/import/classified_nts_pre-weighting.csv')

_nts_import['weighted_trip'] = _nts_import['W1'] * _nts_import['W5xHh'] * _nts_import['W2']

list(_nts_import)

# Pick segments
# _target_output_params = 'Y:/NorMITs Synthesiser/import/trip_length_bands/north_standard_segments.csv'
_target_output_params = 'I:/NorMITs Synthesiser/import/trip_length_bands/gb_standard_plus_ca_segments.csv'



# gb or north
if 'gb_' in _target_output_params:
    geo_area = 'gb'
elif 'north_' in _target_output_params:
    geo_area = 'north'

export = os.path.join(
    'I:/NorMITs Synthesiser/import/trip_length_bands/',
    geo_area,
    'sandbox')

output_params = pd.read_csv(_target_output_params)

weekdays = [1,2,3,4,5]

agg_purp = [13, 14, 15, 18]

north_la = con.NORTH_LA

# TODO: Traveller gender
target_cols = ['SurveyYear', 'TravDay', 'HHoldOSLAUA_B01ID', 'CarAccess_B01ID', 'soc_cat',
               'ns_sec', 'main_mode', 'hb_purpose', 'nhb_purpose', 'Sex_B01ID',
               'trip_origin', 'start_time', 'TripDisIncSW', 'TripOrigGOR_B02ID',
               'TripDestGOR_B02ID', 'tfn_area_type', 'weighted_trip']

output_dat = _nts_import.reindex(target_cols, axis=1)

# CA Map
"""
1	Main driver of company car
2	Other main driver
3	Not main driver of household car
4	Household car but non driver
5	Driver but no car
6	Non driver and no car
7	NA
"""
ca_map = pd.DataFrame({'CarAccess_B01ID':[1, 2, 3, 4, 5, 6, 7],
                       'ca':[2, 2, 2, 2, 1, 1, 1]})

output_dat = output_dat.merge(ca_map,
                              how='left',
                              on='CarAccess_B01ID')

# Aggregate area type application
agg_at = pd.DataFrame({'tfn_area_type':[1,2,3,4,5,6,7,8],
                       'agg_tfn_area_type':[1,1,2,2,3,3,4,4]}) # Check data and revisit

output_dat = output_dat.merge(agg_at,
                              how='left',
                              on='tfn_area_type')

output_dat = output_dat[output_dat['TravDay'].isin(weekdays)].reset_index(drop=True)

# Geo filter
if geo_area == 'north':
    output_dat = output_dat[output_dat['HHoldOSLAUA_B01ID'].isin(north_la)].reset_index(drop=True)

out_mat = []

for index, row in output_params.iterrows():

    print(row)
    op_sub = output_dat.copy()

    # Seed values so they can go MIA
    trip_origin, purpose, mode, tp, soc, ns, tfn_at, agg_at, g = [0,0,0,0,0,
                                                                  0,0,0,0]
    # TODO: Use nones to bypass some of these as required - or else it'll fail except on full seg
    for subset, value in row.iteritems():
        if subset == 'trip_origin':
            op_sub = op_sub[op_sub['trip_origin']==value].reset_index(drop=True)
            trip_origin = value
        if subset == 'p':
            if trip_origin == 'hb':
                op_sub = op_sub[op_sub['hb_purpose']==value].reset_index(drop=True)
            elif trip_origin == 'nhb':
                op_sub = op_sub[op_sub['nhb_purpose']==value].reset_index(drop=True)
            purpose = value
        if subset == 'ca':
            if value != 0:
                op_sub = op_sub[op_sub['ca']==value].reset_index(drop=True)
            ca = value
        if subset == 'm':
            if value != 0:
                op_sub = op_sub[op_sub['main_mode']==value].reset_index(drop=True)
            mode = value
        if subset == 'tp':
            tp = value
            if value != 0:
                # Filter around tp to aggregate
                time_vec = []
                time_vec.append(value)
                
                if purpose in agg_purp:
                    time_vec = [3,4]
                op_sub = op_sub[op_sub['start_time'].isin(time_vec)].reset_index(drop=True)
        if subset == 'soc':
            soc = value
            if value != 0:
                op_sub = op_sub[op_sub['soc_cat']==value].reset_index(drop=True)
        if subset == 'ns':
            ns = value
            if value != 0:
                op_sub = op_sub[op_sub['ns_sec']==value].reset_index(drop=True)
        if subset == 'tfn_area_type':
            tfn_at = value
            if value != 0:
                op_sub = op_sub[op_sub['tfn_area_type']==value].reset_index(drop=True)
        if subset == 'agg_tfn_area_type':
            agg_at = value
            if value != 0:
                op_sub = op_sub[op_sub['agg_tfn_area_type']==value].reset_index(drop=True)
        if subset == 'g':
            g = value
            if value != 0:
                op_sub = op_sub[op_sub['Sex_B01ID']==value].reset_index(drop=True)
    out = trip_length_bands.copy()

    out['ave_km'] = 0
    out['trips'] = 0

    for line, thres in trip_length_bands.iterrows():

        tlb_sub = op_sub.copy()

        lower = thres['lower']
        upper = thres['upper']
        
        tlb_sub = tlb_sub[tlb_sub['TripDisIncSW']>=lower].reset_index(drop=True)
        tlb_sub = tlb_sub[tlb_sub['TripDisIncSW']<upper].reset_index(drop=True)
        
        mean_val = tlb_sub['TripDisIncSW'].mean() * 1.61
        total_trips = tlb_sub['weighted_trip'].sum()

        out['ave_km'].loc[line] = mean_val
        out['trips'].loc[line] = total_trips

        del(mean_val)
        del(total_trips)
    
    out['band_share'] = out['trips']/out['trips'].sum()
    
    name = (trip_origin + '_tlb' + '_p' + str(purpose)+ '_m' + str(mode))
    if ca != 0:
        name = name + '_ca' + str(ca)
    if tfn_at != 0:
        name = name + '_at' + str(tfn_at)
    if agg_at != 0:
        name = name + '_aat' + str(agg_at)
    if tp != 0:
        name = name + '_tp' + str(tp)
    if soc != 0 or (purpose in [1,2,12] and 'soc' in list(output_params)):
        name = name + '_soc' + str(soc)
    if ns != 0 and 'ns' in list(output_params):
        name = name + '_ns' + str(ns)
    if g != 0:
        name = name + '_g' + str(g)
    name += '.csv'
    print(name)

    ex_name = os.path.join(export, name)

    out.to_csv(ex_name, index = False)

    out['mode'] = mode
    out['period'] = tp
    out['ca'] = ca
    out['purpose'] = purpose
    out['soc'] = soc
    out['ns'] = ns
    out['tfn_area_type'] = tfn_at
    out['agg_tfn_area_type'] = agg_at
    out['g'] = g

    out_mat.append(out)

final = pd.concat(out_mat)

full_name = os.path.join(export, 'full_export.csv')

final.to_csv(full_name, index=False)