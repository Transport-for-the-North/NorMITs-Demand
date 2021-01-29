# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 16:47:54 2020

@author: genie
"""
import os

import pandas as pd

trip_length_bands = pd.read_csv('Y:/NorMITs Synthesiser/import/trip_length_bands/single_value_bands_miles.csv')

_nts_import = pd.read_csv('Y:/NTS/import/classified_nts_pre-weighting.csv')

_nts_import['weighted_trip'] = _nts_import['W1'] * _nts_import['W5xHh'] * _nts_import['W2']

list(_nts_import)

# Pick segments
# _target_output_params = 'Y:/NorMITs Synthesiser/import/trip_length_bands/north_standard_segments.csv'
_target_output_params = 'Y:/NorMITs Synthesiser/import/trip_length_bands/north_all_mode_enhanced_segments.csv'

export = 'Y:/NorMITs Synthesiser/import/trip_length_bands/north/sandbox'

# gb or north
geo_area = 'north'

output_params = pd.read_csv(_target_output_params)

weekdays = [1,2,3,4,5]

agg_purp = [13, 14, 15, 18]

north_la = ['E06000001', 'E06000002', 'E06000003', 'E06000004', 'E06000005',
            'E06000006', 'E06000007', 'E06000008', 'E06000009', 'E06000010',
            'E06000011', 'E06000012', 'E06000013', 'E06000014', 'E06000021',
            'E06000047', 'E06000049', 'E06000050', 'E06000057', 'E07000026',
            'E07000027', 'E07000028', 'E07000029', 'E07000030', 'E07000031',
            'E07000033', 'E07000034', 'E07000035', 'E07000037', 'E07000038',
            'E07000117', 'E07000118', 'E07000119', 'E07000120', 'E07000121',
            'E07000122', 'E07000123', 'E07000124', 'E07000125', 'E07000126',
            'E07000127', 'E07000128', 'E07000137', 'E07000142', 'E07000163',
            'E07000164', 'E07000165', 'E07000166', 'E07000167', 'E07000168',
            'E07000169', 'E07000170', 'E07000171', 'E07000174', 'E07000175',
            'E07000198', 'E08000001', 'E08000002', 'E08000003', 'E08000004',
            'E08000005', 'E08000006', 'E08000007', 'E08000008', 'E08000009',
            'E08000010', 'E08000011', 'E08000012', 'E08000013', 'E08000014',
            'E08000015', 'E08000016', 'E08000017', 'E08000018', 'E08000019',
            'E08000021', 'E08000022', 'E08000023', 'E08000024', 'E08000032',
            'E08000033', 'E08000034', 'E08000035', 'E08000036', 'E08000037',
            'W06000001', 'W06000002', 'W06000003', 'W06000004', 'W06000005',
            'W06000006']

# TODO: Traveller gender
target_cols = ['SurveyYear', 'TravDay', 'HHoldOSLAUA_B01ID', 'soc_cat',
               'ns_sec', 'main_mode', 'hb_purpose', 'nhb_purpose', 'Sex_B01ID',
               'trip_origin', 'start_time', 'TripDisIncSW', 'TripOrigGOR_B02ID',
               'TripDestGOR_B02ID', 'tfn_area_type', 'weighted_trip']

output_dat = _nts_import.reindex(target_cols, axis=1)

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