# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 19:34:03 2020

Script to build 24hr costs from a set of Noham 'Time' skims
Weighting based on CTripEnd time split parameters by purpose.

@author: cruella
"""
import os

import pandas as pd

from production_model import get_mode_time_splits
from production_model import aggregate_mode_time_splits

_import_folder = 'Y:/NorMITs Synthesiser/import/'

# Get time split factors
mts_path = _import_folder + 'IRHOmdhsr_FINAL.csv'

mode_time_split = get_mode_time_splits(mts_path)
# Optimise data types in mode time splits

# mode time split [0] is the full thing
# mode time split [1] is mode only
# Use 0 where possible!
mode_time_split = aggregate_mode_time_splits(mode_time_split[0])

# Filter to mode 3
rail_mode_time_split = mode_time_split[0].copy()
rail_mode_time_split = rail_mode_time_split[rail_mode_time_split['mode']==6]

commute_p = [1]
business_p = [2]
other_p = [3,4,5,6,7,8]

commute_time = rail_mode_time_split.copy()
commute_time = commute_time[commute_time['purpose'].isin(commute_p)]

business_time = rail_mode_time_split.copy()
business_time = business_time[business_time['purpose'].isin(business_p)]

other_time = rail_mode_time_split.copy()
other_time = other_time[other_time['purpose'].isin(other_p)]

time_ph = []
time_list = [{'Commute':commute_time},
             {'Business':business_time},
             {'Other':other_time}]

for tl in time_list:
    purpose = list(tl)
    ph = tl[purpose[0]].reindex(['time', 'trip_split'], axis=1)
    # Explicitly drop off peak
    ph = ph[ph['time']!=4]
    # Group and sum
    ph = ph.groupby('time').sum().reset_index()
    total_time = ph['trip_split'].sum()
    ph['time_split'] = ph['trip_split']/total_time
    ph = ph.reindex(['time', 'time_split'], axis=1)
    time_ph.append({purpose[0]:ph})
    del(ph, purpose)

# Set target dir for costs
cost_dir = 'T:/NoRMS_T3/Data/It2_CostSkims/Cost Outputs'

# Index directory
cost_files = os.listdir(cost_dir)
# Filter to time
cost_files = [x for x in cost_files if 'TP' in x]
# Filter to purpose
purpose_list = ['Commute', 'Business', 'Other']
p_ph = []
for p in purpose_list:
    p_ph.append([x for x in cost_files if p in x])
cost_files = [x for y in p_ph for x in y]

# Define user classes to loop over
target_uc = ['CA_from', 'NCA']

# Import & classify by purpose
outputs = []

for uc in target_uc:
    print('Compiling costs for ' + uc)
    uc_files = [x for x in cost_files if uc in x]

    for p in purpose_list:
        print('Compiling costs for ' + p)
        import_files = [x for x in uc_files if p in x]

        weighted_ph = []
        for imp in import_files:
            print(imp)
            if 'TP1' in imp:
                time = 1
                tp = 'tp1'
            elif 'TP2' in imp:
                time = 2
                tp = 'tp2'
            elif 'TP3' in imp:
                time = 3
                tp = 'tp3'
            else:
                # If it's not TP1-3, we don't want it
                break

            ph = pd.read_csv(cost_dir + '/' + imp)
            first_col_name = list(ph)[0]
            ph = pd.melt(ph, id_vars=first_col_name,
                         var_name='a_zone',
                         value_name=tp)
            ph = ph.rename(columns={first_col_name:'p_zone'})

            for item in time_ph:
                if list(item)[0] == p:
                    weight = item[list(item)[0]]
            weight = weight[weight['time']==time]
            weight = weight['time_split'].iloc[0]

            ph[tp] = ph[tp]*weight
            
            new_heading = (p + '_' + uc.replace('_from',''))

            weighted_ph.append(ph)

        purpose_output = weighted_ph[0].merge(weighted_ph[1],
                                    how = 'left',
                                    on = ['p_zone', 'a_zone'])
        purpose_output = purpose_output.merge(weighted_ph[2],
                                              how = 'left',
                                              on = ['p_zone', 'a_zone'])
        purpose_output[new_heading] = purpose_output['tp1'] + purpose_output['tp2'] + purpose_output['tp3']
        purpose_output = purpose_output.reindex(['p_zone','a_zone',new_heading],axis=1)
        outputs.append(purpose_output)
        del(weighted_ph, purpose_output)

final_costs = outputs[0].copy().merge(outputs[1],
                     how='left',
                     on =['p_zone', 'a_zone']).merge(
                             outputs[2],
                             how='left',
                             on=['p_zone', 'a_zone']).merge(
                                     outputs[3],
                                     how='left',
                                     on=['p_zone', 'a_zone']).merge(
                                             outputs[4],
                                             how='left',
                                             on=['p_zone', 'a_zone']).merge(
                                                     outputs[5],
                                                     how='left',
                                                     on=['p_zone', 'a_zone'])

for col in list(final_costs):
    final_costs = final_costs.rename(columns = {col:col.lower()})

final_costs.to_csv('Y:/NorMITs Synthesiser/Norms/Model Zone Lookups/norms_24hr_costs.csv',
                   index=False)
