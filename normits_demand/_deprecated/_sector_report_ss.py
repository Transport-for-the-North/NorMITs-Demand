# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 17:25:35 2020

@author: cruella
"""

import os

import pandas as pd

# TODO: Functionalise
write = True

sector_file = ('Y:/NorMITs Synthesiser/Norms_2015/iter2/Distribution Outputs/' + 
               'Logs & Reports/24hr_pa_distributions_all_distribution_sector_report.csv')

export_folder = ('Y:/NorMITs Synthesiser/Norms_2015/iter2/Distribution Outputs/Logs & Reports/Sector Reports/')

if not os.path.exists(export_folder):
    os.makedirs(export_folder)




sector_report = pd.read_csv(sector_file)

distinct_sector = sector_report['p_sector_name'].drop_duplicates()
distinct_sector_no = sector_report['p_sector'].drop_duplicates()
distinct_purpose = sector_report['purpose'].drop_duplicates()

#### Matrix builder
sector_report_agg_purp = sector_report.copy()
sector_report_agg_purp['p_sector_name'] = (sector_report_agg_purp['p_sector'].astype('str') + ' ' + sector_report_agg_purp['p_sector_name'])
sector_report_agg_purp['a_sector_name'] = (sector_report_agg_purp['a_sector'].astype('str') + ' ' + sector_report_agg_purp['a_sector_name'])

# Reduce mode to model segment
purpose_index = {'purpose':[1,2,3,4,5,6,7,8,12,13,14,15,16,18],
                 'model_purpose':['commute',
                                  'business',
                                  'other',
                                  'other',
                                  'other',
                                  'other',
                                  'other',
                                  'other',
                                  'business',
                                  'other',
                                  'other',
                                  'other',
                                  'other',
                                  'other']} 
# Create DataFrame 
purpose_index = pd.DataFrame(purpose_index) 

sector_report_agg_purp = sector_report_agg_purp.merge(purpose_index,
                                                      how = 'left',
                                                      on = 'purpose')

# Build distinct segments
mode_subset = ['6']
# TODO: This should be a parameter
distinct_params = ['mode', 'model_purpose', 'car_availability']

distinct_segments = sector_report_agg_purp.reindex(distinct_params, axis=1).drop_duplicates().reset_index(drop=True)

distinct_segments = distinct_segments[distinct_segments['mode'].isin(mode_subset)].reset_index(drop=True)



for index, segment in distinct_segments.iterrows():
    subset = sector_report_agg_purp.copy()
    segment_name_ph = []
    for s_index, seg in segment.iteritems():
        subset = subset[subset[s_index]==seg]
        segment_name_ph.append(s_index + '_' + str(seg)) 
    seg_name = ''
    for name in segment_name_ph:
        seg_name = seg_name + name + '_'

    subset = subset.reindex(['p_sector_name', 'a_sector_name', 'dt'],
                            axis=1).groupby(
                                    ['p_sector_name','a_sector_name']).sum(
                                            ).reset_index()
    subset = subset.pivot(index = 'p_sector_name',
                          columns = 'a_sector_name',
                          values = 'dt')
    if write:
        subset.to_csv((export_folder + '/' + seg_name + '.csv'), index=True)
    

####

for index, sector in distinct_sector.iteritems():
    
    print(sector)
    sector_no = distinct_sector_no[index]
    print(sector_no)
    # Do the total loop here
    full_subset = sector_report[sector_report['p_sector_name']==sector]
    for purpose in distinct_purpose:
        purpose_subset = full_subset[full_subset['purpose']==purpose]
        # Group and sum purpose_subset
        purpose_subset_output = purpose_subset.reindex(
                ['p_sector_name',
                 'a_sector_name',
                 'dt'],
                 axis=1).groupby(
                         ['p_sector_name',
                          'a_sector_name']).sum().reset_index()
        
        
        title_name = ('sector ' + str(sector_no) + ', ' + sector + ' purpose ' + str(purpose))
        export_name = ('sector' + str(sector_no) + '_purpose' + str(purpose))

    # Group and sum full subset
    
    full_title_name = ('sector ' + str(sector_no) + ', ' + sector)
    full_export_name = ('sector_' + str(sector_no))

    full_subset_output = full_subset.reindex(
            ['p_sector_name',
             'a_sector_name',
             'dt'],
             axis=1).groupby(
                     ['p_sector_name',
                      'a_sector_name']).sum().reset_index()
    # Graph
    
    # Export graph

"""
Somthing like:::
    
    
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="whitegrid")

# Initialize the matplotlib figure
f, ax = plt.subplots(figsize=(6, 15))

# Load the example car crash dataset
crashes = sns.load_dataset("car_crashes").sort_values("total", ascending=False)

# Plot the total crashes
sns.set_color_codes("pastel")
sns.barplot(x="total", y="abbrev", data=crashes,
            label="Total", color="b")

# Plot the crashes where alcohol was involved
sns.set_color_codes("muted")
sns.barplot(x="alcohol", y="abbrev", data=crashes,
            label="Alcohol-involved", color="b")

# Add a legend and informative axis label
ax.legend(ncol=2, loc="lower right", frameon=True)
ax.set(xlim=(0, 24), ylabel="",
       xlabel="Automobile collisions per billion miles")
sns.despine(left=True, bottom=True)

"""