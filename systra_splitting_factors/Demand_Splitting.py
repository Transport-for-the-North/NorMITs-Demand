# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 10:04:10 2020
@author: tyu2
File purpose: Trip Purpose and Mode Adjustment -- update time period splitting factor
Script version: 1.0
Python version: 3.7
"""

import pandas as pd
import os
import fnmatch
import numpy as np

if __name__ == "__main__":

    root_dir = os.getcwd()
    input_dir = os.path.join(root_dir, 'Inputs')
    output_dir = os.path.join(root_dir, 'Outputs')
    Params_dir = os.path.join(root_dir, 'Params')
    sec_area_zone = pd.read_csv(os.path.join(Params_dir, 'Sector_Area_Zones.csv'))
    splitting_factor = pd.read_csv(r'F:\EFS\Python\Params\Tour_Splitting_Factors.csv')
    #first 5 zones
    splitting_factor = splitting_factor.loc[splitting_factor['Origin'] < 6]
    splitting_factor = splitting_factor.loc[splitting_factor['Destination'] < 6]
    
    # setting file
    files_list = fnmatch.filter(os.listdir(input_dir), '*.csv')
    year_list = ['2018','2033','2035','2050']
    CA_list = ['car_availability1', 'car_availability2']
    mode_list = ['mode6', 'mode3']
    HB_purpose = [1, 2, 3]
    
    df = pd.DataFrame()
    # Loop over car availability, mode and forecast year
    for year in year_list:
        df_year = fnmatch.filter(files_list, '*'+ year +'*')
        for CA in CA_list:
            df_year_ca = fnmatch.filter(df_year, '*'+ CA +'*')
            for mode in mode_list:
                df_year_ca_mode = fnmatch.filter(df_year_ca, '*'+ mode +'*')
                
#                read_in_files()
                
#                group_purpose()
                
#                split_extrnal()
                
                for file in df_year_ca_mode:
                    dir = os.path.join(input_dir, file)
                    print('Reading {}'.format(dir))
                    df_to_append = pd.read_csv(dir)
                    if df.empty:
                        df = df_to_append
                    else:
                        df = df.append(df_to_append)
                        
     
                      
                        
                df = df.drop_duplicates().reset_index(drop=True)
                #first 5 zones
                df = df.loc[df['p_zone'] < 6]
                df = df.loc[df['a_zone'] < 6]
                # Aggregate the 8 purposes into 3 (1: HBW, 2: HBEB, 3: HBO)
                df.loc[df['purpose_id']==1,'Purpose'] = 1
                df.loc[df['purpose_id']==2,'Purpose'] = 2
                df['Purpose'] = df['Purpose'].fillna(3)
                grouped = df.groupby(['p_zone', 'a_zone', 'car_availability_id', 'mode_id', 'time_period_id', 'Purpose'])['dt'].sum().reset_index()
                
                
                
                # apply the sector_area factor to split internal and external demand
                int_ext_fac = pd.merge(grouped, sec_area_zone, how='left', left_on=['p_zone', 'a_zone'], right_on=['Origin', 'Destination'])
                Int_Dem = int_ext_fac[['p_zone', 'a_zone', 'car_availability_id', 'mode_id', 'Purpose', 'time_period_id', 'Factor']]                
                Int_Dem['dt'] = np.where(int_ext_fac['Factor'] != 5, 0, int_ext_fac['dt'])
                
                
                Int_Dem_tot = Int_Dem.groupby(['p_zone', 'a_zone', 'Purpose'])['dt'].sum().reset_index()
                # Calculate the proportion of demand for each period
                Int_Dem_per = Int_Dem.merge(Int_Dem_tot, how='left', on = ['p_zone', 'a_zone', 'Purpose'])
                Int_Dem_per['Dem'] = Int_Dem_per['dt_x'] / Int_Dem_per['dt_y']
                # Take the input splitting factors and calculate the from home totals  
                split_tot = splitting_factor.groupby(['Origin', 'Destination', 'Purpose', 'Car Availability', 'FHPeriod'])['Factor'].sum().reset_index() 
                # Calculate Adjustment Factor as Demand Proportion / Initial From-Home Proportion
                adj_fac = Int_Dem_per.merge(split_tot, how = 'left', left_on= ['p_zone', 'a_zone',  'Purpose', 'car_availability_id', 'time_period_id'], right_on = ['Origin', 'Destination', 'Purpose', 'Car Availability', 'FHPeriod'])
                adj_fac['adj_fac'] = adj_fac['Dem'] / adj_fac['Factor_y']
                # Apply adjustment factors to input splitting factors based on From-Home Period
                updated_splitting = adj_fac.merge(splitting_factor, how = 'left', on = ['Origin', 'Destination', 'Purpose', 'Car Availability', 'FHPeriod'])
                updated_splitting['updated_fac'] = updated_splitting['adj_fac'] * updated_splitting['Factor']
                updated_splitting = updated_splitting[['Origin', 'Destination', 'Purpose', 'Car Availability', 'FHPeriod', 'THPeriod', 'dt_x', 'Dem', 'adj_fac','updated_fac', 'Factor']]
#                check_adj = updated_splitting[['Origin', 'Destination', 'Purpose', 'FHPeriod', 'THPeriod', 'dt_x']]
                
                updated_splitting.to_csv(os.path.join(output_dir, year + '_' + CA + '_' + mode + '_' + 'Int_Dem_test.csv'), index = False)
                # External demand
                Ext_Dem = int_ext_fac[['p_zone', 'a_zone', 'car_availability_id', 'mode_id', 'time_period_id', 'Purpose', 'Factor']] 
                Ext_Dem['dt'] = np.where((int_ext_fac['Factor'] == 1) | (int_ext_fac['Factor'] == 5) | (int_ext_fac['Factor'] == 9), 0, int_ext_fac['dt'])               
                # output to .csv
#                Int_Dem.to_csv(os.path.join(output_dir, year + '_' + CA + '_' + mode + '_' + 'Int_Dem.csv'), index = False)
#                Ext_Dem.to_csv(os.path.join(output_dir, year + '_' + CA + '_' + mode + '_' + 'Ext_Dem.csv'), index = False)
               
