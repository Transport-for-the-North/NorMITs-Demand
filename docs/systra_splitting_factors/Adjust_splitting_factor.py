# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 10:04:10 2020

@author: tyu2
"""

import os
import fnmatch
import logging
from datetime import datetime
import pandas as pd

root_dir = os.getcwd()
input_dir = os.path.join(root_dir, 'Inputs')
output_dir = os.path.join(root_dir, 'Outputs')
params_dir = os.path.join(root_dir, 'Params')

if not os.path.exists(input_dir):
    print('----You need to make an Input folder in the root directory of this script and copy the EFS outputs inside.----')
  
logging.basicConfig(filename = output_dir + "runtime.log", format = '%(levelname)s:%(message)s', level = logging.INFO)
logging.info("*************************************************************************************") 
logging.info("Time Period Splitting Factor STARTED @ " + datetime.now().strftime("%d-%m-%Y,,,%H:%M:%S.%f")) 
logging.info("*************************************************************************************") 

logging.info("Read in files @ " + datetime.now().strftime("%d-%m-%Y,,,%H:%M:%S.%f")) 
sec_area_zone = pd.read_csv(os.path.join(params_dir, 'Sector_Area_Zones.csv'))
purpose_lookup = pd.read_csv(os.path.join(params_dir, 'Purpose_Lookup.csv'))
all_splitting_factors = pd.read_csv(os.path.join(params_dir, 'Tour_Splitting_Factors.csv'))

#split internal factor
logging.info("Merge sector area table @ " + datetime.now().strftime("%d-%m-%Y,,,%H:%M:%S.%f")) 
all_splitting_factors = all_splitting_factors.merge(sec_area_zone, how='left', on=['origin_id', 'destination_id'])
                
#5 zones for checking
#all_splitting_factors = all_splitting_factors.loc[(all_splitting_factors['origin_id'] > 1097) & (all_splitting_factors['origin_id'] < 1103)]
#all_splitting_factors = all_splitting_factors.loc[(all_splitting_factors['destination_id'] > 1097) & (all_splitting_factors['destination_id'] < 1103)]

# setting loop over lists
YearList = ['2018','2033','2035','2050']
CAList = [1,2]
# Rail for now     
ModeList = [6]
HBPurposeList = [1, 2, 3]
full_file_list = fnmatch.filter(os.listdir(input_dir), '*.csv')

logging.info("Set up functions @ " + datetime.now().strftime("%d-%m-%Y,,,%H:%M:%S.%f")) 
def read_in_files(file_list, input_dir):    
    df = pd.DataFrame()
    for file in file_list:
        dir = os.path.join(input_dir, file)
        print('Reading {}'.format(dir))
        df_to_append = pd.read_csv(dir)
        if df.empty:
            df = df_to_append
        else:
            df = df.append(df_to_append)
    df = df.drop_duplicates().reset_index(drop=True)           
    return df

def group_purpose(df, lookup_table):
    print('----Aggregating the 8 purposes into 3----')
    #first 5 zones
#    df = df.loc[(df['p_zone'] > 1097) & (df['p_zone'] < 1103)]
#    df = df.loc[(df['a_zone'] > 1097) & (df['a_zone'] < 1103)]
    # Aggregate the 8 purposes into 3 (1: HBW, 2: HBEB, 3: HBO)
    df = df.merge(lookup_table, how='left', on=['purpose_id'])
    df = df.groupby(['p_zone', 'a_zone', 'car_availability_id', 'mode_id', 'time_period_id', 'uc_id'])['dt'].sum().reset_index()
    return df

def extract_internal_demand(df, area_flag_df):
    # apply the sector flag file to split internal and external demand
    print('----Splitting External Demand----')
    df = df.merge(area_flag_df, how='left', left_on=['p_zone', 'a_zone'], right_on=['origin_id', 'destination_id'])
#    [['p_zone', 'a_zone', 'car_availability_id', 'mode_id', 'uc_id', 'time_period_id', 'area_flag', 'dt']] 
    df = df.loc[df['area_flag'] == 5]
#    df['dt'] = np.where(int_ext_fac['Factor'] != 5, 0, int_ext_fac['dt'])
    return df

def adjust_split_factor(df, splitting_factors_df):
    print('----Calculating New Splitting Factor----')
    # Calculate the proportion of demand for each period
    df = df.merge(df.groupby(['p_zone', 'a_zone', 'uc_id'])['dt'].sum().reset_index(), how='left', on = ['p_zone', 'a_zone', 'uc_id'])
    df['dem_proportion'] = df['dt_x'] / df['dt_y']
    # Take the input splitting factors and calculate the from home totals  
    all_splitting_factors_tot = splitting_factors_df.groupby(['origin_id', 'destination_id', 'purpose_id', 'car_availability_id', 'fh_period_id'])['factor'].sum().reset_index() 
    # Calculate Adjustment Factor as Demand Proportion / Initial From-Home Proportion
    df = df.merge(all_splitting_factors_tot, how = 'left', left_on= ['origin_id', 'destination_id',  'uc_id', 'car_availability_id', 'time_period_id'], 
                  right_on = ['origin_id', 'destination_id', 'purpose_id', 'car_availability_id', 'fh_period_id'])
    df['adj_fac'] = df['dem_proportion'] / df['factor']   
    # Apply adjustment factors to input splitting factors based on From-Home Period and recap internal demand
    df = splitting_factors_df.merge(df, how = 'left', on = ['origin_id', 'destination_id', 'purpose_id', 'car_availability_id', 'fh_period_id'])
    df = df[['origin_id', 'destination_id', 'purpose_id', 'car_availability_id', 'mode_id', 'fh_period_id', 'th_period_id', 'factor_x', 'factor_y', 'adj_fac']].rename(columns = {
            'factor_x': 'original_splitting_factor', 'factor_y': 'demand_splitting_factor', 'adj_fac': 'adjustment_factor'})
   
    # Fill the nan with 1 to keep the unadjustable splitting factor 
    df['adjustment_factor'] = df['adjustment_factor'].fillna(1)
    df['factor'] = df['adjustment_factor'] * df['original_splitting_factor']
    df['t'] = df['fh_period_id'] * 10 + df['th_period_id']
    return df

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)    

if __name__ == "__main__":
          
    logging.info("Start of main code @ " + datetime.now().strftime("%d-%m-%Y,,,%H:%M:%S.%f")) 
    ensure_dir(output_dir)
    # Loop over car availability, mode and forecast year
    for year in YearList:
        for CA in CAList:
            # Filter splititing factor by CA & internal             
            current_splitting_factors= all_splitting_factors.loc[(all_splitting_factors['car_availability_id'] == CA) & (all_splitting_factors['area_flag'] == 5)]
            for mode in ModeList:
                logging.info(" Year " + year + " Car Avail " + str(CA) + " Mode " + str(mode) + " @ " + datetime.now().strftime("%d-%m-%Y,,,%H:%M:%S.%f")) 
                current_file_list = fnmatch.filter(full_file_list,'*'+ year +'*'+'*car_availability'+ str(CA) +'*'+'*mode'+ str(mode) +'*')

                logging.info("        Read in files @ " + datetime.now().strftime("%d-%m-%Y,,,%H:%M:%S.%f")) 
                input_demand = read_in_files(current_file_list, input_dir)
                
                logging.info("        Group uc_ids @ " + datetime.now().strftime("%d-%m-%Y,,,%H:%M:%S.%f")) 
                grouped = group_purpose(input_demand, purpose_lookup)
                
                logging.info("        Split internal/external @ " + datetime.now().strftime("%d-%m-%Y,,,%H:%M:%S.%f")) 
                internal_demand = extract_internal_demand(grouped, sec_area_zone)
                
                logging.info("        Adjust spliting factors @ " + datetime.now().strftime("%d-%m-%Y,,,%H:%M:%S.%f")) 
                updated_splitting_factors = adjust_split_factor(internal_demand, current_splitting_factors)
                
                # Save splitting factor by uc_id
                for purpose in HBPurposeList:
                    logging.info("          Purpose " + str(purpose) + " @ " + datetime.now().strftime("%d-%m-%Y,,,%H:%M:%S.%f")) 

                    output_splitting_factors  = updated_splitting_factors.loc[updated_splitting_factors['purpose_id'] == purpose][['origin_id', 'destination_id', 't', 'factor']] 
                    print('-- Saving results to ' + output_dir)    
                    logging.info("            Output to csv @ " + datetime.now().strftime("%d-%m-%Y,,,%H:%M:%S.%f")) 
                    output_splitting_factors .to_csv(os.path.join(output_dir, year + '_car_availability' + str(CA) + '_mode' + str(mode) + '_Splitting_factor_purpose' + str(purpose) +'.csv'), index = False)
                
#                # External demand
#                Ext_Dem = int_ext_fac[['p_zone', 'a_zone', 'car_availability_id', 'mode_id', 'time_period_id', 'uc_id', 'Factor']] 
#                Ext_Dem['dt'] = np.where((int_ext_fac['Factor'] == 1) | (int_ext_fac['Factor'] == 5) | (int_ext_fac['Factor'] == 9), 0, int_ext_fac['dt'])               
#                # output to .csv
##                Int_Dem.to_csv(os.path.join(output_dir, year + '_' + CA + '_' + mode + '_' + 'Int_Dem.csv'), index = False)
##                Ext_Dem.to_csv(os.path.join(output_dir, year + '_' + CA + '_' + mode + '_' + 'Ext_Dem.csv'), index = False)
#               
logging.info("Time Period Splitting Factor FINISHED @ " + datetime.now().strftime("%d-%m-%Y,,,%H:%M:%S.%f")) 
