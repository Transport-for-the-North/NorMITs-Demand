# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 10:01:48 2022

@author: mishtaiwi1
"""

import os

import pandas as pd
from tqdm import tqdm
import shutil
import logging
from datetime import datetime
import c01_ExportCubeFunctions as pFunc
pd.options.mode.chained_assignment = None  # default='warn'


''' >>> USER INPUTS ---'''
#
#Path to location of the NoRMS Cube Voyager Executable (use '/' insetad of '\\' or r'..)
cube_exe = 'C:/Program Files/Citilabs/CubeVoyager/VOYAGER.EXE'
#Path to location of the NoRMS Cube Catalog (use '/' insetad of '\\' or r'..)
cat_path = 'C:/Work/NorMITs/NorTMS_T3_Model_v8.16b'
#Path to location of the NoRMS Base Run (use '/' insetad of '\\' or r'..)
run_path = 'C:/Work/NorMITs/NorTMS_T3_Model_v8.16b/Runs'
#Base Run ID
run = 'ILP_2018'
#Path to Lookups folder (use '/' insetad of '\\' or r'..)
lookups_path = 'C:/Work/NorMITs/inputs'
#Path to location of Where Outputs to be Saved (use '/' insetad of '\\' or r'..
out_path = 'C:/Work/NorMITs/outputs'
#
''' >>> END OF USER INPUTS'''


'''Process Fixed objects'''
#journey purposes
purposes = {'EB':1, 'Com':2, 'Oth':3}
#time periods
periods = ['AM', 'IP', 'PM', 'OP']



#create new logfile
if os.path.exists(f'{out_path}/Export_BaseMatrices_Logfile.Log'):
    os.remove(f'{out_path}/Export_BaseMatrices_Logfile.Log')
logging.basicConfig(filename = f'{out_path}/Export_BaseMatrices_Logfile.Log', format = '%(levelname)s:%(message)s', level = logging.INFO)
logging.info("######################################################################################")
logging.info("Started Process @ " + datetime.now().strftime("%d-%m-%Y,,,%H:%M:%S.%f"))
logging.info("######################################################################################")


#copy Cube files
for period in periods:
    #read distance matrix
    pFunc.CheckFileExists(f'{run_path}/{run}/Outputs/BaseAssign/{period}_stn2stn_costs.csv')
    shutil.copy2(f'{run_path}/{run}/Outputs/BaseAssign/{period}_stn2stn_costs.csv',
                 f'{out_path}/{period}_stn2stn_costs.csv')
    #read iRSj props
    pFunc.CheckFileExists(f'{run_path}/{run}/Outputs/BaseAssign/{period}_iRSj_probabilities.h5')
    shutil.copy2(f'{run_path}/{run}/Outputs/BaseAssign/{period}_iRSj_probabilities.h5',
                 f'{out_path}/{period}_iRSj_probabilities.h5')
    
    logging.info(f'Distance and Probability matrices for period {period} has been copied')


#produce TLC lookup
pFunc.CheckFileExists(f'{lookups_path}/TLC_Overwrite.csv')
tlc_overwrite = pd.read_csv(f'{lookups_path}/TLC_Overwrite.csv')
stnsTLC = pFunc.StnZone2StnTLC(f'{run_path}/{run}/Inputs/Network/Station_Connectors.csv',
                                f'{run_path}/{run}/Inputs/Network/TfN_Rail_Nodes.csv',
                                f'{run_path}/{run}/Inputs/Network/External_Station_Nodes.csv',
                                tlc_overwrite)
#write TLC Lookup
stnsTLC.to_csv(f'{out_path}/TLCs.csv', index=False)

#PT Demand to time periods F/T
pFunc.PTDemandFromTo(cube_exe, cat_path, run_path + '/' + run, out_path)
logging.info('NoRMS matrices converted to OMX successfully')


#export to CSVs
for period in tqdm(periods, desc='Time Periods Loop ', unit= 'Period'):
    pFunc.ExportMat2CSVViaOMX(cube_exe, out_path + f'/PT_{period}.MAT',
                                      out_path, f'{period}', f'{period}')
    logging.info(f'{period} NoRMS matrices exported to CSVs')


logging.info("######################################################################################")
logging.info('Process Finished Successfully')
logging.info("######################################################################################")