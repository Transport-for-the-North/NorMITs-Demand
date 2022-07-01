# -*- coding: utf-8 -*-
"""
Created on: 26/08/2021
Updated on:

Original author: Nhan Nguyen
Last update made by: Ben Taylor
Other updates made by:

File purpose:
Used to extract data from a NoRMS run to produce a .pkl of tour proportions
Output dictionary is in the format of
level1: ['hbeb', 'hbw', 'hbo', 'nhbeb', 'nhbo', 'ex_eb', 'ex_hbw', 'ex_oth']
level2: ['ca', 'nca']] External = ['ca_fh', 'ca_th', 'nca']
level3: [11, 12, 13, 14, 21, 22, 23, 24, 31, 32, 33, 34, 41, 42, 43, 44]
        where the first number is the From-home tp, and the second number is the
        to-home time period.
        nhbeb and nhbo and external Only have [1, 2, 3, 4] keys at this
        level! NHB splitting factors.
Values: 1300 * 1300 matrix of tour proportion values for the given purpose,
        car_availability, and from_home/to-home segment.

WARNING: Need an instance of CUBE installed in order for this to work.
"""
# Built-Ins
import os
import multiprocessing as mp
import pickle as pk

# Third Party
import tqdm
import pandas as pd
import numpy as np
import subprocess

# Local Imports
from normits_demand import logging as nd_log

# BACKLOG: Rewrite all of CUBE to CSV code to be more generic
#  label: rewrite

LOG = nd_log.get_logger(__name__)

# GLOBAL VARIABLES
SOURCE_DIRECTORY = r'E:\temp\cube\2f ILF 2018\source - test'
OUTPUT_PATH = os.path.join(SOURCE_DIRECTORY, "NoRMS_Tour_Prop.pkl")

### SUBPROCESS - SINGLE ###
def procSingle(cmdList):
    for ts in cmdList:
        pr =subprocess.Popen(ts,creationflags=subprocess.CREATE_NEW_CONSOLE,shell=True)
        pr.wait()

### TEST CUBE VOYAGER ###
def cubeTEST(appCube,appPath,appName='cube_test'):
    appPath = appPath.replace('/','\\')
    toWrite = []
    toWrite.append('RUN PGM=MATRIX')
    toWrite.append('FILEO PRINTO[1]="'+appPath+'\\'+appName+'.prn"')
    toWrite.append('')
    toWrite.append('zones=1')
    toWrite.append("print csv=t list='CUBE software found' printo=1")
    toWrite.append('ENDRUN')
    with open(appPath+'\\'+appName+'.s','w') as script:
        for line in toWrite:
            print(line,file=script)
    procSingle(['del "'+appPath+'\\'+appName+'.prn"','"'+appCube+'" "'+appPath+'\\'+appName+'.s" -Pvdmi /Start /Hide /HideScript'])
    yesCube = True if os.path.isfile(appPath+'\\'+appName+'.prn') else False
    procSingle(['del "'+appPath+'\\'+appName+'.prn"','del "'+appPath+'\\'+appName+'.s"','del "'+appPath+'\\vdmi*.prn"','del "'+appPath+'\\vdmi*.var"'])
    return yesCube

### CONVERT MAT TO CSV ###
def cubeMAT2CSV(appCube,appPath,appName,appExtn,matFile,matFact,matTabx,lokPath,lokName,lokExtn,cubSATW,delFile=0):
    appPath = appPath.replace('/','\\')
    matFile = matFile.replace('/','\\')
    lokPath = lokPath.replace('/','\\')
    cubSATW = cubSATW.split(':')
    toWrite = []
    toWrite.append('RUN PGM=MATRIX')
    toWrite.append('FILEI MATI[1]="'+matFile+'"')
    toWrite.append('FILEO PRINTO[1]="'+appPath+'\\'+appName+appExtn+'"')
    if lokPath.lower() != 'na' and lokName.lower() != 'na':
        toWrite.append('FILEI LOOKUPI[1]="'+lokPath+'\\'+lokName+lokExtn+'"')
    toWrite.append('')
    toWrite.append('parameters maxstring=999')
    if lokPath.lower() != 'na' and lokName.lower() != 'na':
        toWrite.append('lookup name=Cube2Sat,')
        toWrite.append('  lookup[1]={},result={},'.format(cubSATW[0].strip(),cubSATW[1].strip()))
        toWrite.append('  interpolate=false,')
        toWrite.append('  fail=0,0,0,')
        toWrite.append('  lookupi=1')
        toWrite.append('')
        toWrite.append('mw[1]=mi.1.{}*{}'.format(matTabx,matFact))
        toWrite.append('jloop')
        toWrite.append('  Orig=Cube2Sat(1,i)')
        toWrite.append('  Dest=Cube2Sat(1,j)')
        toWrite.append('  print csv=t list=Orig(10.0l),Dest(10.0l),mw[1](30.20l) printo=1')
        toWrite.append('endjloop')
    else:
        toWrite.append('mw[1]=mi.1.{}*{}'.format(matTabx,matFact))
        toWrite.append('jloop')
        toWrite.append('  print csv=t list=i(10.0l),j(10.0l),mw[1](30.20l) printo=1')
        toWrite.append('endjloop')
    toWrite.append('ENDRUN')
    if delFile != 0:
        toWrite.append('*if exist "'+matFile+'" del "'+matFile+'"')
    print(appPath.replace('\\','/')+'/'+appName+'.s')
    with open(appPath.replace('\\','/')+'/'+appName+'.s','w') as script:
        for line in toWrite:
            print(line,file=script)
    procSingle(['"'+appCube+'" "'+appPath+'\\'+appName+'.s" -Pvdmi /Start /Hide /HideScript','del "'+appPath+'\\'+appName+'.s"','del "'+appPath+'\\vdmi*.prn"','del "'+appPath+'\\vdmi*.var"'])

### IMPORT CSV DEMAND TO RAM ###
def csv2Numpy(csvFile):
    outTrip = pd.read_csv(csvFile,header=None,index_col=['O','D'],dtype ={'O':int,'D':int,'Ve':float},names=['O','D','Ve'],low_memory=False)
    outTrip = outTrip.sort_index()
    outTrip = outTrip.reset_index(drop=False).pivot(index='O',columns='D',values='Ve').values
    procSingle(['del "'+csvFile+'"'])
    return outTrip

### Main Application ###
def main(source_directory, output_path):
    mp.freeze_support()
    np.seterr(all = 'ignore')
    numCPUs = os.cpu_count()-2

    intHBPA = 'SplitFactors_DS{}'            #DS1-HBEB, DS2-HBW, DS3-HBO, 1-16 CA, 17-32 NCA
    intNHBx = 'OD_Prop_{}_PT'                #1-NHBEB_CA, 2-NHBEB_NCA, 3-NHBO_CA, 4-NHBO_NCA
    extODhr = 'Time_of_Day_Factors_Zonal_{}' #1-EB_NCA, 2-EB_CA_FH, 3-EB_CA_TH, 4-HBW_NCA, 5-HBW_CA_FH, 6-HBW_CA_TH, 7-Oth_NCA, 8-Oth_CA_FH, 9-Oth_CA_TH

    # Temporary file to store all intermediate outputs
    temp_file_path = os.path.join(source_directory, "temp")
    os.makedirs(temp_file_path, exist_ok=True)

    cubEXES = ''
    cube_passed = False
    cube_locations = [
        r'C:\Program Files (x86)\Citilabs\cubeVoyager\voyager.exe',
        r'C:\Program Files\Citilabs\cubeVoyager\voyager.exe',
    ]
    for app in cube_locations:
        app = app.strip()
        if os.path.isfile(app) == True  and cubeTEST(app, temp_file_path) == True:
            cubEXES = app
            cube_passed = True
            break

    if not cube_passed:
        raise ValueError(
            "Can't find a .exe for CUBE. Tried looking in the following "
            f"places: {cube_locations}"
        )

    #convert MAT to CSV
    LOG.info('Convert MAT to CSV format ...')
    dctFile = {}
    pool = mp.Pool(numCPUs)
    for ds in [1,2,3]:
        dsx = 'hbeb' if ds==1 else 'hbw' if ds==2 else 'hbo'
        dctFile[dsx] = {'ca':{}, 'nca':{}}
        for fh in [1,2,3,4]:
            for th in [1,2,3,4]:
                ts, tsx = 10*fh+th, (fh-1)*4+th
                dctFile[dsx]['ca'][ts]  = intHBPA.format(ds)+'_ca_ts{}'.format(ts)
                dctFile[dsx]['nca'][ts] = intHBPA.format(ds)+'_nca_ts{}'.format(ts)
                pool.apply_async(cubeMAT2CSV,[cubEXES,source_directory,dctFile[dsx]['ca'][ts],'.csv',source_directory+'\\'+intHBPA.format(ds)+'.mat',1.0,tsx,'na','na','na','1:2',0])
                pool.apply_async(cubeMAT2CSV,[cubEXES,source_directory,dctFile[dsx]['nca'][ts],'.csv',source_directory+'\\'+intHBPA.format(ds)+'.mat',1.0,16+tsx,'na','na','na','1:2',0])

    dctFile['nhbeb'] = {'ca':{}, 'nca':{}}
    dctFile['nhbo']  = {'ca':{}, 'nca':{}}
    for ts in [1,2,3,4]:
        tsx = 'am' if ts==1 else 'ip' if ts==2 else 'pm' if ts==3 else 'op'
        dctFile['nhbeb']['ca'][ts]  = intNHBx.format(tsx)+'_nhbeb_ca'
        dctFile['nhbeb']['nca'][ts] = intNHBx.format(tsx)+'_nhbeb_nca'
        dctFile['nhbo']['ca'][ts]   = intNHBx.format(tsx)+'_nhbo_ca'
        dctFile['nhbo']['nca'][ts]  = intNHBx.format(tsx)+'_nhbo_nca'
        pool.apply_async(cubeMAT2CSV,[cubEXES,source_directory,dctFile['nhbeb']['ca'][ts],'.csv',source_directory+'\\'+intNHBx.format(tsx)+'.mat',1.0,1,'na','na','na','1:2',0])
        pool.apply_async(cubeMAT2CSV,[cubEXES,source_directory,dctFile['nhbeb']['nca'][ts],'.csv',source_directory+'\\'+intNHBx.format(tsx)+'.mat',1.0,2,'na','na','na','1:2',0])
        pool.apply_async(cubeMAT2CSV,[cubEXES,source_directory,dctFile['nhbo']['ca'][ts],'.csv',source_directory+'\\'+intNHBx.format(tsx)+'.mat',1.0,3,'na','na','na','1:2',0])
        pool.apply_async(cubeMAT2CSV,[cubEXES,source_directory,dctFile['nhbo']['nca'][ts],'.csv',source_directory+'\\'+intNHBx.format(tsx)+'.mat',1.0,4,'na','na','na','1:2',0])

    dctFile['ex_eb'], dctFile['ex_hbw'], dctFile['ex_oth']  = {'ca_fh':{},'ca_th':{},'nca':{}}, {'ca_fh':{},'ca_th':{},'nca':{}}, {'ca_fh':{},'ca_th':{},'nca':{}}
    for ts in [1,2,3,4]:
        tsx = 'am' if ts==1 else 'ip' if ts==2 else 'pm' if ts==3 else 'op'
        dctFile['ex_eb']['nca'][ts]    = extODhr.format(tsx)+'_eb_nca'
        dctFile['ex_eb']['ca_fh'][ts]  = extODhr.format(tsx)+'_eb_ca_fh'
        dctFile['ex_eb']['ca_th'][ts]  = extODhr.format(tsx)+'_eb_ca_th'
        dctFile['ex_hbw']['nca'][ts]   = extODhr.format(tsx)+'_hbw_nca'
        dctFile['ex_hbw']['ca_fh'][ts] = extODhr.format(tsx)+'_hbw_ca_fh'
        dctFile['ex_hbw']['ca_th'][ts] = extODhr.format(tsx)+'_hbw_ca_th'
        dctFile['ex_oth']['nca'][ts]   = extODhr.format(tsx)+'_oth_nca'
        dctFile['ex_oth']['ca_fh'][ts] = extODhr.format(tsx)+'_oth_ca_fh'
        dctFile['ex_oth']['ca_th'][ts] = extODhr.format(tsx)+'_oth_ca_th'
        pool.apply_async(cubeMAT2CSV,[cubEXES,source_directory,dctFile['ex_eb']['nca'][ts],'.csv',source_directory+'\\'+extODhr.format(tsx)+'.mat',1.0,1,'na','na','na','1:2',0])
        pool.apply_async(cubeMAT2CSV,[cubEXES,source_directory,dctFile['ex_eb']['ca_fh'][ts],'.csv',source_directory+'\\'+extODhr.format(tsx)+'.mat',1.0,2,'na','na','na','1:2',0])
        pool.apply_async(cubeMAT2CSV,[cubEXES,source_directory,dctFile['ex_eb']['ca_th'][ts],'.csv',source_directory+'\\'+extODhr.format(tsx)+'.mat',1.0,3,'na','na','na','1:2',0])
        pool.apply_async(cubeMAT2CSV,[cubEXES,source_directory,dctFile['ex_hbw']['nca'][ts],'.csv',source_directory+'\\'+extODhr.format(tsx)+'.mat',1.0,4,'na','na','na','1:2',0])
        pool.apply_async(cubeMAT2CSV,[cubEXES,source_directory,dctFile['ex_hbw']['ca_fh'][ts],'.csv',source_directory+'\\'+extODhr.format(tsx)+'.mat',1.0,5,'na','na','na','1:2',0])
        pool.apply_async(cubeMAT2CSV,[cubEXES,source_directory,dctFile['ex_hbw']['ca_th'][ts],'.csv',source_directory+'\\'+extODhr.format(tsx)+'.mat',1.0,6,'na','na','na','1:2',0])
        pool.apply_async(cubeMAT2CSV,[cubEXES,source_directory,dctFile['ex_oth']['nca'][ts],'.csv',source_directory+'\\'+extODhr.format(tsx)+'.mat',1.0,7,'na','na','na','1:2',0])
        pool.apply_async(cubeMAT2CSV,[cubEXES,source_directory,dctFile['ex_oth']['ca_fh'][ts],'.csv',source_directory+'\\'+extODhr.format(tsx)+'.mat',1.0,8,'na','na','na','1:2',0])
        pool.apply_async(cubeMAT2CSV,[cubEXES,source_directory,dctFile['ex_oth']['ca_th'][ts],'.csv',source_directory+'\\'+extODhr.format(tsx)+'.mat',1.0,9,'na','na','na','1:2',0])
    pool.close()
    pool.join()

    #Import to python
    LOG.info('Import CSV to python')
    pool = mp.Pool(numCPUs)
    for ds in dctFile:
        for ca in dctFile[ds]:
            for ts in dctFile[ds][ca]:
                dctFile[ds][ca][ts] = pool.apply_async(csv2Numpy,[source_directory+'\\'+dctFile[ds][ca][ts]+'.csv'])
    pool.close()
    pool.join()
    for ds in dctFile:
        for ca in dctFile[ds]:
            for ts in dctFile[ds][ca]:
                dctFile[ds][ca][ts] = dctFile[ds][ca][ts].get()

    #Export to pickle
    LOG.info("Exporting to pickle file %s", output_path)
    with open(output_path,'wb') as log:
        pk.dump(dctFile,log,pk.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main(SOURCE_DIRECTORY, OUTPUT_PATH)
