# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 09:56:10 2022

@author: mishtaiwi1
"""

import os
import sys

import pandas as pd
import typing as typ
import subprocess as sp
import openmatrix as omx
import logging


def CheckFileExists(file):
    '''
    Parameters
    ----------
    file : str
        full path to the file.

    Function
    ----------
    function checks if the file doesn't exist and report when it doesn't
    the function will force quit when a file doesn;t exist

    Returns
    -------
    None.

    '''
    if not os.path.isfile(file):
        print(f' -- File not found - {file}', "red")
        sys.exit()


# subprocess
def proc_single(cmd_list: typ.List):
    '''
    Parameters
    ----------
    cmd_list : list
        list of commands to execute.

    Function
    ----------
    execute processes one after the other in the list order

    Returns
    -------
    None.

    '''
    for ts in cmd_list:
        pr = sp.Popen(ts, creationflags=sp.CREATE_NEW_CONSOLE, shell=True)
        pr.wait()


def MAT2OMX(exe_cube: str, matFile: str, outPath: str, outFile: str):
    '''
    Parameters
    ----------
    exe_cube : str
        path to the cube voyager executable.
    matFile : str
        full path to the .mat file.
    outPath : str
        path to folder where outputs to be saved.
    outFile : str
        name of the output omx file.

    Function
    ----------
    function takes a Cube .MAT file and exports it to .OMX file

    Returns
    -------
    None.

    '''
    #check files exists
    CheckFileExists(exe_cube)
    CheckFileExists(matFile)

    #replace / with \\ for file paths
    matFile = matFile.replace('/', '\\').strip()
    outPath = outPath.replace('/', '\\').strip()

    to_write = [f'convertmat from="{matFile}" to="{outPath}\\{outFile}.omx" format=omx compression=4']
    with open(f'{outPath}\\Mat2OMX.s', 'w') as script:
        for line in to_write:
            print(line, file=script)

    proc_single([f'"{exe_cube}" "{outPath}\\Mat2OMX.s" -Pvdmi /Start /Hide /HideScript',
                 f'del "{outPath}\\*.prn"', f'del "{outPath}\\*.VAR"', f'del "{outPath}\\*.PRJ"',
                 f'del "{outPath}\\Mat2OMX.s"'])

def OMX2DF(omx):
    '''
    Parameters
    ----------
    omx : cArray
        omx single matrix.

    Function
    ----------
    function converts a cArray omx matrix to a pandas dataframe

    Returns
    -------
    df : pandas df
        matrix dataframe.
    '''
    #get omx array to pandas dataframe and reset productions
    df = pd.DataFrame(omx).reset_index().rename(columns={'index': 'from_model_zone_id'})
    #melt DF to get attarctions vector
    df = df.melt(id_vars=['from_model_zone_id'], var_name='to_model_zone_id', value_name='Demand')
    #adjust zone number
    df['from_model_zone_id'] = df['from_model_zone_id'] + 1
    df['to_model_zone_id'] = df['to_model_zone_id'] + 1

    return df


def StnZone2StnTLC(stnZone2Node: str, railNodes: str, extNodes: str, overwrite_TLCs):
    '''
    Parameters
    ----------
    stnZone2Node : str
        full path to the station node to station zone lookup file.
    railNodes : str
        full path to the rail nodes file.
    extNodes : str
        full path to the external rail station nodes file.
    overwrite_TLCs: pandas dataframe
        TLC overwrite dataframe

    Function
    ---------
    produce a stn zone ID to TLC lookup while overwritting the NoRMS TLC with a
    more suitable and EDGE matching TLC

    Returns
    -------
    df : pandas dataframe
        lookup betwee nstation zone ID and station TLC.

    '''
    #check files exists
    CheckFileExists(stnZone2Node)
    CheckFileExists(railNodes)
    CheckFileExists(extNodes)
    #read dataframes
    stnZone2Node = pd.read_csv(stnZone2Node)
    railNodes = pd.read_csv(railNodes)
    extNodes = pd.read_csv(extNodes, names=['N', 'X', 'Y', 'STATIONCODE', 'STATIONNAME',
                                            'ZONEID', 'TFN_FLAG', 'Category_ID', 'Oth'], header=None)
    #keep needed columns
    extNodes = extNodes[['N', 'X', 'Y', 'STATIONCODE', 'STATIONNAME',
                                            'ZONEID', 'TFN_FLAG', 'Category_ID']]
    #concat all rail nodes
    railNodes = pd.concat([railNodes, extNodes], axis=0)
    #keep only stn zones records
    stnZone2Node = stnZone2Node.loc[stnZone2Node['A'] < 10000].reset_index(drop=True)
    #merge zone nodes to rail nodes
    df = stnZone2Node.merge(railNodes, how='left', left_on=['B'], right_on=['N'])
    #keep needed cols
    df = df[['A', 'STATIONCODE', 'STATIONNAME']]
    #rename column
    df = df.rename(columns={'A':'stn_zone_id'})
    #remove '_' from station name and replace with ' '
    df['STATIONNAME'] = df['STATIONNAME'].str.replace('_',' ')
    #overwrite TLCs
    for i, row in overwrite_TLCs.iterrows():
        #get values
        currentTLC = overwrite_TLCs.at[i, 'NoRMS']
        overwriteTLC = overwrite_TLCs.at[i, 'Overwrite']
        #amend value
        df['STATIONCODE'].loc[df['STATIONCODE'] == currentTLC] = overwriteTLC
        #log overwritten station code
        logging.info(f'NoRMS TLC ({currentTLC}) overwritten with ({overwriteTLC})')
    return df


def ExportMat2CSVViaOMX(cube_exe: str, in_mat: str, out_path:str, out_csv: str, segment: str):
    '''
    Parameters
    ----------
    cube_exe : str
        path to the cube voyager executable
    in_mat: str
        full path and name of the input matrix file
    out_path: str
        path to the folder where outputs to be saved
    out_file: str
        name of the output file name
    segment: str
        segment of the overarching loop (e.g. purposes, periods, etc.)

    Function
    ----------
    takes a cube .MAT file and export tabs to .csv files

    Returns
    -------
    None.
    '''
    #Export PT Demand
    MAT2OMX(cube_exe, in_mat, out_path, f'/{out_csv}')
    #open omx
    omx_file = omx.open_file(out_path + f'/{out_csv}.omx')
    #get list of tabs
    omx_tabs = omx_file.list_matrices()
    #loop over MX tabs
    for mx in omx_tabs:
        #convert omx to pd dataframe
        df = OMX2DF(omx_file[mx])
        #export df to CSV
        df.to_csv(f'{out_path}/{out_csv}_{mx}.csv', index=False)
    #close omx
    omx_file.close()
    #delete .omx file
    os.remove(f'{out_path}/{out_csv}.omx')
    #delete .MAT files
    os.remove(f'{out_path}/PT_{segment}.MAT')
    print(f' -- PT {in_mat} Exported Successfully ', 'green')


def PTDemandFromTo(exe_cube: str, cat_folder: str, run_folder: str, output_folder: str):
    '''
    Parameters
    ----------
    exe_cube : str
        path to the cube voyager executable.
    cat_folder : str
        full path to the location of the NoRMS/NorTMS catalog.
    run_folder : str
        full path top the folder containing the .mat input files.
    output_folder : str
        full path top the folder where outputs to be saved.

    Function
    ----------
    function produces PA From and To by time period

    Returns
    -------
    None.

    '''
    #create file paths
    pt_24Hr_demand = run_folder + '/Inputs/Demand/PT_24hr_Demand.MAT'
    area_sectors = cat_folder + '/Params/Demand/Sector_Areas_Zones.MAT'

    splittingfactors_ds1 = run_folder + '/Inputs/Demand/SplitFactors_DS1.MAT'
    splittingfactors_ds2 = run_folder + '/Inputs/Demand/SplitFactors_DS2.MAT'
    splittingfactors_ds3 = run_folder + '/Inputs/Demand/SplitFactors_DS3.MAT'

    timeOfDay_AM = run_folder + '/Inputs/Demand/Time_of_Day_Factors_Zonal_AM.MAT'
    timeOfDay_IP = run_folder + '/Inputs/Demand/Time_of_Day_Factors_Zonal_IP.MAT'
    timeOfDay_PM = run_folder + '/Inputs/Demand/Time_of_Day_Factors_Zonal_PM.MAT'
    timeOfDay_OP = run_folder + '/Inputs/Demand/Time_of_Day_Factors_Zonal_OP.MAT'

    nhb_props_AM = run_folder + '/Inputs/Demand/OD_Prop_AM_PT.MAT'
    nhb_props_IP = run_folder + '/Inputs/Demand/OD_Prop_IP_PT.MAT'
    nhb_props_PM = run_folder + '/Inputs/Demand/OD_Prop_PM_PT.MAT'
    nhb_props_OP = run_folder + '/Inputs/Demand/OD_Prop_OP_PT.MAT'

    #check files exists
    CheckFileExists(exe_cube)

    CheckFileExists(pt_24Hr_demand)
    CheckFileExists(area_sectors)

    CheckFileExists(splittingfactors_ds1)
    CheckFileExists(splittingfactors_ds2)
    CheckFileExists(splittingfactors_ds3)

    CheckFileExists(timeOfDay_AM)
    CheckFileExists(timeOfDay_IP)
    CheckFileExists(timeOfDay_PM)
    CheckFileExists(timeOfDay_OP)

    CheckFileExists(nhb_props_AM)
    CheckFileExists(nhb_props_IP)
    CheckFileExists(nhb_props_PM)
    CheckFileExists(nhb_props_OP)


    #replace / with \\ for file paths
    pt_24Hr_demand = pt_24Hr_demand.replace('/', '\\').strip()
    area_sectors = area_sectors.replace('/', '\\').strip()

    splittingfactors_ds1 = splittingfactors_ds1.replace('/', '\\').strip()
    splittingfactors_ds2 = splittingfactors_ds2.replace('/', '\\').strip()
    splittingfactors_ds3 = splittingfactors_ds3.replace('/', '\\').strip()

    timeOfDay_AM = timeOfDay_AM.replace('/', '\\').strip()
    timeOfDay_IP = timeOfDay_IP.replace('/', '\\').strip()
    timeOfDay_PM = timeOfDay_PM.replace('/', '\\').strip()
    timeOfDay_OP = timeOfDay_OP.replace('/', '\\').strip()

    nhb_props_AM = nhb_props_AM.replace('/', '\\').strip()
    nhb_props_IP = nhb_props_IP.replace('/', '\\').strip()
    nhb_props_PM = nhb_props_PM.replace('/', '\\').strip()
    nhb_props_OP = nhb_props_OP.replace('/', '\\').strip()

    output_folder = output_folder.replace('/', '\\').strip()

    to_write = [f'RUN PGM=MATRIX PRNFILE="{output_folder}\\1st_Print.prn"',
                f'FILEI MATI[1] = "{pt_24Hr_demand}"',
                f'FILEI MATI[13] = "{nhb_props_OP}"',
                f'FILEI MATI[12] = "{nhb_props_PM}"',
                f'FILEI MATI[11] = "{nhb_props_IP}"',
                f'FILEI MATI[10] = "{nhb_props_AM}"',
                f'; input',
                f'; HB demand',
                f'FILEI MATI[2] = "{splittingfactors_ds1}"',
                f'FILEI MATI[3] = "{splittingfactors_ds2}"',
                f'FILEI MATI[4] = "{splittingfactors_ds3}"',
                f'; Sector definition',
                f'FILEI MATI[5] = "{area_sectors}"',
                f'; NHB Demand',
                f'FILEI MATI[6] = "{timeOfDay_AM}"',
                f'FILEI MATI[7] = "{timeOfDay_IP}"',
                f'FILEI MATI[8] = "{timeOfDay_PM}"',
                f'FILEI MATI[9] = "{timeOfDay_OP}"',
                f';--------------------------------',
                f';output',
                f'FILEO MATO[1] = "{output_folder}\\PT_AM.MAT",',
                f'MO=301-310,202,203,201,205,206,204,208,209,207,341-350,',
                f'NAME=HBEBCA_Int, HBEBNCA_Int, NHBEBCA_Int, NHBEBNCA_Int, HBWCA_Int, HBWNCA_Int, HBOCA_Int, HBONCA_Int, NHBOCA_Int, NHBONCA_Int,',
                f'EBCA_Ext_FM, EBCA_Ext_TO, EBNCA_Ext, HBWCA_Ext_FM, HBWCA_Ext_TO, HBWNCA_Ext, OCA_Ext_FM, OCA_Ext_TO, ONCA_Ext,',
                f'HBEBCA_Int_T, HBEBNCA_Int_T, NHBEBCA_Int_T, NHBEBNCA_Int_T, HBWCA_Int_T, HBWNCA_Int_T, HBOCA_Int_T, HBONCA_Int_T, NHBOCA_Int_T, NHBONCA_Int_T,',
                f'DEC=29*D',
                f'FILEO MATO[2] = "{output_folder}\\PT_IP.MAT",',
                f'MO=311-320,212,213,211,215,216,214,218,219,217,351-360,',
                f'NAME=HBEBCA_Int, HBEBNCA_Int, NHBEBCA_Int, NHBEBNCA_Int, HBWCA_Int, HBWNCA_Int, HBOCA_Int, HBONCA_Int, NHBOCA_Int, NHBONCA_Int,',
                f'EBCA_Ext_FM, EBCA_Ext_TO, EBNCA_Ext, HBWCA_Ext_FM, HBWCA_Ext_TO, HBWNCA_Ext, OCA_Ext_FM, OCA_Ext_TO, ONCA_Ext,',
                f'HBEBCA_Int_T, HBEBNCA_Int_T, NHBEBCA_Int_T, NHBEBNCA_Int_T, HBWCA_Int_T, HBWNCA_Int_T, HBOCA_Int_T, HBONCA_Int_T, NHBOCA_Int_T, NHBONCA_Int_T,',
                f'DEC=29*D',
                f'FILEO MATO[3] = "{output_folder}\\PT_PM.MAT",',
                f'MO=321-330,222,223,221,225,226,224,228,229,227,361-370,',
                f'NAME=HBEBCA_Int, HBEBNCA_Int, NHBEBCA_Int, NHBEBNCA_Int, HBWCA_Int, HBWNCA_Int, HBOCA_Int, HBONCA_Int, NHBOCA_Int, NHBONCA_Int,',
                f'EBCA_Ext_FM, EBCA_Ext_TO, EBNCA_Ext, HBWCA_Ext_FM, HBWCA_Ext_TO, HBWNCA_Ext, OCA_Ext_FM, OCA_Ext_TO, ONCA_Ext,',
                f'HBEBCA_Int_T, HBEBNCA_Int_T, NHBEBCA_Int_T, NHBEBNCA_Int_T, HBWCA_Int_T, HBWNCA_Int_T, HBOCA_Int_T, HBONCA_Int_T, NHBOCA_Int_T, NHBONCA_Int_T,',
                f'DEC=29*D',
                f'FILEO MATO[4] = "{output_folder}\\PT_OP.MAT",',
                f'MO=331-340,232,233,231,235,236,234,238,239,237,371-380,',
                f'NAME=HBEBCA_Int, HBEBNCA_Int, NHBEBCA_Int, NHBEBNCA_Int, HBWCA_Int, HBWNCA_Int, HBOCA_Int, HBONCA_Int, NHBOCA_Int, NHBONCA_Int,',
                f'EBCA_Ext_FM, EBCA_Ext_TO, EBNCA_Ext, HBWCA_Ext_FM, HBWCA_Ext_TO, HBWNCA_Ext, OCA_Ext_FM, OCA_Ext_TO, ONCA_Ext,',
                f'HBEBCA_Int_T, HBEBNCA_Int_T, NHBEBCA_Int_T, NHBEBNCA_Int_T, HBWCA_Int_T, HBWNCA_Int_T, HBOCA_Int_T, HBONCA_Int_T, NHBOCA_Int_T, NHBONCA_Int_T,',
                f'DEC=29*D',
                f'PARAMETERS ZONEMSG=100 ',
                f';NB everything is still in PA format when applicable',
                f'; first 10 matrices segmented by ca and nca plus 5 purposes in total 10',
                f'; second 9 matrices segmented by from and to home and 3 purposes ',
                f';"HBEBCA_Int", "HBEBNCA_Int", "NHBEBCA_Int", "NHBEBNCA_Int", "HBWCA_Int", "HBWNCA_Int", "HBOCA_Int", "HBONCA_Int", "NHBOCA_Int", "NHBONCA_Int",',
                f';       "EBCA_Ext_FM", "EBCA_Ext_TO", "EBNCA_Ext", "HBWCA_Ext_FM", "HBWCA_Ext_TO", "HBWNCA" "OCA_Ext_FM", "OCA_Ext_TO", "ONCA_Ext",',
                f';Internal Demand',
                f'FILLMW MW[1]=MI.1.1(10)',
                f';External Demand',
                f'MW[11]=MI.1.13 ;203',
                f'MW[12]=MI.1.11 ;201',
                f'MW[13]=MI.1.12 ;202',
                f'MW[14]=MI.1.16 ;206',
                f'MW[15]=MI.1.14 ;204',
                f'MW[16]=MI.1.15 ;205',
                f'MW[17]=MI.1.19 ;209',
                f'MW[18]=MI.1.17 ;207',
                f'MW[19]=MI.1.18 ;208'
                f';FILLMW MW[21]=MI.1.1.T(10) ; PA to home',
                f';	EMP_NCA	EMP_CA_FM	EMP_CA_TH	COM_NCA	COM_CA_FM	COM_CA_TH	OTH_NCA	OTH_CA_FM	OTH_CA_TH',
                f';OLD	1	2	3	4	5	6	7	8	9',
                f';NEW	3	1	2	6	4	5	9	7	8',
                f';	11	12	13	14	15	16	17	18	19',
                f';Read in "Areas" matrix',
                f'; 1 - Scotland-Scotland',
                f'; 2 - Scotland-TfN',
                f'; 3 - Scotland-South',
                f'; 4 - TfN-Scotland',
                f'; 5 - TfN-TfN',
                f'; 6 - TfN-South',
                f'; 7 - South-Scotland',
                f'; 8 - South-TfN',
                f'; 9 - South-South',
                f';sector matrix',
                f'MW[20]=MI.5.1 ; select areas !=1,9,5',
                f';scaling parameters for MOIRA',
                f'FILLMW MW[101]=MI.2.1(16) ;EB CA',
                f'FILLMW MW[121]=MI.3.1(16) ;Commute CA',
                f'FILLMW MW[141]=MI.4.1(16) ;other CA',
                f'FILLMW MW[601]=MI.2.17(16) ;EB NCA',
                f'FILLMW MW[621]=MI.3.17(16) ;Commute NCA',
                f'FILLMW MW[641]=MI.4.17(16) ;other NCA',
                f';Scaling parameter from PLD (LD Demand)',
                f'FILLMW MW[201]=MI.6.1(9)   ; AM',
                f'FILLMW MW[211]=MI.7.1(9)   ; IP',
                f'FILLMW MW[221]=MI.8.1(9)   ; PM',
                f'FILLMW MW[231]=MI.9.1(9)   ; OP',
                f';scaling parameters for NHB Moira',
                f';Scaling parameter from NHBEBCA NHBEBNCA NHBOTCA NHBOTNCA',
                f'FILLMW MW[401]=MI.10.1(4)   ; AM',
                f'FILLMW MW[411]=MI.11.1(4)   ; IP',
                f'FILLMW MW[421]=MI.12.1(4)   ; PM',
                f'FILLMW MW[431]=MI.13.1(4)   ; OP',
                f'; demand =0 for long distance',
                f'JLOOP',
                f'IF(MW[20]=1 ||MW[20]=9 || MW[20]=5)',
                f'MW[11]=0',
                f'MW[12]=0',
                f'MW[13]=0',
                f'MW[14]=0',
                f'MW[15]=0',
                f'MW[16]=0',
                f'MW[17]=0',
                f'MW[18]=0',
                f'MW[19]=0',
                f'ENDIF',
                f'IF(MW[20]!=5) ;demand = 0 for internal demand',
                f'MW[01]=0',
                f'MW[02]=0',
                f'MW[03]=0',
                f'MW[04]=0',
                f'MW[05]=0',
                f'MW[06]=0',
                f'MW[07]=0',
                f'MW[08]=0',
                f'MW[09]=0',
                f'MW[10]=0',
                f'ENDIF',
                f'ENDJLOOP',
                f';	EMP_NCA	EMP_CA_FM	EMP_CA_TH	COM_NCA	COM_CA_FM	COM_CA_TH	OTH_NCA	OTH_CA_FM	OTH_CA_TH',
                f';OLD	1	2	3	4	5	6	7	8	9',
                f';NEW	3	1	2	6	4	5	9	7	8',
                f';	11	12	13	14	15	16	17	18	19',
                f'LOOP K=1,9',
                f'KD=10+K ;External Demand',
                f'K1=200+K ;AM',
                f'K2=210+K ;IP',
                f'K3=220+K ;PM',
                f'K4=230+K ;OP',
                f'MW[K1]=MW[K1]*MW[KD]',
                f'MW[K2]=MW[K2]*MW[KD]',
                f'MW[K3]=MW[K3]*MW[KD]',
                f'MW[K4]=MW[K4]*MW[KD]',
                f'ENDLOOP',
                f'; PA FROM',
                f';-------------------AM',
                f';From home - internal',
                f'MW[301]=MW[1]*(MW[101]+MW[102]+MW[103]+MW[104])',
                f'MW[302]=MW[2]*(MW[601]+MW[602]+MW[603]+MW[604])',
                f'MW[303]=MW[3]* MW[401];(MW[101]+MW[102]+MW[103]+MW[104])',
                f'MW[304]=MW[4]* MW[402];(MW[101]+MW[102]+MW[103]+MW[104])',
                f'MW[305]=MW[5]*(MW[121]+MW[122]+MW[123]+MW[124])',
                f'MW[306]=MW[6]*(MW[621]+MW[622]+MW[623]+MW[624])',
                f'MW[307]=MW[7]*(MW[141]+MW[142]+MW[143]+MW[144])',
                f'MW[308]=MW[8]*(MW[641]+MW[642]+MW[643]+MW[644])',
                f'MW[309]=MW[9]* MW[403];(MW[141]+MW[142]+MW[143]+MW[144])',
                f'MW[310]=MW[10]*MW[404];(MW[141]+MW[142]+MW[143]+MW[144])',
                f';----------------------------IP',
                f';From home - internal',
                f'MW[311]=MW[1]*(MW[105]+MW[106]+MW[107]+MW[108])',
                f'MW[312]=MW[2]*(MW[605]+MW[606]+MW[607]+MW[608])',
                f'MW[313]=MW[3]*MW[411];(MW[105]+MW[106]+MW[107]+MW[108])',
                f'MW[314]=MW[4]*MW[412];(MW[105]+MW[106]+MW[107]+MW[108])',
                f'MW[315]=MW[5]*(MW[125]+MW[126]+MW[127]+MW[128])',
                f'MW[316]=MW[6]*(MW[625]+MW[626]+MW[627]+MW[628])',
                f'MW[317]=MW[7]*(MW[145]+MW[146]+MW[147]+MW[148])',
                f'MW[318]=MW[8]*(MW[645]+MW[646]+MW[647]+MW[648])',
                f'MW[319]=MW[9]*MW[413];(MW[145]+MW[146]+MW[147]+MW[148])',
                f'MW[320]=MW[10]*MW[414];(MW[145]+MW[146]+MW[147]+MW[148])',
                f';----------------------------PM',
                f';From home - internal',
                f'MW[321]=MW[1]*(MW[109]+MW[110]+MW[111]+MW[112])',
                f'MW[322]=MW[2]*(MW[609]+MW[610]+MW[611]+MW[612])',
                f'MW[323]=MW[3]*MW[421];(MW[109]+MW[110]+MW[111]+MW[112])',
                f'MW[324]=MW[4]*MW[422];(MW[109]+MW[110]+MW[111]+MW[112])',
                f'MW[325]=MW[5]*(MW[129]+MW[130]+MW[131]+MW[132])',
                f'MW[326]=MW[6]*(MW[629]+MW[630]+MW[631]+MW[632])',
                f'MW[327]=MW[7]*(MW[149]+MW[150]+MW[151]+MW[152])',
                f'MW[328]=MW[8]*(MW[649]+MW[650]+MW[651]+MW[652])',
                f'MW[329]=MW[9]*MW[423];(MW[149]+MW[150]+MW[151]+MW[152])',
                f'MW[330]=MW[10]*MW[424];(MW[149]+MW[150]+MW[151]+MW[152])',
                f';----------------------------OP',
                f';From home - internal',
                f'MW[331]=MW[1]*(MW[113]+MW[114]+MW[115]+MW[116])',
                f'MW[332]=MW[2]*(MW[613]+MW[614]+MW[615]+MW[616])',
                f'MW[333]=MW[3]*MW[431];(MW[113]+MW[114]+MW[115]+MW[116])',
                f'MW[334]=MW[4]*MW[432];(MW[113]+MW[114]+MW[115]+MW[116])',
                f'MW[335]=MW[5]*(MW[133]+MW[134]+MW[135]+MW[136])',
                f'MW[336]=MW[6]*(MW[633]+MW[634]+MW[635]+MW[636])',
                f'MW[337]=MW[7]*(MW[153]+MW[154]+MW[155]+MW[156])',
                f'MW[338]=MW[8]*(MW[653]+MW[654]+MW[655]+MW[656])',
                f'MW[339]=MW[9]*MW[433];(MW[153]+MW[154]+MW[155]+MW[156])',
                f'MW[340]=MW[10]*MW[434];(MW[153]+MW[154]+MW[155]+MW[156])',
                f';------------------------------------------------------------------------------------------',
                f';OD To ',
                f';-------------------AM',
                f';To home - internal',
                f'MW[341]=MW[01]*(MW[101]+MW[105]+MW[109]+MW[113])',
                f'MW[342]=MW[02]*(MW[601]+MW[605]+MW[609]+MW[613])',
                f'MW[343]=0 ;MW[03]*;(MW[101]+MW[105]+MW[109]+MW[113])',
                f'MW[344]=0 ;MW[04]*;(MW[101]+MW[105]+MW[109]+MW[113])',
                f'MW[345]=MW[05]*(MW[121]+MW[125]+MW[129]+MW[133])',
                f'MW[346]=MW[06]*(MW[621]+MW[625]+MW[629]+MW[633])',
                f'MW[347]=MW[07]*(MW[141]+MW[145]+MW[149]+MW[153])',
                f'MW[348]=MW[08]*(MW[641]+MW[645]+MW[649]+MW[653])',
                f'MW[349]=0 ;MW[09]*;(MW[141]+MW[145]+MW[149]+MW[153])',
                f'MW[350]=0 ;MW[10]*;(MW[141]+MW[145]+MW[149]+MW[153])',
                f';----------------------------IP',
                f';To home - internal',
                f'MW[351]=MW[01]*(MW[102]+MW[106]+MW[110]+MW[114])',
                f'MW[352]=MW[02]*(MW[602]+MW[606]+MW[610]+MW[614])',
                f'MW[353]=0 ;MW[03]*;(MW[102]+MW[106]+MW[110]+MW[114])',
                f'MW[354]=0 ;MW[04]*;(MW[102]+MW[106]+MW[110]+MW[114])',
                f'MW[355]=MW[05]*(MW[122]+MW[126]+MW[130]+MW[134])',
                f'MW[356]=MW[06]*(MW[622]+MW[626]+MW[630]+MW[634])',
                f'MW[357]=MW[07]*(MW[142]+MW[146]+MW[150]+MW[154])',
                f'MW[358]=MW[08]*(MW[642]+MW[646]+MW[650]+MW[654])',
                f'MW[359]=0 ;MW[09]*;(MW[142]+MW[146]+MW[150]+MW[154])',
                f'MW[360]=0 ;MW[10]*;(MW[142]+MW[146]+MW[150]+MW[154])',
                f';----------------------------PM',
                f';To home - internal',
                f'MW[361]=MW[01]*(MW[103]+MW[107]+MW[111]+MW[115])',
                f'MW[362]=MW[02]*(MW[603]+MW[607]+MW[611]+MW[615])',
                f'MW[363]=0 ;MW[03]*;(MW[103]+MW[107]+MW[111]+MW[115])',
                f'MW[364]=0 ;MW[04]*;(MW[103]+MW[107]+MW[111]+MW[115])',
                f'MW[365]=MW[05]*(MW[123]+MW[127]+MW[131]+MW[135])',
                f'MW[366]=MW[06]*(MW[623]+MW[627]+MW[631]+MW[635])',
                f'MW[367]=MW[07]*(MW[143]+MW[147]+MW[151]+MW[155])',
                f'MW[368]=MW[08]*(MW[643]+MW[647]+MW[651]+MW[655])',
                f'MW[369]=0 ;MW[09]*;(MW[143]+MW[147]+MW[151]+MW[155])',
                f'MW[370]=0 ;MW[10]*;(MW[143]+MW[147]+MW[151]+MW[155])',
                f';----------------------------OP',
                f';To home - internal',
                f'MW[371]=MW[01]*(MW[104]+MW[108]+MW[112]+MW[116])',
                f'MW[372]=MW[02]*(MW[604]+MW[608]+MW[612]+MW[616])',
                f'MW[373]=0 ;MW[03]*;(MW[104]+MW[108]+MW[112]+MW[116])',
                f'MW[374]=0 ;MW[04]*;(MW[104]+MW[108]+MW[112]+MW[116])',
                f'MW[375]=MW[05]*(MW[124]+MW[128]+MW[132]+MW[136])',
                f'MW[376]=MW[06]*(MW[624]+MW[628]+MW[632]+MW[636])',
                f'MW[377]=MW[07]*(MW[144]+MW[148]+MW[152]+MW[156])',
                f'MW[378]=MW[08]*(MW[644]+MW[648]+MW[652]+MW[656])',
                f'MW[379]=0 ;MW[09]*;(MW[144]+MW[148]+MW[152]+MW[156])',
                f'MW[380]=0 ;MW[10]*;(MW[144]+MW[148]+MW[152]+MW[156])',
                f'ENDRUN']






    with open(f'{output_folder}\\PT_Periods_FT.s', 'w') as script:
        for line in to_write:
            print(line, file=script)

    proc_single([f'"{exe_cube}" "{output_folder}\\PT_Periods_FT.s" -Pvdmi /Start /Hide /HideScript',
                 f'del "{output_folder}\\*.prn"', f'del "{output_folder}\\*.VAR"', f'del "{output_folder}\\*.PRJ"',
                 f'del "{output_folder}\\PT_Periods_FT.s"'])



def PA2OD(exe_cube: str, mats_folder: str):
    '''
    Parameters
    ----------
    exe_cube : str
        path to the cube voyager executable.
    mats_folder : str
        full path to the input matrix files folder. this is uusually where matrices
        from "PTDemandFromTo" are saved, it's also where output matrices will be saved


    Function
    ----------
    transposes the T matrices to get Od matrices by period and demand segmnet

    Returns
    -------
    None.

    '''
    #create file paths
    amDemand = mats_folder + '/PT_AM.MAT'
    ipDemand = mats_folder + '/PT_IP.MAT'
    pmDemand = mats_folder + '/PT_PM.MAT'
    opDemand = mats_folder + '/PT_OP.MAT'

    #check files exists
    CheckFileExists(exe_cube)

    CheckFileExists(amDemand)
    CheckFileExists(ipDemand)
    CheckFileExists(pmDemand)
    CheckFileExists(opDemand)


    #replace / with \\ for file paths
    amDemand = amDemand.replace('/', '\\').strip()
    ipDemand = ipDemand.replace('/', '\\').strip()
    pmDemand = pmDemand.replace('/', '\\').strip()
    opDemand = opDemand.replace('/', '\\').strip()


    mats_folder = mats_folder.replace('/', '\\').strip()

    to_write = [f'RUN PGM=MATRIX PRNFILE="{mats_folder}2nd_Print.prn"',
                f'FILEI MATI[1] = "{amDemand}"',
                f'FILEI MATI[2] = "{ipDemand}"',
                f'FILEI MATI[3] = "{pmDemand}"',
                f'FILEI MATI[4] = "{opDemand}"',
                f';--------------------------------',
                f';output',
                f'FILEO MATO[1] = "{mats_folder}PT_AM_OD.MAT",',
                f'MO=1-10, 20-29,'
                f'NAME=HBEBCA_F,HBEBNCA_F,NHBEBCA_F,NHBEBNCA_F,HBWCA_F,HBWNCA_F,HBOCA_F,',
                f'HBONCA_F,NHBOCA_F,NHBONCA_F,',
                f'HBEBCA_T,HBEBNCA_T,NHBEBCA_T,NHBEBNCA_T,HBWCA_T,HBWNCA_T,HBOCA_T,',
                f'HBONCA_T,NHBOCA_T,NHBONCA_T,',
                f'DEC=20*D',
                f'FILEO MATO[2] = "{mats_folder}PT_IP_OD.MAT",',
                f'MO=31-40, 50-59,'
                f'NAME=HBEBCA_F,HBEBNCA_F,NHBEBCA_F,NHBEBNCA_F,HBWCA_F,HBWNCA_F,HBOCA_F,',
                f'HBONCA_F,NHBOCA_F,NHBONCA_F,',
                f'HBEBCA_T,HBEBNCA_T,NHBEBCA_T,NHBEBNCA_T,HBWCA_T,HBWNCA_T,HBOCA_T,',
                f'HBONCA_T,NHBOCA_T,NHBONCA_T,',
                f'DEC=20*D',
                f'FILEO MATO[3] = "{mats_folder}PT_PM_OD.MAT",',
                f'MO=61-70, 80-89,'
                f'NAME=HBEBCA_F,HBEBNCA_F,NHBEBCA_F,NHBEBNCA_F,HBWCA_F,HBWNCA_F,HBOCA_F,',
                f'HBONCA_F,NHBOCA_F,NHBONCA_F,',
                f'HBEBCA_T,HBEBNCA_T,NHBEBCA_T,NHBEBNCA_T,HBWCA_T,HBWNCA_T,HBOCA_T,',
                f'HBONCA_T,NHBOCA_T,NHBONCA_T,',
                f'DEC=20*D',
                f'FILEO MATO[4] = "{mats_folder}PT_OP_OD.MAT",',
                f'MO=91-100, 110-119,'
                f'NAME=HBEBCA_F,HBEBNCA_F,NHBEBCA_F,NHBEBNCA_F,HBWCA_F,HBWNCA_F,HBOCA_F,',
                f'HBONCA_F,NHBOCA_F,NHBONCA_F,',
                f'HBEBCA_T,HBEBNCA_T,NHBEBCA_T,NHBEBNCA_T,HBWCA_T,HBWNCA_T,HBOCA_T,',
                f'HBONCA_T,NHBOCA_T,NHBONCA_T,',
                f'DEC=20*D',

                f';P>A',
                f'FILLMW MW[01]=MI.1.1(19) ;AM 1-29',
                f'FILLMW MW[31]=MI.2.1(19) ;IP 30-59',
                f'FILLMW MW[61]=MI.3.1(19) ;PM 60-89',
                f'FILLMW MW[91]=MI.4.1(19) ;OP 91-119',

                f';A>P (transpose)',
                f'FILLMW	MW[20]=Mi.1.20.T(10)',
                f'FILLMW	MW[50]=Mi.2.20.T(10)',
                f'FILLMW	MW[80]=Mi.3.20.T(10)',
                f'FILLMW	MW[110]=Mi.4.20.T(10)',

                f';Add Externals  - AM',
                f'MW[1] = MW[1] + MW[11]    ;EBCA_F',
                f'MW[20] = MW[20] + MW[12]  ;EBCA_T',
                f'MW[4] = MW[4] + MW[13]    ;EBNCA',
                f'MW[5] = MW[5] + MW[14]    ;HBWCA_F',
                f'MW[24] = MW[24] + MW[15]  ;HBWCA_T',
                f'MW[6] = MW[6] + MW[16]    ;HBWNCA',
                f'MW[7] = MW[7] + MW[17]    ;OCA_F',
                f'MW[26] = MW[26] + MW[18]  ;OCA_T',
                f'MW[10] = MW[10] + MW[19]  ;ONCA',

                f';Add Externals  - IP',
                f'MW[31] = MW[31] + MW[41] ;EBCA_F',
                f'MW[50] = MW[50] + MW[42] ;EBCA_T',
                f'MW[34] = MW[34] + MW[43] ;EBNCA',
                f'MW[35] = MW[35] + MW[44] ;HBWCA_F',
                f'MW[54] = MW[54] + MW[45] ;HBWCA_T',
                f'MW[36] = MW[36] + MW[46] ;HBWNCA',
                f'MW[37] = MW[37] + MW[47] ;OCA_F',
                f'MW[56] = MW[56] + MW[48] ;OCA_T',
                f'MW[40] = MW[40] + MW[49] ;ONCA',

                f';Add Externals  - PM',
                f'MW[61] = MW[61] + MW[71] ;EBCA_F',
                f'MW[80] = MW[80] + MW[72] ;EBCA_T',
                f'MW[64] = MW[64] + MW[73] ;EBNCA',
                f'MW[65] = MW[65] + MW[74] ;HBWCA_F',
                f'MW[84] = MW[84] + MW[75] ;HBWCA_T',
                f'MW[66] = MW[66] + MW[76] ;HBWNCA',
                f'MW[87] = MW[67] + MW[77] ;OCA_F',
                f'MW[86] = MW[86] + MW[78] ;OCA_T',
                f'MW[70] = MW[70] + MW[79] ;ONCA',

                f';Add Externals  - OP',
                f'MW[91] = MW[91] + MW[101]   ;EBCA_F',
                f'MW[110] = MW[110] + MW[102] ;EBCA_T',
                f'MW[94] = MW[94] + MW[103]   ;EBNCA',
                f'MW[95] = MW[95] + MW[104]   ;HBWCA_F',
                f'MW[114] = MW[114] + MW[105] ;HBWCA_T',
                f'MW[96] = MW[96] + MW[106]   ;HBWNCA',
                f'MW[97] = MW[97] + MW[107] ;OCA_F',
                f'MW[116] = MW[116] + MW[108] ;OCA_T',
                f'MW[100] = MW[100] + MW[109] ;ONCA',

                f'ENDRUN']



    with open(f'{mats_folder}\\PA2OD.s', 'w') as script:
        for line in to_write:
            print(line, file=script)

    proc_single([f'"{exe_cube}" "{mats_folder}\\PA2OD.s" -Pvdmi /Start /Hide /HideScript',
                 f'del "{mats_folder}\\*.prn"', f'del "{mats_folder}\\*.VAR"', f'del "{mats_folder}\\*.PRJ"',
                 f'del "{mats_folder}\\PA2OD.s"'])
