# -*- coding: utf-8 -*-
"""EDGE Cube extractor functions."""
# ## IMPORTS ## #
# Standard imports
import os

# Third party imports
import typing as typ
import subprocess as sp
import logging
import pathlib
import openmatrix as omx
import pandas as pd
import numpy as np

# Local imports
from normits_demand.matrices.cube_mat_converter import CUBEMatConverter
from normits_demand.utils import file_ops

# from normits_demand.matrices import omx_file

# CONSTANTS
LOG = logging.getLogger(__name__)

# subprocess
def proc_single(cmd_list: typ.List):
    """Execute single process at a time.

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

    """
    for t_s in cmd_list:
        p_r = sp.Popen(t_s, creationflags=sp.CREATE_NEW_CONSOLE, shell=True)
        p_r.wait()


def omx_2_df(omx_mx: np.array):
    """Read omx to a pandas dataframe.

    Parameters
    ----------
    omx_mx : np.array
        omx single matrix.

    Function
    ----------
    function converts a cArray omx matrix to a pandas dataframe

    Returns
    -------
    mx_df : pandas df
        matrix dataframe.
    """
    # get omx array to pandas dataframe and reset productions
    mx_df = pd.DataFrame(omx_mx).reset_index().rename(columns={"index": "from_model_zone_id"})
    # melt DF to get attarctions vector
    mx_df = mx_df.melt(
        id_vars=["from_model_zone_id"], var_name="to_model_zone_id", value_name="Demand"
    )
    # adjust zone number
    mx_df["from_model_zone_id"] = mx_df["from_model_zone_id"] + 1
    mx_df["to_model_zone_id"] = mx_df["to_model_zone_id"] + 1

    return mx_df


def stnzone_2_stn_tlc(
    stn_zone_to_node: str, rail_nodes: str, ext_nodes: str, overwrite_tlcs: pd.DataFrame
):
    """Prepare the NoRMS 2 EDGE TLC codes dataframe.

    Parameters
    ----------
    stn_zone_to_node : str
        full path to the station node to station zone lookup file.
    rail_nodes : str
        full path to the rail nodes file.
    ext_nodes : str
        full path to the external rail station nodes file.
    overwrite_tlcs: pandas dataframe
        TLC overwrite dataframe

    Function
    ---------
    produce a stn zone ID to TLC lookup while overwritting the NoRMS TLC with a
    more suitable and EDGE matching TLC

    Returns
    -------
    zones_df : pandas dataframe
        lookup betwee nstation zone ID and station TLC.

    """
    # check files exists
    file_ops.check_file_exists(stn_zone_to_node)
    file_ops.check_file_exists(rail_nodes)
    file_ops.check_file_exists(ext_nodes)
    # read dataframes
    stn_zone_to_node = file_ops.read_df(stn_zone_to_node)
    rail_nodes = file_ops.read_df(rail_nodes)
    ext_nodes = file_ops.read_df(
        ext_nodes,
        names=[
            "N",
            "X",
            "Y",
            "STATIONCODE",
            "STATIONNAME",
            "ZONEID",
            "TFN_FLAG",
            "Category_ID",
        ],
        usecols=[0, 1, 2, 3, 4, 5, 6, 7],
        header=None,
    )

    # concat all rail nodes
    rail_nodes = pd.concat([rail_nodes, ext_nodes], axis=0)
    # keep only stn zones records
    stn_zone_to_node = stn_zone_to_node.loc[stn_zone_to_node["A"] < 10000].reset_index(
        drop=True
    )
    # merge zone nodes to rail nodes
    zones_df = stn_zone_to_node.merge(rail_nodes, how="left", left_on=["B"], right_on=["N"])
    # keep needed cols
    zones_df = zones_df[["A", "STATIONCODE", "STATIONNAME"]]
    # rename column
    zones_df = zones_df.rename(columns={"A": "stn_zone_id"})
    # remove '_' from station name and replace with ' '
    zones_df["STATIONNAME"] = zones_df["STATIONNAME"].str.replace("_", " ")
    # overwrite TLCs
    for i, row in overwrite_tlcs.iterrows():
        # get values
        current_tlc = overwrite_tlcs.at[i, "NoRMS"]
        overwrite_tlc = overwrite_tlcs.at[i, "Overwrite"]
        # amend value
        zones_df.loc[zones_df["STATIONCODE"] == current_tlc, "STATIONCODE"] = overwrite_tlc
        # log overwritten station code
        LOG.info("NoRMS TLC (%s) overwritten with (%s)", current_tlc, overwrite_tlc)
    return zones_df


def export_mat_2_csv_via_omx(
    cube_exe: pathlib.Path,
    in_mat: pathlib.Path,
    out_path: pathlib.Path,
    out_csv: str,
    segment: str,
):
    """Export Cube .MAT to .csv through .OMX.

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
    """
    # Export PT Demand
    c_m = CUBEMatConverter(cube_exe)
    c_m.mat_2_omx(in_mat, out_path, f"{out_csv}")
    # open omx
    omx_file = omx.open_file(pathlib.Path(out_path, f"{out_csv}.omx"))
    # get list of tabs
    omx_tabs = omx_file.list_matrices()
    # loop over MX tabs
    for single_mx in omx_tabs:
        # convert omx to pd dataframe
        mx_df = omx_2_df(omx_file[single_mx])
        # export df to CSV
        file_ops.write_df(
            mx_df, pathlib.Path(out_path, f"{out_csv}_{single_mx}.csv", index=False)
        )

    # with omx_file.OMXFile(pathlib.Path(out_path, f"{out_csv}.omx")) as omx_mat:
    #     for mx in omx_mat.matrix_levels:
    #         mat = pd.DataFrame(omx_mat.get_matrix_level(mx),
    #                            index=omx_mat.zones,
    #                            columns=omx_mat.zones)
    #         file_ops.write_df(mat, f"{out_path}/{out_csv}_{mx}.csv", index=False)

    # close omx
    omx_file.close()
    # delete .omx file
    os.remove(f"{out_path}/{out_csv}.omx")
    # delete .MAT files
    os.remove(f"{out_path}/PT_{segment}.MAT")


def pt_demand_from_to(
    exe_cube: pathlib.Path,
    cat_folder: pathlib.Path,
    run_folder: pathlib.Path,
    output_folder: pathlib.Path,
):
    """Create PA From/To Home matrices.

    Parameters
    ----------
    exe_cube : Path
        path to the cube voyager executable.
    cat_folder : Path
        full path to the location of the NoRMS/NorTMS catalog.
    run_folder : Path
        full path top the folder containing the .mat input files.
    output_folder : Path
        full path top the folder where outputs to be saved.

    Function
    ----------
    function produces PA From and To by time period

    Returns
    -------
    None.

    """
    # create file paths
    area_sectors = pathlib.Path(cat_folder, "Params/Demand/Sector_Areas_Zones.MAT")
    pt_24hr_demand = pathlib.Path(run_folder, "Inputs/Demand/PT_24hr_Demand.MAT")

    splittingfactors_ds1 = pathlib.Path(run_folder, "Inputs/Demand/SplitFactors_DS1.MAT")
    splittingfactors_ds2 = pathlib.Path(run_folder, "Inputs/Demand/SplitFactors_DS2.MAT")
    splittingfactors_ds3 = pathlib.Path(run_folder, "Inputs/Demand/SplitFactors_DS3.MAT")

    time_of_day_am = pathlib.Path(run_folder, "Inputs/Demand/Time_of_Day_Factors_Zonal_AM.MAT")
    time_of_day_ip = pathlib.Path(run_folder, "Inputs/Demand/Time_of_Day_Factors_Zonal_IP.MAT")
    time_of_day_pm = pathlib.Path(run_folder, "Inputs/Demand/Time_of_Day_Factors_Zonal_PM.MAT")
    time_of_day_op = pathlib.Path(run_folder, "Inputs/Demand/Time_of_Day_Factors_Zonal_OP.MAT")

    nhb_props_am = pathlib.Path(run_folder, "Inputs/Demand/OD_Prop_AM_PT.MAT")
    nhb_props_ip = pathlib.Path(run_folder, "Inputs/Demand/OD_Prop_IP_PT.MAT")
    nhb_props_pm = pathlib.Path(run_folder, "Inputs/Demand/OD_Prop_PM_PT.MAT")
    nhb_props_op = pathlib.Path(run_folder, "Inputs/Demand/OD_Prop_OP_PT.MAT")

    # list of all files
    file_list = [
        exe_cube,
        pt_24hr_demand,
        area_sectors,
        splittingfactors_ds1,
        splittingfactors_ds2,
        splittingfactors_ds3,
        time_of_day_am,
        time_of_day_ip,
        time_of_day_pm,
        time_of_day_op,
        nhb_props_am,
        nhb_props_ip,
        nhb_props_pm,
        nhb_props_op,
    ]
    # check files exists
    for file in file_list:
        file_ops.check_file_exists(file)

    to_write = [
        f'RUN PGM=MATRIX PRNFILE="{output_folder}\\1st_Print.prn"',
        f'FILEI MATI[1] = "{pt_24hr_demand}"',
        f'FILEI MATI[13] = "{nhb_props_op}"',
        f'FILEI MATI[12] = "{nhb_props_pm}"',
        f'FILEI MATI[11] = "{nhb_props_ip}"',
        f'FILEI MATI[10] = "{nhb_props_am}"',
        "; input",
        "; HB demand",
        f'FILEI MATI[2] = "{splittingfactors_ds1}"',
        f'FILEI MATI[3] = "{splittingfactors_ds2}"',
        f'FILEI MATI[4] = "{splittingfactors_ds3}"',
        "; Sector definition",
        f'FILEI MATI[5] = "{area_sectors}"',
        "; NHB Demand",
        f'FILEI MATI[6] = "{time_of_day_am}"',
        f'FILEI MATI[7] = "{time_of_day_ip}"',
        f'FILEI MATI[8] = "{time_of_day_pm}"',
        f'FILEI MATI[9] = "{time_of_day_op}"',
        ";--------------------------------",
        ";output",
        f'FILEO MATO[1] = "{output_folder}\\PT_AM.MAT",',
        "MO=301-310,202,203,201,205,206,204,208,209,207,341-350,",
        "NAME=HBEBCA_Int, HBEBNCA_Int, NHBEBCA_Int, NHBEBNCA_Int, "
        "HBWCA_Int, HBWNCA_Int, HBOCA_Int, HBONCA_Int, NHBOCA_Int, NHBONCA_Int,",
        "EBCA_Ext_FM, EBCA_Ext_TO, EBNCA_Ext, HBWCA_Ext_FM, HBWCA_Ext_TO, "
        "HBWNCA_Ext, OCA_Ext_FM, OCA_Ext_TO, ONCA_Ext,",
        "HBEBCA_Int_T, HBEBNCA_Int_T, NHBEBCA_Int_T, NHBEBNCA_Int_T, HBWCA_Int_T, "
        "HBWNCA_Int_T, HBOCA_Int_T, HBONCA_Int_T, NHBOCA_Int_T, NHBONCA_Int_T,",
        "DEC=29*D",
        f'FILEO MATO[2] = "{output_folder}\\PT_IP.MAT",',
        "MO=311-320,212,213,211,215,216,214,218,219,217,351-360,",
        "NAME=HBEBCA_Int, HBEBNCA_Int, NHBEBCA_Int, NHBEBNCA_Int, HBWCA_Int, "
        "HBWNCA_Int, HBOCA_Int, HBONCA_Int, NHBOCA_Int, NHBONCA_Int,",
        "EBCA_Ext_FM, EBCA_Ext_TO, EBNCA_Ext, HBWCA_Ext_FM, HBWCA_Ext_TO, "
        "HBWNCA_Ext, OCA_Ext_FM, OCA_Ext_TO, ONCA_Ext,",
        "HBEBCA_Int_T, HBEBNCA_Int_T, NHBEBCA_Int_T, NHBEBNCA_Int_T, HBWCA_Int_T, "
        "HBWNCA_Int_T, HBOCA_Int_T, HBONCA_Int_T, NHBOCA_Int_T, NHBONCA_Int_T,",
        "DEC=29*D",
        f'FILEO MATO[3] = "{output_folder}\\PT_PM.MAT",',
        "MO=321-330,222,223,221,225,226,224,228,229,227,361-370,",
        "NAME=HBEBCA_Int, HBEBNCA_Int, NHBEBCA_Int, NHBEBNCA_Int, HBWCA_Int, "
        "HBWNCA_Int, HBOCA_Int, HBONCA_Int, NHBOCA_Int, NHBONCA_Int,",
        "EBCA_Ext_FM, EBCA_Ext_TO, EBNCA_Ext, HBWCA_Ext_FM, HBWCA_Ext_TO, "
        "HBWNCA_Ext, OCA_Ext_FM, OCA_Ext_TO, ONCA_Ext,",
        "HBEBCA_Int_T, HBEBNCA_Int_T, NHBEBCA_Int_T, NHBEBNCA_Int_T, "
        "HBWCA_Int_T, HBWNCA_Int_T, HBOCA_Int_T, HBONCA_Int_T, NHBOCA_Int_T, NHBONCA_Int_T,",
        "DEC=29*D",
        f'FILEO MATO[4] = "{output_folder}\\PT_OP.MAT",',
        "MO=331-340,232,233,231,235,236,234,238,239,237,371-380,",
        "NAME=HBEBCA_Int, HBEBNCA_Int, NHBEBCA_Int, NHBEBNCA_Int, HBWCA_Int, "
        "HBWNCA_Int, HBOCA_Int, HBONCA_Int, NHBOCA_Int, NHBONCA_Int,",
        "EBCA_Ext_FM, EBCA_Ext_TO, EBNCA_Ext, HBWCA_Ext_FM, HBWCA_Ext_TO, "
        "HBWNCA_Ext, OCA_Ext_FM, OCA_Ext_TO, ONCA_Ext,",
        "HBEBCA_Int_T, HBEBNCA_Int_T, NHBEBCA_Int_T, NHBEBNCA_Int_T, HBWCA_Int_T, "
        "HBWNCA_Int_T, HBOCA_Int_T, HBONCA_Int_T, NHBOCA_Int_T, NHBONCA_Int_T,",
        "DEC=29*D",
        "PARAMETERS ZONEMSG=100 ",
        ";NB everything is still in PA format when applicable",
        "; first 10 matrices segmented by ca and nca plus 5 purposes in total 10",
        "; second 9 matrices segmented by from and to home and 3 purposes ",
        ';"HBEBCA_Int", "HBEBNCA_Int", "NHBEBCA_Int", "NHBEBNCA_Int", "\
            "HBWCA_Int", "HBWNCA_Int", "HBOCA_Int", "HBONCA_Int", "NHBOCA_Int", "NHBONCA_Int",',
        ';       "EBCA_Ext_FM", "EBCA_Ext_TO", "EBNCA_Ext", "HBWCA_Ext_FM", "\
            "HBWCA_Ext_TO", "HBWNCA" "OCA_Ext_FM", "OCA_Ext_TO", "ONCA_Ext",',
        ";Internal Demand",
        "FILLMW MW[1]=MI.1.1(10)",
        ";External Demand",
        "MW[11]=MI.1.13 ;203",
        "MW[12]=MI.1.11 ;201",
        "MW[13]=MI.1.12 ;202",
        "MW[14]=MI.1.16 ;206",
        "MW[15]=MI.1.14 ;204",
        "MW[16]=MI.1.15 ;205",
        "MW[17]=MI.1.19 ;209",
        "MW[18]=MI.1.17 ;207",
        "MW[19]=MI.1.18 ;208 ;FILLMW MW[21]=MI.1.1.T(10) ; PA to home",
        ";	EMP_NCA	EMP_CA_FM	EMP_CA_TH	COM_NCA	COM_CA_FM	COM_CA_TH	OTH_NCA	OTH_CA_FM	OTH_CA_TH",
        ";OLD	1	2	3	4	5	6	7	8	9",
        ";NEW	3	1	2	6	4	5	9	7	8",
        ";	11	12	13	14	15	16	17	18	19",
        ';Read in "Areas" matrix',
        "; 1 - Scotland-Scotland",
        "; 2 - Scotland-TfN",
        "; 3 - Scotland-South",
        "; 4 - TfN-Scotland",
        "; 5 - TfN-TfN",
        "; 6 - TfN-South",
        "; 7 - South-Scotland",
        "; 8 - South-TfN",
        "; 9 - South-South",
        ";sector matrix",
        "MW[20]=MI.5.1 ; select areas !=1,9,5",
        ";scaling parameters for MOIRA",
        "FILLMW MW[101]=MI.2.1(16) ;EB CA",
        "FILLMW MW[121]=MI.3.1(16) ;Commute CA",
        "FILLMW MW[141]=MI.4.1(16) ;other CA",
        "FILLMW MW[601]=MI.2.17(16) ;EB NCA",
        "FILLMW MW[621]=MI.3.17(16) ;Commute NCA",
        "FILLMW MW[641]=MI.4.17(16) ;other NCA",
        ";Scaling parameter from PLD (LD Demand)",
        "FILLMW MW[201]=MI.6.1(9)   ; AM",
        "FILLMW MW[211]=MI.7.1(9)   ; IP",
        "FILLMW MW[221]=MI.8.1(9)   ; PM",
        "FILLMW MW[231]=MI.9.1(9)   ; OP",
        ";scaling parameters for NHB Moira",
        ";Scaling parameter from NHBEBCA NHBEBNCA NHBOTCA NHBOTNCA",
        "FILLMW MW[401]=MI.10.1(4)   ; AM",
        "FILLMW MW[411]=MI.11.1(4)   ; IP",
        "FILLMW MW[421]=MI.12.1(4)   ; PM",
        "FILLMW MW[431]=MI.13.1(4)   ; OP",
        "; demand =0 for long distance",
        "JLOOP",
        "IF(MW[20]=1 ||MW[20]=9 || MW[20]=5)",
        "MW[11]=0",
        "MW[12]=0",
        "MW[13]=0",
        "MW[14]=0",
        "MW[15]=0",
        "MW[16]=0",
        "MW[17]=0",
        "MW[18]=0",
        "MW[19]=0",
        "ENDIF",
        "IF(MW[20]!=5) ;demand = 0 for internal demand",
        "MW[01]=0",
        "MW[02]=0",
        "MW[03]=0",
        "MW[04]=0",
        "MW[05]=0",
        "MW[06]=0",
        "MW[07]=0",
        "MW[08]=0",
        "MW[09]=0",
        "MW[10]=0",
        "ENDIF",
        "ENDJLOOP",
        ";	EMP_NCA	EMP_CA_FM	EMP_CA_TH	COM_NCA	COM_CA_FM	COM_CA_TH	OTH_NCA	OTH_CA_FM	OTH_CA_TH",
        ";OLD	1	2	3	4	5	6	7	8	9",
        ";NEW	3	1	2	6	4	5	9	7	8",
        ";	11	12	13	14	15	16	17	18	19",
        "LOOP K=1,9",
        "KD=10+K ;External Demand",
        "K1=200+K ;AM",
        "K2=210+K ;IP",
        "K3=220+K ;PM",
        "K4=230+K ;OP",
        "MW[K1]=MW[K1]*MW[KD]",
        "MW[K2]=MW[K2]*MW[KD]",
        "MW[K3]=MW[K3]*MW[KD]",
        "MW[K4]=MW[K4]*MW[KD]",
        "ENDLOOP",
        "; PA FROM",
        ";-------------------AM",
        ";From home - internal",
        "MW[301]=MW[1]*(MW[101]+MW[102]+MW[103]+MW[104])",
        "MW[302]=MW[2]*(MW[601]+MW[602]+MW[603]+MW[604])",
        "MW[303]=MW[3]* MW[401];(MW[101]+MW[102]+MW[103]+MW[104])",
        "MW[304]=MW[4]* MW[402];(MW[101]+MW[102]+MW[103]+MW[104])",
        "MW[305]=MW[5]*(MW[121]+MW[122]+MW[123]+MW[124])",
        "MW[306]=MW[6]*(MW[621]+MW[622]+MW[623]+MW[624])",
        "MW[307]=MW[7]*(MW[141]+MW[142]+MW[143]+MW[144])",
        "MW[308]=MW[8]*(MW[641]+MW[642]+MW[643]+MW[644])",
        "MW[309]=MW[9]* MW[403];(MW[141]+MW[142]+MW[143]+MW[144])",
        "MW[310]=MW[10]*MW[404];(MW[141]+MW[142]+MW[143]+MW[144])",
        ";----------------------------IP",
        ";From home - internal",
        "MW[311]=MW[1]*(MW[105]+MW[106]+MW[107]+MW[108])",
        "MW[312]=MW[2]*(MW[605]+MW[606]+MW[607]+MW[608])",
        "MW[313]=MW[3]*MW[411];(MW[105]+MW[106]+MW[107]+MW[108])",
        "MW[314]=MW[4]*MW[412];(MW[105]+MW[106]+MW[107]+MW[108])",
        "MW[315]=MW[5]*(MW[125]+MW[126]+MW[127]+MW[128])",
        "MW[316]=MW[6]*(MW[625]+MW[626]+MW[627]+MW[628])",
        "MW[317]=MW[7]*(MW[145]+MW[146]+MW[147]+MW[148])",
        "MW[318]=MW[8]*(MW[645]+MW[646]+MW[647]+MW[648])",
        "MW[319]=MW[9]*MW[413];(MW[145]+MW[146]+MW[147]+MW[148])",
        "MW[320]=MW[10]*MW[414];(MW[145]+MW[146]+MW[147]+MW[148])",
        ";----------------------------PM",
        ";From home - internal",
        "MW[321]=MW[1]*(MW[109]+MW[110]+MW[111]+MW[112])",
        "MW[322]=MW[2]*(MW[609]+MW[610]+MW[611]+MW[612])",
        "MW[323]=MW[3]*MW[421];(MW[109]+MW[110]+MW[111]+MW[112])",
        "MW[324]=MW[4]*MW[422];(MW[109]+MW[110]+MW[111]+MW[112])",
        "MW[325]=MW[5]*(MW[129]+MW[130]+MW[131]+MW[132])",
        "MW[326]=MW[6]*(MW[629]+MW[630]+MW[631]+MW[632])",
        "MW[327]=MW[7]*(MW[149]+MW[150]+MW[151]+MW[152])",
        "MW[328]=MW[8]*(MW[649]+MW[650]+MW[651]+MW[652])",
        "MW[329]=MW[9]*MW[423];(MW[149]+MW[150]+MW[151]+MW[152])",
        "MW[330]=MW[10]*MW[424];(MW[149]+MW[150]+MW[151]+MW[152])",
        ";----------------------------OP",
        ";From home - internal",
        "MW[331]=MW[1]*(MW[113]+MW[114]+MW[115]+MW[116])",
        "MW[332]=MW[2]*(MW[613]+MW[614]+MW[615]+MW[616])",
        "MW[333]=MW[3]*MW[431];(MW[113]+MW[114]+MW[115]+MW[116])",
        "MW[334]=MW[4]*MW[432];(MW[113]+MW[114]+MW[115]+MW[116])",
        "MW[335]=MW[5]*(MW[133]+MW[134]+MW[135]+MW[136])",
        "MW[336]=MW[6]*(MW[633]+MW[634]+MW[635]+MW[636])",
        "MW[337]=MW[7]*(MW[153]+MW[154]+MW[155]+MW[156])",
        "MW[338]=MW[8]*(MW[653]+MW[654]+MW[655]+MW[656])",
        "MW[339]=MW[9]*MW[433];(MW[153]+MW[154]+MW[155]+MW[156])",
        "MW[340]=MW[10]*MW[434];(MW[153]+MW[154]+MW[155]+MW[156])",
        ";-----------------------------------------------------------------",
        ";OD To ",
        ";-------------------AM",
        ";To home - internal",
        "MW[341]=MW[01]*(MW[101]+MW[105]+MW[109]+MW[113])",
        "MW[342]=MW[02]*(MW[601]+MW[605]+MW[609]+MW[613])",
        "MW[343]=0 ;MW[03]*;(MW[101]+MW[105]+MW[109]+MW[113])",
        "MW[344]=0 ;MW[04]*;(MW[101]+MW[105]+MW[109]+MW[113])",
        "MW[345]=MW[05]*(MW[121]+MW[125]+MW[129]+MW[133])",
        "MW[346]=MW[06]*(MW[621]+MW[625]+MW[629]+MW[633])",
        "MW[347]=MW[07]*(MW[141]+MW[145]+MW[149]+MW[153])",
        "MW[348]=MW[08]*(MW[641]+MW[645]+MW[649]+MW[653])",
        "MW[349]=0 ;MW[09]*;(MW[141]+MW[145]+MW[149]+MW[153])",
        "MW[350]=0 ;MW[10]*;(MW[141]+MW[145]+MW[149]+MW[153])",
        ";----------------------------IP",
        ";To home - internal",
        "MW[351]=MW[01]*(MW[102]+MW[106]+MW[110]+MW[114])",
        "MW[352]=MW[02]*(MW[602]+MW[606]+MW[610]+MW[614])",
        "MW[353]=0 ;MW[03]*;(MW[102]+MW[106]+MW[110]+MW[114])",
        "MW[354]=0 ;MW[04]*;(MW[102]+MW[106]+MW[110]+MW[114])",
        "MW[355]=MW[05]*(MW[122]+MW[126]+MW[130]+MW[134])",
        "MW[356]=MW[06]*(MW[622]+MW[626]+MW[630]+MW[634])",
        "MW[357]=MW[07]*(MW[142]+MW[146]+MW[150]+MW[154])",
        "MW[358]=MW[08]*(MW[642]+MW[646]+MW[650]+MW[654])",
        "MW[359]=0 ;MW[09]*;(MW[142]+MW[146]+MW[150]+MW[154])",
        "MW[360]=0 ;MW[10]*;(MW[142]+MW[146]+MW[150]+MW[154])",
        ";----------------------------PM",
        ";To home - internal",
        "MW[361]=MW[01]*(MW[103]+MW[107]+MW[111]+MW[115])",
        "MW[362]=MW[02]*(MW[603]+MW[607]+MW[611]+MW[615])",
        "MW[363]=0 ;MW[03]*;(MW[103]+MW[107]+MW[111]+MW[115])",
        "MW[364]=0 ;MW[04]*;(MW[103]+MW[107]+MW[111]+MW[115])",
        "MW[365]=MW[05]*(MW[123]+MW[127]+MW[131]+MW[135])",
        "MW[366]=MW[06]*(MW[623]+MW[627]+MW[631]+MW[635])",
        "MW[367]=MW[07]*(MW[143]+MW[147]+MW[151]+MW[155])",
        "MW[368]=MW[08]*(MW[643]+MW[647]+MW[651]+MW[655])",
        "MW[369]=0 ;MW[09]*;(MW[143]+MW[147]+MW[151]+MW[155])",
        "MW[370]=0 ;MW[10]*;(MW[143]+MW[147]+MW[151]+MW[155])",
        ";----------------------------OP",
        ";To home - internal",
        "MW[371]=MW[01]*(MW[104]+MW[108]+MW[112]+MW[116])",
        "MW[372]=MW[02]*(MW[604]+MW[608]+MW[612]+MW[616])",
        "MW[373]=0 ;MW[03]*;(MW[104]+MW[108]+MW[112]+MW[116])",
        "MW[374]=0 ;MW[04]*;(MW[104]+MW[108]+MW[112]+MW[116])",
        "MW[375]=MW[05]*(MW[124]+MW[128]+MW[132]+MW[136])",
        "MW[376]=MW[06]*(MW[624]+MW[628]+MW[632]+MW[636])",
        "MW[377]=MW[07]*(MW[144]+MW[148]+MW[152]+MW[156])",
        "MW[378]=MW[08]*(MW[644]+MW[648]+MW[652]+MW[656])",
        "MW[379]=0 ;MW[09]*;(MW[144]+MW[148]+MW[152]+MW[156])",
        "MW[380]=0 ;MW[10]*;(MW[144]+MW[148]+MW[152]+MW[156])",
        "ENDRUN",
    ]

    with open(pathlib.Path(output_folder, "PT_Periods_FT.s"), "w") as script:
        script.write("\n".join(to_write))

    proc_single(
        [
            f'"{exe_cube}" "{output_folder}\\PT_Periods_FT.s" -Pvdmi /Start /Hide /HideScript',
            f'del "{output_folder}\\*.prn"',
            f'del "{output_folder}\\*.VAR"',
            f'del "{output_folder}\\*.PRJ"',
            f'del "{output_folder}\\PT_Periods_FT.s"',
        ]
    )


def pa_2_od(exe_cube: str, mats_folder: str):
    """Create Cube Script to convert NoRMS Matrices from PA to OD.

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

    """
    # create file paths
    exe_cube = pathlib.Path(exe_cube)
    exe_cube = pathlib.Path(mats_folder)
    am_demand = pathlib.Path(mats_folder, "PT_AM.MAT")
    ip_demand = pathlib.Path(mats_folder, "PT_IP.MAT")
    pm_demand = pathlib.Path(mats_folder, "PT_PM.MAT")
    op_demand = pathlib.Path(mats_folder, "PT_OP.MAT")

    files_list = [exe_cube, am_demand, ip_demand, pm_demand, op_demand]
    # check files exists
    for file in files_list:
        file_ops.check_file_exists(file)

    to_write = [
        f'RUN PGM=MATRIX PRNFILE="{mats_folder}2nd_Print.prn"',
        f'FILEI MATI[1] = "{am_demand}"',
        f'FILEI MATI[2] = "{ip_demand}"',
        f'FILEI MATI[3] = "{pm_demand}"',
        f'FILEI MATI[4] = "{op_demand}"',
        ";--------------------------------",
        ";output",
        'FILEO MATO[1] = "{mats_folder}PT_AM_OD.MAT",',
        "MO=1-10, 20-29,"
        "NAME=HBEBCA_F,HBEBNCA_F,NHBEBCA_F,NHBEBNCA_F,HBWCA_F,HBWNCA_F,HBOCA_F,",
        "HBONCA_F,NHBOCA_F,NHBONCA_F,",
        "HBEBCA_T,HBEBNCA_T,NHBEBCA_T,NHBEBNCA_T,HBWCA_T,HBWNCA_T,HBOCA_T,",
        "HBONCA_T,NHBOCA_T,NHBONCA_T,",
        "DEC=20*D",
        f'FILEO MATO[2] = "{mats_folder}PT_IP_OD.MAT",',
        "MO=31-40, 50-59,"
        "NAME=HBEBCA_F,HBEBNCA_F,NHBEBCA_F,NHBEBNCA_F,HBWCA_F,HBWNCA_F,HBOCA_F,",
        "HBONCA_F,NHBOCA_F,NHBONCA_F,",
        "HBEBCA_T,HBEBNCA_T,NHBEBCA_T,NHBEBNCA_T,HBWCA_T,HBWNCA_T,HBOCA_T,",
        "HBONCA_T,NHBOCA_T,NHBONCA_T,",
        "DEC=20*D",
        f'FILEO MATO[3] = "{mats_folder}PT_PM_OD.MAT",',
        "MO=61-70, 80-89,",
        "NAME=HBEBCA_F,HBEBNCA_F,NHBEBCA_F,NHBEBNCA_F,HBWCA_F,HBWNCA_F,HBOCA_F,",
        "HBONCA_F,NHBOCA_F,NHBONCA_F,",
        "HBEBCA_T,HBEBNCA_T,NHBEBCA_T,NHBEBNCA_T,HBWCA_T,HBWNCA_T,HBOCA_T,",
        "HBONCA_T,NHBOCA_T,NHBONCA_T,",
        "DEC=20*D",
        f'FILEO MATO[4] = "{mats_folder}PT_OP_OD.MAT",',
        "MO=91-100, 110-119,",
        "NAME=HBEBCA_F,HBEBNCA_F,NHBEBCA_F,NHBEBNCA_F,HBWCA_F,HBWNCA_F,HBOCA_F,",
        "HBONCA_F,NHBOCA_F,NHBONCA_F,",
        "HBEBCA_T,HBEBNCA_T,NHBEBCA_T,NHBEBNCA_T,HBWCA_T,HBWNCA_T,HBOCA_T,",
        "HBONCA_T,NHBOCA_T,NHBONCA_T,",
        "DEC=20*D",
        ";P>A",
        "FILLMW MW[01]=MI.1.1(19) ;AM 1-29",
        "FILLMW MW[31]=MI.2.1(19) ;IP 30-59",
        "FILLMW MW[61]=MI.3.1(19) ;PM 60-89",
        "FILLMW MW[91]=MI.4.1(19) ;OP 91-119",
        ";A>P (transpose)",
        "FILLMW	MW[20]=Mi.1.20.T(10)",
        "FILLMW	MW[50]=Mi.2.20.T(10)",
        "FILLMW	MW[80]=Mi.3.20.T(10)",
        "FILLMW	MW[110]=Mi.4.20.T(10)",
        ";Add Externals  - AM",
        "MW[1] = MW[1] + MW[11]    ;EBCA_F",
        "MW[20] = MW[20] + MW[12]  ;EBCA_T",
        "MW[4] = MW[4] + MW[13]    ;EBNCA",
        "MW[5] = MW[5] + MW[14]    ;HBWCA_F",
        "MW[24] = MW[24] + MW[15]  ;HBWCA_T",
        "MW[6] = MW[6] + MW[16]    ;HBWNCA",
        "MW[7] = MW[7] + MW[17]    ;OCA_F",
        "MW[26] = MW[26] + MW[18]  ;OCA_T",
        "MW[10] = MW[10] + MW[19]  ;ONCA",
        ";Add Externals  - IP",
        "MW[31] = MW[31] + MW[41] ;EBCA_F",
        "MW[50] = MW[50] + MW[42] ;EBCA_T",
        "MW[34] = MW[34] + MW[43] ;EBNCA",
        "MW[35] = MW[35] + MW[44] ;HBWCA_F",
        "MW[54] = MW[54] + MW[45] ;HBWCA_T",
        "MW[36] = MW[36] + MW[46] ;HBWNCA",
        "MW[37] = MW[37] + MW[47] ;OCA_F",
        "MW[56] = MW[56] + MW[48] ;OCA_T",
        "MW[40] = MW[40] + MW[49] ;ONCA",
        ";Add Externals  - PM",
        "MW[61] = MW[61] + MW[71] ;EBCA_F",
        "MW[80] = MW[80] + MW[72] ;EBCA_T",
        "MW[64] = MW[64] + MW[73] ;EBNCA",
        "MW[65] = MW[65] + MW[74] ;HBWCA_F",
        "MW[84] = MW[84] + MW[75] ;HBWCA_T",
        "MW[66] = MW[66] + MW[76] ;HBWNCA",
        "MW[87] = MW[67] + MW[77] ;OCA_F",
        "MW[86] = MW[86] + MW[78] ;OCA_T",
        "MW[70] = MW[70] + MW[79] ;ONCA",
        ";Add Externals  - OP",
        "MW[91] = MW[91] + MW[101]   ;EBCA_F",
        "MW[110] = MW[110] + MW[102] ;EBCA_T",
        "MW[94] = MW[94] + MW[103]   ;EBNCA",
        "MW[95] = MW[95] + MW[104]   ;HBWCA_F",
        "MW[114] = MW[114] + MW[105] ;HBWCA_T",
        "MW[96] = MW[96] + MW[106]   ;HBWNCA",
        "MW[97] = MW[97] + MW[107] ;OCA_F",
        "MW[116] = MW[116] + MW[108] ;OCA_T",
        "MW[100] = MW[100] + MW[109] ;ONCA",
        "ENDRUN",
    ]

    with open(pathlib.Path(mats_folder, "PA2OD.s"), "w") as script:
        script.write("\n".join(to_write))

    proc_single(
        [
            f'"{exe_cube}" "{mats_folder}\\PA2OD.s" -Pvdmi /Start /Hide /HideScript',
            f'del "{mats_folder}\\*.prn"',
            f'del "{mats_folder}\\*.VAR"',
            f'del "{mats_folder}\\*.PRJ"',
            f'del "{mats_folder}\\PA2OD.s"',
        ]
    )
