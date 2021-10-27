"""
TODO: add some more comments here

MND pre-processing
Aim - remove LGV, HGV and bus from MND highways records to leave car

MND pickle file from
'Y:/Mobile Data/Processing'

LGV, HGV from
'Y:/NoHAM/7.Network_Builder/4.NoHAM_IPBA-June21/Demand'

Bus & Car from
'I:/Transfer/External Model OD'

LGV Distance skim
'Y:/NoHAM/17.TUBA_Runs/-TPT/Skims/RefDist_Skims'

"""

import pandas as pd
import numpy as np
import pickle as pk

#Locations
GV_Demand_loc = 'Y:/NoHAM/7.Network_Builder/4.NoHAM_IPBA-June21/Demand'
BusCar_Demand_loc = 'I:/Transfer/External Model OD'
LGV_Dist_loc = 'Y:/NoHAM/17.TUBA_Runs/-TPT/Skims/RefDist_Skims'
gv_nts_loc = 'Y:/NTS/outputs'
MPD_Processing = 'Y:/Mobile Data/Processing'

class MDDPreProcess:
    # TODO: start then finish this
    print("only here so Pycharm is happy with the indented comments above")
    
    #MND and NTS purposes
    mndPurp = {1:'HBW_fr',2:'HBW_to',3:'HBO_fr',4:'HBO_to',5:'NHB'}
    ntsPurp = {1:['HBW',1,2],2:['HBEB',3,1],3:['HBED',3,3], #[nts]:[name,mnd,uc]
               4:['HBShop',3,3],5:['HBPB',3,3],6:['HBSoc',3,3],7:['HBVF',3,3],8:['HBHol',3,3],
               12:['NHBEB',5,1],13:['NHBED',5,3],14:['NHBShop',5,3],15:['NHBPB',5,3],16:['NHBSoc',5,3],18:['NHBHol',5,3]}
    ntsGORs = {1:['NE',1],2:['NW',2],3:['YH',3],
               4:['EM',4],5:['WM',4],6:['East',4],7:['Lon',4],8:['SE',4],9:['SW',4],
               10:['Wales',4],11:['Scot',5]}
    
    #TLD distance band
    tldDist = [0,1,2,3,4,5,6,7,8,9,10,12.5,15,17.5,20,25,30,35,40,50,75,100,150,200,250,300,400,600,999]
    tldBand = {'band':[],'lower':[],'upper':[]}
    
    for row in zip(range(1,len(tldDist)),tldDist[:-1],tldDist[1:]):
        tldBand['band'].append(row[0])
        tldBand['lower'].append(row[1])
        tldBand['upper'].append(row[2])
    #pd.Series(tldBand).head()
    #tldBand.items()
    
    #Import and pickle LGV, HGV demand
    def gv_package():
        """ packages LGV/HGV csv matrices into single nested dictionary pickle file """
        dctmode = {4: 'LGV', 5: 'HGV'}
        dctday = {1: 'Weekday'}
        dcttp = {1: 'AM', 2: 'IP', 3: 'PM'}
        dctgv = {}
    
        unq_zones = list(range(1, 2771))
    
        for md in dctmode:
            dctgv[md] = {}
            for wd in dctday:
                dctgv[md][wd] = {}
                for tp in dcttp:
                    print('+++ {} - {} - {} +++'.
                          format(dctmode[md][0], dctday[wd][0], dcttp[tp][0]))
                    gv_mat = np.genfromtxt(
                        GV_Demand_loc
                        + '/Base_2018_Hwy_TS' + str(tp) + '_i8c_' + str(dctmode[md]) + '.csv',
                        delimiter=',',
                        skip_header=0)
                    dctgv[md][wd][tp] = gv_mat
    
        with open(GV_Demand_loc + '/dctGV.pkl', 'wb') as log:
            pk.dump(dctgv, log, pk.HIGHEST_PROTOCOL)
        print("matrices packaged")
        
    gv_package()
    
    #Read GV pickle file
    dctgv = pk.load(open(GV_Demand_loc + '/dctGV.pkl', 'rb'))
    
    #GV from PCU to vehicles and time period adjustment
    """ 24hr = (3AM + 6IP + 3*PM)*op_scalar
        HGV op_scalar = 0.33
        LGV op_scalar = 0.22
        HGV PCU = 2.5 * vehicles 
        Scalar element-wise multiplication and division"""
    dctmode = {4: 'LGV', 5: 'HGV'}
    dctday = {1: 'Weekday'}
    dcttp = {1: ['AM', 3], 2: ['IP', 6], 3: ['PM', 3]}
    dctgv_vehtp = {}
    
    for md in dctmode:
        dctgv_vehtp[md] = {}
        if md == 5: 
            veh_scalar = 2.5
            op_scalar = 0.33
        else: 
            veh_scalar = 1
            op_scalar = 0.22
        for wd in dctday:
            dctgv_vehtp[md][wd] = {}
            for tp in dcttp:
                dctgv_vehtp[md][wd][tp] = (dctgv[md][wd][tp] * dcttp[tp][1]) / veh_scalar
            #Add OP period
            dctgv_vehtp[md][wd][4] = (dctgv_vehtp[md][wd][1] + dctgv_vehtp[md][wd][2] + dctgv_vehtp[md][wd][3]) * op_scalar
            #Add 24hr
            dctgv_vehtp[md][wd][5] = (dctgv_vehtp[md][wd][1] + dctgv_vehtp[md][wd][2] + 
                                      dctgv_vehtp[md][wd][3] + dctgv_vehtp[md][wd][4])
        
    """TODO:
        Apply Van occupancy to get person trips from vehicles & Split LGV into purposes using NTS tour proportions
            Import LGV distance skim & tour proportions
            Import NTS LGV occupancies by distance band
            Import zone-GOR lookup
            Assign TLD band to dist skim
            Apply purpose tour proportions by GOR to 24hr LGV matrix (note this doesn't account for land use)
            Apply LGV occupancy multipliers by purpose and TLD band
            
            Import Bus & car matrices
            Calculate mode proportions by purpose for car, bus, LGV, HGV
            
    """

    #Import LGV distance skim - convert to numpy array
    lgvDist = pd.read_csv(LGV_Dist_loc + '/NoHAM_Base_2018_TS2_v106_Dist_LGV.csv', header=None, index_col=None, low_memory=False)
    lgvDist2 = lgvDist.pivot(index=0, columns=1, values=2)
    lgvDist2 = lgvDist2.to_numpy()
    lgvDist3 = np.nan_to_num(lgvDist2)

    #Import LGV tour proportions
    lgv_tour = pd.read_excel(gv_nts_loc + '/van_tour_props_check.xlsx', sheet_name='van_tour_props', header=0, index_col=None, usecols=list(range(0,8)))
    
    #Import NTS LGV occupancies by distance band
    lgv_oc = pd.read_excel(gv_nts_loc + '/van_co_check.xlsx', sheet_name='van_co2', header=0, index_col=None)
    
    #Import zone-GOR lookup
    znList = pd.read_csv(MPD_Processing +
                         '/NoHAM_zone-GOR.csv',
                         header=0)
    
    # Join tour props to zone list
    znList_tour = pd.merge(znList, lgv_tour, how='right', left_on=['TFN_GOR2'], right_on=['agg_gor_from'])
    # Remove extra data columns
    znList_tourtidy = znList_tour[['Zone', 'p', 'trip_direction', 'start_time', 'prop']]
    # Import purp list
    toMndPurpList = pd.read_csv(MPD_Processing +
                         '/toMndPurp.csv',
                         header=0)
    # add mnd purpose to tour props
    znList_tourTidyPurp = pd.merge(znList_tourtidy, toMndPurpList, how='left', on=['p', 'trip_direction'])
    # Reset index to mnd purpose & start time
    znList_tourTidyPurp = znList_tourTidyPurp.set_index(['start_time', 'MndPurp'])
    # Print example zone list
    # print(znList_tourTidyPurp.loc[([1], [3]), ['Zone', 'prop']].sort_values(by=['Zone']).to_numpy)
    # znList_tourTidyPurp_check = znList_tourTidyPurp.loc[([1], [3]), ['Zone', 'prop']].sort_values(by=['Zone']).to_numpy
    # znList_tourTidyPurp.loc[([1], [3]), ['Zone', 'prop']].to_csv(MPD_Processing + '/LGV_prop.csv', index=False, header=True)
    """ create test array
    temp_array = np.array([[1, 2, 3], [2, 4, 6], [3, 6, 9]])
    temp_array2 = np.array([0, 1, 2])
    temp_array3 = temp_array * temp_array2[:, np.newaxis]
    print(temp_array3)
    # create test nested dictionaries
    dctgv_vehtp = {}
    for md in dctmode:
        dctgv_vehtp[md] = {}
        for wd in dctday:
            dctgv_vehtp[md][wd] = {}
            for tp in dcttp:
                dctgv_vehtp[md][wd][tp] = temp_array
    """
    # duplicate 24hr into tps
    dctlgv_tours= {}
    dctlgv_tours[4] = {}
    for wd in dctday:
        dctlgv_tours[4][wd] = {}
        for tp in dcttp:
            dctlgv_tours[4][wd][tp] = {}
            for p in mndPurp:
                dctlgv_tours[4][wd][tp][p] = {}
                dctlgv_tours[4][wd][tp][p] = dctgv_vehtp[4][wd][5]

    for wd in dctday:
        for tp in dcttp:
            for p in mndPurp:
                tourProp1 = znList_tourTidyPurp.loc[([tp], [p]), ['Zone', 'prop']].sort_values(by=['Zone'])
                tourProp2 = tourProp1.head(2770)
                tourProp3 = tourProp2['prop'].to_numpy()
                dctlgv_tours[md][wd][tp][p] = dctlgv_tours[md][wd][tp][p] * tourProp3[:, np.newaxis]

    # Import dbands
    dbandList = pd.read_csv(MPD_Processing +
                         '/dbands.csv',
                         header=0)
    # Covert distance matrix to dband integers
    # TODO: update dband list to correct bands
    lgvDistBand = np.digitize(lgvDist3, dbandList['Min'])
    # Loop indexed lgv oc list and convert dband integers with occupancy factor
    # TODO: find code used to create & index lgv_oc
    # TODO: duplicate dband integer matrices into nested purpose dict
    # TODO: build loop to convert by purpose
    for tp:
        for band, purp, oc in lgv_oc[lgv_oc['purpose'] == 'HBW'].itertuples(index=False):
            print(type(band), oc)
            if purp in ['HBW'] p = 1 ifelse
            lgvDistBand[tp][p] = np.where(lgvDistBand == band, oc, lgvDistBand)
    # TODO: apply occupancy adjustment to TP LGVs

    # TODO: get and format Bus data
    # TODO: build all mode dictionary
    # TODO: check tld by mode/purpose
    # TODO: check mode share by origin/purpose

    
