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

#MND and NTS purposes
mndPurp = {1:['HBW','HBW_fr'],2:['HBW','HBW_to'],3:['HBO','HBO_fr'],4:['HBO','HBO_to'],5:['NHB','NHB']}
ntsPurp = {1:['HBW',1,2],2:['HBEB',3,1],3:['HBED',3,3], #[nts]:[name,mnd,uc]
           4:['HBShop',3,3],5:['HBPB',3,3],6:['HBSoc',3,3],7:['HBVF',3,3],8:['HBHol',3,3],
           12:['NHBEB',5,1],13:['NHBED',5,3],14:['NHBShop',5,3],15:['NHBPB',5,3],16:['NHBSoc',5,3],18:['NHBHol',5,3]}
ntsGORs = {1:['NE',1],2:['NW',2],3:['YH',3],
           4:['EM',4],5:['WM',4],6:['East',4],7:['Lon',4],8:['SE',4],9:['SW',4],
           10:['Wales',4],11:['Scot',5]}

# looping dictionaries
dctmddpurp = {1: ['HBW', 'HBW_fr'], 2: ['HBW', 'HBW_to'],
              3: ['HBO', 'HBO_fr'], 4: ['HBO', 'HBO_to'],
              5: ['NHB', 'NHB']}
dcttp = {1: ['AM', 3], 2: ['IP', 6], 3: ['PM', 3], 4: ['OP', 4]}

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


def gv_processing(totals_check=False, check_location='Y:\\Mobile Data\\Processing\\9_Totals_Check'):
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

    with open(r'Y:\Mobile Data\Processing\dctGV_24hr.pkl', 'wb') as log:
        pk.dump(dctgv_vehtp, log, pk.HIGHEST_PROTOCOL)

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
    znList_tourtidy = znList_tour[['Zone', 'p', 'trip_direction', 'start_time', 'prop_p']]
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
    dctlgv_tours= {7: {}}
    for wd in dctday:
        dctlgv_tours[7][wd] = {}
        for pp in mndPurp:
            dctlgv_tours[7][wd][pp] = {}
            for tp in dcttp:
                dctlgv_tours[7][wd][pp][tp] = {}
                dctlgv_tours[7][wd][pp][tp] = dctgv_vehtp[4][wd][5]

    for wd in dctday:
        for pp in mndPurp:
            for tp in dcttp:
                tourProp1 = znList_tourTidyPurp.loc[([tp], [pp]), ['Zone', 'prop_p']].sort_values(by=['Zone'])
                tourProp2 = tourProp1.head(2770)
                tourProp3 = tourProp2['prop_p'].to_numpy()
                dctlgv_tours[7][wd][pp][tp] = (dctlgv_tours[7][wd][pp][tp] * tourProp3[:, np.newaxis])

    # Export totals if required
    if totals_check:
        # Save off dctlgv_tours
        with open(r'Y:\Mobile Data\Processing\dct_LGV_tours.pkl', 'wb') as log:
            pk.dump(dctlgv_tours, log, pk.HIGHEST_PROTOCOL)
        # Build totals dictionary
        dct_total = {7: {1: {}}}
        for pp in mndPurp:
            dct_total[7][1][pp] = {}
            for tp in dcttp:
                print(str(7) + '-' + str(1) + '-' + str(pp) + '-' + str(tp))
                dct_total[7][1][pp][tp] = np.sum(dctlgv_tours[7][1][pp][tp][:2770, :2770])
        # Convert dictionary to dataframe
        df_totals = pd.DataFrame.from_dict({(i, j, k): dct_total[i][j][k]
                                            for i in dct_total.keys()
                                            for j in dct_total[i].keys()
                                            for k in dct_total[i][j].keys()},
                                           orient='index')
        # Export to csv
        df_totals.to_csv(check_location + '\\LGV_tours_Totals.csv')
    # Export compiled dictionary


    # Import dbands
    '''
    dbandList = pd.read_csv(MPD_Processing +
                         '/dbands.csv',
                         header=0)
    lgvDistBand = np.digitize(lgvDist3, dbandList['Min'])
    '''
    # Covert distance matrix to dband integers
    # miles breaks = c(-1, 1, 2, 3, 5, 10, 15, 25, 35, 50, 100, 200, Inf)
    lgvDistBand = np.digitize(lgvDist3, pd.Series(list(map(int, tldBand['lower']))))

    # Loop indexed lgv oc list and convert dband integers with occupancy factor
    # Duplicate dband integer matrices into nested purpose dict
    dctlgv_oc = {7: {}}
    for wd in dctday:
        dctlgv_oc[7][wd] = {}
        for pp in mndPurp:
            dctlgv_oc[7][wd][pp] = {}
            for tp in dcttp:
                dctlgv_oc[7][wd][pp][tp] = {}
                dctlgv_oc[7][wd][pp][tp] = lgvDistBand

    # Loops purpose and replaces dband integer with occupancy
    for wd in dctday:
        for pp in mndPurp:
            purpose = ''.join(mndPurp[pp][:1])
            print(purpose)
            for tp in dcttp:
                for band, purp, oc in lgv_oc[lgv_oc['purpose'] == purpose].itertuples(index=False):
                    print(band, purp, oc)
                    dctlgv_oc[7][wd][pp][tp] = np.where(dctlgv_oc[7][wd][pp][tp] == band, oc, dctlgv_oc[7][wd][pp][tp])

    # Apply occupancy adjustment to TP LGVs
    dctlgv_persons = {7: {}}
    for wd in dctday:
        dctlgv_persons[7][wd] = {}
        for pp in mndPurp:
            dctlgv_persons[7][wd][pp] = {}
            for tp in dcttp:
                dctlgv_persons[7][wd][pp][tp] = {}
                dctlgv_persons[7][wd][pp][tp] = (dctlgv_tours[7][wd][pp][tp] * dctlgv_oc[7][wd][pp][tp])

    if totals_check:
        # Save off dctlgv_tours
        # Build totals dictionary
        dct_total = {7: {1: {}}}
        for pp in mndPurp:
            dct_total[7][1][pp] = {}
            for tp in dcttp:
                print(str(7) + '-' + str(1) + '-' + str(pp) + '-' + str(tp))
                dct_total[7][1][pp][tp] = np.sum(dctlgv_persons[7][1][pp][tp][:2770, :2770])
        # Convert dictionary to dataframe
        df_totals = pd.DataFrame.from_dict({(i, j, k): dct_total[i][j][k]
                                            for i in dct_total.keys()
                                            for j in dct_total[i].keys()
                                            for k in dct_total[i][j].keys()},
                                           orient='index')
        # Export to csv
        df_totals.to_csv(check_location + '\\LGV_person_Totals.csv')

    # TODO: save updated lgv dictionary
    with open(r'Y:\Mobile Data\Processing\dctlgv_persons.pkl', 'wb') as log:
        pk.dump(dctlgv_persons, log, pk.HIGHEST_PROTOCOL)


def produce_combined(totals_check=False, check_location='Y:\\Mobile Data\\Processing\\9_Totals_Check'):
    # TODO: build all mode dictionary
    # Import
    with open(r'Y:\Mobile Data\Processing\dctNoHAM_mddpurp.pkl', 'rb') as log:
        dctnoham = pk.load(log)

    with open(r'Y:\Mobile Data\Processing\dctNoTEM_mddpurp.pkl', 'rb') as log:
        dctnotem = pk.load(log)

    with open(r'Y:\Mobile Data\Processing\dctGV_24hr.pkl', 'rb') as log:
        dctgv = pk.load(log)

    with open(r'Y:\Mobile Data\Processing\dctlgv_persons.pkl', 'rb') as log:
        dctlgv = pk.load(log)

    # TODO: define all mode dictionary
    dctmode = {3: 'Car', 5: 'Bus', 7: 'LGV', 8: 'HGV'}
    dctday = {1: 'Weekday'}
    dctmddpurp = {1: ['HBW', 'HBW_fr'], 2: ['HBW', 'HBW_to'], 3: ['HBO', 'HBO_fr'], 4: ['HBO', 'HBO_to'],
               5: ['NHB', 'NHB']}
    dcthgvpurp = {5: ['NHB', 'NHB']}
    dcttp = {1: ['AM', 3], 2: ['IP', 6], 3: ['PM', 3], 4: ['OP', 4]}

    # TODO: amend below
    dctcombined = {}
    for md in dctmode:
        if md in [3, 5, 7]:
            dctcombined[md] = {}
            for wd in dctday:
                dctcombined[md][wd] = {}
                for pp in dctmddpurp:
                    dctcombined[md][wd][pp] = {}
                    for tp in dcttp:
                        print(str(md) + '-' + str(wd) + '-' + str(pp) + '-' + str(tp))
                        if md == 3:
                            print('Car')
                            print(np.sum(dctnoham[md][wd][pp][tp]))
                            dctcombined[md][wd][pp][tp] = dctnoham[md][wd][pp][tp]
                        elif md == 5:
                            print('Bus')
                            print(np.sum(dctnotem[md][wd][pp][tp]))
                            dctcombined[md][wd][pp][tp] = dctnotem[md][wd][pp][tp]
                        elif md == 7:
                            print('LGV')
                            print(np.sum(dctlgv[md][wd][pp][tp]))
                            dctcombined[md][wd][pp][tp] = dctlgv[md][wd][pp][tp]
        elif md in [8]:
            md_in = 5
            dctcombined[md] = {}
            for wd in dctday:
                dctcombined[md][wd] = {}
                for pp in dcthgvpurp:
                    pp_in = 1
                    dctcombined[md][wd][pp] = {}
                    for tp in dcttp:
                        print(str(md_in) + '-' + str(wd) + '-' + str(pp_in) + '-' + str(tp))
                        print(str(md) + '-' + str(wd) + '-' + str(pp) + '-' + str(tp))
                        print('HGV')
                        print(np.sum(dctgv[md_in][pp_in][tp]))
                        dctcombined[md][wd][pp][tp] = dctgv[md_in][pp_in][tp]
    # Export totals if required
    if totals_check:
        # Build totals dictionary
        dct_total = {}
        for md in [3, 5, 7]:
            dct_total[md] = {}
            for wd in dctday:
                dct_total[md][wd] = {}
                for pp in dctmddpurp:
                    dct_total[md][wd][pp] = {}
                    for tp in dcttp:
                        print(str(md) + '-' + str(wd) + '-' + str(pp) + '-' + str(tp))
                        dct_total[md][wd][pp][tp] = np.sum(dctcombined[md][wd][pp][tp][:2770, :2770])
        # Convert dictionary to dataframe
        df_totals = pd.DataFrame.from_dict({(i, j, k): dct_total[i][j][k]
                                            for i in dct_total.keys()
                                            for j in dct_total[i].keys()
                                            for k in dct_total[i][j].keys()},
                                           orient='index')
        # Export to csv
        df_totals.to_csv(check_location + '\\CombinedMode_Person_Totals.csv')
    # Export combined mode
    with open(r'Y:\Mobile Data\Processing\dct_CombinedMode.pkl', 'wb') as log:
        pk.dump(dctcombined, log, pk.HIGHEST_PROTOCOL)


def calc_mode_share():

    dctmode = {3: 'Car', 5: 'Bus', 7: 'LGV', 8: 'HGV'}
    dctday = {1: 'Weekday'}
    dctmddpurp = {1: ['HBW', 'HBW_fr'], 2: ['HBW', 'HBW_to'], 3: ['HBO', 'HBO_fr'], 4: ['HBO', 'HBO_to'],
                  5: ['NHB', 'NHB']}
    dcthgvpurp = {5: ['NHB', 'NHB']}
    dcttp = {1: ['AM', 3], 2: ['IP', 6], 3: ['PM', 3], 4: ['OP', 4]}

    with open(r'Y:\Mobile Data\Processing\dct_CombinedMode.pkl', 'rb') as log:
        dctcombined = pk.load(log)

    # Sum mode trips to calc totals
    dcttotal = {}
    for wd in dctday:
        dcttotal[wd] = {}
        for pp in dctmddpurp:
            dcttotal[wd][pp] = {}
            for tp in dcttp:
                print(str(wd) + '-' + str(pp) + '-' + str(tp))
                if pp == 5:
                    print('NHB')
                    dcttotal[wd][pp][tp] = (dctcombined[3][wd][pp][tp] + dctcombined[5][wd][pp][tp] +
                                            dctcombined[7][wd][pp][tp] + dctcombined[8][wd][pp][tp])
                elif pp in [1, 2, 3, 4]:
                    print('Car, Bus & LGV')
                    dcttotal[wd][pp][tp] = (dctcombined[3][wd][pp][tp] + dctcombined[5][wd][pp][tp] +
                                            dctcombined[7][wd][pp][tp])
                else:
                    print('purpose variable outside anticipated range')

    # Calculate wider share
    dctglobalshare = {}
    for md in dctmode:
        if md in [3, 5, 7]:
            dctglobalshare[md] = {}
            for wd in dctday:
                dctglobalshare[md][wd] = {}
                for pp in dctmddpurp:
                    dctglobalshare[md][wd][pp] = {}
                    for tp in dcttp:
                        print(str(md) + '-' + str(wd) + '-' + str(pp) + '-' + str(tp))
                        dctglobalshare[md][wd][pp][tp] = (np.sum(dctcombined[md][wd][pp][tp]) / np.sum(dcttotal[wd][pp][tp]))
                        dctglobalshare[md][wd][pp][tp] = np.nan_to_num(dctglobalshare[md][wd][pp][tp])
        elif md in [8]:
            dctglobalshare[md] = {}
            for wd in dctday:
                dctglobalshare[md][wd] = {}
                for pp in dcthgvpurp:
                    dctglobalshare[md][wd][pp] = {}
                    for tp in dcttp:
                        print(str(md) + '-' + str(wd) + '-' + str(pp) + '-' + str(tp))
                        dctglobalshare[md][wd][pp][tp] = (np.sum(dctcombined[md][wd][pp][tp]) / np.sum(dcttotal[wd][pp][tp]))
                        dctglobalshare[md][wd][pp][tp] = np.nan_to_num(dctglobalshare[md][wd][pp][tp])

    # Calculate mode share
    dctshare = {}
    for md in dctmode:
        if md in [3, 5, 7]:
            dctshare[md] = {}
            for wd in dctday:
                dctshare[md][wd] = {}
                for pp in dctmddpurp:
                    dctshare[md][wd][pp] = {}
                    for tp in dcttp:
                        print(str(md) + '-' + str(wd) + '-' + str(pp) + '-' + str(tp))
                        dctshare[md][wd][pp][tp] = (dctcombined[md][wd][pp][tp]/dcttotal[wd][pp][tp])
                        dctshare[md][wd][pp][tp] = np.nan_to_num(dctshare[md][wd][pp][tp], nan=dctglobalshare[md][wd][pp][tp])
        elif md in [8]:
            dctshare[md] = {}
            for wd in dctday:
                dctshare[md][wd] = {}
                for pp in dcthgvpurp:
                    dctshare[md][wd][pp] = {}
                    for tp in dcttp:
                        print(str(md) + '-' + str(wd) + '-' + str(pp) + '-' + str(tp))
                        dctshare[md][wd][pp][tp] = (dctcombined[md][wd][pp][tp] / dcttotal[wd][pp][tp])
                        dctshare[md][wd][pp][tp] = np.nan_to_num(dctshare[md][wd][pp][tp], nan=dctglobalshare[md][wd][pp][tp])

    with open(r'Y:\Mobile Data\Processing\dct_ModeShare.pkl', 'wb') as log:
        pk.dump(dctshare, log, pk.HIGHEST_PROTOCOL)

    # Sum check mode shares
    dctcheck = {}
    for wd in dctday:
        dctcheck[wd] = {}
        for pp in dctmddpurp:
            dctcheck[wd][pp] = {}
            for tp in dcttp:
                print(str(wd) + '-' + str(pp) + '-' + str(tp))
                if pp == 5:
                    print('NHB')
                    dctcheck[wd][pp][tp] = (dctshare[3][wd][pp][tp] + dctshare[5][wd][pp][tp] +
                                            dctshare[7][wd][pp][tp] + dctshare[8][wd][pp][tp])
                    print(np.sum(dctcheck[wd][pp][tp]))
                elif pp in [1, 2, 3, 4]:
                    print('Car, Bus & LGV')
                    dctcheck[wd][pp][tp] = (dctshare[3][wd][pp][tp] + dctshare[5][wd][pp][tp] +
                                            dctshare[7][wd][pp][tp])
                    print(np.sum(dctcheck[wd][pp][tp]))
                else:
                    print('purpose variable outside anticipated range')

    with open(r'Y:\Mobile Data\Processing\dct_check.pkl', 'wb') as log:
        pk.dump(dctcheck, log, pk.HIGHEST_PROTOCOL)


def split_mdd_hw(totals_check=False, check_location='Y:\\Mobile Data\\Processing\\9_Totals_Check'):
    # Import previous mdd data pickle file
    with open(r'Y:\Mobile Data\Processing\dctMODD_trip.pkl', 'rb') as log:
        dct_MODD = pk.load(log)
    # Export totals if required
    if totals_check:
        # Build totals dictionary
        dct_total = {3: {1: {}}}
        for pp in dctmddpurp:
            dct_total[3][1][pp] = {}
            for tp in dcttp:
                print(str(3) + '-' + str(1) + '-' + str(pp) + '-' + str(tp))
                dct_total[3][1][pp][tp] = np.sum(dct_MODD[1][2][pp][tp][:2770, :2770])
        # Convert dictionary to dataframe
        df_totals = pd.DataFrame.from_dict({(i, j, k): dct_total[i][j][k]
                                            for i in dct_total.keys()
                                            for j in dct_total[i].keys()
                                            for k in dct_total[i][j].keys()},
                                           orient='index')
        # Export to csv
        df_totals.to_csv(check_location + '\\MODD_HW_Totals.csv')
    # Build dictionary with future matching mode and weekday keys
    dct_MDDHW = {3: {}}
    dct_MDDHW[3][1] = {}
    for pp in dctmddpurp:
        dct_MDDHW[3][1][pp] = {}
        for tp in dcttp:
            dct_MDDHW[3][1][pp][tp] = (dct_MODD[1][2][pp][tp][:2770, :2770]/240)
    # Export revised totals if required
    if totals_check:
        # Build totals dictionary
        dct_total = {3: {1: {}}}
        for pp in dctmddpurp:
            dct_total[3][1][pp] = {}
            for tp in dcttp:
                print(str(3) + '-' + str(1) + '-' + str(pp) + '-' + str(tp))
                dct_total[3][1][pp][tp] = np.sum(dct_MDDHW[3][1][pp][tp])
        df_totals = pd.DataFrame.from_dict({(i, j, k): dct_total[i][j][k]
                                            for i in dct_total.keys()
                                            for j in dct_total[i].keys()
                                            for k in dct_total[i][j].keys()},
                                           orient='index')
        df_totals.to_csv(check_location + '\\MDD_HW_Totals-Average_WeekDay.csv')
    # Export to MDDHW pickle file
    with open(r'Y:\Mobile Data\Processing\dct_MDDHW.pkl', 'wb') as log:
        pk.dump(dct_MDDHW, log, pk.HIGHEST_PROTOCOL)


def factor_mdd():
    dctmddpurp = {1: ['HBW', 'HBW_fr'], 2: ['HBW', 'HBW_to'], 3: ['HBO', 'HBO_fr'], 4: ['HBO', 'HBO_to'],
                  5: ['NHB', 'NHB']}
    dcttp = {1: ['AM', 3], 2: ['IP', 6], 3: ['PM', 3], 4: ['OP', 4]}

    with open(r'Y:\Mobile Data\Processing\dct_MDDHW.pkl', 'rb') as log:
        dct_mddhw = pk.load(log)

    with open(r'Y:\Mobile Data\Processing\dct_ModeShare.pkl', 'rb') as log:
        dctmodeshare = pk.load(log)

    dct_mdd_car = {3: {}}
    dct_mdd_car[3][1] = {}
    for pp in dctmddpurp:
        dct_mdd_car[3][1][pp] = {}
        for tp in dcttp:
            dct_mdd_car[3][1][pp][tp] = (dct_mddhw[3][1][pp][tp] * dctmodeshare[3][1][pp][tp])

    with open(r'Y:\Mobile Data\Processing\dct_MDDCar.pkl', 'wb') as log:
        pk.dump(dct_mdd_car, log, pk.HIGHEST_PROTOCOL)
    # TODO: convert to PCUs
    # TODO: assign
    # TODO: check tld by mode/purpose
    # TODO: check mode share by origin/purpose


def expand_mdd(totals_check=False, check_location='Y:\\Mobile Data\\Processing\\9_Totals_Check'):
    dctmddpurp = {1: ['HBW', 'HBW_fr'], 2: ['HBW', 'HBW_to'], 3: ['HBO', 'HBO_fr'], 4: ['HBO', 'HBO_to'],
                  5: ['NHB', 'NHB']}
    dcttp = {1: ['AM', 3], 2: ['IP', 6], 3: ['PM', 3], 4: ['OP', 4]}
    # import input dictionaries
    mdd_car_file = 'Y:\\Mobile Data\\Processing\\dct_MDDCar.pkl'
    with open(mdd_car_file, 'rb') as log:
        dct_mdd_car = pk.load(log)
    mdd_global_factor_file = 'Y:\\Mobile Data\\Processing\\dict_mnfactor_arrays_v1.pkl'
    with open(mdd_global_factor_file, 'rb') as log:
        mdd_global_factor = pk.load(log)
    dct_mdd_expanded = {3: {}}
    dct_mdd_expanded[3][1] = {}
    for pp in dctmddpurp:
        dct_mdd_expanded[3][1][pp] = {}
        for tp in dcttp:
            dct_mdd_expanded[3][1][pp][tp] = (dct_mdd_car[3][1][pp][tp] * mdd_global_factor[3][1][pp][tp])
    # If required export totals
    if totals_check:
        # Build totals dictionary
        dct_total = {3: {1: {}}}
        for pp in dctmddpurp:
            dct_total[3][1][pp] = {}
            for tp in dcttp:
                print(str(3) + '-' + str(1) + '-' + str(pp) + '-' + str(tp))
                dct_total[3][1][pp][tp] = np.sum(dct_mdd_expanded[3][1][pp][tp])
        df_totals = pd.DataFrame.from_dict({(i, j, k): dct_total[i][j][k]
                                            for i in dct_total.keys()
                                            for j in dct_total[i].keys()
                                            for k in dct_total[i][j].keys()},
                                           orient='index')
        df_totals.to_csv(check_location + '\\MDD_Car_expanded_Totals-v2.csv')
    with open(r'Y:\Mobile Data\Processing\dct_MDDCar_expanded-v2.pkl', 'wb') as log:
        pk.dump(dct_mdd_expanded, log, pk.HIGHEST_PROTOCOL)


def convert_mdd_purp_to_noham_uc(totals_check=False, check_location='Y:\\Mobile Data\\Processing\\9_Totals_Check'):
    dctmddpurp = {1: ['HBW', 'HBW_fr', 'Commute'], 2: ['HBW', 'HBW_to', 'Commute'], 3: ['HBO', 'HBO_fr', 'Other'],
                  4: ['HBO', 'HBO_to', 'Other'], 5: ['NHB', 'NHB', 'Other']}
    dctmddpurpuc = {1: ['HBW', 'HBW_fr', 'Commute'], 2: ['HBW', 'HBW_to', 'Commute'],
                    31: ['HBO', 'HBO_fr', 'Other'], 33: ['HBO', 'HBO_fr', 'Other'],
                    41: ['HBO', 'HBO_to', 'Other'], 43: ['HBO', 'HBO_to', 'Other'],
                    51: ['NHB', 'NHB', 'Other'], 53: ['NHB', 'NHB', 'Other']}
    dctuc = {1: ['Business'],
             2: ['Commute'],
             3: ['Other']}
    dcttp = {1: ['AM', 3], 2: ['IP', 6], 3: ['PM', 3], 4: ['OP', 4]}

    # import input dictionaries
    #mdd_car_file = 'Y:\\Mobile Data\\Processing\\dct_MDDCar_expanded.pkl'
    #mdd_car_file = 'Y:\\Mobile Data\\Processing\\dct_MDDHW.pkl'
    mdd_car_file = 'Y:\\Mobile Data\\Processing\\dct_MDDCar_expanded-v2.pkl'
    with open(mdd_car_file, 'rb') as log:
        dct_mdd_car = pk.load(log)
    """
    mdd_car_file = 'Y:\\Mobile Data\\Processing\\dct_MDDCar.pkl'
    with open(mdd_car_file, 'rb') as log:
        dct_mdd_car = pk.load(log)
    """
    mdd_uc_split_file = 'Y:\\Mobile Data\\Processing\\dctmdd_uc_split.pkl'
    with open(mdd_uc_split_file, 'rb') as log:
        dct_mdd_uc_split = pk.load(log)

    dct_mdd_demand_uc = {3: {}}
    dct_mdd_demand_uc[3][1] = {}
    for pp in dctmddpurpuc:
        dct_mdd_demand_uc[3][1][pp] = {}
        purpose = (3 if pp in [31] else
                   3 if pp in [33] else
                   4 if pp in [41] else
                   4 if pp in [43] else
                   5 if pp in [51] else
                   5 if pp in [53] else
                   1 if pp in [1] else
                   2 if pp in [2] else
                   6)
        uc_split = ('1_hb_from' if pp in [31] else
                    '3_hb_from' if pp in [33] else
                    '1_hb_to' if pp in [41] else
                    '3_hb_to' if pp in [43] else
                    '1_nhb' if pp in [51] else
                    '3_nhb' if pp in [53] else
                    'commute' if pp in [1, 2] else
                    'missing')
        for tp in dcttp:
            if pp in [1, 2]:
                # multiply by 1
                dct_mdd_demand_uc[3][1][pp][tp] = dct_mdd_car[3][1][purpose][tp] * 1
            else:
                dct_mdd_demand_uc[3][1][pp][tp] = dct_mdd_car[3][1][purpose][tp] * dct_mdd_uc_split[3][1][uc_split][tp]

    temp_array = np.zeros((2770, 2770))
    dct_mdd_demand_uc_comb = {3: {}}
    dct_mdd_demand_uc_comb[3][1] = {}
    for uc in dctuc:
        dct_mdd_demand_uc_comb[3][1][uc] = {}
        for tp in dcttp:
            dct_mdd_demand_uc_comb[3][1][uc][tp] = temp_array

    for pp in dctmddpurpuc:
        userclass = (1 if pp in [31, 41, 51] else
                     2 if pp in [1, 2] else
                     3 if pp in [33, 43, 53] else
                     4)
        for tp in dcttp:
            dct_mdd_demand_uc_comb[3][1][userclass][tp] = (dct_mdd_demand_uc_comb[3][1][userclass][tp] + dct_mdd_demand_uc[3][1][pp][tp])
    # If required export totals
    if totals_check:
        # Build totals dictionary
        dct_total = {3: {1: {}}}
        for uc in dctuc:
            dct_total[3][1][uc] = {}
            for tp in dcttp:
                print(str(3) + '-' + str(1) + '-' + str(uc) + '-' + str(tp))
                dct_total[3][1][uc][tp] = np.sum(dct_mdd_demand_uc_comb[3][1][uc][tp])
        df_totals = pd.DataFrame.from_dict({(i, j, k): dct_total[i][j][k]
                                            for i in dct_total.keys()
                                            for j in dct_total[i].keys()
                                            for k in dct_total[i][j].keys()},
                                           orient='index')
        df_totals.to_csv(check_location + '\\MDD_HW_Totals.csv')
    # Export to MDDHW pickle file
    with open(r'Y:\Mobile Data\Processing\dct_MDDCar_expanded-v2_UC.pkl', 'wb') as log:
        pk.dump(dct_mdd_demand_uc_comb, log, pk.HIGHEST_PROTOCOL)
    """
    with open(r'Y:\Mobile Data\Processing\dct_MDDHW.pkl', 'wb') as log:
        pk.dump(dct_mdd_demand_uc_comb, log, pk.HIGHEST_PROTOCOL)

    with open(r'Y:\Mobile Data\Processing\dct_MDDCar_UC.pkl', 'wb') as log:
        pk.dump(dct_mdd_demand_uc_comb, log, pk.HIGHEST_PROTOCOL)
    """

def main():

    run_gv_package = False
    run_gv_processing = False
    run_produce_combined = False
    run_calc_mode_share = False
    run_split_mdd_hw = False
    run_factor_mdd = True

    if run_gv_package:
        gv_package()

    if run_gv_processing:
        gv_processing()

    if run_produce_combined:
        produce_combined()

    if run_calc_mode_share:
        calc_mode_share()

    if run_split_mdd_hw:
        split_mdd_hw()

    if run_factor_mdd:
        factor_mdd()

    print("end of main")


if __name__ == '__main__':
    main()
