

import numpy as np
import pickle as pk
import pandas as pd
import normits_demand as nd
from normits_demand import efs_constants as efs_consts
from normits_demand.utils import vehicle_occupancy as vo
from pathlib import Path

dctmode = {3: ['Car']}
dctday = {1: ['Weekday']}
purp_hb = [1, 2, 3, 4, 5, 6, 7, 8]
purp_nhb = [12, 13, 14, 15, 16, 18]
keys = range((len(purp_hb) * 2) + len(purp_nhb))
dctpurp = {}
for i in keys:
    if i < len(purp_hb):
        # print([purp_hb[i]])
        dctpurp[i] = ['hb', 'from'] + [purp_hb[i]]
    elif len(purp_hb) <= i < (len(purp_hb) * 2):
        # print([purp_hb[i-len(purp_hb)]])
        dctpurp[i] = ['hb', 'to'] + [purp_hb[i - len(purp_hb)]]
    elif i >= (len(purp_hb) * 2):
        # print([purp_nhb[i-(len(purp_hb)*2)]])
        dctpurp[i] = ['nhb', ''] + [purp_nhb[i - (len(purp_hb) * 2)]]
    else:
        print('Value outside expected range')
dctmddpurp = {1: ['HBW', 'HBW_fr', 'commute'], 2: ['HBW', 'HBW_to', 'commute'], 3: ['HBO', 'HBO_fr', 'other'],
              4: ['HBO', 'HBO_to', 'other'], 5: ['NHB', 'NHB', 'other']}
dctuc = {1: ['business'],
         2: ['commute'],
         3: ['other']}
dcttp = {1: ['AM'], 2: ['IP'], 3: ['PM'], 4: ['OP']}

unq_zones = list(range(1, 2771))


def mdd_export():
    # some notes
    with open(r'Y:\Mobile Data\Processing\dct_MDDCar.pkl', 'wb') as log:
        dctmddcar = pk.load(log)

    with open(r'Y:\Mobile Data\Processing\dctmdd_uc_split.pkl', 'rb') as log:
        dctmddsplit = pk.load(log)

    dctuc = {'1_hb_from', '1_hb_to', '2_hb_from', '2_hb_to', '3_hb_from', '3_hb_to', '1_nhb', '3_nhb'}

    # UC1 Employers Business
    (dctmmdcar[3][1][3][1]*dctmddsplit[3][1]['1_hb_fr'][1])+(dctmmdcar[3][1][4][1]*dctmddsplit[3][1]['1_hb_to'][1])+(dctmmdcar[3][1][5][1]*dctmddsplit[3][1]['1_nhb'][1])

    # UC2 Commute
    (dctmmdcar[3][1][1][1]*dctmddsplit[3][1]['2_hb_fr'][1])+(dctmmdcar[3][1][2][1]*dctmddsplit[3][1]['2_hb_to'][1])

    # UC3 Other
    (dctmmdcar[3][1][3][1]*dctmddsplit[3][1]['3_hb_fr'][1])+(dctmmdcar[3][1][4][1]*dctmddsplit[3][1]['3_hb_to'][1])+(dctmmdcar[3][1][5][1]*dctmddsplit[3][1]['3_nhb'][1])

    np.savetxt("Y:/Mobile Data/Processing/MDD_Demand/MDD_Car_T1_UC3.csv", (dct_MDDCar[3][1][3][1] + np.transpose(dct_MDDCar[3][1][4][1])), fmt='%1.5f', delimiter=',')
    np.savetxt("Y:/Mobile Data/Processing/MDD_Demand/MDD_Car_T1_UC2.csv", (dct_MDDCar[3][1][1][1] + np.transpose(dct_MDDCar[3][1][2][1])), fmt='%1.5f', delimiter=',')
    np.savetxt("Y:/Mobile Data/Processing/MDD_Demand/MDD_Car_T1_UC1.csv", (dct_MDDCar[3][1][5][1]), fmt='%1.5f', delimiter=',')


def mdd_person_export():
    # Set local variables
    version = 2
    export_folder = 'Y:/Mobile Data/Processing/NoHAM_Demand'

    # TODO: Load MMD Car pickle
    with open(r'Y:\Mobile Data\Processing\dct_MDDCar.pkl', 'rb') as log:
        dct_mdd_car = pk.load(log)

    # TODO: Loop export into PersonTrips folder with pandas out method
    for md in dctmode:
        for wd in dctday:
            for pp in dctmddpurp:
                for tp in dcttp:
                    file_path = (export_folder + '/v' + str(version) + '/PersonTrips/' +
                                 'od_' + str(dctmddpurp[pp][2]) + '_p' + str(pp) +
                                 '_yr2018_m' + str(md) +
                                 '_tp' + str(tp) + '.csv')
                    print(file_path)
                    export_array = dct_mdd_car[md][wd][pp][tp]
                    export_df = pd.DataFrame(data=export_array, index=unq_zones, columns=unq_zones)
                    export_df.to_csv(file_path)
    print('mdd person trip matrices exported')


def mdd_person_uc_export():
    # Set local variables
    version = 4
    export_folder = 'Y:/Mobile Data/Processing/2-1_MDD_Demand'

    # TODO: Load MMD Car pickle
    with open(r'Y:\Mobile Data\Processing\dct_MDDCar_expanded-v2_UC.pkl', 'rb') as log:
        dct_mdd_car_uc = pk.load(log)

    # TODO: Loop export into PersonTrips folder with pandas out method
    for md in dctmode:
        for wd in dctday:
            for uc in dctuc:
                for tp in dcttp:
                    folder_path = (export_folder + '/v' + str(version) + '/PersonTrips')
                    Path(folder_path).mkdir(parents=True, exist_ok=True)
                    file_path = (folder_path + '/' +
                                 'od_' + str(dctuc[uc][0]) + '_p' + str(uc) +
                                 '_yr2018_m' + str(md) +
                                 '_tp' + str(tp) + '.csv')
                    print(file_path)
                    export_array = dct_mdd_car_uc[md][wd][uc][tp]
                    export_df = pd.DataFrame(data=export_array, index=unq_zones, columns=unq_zones)
                    export_df.to_csv(file_path)
    print('mdd person userclass trip matrices exported')


def mdd_per_to_veh():
    # Set local variables
    version = 4
    working_folder = 'Y:/Mobile Data/Processing/2-1_MDD_Demand/'
    import_path = working_folder + '/v' + str(version) + '/PersonTrips'
    export_path = working_folder + '/v' + str(version) + '/PCUs'
    Path(export_path).mkdir(parents=True, exist_ok=True)
    import_folder = 'I:/NorMITs Demand/import'

    vo.people_vehicle_conversion(
        mat_import=import_path,
        mat_export=export_path,
        import_folder=import_folder,
        mode=3,
        method='to_vehicles',
        out_format='wide',
        hourly_average=True,
        header=True
    )


def package_mdd_pcus(totals_check=False, check_location='Y:\\Mobile Data\\Processing\\9_Totals_Check'):

    version = 4
    working_folder = 'Y:/Mobile Data/Processing/2-1_MDD_Demand'
    import_path = working_folder + '/v' + str(version) + '/PCUs'

    md = 3

    dct_mdd_pcu = {md: {}}
    dct_mdd_pcu[md][1] = {}
    for uc in dctuc:
        dct_mdd_pcu[md][1][uc] = {}
        for tp in dcttp:
            print(str(md) + '-' + str(1) + '-' + str(uc) + '-' + str(tp))
            path = (import_path + '\\'
                    + 'od_' + str(dctuc[uc][0])
                    + '_p' + str(uc)
                    + '_yr2018_m' + str(md)
                    + '_tp' + str(tp) + '.csv')
            print(path)
            mdd_car = np.genfromtxt(path,
                                      delimiter=',',
                                      skip_header=1,
                                      usecols=unq_zones)
            dct_mdd_pcu[md][1][uc][tp] = mdd_car
    # export totals if needed
    if totals_check:
        # Build totals dictionary
        dct_total = {3: {1: {}}}
        for uc in dctuc:
            dct_total[3][1][uc] = {}
            for tp in dcttp:
                print(str(3) + '-' + str(1) + '-' + str(uc) + '-' + str(tp))
                dct_total[3][1][uc][tp] = np.sum(dct_mdd_pcu[3][1][uc][tp])
        df_totals = pd.DataFrame.from_dict({(i, j, k): dct_total[i][j][k]
                                            for i in dct_total.keys()
                                            for j in dct_total[i].keys()
                                            for k in dct_total[i][j].keys()},
                                           orient='index')
        df_totals.to_csv(check_location + '\\MDD_Car_expanded-v2_UC_pcu_Totals.csv')
    # Export to MDDHW pickle file
    with open(r'Y:\Mobile Data\Processing\dct_MDDCar_expanded-v2_UC_pcu.pkl', 'wb') as log:
        pk.dump(dct_mdd_pcu, log, pk.HIGHEST_PROTOCOL)


"""
TEMPLATE CODE
def export_noham_car():
    # Set local variables
    version = 1
    export_folder = 'Y:/Mobile Data/Processing/NoHAM_Demand'

    zns = np.arange(1, 2771)
    rows = zns.reshape(2770, 1)
    columns = np.concatenate((np.array(['']), zns))

    # Load NoHAM_mddpurp pickle
    with open(r'Y:\Mobile Data\Processing\dctNoHAM_mddpurp.pkl', 'rb') as log:
        dctnoham_mddpurp = pk.load(log)

    # Loop dictionaries and save to export location
    for md in dctmode:
        for wd in dctday:
            for pp in dctmddpurp:
                for tp in dcttp:
                    file_path = (export_folder + '/v' + str(version) + '/PersonTrips/' +
                                 'od_p' + str(pp) +
                                 '_yr2018_m' + str(md) +
                                 '_tp' + str(tp) + '.csv')
                    print(file_path)
                    export_array = dctnoham_mddpurp[md][wd][pp][tp]
                    export_array = np.hstack((rows, export_array))
                    export_array = np.vstack((columns, export_array))
                    np.savetxt(file_path,
                               export_array,
                               fmt='%1.5f', delimiter=',')

def package_noham_car_pcu():
    # Set local variables
    version = 1
    import_folder = 'Y:/Mobile Data/Processing/NoHAM_Demand'

    # Loop dictionaries and save to export location
    dctnoham_mddpurp_pcu = {}
    for md in dctmode:
        dctnoham_mddpurp_pcu[md] = {}
        for wd in dctday:
            dctnoham_mddpurp_pcu[md][wd] = {}
            for pp in dctmddpurp:
                dctnoham_mddpurp_pcu[md][wd][pp] = {}
                for tp in dcttp:
                    file_path = (import_folder + '/v' + str(version) + '/PCUs/' +
                                 'od_' + str(dctmddpurp[pp][2]) + '_p' + str(pp) +
                                 '_yr2018_m' + str(md) +
                                 '_tp' + str(tp) + '.csv')
                    print(file_path)
                    noham_car = np.genfromtxt(file_path,
                                              delimiter=',')
                    dctnoham_mddpurp_pcu[md][wd][pp][tp] = noham_car

    with open(r'Y:\Mobile Data\Processing\dctNoHAM_mddpurp_pcu.pkl', 'wb') as log:
        pk.dump(dctnoham_mddpurp_pcu, log, pk.HIGHEST_PROTOCOL)
    print("noham pcus packaged")
"""


def main():
    run_mdd_export = False
    run_mdd_person_export = False
    run_mdd_person_uc_export = False
    run_mdd_per_to_veh = True
    run_package_mdd_pcus = True

    if run_mdd_export:
        mdd_export()
    if run_mdd_person_export:
        mdd_person_export()
    if run_mdd_person_uc_export:
        mdd_person_uc_export()
    if run_mdd_per_to_veh:
        mdd_per_to_veh()
    if run_package_mdd_pcus:
        package_mdd_pcus(totals_check=True)

    print("end of main")


if __name__ == '__main__':
    main()
