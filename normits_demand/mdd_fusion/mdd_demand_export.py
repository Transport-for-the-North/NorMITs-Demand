

import numpy as np
import pickle as pk
import normits_demand as nd
from normits_demand import efs_constants as efs_consts
from normits_demand.utils import vehicle_occupancy as vo

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
    # TODO: Load MMD Car pickle
    # TODO: Loop export into PersonTrips folder with pandas out method
    print('mdd person trip matrices exported')


def mdd_per_to_veh():
    import_path = r'Y:\Mobile Data\Processing\MDD_Demand\v0\PersonTrips'
    export_path = r'Y:\Mobile Data\Processing\MDD_Demand\v0\PCUs'
    import_folder = r'I:\NorMITs Demand\import'

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
    run_mdd_per_to_veh = True

    if run_mdd_export:
        mdd_export()

    if run_mdd_person_export:
        mdd_person_export()

    if run_mdd_per_to_veh:
        mdd_per_to_veh()

    print("end of main")


if __name__ == '__main__':
    main()