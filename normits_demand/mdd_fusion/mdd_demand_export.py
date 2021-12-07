

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

def mdd_per_to_veh():
    import_path = r'Y:\Mobile Data\Processing\MDD_Demand\v0\PersonTrips'
    export_path = r'Y:\Mobile Data\Processing\MDD_Demand\v0\PCUs'
    import_folder = r'I:\NorMITs Demand\import'

    vo.people_vehicle_conversion(
        mat_import=import_path,
        mat_export=export_path,
        import_folder=import_folder,
        mode=3,
        method='to_vehicle',
        out_format='wide'
    )

def main():
    run_mdd_Export = False
    run_mdd_per_to_veh = True

    if run_noham_car_package:
        noham_car_package()

    if run_mdd_per_to_veh:
        mdd_per_to_veh()

    print("end of main")


if __name__ == '__main__':
    main()