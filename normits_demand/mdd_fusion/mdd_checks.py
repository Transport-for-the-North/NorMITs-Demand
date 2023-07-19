

import numpy as np
import pickle as pk
import pandas as pd

dctmode = {3: 'Car', 5: 'Bus', 7: 'LGV', 8: 'HGV'}
dctday = {1: 'Weekday'}
dctmddpurp = {1: ['HBW', 'HBW_fr'], 2: ['HBW', 'HBW_to'], 3: ['HBO', 'HBO_fr'], 4: ['HBO', 'HBO_to'],
              5: ['NHB', 'NHB']}
dcthgvpurp = {5: ['NHB', 'NHB']}
dcttp = {1: ['AM', 3], 2: ['IP', 6], 3: ['PM', 3]}

def noham_combined_totals():
    # Import combined dict
    with open(r'Y:\Mobile Data\Processing\dct_CombinedMode.pkl', 'rb') as log:
        dct_combined = pk.load(log)

    dct_total = {}
    for md in dctmode:
        if md in [3, 5, 7]:
            dct_total[md] = {}
            for wd in dctday:
                dct_total[md][wd] = {}
                for pp in dctmddpurp:
                    dct_total[md][wd][pp] = {}
                    for tp in dcttp:
                        print(str(md) + '-' + str(wd) + '-' + str(pp) + '-' + str(tp))
                        dct_total[md][wd][pp][tp] = np.sum(dct_combined[md][wd][pp][tp])
        elif md in [8]:
            dct_total[md] = {}
            for wd in dctday:
                dct_total[md][wd] = {}
                for pp in dcthgvpurp:
                    dct_total[md][wd][pp] = {}
                    for tp in dcttp:
                        print(str(md) + '-' + str(wd) + '-' + str(pp) + '-' + str(tp))
                        dct_total[md][wd][pp][tp] = np.sum(dct_combined[md][wd][pp][tp])

    df_totals = pd.DataFrame.from_dict({(i, j, k): dct_total[i][j][k]
                                        for i in dct_total.keys()
                                        for j in dct_total[i].keys()
                                        for k in dct_total[i][j].keys()},
                                       orient='index')

    df_totals.to_csv(r'Y:\Mobile Data\Processing\NoHAM_Combined_Totals.csv')


def mdd_totals():
    # Import combined dict
    with open(r'Y:\Mobile Data\Processing\dct_CombinedMode.pkl', 'rb') as log:
        dct_combined = pk.load(log)


def mdd_hw_totals():
    dcttp = {1: ['AM', 3], 2: ['IP', 6], 3: ['PM', 3], 4: ['OP', 12]}
    # Import combined dict
    with open(r'Y:\Mobile Data\Processing\dct_MDDHW_incOP.pkl', 'rb') as log:
        dct_mddhw = pk.load(log)

    dct_total = {3: {1: {}}}
    for pp in dctmddpurp:
        dct_total[3][1][pp] = {}
        for tp in dcttp:
            print(str(3) + '-' + str(1) + '-' + str(pp) + '-' + str(tp))
            dct_total[3][1][pp][tp] = np.sum(dct_mddhw[3][1][pp][tp])

    df_totals = pd.DataFrame.from_dict({(i, j, k): dct_total[i][j][k]
                                        for i in dct_total.keys()
                                        for j in dct_total[i].keys()
                                        for k in dct_total[i][j].keys()},
                                       orient='index')

    df_totals.to_csv(r'Y:\Mobile Data\Processing\MDD_HW_Totals_incOP.csv')

def mdd_car_totals(totals_check=False, check_location='Y:\\Mobile Data\\Processing\\9_Totals_Check'):
    # Import combined dict
    with open(r'Y:\Mobile Data\Processing\dct_MDDCar.pkl', 'rb') as log:
        dct_mdd_car = pk.load(log)

    dct_total = {3: {1: {}}}
    for pp in dctmddpurp:
        dct_total[3][1][pp] = {}
        for tp in dcttp:
            print(str(3) + '-' + str(1) + '-' + str(pp) + '-' + str(tp))
            dct_total[3][1][pp][tp] = np.sum(dct_mddcar[3][1][pp][tp])

    df_totals = pd.DataFrame.from_dict({(i, j, k): dct_total[i][j][k]
                                        for i in dct_total.keys()
                                        for j in dct_total[i].keys()
                                        for k in dct_total[i][j].keys()},
                                       orient='index')

    df_totals.to_csv(check_location + '\\MDD_Car_Totals.csv')


def main():
    run_mdd_combined_totals = False
    run_mdd_person_export = False
    run_mdd_per_to_veh = True

    if run_mdd_combined_totals:
        mdd_combined_totals()

    if run_mdd_person_export:
        mdd_person_export()

    if run_mdd_per_to_veh:
        mdd_per_to_veh()

    print("end of main")


if __name__ == '__main__':
    main()
