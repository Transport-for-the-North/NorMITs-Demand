

import numpy as np
import pickle as pk
import pandas as pd

dctmode = {3: 'Car', 5: 'Bus', 7: 'LGV', 8: 'HGV'}
dctday = {1: 'Weekday'}
dctmddpurp = {1: ['HBW', 'HBW_fr'], 2: ['HBW', 'HBW_to'], 3: ['HBO', 'HBO_fr'], 4: ['HBO', 'HBO_to'],
              5: ['NHB', 'NHB']}
dcthgvpurp = {5: ['NHB', 'NHB']}
dcttp = {1: ['AM', 3], 2: ['IP', 6], 3: ['PM', 3]}

def mdd_combined_totals():
    # Import combined dict
    with open(r'Y:\Mobile Data\Processing\dct_CombinedMode.pkl', 'rb') as log:
        dct_combined = pk.load(log)

    dct_total = {}
    for wd in dctday:
        dct_total[wd] = {}
        for pp in dctmddpurp:
            dct_total[wd][pp] = {}
            for tp in dcttp:
                print(str(wd) + '-' + str(pp) + '-' + str(tp))
                if pp == 5:
                    print('NHB')
                    dct_total[wd][pp][tp] = (dct_combined[3][wd][pp][tp] + dct_combined[5][wd][pp][tp] +
                                            dct_combined[7][wd][pp][tp] + dct_combined[8][wd][pp][tp])
                elif pp in [1, 2, 3, 4]:
                    print('Car, Bus & LGV')
                    dct_total[wd][pp][tp] = (dct_combined[3][wd][pp][tp] + dct_combined[5][wd][pp][tp] +
                                            dct_combined[7][wd][pp][tp])
                else:
                    print('purpose variable outside anticipated range')

    df_totals = pd.DataFrame.from_dict({(i, j): dct_total[i][j]
                                        for i in dct_total.keys()
                                        for j in dct_total[i].keys()},
                                       orient='index')

    df_totals.to_csv(r'Y:\Mobile Data\Processing\Combined_Totals')

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
