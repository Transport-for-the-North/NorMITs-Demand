"""
# TODO: add further information
"""

import numpy as np
import pickle as pk
import bz2

dctmode = {5: ['Bus']}
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
dcttp = {1: ['AM'], 2: ['IP'], 3: ['PM'], 4: ['OP']}
dctmddpurp = {1: ['HBW', 'HBW_fr'], 2: ['HBW', 'HBW_to'], 3: ['HBO', 'HBO_fr'], 4: ['HBO', 'HBO_to'],
              5: ['NHB', 'NHB']}


def decompress_pickle(file):
    data = bz2.BZ2File(file, 'rb')
    data = pk.load(data)
    return data


def notem_bus_package_test():
    temp_data = decompress_pickle(
        r'I:\Transfer\External Model OD\NoTEM iter4.2\bus\hb_synthetic_od_from_yr2018_p1_m5_tp1.pbz2')
    print(temp_data)


def notem_bus_package():
    dctnotem = {}
    for md in dctmode:
        dctnotem[md] = {}
        for wd in dctday:
            dctnotem[md][wd] = {}
            for pp in dctpurp:
                dctnotem[md][wd][pp] = {}
                for tp in dcttp:
                    print('+++ {} - {} - {} - {} +++'.
                          format(dctmode[md][0], dctday[wd][0], dctpurp[pp], dcttp[tp][0]))
                    if dctpurp[pp][0] == 'hb':
                        path = ('I:/Transfer/External Model OD/NoTEM iter4.2/bus/'
                                + str(dctpurp[pp][0]) + '_synthetic_od_'
                                + str(dctpurp[pp][1]) + '_yr2018_p'
                                + str(dctpurp[pp][2]) + '_m'
                                + str(md) + '_tp'
                                + str(tp) + '.pbz2')
                    elif dctpurp[pp][0] == 'nhb':
                        path = ('I:/Transfer/External Model OD/NoTEM iter4.2/bus/'
                                + str(dctpurp[pp][0]) + '_synthetic_od_'
                                + str(dctpurp[pp][1]) + 'yr2018_p'
                                + str(dctpurp[pp][2]) + '_m'
                                + str(md) + '_tp'
                                + str(tp) + '.pbz2')
                    else:
                        print('Value outside expected range')
                    print(path)
                    notem_bus = decompress_pickle(path)
                    dctnotem[md][wd][pp][tp] = notem_bus

    with open(r'Y:\Mobile Data\Processing\dctNoTEM.pkl', 'wb') as log:
        pk.dump(dctnotem, log, pk.HIGHEST_PROTOCOL)


def notem_bus_merge():
    temp_array = np.zeros((2770, 2770))

    dctnotem_mddpurp = {}
    for md in dctmode:
        dctnotem_mddpurp[md] = {}
        for wd in dctday:
            dctnotem_mddpurp[md][wd] = {}
            for pp in dctmddpurp:
                dctnotem_mddpurp[md][wd][pp] = {}
                for tp in dcttp:
                    dctnotem_mddpurp[md][wd][pp][tp] = temp_array

    with open(r'Y:\Mobile Data\Processing\dctNoTEM.pkl', 'rb') as log:
        dctnotem = pk.load(log)

    for md in dctmode:
        for wd in dctday:
            for pp in dctpurp:
                mddpurp = (1 if dctpurp[pp][2] in [1] and dctpurp[pp][1] in ['from'] else
                           2 if dctpurp[pp][2] in [1] and dctpurp[pp][1] in ['to'] else
                           3 if dctpurp[pp][2] in [2, 3, 4, 5, 6, 7, 8] and dctpurp[pp][1] in ['from'] else
                           4 if dctpurp[pp][2] in [2, 3, 4, 5, 6, 7, 8] and dctpurp[pp][1] in ['to'] else
                           5 if dctpurp[pp][2] in [12, 13, 14, 15, 16, 18] else
                           6)
                print('Purpose: ' + str(pp) + ' into mdd ' + str(mddpurp))
                for tp in dcttp:
                    # print(str(md) + '-' + str(wd) + '-' + str(pp) + '-' + str(tp))
                    # print(str(md) + '-' + str(wd) + '-' + str(mddpurp) + '-' + str(tp))
                    dctnotem_mddpurp[md][wd][mddpurp][tp] = (dctnotem_mddpurp[md][wd][mddpurp][tp] +
                                                             dctnotem[md][wd][pp][tp].to_numpy())

    with open(r'Y:\Mobile Data\Processing\dctNoTEM_mddpurp.pkl', 'wb') as log:
        pk.dump(dctnotem_mddpurp, log, pk.HIGHEST_PROTOCOL)


def main():

    run_notem_bus_package_test = False
    run_notem_bus_package = False
    run_notem_bus_merge = True

    if run_notem_bus_package_test:
        notem_bus_package_test()

    if run_notem_bus_package:
        notem_bus_package()

    if run_notem_bus_merge:
        notem_bus_merge()

    print("end of main")


if __name__ == '__main__':
    main()
