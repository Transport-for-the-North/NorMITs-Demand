"""
TODO: add some more comments here
"""
import numpy as np
import pickle as pk

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
dctmddpurp = {1: ['HBW', 'HBW_fr'], 2: ['HBW', 'HBW_to'], 3: ['HBO', 'HBO_fr'], 4: ['HBO', 'HBO_to'],
              5: ['NHB', 'NHB']}
dcttp = {1: ['AM'], 2: ['IP'], 3: ['PM'], 4: ['OP']}


def noham_car_package():
    """ packages NoHAM car matrices into single nested dictionary pickle file """
    # TODO: add import and export locations to function variables

    unq_zones = list(range(1, 2771))

    dctnoham = {}
    for md in dctmode:
        dctnoham[md] = {}
        for wd in dctday:
            dctnoham[md][wd] = {}
            for pp in dctpurp:
                dctnoham[md][wd][pp] = {}
                for tp in dcttp:
                    print('+++ {} - {} - {} - {} +++'.
                          format(dctmode[md][0], dctday[wd][0], dctpurp[pp][0], dcttp[tp][0]))
                    if dctpurp[pp][0] == 'hb':
                        path = ('I:/NorMITs Demand/noham/EFS/iter3i/NTEM/Matrices/OD Matrices/'
                                + str(dctpurp[pp][0]) + '_od_'
                                + str(dctpurp[pp][1]) + '_yr2018_p'
                                + str(dctpurp[pp][2]) + '_m'
                                + str(md) + '_tp'
                                + str(tp) + '.csv')
                    elif dctpurp[pp][0] == 'nhb':
                        path = ('I:/NorMITs Demand/noham/EFS/iter3i/NTEM/Matrices/OD Matrices/'
                                + str(dctpurp[pp][0]) + '_od_'
                                + str(dctpurp[pp][1]) + 'yr2018_p'
                                + str(dctpurp[pp][2]) + '_m'
                                + str(md) + '_tp'
                                + str(tp) + '.csv')
                    else:
                        print('Value outside expected range')
                    print(path)
                    noham_car = np.genfromtxt(path,
                                              delimiter=',',
                                              skip_header=1,
                                              usecols=unq_zones)
                    dctnoham[md][wd][pp][tp] = noham_car

    with open(r'Y:\Mobile Data\Processing\dctNoHAM.pkl', 'wb') as log:
        pk.dump(dctnoham, log, pk.HIGHEST_PROTOCOL)
    print("matrices packaged")


def noham_compiled_pcu_import():
    import numpy as np
    import pickle as pk

    dctmode = {3: ['Car']}
    dctday = {1: ['Weekday']}
    dctuc = {1: ['business'],
             2: ['commute'],
             3: ['other']}
    dcttp = {1: ['AM'], 2: ['IP'], 3: ['PM'], 4: ['OP']}

    unq_zones = list(range(1, 2771))

    dctnoham_uc = {}
    for md in dctmode:
        dctnoham_uc[md] = {}
        for wd in dctday:
            dctnoham_uc[md][wd] = {}
            for uc in dctuc:
                dctnoham_uc[md][wd][uc] = {}
                for tp in dcttp:
                    path = ('I:/NorMITs Demand/noham/EFS/iter3i/NTEM/Matrices/Compiled OD Matrices PCU/'
                            + 'od_'
                            + str(dctuc[uc][0]) + '_yr2018'
                            + '_m' + str(md)
                            + '_tp' + str(tp) + '.csv')
                    print(path)
                    noham_uc = np.genfromtxt(path,
                                             delimiter=',',
                                             skip_header=1,
                                             usecols=unq_zones)
                    print(str(md) + str(wd) + str(uc) + str(tp))
                    print(round(np.sum(noham_uc), ndigits=0))
                    dctnoham_uc[md][wd][uc][tp] = noham_uc

    with open(r'Y:\Mobile Data\Processing\dctNoHAM_compiled_PCU.pkl', 'wb') as log:
        pk.dump(dctnoham_uc, log, pk.HIGHEST_PROTOCOL)
    print("matrices packaged")


def noham_car_merge():
    temp_array = np.zeros((2770, 2770))

    dctnoham_mddpurp = {}
    for md in dctmode:
        dctnoham_mddpurp[md] = {}
        for wd in dctday:
            dctnoham_mddpurp[md][wd] = {}
            for pp in dctmddpurp:
                dctnoham_mddpurp[md][wd][pp] = {}
                for tp in dcttp:
                    dctnoham_mddpurp[md][wd][pp][tp] = temp_array

    with open(r'Y:\Mobile Data\Processing\dctNoHAM.pkl', 'rb') as log:
        dctnoham = pk.load(log)  # [md][wd][pp][hr]

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
                    dctnoham_mddpurp[md][wd][mddpurp][tp] = (dctnoham_mddpurp[md][wd][mddpurp][tp] +
                                                             dctnoham[md][wd][pp][tp])

    with open(r'Y:\Mobile Data\Processing\dctNoHAM_mddpurp.pkl', 'wb') as log:
        pk.dump(dctnoham_mddpurp, log, pk.HIGHEST_PROTOCOL)


def mdd_to_uc():
    # Import NoHAM purpose dictionary
    with open(r'Y:\Mobile Data\Processing\dctNoHAM.pkl', 'rb') as log:
        dctnohampp = pk.load(log)

    with open(r'Y:\Mobile Data\Processing\dctNoHAM_mddpurp.pkl', 'rb') as log:
        dctnohammdd = pk.load(log)

    dctmdduc = {'1_hb_from': [1], '1_hb_to': [1],
                '2_hb_from': [2], '2_hb_to': [2],
                '3_hb_from': [3], '3_hb_to': [3],
                '1_nhb': [1], '3_nhb': [3]}
    temp_array = np.zeros((2770, 2770))

    dctnohammdduc = {}
    for md in dctmode:
        dctnohammdduc[md] = {}
        for wd in dctday:
            dctnohammdduc[md][wd] = {}
            for uc in dctmdduc:
                dctnohammdduc[md][wd][uc] = {}
                for tp in dcttp:
                    dctnohammdduc[md][wd][uc][tp] = temp_array

    for md in dctmode:
        for wd in dctday:
            for pp in dctpurp:
                uc = (  # Commute hb from
                    '2_hb_from' if dctpurp[pp][2] in [1] and dctpurp[pp][1] in ['from'] else
                    # Commute hb to
                    '2_hb_to' if dctpurp[pp][2] in [1] and dctpurp[pp][1] in ['to'] else
                    # Employers Business hb from
                    '1_hb_from' if dctpurp[pp][2] in [2] and dctpurp[pp][1] in ['from'] else
                    # Employers Business hb to
                    '1_hb_to' if dctpurp[pp][2] in [2] and dctpurp[pp][1] in ['to'] else
                    # Other hb from
                    '3_hb_from' if dctpurp[pp][2] in [3, 4, 5, 6, 7, 8] and dctpurp[pp][1] in ['from'] else
                    # Other hb to
                    '3_hb_to' if dctpurp[pp][2] in [3, 4, 5, 6, 7, 8] and dctpurp[pp][1] in ['to'] else
                    # Employers Business nhb
                    '1_nhb' if dctpurp[pp][2] in [12] else
                    # Other nhb
                    '3_nhb' if dctpurp[pp][2] in [13, 14, 15, 16, 18] else
                    '6')
                for tp in dcttp:
                    print('+++ {} - {} - {} - {} +++ uc: {}'.
                          format(dctmode[md][0], dctday[wd][0], dctpurp[pp][0], dcttp[tp][0], uc))
                    dctnohammdduc[md][wd][uc][tp] = (
                                dctnohammdduc[md][wd][uc][tp] + dctnohampp[md][wd][pp][tp][:2770, :2770])

    with open(r'Y:\Mobile Data\Processing\dctNoHAM_car_uc.pkl', 'wb') as log:
        pk.dump(dctnohammdduc, log, pk.HIGHEST_PROTOCOL)

    dctglobalucshare = {}
    for md in dctmode:
        dctglobalucshare[md] = {}
        for wd in dctday:
            dctglobalucshare[md][wd] = {}
            for uc in dctmdduc:
                dctglobalucshare[md][wd][uc] = {}
                mddpurp = (3 if uc in ['1_hb_from'] else
                           4 if uc in ['1_hb_to'] else
                           1 if uc in ['2_hb_from'] else
                           2 if uc in ['2_hb_to'] else
                           3 if uc in ['3_hb_from'] else
                           4 if uc in ['3_hb_to'] else
                           5 if uc in ['1_nhb'] else
                           5 if uc in ['3_nhb'] else
                           6)
                for tp in dcttp:
                    print(str(md) + str(wd) + str(uc) + str(mddpurp) + str(tp))
                    dctglobalucshare[md][wd][uc][tp] = (
                                np.sum(dctglobalucshare[md][wd][uc][tp]) / np.sum(dctnohammdd[md][wd][mddpurp][tp]))
                    dctglobalucshare[md][wd][uc][tp] = np.nan_to_num(dctglobalucshare[md][wd][uc][tp])

    dctmddsplit = {}
    for md in dctmode:
        dctmddsplit[md] = {}
        for wd in dctday:
            dctmddsplit[md][wd] = {}
            for uc in dctmdduc:
                dctmddsplit[md][wd][uc] = {}
                mddpurp = (3 if uc in ['1_hb_from'] else
                           4 if uc in ['1_hb_to'] else
                           1 if uc in ['2_hb_from'] else
                           2 if uc in ['2_hb_to'] else
                           3 if uc in ['3_hb_from'] else
                           4 if uc in ['3_hb_to'] else
                           5 if uc in ['1_nhb'] else
                           5 if uc in ['3_nhb'] else
                           6)
                for tp in dcttp:
                    print(str(md) + str(wd) + str(uc) + str(mddpurp) + str(tp))
                    dctmddsplit[md][wd][uc][tp] = (dctnohammdduc[md][wd][uc][tp] / dctnohammdd[md][wd][mddpurp][tp])
                    dctmddsplit[md][wd][uc][tp] = np.nan_to_num(dctmddsplit[md][wd][uc][tp],
                                                                nan=dctglobalucshare[md][wd][uc][tp])

    with open(r'Y:\Mobile Data\Processing\dctmdd_uc_split.pkl', 'wb') as log:
        pk.dump(dctmddsplit, log, pk.HIGHEST_PROTOCOL)

    # TODO: convert MDD and assign
    # TODO: re-split MDD Car


def main():
    run_noham_car_package = False
    run_noham_car_merge = True

    if run_noham_car_package:
        noham_car_package()

    if run_noham_car_merge:
        noham_car_merge()

    print("end of main")


if __name__ == '__main__':
    main()
