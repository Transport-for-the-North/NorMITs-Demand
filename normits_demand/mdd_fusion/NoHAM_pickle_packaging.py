"""
TODO: add some more comments here
"""
import numpy as np
import pickle as pk


def noham_car_package():
    """ packages NoHAM car matrices into single nested dictionary pickle file """
    # TODO: add import and export locations to function variables
    # TODO: could these be linked to some constants?
    dctmode = {3: ['Car']}
    dctday = {1: ['Weekday']}
    dctpurp = {1: ['Purpose-1'], 2: ['Purpose-2'], 3: ['Purpose-3'], 4: ['Purpose-4'], 5: ['Purpose-5'],
               6: ['Purpose-6'], 7: ['Purpose-7'], 8: ['Purpose-8']}
    dcttp = {1: ['AM'], 2: ['IP'], 3: ['PM'], 4: ['OP']}
    dctnoham = {}

    unq_zones = list(range(1, 2771))

    for md in dctmode:
        dctnoham[md] = {}
        for wd in dctday:
            dctnoham[md][wd] = {}
            for pp in dctpurp:
                dctnoham[md][wd][pp] = {}
                for tp in dcttp:
                    print('+++ {} - {} - {} - {} +++'.
                          format(dctmode[md][0], dctday[wd][0], dctpurp[pp][0], dcttp[tp][0]))
                    noham_car = np.genfromtxt(
                        r'I:\NorMITs Demand\noham\EFS\iter3i\NTEM\Matrices\OD Matrices\hb_od_from_yr2018_p'
                        + str(pp) + '_m' + str(md) + '_tp' + str(tp) + '.csv',
                        delimiter=',',
                        skip_header=1,
                        usecols=unq_zones)
                    dctnoham[md][wd][pp][tp] = noham_car

    with open(r'Y:\Mobile Data\Processing\dctNoHAM_hbfrom.pkl', 'wb') as log:
        pk.dump(dctnoham, log, pk.HIGHEST_PROTOCOL)
    print("matrices packaged")


def noham_hb_to_car_package():
    """ packages NoHAM car matrices into single nested dictionary pickle file """
    # TODO: add import and export locations to function variables
    # TODO: could these be linked to some constants?
    dctmode = {3: ['Car']}
    dctday = {1: ['Weekday']}
    dctpurp = {1: ['Purpose-1'], 2: ['Purpose-2'], 3: ['Purpose-3'], 4: ['Purpose-4'], 5: ['Purpose-5'],
               6: ['Purpose-6'], 7: ['Purpose-7'], 8: ['Purpose-8']}
    dcttp = {1: ['AM'], 2: ['IP'], 3: ['PM'], 4: ['OP']}
    dctnoham = {}

    unq_zones = list(range(1, 2771))

    for md in dctmode:
        dctnoham[md] = {}
        for wd in dctday:
            dctnoham[md][wd] = {}
            for pp in dctpurp:
                dctnoham[md][wd][pp] = {}
                for tp in dcttp:
                    print('+++ {} - {} - {} - {} +++'.
                          format(dctmode[md][0], dctday[wd][0], dctpurp[pp][0], dcttp[tp][0]))
                    noham_car = np.genfromtxt(
                        r'I:\NorMITs Demand\noham\EFS\iter3i\NTEM\Matrices\OD Matrices\hb_od_to_yr2018_p'
                        + str(pp) + '_m' + str(md) + '_tp' + str(tp) + '.csv',
                        delimiter=',',
                        skip_header=1,
                        usecols=unq_zones)
                    dctnoham[md][wd][pp][tp] = noham_car

    with open(r'Y:\Mobile Data\Processing\dctNoHAM_hbto.pkl', 'wb') as log:
        pk.dump(dctnoham, log, pk.HIGHEST_PROTOCOL)
    print("matrices packaged")


def noham_nhb_car_package():
    """ packages NoHAM car matrices into single nested dictionary pickle file """
    # TODO: add import and export locations to function variables
    # TODO: could these be linked to some constants?
    dctmode = {3: ['Car']}
    dctday = {1: ['Weekday']}
    dctpurp = {12: ['Purpose-12'], 13: ['Purpose-13'], 14: ['Purpose-14'], 15: ['Purpose-15'], 16: ['Purpose-16'],
               18: ['Purpose-18']}
    dcttp = {1: ['AM'], 2: ['IP'], 3: ['PM'], 4: ['OP']}
    dctnoham = {}

    unq_zones = list(range(1, 2771))

    for md in dctmode:
        dctnoham[md] = {}
        for wd in dctday:
            dctnoham[md][wd] = {}
            for pp in dctpurp:
                dctnoham[md][wd][pp] = {}
                for tp in dcttp:
                    print('+++ {} - {} - {} - {} +++'.
                          format(dctmode[md][0], dctday[wd][0], dctpurp[pp][0], dcttp[tp][0]))
                    noham_car = np.genfromtxt(
                        r'I:\NorMITs Demand\noham\EFS\iter3i\NTEM\Matrices\OD Matrices\nhb_od_yr2018_p'
                        + str(pp) + '_m' + str(md) + '_tp' + str(tp) + '.csv',
                        delimiter=',',
                        skip_header=1,
                        usecols=unq_zones)
                    dctnoham[md][wd][pp][tp] = noham_car

    with open(r'Y:\Mobile Data\Processing\dctNoHAM_nhb.pkl', 'wb') as log:
        pk.dump(dctnoham, log, pk.HIGHEST_PROTOCOL)
    print("matrices packaged")


def noham_car_hb_from_merge():
    dctmode = {3: ['Car']}
    dctday = {1: ['Weekday']}
    dctpurp = {1: ['Purpose-1'], 2: ['Purpose-2'], 3: ['Purpose-3'], 4: ['Purpose-4'], 5: ['Purpose-5'],
               6: ['Purpose-6'], 7: ['Purpose-7'], 8: ['Purpose-8']}
    dcttp = {1: ['AM'], 2: ['IP'], 3: ['PM'], 4: ['OP']}

    dctuserclass = {1: ['Commute'], 2: ['Business'], 3: ['Other']}

    temp_array = np.zeros((2770, 2770))

    dctnohamhbfromuc = {}
    for md in dctmode:
        dctnohamhbfromuc[md] = {}
        for wd in dctday:
            dctnohamhbfromuc[md][wd] = {}
            for uc in dctuserclass:
                dctnohamhbfromuc[md][wd][uc] = {}
                for tp in dcttp:
                    dctnohamhbfromuc[md][wd][uc][tp] = temp_array

    with open(r'Y:\Mobile Data\Processing\dctNoHAM_hbfrom.pkl', 'rb') as log:
        dctnohamhbfrompp = pk.load(log)  # [md][wd][pp][hr]

    for md in dctmode:
        for wd in dctday:
            for pp in dctpurp:
                uc = 1 if pp in [1] else 2 if pp in [2] else 3 if pp in [3, 4, 5, 6, 7, 8] else 4
                for tp in dcttp:
                    print('+++ {} - {} - {} - {} +++'.
                          format(dctmode[md][0], dctday[wd][0], dctpurp[pp][0], dcttp[tp][0]))
                    print('+++ {} - {} - {} - {} +++'.
                          format(dctmode[md][0], dctday[wd][0], dctuserclass[uc][0], dcttp[tp][0]))
                    print(str(md) + '-' + str(wd) + '-' + str(pp) + '-' + str(tp))
                    print(str(md) + '-' + str(wd) + '-' + str(uc) + '-' + str(tp))
                    dctnohamhbfromuc[md][wd][uc][tp] += dctnohamhbfrompp[md][wd][pp][tp][:2770, :2770]

    with open(r'Y:\Mobile Data\Processing\dctNoHAM_hb_from_uc.pkl', 'wb') as log:
        pk.dump(dctnohamhbfromuc, log, pk.HIGHEST_PROTOCOL)


def noham_car_hb_to_merge():
    dctmode = {3: ['Car']}
    dctday = {1: ['Weekday']}
    dctpurp = {1: ['Purpose-1'], 2: ['Purpose-2'], 3: ['Purpose-3'], 4: ['Purpose-4'], 5: ['Purpose-5'],
               6: ['Purpose-6'], 7: ['Purpose-7'], 8: ['Purpose-8']}
    dcttp = {1: ['AM'], 2: ['IP'], 3: ['PM'], 4: ['OP']}

    dctuserclass = {1: ['Commute'], 2: ['Business'], 3: ['Other']}

    temp_array = np.zeros((2770, 2770))

    dctnohamhbtouc = {}
    for md in dctmode:
        dctnohamhbtouc[md] = {}
        for wd in dctday:
            dctnohamhbtouc[md][wd] = {}
            for uc in dctuserclass:
                dctnohamhbtouc[md][wd][uc] = {}
                for tp in dcttp:
                    dctnohamhbtouc[md][wd][uc][tp] = temp_array

    with open(r'Y:\Mobile Data\Processing\dctNoHAM_hbto.pkl', 'rb') as log:
        dctnohamhbtopp = pk.load(log)  # [md][wd][pp][hr]

    for md in dctmode:
        for wd in dctday:
            for pp in dctpurp:
                uc = 1 if pp in [1] else 2 if pp in [2] else 3 if pp in [3, 4, 5, 6, 7, 8] else 4
                for tp in dcttp:
                    print('+++ {} - {} - {} - {} +++'.
                          format(dctmode[md][0], dctday[wd][0], dctpurp[pp][0], dcttp[tp][0]))
                    print('+++ {} - {} - {} - {} +++'.
                          format(dctmode[md][0], dctday[wd][0], dctuserclass[uc][0], dcttp[tp][0]))
                    print(str(md) + '-' + str(wd) + '-' + str(pp) + '-' + str(tp))
                    print(str(md) + '-' + str(wd) + '-' + str(uc) + '-' + str(tp))
                    dctnohamhbtouc[md][wd][uc][tp] += dctnohamhbtopp[md][wd][pp][tp][:2770, :2770]

    with open(r'Y:\Mobile Data\Processing\dctNoHAM_hb_to_uc.pkl', 'wb') as log:
        pk.dump(dctnohamhbtouc, log, pk.HIGHEST_PROTOCOL)


def noham_car_nhb_merge():
    dctmode = {3: ['Car']}
    dctday = {1: ['Weekday']}
    dctpurp = {12: ['Purpose-12'], 13: ['Purpose-13'], 14: ['Purpose-14'], 15: ['Purpose-15'], 16: ['Purpose-16'],
               18: ['Purpose-18']}
    dcttp = {1: ['AM'], 2: ['IP'], 3: ['PM'], 4: ['OP']}

    dctuserclass = {1: ['Commute'], 2: ['Business'], 3: ['Other']}

    temp_array = np.zeros((2770, 2770))

    dctnohamnhbuc = {}
    for md in dctmode:
        dctnohamnhbuc[md] = {}
        for wd in dctday:
            dctnohamnhbuc[md][wd] = {}
            for uc in dctuserclass:
                dctnohamnhbuc[md][wd][uc] = {}
                for tp in dcttp:
                    dctnohamnhbuc[md][wd][uc][tp] = temp_array

    with open(r'Y:\Mobile Data\Processing\dctNoHAM_nhb.pkl', 'rb') as log:
        dctnohamnhbpp = pk.load(log)  # [md][wd][pp][hr]

    for md in dctmode:
        for wd in dctday:
            for pp in dctpurp:
                uc = 1 if pp in [1] else 2 if pp in [2, 12] else 3 if pp in [3, 4, 5, 6, 7, 8, 13, 14, 15, 16, 18] else 4
                for tp in dcttp:
                    print('+++ {} - {} - {} - {} +++'.
                          format(dctmode[md][0], dctday[wd][0], dctpurp[pp][0], dcttp[tp][0]))
                    print('+++ {} - {} - {} - {} +++'.
                          format(dctmode[md][0], dctday[wd][0], dctuserclass[uc][0], dcttp[tp][0]))
                    print(str(md) + '-' + str(wd) + '-' + str(pp) + '-' + str(tp))
                    print(str(md) + '-' + str(wd) + '-' + str(uc) + '-' + str(tp))
                    dctnohamnhbuc[md][wd][uc][tp] += dctnohamnhbpp[md][wd][pp][tp][:2770, :2770]

    with open(r'Y:\Mobile Data\Processing\dctNoHAM_nhb_uc.pkl', 'wb') as log:
        pk.dump(dctnohamnhbuc, log, pk.HIGHEST_PROTOCOL)


def main():

    run_noham_car_package = False
    run_noham_hb_to_car_package = False
    run_noham_nhb_car_package = False
    run_noham_car_hb_from_merge = False
    run_noham_car_hb_to_merge = False
    run_noham_car_nhb_merge = True

    if run_noham_car_package:
        noham_car_package()

    if run_noham_hb_to_car_package:
        noham_hb_to_car_package()

    if run_noham_nhb_car_package:
        noham_nhb_car_package()

    if run_noham_car_hb_from_merge:
        noham_car_hb_to_merge()

    if run_noham_car_hb_to_merge:
        noham_car_hb_to_merge()

    if run_noham_car_nhb_merge:
        noham_car_nhb_merge()

    print("end of main")


if __name__ == '__main__':
    main()
