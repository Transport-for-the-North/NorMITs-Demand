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

    with open(r'Y:\Mobile Data\Processing\dctNoHAM.pkl', 'wb') as log:
        pk.dump(dctnoham, log, pk.HIGHEST_PROTOCOL)
    print("matrices packaged")


def main():

    run_noham_car_package = True

    if run_noham_car_package:
        noham_car_package()

    print("end of main")


if __name__ == '__main__':
    main()
