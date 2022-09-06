
import os
from pathlib import Path

import normits_demand.utils.ntem_extractor as te


def main():

    out_path = None
    output_years = [2018, 2027, 2033, 2035, 2038, 2040, 2045, 2050]
    scenarios = ['High', 'Low', 'Regional', 'Core', 'Behavioural']
    scenarios = [None]

    get_planning_data = True
    get_car_ownership = False
    get_trip_ends = False
    write = True
    verbose = True

    ntem_version = 7.2

    out_path = r'I:\Data\NTEM\Extracted Outputs'

    for scenario in scenarios:

        safe_scenario = '' if scenario is None else scenario
        safe_version = '' if ntem_version is None else '_' + str(ntem_version)

        # Build object
        parser = te.TemproParser(output_years=output_years,
                                 ntem_version=ntem_version,
                                 scenario=scenario)

        # Get planning data
        if get_planning_data:
            pd_out = parser.get_planning_data(compile_planning_data=True,
                                              verbose=verbose)
            if write:
                # Like this when format dict is name, dat:
                pd_out.to_csv(os.path.join(out_path,
                                           'ntem%s_%s_planning_data.csv' % (safe_version, safe_scenario)),
                              index=False)

        # Get car ownership
        if get_car_ownership:
            co_out = parser.get_household_co_data(verbose=verbose)
            if write:
                co_out.to_csv(
                    os.path.join(out_path,
                                 'ntem%s_%s_ca_data.csv' % (safe_version, safe_scenario)), index=False)

        # Get trip ends
        if get_trip_ends:
            te_out = parser.get_trip_ends(trip_type=None,
                                          all_commute_hb=False,
                                          aggregate_car=False,
                                          average_weekday=False,
                                          verbose=verbose)
            pa = ['productions', 'attractions']
            od = ['origins', 'destinations']

            if write:
                te_out[te_out['trip_end_type'].isin(pa)].to_csv(
                    os.path.join(out_path,
                                 'ntem%s_%s_pa_data.csv' % (safe_version, safe_scenario)), index=False)
                te_out[te_out['trip_end_type'].isin(od)].to_csv(
                    os.path.join(out_path,
                                 'ntem%s_%s_od_data.csv' % (safe_version, safe_scenario)), index=False)


if __name__ == '__main__':

    main()
