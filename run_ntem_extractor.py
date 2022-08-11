
import os
from pathlib import Path

import normits_demand.utils.ntem_extractor as te


def main(out_path: Path = None,
         output_years = list(),
         scenario: str = None,
         get_planning_data: bool = False,
         get_car_ownership: bool = False,
         get_trip_ends = True,
         write: bool = True,
         verbose: bool = True):

    safe_scenario = '' if scenario is None else scenario

    # Build object
    parser = te.TemproParser(output_years=output_years,
                             scenario=scenario)

    # Get planning data
    if get_planning_data:
        pd_out = parser.get_planning_data(compile_planning_data=True,
                                          verbose=verbose)
        if write:
            # Like this when format dict is name, dat:
            pd_out.to_csv(os.path.join(out_path,
                                       'ntem_%s_planning_data.csv' % safe_scenario),
                          index=False)

    # Get car ownership
    if get_car_ownership:
        co_out = parser.get_household_co_data(verbose=verbose)
        if write:
            co_out.to_csv(
                os.path.join(out_path,
                             'ntem_%s_ca_data.csv' % safe_scenario), index=False)

    # Get trip ends
    if get_trip_ends:
        te_out = parser.get_trip_ends(trip_type=None,
                                      all_commute_hb=True,
                                      aggregate_car=False,
                                      average_weekday=False,
                                      verbose=verbose)
        pa = ['productions', 'attractions']
        od = ['origins', 'destination']

        if write:
            te_out[te_out['trip_end_type'].isin(pa)].to_csv(
                os.path.join(out_path,
                             'ntem_%s_pa_data.csv' % safe_scenario), index=False)
            te_out[te_out['trip_end_type'].isin(od)].to_csv(
                os.path.join(out_path,
                             'ntem_%s_od_data.csv' % safe_scenario), index=False)


if __name__ == '__main__':

    # Config
    # TODO: some tying up params to funcs here

    output_years = [2018, 2027, 2033, 2035, 2038, 2040, 2045, 2050, 2060]
    # output_years = list(range(2011, 2051))

    scenarios = ['High', 'Low', 'Regional', 'Core', 'Behavioural']

    write = True
    verbose = True

    get_planning_data = True
    get_car_ownership = True
    get_trip_ends = True

    out_path = r'I:\Data\NTEM\Extracted Outputs'

    for scenario in scenarios:
        main(
            out_path=out_path,
            output_years=output_years,
            scenario=scenario,
            get_planning_data=get_planning_data,
            get_car_ownership=get_car_ownership,
            get_trip_ends=get_trip_ends,
            write=write,
            verbose=verbose)
