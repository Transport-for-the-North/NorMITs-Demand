
import os

import pandas as pd

import normits_demand.utils.tempro_extractor as te

if __name__ == '__main__':

    # Config
    # TODO: some tying up params to funcs here
    get_planning_data = True
    get_car_ownership = True  
    get_trip_ends = True
    
    out_path = os.path.join(
        'C:/Users/',
        os.getlogin(),
        'Documents')

    write = True
    verbose = True

    # Build object
    parser = te.TemproParser()
    
    # Get planning data
    if get_planning_data:
        pd_out = parser.get_planning_data(compile_planning_data=True)
        if write:
            # Like this when format dict is name, dat:
            pd_out.to_csv(os.path.join(out_path,
                                       'tempro_planning_data.csv'),
                          index=False)
    
    # Get car ownership
    if get_car_ownership:
        co_out = parser.get_household_co_data()
        if write:
            co_out.to_csv(
                os.path.join(out_path,
                             'tempro_ca_data.csv'), index=False)
    
    # Get trip ends
    if get_trip_ends:
        te_out = te.get_trip_ends(trip_type='pa',
                                  aggregate_car=True)
        # TODO: Finish this w/ export - underlying query is still messy