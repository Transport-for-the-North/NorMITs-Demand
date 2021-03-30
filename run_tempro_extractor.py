
import os
import normits_demand.utils.tempro_extractor as te

if __name__ == '__main__':
    """
    """
    
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
        pd_out = parser.get_planning_data()
        if write:
            # TODO: Out handling for planning data
            # Like this when format dict is name, dat:
            for out in pd_out:           
                export_name = 'in_a_dict'
                print('%s.csv' % export_name)
                pd_out[0].to_csv(os.path.join(out_path, 'dat_name.csv'), index=False)
    
    # Get car ownership
    if get_car_ownership:
        co_out = parser.get_co_future_data()
        if write:
            co_out[0].to_csv(
                os.path.join(out_path, 'ca_share_factors.csv'), index=False)
            co_out[1].to_csv(
                os.path.join(out_path, 'ca_growth_factors.csv', index=False))
    
    # Get trip ends