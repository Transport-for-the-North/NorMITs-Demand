# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
import sys # File ops & sys config
sys.path.append('C:/Users/' + os.getlogin() + 
                '/S/NorMITs Travel Market Synthesiser/Python')
import distribution_model as dm
import pandas as pd

distribution_segments = ['purpose', 'mode', 'car_availability']

def prepare_betas(distribution_segments):
    
    # Function to create a set of preseeded betas to a st of distribution segments.
    # Should use curve derived from model run outputs.
    # That should live in import folder

    beta_list = ['betas']    
    return(beta_list)
    
    
output_log_path = 'Y:/NorMITs Synthesiser/Norms/iter1/Distribution Outputs/Logs & Reports/distribution_run_log.csv'
model_lookup_path = 'Y:/NorMITs Synthesiser/Norms/Model Zone Lookups/'

log_col_headings = ['atl', 'beta', 'car_availability', 'end_time',
                    'mode', 'purpose', 'run_date', 'solution_steps',
                    'start_time', 'ttl']

def update_original_betas(model_lookup_path,
                          model_name,
                          output_log_path,
                          distribution_segments,
                          distribution_type = 'hb'
                          ):

    """
    Function to take run logs from a distribution and allocate best match beta
    values from run log back into the initial beta file.
    
    Parameters
    ----------
    model_lookup_path:
        sgagr

    model_name:
        rgwregrg

    distribution_type:
        'hb' or 'nhb'

    output_log_path:
        Path to output log csv.

    distribution_segments:
        Segments to build distributions by.

    Returns
    ----------
    [0] import_folder_path:
        Path to all Synthesiser import parameters.

    [1] model_lookup_path:
        Path to model lookups, translations, costs etc.
    """
    # Import output log - assumes it's all clean, which it isn't yet
    new_betas = pd.read_csv(output_log_path)

    # Calculate absolute percentage calibration
    new_betas['cal_perc'] = 1-(abs(1-new_betas['ttl']/new_betas['atl']))

    # Make list of col names for all distribution segments
    bd_cols = []
    for d_col in distribution_segments:
        bd_cols.append(d_col)
    # Will need cal_perc too, so append
    bd_cols.append('cal_perc')

    # Get distributions with only the best calibration
    best_dists = new_betas.reindex(
            bd_cols,
            axis=1).groupby(
                    distribution_segments).max(
                            ).reset_index()

    # Make a seperate set with beta included for joins
    bbd_cols = bd_cols.copy()
    bbd_cols.append('beta')

    # Get betas and distributions only
    betas_and_dists = new_betas.reindex(bbd_cols, axis=1)

    # Join best distributions on to get new betas
    best_dists = best_dists.merge(betas_and_dists,
                                  how = 'inner',
                                  on = bd_cols).drop_duplicates().reset_index(drop=True)

    # Import old betas
    initial_betas = dm.get_initial_betas(model_lookup_path,
                                         distribution_type='hb',
                                         model_name=model_name)
    # Sidepot initial betas col names
    ib_cols = list(initial_betas)

    # Attach changed betas
    changed_betas = initial_betas.merge(best_dists,
                                        how='left',
                                        on = distribution_segments)

    # Round beta, don't be overly exact
    changed_betas['beta'] = changed_betas['beta'].round(5)
    # Replace null values with prior values
    changed_betas['beta'] = changed_betas['beta'].fillna(changed_betas['initial_beta'])
    # Get rid of the old intial betas
    del(changed_betas['initial_beta'])

    # Make new betas initial betas
    changed_betas = changed_betas.rename(columns={'beta':'initial_beta'})
    # Reindex back to original cols
    changed_betas = changed_betas.reindex(ib_cols, axis=1).reset_index(drop=True)
    
    
    
    changed_betas.to_csv('Y:/norms_initial_betas.csv', index=False)

    return(changed_betas)

def chunk_betas():
    # Function to replicate the R split.
    
    
    return(None)