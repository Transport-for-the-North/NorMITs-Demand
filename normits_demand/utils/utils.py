# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 14:07:32 2020

@author: cruella
"""

import gc
import os
import pathlib
import sys
import time
import math
import pickle
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from normits_demand.types import PathLike

_default_home_dir = 'C:/'
_default_iter = 'iter0'

_M_KM = 1.609344


# Index functions - functions to aggregate columns into new category variables
def create_project_folder(projectName, echo=True):
    """
    """

    if not os.path.exists(os.getcwd() + '/' + projectName):
        os.makedirs(os.getcwd() + '/' + projectName)
        os.chdir(os.getcwd() + '/' + projectName)
        print_w_toggle('New project folder created in ' + os.getcwd() + ', wd set there', echo)
    else:
        os.chdir(os.getcwd() + '/' + projectName)
        print_w_toggle('Project folder already exists, wd set there', echo)


def create_folder(folder, chDir=False, verbose=True):
    """
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
        if chDir:
            os.chdir(folder)
        print_w_toggle("New project folder created in " + folder, verbose=verbose)
    else:
        if chDir:
            os.chdir(folder)
        print_w_toggle('Folder already exists', verbose=verbose)


def set_time():
    """
    Gets time from time module.

    Parameters
    ----------
    Null

    Returns
    ----------
    time_stamp:
        Day and time in format 'hour:minute:second'
    """
    time_stamp = time.strftime('%H:%M:%S')
    return (time_stamp)


def set_wd(home_dir=_default_home_dir, iteration=_default_iter):
    # TODO: I've written so many of these it should go into Utils
    """
    This function sets a working directory and creates a project folder
    if required.

    Parameters
    ----------
    home_dir:
        Path to base directory to sit under project.

    iteration:
        Project folder for efs_exports

    Returns
    ----------
    [folder]
    """
    os.chdir(home_dir)
    create_project_folder(iteration)
    return ()


# Index functions
def build_index(dataframe,
                index_cols,
                new_index_name):
    """
    Function to combine 2 or more category variables into a single integer
    index.
    This function builds the index only, and requires replace_index_columns
    to apply the index to a DataFrame.

    Parameters
    ----------
    dataframe:
        A DataFrame containing category variables.

    index_cols:
        list of column names to index by.

    new_index_name:
        The name of the new index. Will be an integer column with this name.

    Returns
    ----------
    new_index:
        Index as DataFrame containing original category variable and integer
        index lookup.
    """
    new_index = dataframe.reindex(
        index_cols,
        axis=1
    ).drop_duplicates(
    ).sort_values(
        by=index_cols
    ).reset_index(
        drop=True
    )
    new_index.index = new_index.index + 1
    new_index = new_index.reset_index()
    new_index = new_index.rename(columns={'index': new_index_name})
    return (new_index)


def replace_index_columns(dataframe,
                          index):
    """
    This function uses an index to replace category columns in a DataFrame
    with a single lookup column. Takes an index built in build_index.

    Parameters
    ----------
    dataframe:
        A DataFrame containing category variables.

    index:
        Index as DataFrame containing original category variable and integer
        index lookup.

    Returns
    ----------
    dataframe:
        A DataFrame with the integer lookup of the index added and the other
        category variables dropped.
    """
    drop_cols = list(index)[1:]
    dataframe = dataframe.merge(index,
                                how='left',
                                on=drop_cols)
    for col in drop_cols:
        del (dataframe[col])
    return (dataframe)


def reapply_index(dataframe, index):
    """
    This function divides a DataFrame back up by its original category
    variables and deletes the id column of the index.

    Parameters
    ----------
    dataframe:
        A DataFrame containing category variables.

    index:
        Index as DataFrame containing original category variable and integer
        index lookup.

    Returns
    ----------
    dataframe:
        A DataFrame with the category cols of the index added back in.
    """
    id_col = list(index)[0]
    dataframe = dataframe.merge(index,
                                how='left',
                                on=id_col)
    del (dataframe[id_col])
    return (dataframe)


# Optimise functions

def optimise_data_types(dataframe, verbose=True):
    """
    Function to iterate over a DataFrames columns and change data types
    to minimise memory use.
    Required to get production model to pass at MSOA level with 32GB.

    Parameters
    ----------
    dataframe:
        a DataFrame with a single zone and category variables.
        Designed for a production vector.

    verbose:
        Toggle output describing conversion process.

    Returns
    ----------
    dataframe:
        a DataFrame with optimised data types.
    """
    # TODO: Integrate length into calculation, allocate memory to smaller dfs.
    # Get list of columns in dataframe
    # TODO: Convert series to values
    lu_cols = list(dataframe)
    # Get in-memory size of dataframe to be optimised in bytes
    before = sys.getsizeof(dataframe)
    # Define long threshold
    long_df = np.power(10, 7)
    # Iterate over column names
    for col in lu_cols:
        # Get col type (numpy data type)
        col_np_type = type(dataframe[col][0]).__name__
        # Get col len
        col_len = len(dataframe[col])
        # Get number of unique values
        unq_val = len(dataframe[col].drop_duplicates())
        print_w_toggle(col, 'is', col_np_type, verbose=verbose)
        # If unsigned int8, make signed int8 (neutral)
        if col_np_type == 'uint8':
            dataframe[col] = dataframe[col].astype('int8')
            print_w_toggle('converted to', type(dataframe[col][0]).__name__,
                           verbose=verbose)
        # If unsigned int64, make int8 or int32 depending on length
        elif col_np_type == 'int64':
            if unq_val < 127:
                dataframe[col] = dataframe[col].astype('int8')
            else:
                dataframe[col] = dataframe[col].astype('int16')
            print_w_toggle('converted to', type(dataframe[col][0]).__name__,
                           verbose=verbose)
        # If float64 () make float16
        elif col_np_type == 'float64':
            # if df is less than long threshold, give it a bit of space
            dataframe[col] = dataframe[col].astype('float32')
            print_w_toggle('converted to', type(dataframe[col][0]).__name__,
                           verbose=verbose)

    # Get size after optimisation
    # TODO: Look at deepgetsizeof
    after = sys.getsizeof(dataframe)
    # Get improvement value (bytes)
    shrink = before - after
    print_w_toggle('optimised for', shrink, 'bytes', verbose=verbose)
    return (dataframe)


def refresh():
    """
    Call garbage collector. Left blank to add other memory release tricks.

    Parameters
    ----------
    null

    Returns
    ----------
    null

    """
    gc.collect()
    return ()


def frame_audit(dataframe, trips_var=None):
    """
    Take a DataFrame and form a list containing log variables for production
    run.
    Exports the length, columns, productions, df size as GB & row size.

    Parameters
    ----------
    dataframe:
        Path to ntem

    trips_var:
        A column heading to count trips in. Applies to any numeric variable.

    Returns
    ----------
    length:
        Number of rows in the DataFrame.

    cols:
        List of the column names of the DataFrame.

    productions:
        Total productions from some of 'trips', otherwise returns 'N/A'

    size_gb:
        Size of the DataFrame in GB.

    row_size:
        Size of the first row of the DataFrame
    """
    # Get df length
    length = len(dataframe)
    # Get list of df cols
    cols = list(dataframe)
    # If there's a trips col - they're productions, count them
    if trips_var:
        if 'trips' in cols:
            productions = sum(dataframe[trips_var])
    else:
        productions = 'N/A'

    # Get df size
    size = sys.getsizeof(dataframe)
    # Change size from bytes to GB
    size_gb = size / 1073741824
    # Get size of a row in bytes
    row_size = sys.getsizeof(dataframe[:1])
    # Get time at audit
    time_stamp = set_time()
    return length, cols, productions, size_gb, row_size, time_stamp


def glimpse(dataframe):
    """
    Get readable header without mucking about with GUI settings.
    """
    gl = dataframe.iloc[0:5]
    return gl


def aggregate_merger(dataframe,
                     target_segments,
                     merge_dat,
                     join_type,
                     join_cols,
                     drop_cols=False):
    """
    Placeholder for a one stop that breaks up dataframes to aggregate,
    assigns to a pot and recomplies with thinner segments.
    Only placeholder code for now.
    Could do something really fun that looks across the combinations of
    unique segments and decides which ones to to.
    """
    unq_segs = dataframe.reindex(
        [target_segments], axis=1).drop_duplicates().reset_index(drop=True)

    df_bin = []

    for index, row in unq_segs.iterrows():

        print('subsetting segment', index + 1)
        subset = dataframe.copy()

        for col in row:
            subset = dataframe[dataframe[col] == row[col]]

        print('merging time splits')
        subset = subset.merge(merge_dat,
                              how=join_type,
                              on=join_cols)

        if drop_cols:
            for col in drop_cols:
                del ()
        del (subset['purpose'], subset['time'])
        subset = subset.rename(columns={'purpose_to_home': 'purpose',
                                        'time_to_home': 'time'})
        subset['dt'] = subset['dt'].values * subset['direction_factor'].values
        del (subset['direction_factor'])
        print('re-aggregating with new mode and time')

        # TODO: Another loop here to recompile?
        subset = subset.groupby(['o_zone', 'd_zone', 'mode',
                                 'time', 'purpose',
                                 'od_index']).sum().reset_index()
        print(len(subset))
        print('subset dt', subset['dt'].sum())
        df_bin.append(subset)
        del (subset)

    dataframe = pd.concat(df_bin, sort=True)
    del (df_bin)

    return (dataframe)


def df_to_np(df,
             values,
             unq_internal_zones,
             v_heading,
             h_heading=None,
             verbose=False):
    """
    df: A Dataframe

    v_heading: heading to use as row index

    h_heading: heading to use as column index

    unq_internal_zones: unq zones to use as placeholder

    verbose = True:
        Indicates whether to print a log of the process to the terminal.
        Useful to set verbose=False when using multi-threaded loops
    """
    df = df.copy()

    placeholder = pd.DataFrame(unq_internal_zones).copy()
    col_name = list(placeholder)[0]

    if h_heading is None:
        placeholder = placeholder.rename(columns={
            list(placeholder)[0]: v_heading})
        full_placeholder = placeholder.merge(df,
                                             how='left',
                                             on=[v_heading])
        # Replace NAs with zeroes
        full_placeholder[values] = full_placeholder[values].fillna(0)

        array = full_placeholder[values].copy().values

    else:
        # Build placeholders
        ph_v = placeholder.copy()
        ph_v = ph_v.rename(columns={col_name: v_heading})
        ph_v['ph'] = 0
        ph_h = placeholder.copy()
        ph_h = ph_h.rename(columns={col_name: h_heading})
        ph_h['ph'] = 0

        # Join placeholders
        placeholder = ph_v.merge(ph_h,
                                 how='left',
                                 on='ph')
        placeholder = placeholder.drop(['ph'], axis=1)

        # Merge df onto placeholder
        full_placeholder = placeholder.merge(df,
                                             how='left',
                                             on=[h_heading, v_heading])

        # Replace NAs with zeroes
        full_placeholder[values] = full_placeholder[values].fillna(0)

        # Pivot to array
        # TODO: u - test
        array = full_placeholder.sort_values([
            v_heading,
            h_heading
        ]).pivot(
            index=v_heading,
            columns=h_heading,
            values=values
        ).values

    # Array len should be same as length of unq values
    if verbose and len(array) == len(unq_internal_zones):
        print('Matrix length=%d. Matches input constraint.' % (len(array)))

    return array


def build_path(
    base_path: PathLike,
    calib_params: Dict[str, Any],
    tp: Optional[str] = None,
    no_csv: bool = False
    ) -> PathLike:
    """
    Build a finished param path from a base string containing file location
    and a list of input params for a given run.
    """
    # BACKLOG: Update TMS filenames to include the year.
    #  Will always be 2018 in TMS.
    #  labels: demand merge, TMS
    path_output = isinstance(base_path, pathlib.Path)
    path = pathlib.Path(base_path)

    suffixes = "".join(path.suffixes)
    name = path.name.removesuffix(suffixes)

    for index, cp in calib_params.items():
        # Ignore trip length bands
        if index != 'tlb':
            # Ignore null segments
            if cp != 'none':
                cp_ph = ('_' + index + str(cp))
                name += cp_ph
    if tp:
        name += ('_tp' + str(tp))

    if not no_csv:
        name += '.csv'

    if path_output:
        return path.parent / name
    return str(path.parent / name)

# BACKLOG: Replace compile_od() with mat_p.compile_matrices()
#  labels: demand merge, EFS, TMS
def compile_od(od_folder,
               write_folder,
               compile_param_path,
               build_factor_pickle=False,
               factor_pickle_path=None):
    """
    Function to compile model format od matrices to a given specification
    """
    import_params = pd.read_csv(compile_param_path)

    # Define cols
    compilations = import_params.drop(
        'distribution_name',
        axis=1).drop_duplicates().reset_index(drop=True)

    # Some sort of check on the files
    files = os.listdir(od_folder)
    # Filter pickles or anything else odd in there
    files = [x for x in files if '.csv' in x]

    comp_ph = []
    od_pickle = {}
    for index, row in compilations.iterrows():
        compilation_name = row['compilation']

        if row['format'] == 'long':
            target_format = 'long'
        else:
            target_format = 'wide'

        subset = import_params[import_params['compilation'] == compilation_name]
        import_me = subset['distribution_name'].drop_duplicates()

        ph = []
        squares = []

        for each_one in import_me:
            reader = (od_folder + '/' + each_one)
            print('Importing ' + reader)
            temp = pd.read_csv(reader)

            if build_factor_pickle:
                square = temp.copy().drop(list(temp)[0], axis=1).values
                squares.append({each_one.replace('.csv', ''): square})
                del (square)

            temp = temp.rename(columns={list(temp)[0]: 'o_zone'})

            temp = pd.melt(temp, id_vars=['o_zone'],
                           var_name='d_zone', value_name='dt', col_level=0)

            ph.append(temp)

        if build_factor_pickle:
            compilation_dict = {}
            # Get size of first square matrix
            for key, dat in squares[0].items():
                ms = len(dat)
            ph_sq = np.zeros([ms, ms])
            # Build empty matrix
            for square in squares:
                for key, dat in square.items():
                    ph_sq = ph_sq + dat
            # If nothing: nothing, just dont div0
            ph_sq = np.where(ph_sq == 0, 0.0001, ph_sq)
            # Divide each matrix by total
            for square in squares:
                for key, dat in square.items():
                    od_factors = dat / ph_sq
                    od_factors = np.float64(od_factors)
                    compilation_dict.update({key: od_factors})

            od_pickle.update({row['compilation'].replace('.csv',
                                                         ''): compilation_dict})

        # Copy the od columns over to a placeholder for joins
        final = ph[0].copy()
        final = final.drop('dt', axis=1)

        mat_len = len(ph)

        loop = 1
        for mat in ph:
            mat = mat.rename(columns={'dt': 'dt_' + str(loop)})
            final = final.merge(mat, how='left',
                                on=['o_zone', 'd_zone'])
            loop = loop + 1

        final['dt'] = 0

        for add in range(mat_len):
            print('adding dt_' + str(add + 1))
            final['dt'] = final['dt'] + final['dt_' + str(add + 1)]
            final = final.drop('dt_' + str(add + 1), axis=1)

        # Change to numeric to order columns properly
        final['o_zone'] = final['o_zone'].astype('int32')
        final['d_zone'] = final['d_zone'].astype('int32')

        final = final.reindex(['o_zone', 'd_zone', 'dt'],
                              axis=1).groupby(['o_zone', 'd_zone']).sum(
        ).sort_values(
            ['o_zone', 'd_zone']).reset_index()

        if target_format == 'wide':
            print('translating back to wide')
            final = final.pivot(index='o_zone',
                                columns='d_zone',
                                values='dt')

        export_dict = {compilation_name: final}

        comp_ph.append(export_dict)

        # Write if you can
        if write_folder is not None:
            for mat in comp_ph:
                # Write compiled od
                for key, value in mat.items():
                    print(key)
                    if key[-4:] == '.csv':
                        c_od_out = os.path.join(write_folder, key)
                    else:
                        c_od_out = os.path.join(write_folder, key + '.csv')
                    print(c_od_out)
                    value.to_csv(c_od_out, index=True)
            if build_factor_pickle:
                fname = 'od_compilation_factors.pickle'
                if factor_pickle_path is None:
                    p_path = os.path.join(write_folder, fname)
                else:
                    p_path = os.path.join(factor_pickle_path, fname)
                print('Writing factor pickle - might take a while')
                with open(p_path, 'wb') as handle:
                    pickle.dump(od_pickle, handle,
                                protocol=pickle.HIGHEST_PROTOCOL)
    return (comp_ph)


def compile_pa(pa_folder,
               compile_param_path):
    """
    Function to compile pa matrices to a given specification

    output - takes 'normal' or 'fusion'
    """

    # Import parameters
    import_params = pd.read_csv(compile_param_path)

    # Define cols
    compilations = import_params.drop(
        'distribution_name',
        axis=1).drop_duplicates().reset_index(drop=True)

    if 'split_time' in list(compilations):
        split_time = True
        print('Splitting on time period')
    else:
        split_time = False
        print('Not splitting on time period')

    # Some sort of check on the files
    files = os.listdir(pa_folder)
    print(files)

    comp_ph = []
    for index, row in compilations.iterrows():
        compilation_name = row['compilation']

        print('Compiling ' + compilation_name)

        # Get files to import
        subset = import_params[import_params['compilation'] == compilation_name]
        # Get rid of any duplicates?
        import_list = subset['distribution_name'].drop_duplicates()

        ph = []
        for import_file in import_list:
            reader = (pa_folder + '/' + import_file)
            print('Importing ' + reader)
            temp = pd.read_csv(reader)
            temp = temp.rename(columns={list(temp)[0]: 'p_zone'})
            temp = pd.melt(temp, id_vars=['p_zone'],
                           var_name='a_zone', value_name='dt', col_level=0)
            ph.append(temp)

        current_comp = pd.concat(ph, sort=False)

        # Sidepot to handle time splits
        if split_time:
            # Get time params
            unq_time = current_comp['time'].drop_duplicates(
            ).reset_index(drop=True)

            # Loop over unique times
            for time_period in unq_time:
                print(time_period)
                # Subset
                time_sub = current_comp[current_comp['time'] == time_period]
                # Group and sum
                time_sub = time_sub.drop('time', axis=1)
                time_sub = time_sub.groupby(
                    ['p_zone', 'a_zone']).sum().reset_index()
                # Reset name
                time_sub_name = compilation_name.replace('*', str(time_period))
                # Append to output pot
                comp_ph.append({time_sub_name: time_sub})
        else:
            current_comp['p_zone'] = current_comp['p_zone'].astype(int)
            current_comp['a_zone'] = current_comp['a_zone'].astype(int)

            current_comp = current_comp.groupby(
                ['p_zone', 'a_zone']).sum().sort_values(
                ['p_zone', 'a_zone']).reset_index()
            # Append to output pot
            comp_ph.append({compilation_name: current_comp})

    return (comp_ph)


def print_w_toggle(*args, verbose):
    """
    Small wrapper to only print when verbose=True

    Parameters
    ----------
    *args:
        The text to print - can be passed in the same format as a usual
        print function

    echo:
        Whether to print the text or not
    """
    if verbose:
        print(*args)


def filter_distribution_p(internal_24hr_productions,
                          ia_name,
                          calib_params,
                          round_val=3,
                          verbose=True):
    """
    This function adds new balancing factors in to a matrix. They are returned
    in the dt col and added to whichever col comes through in zone_col
    parameter.

    Parameters
    ----------
    internal_24hr_productions:
        Internal area productions.

    ia_name:
        Internal area name for selecting zone column.

    calib_params:
        Dictionary of calibration parameters.

    verbose = True:
        Indicates whether to print a log of the process to the terminal.
        Useful to set verbose=False when using multi-threaded loops

    Returns:
    ----------
    distribution_p:
        Filtered DataFrame of distributed productions.
    """
    dp = internal_24hr_productions.copy()

    for index, cp in calib_params.items():
        # except trip length bands
        if index != 'tlb':
            if cp != 'none':
                # Ignore nulled out segments (soc or ns)
                # Force the parameter to integer, or it drops trips
                param = cp
                dp = dp[dp[index] == param]
                if verbose:
                    print(index, cp)
            else:
                print_w_toggle('Ignoring ' + index, verbose=verbose)
        else:
            print_w_toggle('Ignoring trip length bands', verbose=verbose)

    dp_cols = [ia_name, 'trips']
    dp = dp.reindex(dp_cols, axis=1)
    dp = dp.rename(columns={'trips': 'productions'})
    # Aggregate to zones
    dp = dp.groupby(ia_name).sum().reset_index()

    # Round, if it wants
    if round_val is not None:
        total_dp = dp['productions'].sum()
        dp['productions'] = dp['productions'].round(round_val)

        if verbose:
            print('Productions=%f before rounding.' % total_dp)
            print('Productions=%f after rounding.' % (dp['productions'].sum()))
            print('Same=%s' % str(total_dp == dp['productions'].sum()))
    else:
        total_dp = None

    return (dp, total_dp)


def filter_pa_vector(pa_vector,
                     ia_name,
                     calib_params,
                     value_var='trips',
                     round_val=3,
                     verbose=False):
    """
    Filter productions to target distribution type.

    Parameters
    ----------
    pa_vector:
        Internal area productions.

    ia_name:
        Internal area name for selecting zone column.

    calib_params:
        Dictionary of calibration parameters.

    value_var:
        name of total to sum

    verbose:
        Indicates whether to print a log of the process to the terminal.
        Useful to set verbose=False when using multi-threaded loops

    Return
    ----------
    distribution_p:
        Filtered DataFrame of distributed productions.
    """
    dp = pa_vector.copy()
    dp_cols = list(dp)

    for index, cp in calib_params.items():
        # except trip length bands
        if index in dp_cols:
            if not isinstance(cp, (float, int)) or not np.isnan(cp):
                # Ignore nulled out segments (soc or ns)
                dp = dp.loc[dp[index] == cp]
                if verbose:
                    print(index, cp)
            else:
                print_w_toggle('Ignoring ' + index, verbose=verbose)
        else:
            print_w_toggle('Ignoring trip length bands', verbose=verbose)

    dp_ri = [ia_name, value_var]
    dp = dp.reindex(dp_ri, axis=1)

    # Aggregate to zones
    dp = dp.groupby(ia_name).sum().reset_index()

    # Round, if it wants
    if round_val is not None:
        total_dp = dp[value_var].sum()
        dp[value_var] = dp[value_var].round(round_val)

        if verbose:
            print('Values=%f before rounding.' % total_dp)
            print('Values=%f after rounding.' % (dp[value_var].sum()))
            print('Same=%s' % str(total_dp == dp[value_var].sum()))
    else:
        total_dp = None

    return dp, total_dp


def filter_pa_cols(pa_frame,
                   ia_name,
                   calib_params,
                   round_val=3,
                   verbose=True):
    """
    Returns data for a unique segmentation and it sum
    """
    dp = pa_frame.copy()
    col_names = list(dp)
    target_col = col_names.copy()

    # This is hanging on the thread that tp and p never conflict
    # TODO: Think of something to guarantee that works
    for index, cp in calib_params.items():
        if index != 'tlb':
            prior = target_col
            target_col = [x for x in target_col if (index + str(cp)) in x]
            if len(target_col) == 0:
                print_w_toggle('Col lookup %s failed' % index, verbose=verbose)
                target_col = prior

    if len(target_col) > 1:
        print_w_toggle('Search returned >1 col', verbose=verbose)
        print(target_col)
        print_w_toggle('Picking ' + target_col[0], verbose=verbose)
    target_col = target_col[0]

    dp = dp.reindex([ia_name, target_col], axis=1)
    print(dp)
    total_dp = dp[target_col].sum()

    return dp, total_dp


def get_costs(model_lookup_path,
              calib_params,
              tp='24hr',
              iz_infill=0.5,
              replace_nhb_with_hb=False,
              ):
    """
    # TODO(BT): Stop calling this function. Move to new method in utils.costs
    """
    # units takes different parameters
    # TODO: Needs a config guide for the costs somewhere
    """
    This function imports distances or costs from a given path.

    Parameters
    ----------
    model_lookup_path:
        Model folder to look in for distances/costs. Should be in call or global.

    calib_params:
        Calibration parameters dictionary'

    tp:
        Should ultimately take 24hr & tp, usually 24hr for hb and tp for NHB.

    direction = None:
        Takes None, 'To', 'From'

    car_available = None:
        Takes None, True, False

    seed_intrazonal = True:
        Takes True or False - whether to add a value half the minimum
        interzonal value to the intrazonal cells. Currently needed for distance
        but not cost.

    Returns:
    ----------
    dat:
        DataFrame containing required cost or distance values.
    """
    # TODO: Adapt model input costs to take time periods
    # TODO: The name cost_cols is misleading
    file_sys = os.listdir(os.path.join(model_lookup_path, 'costs'))
    tp_path = [x for x in file_sys if tp in x]

    dat = pd.read_csv(os.path.join(model_lookup_path,
                                   'costs',
                                   tp_path[0]))
    cols = list(dat)

    # Get purpose and direction from calib_params
    ca = None
    purpose = None
    time_period = None

    for index, param in calib_params.items():
        # Need a purpose, if a ca is not picked up returns none
        if index == 'p':
            purpose = param
        if index == 'ca':
            if param == 1:
                ca = 'nca'
            elif param == 2:
                ca = 'ca'
        if index == 'tp':
            time_period = param

    # Purpose to string
    commute = [1]
    business = [2, 12]
    other = [3, 4, 5, 6, 7, 8, 13, 14, 15, 16, 18]
    if purpose in commute:
        str_purpose = 'commute'
    elif purpose in business:
        str_purpose = 'business'
    elif purpose in other:
        str_purpose = 'other'
    else:
        raise ValueError("Cannot convert purpose to string. " +
                         "Got %s." % str(purpose))

    # Filter down on purpose
    cost_cols = [x for x in cols if str_purpose in x]
    # Handle if we have numeric purpose costs, hope so, they're better!
    if len(cost_cols) == 0:
        if replace_nhb_with_hb:
            if purpose >= 10:
                purpose -= 10
        cost_cols = [x for x in cols if ('p' + str(purpose)) in x]

    # Filter down on car availability
    if ca is not None:
        # Have to be fussy as ca is in nca...
        if ca == 'ca':
            cost_cols = [x for x in cost_cols if 'nca' not in x]
        elif ca == 'nca':
            cost_cols = [x for x in cost_cols if 'nca' in x]

    if time_period is not None:
        cost_cols = [x for x in cost_cols if str(time_period) in x]

    target_cols = ['p_zone', 'a_zone']
    for col in cost_cols:
        target_cols.append(col)
    cost_return_name = cost_cols[0]

    dat = dat.reindex(target_cols, axis=1)
    dat = dat.rename(columns={cost_cols[0]: 'cost'})

    # Redefine cols
    cols = list(dat)

    if iz_infill is not None:
        dat = dat.copy()
        min_inter_dat = dat[dat[cols[2]] > 0]
        # Derive minimum intrazonal
        min_inter_dat = min_inter_dat.groupby(cols[0]).min().reset_index().drop(cols[1], axis=1)
        intra_dat = min_inter_dat.copy()
        intra_dat[cols[2]] = intra_dat[cols[2]] * iz_infill
        iz = dat[dat[cols[0]] == dat[cols[1]]]
        non_iz = dat[dat[cols[0]] != dat[cols[1]]]
        iz = iz.drop(cols[2], axis=1)
        # Rejoin
        iz = iz.merge(intra_dat, how='inner', on=cols[0])
        dat = pd.concat([iz, non_iz], axis=0, sort=True).reset_index(drop=True)

    return (dat, cost_return_name)


def get_distance(model_lookup_path,
                 journey_purpose=None,
                 direction=None,
                 seed_intrazonal=True):
    """
    This function imports distances or costs from a given path.

    Parameters
    ----------
    model_lookup_path:
        Model folder to look in for distances/costs. Should be in call or global.

    journey_purpose = None:
        Takes None, 'commute', 'business' or 'other'.

    direction = None:
        Takes None, 'To', 'From'

    seed_intrazonal = True:
        Takes True or False - whether to add a value half the minimum
        interzonal value to the intrazonal cells. Currently needed for distance
        but not cost.

    Returns:
    ----------
    dat:
        DataFrame containing required distance values.
    """
    file_sys = os.listdir(model_lookup_path)
    distance_path = [x for x in file_sys if 'distance' in x]
    print(distance_path)
    distance_path = distance_path[0]
    dat = pd.read_csv(model_lookup_path + '/' + distance_path)
    cols = list(dat)

    if seed_intrazonal:
        dat = dat.copy()
        min_inter_dat = dat[dat[cols[2]] > 0]
        # Derive minimum intrazonal
        min_inter_dat = min_inter_dat.groupby(
            cols[0]).min().reset_index().drop(cols[1], axis=1)
        intra_dat = min_inter_dat.copy()
        intra_dat[cols[2]] = intra_dat[cols[2]] / 2
        iz = dat[dat[cols[0]] == dat[cols[1]]]
        non_iz = dat[dat[cols[0]] != dat[cols[1]]]
        iz = iz.drop(cols[2], axis=1)
        # Rejoin
        iz = iz.merge(intra_dat, how='inner', on=cols[0])
        dat = pd.concat([iz, non_iz], axis=0, sort=True).reset_index(drop=True)

    return (dat)


def balance_by_band(band_atl,
                    distance,
                    internal_pa,
                    echo=True):
    """
    Balance based on segments.
    A lot of duplication from trip length by band
    """

    # Get total p and total a
    total_p = internal_pa.sum(axis=1).sum()
    total_a = internal_pa.sum(axis=0).sum()

    # Get min max for each
    if 'tlb_desc' in list(band_atl):
        # R built
        ph = band_atl['tlb_desc'].str.split('-', n=1, expand=True)
        band_atl['min'] = ph[0].str.replace('(', '')
        band_atl['max'] = ph[1].str.replace('[', '')
        band_atl['min'] = band_atl['min'].str.replace('(', '').values
        band_atl['max'] = band_atl['max'].str.replace(']', '').values
        del (ph)
    elif 'lower' in list(band_atl):
        # Python built
        # Convert bands to km
        band_atl['min'] = band_atl['lower'] * 1.61
        band_atl['max'] = band_atl['upper'] * 1.61

    round_mat = []
    for index, row in band_atl.iterrows():

        # Get total distance
        band_mat = np.where(
            (
                    (distance >= float(row['min']))
                    &
                    (distance < float(row['max']))
            ),
            distance,
            0)

        distance_bool = np.where(band_mat == 0, band_mat, 1)
        band_trips = internal_pa * distance_bool

        band_p = band_trips.sum(axis=1).round(3)
        # Just in:
        band_a = band_trips.sum(axis=0).round(3)

        round_trips = band_trips.round(3)
        balance_p = round_trips.sum(axis=1).sum().round(3)

        if balance_p.sum() > 0:
            refactor = band_p / balance_p
            print_w_toggle(refactor, echo=echo)
            round_trips = round_trips * refactor.round(3)

        round_mat.append(round_trips)

    balanced_pa = np.zeros((len(internal_pa), len(internal_pa)))

    for r_mat in round_mat:
        balanced_pa = r_mat + balanced_pa

    # Get output totals
    final_p = internal_pa.sum(axis=1).sum()
    final_a = internal_pa.sum(axis=0).sum()

    if echo:
        print('Productions were: ' + str(total_p))
        print('Productions now:  ' + str(final_p))
        print('Attractions were: ' + str(total_a))
        print('Attractions now:  ' + str(final_a))

    return (balanced_pa)


def single_balance(achieved_pa,
                   target_attractions,
                   target_productions,
                   echo=True):
    """
    Balance achieved attractions to target.
    Balance achieved productions to target.
    Then stop. Output will have slightly broken attractions.

    acheived_pa = distributed pa as numpy matrix,
    w format p/a, ie. p=rows, a=index.

    target_attractions = original attractions as numpy vector

    target_productions = original productions as numpy vector
    """

    # TODO: Check

    curr_a = achieved_pa.sum(axis=0)
    curr_a = np.where(curr_a == 0, 0.0001, curr_a)

    # Fill in target attractions
    targ_a = np.where(target_attractions == 0,
                      0.0001,
                      target_attractions)

    # Target over current, multiply across rows by factor
    corr_fac_a = targ_a / curr_a
    corr_fac_a = np.broadcast_to(corr_fac_a,
                                 (len(corr_fac_a),
                                  len(corr_fac_a)))

    # Apply
    achieved_pa = achieved_pa * corr_fac_a

    # Sometimes this contains NaN and outputs empty matrices. Warn the user
    if np.isnan(achieved_pa).any():
        print("WARNING: Achieved PA contains NaN. " +
              "Something has gone wrong - possibly a math error.\n" +
              "Check for RuntimeWarning above! "
              "This PA Matrix will probably be all empty")
        # Add in print in outer loop telling us which P and M combo we are on

    if echo:
        # Check and print totals
        print('Attractions balanced')
        print('Current ' + str(achieved_pa.sum(axis=0)))
        print('Target ' + str(targ_a.sum()))

    curr_p = achieved_pa.sum(axis=1)
    curr_p = np.where(curr_p == 0, 0.0001, curr_p)

    targ_p = np.where(target_productions == 0,
                      0.0001,
                      target_productions)

    # Target over current, multiply across rows by factor
    corr_fac_p = targ_p / curr_p
    # Same as before but transposed to rows
    corr_fac_p = np.broadcast_to(corr_fac_p,
                                 (len(corr_fac_p),
                                  len(corr_fac_p))).T

    # Apply
    achieved_pa = achieved_pa * corr_fac_p

    if echo:
        # Check and print totals
        print('Productions balanced')
        print('Current ' + str(achieved_pa.sum(axis=1)))
        print('Target ' + str(targ_p.sum()))

    return (achieved_pa)


def build_distribution_bins(internal_distance,
                            distribution,
                            cost_col='cost',
                            verbose=True):
    """
    This takes a distribution and rounds the trip lengths to the nearest
    whole number, then counts the number of trips by each km.

    Parameters
    ----------
    internal_distance:
        Distance in long format with p/a headings = should be flexible

    distribution:
        PA matrix containing numbers of zone to zone trips.

    Returns
    ----------
    dist_bins:
        Flat file containing counts of trips by distance.
    """
    # TODO: You can't have zero distance trips so round to ceiling
    # TODO: Broken in a few ways - needs to be fixed
    # TODO: Rewrite  exists in trip_length_audit, maybe just call that

    # Join distance
    distribution = distribution.merge(internal_distance,
                                      how='left',
                                      on=['p_zone', 'a_zone'])
    # Output trips by target trip length distribution
    dist_cols = ['dt', cost_col]
    dist_bins = distribution.reindex(dist_cols, axis=1)
    dist_bins[cost_col] = dist_bins[cost_col].round(0)
    dist_bins = dist_bins.groupby(cost_col).sum().reset_index()
    print_w_toggle('outputting distribution bin', verbose=verbose)
    return dist_bins


def define_internal_external_areas(model_lookup_path):
    """
    This function imports an internal area definition from a model folder.

    Parameters
    ----------
    model_lookup_path:
        Takes a model folder to look for an internal area definition.

    Returns
    ----------
    [0] internal_area:
        The internal area of a given model.

    [1] external_area:
        The external area of a given model.
    """
    file_sys = os.listdir(model_lookup_path)
    internal_file = [x for x in file_sys if 'internal_area' in x][0]
    external_file = [x for x in file_sys if 'external_area' in x][0]

    internal_area = pd.read_csv(model_lookup_path + '/' + internal_file)
    external_area = pd.read_csv(model_lookup_path + '/' + external_file)

    return internal_area, external_area


def get_init_params(path,
                    distribution_type='hb',
                    model_name=None,
                    mode_subset=None,
                    purpose_subset=None):
    """
    This function imports beta values for deriving distributions from
    a given path. Chunk exists as a filename in the target folder.

    Parameters
    ----------
    path:
        Path to folder containing containing required beta values.

    distribution_type = hb:
        Distribution type. Takes hb or nhb.

    model_name:
        Name of model. For pathing to new lookups.

    chunk:
        Number of chunk to take if importing by chunks. This is designed to
        make it easy to multi-process in future, and can be used to run
        distributions in parallel IDEs now.

    Return
    ----------
    initial_betas:
        DataFrame containing target betas for distribution.
    """

    if model_name is None:
        path = os.path.join(path,
                            'init_params_' + distribution_type + '.csv')
    else:
        path = os.path.join(path,
                            ''.join([model_name.lower(),
                                     '_init_params_',
                                     str(distribution_type),
                                     '.csv']))

    init_params = pd.read_csv(path)

    if mode_subset:
        init_params = init_params[
            init_params['m'].isin(mode_subset)]
    if purpose_subset:
        init_params = init_params[
            init_params['p'].isin(purpose_subset)]

    return init_params


def get_cjtw(model_lookup_path,
             model_name,
             subset=None,
             reduce_to_pa_factors=True):
    """
    This function imports census journey to work and converts types
    to ntem journey types

    Parameters
    ----------
    model_lookup_path:
        Takes a model folder to look for a cjtw zonal conversion

    subset:
        Takes a vector of model zones to filter by. Mostly for test model runs.

    Returns
    ----------
    [0] cjtw:
        A census journey to work distribution in the required zonal format.
    """
    # Lower model name for use
    mn = model_name.lower()

    # TODO: If no cjtw to model zone conversion there - run one

    file_sys = os.listdir(model_lookup_path)
    cjtw_path = [x for x in file_sys if ('cjtw_' + mn) in x][0]
    cjtw = pd.read_csv(model_lookup_path + '/' + cjtw_path)

    # CTrip End Categories
    # 1 Walk
    # 2 Cycle
    # 3 Car driver
    # 4 Car passenger
    # 5 Bus
    # 6 Rail / underground

    if subset is not None:
        sub_col = list(subset)
        sub_zones = subset[sub_col].squeeze()
        cjtw = cjtw[cjtw['1_' + mn + 'Areaofresidence'].isin(sub_zones)]
        cjtw = cjtw[cjtw['2_' + mn + 'Areaofworkplace'].isin(sub_zones)]

    method_to_mode = {'4_Workmainlyatorfromhome': '1_walk',
                      '5_Undergroundmetrolightrailtram': '6_rail_ug',
                      '6_Train': '6_rail_ug',
                      '7_Busminibusorcoach': '5_bus',
                      '8_Taxi': '3_car',
                      '9_Motorcyclescooterormoped': '2_cycle',
                      '10_Drivingacarorvan': '3_car',
                      '11_Passengerinacarorvan': '3_car',
                      '12_Bicycle': '2_cycle',
                      '13_Onfoot': '1_walk',
                      '14_Othermethodoftraveltowork': '1_walk'}
    modeCols = list(method_to_mode.keys())

    for col in modeCols:
        cjtw = cjtw.rename(columns={col: method_to_mode.get(col)})

    cjtw = cjtw.drop('3_Allcategories_Methodoftraveltowork', axis=1)
    cjtw = cjtw.groupby(cjtw.columns, axis=1).sum()
    cjtw = cjtw.reindex(['1_' + mn + 'Areaofresidence',
                         '2_' + mn + 'Areaofworkplace',
                         '1_walk', '2_cycle', '3_car',
                         '5_bus', '6_rail_ug'], axis=1)
    # Redefine mode cols for new aggregated modes
    modeCols = ['1_walk', '2_cycle', '3_car', '5_bus', '6_rail_ug']
    # Pivot
    cjtw = pd.melt(cjtw, id_vars=['1_' + mn + 'Areaofresidence',
                                  '2_' + mn + 'Areaofworkplace'],
                   var_name='mode', value_name='trips')
    cjtw['mode'] = cjtw['mode'].str[0]

    # Build distribution factors
    hb_totals = cjtw.drop(
        '2_' + mn + 'Areaofworkplace',
        axis=1
    ).groupby(
        ['1_' + mn + 'Areaofresidence', 'mode']
    ).sum().reset_index()

    hb_totals = hb_totals.rename(columns={'trips': 'zonal_mode_total_trips'})
    hb_totals = hb_totals.reindex(
        ['1_' + mn + 'Areaofresidence', 'mode', 'zonal_mode_total_trips'],
        axis=1
    )

    cjtw = cjtw.merge(hb_totals,
                      how='left',
                      on=['1_' + mn + 'Areaofresidence', 'mode'])

    # Divide by total trips to get distribution factors

    if reduce_to_pa_factors:
        cjtw['distribution'] = cjtw['trips'] / cjtw['zonal_mode_total_trips']
        cjtw = cjtw.drop(['trips', 'zonal_mode_total_trips'], axis=1)
    else:
        cjtw = cjtw.drop(['zonal_mode_total_trips'], axis=1)

    return cjtw


def single_constraint(balance,
                      constraint,
                      alpha=None,
                      beta=None,
                      cost=None):
    """
    This function applies a single constrained distribution function
    to a pa matrix to derive new balancing factors for interating a solution.

    Parameters
    ----------
    row:
        A row of data in a dataframe. Will pick up automatically if used
        in pd.apply.

    constraint = p:
        Variable to constrain by. Takes 'p' to constrain to production or 'a'
        to constrain to attraction.

    beta = -0.1:
        Beta to use in the function. Should be passed externally. Defaults
        to 1 but this should never be used (unless -0.1 gives the right
        distribution)

    Returns
    ----------
    dt = New balancing factor. Should be added to column.
    """

    if alpha is not None and beta is not None and cost is not None:
        t = (cost ** alpha) * np.exp(beta * cost)
    else:
        t = 1
    dt = balance * constraint * t

    # Log normal
    # Normal start values: mu ~ 5 sigma ~ 2
    # 1/(Cij*sigma*(2pi)**0.5)*exp(-nlog(Cij)-mu)**2/(2/sigma**2)
    # TODO: Look at graph

    return (dt)


def double_constraint(ba,
                      p,
                      bb,
                      a,
                      alpha=None,
                      beta=None,
                      cost=None):
    """
    This function applies a double constrained distribution function
    to a pa matrix to derive distributed trip rates.

    Parameters
    ----------
    row:
        A row of data in a dataframe. Will pick up automatically if used
        in pd.apply.

    beta = -0.1:
        Beta to use in the function. Should be passed externally. Defaults
        to 1 but this should never be used (unless -0.1 gives the right
        distribution)

    Returns
    ----------
    dt = Distributed trips for a given interzonal.
    """
    if alpha is not None and beta is not None and cost is not None:
        t = (cost ** alpha) * np.exp(beta * cost)
    else:
        t = 1
    dt = p * ba * a * bb * t

    return dt


def get_internal_area(lookup_folder):
    """
    Get internal area - just takes a lookup folder.
    Just functionalised so it's obvious what's failed.
    """
    directory = os.listdir(lookup_folder)
    int_path = [x for x in directory if 'internal_area' in x][0]
    int_area = pd.read_csv(os.path.join(lookup_folder, int_path))

    # Assume first col it model zone
    zone_col = int_area.columns[0]
    int_area = int_area[zone_col].tolist()

    return int_area


def get_external_area(lookup_folder):
    """
    Get internal area - just takes a lookup folder.
    Just functionalised so it's obvious what's failed.
    """
    directory = os.listdir(lookup_folder)
    ext_path = [x for x in directory if 'external_area' in x][0]
    ext_area = pd.read_csv(os.path.join(lookup_folder, ext_path))

    # Assume first col it model zone
    zone_col = ext_area.columns[0]
    ext_area = ext_area[zone_col].tolist()

    return ext_area


def get_zone_range(zone_vector):
    """
    Zone vector as series - returns contiguous zones
    Needed to avoid placeholder zones being an issue
    """
    min_zone = int(zone_vector.min())
    max_zone = int(zone_vector.max())
    unq_zones = [i for i in range(min_zone, max_zone + 1)]

    return unq_zones


def equal_ignore_order(a, b):
    """
    Return whether if a and b contain the same items, ignoring order.

    Only use when elements are neither hashable nor sortable, as this
    method is quite slow.
    if hashable use: set(a) == set(b)
    if sortable use: sorted(a) ==  sorted(b)
    """
    unmatched = list(b)
    for element in a:
        try:
            unmatched.remove(element)
        except ValueError:
            return False
    return not unmatched


def generate_distribution_name(calib_params, segments=None):
    """
    Returns a string of the key val pairs separated by spaces

    Parameters
    ----------
    calib_params:
        A dictionary in the from {'p':1, 'm':1}

    segments:
        Which keys in calib params should be used to make up
        the distribution name.
        Defaults to ['p', 'm'].

    Returns
    ----------
    distribution_name:
        A string in the form 'p1 m1'
    """
    if segments is None:
        segments = ['p', 'm', 'tp']

    dist_name = ''
    for seg, num in calib_params.items():
        if seg in segments:
            dist_name += str(seg) + str(num) + ' '
    return dist_name.strip()


def print_dict_as_table(print_dict, rounding_factor=None):
    """
    Prints a dictionary as a table using the tabulate module

    Parameters
    ----------
    print_dict:
        Keys should be headers of the table. Values should be column
        values for that header.

    rounding_factor:
        How many decimal places to round all numbers to.
        Uses built-in round function. If None, no rounding is used.
        Defaults to None.
    """
    header, vals = zip(*print_dict.items())
    vals = [[x] if not isinstance(x, list) else x for x in vals]
    if rounding_factor:
        vals = [[round(float(x[0]), rounding_factor)] for x in vals]
    print(pd.DataFrame(dict(zip(header, vals))))


def log_change_generator(max_val, min_val, n_iters, increase=False):
    """
    Yields successive values between max_val and min_val changing on a log scale.

    Due to the log scale the rate of change will start off slow. The rate of
    change will then slowly increase as the number of yields reaches n_iters.

    Parameters
    ----------
    max_val:
        The maximum value to return.

    min_val:
        The minimum value to return.

    n_iters:
        The number of iterations to run for.

    increase:
        If True, start at min_val, end at max_val.
        If False, start at max_val, end at min_val.
        Defaults to False.

    Yields
    -------
    val:
        A float somewhere between max_val and min_val
    """
    for i in range(n_iters):
        # As i increases, factor change approaches 0 (from 1) quicker
        factor_change = math.log(n_iters - i, n_iters)
        if increase:
            yield max_val - (factor_change * (max_val - min_val))
        else:
            yield min_val + (factor_change * (max_val - min_val))


def convert_table_desc_to_min_max(band_atl, in_place=False):
    """
    Returns a copy of the dataframe with added min and max columns taken from
    the table_desc column

    Parameters
    ----------
    band_atl:
        pandas.DataFrame containing a tlb_dec column.

    in_place:
        if in_place is True then a copy is not made.

    Returns
    -------
    edited_band_atl:
        A copy of the dataframe with added min and max
    """
    if not in_place:
        band_atl = band_atl.copy()

    # Get min max for each band
    if 'tlb_desc' in list(band_atl):
        # R built
        ph = band_atl['tlb_desc'].str.split('-', n=1, expand=True)
        band_atl['min'] = ph[0].str.replace('(', '')
        band_atl['max'] = ph[1].str.replace('[', '')
        band_atl['min'] = band_atl['min'].str.replace('(', '').values
        band_atl['max'] = band_atl['max'].str.replace(']', '').values
        del (ph)
    elif 'lower' in list(band_atl):
        # Python built
        # Convert bands to km
        band_atl['min'] = band_atl['lower'] * 1.61
        band_atl['max'] = band_atl['upper'] * 1.61

    return band_atl


def get_observed_estimated_trips(band_atl,
                                 distance,
                                 internal_pa):
    """
    Returns the observed, estimated, and average trip length for
    each trip length band in band_atl.

    Parameters
    ----------
    band_atl:
        pandas.dataframe containing the observed (or target) data.
        Needs the following columns: ['tbl_desc', 'band_share', 'atl']

    distance:
        The distance cost matrix of travel between zones

    internal_pa:
        The estimated internal pa. Used to determine the estimated trips

    Returns
    -------
    est_trips:
        np.array vector of estimated trips for each band share

    obs_trips:
        np.array vector of observed trips for each band share

    atl_trips:
        np.array vector of average trip length for each band share

    """
    # Loop setup
    total_est_trips = internal_pa.sum(axis=1).sum()

    band_atl = convert_table_desc_to_min_max(band_atl)
    est_trips = list()
    obs_trips = list()
    atl_trips = list()

    # Calculate the return values for each band share
    for index, row in band_atl.iterrows():
        band_mask = ((distance >= float(row['min']))
                     & (distance < float(row['max'])))

        est_trips.append(np.sum(np.where(band_mask, internal_pa, 0)))
        obs_trips.append(row['band_share'] * total_est_trips)
        atl_trips.append(row['atl'])

    return np.array(est_trips), np.array(obs_trips), np.array(atl_trips)


def get_convergence_adjustment_factors(est_trips,
                                       obs_trips,
                                       atl_trips):
    """
    Returns factors for calculating convergence and alpha/beta adjustment

    Parameters
    ----------
    est_trips:
        np.array vector of estimated trips for each band share

    obs_trips:
        np.array vector of observed trips for each band share

    atl_trips:
        np.array vector of average trip length for each band share

    Returns
    -------
    convergence_adjustment_factors:
        A list of floats. These are factors used for calculating convergence
        and alpha/beta adjustment. They are in the following order:
        alpha_fix, alpha_gradient, alpha_current,
        beta_fix, beta_gradient, beta_current
    """
    atl_mask = (atl_trips > 0)

    alpha_fix = np.sum(obs_trips * np.log(atl_trips, where=atl_mask))
    alpha_gradient = np.sum((est_trips - obs_trips) * np.log(atl_trips, where=atl_mask))
    alpha_current = np.sum(est_trips * np.log(atl_trips, where=atl_mask))

    beta_fix = np.sum(obs_trips * atl_trips)
    beta_gradient = np.sum((est_trips - obs_trips) * atl_trips)
    beta_current = np.sum(est_trips * atl_trips)

    return alpha_fix, alpha_gradient, alpha_current, beta_fix, beta_gradient, beta_current


def r_squared(estimated, observed):
    """

    Parameters
    ----------
    estimated:
        np.array vector of estimated values. Should be same size as observed
    observed

    Returns
    -------
    r_squared:
        The R squared value of estimated against observed
    """
    x = (
            1
            -
            np.sum((estimated - observed) ** 2)
            /
            np.sum((observed - np.sum(observed) / len(observed)) ** 2)
    )
    return max(x, 0)


def least_squares(df, achieved_col, target_col):
    """
    Returns the least squares difference of the achieved_col - target_col.

    Parameters
    ----------
    df:
        The pandas dataframe containing the data.

    achieved_col:
        The df column containing the achieved data.

    target_col:
        The df column containing the target data.

    Returns
    -------
    least_squares:
        least squares difference of the achieved_col - target_col
    """
    estimated = np.array(df[achieved_col].values)
    target = np.array(df[target_col].values)
    return r_squared(estimated, target)


def get_band_adjustment_factors(band_df):
    """
    Take output DF of band shares and multiply to get adjustment factors.
    """

    band_df['adj_fac'] = band_df['tbs'] / band_df['bs']
    adj_fac = band_df.drop(['tbs', 'bs'], axis=1)
    # Fill na with 0
    adj_fac['adj_fac'] = adj_fac['adj_fac'].replace(np.inf, 0)

    return adj_fac


def get_compilation_params(lookup_folder,
                           get_pa=True,
                           get_od=True):
    """
    """
    out_dict = []

    lookup_list = os.listdir(lookup_folder)

    if get_pa:
        pa_path = [x for x in lookup_list if 'pa_matrix_params' in x][0]
        pa = pd.read_csv(lookup_folder + '/' + pa_path)
        out_dict.append(pa)

    if get_od:
        od_path = [x for x in lookup_list if 'od_matrix_params' in x][0]
        od = pd.read_csv(lookup_folder + '/' + od_path)
        out_dict.append(od)

    return out_dict


def parse_mat_output(list_dir,
                     sep='_',
                     mat_type='dat',
                     file_format='.csv',
                     file_name='file'):
    """
    """

    # Define UC format
    uc = ['commute', 'business', 'other',
          'Commute', 'Business', 'Other']

    # Get target file format only
    unq_files = [x for x in list_dir if file_format in x]
    # If no numbers in then drop
    unq_files = [x for x in list_dir if any(c.isdigit() for c in x)]

    split_list = []
    for file in unq_files:
        split_dict = {file_name: file}
        file = file.replace(file_format, '')
        test = str(file).split('_')
        for item in test:
            if 'hb' in item:
                name = 'trip_origin'
                dat = item
            elif item in uc:
                name = 'p'
                dat = item
            elif item == mat_type:
                name = 'mat_type'
                dat = item
            else:
                name = ''
                dat = ''
                # name = letters, dat = numbers
                for char in item:
                    if char.isalpha():
                        name += str(char)
                    else:
                        dat += str(char)
            # Return None not nan
            if len(dat) == 0:
                dat = 'none'
            split_dict.update({name: dat})
        split_list.append(split_dict)

    segments = pd.DataFrame(split_list)
    segments = segments.replace({np.nan: 'none'})

    return segments


def unpack_tlb(tlb,
               ):
    """
    Function to unpack a trip length band table into constituents.
    Parameters
    ----------
    tlb:
        A trip length band DataFrame
    Returns
    ----------
    min_dist:
        ndarray of minimum distance by band
    max_dist:
        ndarray of maximum distance by band
    obs_trip:
        Band share by band as fraction of 1
    obs_dist:

    """
    # Convert miles from raw NTS to km
    min_dist = tlb['lower'].astype('float').to_numpy() * _M_KM
    max_dist = tlb['upper'].astype('float').to_numpy() * _M_KM
    obs_trip = tlb['band_share'].astype('float').to_numpy()
    # TODO: Check that this works!!
    obs_dist = tlb['ave_km'].astype(float).to_numpy()

    return min_dist, max_dist, obs_trip, obs_dist
