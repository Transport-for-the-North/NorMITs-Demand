# -*- coding: utf-8 -*-
"""
Created on: Fri September 11 12:46:25 2020
Updated on:

Original author: Ben Taylor
Last update made by:
Other updates made by:

File purpose:
Collection of functions for translating PA matrices into OD matrices.
TODO: After integrations with TMS, combine with pa_to_od.py
  to create a general pa_to_od.py file

"""

import numpy as np
import pandas as pd

from typing import Any
from typing import List
from typing import Dict
from itertools import product

from tqdm import tqdm

# self imports
import normits_demand as nd

from normits_demand import constants as consts
from normits_demand import efs_constants as efs_consts
from normits_demand.concurrency import multiprocessing

from normits_demand.matrices import utils as mat_utils
from normits_demand.utils import general as du
from normits_demand.utils import file_ops
from normits_demand.utils import math_utils

# Can call tms pa_to_od.py functions from here
from normits_demand.matrices.tms_pa_to_od import *

def trip_end_pa_to_od(pa_productions,
                      phi_lookup_folder: str,
                      phi_type: str,
                      modes: List[int],
                      tp_col: str = 'tp',
                      trip_col: str = 'trips',
                      round_dp: int = 4,
                      aggregate_to_wday: bool = True,
                      verbose: bool = True):
    
    """
    pa_productions
    """
    # Check for TP and trips
    # What is this - error handling !?
    if 'tp' not in list(pa_productions):
        if tp_col == 'tp':
            raise ValueError('No time period column in trip end vector')
    if 'trips' not in list(pa_productions):
        if trip_col == 'trips':
            raise ValueError('No trips column in trip end vector')

    # Initialise group and sum cols
    out_cols = list(pa_productions)
    out_cols.remove(trip_col)
    group_cols = out_cols.copy()
    append_cols = ['o_' + trip_col, 'd_' + trip_col]
    [out_cols.append(x) for x in append_cols]
    del(append_cols)
    toh_cols = out_cols.copy()
    toh_cols.remove('o_' + trip_col)

    # initialise time periods
    tp_nos = [1,2,3,4]
    tp_list = []
    [tp_list.append(tp_col+str(x)) for x in tp_nos]

    # Do subset by mode
    mode_subs = []
    for mode in modes:
        mode_sub = pa_productions.copy()
        mode_sub = mode_sub[mode_sub['m']==mode]

        phi_factors = get_time_period_splits(
            mode,
            phi_type,
            aggregate_to_wday=aggregate_to_wday,
            lookup_folder=phi_lookup_folder)
        
        # Rename phi factors
        phi_factors = phi_factors.rename(
            columns={'purpose_from_home':'p',
                     'time_from_home':tp_col})
        # BACKLOG: Handle different types of phi factor

        time_subs = []
        for time in tp_nos:
            from_home = mode_sub.copy()
            from_home = from_home[from_home[tp_col]==time]
            from_home = from_home.rename(columns={trip_col:'o_' + trip_col})
            to_home = from_home.copy()
            to_home = to_home.merge(
                phi_factors,
                how='left',
                on=['p',tp_col])
            to_home['d_' + trip_col] = (to_home['o_trips'] *
                                  to_home['direction_factor'])
            to_home = to_home.drop(
                ['tp', 'direction_factor', 'o_trips'], axis=1)
            to_home = to_home.rename(
                columns = {'time_to_home':'tp'})
            to_home = to_home.groupby(group_cols).sum().reset_index()
            to_home = to_home.sort_values(toh_cols)
            
            time_sub = from_home.merge(to_home,
                                       how='left',
                                       on=group_cols)
            time_subs.append(time_sub)
        
        mode_subs.append(pd.concat(time_subs))
    
    od_productions = pd.concat(mode_subs)
    od_productions = od_productions.groupby(
        group_cols).sum().reset_index()
    od_productions = od_productions.sort_values(
        group_cols).reset_index(drop=True)
    
    # Round
    od_productions['o_'+trip_col] = od_productions[
        'o_'+trip_col].round(round_dp)
    od_productions['d_'+trip_col] = od_productions[
        'd_'+trip_col].round(round_dp)
    
    totals = {'o_total':od_productions['o_'+trip_col].sum(),
              'd_total':od_productions['o_'+trip_col].sum()}
    if verbose:
        print(totals)

    return od_productions, totals


def simplify_phi_factors(time_period_splits: pd.DataFrame):
    """
    Simplifies time_period_splits to a case where the purpose_from_home
    is always the same as the purpose_to_home

    Parameters
    ----------
    time_period_splits:
        A time_period_splits dataframe extracted using get_time_period_splits()

    Returns
    -------
    time_period_splits only where the purpose_from_home
    is the same as purpose_to_home
    """
    time_period_splits = time_period_splits.copy()

    # Problem column doesn't exist in this case
    if 'purpose_to_home' not in time_period_splits.columns:
        return time_period_splits

    # Build a mask where purposes match
    unq_purpose = time_period_splits['purpose_from_home'].drop_duplicates()
    keep_rows = np.array([False] * len(time_period_splits))
    for p in unq_purpose:
        purpose_mask = (
            (time_period_splits['purpose_from_home'] == p)
            & (time_period_splits['purpose_to_home'] == p)
        )
        keep_rows = keep_rows | purpose_mask

    time_period_splits = time_period_splits.loc[keep_rows]

    # Filter down to just the needed col and return
    needed_cols = [
        'purpose_from_home',
        'time_from_home',
        'time_to_home',
        'direction_factor']
    return time_period_splits.reindex(needed_cols, axis='columns')


def _build_tp_pa_internal(pa_import,
                          pa_export,
                          tp_splits,
                          model_zone_col,
                          model_name,
                          matrix_format,
                          year,
                          purpose,
                          mode,
                          segment,
                          car_availability,
                          round_dp,
                          ):
    """
    The internals of build_tp_pa(). Useful for making the code more
    readable due to the number of nested loops needed

    Returns
    -------
    None
    """
    # ## READ IN TIME PERIOD SPLITS FILE ## #
    if purpose in consts.ALL_NHB_P:
        trip_origin = 'nhb'
        in_zone_col = 'o_zone'
    elif purpose in consts.ALL_HB_P:
        trip_origin = 'hb'
        in_zone_col = 'p_zone'
    else:
        raise ValueError(
            "%s is neither a home based nor non-home based purpose."
            % str(purpose)
        )

    # Rename model zone col to be more accurate
    tp_splits = tp_splits.rename(columns={model_zone_col: in_zone_col})

    # ## Read in 24hr matrix ## #
    dist_fname = du.get_dist_name(
        trip_origin,
        matrix_format,
        str(year),
        str(purpose),
        str(mode),
        str(segment),
        str(car_availability),
        csv=True
    )
    path = os.path.join(pa_import, dist_fname)
    path = file_ops.find_filename(path)
    pa_24hr = file_ops.read_df(path, index_col=0)

    # Pull the zoning system out of the index if we need to
    zoning_system = "%s_zone_id" % model_name
    if pa_24hr.columns[0] != zoning_system:
        pa_24hr.index.name = zoning_system
        pa_24hr = pa_24hr.reset_index()

    print("Working on splitting %s..." % dist_fname)

    # Convert from wide to long format
    out_zone_col = 'a_zone' if in_zone_col == 'p_zone' else 'd_zone'
    pa_24hr = du.expand_distribution(
        pa_24hr,
        year,
        purpose,
        mode,
        segment,
        car_availability,
        id_vars=in_zone_col,
        var_name=out_zone_col,
        value_name='trips'
    )

    # ## Narrow tp_split down to just the segment here ## #
    segment_id = 'soc' if purpose in consts.SOC_P else 'ns'
    segmentation_mask = du.get_segmentation_mask(
        tp_splits,
        col_vals={
            'yr': year,
            'p': purpose,
            'm': mode,
            segment_id: str(segment),
            'ca': car_availability,
        },
        ignore_missing_cols=True
    )
    tp_splits = tp_splits.loc[segmentation_mask]
    tp_splits = tp_splits.rename(columns={str(year): 'tp_split_factor'})

    # Drop either soc or ns, whichever is none is productions
    if pa_24hr['soc'].dtype == object and pa_24hr['soc'].unique()[0] == 'none':
        pa_24hr = pa_24hr.drop(columns=['soc'])
        tp_splits = tp_splits.drop(columns=['soc'])

        pa_24hr['ns'] = pa_24hr['ns'].astype(int)
        tp_splits['ns'] = tp_splits['ns'].astype(int)

    if pa_24hr['ns'].dtype == object and pa_24hr['ns'].unique()[0] == 'none':
        pa_24hr = pa_24hr.drop(columns=['ns'])
        tp_splits = tp_splits.drop(columns=['ns'])

        pa_24hr['soc'] = pa_24hr['soc'].astype(int)
        tp_splits['soc'] = tp_splits['soc'].astype(int)

    # ## Aggregate to tp_splits to match pa_24hr segmentation ## #
    merge_cols = du.intersection(list(tp_splits), list(pa_24hr))
    group_cols = merge_cols.copy() + ['tp']
    index_cols = group_cols.copy() + ['tp_split_factor']

    tp_splits = tp_splits.reindex(columns=index_cols)
    tp_splits = tp_splits.groupby(group_cols).sum().reset_index()

    # ## Apply tp-split factors to total pa_24hr ## #
    unq_time = tp_splits['tp'].drop_duplicates()
    for time in unq_time:
        # Left join to make sure we don't drop any demand
        time_factors = tp_splits[tp_splits['tp'] == time]
        tp_split_pa = pd.merge(
            pa_24hr,
            time_factors,
            on=merge_cols,
            how='left'
        )

        # Fill in any NaNs from the left join
        tp_split_pa['tp_split_factor'] = tp_split_pa['tp_split_factor'].fillna(0)
        tp_split_pa['tp'] = tp_split_pa['tp'].fillna(time).astype(int)

        # Calculate the number of trips for this time_period
        tp_split_pa['trips'] *= tp_split_pa['tp_split_factor']

        # ## Aggregate back up to our segmentation ## #
        seg_cols = du.list_safe_remove(merge_cols, [in_zone_col])
        group_cols = [in_zone_col, out_zone_col] + seg_cols + ['tp']
        index_cols = group_cols.copy() + ['trips']

        tp_split_pa = tp_split_pa.reindex(columns=index_cols)
        tp_split_pa = tp_split_pa.groupby(group_cols).sum().reset_index()

        # Build write path
        tp_pa_name = du.get_dist_name(
            str(trip_origin),
            str(matrix_format),
            str(year),
            str(purpose),
            str(mode),
            str(segment),
            str(car_availability),
            tp=str(time)
        )
        tp_pa_fname = tp_pa_name + '.csv'
        out_tp_pa_path = os.path.join(
            pa_export,
            tp_pa_fname
        )

        # Convert table from long to wide format and save
        # TODO: Generate header based on model used
        du.long_to_wide_out(
            tp_split_pa.rename(columns={in_zone_col: zoning_system}),
            v_heading=zoning_system,
            h_heading=out_zone_col,
            values='trips',
            round_dp=round_dp,
            out_path=out_tp_pa_path
        )


def efs_build_tp_pa(pa_import: str,
                    pa_export: str,
                    tp_splits: pd.DataFrame,
                    model_zone_col: str,
                    model_name: str,
                    years_needed: List[int],
                    p_needed: List[int],
                    m_needed: List[int],
                    soc_needed: List[int] = None,
                    ns_needed: List[int] = None,
                    ca_needed: List[int] = None,
                    matrix_format: str = 'pa',
                    round_dp: int = consts.DEFAULT_ROUNDING,
                    process_count: int = consts.PROCESS_COUNT
                    ) -> None:
    """
    Converts the 24hr matrices in pa_import into time_period segmented
    matrices - outputting to pa_export

    Parameters
    ----------
    pa_import:
        Path to the directory containing the 24hr matrices

    pa_export:
        Path to the directory to export the tp split matrices

    tp_splits:
        pandas DataFrame containing the time period splitting factors. The
        more segmented this dataframe is, the better the splitting will be

    model_zone_col:
        The name of the column in tp_splits that contains the model zone
        information

    years_needed:
        A list of which years of 24hr Matrices to convert.

    p_needed:
        A list of which purposes of 24hr Matrices to convert.

    m_needed:
        A list of which modes of 24hr Matrices to convert.

    soc_needed:
        A list of which soc of 24hr Matrices to convert.

    ns_needed:
        A list of which ns of 24hr Matrices to convert.

    ca_needed:
        A list of which car availabilities of 24hr Matrices to convert.

    matrix_format:
        Which format the matrix is in. Either 'pa' or 'od'

    round_dp:
        The number of decimal places to round the output values to.
        Uses efs_consts.DEFAULT_ROUNDING by default.

    process_count:
        The number of processes to use when multiprocessing. Negative numbers
        use that many processes less than the max. i.e. -1 ->
        os.cpu_count() - 1

    Returns
    -------
        None

    """
    # Validate inputs
    if matrix_format not in consts.VALID_MATRIX_FORMATS:
        raise ValueError("'%s' is not a valid matrix format."
                         % str(matrix_format))

    # Init
    soc_needed = [None] if soc_needed is None else soc_needed
    ns_needed = [None] if ns_needed is None else ns_needed
    ca_needed = [None] if ca_needed is None else ca_needed

    # ## MULTIPROCESS ## #
    unchanging_kwargs = {
        'pa_import': pa_import,
        'pa_export': pa_export,
        'matrix_format': matrix_format,
        'model_name': model_name,
        'tp_splits': tp_splits,
        'model_zone_col': model_zone_col,
        'round_dp': round_dp,
    }

    # Build a list of the changing arguments
    kwargs_list = list()
    for year in years_needed:
        loop_generator = du.segmentation_loop_generator(
            p_needed,
            m_needed,
            soc_needed,
            ns_needed,
            ca_needed
        )

        for p, m, seg, ca in loop_generator:
            kwargs = unchanging_kwargs.copy()
            kwargs.update({
                'year': year,
                'purpose': p,
                'mode': m,
                'segment': seg,
                'car_availability': ca
            })
            kwargs_list.append(kwargs)

    # Multiprocess - split by time period and write to disk
    multiprocessing.multiprocess(
        _build_tp_pa_internal,
        kwargs=kwargs_list,
        process_count=process_count
    )


def _build_od_internal(pa_import,
                       od_export,
                       model_name,
                       calib_params,
                       phi_lookup_folder,
                       phi_type,
                       aggregate_to_wday,
                       round_dp,
                       full_od_out=False,
                       echo=True):
    """
    The internals of build_od(). Useful for making the code more
    readable du to the number of nested loops needed

    TODO: merge with TMS - NOTE:
    All this code below has been mostly copied from TMS pa_to_od.py
    function of the same name. A few filenames etc have been changed
    to make sure it properly works with NorMITs demand files (This is
    du to NorMITs demand needing moving in entirety over to the Y drive)

    Returns
    -------

    """
    # Init
    tps = ['tp1', 'tp2', 'tp3', 'tp4']
    matrix_totals = list()
    dir_contents = os.listdir(pa_import)
    mode = calib_params['m']
    purpose = calib_params['p']

    model_zone_col = model_name + '_zone_id'

    # Print out some info
    dist_name = du.calib_params_to_dist_name('hb', 'od', calib_params)
    print("Generating %s..." % dist_name)

    # Get appropriate phis and filter
    phi_factors = get_time_period_splits(
        mode,
        phi_type,
        aggregate_to_wday=aggregate_to_wday,
        lookup_folder=phi_lookup_folder)
    phi_factors = simplify_phi_factors(phi_factors)
    phi_factors = phi_factors[phi_factors['purpose_from_home'] == purpose]

    # Get the relevant filenames from the dir
    dir_subset = dir_contents.copy()
    for name, param in calib_params.items():
        # Work around for 'p2' clashing with 'tp2'
        if name == 'p':
            dir_subset = [x for x in dir_subset if '_' + name + str(param) in x]
        else:
            dir_subset = [x for x in dir_subset if (name + str(param)) in x]

    # Build dict of tp names to filenames
    tp_names = {}
    for tp in tps:
        tp_names.update({tp: [x for x in dir_subset if tp in x][0]})

    # ## Build from_home dict from imported from_home PA ## #
    frh_dist = {}
    for tp, path in tp_names.items():
        dist_df = pd.read_csv(os.path.join(pa_import, path))
        zone_nums = dist_df[model_zone_col]     # Save to re-attach later
        dist_df = dist_df.drop(model_zone_col, axis=1)
        frh_dist.update({tp: dist_df})

    # ## Build to_home matrices from the from_home PA ## #
    frh_ph = {}
    for tp_frh in tps:
        du.print_w_toggle('From from_h ' + str(tp_frh), verbose=echo)
        frh_int = int(tp_frh.replace('tp', ''))
        phi_frh = phi_factors[phi_factors['time_from_home'] == frh_int]

        # Transpose to flip P & A
        frh_base = frh_dist[tp_frh].copy()
        frh_base = frh_base.values.T

        toh_dists = {}
        for tp_toh in tps:
            # Get phi
            du.print_w_toggle('\tBuilding to_h ' + str(tp_toh), verbose=echo)
            toh_int = int(tp_toh.replace('tp', ''))
            phi_toh = phi_frh[phi_frh['time_to_home'] == toh_int]
            phi_toh = phi_toh['direction_factor']

            # Cast phi toh
            phi_mat = np.broadcast_to(phi_toh,
                                      (len(frh_base),
                                       len(frh_base)))
            tp_toh_mat = frh_base * phi_mat
            toh_dists.update({tp_toh: tp_toh_mat})
        frh_ph.update({tp_frh: toh_dists})

    # ## Aggregate to_home matrices by time period ## #
    # removes the from_home splits
    tp1_list = list()
    tp2_list = list()
    tp3_list = list()
    tp4_list = list()
    for item, toh_dict in frh_ph.items():
        for toh_tp, toh_dat in toh_dict.items():
            if toh_tp == 'tp1':
                tp1_list.append(toh_dat)
            elif toh_tp == 'tp2':
                tp2_list.append(toh_dat)
            elif toh_tp == 'tp3':
                tp3_list.append(toh_dat)
            elif toh_tp == 'tp4':
                tp4_list.append(toh_dat)

    toh_dist = {
        'tp1': np.sum(tp1_list, axis=0),
        'tp2': np.sum(tp2_list, axis=0),
        'tp3': np.sum(tp3_list, axis=0),
        'tp4': np.sum(tp4_list, axis=0)
    }

    # ## Output the from_home and to_home matrices ## #
    for tp in tps:
        # Get output matrices
        output_name = tp_names[tp]

        output_from = frh_dist[tp]
        from_total = output_from.sum().sum()
        output_from_name = output_name.replace('pa', 'od_from')

        output_to = toh_dist[tp]
        to_total = output_to.sum().sum()
        output_to_name = output_name.replace('pa', 'od_to')

        # ## Gotta fudge the row/column names ## #
        # Add the zone_nums back on
        output_from = pd.DataFrame(output_from).reset_index()
        # noinspection PyUnboundLocalVariable
        output_from['index'] = zone_nums
        output_from.columns = [model_zone_col] + zone_nums.tolist()
        output_from = output_from.set_index(model_zone_col)

        output_to = pd.DataFrame(output_to).reset_index()
        output_to['index'] = zone_nums
        output_to.columns = [model_zone_col] + zone_nums.tolist()
        output_to = output_to.set_index(model_zone_col)

        # With columns fixed, created full OD output
        output_od = output_from + output_to
        output_od_name = output_name.replace('pa', 'od')

        du.print_w_toggle('Exporting ' + output_from_name, verbose=echo)
        du.print_w_toggle('& ' + output_to_name, verbose=echo)
        if full_od_out:
            du.print_w_toggle('& ' + output_od_name, verbose=echo)
        du.print_w_toggle('To ' + od_export, verbose=echo)

        # Output from_home, to_home and full OD matrices
        output_from_path = os.path.join(od_export, output_from_name)
        output_to_path = os.path.join(od_export, output_to_name)
        output_od_path = os.path.join(od_export, output_od_name)

        # Round the outputs
        output_from = output_from.round(decimals=round_dp)
        output_to = output_to.round(decimals=round_dp)
        output_od = output_od.round(decimals=round_dp)

        # BACKLOG: Add tidality checks into efs_build_od()
        #  labels: demand merge, audits, EFS
        # Auditing checks - tidality
        # OD from = PA
        # OD to = if it leaves it should come back
        # OD = 2(PA)
        output_from.to_csv(output_from_path)
        output_to.to_csv(output_to_path)
        if full_od_out:
            output_od.to_csv(output_od_path)

        matrix_totals.append([output_name, from_total, to_total])

    return matrix_totals


def efs_build_od(pa_import: str,
                 od_export: str,
                 model_name: str,
                 p_needed: List[int],
                 m_needed: List[int],
                 soc_needed: List[int],
                 ns_needed: List[int],
                 ca_needed: List[int],
                 years_needed: List[int],
                 phi_lookup_folder: str = None,
                 phi_type: str = 'fhp_tp',
                 aggregate_to_wday: bool = True,
                 verbose: bool = True,
                 round_dp: int = consts.DEFAULT_ROUNDING,
                 process_count: int = consts.PROCESS_COUNT
                 ) -> None:
    """
     This function imports time period split factors from a given path.
    TODO: write efs_build_od() docs

    Parameters
    ----------
    pa_import
    od_export
    model_name
    p_needed
    m_needed
    soc_needed
    ns_needed
    ca_needed
    years_needed
    phi_lookup_folder
    phi_type
    aggregate_to_wday
    verbose
    round_dp:
        The number of decimal places to round the output values to.
        Uses efs_consts.DEFAULT_ROUNDING by default.

    process_count:
        The number of processes to use when multiprocessing. Set to 0 to not
        use multiprocessing at all. Set to -1 to use all expect 1 available
        CPU.

    Returns
    -------
    None
    """

    # BACKLOG: Dynamically generate the path to phi_factors
    #  labels: EFS
    # Init
    if phi_lookup_folder is None:
        phi_lookup_folder = 'I:/NorMITs Demand/import/phi_factors'

    # ## MULTIPROCESS ## #
    unchanging_kwargs = {
       'pa_import': pa_import,
       'od_export': od_export,
       'model_name': model_name,
       'phi_lookup_folder': phi_lookup_folder,
       'phi_type': phi_type,
       'aggregate_to_wday': aggregate_to_wday,
       'round_dp': round_dp,
       'echo': verbose
    }

    # Build a list of the changing arguments
    kwargs_list = list()
    for year in years_needed:
        loop_generator = du.cp_segmentation_loop_generator(
            p_needed,
            m_needed,
            soc_needed,
            ns_needed,
            ca_needed
        )

        for calib_params in loop_generator:
            calib_params['yr'] = year
            kwargs = unchanging_kwargs.copy()
            kwargs.update({
                'calib_params': calib_params,
            })
            kwargs_list.append(kwargs)

    # Multiprocess - split by time period and write to disk
    matrix_totals = multiprocessing.multiprocess(
        _build_od_internal,
        kwargs=kwargs_list,
        process_count=process_count,
        in_order=True
    )

    # Make sure individual process outputs are concatenated together
    return [y for x in matrix_totals for y in x]


def maybe_get_aggregated_tour_proportions(orig: int,
                                          dest: int,
                                          model_tour_props: Dict[int, Dict[int, np.array]],
                                          lad_tour_props: Dict[int, Dict[int, np.array]],
                                          tfn_tour_props: Dict[int, Dict[int, np.array]],
                                          model2lad: Dict[int, int],
                                          model2tfn: Dict[int, int],
                                          cell_demand: float,
                                          ) -> np.array:
    # Translate to the aggregated zones
    lad_orig = model2lad.get(orig, -1)
    lad_dest = model2lad.get(dest, -1)
    tfn_orig = model2tfn.get(orig, -1)
    tfn_dest = model2tfn.get(dest, -1)

    # If the model zone tour proportions are zero, fall back on the
    # aggregated tour proportions
    bad_key = False
    if not cell_demand > 0:
        # The cell demand is zero - it doesn't matter which tour props
        # we use
        od_tour_props = model_tour_props[orig][dest]

    elif model_tour_props[orig][dest].sum() != 0:
        od_tour_props = model_tour_props[orig][dest]

    elif lad_tour_props[lad_orig][lad_dest].sum() != 0:
        # First - fall back to LAD aggregation
        od_tour_props = lad_tour_props[lad_orig][lad_dest]

        # We have a problem if this used a negative key
        bad_key = lad_orig < 0 or lad_dest < 0

    elif tfn_tour_props[tfn_orig][tfn_dest].sum() != 0:
        # Second - Try fall back to TfN Sector aggregation
        od_tour_props = tfn_tour_props[tfn_orig][tfn_dest]

        # We have a problem if this used a negative key
        bad_key = tfn_orig < 0 or tfn_dest < 0

    else:
        # If all aggregations are zero, and the zone has grown
        # we probably have a problem elsewhere
        raise ValueError(
            "Could not find a non-zero tour proportions for (O, D) pair "
            "(%s, %s). This likely means there was a problem when "
            "generating these tour proportions."
            % (str(orig), str(dest))
        )

    if bad_key:
        raise KeyError(
            "A negative key was used to get aggregated tour proportions. "
            "This probably means that either the origin or destination "
            "zone could not be found in the zone translation files. Check "
            "the zone translation files for (O, D) pair (%s, %s) "
            "to make sure."
            % (str(orig), str(dest))
        )

    return od_tour_props


def to_od_via_tour_props(n_od_vals,
                         pa_24,
                         fh_factor_dict,
                         th_factor_dict,
                         tp_needed,
                         ):
    # TODO: Write to_od_via_tour_props() docs
    # Make sure the given factors are the correct shape
    mat_utils.check_fh_th_factors(
        factor_dict=fh_factor_dict,
        tp_needed=tp_needed,
        n_row_col=n_od_vals,
    )

    mat_utils.check_fh_th_factors(
        factor_dict=th_factor_dict,
        tp_needed=tp_needed,
        n_row_col=n_od_vals,
    )

    # Create the from home OD matrices
    fh_mats = dict.fromkeys(fh_factor_dict.keys())
    for tp, factor_mat in fh_factor_dict.items():
        fh_mats[tp] = pa_24 * factor_mat

    # Create the to home OD matrices
    th_mats = dict.fromkeys(th_factor_dict.keys())
    for tp, factor_mat in th_factor_dict.items():
        th_mats[tp] = pa_24 * factor_mat

    # Validate return matrix totals
    fh_total = np.sum([x.values.sum() for x in fh_mats.values()])
    th_total = np.sum([x.values.sum() for x in th_mats.values()])
    od_total = fh_total + th_total

    # From home and to home should be the same total
    if not math_utils.is_almost_equal(fh_total, th_total):
        raise nd.NormitsDemandError(
            "From-home and to-home OD matrix totals are not the same."
            "Are the given splitting factors correct?\n"
            "from-home total: %.2f\n"
            "to-home total: %.2f\n"
            % (float(fh_total), float(th_total))
        )

    # OD total should be double the input PA
    if not math_utils.is_almost_equal(od_total, pa_24.values.sum() * 2):
        raise nd.NormitsDemandError(
            "OD Matrices total is not 2 * the input PA input."
            "Are the given splitting factors correct?"
            "2 * PA total total: %.2f\n"
            "OD total: %.2f\n"
            % (float(pa_24.values.sum() * 2), float(od_total))
        )

    return fh_mats, th_mats


def _tms_od_from_fh_th_factors_internal(pa_import,
                                        od_export,
                                        fh_th_factors_dir,
                                        trip_origin,
                                        pa_matrix_desc,
                                        od_to_matrix_desc,
                                        od_from_matrix_desc,
                                        base_year,
                                        year,
                                        p,
                                        m,
                                        seg,
                                        ca,
                                        tp_needed
                                        ) -> None:
    # TODO: Write _tms_od_from_tour_props_internal docs()
    # Load in 24hr PA
    input_dist_name = du.get_dist_name(
        trip_origin=trip_origin,
        matrix_format=pa_matrix_desc,
        year=str(year),
        purpose=str(p),
        mode=str(m),
        segment=str(seg),
        car_availability=str(ca),
        csv=True
    )
    path = os.path.join(pa_import, input_dist_name)
    pa_24 = nd.read_df(path, index_col=0, find_similar=True)
    to_numeric = lambda a: pd.to_numeric(a, errors="ignore", downcast="integer")
    pa_24.columns = to_numeric(pa_24.columns)
    pa_24.index = to_numeric(pa_24.index)

    # ## Load the from home and to home factors - always generated on base year ## #
    # Load the model zone tour proportions
    fh_factor_fname = du.get_dist_name(
        trip_origin=trip_origin,
        matrix_format='fh_factors',
        year=str(base_year),
        purpose=str(p),
        mode=str(m),
        segment=str(seg),
        car_availability=str(ca),
        suffix='.pkl'
    )
    fh_factor_dict = pd.read_pickle(os.path.join(fh_th_factors_dir, fh_factor_fname))

    th_factor_fname = fh_factor_fname.replace('fh_factors', 'th_factors')
    th_factor_dict = pd.read_pickle(os.path.join(fh_th_factors_dir, th_factor_fname))

    fh_mats, th_mats = to_od_via_tour_props(
        n_od_vals=len(pa_24.index),
        pa_24=pa_24,
        fh_factor_dict=fh_factor_dict,
        th_factor_dict=th_factor_dict,
        tp_needed=tp_needed,
    )

    # Save the generated from_home matrices
    for tp, mat in fh_mats.items():
        dist_name = du.get_dist_name(
            trip_origin=trip_origin,
            matrix_format=od_from_matrix_desc,
            year=str(year),
            purpose=str(p),
            mode=str(m),
            segment=str(seg),
            car_availability=str(ca),
            tp=str(tp),
            compressed=True
        )
        file_ops.write_df(mat, os.path.join(od_export, dist_name))

    # Save the generated to_home matrices
    for tp, mat in th_mats.items():
        dist_name = du.get_dist_name(
            trip_origin=trip_origin,
            matrix_format=od_to_matrix_desc,
            year=str(year),
            purpose=str(p),
            mode=str(m),
            segment=str(seg),
            car_availability=str(ca),
            tp=str(tp),
            compressed=True
        )
        # Need to transpose to_home before writing
        file_ops.write_df(mat.T, os.path.join(od_export, dist_name))


def _tms_od_from_fh_th_factors(pa_import: str,
                               od_export: str,
                               fh_th_factors_dir: str,
                               base_year: str = efs_consts.BASE_YEAR,
                               years_needed: List[int] = efs_consts.FUTURE_YEARS,
                               p_needed: List[int] = consts.ALL_HB_P,
                               m_needed: List[int] = efs_consts.MODES_NEEDED,
                               soc_needed: List[int] = None,
                               ns_needed: List[int] = None,
                               ca_needed: List[int] = None,
                               tp_needed: List[int] = efs_consts.TIME_PERIODS,
                               pa_matrix_desc: str = 'pa',
                               od_to_matrix_desc: str = 'od_to',
                               od_from_matrix_desc: str = 'od_from',
                               process_count: int = consts.PROCESS_COUNT,
                               ) -> None:
    """Internal function of build_od_from_fh_th_factors to handle 'tms' seg_level

    Reads in each of the matrices, as defined by the possible segments in
    p_needed, m_needed, soc_needed, ns_needed, ca_needed, and tp_needed.
    Converts the pa matrices, in pa_import, into OD matrices. Writing them out
    to od_export.
    It is assumed that the base year from-home and to-home factors should be
    used from fh_th_factors_dir.

    Parameters
    ----------
    pa_import
    od_export
    fh_th_factors_dir
    base_year
    years_needed
    p_needed
    m_needed
    soc_needed
    ns_needed
    ca_needed
    tp_needed
    process_count

    Returns
    -------

    """
    # TODO: Write _tms_od_from_tour_props() docs
    # Init
    soc_needed = [None] if soc_needed is None else soc_needed
    ns_needed = [None] if ns_needed is None else ns_needed
    ca_needed = [None] if ca_needed is None else ca_needed

    # Make sure all purposes are home based
    for p in p_needed:
        if p not in consts.ALL_HB_P:
            raise ValueError("Got purpose '%s' which is not a home based "
                             "purpose. generate_tour_proportions() cannot "
                             "handle nhb purposes." % str(p))
    trip_origin = 'hb'

    # MP placed inside this loop to prevent too much Memory being used
    for year in years_needed:
        loop_generator = du.segmentation_loop_generator(
            p_list=p_needed,
            m_list=m_needed,
            soc_list=soc_needed,
            ns_list=ns_needed,
            ca_list=ca_needed
        )

        # ## MULTIPROCESS ## #
        pbar_kwargs = {
            'desc': 'Converting segments %s from PA to OD' % year,
            'unit': 'segment',
            'disable': False,
        }

        unchanging_kwargs = {
            'pa_import': pa_import,
            'od_export': od_export,
            'fh_th_factors_dir': fh_th_factors_dir,
            'trip_origin': trip_origin,
            'pa_matrix_desc': pa_matrix_desc,
            'od_to_matrix_desc': od_to_matrix_desc,
            'od_from_matrix_desc': od_from_matrix_desc,
            'base_year': base_year,
            'year': year,
            'tp_needed': tp_needed
        }

        kwargs_list = list()
        for p, m, seg, ca in loop_generator:
            kwargs = unchanging_kwargs.copy()
            kwargs.update({
                'p': p,
                'm': m,
                'seg': seg,
                'ca': ca
            })
            kwargs_list.append(kwargs)

        multiprocessing.multiprocess(
            _tms_od_from_fh_th_factors_internal,
            kwargs=kwargs_list,
            pbar_kwargs=pbar_kwargs,
            process_count=process_count,
        )

        # Repeat loop for every wanted year


def _vdm_od_from_fh_th_factors_internal(pa_import,
                                        od_export,
                                        fh_th_factors_dir,
                                        trip_origin,
                                        pa_matrix_desc,
                                        od_to_matrix_desc,
                                        od_from_matrix_desc,
                                        base_year,
                                        year,
                                        uc,
                                        m,
                                        ca,
                                        tp_needed
                                        ) -> None:
    # TODO: Write _vdm_od_from_tour_props_internal docs()
    # TODO: Is there a way to combine get_vdm_dist_name and get_dist_name?
    #  Cracking this would make all future code super easy flexible!
    # Load in 24hr PA
    input_dist_name = du.get_vdm_dist_name(
        trip_origin=trip_origin,
        matrix_format=pa_matrix_desc,
        year=str(year),
        user_class=str(uc),
        mode=str(m),
        ca=ca,
        csv=True
    )
    path = os.path.join(pa_import, input_dist_name)
    pa_24 = nd.read_df(path, index_col=0, find_similar=True)
    pa_24.columns = pa_24.columns.astype(int)
    pa_24.index = pa_24.index.astype(int)

    # Get a list of the zone names for iterating - make sure integers
    orig_vals = [int(x) for x in pa_24.index.values]
    dest_vals = [int(x) for x in list(pa_24)]

    # ## Load the from home and to home factors - always generated on base year ## #
    # Load the model zone tour proportions
    fh_factor_fname = du.get_vdm_dist_name(
        trip_origin=trip_origin,
        matrix_format='fh_factors',
        year=str(year),
        user_class=str(uc),
        mode=str(m),
        ca=ca,
        suffix='.pkl'
    )
    fh_factor_dict = pd.read_pickle(os.path.join(fh_th_factors_dir, fh_factor_fname))

    th_factor_fname = fh_factor_fname.replace('fh_factors', 'th_factors')
    th_factor_dict = pd.read_pickle(os.path.join(fh_th_factors_dir, th_factor_fname))

    fh_mats, th_mats = to_od_via_tour_props(
        n_od_vals=len(orig_vals),
        pa_24=pa_24,
        fh_factor_dict=fh_factor_dict,
        th_factor_dict=th_factor_dict,
        tp_needed=tp_needed,
    )

    print("Writing %s converted matrices to disk..." % input_dist_name)

    # Save the generated from_home matrices
    for tp, mat in fh_mats.items():
        dist_name = du.get_vdm_dist_name(
            trip_origin=trip_origin,
            matrix_format=od_from_matrix_desc,
            year=str(year),
            user_class=str(uc),
            mode=str(m),
            ca=ca,
            tp=str(tp),
            csv=True
        )
        mat.to_csv(os.path.join(od_export, dist_name))

    # Save the generated to_home matrices
    for tp, mat in th_mats.items():
        dist_name = du.get_vdm_dist_name(
            trip_origin=trip_origin,
            matrix_format=od_to_matrix_desc,
            year=str(year),
            user_class=str(uc),
            mode=str(m),
            ca=ca,
            tp=str(tp),
            csv=True
        )
        # Need to transpose to_home before writing
        mat.T.to_csv(os.path.join(od_export, dist_name))


def _vdm_od_from_fh_th_factors(pa_import: str,
                               od_export: str,
                               fh_th_factors_dir: str,
                               base_year: str = efs_consts.BASE_YEAR,
                               years_needed: List[int] = efs_consts.FUTURE_YEARS,
                               to_needed: List[str] = efs_consts.VDM_TRIP_ORIGINS,
                               uc_needed: List[str] = efs_consts.USER_CLASSES,
                               m_needed: List[int] = efs_consts.MODES_NEEDED,
                               ca_needed: List[int] = None,
                               tp_needed: List[int] = efs_consts.TIME_PERIODS,
                               pa_matrix_desc: str = 'pa',
                               od_to_matrix_desc: str = 'od_to',
                               od_from_matrix_desc: str = 'od_from',
                               process_count: int = os.cpu_count() - 2
                               ):
    # TODO: Write _vdm_od_from_tour_props() docs
    # Init
    ca_needed = [None] if ca_needed is None else ca_needed

    # MP placed inside this loop to prevent too much Memory being used
    for year in years_needed:
        loop_generator = du.vdm_segment_loop_generator(
            to_list=to_needed,
            uc_list=uc_needed,
            m_list=m_needed,
            ca_list=ca_needed
        )

        # ## MULTIPROCESS ## #
        unchanging_kwargs = {
            'pa_import': pa_import,
            'od_export': od_export,
            'fh_th_factors_dir': fh_th_factors_dir,
            'year': year,
            'pa_matrix_desc': pa_matrix_desc,
            'od_to_matrix_desc': od_to_matrix_desc,
            'od_from_matrix_desc': od_from_matrix_desc,
            'tp_needed': tp_needed
        }

        kwargs_list = list()
        for to, uc, m, ca in loop_generator:
            kwargs = unchanging_kwargs.copy()
            kwargs.update({
                'trip_origin': to,
                'uc': uc,
                'm': m,
                'ca': ca
            })
            kwargs_list.append(kwargs)

        multiprocessing.multiprocess(
            _vdm_od_from_fh_th_factors_internal,
            kwargs=kwargs_list,
            process_count=process_count
        )

        # Repeat loop for every wanted year


def build_od_from_fh_th_factors(pa_import: str,
                                od_export: str,
                                fh_th_factors_dir: str,
                                seg_level: str,
                                seg_params: Dict[str, Any],
                                base_year: str = efs_consts.BASE_YEAR,
                                years_needed: List[int] = efs_consts.FUTURE_YEARS,
                                pa_matrix_desc: str = 'pa',
                                od_to_matrix_desc: str = 'od_to',
                                od_from_matrix_desc: str = 'od_from',
                                process_count: int = consts.PROCESS_COUNT,
                                ) -> None:
    """Builds OD Matrices from PA using the factors in fh_th_factors_dir

    Builds OD matrices based on the base year tour proportions
    at fh_th_factors_dir.

    Parameters
    ----------
    pa_import:
        Path to the directory containing the 24hr matrices.

    od_export:
        Path to the directory to export the future year tp split OD matrices.

    fh_th_factors_dir:
        Path to the directory containing the base year from-home and to-home
        splitting factors across time periods.

    seg_level:
        The name of the segmentation level to use. Should be one of
        efs_consts.SEG_LEVELS. Currently only 'tms' and 'vdm' are supported.

    seg_params:
        A dictionary defining the possible values for each of the segments.
        This is like a kwarg dictionary to pass through to the underlying
        function of the seg level. For seg_level='tms', this should look
        something like:
        {
            'p_needed': [1, 2, 3, 4, 5, 6, 7, 8],
            'm_needed': [3],
            'ca_needed': [1, 2],
        }

    base_year:
        The base year that the tour proportions were generated for

    years_needed:
        The year of the matrices that need to be converted from PA to OD

    pa_matrix_desc:
        The name used to describe the pa matrices. Usually just 'pa', but
        will sometimes be 'synthetic_pa' when dealing with TMS synthetic
        matrices.

    od_to_matrix_desc:
        The name used to describe the od to matrices. Usually just 'od_to', but
        will sometimes be 'synthetic_od_to' when dealing with TMS synthetic
        matrices.

    od_from_matrix_desc:
        The name used to describe the od from matrices. Usually just
        'od_from', but will sometimes be 'synthetic_od_from' when dealing
        with TMS synthetic matrices.

    process_count:
        The number of processes to use when multiprocessing. Set to 0 to not
        use multiprocessing at all. Set to -1 to use all expect 1 available
        CPU.

    Returns
    -------
    None
    """
    # Init
    seg_level = du.validate_seg_level(seg_level)

    # Call the correct mid-level function to deal with the segmentation
    if seg_level == 'tms':
        to_od_fn = _tms_od_from_fh_th_factors
    elif seg_level == 'vdm':
        to_od_fn = _vdm_od_from_fh_th_factors
    else:
        raise NotImplementedError(
            "'%s' is a valid segmentation level, however, we do not have a "
            "mid-level function to deal with it at the moment."
            % seg_level
        )

    to_od_fn(
        pa_import=pa_import,
        od_export=od_export,
        fh_th_factors_dir=fh_th_factors_dir,
        base_year=base_year,
        years_needed=years_needed,
        pa_matrix_desc=pa_matrix_desc,
        od_to_matrix_desc=od_to_matrix_desc,
        od_from_matrix_desc=od_from_matrix_desc,
        process_count=process_count,
        **seg_params
    )
