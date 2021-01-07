# -*- coding: utf-8 -*-
"""
Created on: Mon Nov 2 08:56:35 2020
Updated on:

Original Author: Ben Taylor
Last update made by:

File Purpose:
Module of all distribution functions for EFS
"""
import os

import pandas as pd
import numpy as np
from numpy.testing import assert_approx_equal

from typing import List
from functools import reduce

import audits
import efs_constants as consts
from demand_utilities import utils as du
from demand_utilities import concurrency as conc


def doubly_constrained_furness(seed_vals: np.array,
                               row_targets: np.array,
                               col_targets: np.array,
                               tol: float = 1e-9,
                               max_iters: int = 5000
                               ) -> np.array:
    """
    Performs a doubly constrained furness for max_iters or until tol is met

    Parameters
    ----------
    seed_vals:
        Initial values for the furness. Must be of shape
        (len(n_rows), len(n_cols)).

    row_targets:
        The target values for the sum of each row.
        i.e np.sum(matrix, axis=1)

    col_targets:
        The target values for the sum of each column
        i.e np.sum(matrix, axis=0)

    tol:
        The maximum difference between the achieved and the target values
        to tolerate before exiting early. R^2 is used to calculate the
        difference.

    max_iters:
        The maximum number of iterations to complete before exiting.

    Returns
    -------
    furnessed_matrix:
        The final furnessed matrix
    """
    # Error check
    if seed_vals.shape != (len(row_targets), len(col_targets)):
        raise ValueError(
            "The shape of the seed values given does not match the row "
            "and col targets. Seed_vals are shape %s. Expected shape (%d, %d)."
            % (str(seed_vals.shape), len(row_targets), len(col_targets))
        )

    if row_targets.sum() == 0 or col_targets.sum() == 0:
        return np.zeros(seed_vals.shape)

    # Init
    furnessed_mat = seed_vals.copy()
    early_exit = False
    cur_diff = tol + 10

    for _ in range(max_iters):
        # ## ROW CONSTRAIN ## #
        # Calculate difference factor
        row_ach = np.sum(furnessed_mat, axis=1)
        row_ach = np.where(row_ach == 0, 1, row_ach)
        diff_factor = row_targets / row_ach

        # adjust rows
        furnessed_mat = (furnessed_mat.T * diff_factor).T

        # ## COL CONSTRAIN ## #
        # Calculate difference factor
        col_ach = np.sum(furnessed_mat, axis=0)
        col_ach = np.where(col_ach == 0, 1, col_ach)
        diff_factor = col_targets / col_ach

        # adjust cols
        furnessed_mat = furnessed_mat * diff_factor

        # Calculate the diff - leave early if met
        row_diff = (row_targets - np.sum(furnessed_mat, axis=1)) ** 2
        col_diff = (col_targets - np.sum(furnessed_mat, axis=0)) ** 2
        cur_diff = np.sum(row_diff + col_diff) ** .5
        if cur_diff < tol:
            early_exit = True
            break

        if np.isnan(cur_diff):
            return np.zeros(furnessed_mat.shape)

    # Warn the user if we exhausted our number of loops
    if not early_exit:
        print("WARNING! The doubly constrained furness exhausted its max "
              "number of loops (%d), while achieving an R^2 difference of "
              "%f. The values returned may not be accurate."
              % (max_iters, cur_diff))

    return furnessed_mat


def _distribute_pa_internal(productions,
                            attraction_weights,
                            seed_dist_dir,
                            trip_origin,
                            calib_params,
                            zone_col,
                            p_col,
                            m_col,
                            seg_col,
                            ca_col,
                            tp_col,
                            max_iters,
                            seed_infill,
                            furness_tol,
                            seed_mat_format,
                            echo,
                            audit_out,
                            dist_out,
                            ):
    """
    Internal function of distribute_pa(). See that for full documentation.
    """
    # Init
    productions = productions.copy()
    a_weights = attraction_weights.copy()
    unique_col = 'trips'

    out_dist_name = du.calib_params_to_dist_name(
        trip_origin=trip_origin,
        matrix_format='pa',
        calib_params=calib_params,
        csv=True
    )
    print("Furnessing %s ..." % out_dist_name)

    # Don't need year in calib params for much of this
    year = calib_params['yr']
    del calib_params['yr']

    # Read in the seed distribution - ignoring year
    seed_fname = du.calib_params_to_dist_name(
        trip_origin=trip_origin,
        matrix_format=seed_mat_format,
        calib_params=calib_params,
        csv=True
    )
    seed_dist = pd.read_csv(os.path.join(seed_dist_dir, seed_fname), index_col=0)
    seed_dist.columns = seed_dist.columns.astype(int)

    # Quick check that seed is valid
    if len(seed_dist.columns.difference(seed_dist.index)) > 0:
        raise ValueError("The index and columns of the seed distribution"
                         "'%s' do not match."
                         % seed_fname)

    # Assume seed contains all the zones numbers
    unique_zones = list(seed_dist.index)

    # ## FILTER P/A TO SEGMENTATION ## #
    if calib_params.get('soc') is not None:
        seg = calib_params.get('soc')
    else:
        seg = calib_params.get('ns')

    base_filter = {
        p_col: calib_params.get('p'),
        m_col: calib_params.get('m'),
        seg_col: str(seg),
        ca_col: calib_params.get('ca'),
        tp_col: calib_params.get('tp')
    }

    productions = du.filter_by_segmentation(productions,
                                            df_filter=base_filter,
                                            fit=True)

    # Soc0 is always special - do this to avoid dropping demand
    # This is saying: If soc is 0, ignore soc segmentation!
    # Can we fo this for productions too?
    if base_filter.get('soc', -1) == '0':
        base_filter_t = base_filter.copy()
        base_filter_t.pop('soc')
        a_weights = du.filter_by_segmentation(a_weights, base_filter_t, fit=True)
    else:
        a_weights = du.filter_by_segmentation(a_weights, base_filter, fit=True)

    # Rename columns for furness
    productions = productions.rename(columns={str(year): unique_col})
    a_weights = a_weights.rename(columns={str(year): unique_col})

    # Tidy up
    productions = productions.reindex([zone_col, unique_col], axis='columns')
    productions = productions.groupby(zone_col).sum().reset_index()
    a_weights = a_weights.reindex([zone_col, unique_col], axis='columns')
    a_weights = a_weights.groupby(zone_col).sum().reset_index()

    # ## MATCH P/A ZONES ## #
    if productions.empty:
        raise ValueError(
            "Something has gone wrong. I have no productions. Perhaps none "
            "exist for the given segmentation: %s" % str(calib_params)
        )

    if a_weights.empty:
        raise ValueError(
            "Something has gone wrong. I have no attractions. Perhaps none "
            "exist for the given segmentation: %s" % str(calib_params)
        )

    productions, a_weights = du.match_pa_zones(
        productions=productions,
        attractions=a_weights,
        zone_col=zone_col,
        unique_zones=unique_zones
    )

    # ## BALANCE P/A FORECASTS ## #
    if productions[unique_col].sum() != a_weights[unique_col].sum():
        du.print_w_toggle("Row and Column targets do not match. Balancing...",
                          echo=echo)
        a_weights[unique_col] /= (
            a_weights[unique_col].sum() / productions[unique_col].sum()
        )

    pa_dist = furness_pandas_wrapper(
        row_targets=productions,
        col_targets=a_weights,
        seed_values=seed_dist,
        max_iters=max_iters,
        seed_infill=seed_infill,
        idx_col=zone_col,
        unique_col=unique_col,
        tol=furness_tol
    )

    if audit_out is not None:
        # Create output filename
        audit_fname = seed_fname.replace('_pa_', '_dist_audit_')
        audit_path = os.path.join(audit_out, audit_fname)

        audits.audit_furness(
            row_targets=productions,
            col_targets=a_weights,
            furness_out=pa_dist,
            output_path=audit_path,
            idx_col=zone_col,
            unique_col=unique_col,
            row_prefix='p',
            col_prefix='a',
            index_name='zone'
        )

    # ## OUTPUT TO DISK ## #
    # MODEL ZONE!
    output_path = os.path.join(dist_out, out_dist_name)
    pa_dist.to_csv(output_path)


def distribute_pa(productions: pd.DataFrame,
                  attraction_weights: pd.DataFrame,
                  seed_dist_dir: str,
                  dist_out: str,
                  years_needed: List[str],
                  p_needed: List[int],
                  m_needed: List[int],
                  soc_needed: List[int] = None,
                  ns_needed: List[int] = None,
                  ca_needed: List[int] = None,
                  tp_needed: List[int] = None,
                  zone_col: str = 'model_zone_id',
                  p_col: str = 'p',
                  m_col: str = 'm',
                  soc_col: str = 'soc',
                  ns_col: str = 'ns',
                  ca_col: str = 'ca',
                  tp_col: str = 'tp',
                  trip_origin: str = 'hb',
                  max_iters: int = 5000,
                  seed_infill: float = 1e-5,
                  furness_tol: float = 1e-2,
                  seed_mat_format: str = 'enhpa',
                  echo: bool = False,
                  audit_out: str = None,
                  process_count: int = consts.PROCESS_COUNT
                  ) -> None:
    """
    Furnesses the given productions and attractions

    years_needed, p_needed, m_needed, soc_needed, ns_needed, ca_needed, and
    tp_needed can all be used to control to segmentation used when furnessing
    to productions and attractions. If no productions or attractions exist
    for the given segmentation then an error will be thrown.

    Parameters
    ----------
    productions:
        Dataframe of segmented productions values across all years in
        years_needed.

    attraction_weights:
        Dataframe of segmented attraction weight factors across all years in
        years_needed.

    seed_dist_dir:
        Path to the folder where the seed distributions are located. Filenames
        will automatically be generated based on the segmentation being
        furnessed. If a file cannot be found, and error will be raised

    dist_out:
        Path to the directory to output the furnessed distributions. Filenames
        will automatically be generated based on the segmentation that has
        been furnessed.

    years_needed:
        A list of the base and future years that should be furnessed. All years
        in this list must have a column in productions and attraction_weights.

    p_needed:
        A list of the purposes that should be distributed. Any purposes that are
        not in this list, but do exist in the productions/attraction_weights
        will be ignored.

    m_needed:
        A list of the modes that should be distributed (at the moment this
        should only be one mode, but this allows multi-modal in the future).
        All productions and attraction_weights given are assumed to be of the
        mode in this List.

    soc_needed:
        A list of the soc categories that should be distributed. Any soc
        categories that are not in this list, but do exist in the productions/
        attraction_weights will be ignored.

    ns_needed:
        A list of the ns-sec categories that should be distributed. Any ns-sec
        categories that are not in this list, but do exist in the productions/
        attraction_weights will be ignored.

    ca_needed:
        A list of the car availabilities that should be distributed. Any car
        availabilities that are not in this list, but do exist in the
        productions/attraction_weights will be ignored.

    tp_needed:
        A list of time periods that should be distributed. Any time periods
        that are not in this list, but do exist in the productions/
        attraction_weights will be ignored.

    zone_col:
        Name of the column in productions/attraction_weights that contains
        the zone data.

    p_col:
        Name of the column in productions/attraction_weights that contains
        the purpose data (if segmenting by purpose).

    m_col:
        Name of the column in productions/attraction_weights that contains
        the mode data (if segmenting by mode).

    soc_col:
        Name of the column in productions/attraction_weights that contains
        the soc data (if segmenting by soc).

    ns_col:
        Name of the column in productions/attraction_weights that contains
        the ns-sec data (if segmenting by ns-sec).

    ca_col:
        Name of the column in productions/attraction_weights that contains
        the car availability data (if segmenting by car availability).

    tp_col:
        Name of the column in productions/attraction_weights that contains
        the time periods data (if segmenting by time periods).

    trip_origin:
        The origin of the trips being distributed. This will be used to
        generate the seed and output filenames. Will accept 'hb' or 'nhb'.

    max_iters
        The maximum number of iterations to complete within the furness process
        before exiting, if a solution has not been found a message will be
        printed to the terminal.

    seed_infill:
        The value to infill any seed values that are 0.

    furness_tol:
        The maximum difference between the achieved and the target values
        to tolerate before exiting the furness early. R^2 is used to calculate
        the difference.

    seed_mat_format:
        The format of the seed matrices. Usually 'enhpa' from TMS disaggregator

    echo:
        Controls the amount of printing to the terminal. If False, most of the
        print outs will be ignored.

    audit_out:
        Path to a directory to output all audit checks.

    process_count:
        The number of processes to use when distributing all segmentations.
        Positive numbers equate to the number of processes to call - this
        should not usually be more than the number of cores available.
        Negative numbers equate to the maximum number of cores available, less
        that amount. If Multiprocessing should not be used, set this value to
        0.

    Returns
    -------
    None
    """
    # Init
    productions = productions.copy()
    attraction_weights = attraction_weights.copy()
    soc_needed = [None] if soc_needed is None else soc_needed
    ns_needed = [None] if ns_needed is None else ns_needed
    ca_needed = [None] if ca_needed is None else ca_needed
    tp_needed = [None] if tp_needed is None else tp_needed

    # Make sure the soc and ns columns are strings
    if 'soc' in list(productions):
        productions['soc'] = productions['soc'].astype(str)
    if 'ns' in list(productions):
        productions['ns'] = productions['ns'].astype(str)
    if 'soc' in list(attraction_weights):
        attraction_weights['soc'] = attraction_weights['soc'].astype(str)

    # ## Make sure the segmentations we're asking for exist in P/A ## #
    # Build a dict of the common arguments
    kwargs = {
        'p_needed': p_needed,
        'm_needed': m_needed,
        'soc_needed': soc_needed,
        'ns_needed': ns_needed,
        'ca_needed': ca_needed,
        'tp_needed': tp_needed,
        'p_col': p_col,
        'm_col': m_col,
        'soc_col': soc_col,
        'ns_col': ns_col,
        'ca_col': ca_col,
        'tp_col': tp_col,

    }

    # Check the productions and attractions
    productions = du.ensure_segmentation(productions, **kwargs)
    attraction_weights = du.ensure_segmentation(attraction_weights,
                                                ignore_ns=True,
                                                ignore_ca=True,
                                                **kwargs)

    # Get P/A columns
    p_cols = list(productions.columns)
    for year in years_needed:
        p_cols.remove(year)

    a_cols = list(attraction_weights.columns)
    for year in years_needed:
        a_cols.remove(year)

    # TODO: Fix area_type in production model
    if 'area_type' in p_cols:
        p_cols.remove('area_type')
        productions = productions.drop('area_type', axis='columns')
        productions = productions.groupby(p_cols).sum().reset_index()

    # Distribute P/A per segmentation required
    for year in years_needed:
        # Filter P/A for this year
        p_index = p_cols.copy() + [year]
        yr_productions = productions.reindex(p_index, axis='columns')

        a_index = a_cols.copy() + [year]
        yr_a_weights = attraction_weights.reindex(a_index, axis='columns')

        # Loop through segmentations for this year
        loop_generator = du.cp_segmentation_loop_generator(
            p_list=p_needed,
            m_list=m_needed,
            soc_list=soc_needed,
            ns_list=ns_needed,
            ca_list=ca_needed,
            tp_list=tp_needed,
        )

        # ## MULTIPROCESS ## #
        unchanging_kwargs = {
            'productions': yr_productions,
            'attraction_weights': yr_a_weights,
            'seed_dist_dir': seed_dist_dir,
            'trip_origin': trip_origin,
            'zone_col': zone_col,
            'p_col': p_col,
            'm_col': m_col,
            'ca_col': ca_col,
            'tp_col': tp_col,
            'max_iters': max_iters,
            'seed_infill': seed_infill,
            'furness_tol': furness_tol,
            'seed_mat_format': seed_mat_format,
            'echo': echo,
            'audit_out': audit_out,
            'dist_out': dist_out,
        }

        # Build a list of all kw arguments
        kwargs_list = list()
        for calib_params in loop_generator:
            # Set the column name of the ns/soc column
            if calib_params['p'] in consts.SOC_P:
                seg_col = soc_col
            elif calib_params['p'] in consts.NS_P:
                seg_col = ns_col
            else:
                raise ValueError("'%s' does not seem to be a valid soc or ns "
                                 "purpose." % str(calib_params['p']))

            # Add in year
            calib_params['yr'] = int(year)

            kwargs = unchanging_kwargs.copy()
            kwargs.update({
                'calib_params': calib_params,
                'seg_col': seg_col,
            })
            kwargs_list.append(kwargs)

        conc.multiprocess(
            fn=_distribute_pa_internal,
            kwargs=kwargs_list,
            process_count=process_count
        )


def furness_pandas_wrapper(seed_values: pd.DataFrame,
                           row_targets: pd.DataFrame,
                           col_targets: pd.DataFrame,
                           max_iters: int = 2000,
                           seed_infill: float = 1e-3,
                           tol: float = 1e-9,
                           idx_col: str = 'model_zone_id',
                           unique_col: str = 'trips',
                           ):
    """
    Wrapper around doubly_constrained_furness() to handle pandas in/out

    Internally checks and converts the pandas inputs into numpy in order to
    run doubly_constrained_furness(). Converts the output back into pandas
    at the end

    Parameters
    ----------
    seed_values:
        The seed values to use for the furness. The index and columns must
        match the idx_col of row_targets and col_targets.

    row_targets:
        The target values for the sum of each row. In production/attraction
        furnessing, this would be the productions. The idx_col must match
        the idx_col of col_targets.

    col_targets:
        The target values for the sum of each column. In production/attraction
        furnessing, this would be the attractions. The idx_col must match
        the idx_col of row_targets.

    max_iters:
        The maximum number of iterations to complete before exiting.

    tol:
        The maximum difference between the achieved and the target values
        to tolerate before exiting early. R^2 is used to calculate the
        difference.

    seed_infill:
        The value to infill any seed values that are 0.

    idx_col:
        Name of the columns in row_targets and col_targets that contain the
        index data that matches seed_values index/columns

    unique_col:
        Name of the columns in row_targets and col_targets that contain the
        values to target during the furness.

    Returns
    -------
    furnessed_matrix:
        The final furnessed matrix, in the same format as seed_values
    """
    # Init
    row_targets = row_targets.copy()
    col_targets = col_targets.copy()
    seed_values = seed_values.copy()

    row_targets = row_targets.reindex([idx_col, unique_col], axis='columns')
    col_targets = col_targets.reindex([idx_col, unique_col], axis='columns')
    row_targets = row_targets.set_index(idx_col)
    col_targets = col_targets.set_index(idx_col)

    # ## VALIDATE INPUTS ## #
    ref_index = row_targets.index
    if len(ref_index.difference(col_targets.index)) > 0:
        raise ValueError("Row and Column target indexes do not match.")

    if len(ref_index.difference(seed_values.index)) > 0:
        raise ValueError("Row and Column target indexes do not match "
                         "seed index.")

    if len(ref_index.difference(seed_values.columns)) > 0:
        raise ValueError("Row and Column target indexes do not match "
                         "seed columns.")

    assert_approx_equal(
        row_targets[unique_col].sum(),
        col_targets[unique_col].sum(),
        err_msg="Row and Column target totals do not match. Cannot Furness."
    )

    # Now we know everything matches, we can convert to numpy
    row_targets = row_targets.values.flatten()
    col_targets = col_targets.values.flatten()
    seed_values = seed_values.values

    # ## TIDY AND INFILL SEED ## #
    seed_values = np.where(seed_values <= 0, seed_infill, seed_values)
    seed_values /= seed_values.sum()

    furnessed_mat = doubly_constrained_furness(
        seed_vals=seed_values,
        row_targets=row_targets,
        col_targets=col_targets,
        tol=tol,
        max_iters=max_iters
    )

    # ## STICK BACK INTO PANDAS ## #
    furnessed_mat = pd.DataFrame(
        index=ref_index,
        columns=ref_index,
        data=furnessed_mat
    )

    return furnessed_mat


def nhb_furness(p_import: str,
                a_import: str,
                seed_dist_dir: str,
                pa_export: str,
                model_name: str,
                years_needed: List[str] = consts.ALL_YEARS,
                p_needed: List[int] = consts.NHB_PURPOSES_NEEDED,
                m_needed: List[int] = consts.MODES_NEEDED,
                soc_needed: List[int] = None,
                ns_needed: List[int] = None,
                ca_needed: List[int] = None,
                tp_needed: List[int] = None,
                nhb_productions_fname: str = 'nhb_productions.csv',
                zone_col: str = 'model_zone_id',
                p_col: str = 'purpose_id',
                m_col: str = 'mode_id',
                soc_col: str = 'soc',
                ns_col: str = 'ns',
                ca_col: str = 'car_availability_id',
                tp_col: str = 'tp',
                trip_origin: str = 'nhb',
                unique_col: str = 'trips',
                max_iters: int = 5000,
                seed_infill: float = 1e-5,
                echo: bool = False,
                audit_out: str = None
                ) -> None:
    # TODO: Write nhb_furness() docs
    # Productions and attractions need to be in the same zoning system
    # ## GET PRODUCTIONS ## #
    # Read from disk
    productions = list()
    for year in years_needed:
        year_p_fname = '_'.join(["yr" + str(year), nhb_productions_fname])
        df = pd.read_csv(os.path.join(p_import, year_p_fname))
        productions.append(df.rename(columns={
            unique_col: year,
            'p_zone': zone_col,
            'p': p_col,
            'm': m_col
        }))

    # merge all productions into one dataframe
    p_cols = [list(p) for p in productions]
    merge_cols = reduce(lambda x, y: du.intersection(x, y), p_cols)
    productions = reduce(lambda x, y: pd.merge(x, y, on=merge_cols), productions)

    # ## GET ATTRACTIONS ##
    # Read from disk
    attractions_fname = '_'.join([model_name, 'nhb_attractions.csv'])
    attractions = pd.read_csv(os.path.join(a_import, attractions_fname))

    # Convert to weights
    attraction_weights = du.convert_to_weights(attractions, years_needed)

    return distribute_pa(
        productions,
        attraction_weights,
        seed_dist_dir,
        dist_out=pa_export,
        years_needed=years_needed,
        p_needed=p_needed,
        m_needed=m_needed,
        soc_needed=soc_needed,
        ns_needed=ns_needed,
        ca_needed=ca_needed,
        tp_needed=tp_needed,
        zone_col=zone_col,
        p_col=p_col,
        m_col=m_col,
        soc_col=soc_col,
        ns_col=ns_col,
        ca_col=ca_col,
        tp_col=tp_col,
        trip_origin=trip_origin,
        max_iters=max_iters,
        seed_infill=seed_infill,
        echo=echo,
        audit_out=audit_out
    )


def furness_old(productions: pd.DataFrame,
            attractions: pd.DataFrame,
            distributions: pd.DataFrame,
            max_iters: int = 1000,
            replace_zero_values: bool = True,
            constrain_on_production: bool = True,
            constrain_on_attraction: bool = True,
            zero_replacement_value: float = 0.01,
            target_percentage: float = 0.7,
            exit_early_tol: int = 40,
            audit_outputs: bool = False,
            echo=True,
            zone_col: str = 'model_zone_id'
            ) -> pd.DataFrame:
    """
    Run function for the Furness Process class.

    Provides a full Furness with options for number of
    iterations, whether to replace zero values on the
    seed, whether to constrain on production or attraction
    or both and what to replace zero values on the seed
    with.

    TODO: Make use of max_iters

    Parameters
    ----------
    productions:
        A dataframe with the columns "model_zone_id" and
        "production_forecast".

    attractions:
        A dataframe with the columns "model_zone_id" and
        "attraction_forecast".

    distributions:
        A dataframe with the columns "p_zone", "a_zone"
        and "seed_values".

    number_of_iterations:
        Amount of Furness iterations to be done.
        Default value is 1.

    replace_zero_values:
        Whether to replace zero values in the seed
        distribution dataframe.
        Default value is True.
        Possible values are True and False.

    constrain_on_production:
        Whether to constraining on production.
        Defaults to True.
        Possible values are True and False.

    constrain_on_attraction:
        Whether to constraining on attraction.
        Defaults to True.
        Possible values are True and False.

    zero_replacement_value:
        What value to replace zero values in the seed
        distribution dataframe with.
        Defaults to 0.01.

    target_percentage:
        TODO: Write this docstring

    audit_outputs:
        Whether to produce audit outputs.
        Defaults to True.
        Possible values are True and False.

    Return
    ----------
    furnessed_frame:
        The complete final furnessed frame with the
        columns "p_zone", "a_zone" and "dt".
    """

    print("WARNING! This code is now deprecated. Use furness_pandas_wrapper() "
          "instead, it is much much faster and converts to numpy!")

    # Grab only the necessary columns
    productions = productions[[zone_col, "production_forecast"]].copy()
    attractions = attractions[[zone_col, "attraction_forecast"]].copy()
    distributions = distributions[["p_zone", "a_zone", "seed_values"]].copy()

    # Ensure correct formats
    productions[zone_col] = productions[zone_col].astype(int)
    attractions[zone_col] = attractions[zone_col].astype(int)
    distributions['p_zone'] = distributions['p_zone'].astype(int)
    distributions['a_zone'] = distributions['a_zone'].astype(int)

    # Get a set of production and attraction zone ids (for checks)
    production_zones = set(productions["model_zone_id"].tolist())
    attraction_zones = set(attractions["model_zone_id"].tolist())

    # Get a set of distribution zone ids (for checks)
    distribution_zones = set(distributions["p_zone"].tolist())

    # ensure production and attraction zones match
    if production_zones != attraction_zones:
        raise ValueError("In FurnessProcess.run(): "
                         + "Production and attraction zones "
                         + "do not match.")

    # checking production total versus attraction total
    if (productions["production_forecast"].sum()
            !=
            attractions["attraction_forecast"].sum()):
        print("In FurnessProcess.run(): "
              + "Attraction forecast and production forecast do not match, "
              + "shifting attraction forecast to match to production "
              + "forecast.")

        # production and attraction do not match
        # need to scale attraction forecast to production forecast
        attractions["attraction_forecast"] /= (
            attractions["attraction_forecast"].sum()
            /
            productions["production_forecast"].sum()
        )

    if replace_zero_values:
        zero_seed_mask = (distributions["seed_values"] == 0)
        distributions.loc[zero_seed_mask, "seed_values"] = zero_replacement_value

    # Get percentage of productions to each a_zone from each p_zone
    # ph = list()
    # for zone in distribution_zones:
    #     temp_dists = distributions[distributions["p_zone"] == zone].copy()
    #     temp_dists['seed_values'] /= temp_dists['seed_values'].sum()
    #     ph.append(temp_dists)
    # distributions = pd.concat(ph)

    # Loop Init
    furnessed_frame = pd.merge(
        distributions.copy(),
        productions,
        left_on='p_zone',
        right_on='model_zone_id'
    ).drop('model_zone_id', axis='columns')

    furnessed_frame = pd.merge(
        furnessed_frame,
        attractions,
        left_on='a_zone',
        right_on='model_zone_id'
    ).drop('model_zone_id', axis='columns')

    # Calculate trip distributions
    furnessed_frame["dt"] = 0
    furnessed_frame["dt"] = (
        (
            (
                furnessed_frame["seed_values"].values
                *
                furnessed_frame["production_forecast"].values
            )
            +
            (
                furnessed_frame["seed_values"].values
                *
                furnessed_frame["attraction_forecast"].values
            )
        )
        /
        2
    )

    # Calculate the total production of each zone
    production_zone_total = furnessed_frame[["p_zone", "dt"]].groupby(
        by=["p_zone"],
        as_index=False
    ).sum()
    production_zone_total.rename(
        columns={"dt": "production_zone_total"},
        inplace=True
    )

    production_zone_total = pd.merge(
        production_zone_total,
        furnessed_frame[["p_zone", "production_forecast"]].drop_duplicates(
            subset=["p_zone"]
        ),
        on="p_zone"
    )

    # Calculate production accuracy
    production_zone_total["production_zone_accuracy"] = (
        abs(
            (
                production_zone_total["production_forecast"].values
                /
                production_zone_total["production_zone_total"].values
            )
            -
            1
        )
    )

    # Calculate the total attraction of each zone
    attraction_zone_total = furnessed_frame[["a_zone", "dt"]].groupby(
        by=["a_zone"],
        as_index=False
    ).sum()
    attraction_zone_total.rename(
        columns={
            "dt": "attraction_zone_total"
        },
        inplace=True
    )

    attraction_zone_total = pd.merge(
        attraction_zone_total,
        furnessed_frame[["a_zone", "attraction_forecast"]].drop_duplicates(
            subset=["a_zone"]
        ),
        on="a_zone"
    )

    # Calculate attraction accuracy
    attraction_zone_total["attraction_zone_accuracy"] = (
        abs(
            (
                attraction_zone_total["attraction_forecast"].values
                /
                attraction_zone_total["attraction_zone_total"].values
            )
            -
            1
        )
    )

    pa_acc = (
        1
        -
        (
            (
                production_zone_total["production_zone_accuracy"].mean()
                +
                attraction_zone_total["attraction_zone_accuracy"].mean()
            )
            /
            2
        )
    )

    furnessed_frame = pd.merge(
        furnessed_frame,
        production_zone_total[["p_zone", "production_zone_total"]],
        on="p_zone"
    )
    furnessed_frame = pd.merge(
        furnessed_frame,
        attraction_zone_total[["a_zone", "attraction_zone_total"]],
        on="a_zone"
    )

    # TODO: Refactor this to avoid the code duplication before entering while loop
    i = 1

    best_pa_acc = (pa_acc, i)

    while pa_acc < target_percentage:
        du.print_w_toggle("Distribution iteration: %d" % i, echo=echo)
        du.print_w_toggle("Distribution iteration: %.9f" % pa_acc, echo=echo)
        if constrain_on_production and constrain_on_attraction:
            furnessed_frame["dt"] = (
                (
                    (
                        furnessed_frame["dt"].values
                        *
                        (
                            furnessed_frame["production_forecast"].values
                            /
                            furnessed_frame["production_zone_total"].values
                        )
                    )
                    +
                    (
                        furnessed_frame["dt"].values
                        *
                        (
                            furnessed_frame["attraction_forecast"].values
                            /
                            furnessed_frame["attraction_zone_total"].values
                        )
                    )
                )
                / 2
            )

            # Calculate the total production of each zone
            production_zone_total = furnessed_frame[["p_zone", "dt"]]
            production_zone_total = production_zone_total.groupby(
                by=["p_zone"],
                as_index=False
            ).sum()
            production_zone_total.rename(
                columns={
                    "dt": "production_zone_total"
                },
                inplace=True
            )

            production_zone_total = pd.merge(
                production_zone_total,
                furnessed_frame[["p_zone", "production_forecast"]].drop_duplicates(
                    subset=["p_zone"]
                ),
                on="p_zone"
            )

            # Calculate production accuracy
            production_zone_total["production_zone_accuracy"] = (
                abs(
                    (
                        production_zone_total["production_forecast"].values
                        /
                        production_zone_total["production_zone_total"].values
                    )
                    -
                    1
                )
            )

            # Calculate the total attraction of each zone
            attraction_zone_total = furnessed_frame[["a_zone", "dt"]]
            attraction_zone_total = attraction_zone_total.groupby(
                by=["a_zone"],
                as_index=False
            ).sum()
            attraction_zone_total.rename(
                columns={"dt": "attraction_zone_total"},
                inplace=True
            )

            attraction_zone_total = pd.merge(
                attraction_zone_total,
                furnessed_frame[["a_zone", "attraction_forecast"]].drop_duplicates(
                    subset=["a_zone"]
                ),
                on="a_zone"
            )

            # Calculate attraction accuracy
            attraction_zone_total["attraction_zone_accuracy"] = (
                abs(
                    (
                        attraction_zone_total["attraction_forecast"].values
                        /
                        attraction_zone_total["attraction_zone_total"].values
                    )
                    -
                    1
                )
            )

            pa_acc = (
                1
                -
                (
                    (
                        production_zone_total["production_zone_accuracy"].mean()
                        +
                        attraction_zone_total["attraction_zone_accuracy"].mean()
                    )
                    /
                    2
                )
            )

            furnessed_frame = furnessed_frame.drop(
                columns=["production_zone_total", "attraction_zone_total"],
                axis=1
            )
            furnessed_frame = pd.merge(
                furnessed_frame,
                production_zone_total[["p_zone", "production_zone_total"]],
                on="p_zone"
            )
            furnessed_frame = pd.merge(
                furnessed_frame,
                attraction_zone_total[["a_zone", "attraction_zone_total"]],
                on="a_zone"
            )

            # TODO: Write a log of pa_acc achieved for each distribution
            #  Warn user when the target cannot be met
            # Log the best pa_acc achieved
            if pa_acc > best_pa_acc[0]:
                best_pa_acc = (pa_acc, i)
            else:
                # Exit early if we have been stuck here a while
                if i - best_pa_acc[1] > exit_early_tol:
                    print("WARNING: Couldn't reach target accuracy! Exiting "
                          "furnessing early...")
                    break

            # TODO: Turn this into a for loop
            i = i + 1

    # if we're performing checks to see drift of production vs trips
    if audit_outputs:
        # do this for each zone
        for zone in production_zones:
            # pre-distribution predictions
            zone_production = productions[
                productions["model_zone_id"] == zone
                ][
                "production_forecast"
            ].values[0]

            zone_attraction = attractions[
                attractions["model_zone_id"] == zone
                ][
                "attraction_forecast"
            ].values[0]

            # post-distribution predictions
            zone_from = furnessed_frame[
                furnessed_frame["p_zone"] == zone
            ].sum()["dt"]
            zone_to = furnessed_frame[
                furnessed_frame["a_zone"] == zone
            ].sum()["dt"]

            # difference between post and pre distribution predictions
            production_difference = zone_from - zone_production
            attraction_difference = zone_to - zone_attraction

            # print outputs
            # TODO: Offer option to build and save to csv
            print("Audit outputs for Zone " + str(zone) + ":")
            print("----------------------------------")
            print("Predicted Production from Zone " + str(zone) + ": " + str(zone_production))
            print("Trips from Zone " + str(zone) + ": " + str(zone_from))
            print("Difference is: " + str(production_difference))
            print("----------------------------------")
            print("Predicted Attraction to Zone " + str(zone) + ": " + str(zone_attraction))
            print("Trips to Zone " + str(zone) + ": " + str(zone_to))
            print("Difference is: " + str(attraction_difference))
            print("----------------------------------")
            print("")

    # return the completed furnessed frame
    return furnessed_frame[["p_zone", "a_zone", "dt"]]
