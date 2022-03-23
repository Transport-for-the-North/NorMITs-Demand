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
import operator
import warnings

import pandas as pd
import numpy as np
from numpy.testing import assert_approx_equal

from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Callable

# self imports
import normits_demand as nd
from normits_demand import constants as consts

from normits_demand.utils import timing
from normits_demand.utils import file_ops
from normits_demand.utils import math_utils
from normits_demand.utils import general as du
from normits_demand.utils import pandas_utils as pd_utils

from normits_demand.concurrency import multiprocessing
from normits_demand.audits import audits
from normits_demand.cost import utils as cost_utils


class Furness3D:
    # TODO(BT): Write Furness3D docs

    _target_cost_distribution_cols = ['min', 'max', 'ave_km', 'band_share']

    def __init__(self,
                 row_targets: np.ndarray,
                 col_targets: np.ndarray,
                 cost_matrix: np.ndarray,
                 base_matrix: np.ndarray,
                 calibration_matrix: np.ndarray,
                 target_cost_distributions: Dict[Any, pd.DataFrame],
                 calibration_naming: Dict[Any, Any],
                 target_convergence: float,
                 furness_max_iters: int,
                 furness_tol: float,
                 calibration_ignore_val: Any = -1,
                 running_log_path: nd.PathLike = None,
                 ):

        # TODO(BT): Write Furness3D __init__ docs
        # Validate attributes
        for key, tcd in target_cost_distributions.items():
            tcd = pd_utils.reindex_cols(tcd, self._target_cost_distribution_cols)
            target_cost_distributions[key] = tcd

        if running_log_path is not None:
            dir_name, _ = os.path.split(running_log_path)
            if not os.path.exists(dir_name):
                raise FileNotFoundError(
                    "Cannot find the defined directory to write out a"
                    "log. Given the following path: %s"
                    % dir_name
                )

            if os.path.isfile(running_log_path):
                warnings.warn(
                    "Given a log path to a file that already exists. Logs "
                    "will be appended to the end of the file at: %s"
                    % running_log_path
                )

        # Get a list of keys for calibration zones
        calib_keys = np.unique(calibration_matrix).tolist()
        calib_keys = du.list_safe_remove(calib_keys, [calibration_ignore_val])

        # Set attributes
        self.row_targets = row_targets
        self.col_targets = col_targets
        self.cost_matrix = cost_matrix
        self.base_matrix = base_matrix
        self.calibration_keys = calib_keys
        self.calibration_matrix = calibration_matrix
        self.target_cost_distributions = target_cost_distributions
        self.calibration_naming = calibration_naming
        self.calibration_ignore_val = calibration_ignore_val
        self.furness_max_iters = furness_max_iters
        self.furness_tol = furness_tol
        self.running_log_path = running_log_path

        self.target_convergence = target_convergence

        # Additional attributes
        self.initial_convergences = None
        self.achieved_band_shares = None
        self.achieved_convergences = None
        self.achieved_distribution = None

    def _correct_band_share(self,
                            matrix: np.ndarray,
                            ) -> np.ndarray:
        """Correct matrix to move band shares towards the target"""
        # Init
        out_matrix = np.zeros_like(matrix)

        # Just copy over where there are ignore values
        ignore_mask = (self.calibration_matrix == self.calibration_ignore_val)
        ignore_values = matrix * ignore_mask
        out_matrix += ignore_values

        # Adjust band shares by calibration areas
        for calib_key in self.calibration_keys:
            # Filter down to this area
            area_mask = (self.calibration_matrix == calib_key)
            area_tcd = self.target_cost_distributions[calib_key]

            area_matrix_values = matrix * area_mask
            area_total = area_matrix_values.sum()
            area_cost = self.cost_matrix * area_mask

            # Adjust bands one at a time
            for _, row in area_tcd.iterrows():
                # Get proportion of all trips that should be in this band
                target_band_share = row['band_share']
                min_val = float(row['min'])
                max_val = float(row['max'])

                # Get proportion of all trips that are in this band
                distance_mask = (area_cost >= min_val) & (area_cost < max_val)
                band_trips = area_matrix_values * distance_mask

                # Figure out the target and achieved
                target_band_total = area_total * target_band_share
                ach_band_total = band_trips.sum()

                # We can't adjust if there are no trips in this band
                if ach_band_total <= 0:
                    adj_mat = band_trips

                elif target_band_total <= 0:
                    # Set to a really small value so furness can use still
                    adj_mat = np.where(band_trips != 0, 1e-7, 0)

                else:
                    # Adjust the matrix towards target
                    adjustment = target_band_total / ach_band_total
                    adj_mat = band_trips * adjustment

                # Add into the return matrix
                out_matrix += adj_mat

        return out_matrix

    def fit(self,
            outer_max_iters: int = 100,
            calibrate: bool = True,
            ):
        # Make sure we always do at least 1 loop!
        if outer_max_iters < 1 or not calibrate:
            outer_max_iters = 1

        # Seed base
        matrix = self.base_matrix.copy()

        # Perform a 3D furness
        for iter_num in range(outer_max_iters):
            iter_start_time = timing.current_milli_time()

            # Push band shares towards targets
            matrix = self._correct_band_share(matrix=matrix)

            # Furness across the other 2 dimensions
            matrix, furn_iters, furn_rmse = doubly_constrained_furness(
                seed_vals=matrix,
                row_targets=self.row_targets,
                col_targets=self.col_targets,
                tol=self.furness_tol,
                max_iters=self.furness_max_iters,
            )

            # ## EVALUATE AND LOG THIS ITERATION ## #
            # Initialise the log
            iter_end_time = timing.current_milli_time()
            time_taken = iter_end_time - iter_start_time

            log_dict = {
                'Loop Num': str(iter_num),
                'run_time(s)': time_taken / 1000,
                'furness_loops': furn_iters,
                'furness_rmse': furn_rmse,
            }

            # Evaluate per calibration area
            area_convergence_dict = dict()
            area_band_share_dict = dict()
            for calib_key in self.calibration_keys:
                # Filter down to this area
                area_mask = (self.calibration_matrix == calib_key)
                area_tcd = self.target_cost_distributions[calib_key]

                area_matrix_values = matrix * area_mask
                area_cost = self.cost_matrix * area_mask

                # Calculate the convergence of this area
                achieved_band_shares = cost_utils.calculate_cost_distribution(
                    matrix=area_matrix_values,
                    cost_matrix=area_cost,
                    min_bounds=area_tcd['min'].tolist(),
                    max_bounds=area_tcd['max'].tolist(),
                )
                area_convergence = math_utils.curve_convergence(
                    area_tcd['band_share'].values,
                    achieved_band_shares,
                )

                # Store for recording runs
                area_convergence_dict[calib_key] = area_convergence
                area_band_share_dict[calib_key] = achieved_band_shares

                # Add to log
                area_name = '%s_bs_con' % self.calibration_naming[calib_key]
                log_dict.update({area_name: np.round(area_convergence, 6)})

            # Append this iteration to log file
            file_ops.safe_dataframe_to_csv(
                pd.DataFrame(log_dict, index=[0]),
                self.running_log_path,
                mode='a',
                header=(not os.path.exists(self.running_log_path)),
                index=False,
            )

            # log initial convergence if first iter
            if iter_num == 0:
                self.initial_convergences = area_convergence_dict

            # Log current best performance
            self.achieved_band_shares = area_band_share_dict
            self.achieved_convergences = area_convergence_dict
            self.achieved_distribution = matrix

            # ## EXIT EARLY IF CONDITIONS MET ## #
            if all([x > self.target_convergence for x in area_convergence_dict.values()]):
                break


def doubly_constrained_furness(seed_vals: np.array,
                               row_targets: np.array,
                               col_targets: np.array,
                               tol: float = 1e-9,
                               max_iters: int = 5000,
                               warning: bool = True,
                               ) -> Tuple[np.array, int, float]:
    """
    Performs a doubly constrained furness for max_iters or until tol is met

    Controls numpy warnings to warn of any overflow errors encountered

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

    warning:
        Whether to print a warning or not when the tol cannot be met before
        max_iters.

    Returns
    -------
    furnessed_matrix:
        The final furnessed matrix

    completed_iters:
        The number of completed iterations before exiting

    achieved_rmse:
        The Root Mean Squared Error difference achieved before exiting
    """
    # Error check
    if seed_vals.shape != (len(row_targets), len(col_targets)):
        raise ValueError(
            "The shape of the seed values given does not match the row "
            "and col targets. Seed_vals are shape %s. Expected shape (%d, %d)."
            % (str(seed_vals.shape), len(row_targets), len(col_targets))
        )

    # Init
    furnessed_mat = seed_vals.copy()
    early_exit = False
    cur_rmse = np.inf
    iter_num = 0
    n_vals = len(row_targets)

    # Can return early if all 0 - probably shouldn't happen!
    if row_targets.sum() == 0 or col_targets.sum() == 0:
        warnings.warn("Furness given targets of 0. Returning all 0's")
        return np.zeros(seed_vals.shape), iter_num, cur_rmse

    # Set up numpy overflow errors
    with np.errstate(over='raise'):

        for iter_num in range(max_iters):
            # ## COL CONSTRAIN ## #
            # Calculate difference factor
            col_ach = np.sum(furnessed_mat, axis=0)
            diff_factor = np.divide(
                col_targets,
                col_ach,
                where=col_ach != 0,
                out=np.ones_like(col_targets, dtype=float),
            )

            # adjust cols
            furnessed_mat *=
                diff_factor

            # ## ROW CONSTRAIN ## #
            # Calculate difference factor
            row_ach = np.sum(furnessed_mat, axis=1)
            diff_factor = np.divide(
                row_targets,
                row_ach,
                where=row_ach != 0,
                out=np.ones_like(row_targets, dtype=float),
            )

            # adjust rows
            furnessed_mat *=
                np.atleast_2d(diff_factor).T

            # Calculate the diff - leave early if met
            row_diff = (row_targets - np.sum(furnessed_mat, axis=1)) ** 2
            col_diff = (col_targets - np.sum(furnessed_mat, axis=0)) ** 2
            cur_rmse = (np.sum(row_diff + col_diff) / n_vals) ** 0.5
            if cur_rmse < tol:
                early_exit = True
                break

            # We got a NaN! Make sure to point out we didn't converge
            if np.isnan(cur_rmse):
                return np.zeros(furnessed_mat.shape), iter_num, np.inf

    # Warn the user if we exhausted our number of loops
    if not early_exit and warning:
        print("WARNING! The doubly constrained furness exhausted its max "
              "number of loops (%d), while achieving an RMSE difference of "
              "%f. The values returned may not be accurate."
              % (max_iters, cur_rmse))

    return furnessed_mat, iter_num + 1, cur_rmse


def _distribute_pa_internal(productions,
                            attraction_weights,
                            seed_year,
                            seed_dist_dir,
                            trip_origin,
                            calib_params,
                            unique_zones,
                            unique_zones_join_fn,
                            zone_col,
                            p_col,
                            m_col,
                            seg_col,
                            ca_col,
                            tp_col,
                            max_iters,
                            seed_infill,
                            normalise_seeds,
                            furness_tol,
                            seed_mat_format,
                            echo,
                            report_out,
                            dist_out,
                            round_dp,
                            fname_suffix,
                            csv_out,
                            compress_out,
                            ) -> Dict[str, Any]:
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
        suffix=fname_suffix,
        csv=csv_out,
        compressed=compress_out,
    )
    print("Furnessing %s ..." % out_dist_name)

    # Build seg_params for the seed values
    seed_seg_params = calib_params.copy()
    seed_seg_params['yr'] = seed_year

    # Read in the seed distribution - ignoring year
    seed_fname = du.calib_params_to_dist_name(
        trip_origin=trip_origin,
        matrix_format=seed_mat_format,
        calib_params=seed_seg_params,
        csv=True
    )
    seed_dist = pd.read_csv(os.path.join(seed_dist_dir, seed_fname), index_col=0)
    seed_dist.columns = seed_dist.columns.astype(int)

    # Pull the seed matrix into line with unique zones
    if unique_zones is not None:
        # Get the mask and extract the data
        mask = pd_utils.get_wide_mask(
            df=seed_dist,
            zones=unique_zones,
            join_fn=unique_zones_join_fn,
        )
        seed_dist = seed_dist.where(mask, 0)

    # Quick check that seed is valid
    if len(seed_dist.columns.difference(seed_dist.index)) > 0:
        raise ValueError(
            "The index and columns of the seed distribution"
            "'%s' do not match." % seed_fname
        )

    # Assume seed contains all the zones numbers
    seed_zones = list(seed_dist.index)

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

    productions = du.filter_df(productions, df_filter=base_filter, fit=True)

    a_weights = du.filter_df(a_weights, df_filter=base_filter, fit=True)

    # Rename columns for furness
    year = calib_params['yr']
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
        unique_zones=seed_zones
    )

    # ## BALANCE P/A FORECASTS ## #
    if productions[unique_col].sum() != a_weights[unique_col].sum():
        du.print_w_toggle("Row and Column targets do not match. Balancing...", verbose=echo)
        bal_fac = productions[unique_col].sum() / a_weights[unique_col].sum()
        a_weights[unique_col] *= bal_fac

    pa_dist, n_iters, achieved_rmse = furness_pandas_wrapper(
        row_targets=productions,
        col_targets=a_weights,
        seed_values=seed_dist,
        max_iters=max_iters,
        seed_infill=seed_infill,
        normalise_seeds=normalise_seeds,
        idx_col=zone_col,
        unique_col=unique_col,
        tol=furness_tol,
        round_dp=round_dp,
        unique_zones=unique_zones,
        unique_zones_join_fn=unique_zones_join_fn,
    )

    # Build a report of the furness
    report = {
        'name': out_dist_name,
        'iterations': n_iters,
        'furness_RMSE': achieved_rmse,
        'tolerance': furness_tol,
    }

    if report_out is not None:
        # Create output filename
        audit_fname = out_dist_name.replace('_pa_', '_dist_audit_')
        audit_fname = audit_fname.replace(consts.COMPRESSION_SUFFIX, '.csv')
        audit_path = os.path.join(report_out, audit_fname)

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
    file_ops.write_df(pa_dist, output_path)

    return report


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
                  unique_zones: List[int] = None,
                  unique_zones_join_fn: Callable = operator.and_,
                  zone_col: str = 'model_zone_id',
                  p_col: str = 'p',
                  m_col: str = 'm',
                  soc_col: str = 'soc',
                  ns_col: str = 'ns',
                  ca_col: str = 'ca',
                  tp_col: str = 'tp',
                  trip_origin: str = 'hb',
                  seed_year: str = None,
                  max_iters: int = 5000,
                  seed_infill: float = 1e-5,
                  normalise_seeds: bool = True,
                  furness_tol: float = 1e-2,
                  seed_mat_format: str = 'pa',
                  fname_suffix: str = None,
                  csv_out: bool = True,
                  compress_out: bool = True,
                  echo: bool = False,
                  report_out: str = None,
                  round_dp: int = consts.DEFAULT_ROUNDING,
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

    unique_zones:
        A list of unique zones to keep in the seed matrix when starting the
        furness. The given productions and attractions will also be limited
        to these zones as well.

    unique_zones_join_fn:
        The function to call on the column and index masks to join them for
        the seed matrices. By default, a bitwise and is used. See pythons
        builtin operator library for more options.

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

    seed_year:
        The base year of the seed matrices. If None, then no year is looked
        for in the filenames of the seed matrices.

    max_iters
        The maximum number of iterations to complete within the furness process
        before exiting, if a solution has not been found a message will be
        printed to the terminal.

    seed_infill:
        The value to infill any seed values that are 0.

    normalise_seeds:
        Whether to normalise the seeds so they total to one before
        sending them to the furness.

    furness_tol:
        The maximum difference between the achieved and the target values
        to tolerate before exiting the furness early. R^2 is used to calculate
        the difference.

    seed_mat_format:
        The format of the seed matrices.

    fname_suffix:
        Any additional suffix to add to the filename when writing out to disk.
        Will be added at the end of the filename, before the ftype suffix.

    csv_out:
        Whether to write the matrices out as csv or not. IF both this and
        compress_out are True, compress_out is ignored.

    compress_out:
        Whether to write the matrices out as a compressed file or not.
        If both this and csv_out are True, this argument is ignored.

    echo:
        Controls the amount of printing to the terminal. If False, most of the
        print outs will be ignored.

    report_out:
        Path to a directory to output all reports.
        
    round_dp:
        The number of decimal places to round the output values of the
        furness to. Uses 4 by default.

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
    attraction_weights = du.ensure_segmentation(attraction_weights, **kwargs)

    # Get P/A columns
    p_cols = list(productions.columns)
    for year in years_needed:
        p_cols.remove(year)

    a_cols = list(attraction_weights.columns)
    for year in years_needed:
        a_cols.remove(year)

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
            'seed_year': seed_year,
            'seed_dist_dir': seed_dist_dir,
            'trip_origin': trip_origin,
            'unique_zones': unique_zones,
            'unique_zones_join_fn': unique_zones_join_fn,
            'zone_col': zone_col,
            'p_col': p_col,
            'm_col': m_col,
            'ca_col': ca_col,
            'tp_col': tp_col,
            'max_iters': max_iters,
            'seed_infill': seed_infill,
            'normalise_seeds': normalise_seeds,
            'furness_tol': furness_tol,
            'seed_mat_format': seed_mat_format,
            'echo': echo,
            'report_out': report_out,
            'dist_out': dist_out,
            'round_dp': round_dp,
            'fname_suffix': fname_suffix,
            'csv_out': csv_out,
            'compress_out': compress_out,
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

        reports = multiprocessing.multiprocess(
            fn=_distribute_pa_internal,
            kwargs=kwargs_list,
            process_count=process_count
        )

        # Finally - create aan audit summary
        if report_out is not None:
            audits.summarise_audit_furness(
                report_out,
                trip_origin=trip_origin,
                format_name='dist_audit',
                year=year,
                p_needed=p_needed,
                m_needed=m_needed,
                soc_needed=soc_needed,
                ns_needed=ns_needed,
                ca_needed=ca_needed,
                tp_needed=tp_needed,
                fname_suffix=fname_suffix,
            )

            # Write out the furness stats for the year
            fname = '%s_%s_furness_stats.csv' % (trip_origin, year)
            out_path = os.path.join(report_out, fname)
            pd.DataFrame(reports).to_csv(out_path, index=False)


def furness_pandas_wrapper(seed_values: pd.DataFrame,
                           row_targets: pd.DataFrame,
                           col_targets: pd.DataFrame,
                           max_iters: int = 2000,
                           seed_infill: float = 1e-3,
                           normalise_seeds: bool = True,
                           tol: float = 1e-9,
                           idx_col: str = 'model_zone_id',
                           unique_col: str = 'trips',
                           round_dp: int = consts.DEFAULT_ROUNDING,
                           unique_zones: List[int] = None,
                           unique_zones_join_fn: Callable = operator.and_,
                           ) -> Tuple[pd.DataFrame, int, float]:
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

    normalise_seeds:
        Whether to normalise the seeds so they total to one before
        sending them to the furness.

    idx_col:
        Name of the columns in row_targets and col_targets that contain the
        index data that matches seed_values index/columns

    unique_col:
        Name of the columns in row_targets and col_targets that contain the
        values to target during the furness.

    round_dp:
        The number of decimal places to round the output values of the
        furness to. Uses 4 by default.

    unique_zones:
        A list of unique zones to keep in the seed matrix when starting the
        furness. The given productions and attractions will also be limited
        to these zones as well.

    unique_zones_join_fn:
        The function to call on the column and index masks to join them for
        the seed matrices. By default, a bitwise and is used. See pythons
        builtin operator library for more options.

    Returns
    -------
    furnessed_matrix:
        The final furnessed matrix, in the same format as seed_values

    completed_iters:
        The number of completed iterations before exiting

    achieved_rmse:
        The Root Mean Squared Error difference achieved before exiting
    """
    # Init
    row_targets = row_targets.copy()
    col_targets = col_targets.copy()
    seed_values = seed_values.copy()

    row_targets = row_targets.reindex(columns=[idx_col, unique_col])
    col_targets = col_targets.reindex(columns=[idx_col, unique_col])
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

    # ## TIDY AND INFILL SEED ## #
    # Infill the 0 zones
    seed_values = seed_values.mask(seed_values <= 0, seed_infill)
    if normalise_seeds:
        seed_values /= seed_values.sum()

    # If we were given certain zones, make sure everything else is 0
    if unique_zones is not None:
        # Get the mask and extract the data
        mask = pd_utils.get_wide_mask(
            df=seed_values,
            zones=unique_zones,
            join_fn=unique_zones_join_fn,
        )
        seed_values = seed_values.where(mask, 0)

    # ## CONVERT TO NUMPY AND FURNESS ## #
    row_targets = row_targets.values.flatten()
    col_targets = col_targets.values.flatten()
    seed_values = seed_values.values

    furnessed_mat, n_iters, achieved_rmse = doubly_constrained_furness(
        seed_vals=seed_values,
        row_targets=row_targets,
        col_targets=col_targets,
        tol=tol,
        max_iters=max_iters
    )

    furnessed_mat = np.round(furnessed_mat, round_dp)

    # ## STICK BACK INTO PANDAS ## #
    furnessed_mat = pd.DataFrame(
        index=ref_index,
        columns=ref_index,
        data=furnessed_mat
    ).round(round_dp)

    return furnessed_mat, n_iters, achieved_rmse
