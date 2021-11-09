# -*- coding: utf-8 -*-
"""
Created on: 06/10/2021
Updated on:

Original author: Ben Taylor
Last update made by: Ben Taylor
Other updates made by: Chris Storey

File purpose:

"""
# Built-Ins
import os

from typing import Any
from typing import Dict
from typing import Optional

# Third Party
import numpy as np
import pandas as pd

# Local Imports
import normits_demand as nd
from normits_demand import constants

from normits_demand import cost

from normits_demand.utils import timing
from normits_demand.utils import file_ops
from normits_demand.utils import math_utils
from normits_demand.utils import costs as cost_utils
from normits_demand.utils import pandas_utils as pd_utils
from normits_demand.utils import trip_length_distributions as tld_utils

from normits_demand.distribution import gravity_model

from normits_demand.validation import checks
from normits_demand.concurrency import multiprocessing

from normits_demand.pathing.travel_market_synthesiser import GravityModelExportPaths

# BACKLOG: Re-write Gravity Model to use scipy.optimize.curve_fit
#  and numpy based furness
#  labels: TMS, optimisation


class GravityModel(GravityModelExportPaths):
    _log_fname = "Gravity_Model_log.log"

    _base_zone_col = "%s_zone_id"
    _pa_val_col = 'trips'

    _internal_only_suffix = 'int'

    def __init__(self,
                 year: int,
                 running_mode: nd.Mode,
                 zoning_system: nd.core.ZoningSystem,
                 export_home: nd.PathLike,
                 zone_col: str = None,
                 process_count: Optional[int] = constants.PROCESS_COUNT,
                 ):
        # Validate inputs
        if not isinstance(zoning_system, nd.core.zoning.ZoningSystem):
            raise ValueError(
                "Expected and instance of a normits_demand ZoningSystem. "
                "Got a %s instance instead."
                % type(zoning_system)
            )

        # Assign attributes
        self.zoning_system = zoning_system
        self.zone_col = zone_col
        self.process_count = process_count

        if self.zone_col is None:
            self.zone_col = zoning_system.col_name

        # Make sure the reports paths exists
        report_home = os.path.join(export_home, "Logs & Reports")
        file_ops.create_folder(report_home)

        # Build the output paths
        super().__init__(
            year=year,
            running_mode=running_mode,
            export_home=export_home,
        )

        # Create a logger
        logger_name = "%s.%s" % (nd.get_package_logger_name(), self.__class__.__name__)
        log_file_path = os.path.join(self.export_home, self._log_fname)
        self._logger = nd.get_logger(
            logger_name=logger_name,
            log_file_path=log_file_path,
            instantiate_msg="Initialised new Gravity Model Logger",
        )

    def run(self,
            trip_origin: str,
            running_segmentation: nd.core.segments.SegmentationLevel,
            productions: pd.DataFrame,
            attractions: pd.DataFrame,
            init_params: pd.DataFrame,
            target_tld_dir: pd.DataFrame,
            costs_path: nd.PathLike,
            cost_function: str,
            intrazonal_cost_infill: Optional[float] = 0.5,
            pa_val_col: Optional[str] = 'val',
            apply_k_factoring: bool = True,
            convergence_target: float = 0.95,
            fitting_loops: int = 100,
            furness_max_iters: int = 5000,
            furness_tol: float = 1.0,
            init_param_1_col: str = 'init_param_a',
            init_param_2_col: str = 'init_param_b',
            ):
        # Validate the trip origin
        trip_origin = checks.validate_trip_origin(trip_origin)

        # ## MULTIPROCESS ACROSS SEGMENTS ## #
        unchanging_kwargs = {
            'trip_origin': trip_origin,
            'running_segmentation': running_segmentation,
            'target_tld_dir': target_tld_dir,
            'costs_path': costs_path,
            'cost_function': cost_function,
            'intrazonal_cost_infill': intrazonal_cost_infill,
            'apply_k_factoring': apply_k_factoring,
            'furness_max_iters': furness_max_iters,
            'fitting_loops': fitting_loops,
            'convergence_target': convergence_target,
            'furness_tol': furness_tol,
        }

        pbar_kwargs = {
            'desc': 'Gravity model',
            'unit': 'segment',
        }

        # Build a list of kwargs
        kwarg_list = list()
        for segment_params in running_segmentation:
            # ## GET P/A VECTORS FOR THIS SEGMENT ## #
            # Figure out which columns we need
            segments = list(segment_params.keys())
            rename_cols = {pa_val_col: self._pa_val_col}
            needed_cols = segments + [self._pa_val_col]

            # Filter productions
            seg_productions = pd_utils.filter_df(
                df=productions,
                df_filter=segment_params,
                throw_error=True,
            )
            seg_productions = seg_productions.rename(columns=rename_cols)
            seg_productions = seg_productions.set_index(self.zone_col)
            seg_productions = seg_productions.reindex(
                index=self.zoning_system.unique_zones,
                columns=needed_cols,
                fill_value=0,
            ).reset_index()

            # Filter attractions
            seg_attractions = pd_utils.filter_df(
                df=attractions,
                df_filter=segment_params,
                throw_error=True,
            )
            seg_attractions = seg_attractions.rename(columns=rename_cols)
            seg_attractions = seg_attractions.set_index(self.zone_col)
            seg_attractions = seg_attractions.reindex(
                index=self.zoning_system.unique_zones,
                columns=needed_cols,
                fill_value=0,
            ).reset_index()

            # Check we actually got something
            production_sum = seg_productions[self._pa_val_col].values.sum()
            attraction_sum = seg_attractions[self._pa_val_col].values.sum()
            if production_sum <= 0 or attraction_sum <= 0:
                raise nd.NormitsDemandError(
                    "Missing productions and/or attractions after filtering to "
                    "this segment.\n"
                    "\tSegment: %s\n"
                    "\tProductions sum: %s\n"
                    "\tAttractions sum: %s"
                    % (segment_params, production_sum, attraction_sum)
                )

            # Balance A to P
            adj_factor = production_sum / attraction_sum
            seg_attractions[self._pa_val_col] *= adj_factor

            # ## GET INIT PARAMS FOP THIS SEGMENT ## #
            seg_init_params = pd_utils.filter_df(init_params, segment_params)

            if len(seg_init_params) > 1:
                seg_name = running_segmentation.generate_file_name(segment_params)
                raise ValueError(
                    "%s rows found in init_params for segment %s. "
                    "Expecting only 1 row."
                    % (len(seg_init_params), seg_name)
                )

            # Make sure the columns we need do exist
            seg_init_params = pd_utils.reindex_cols(
                df=seg_init_params,
                columns=[init_param_1_col, init_param_2_col],
                dataframe_name='init_params',
            )

            # Build the kwargs
            kwargs = unchanging_kwargs.copy()
            kwargs.update({
                'segment_params': segment_params,
                'seg_productions': seg_productions,
                'seg_attractions': seg_attractions,
                'init_param_a': seg_init_params[init_param_1_col].squeeze(),
                'init_param_b': seg_init_params[init_param_2_col].squeeze(),
            })
            kwarg_list.append(kwargs)

        # Multiprocess
        multiprocessing.multiprocess(
            fn=self._run_internal,
            kwargs=kwarg_list,
            pbar_kwargs=pbar_kwargs,
            process_count=0,
            # process_count=self.process_count,
        )

    def _run_internal(self,
                      segment_params: Dict[str, Any],
                      trip_origin: str,
                      running_segmentation: nd.core.segments.SegmentationLevel,
                      seg_productions: pd.DataFrame,
                      seg_attractions: pd.DataFrame,
                      init_param_a: float,
                      init_param_b: float,
                      target_tld_dir: pd.DataFrame,
                      costs_path: nd.PathLike,
                      cost_function: str,
                      intrazonal_cost_infill: Optional[float] = 0.5,
                      apply_k_factoring: bool = True,
                      convergence_target: float = 0.95,
                      fitting_loops: int = 100,
                      furness_max_iters: int = 5000,
                      furness_tol: float = 1.0,
                      ):
        seg_name = running_segmentation.generate_file_name(segment_params)
        self._logger.info("Running for %s" % seg_name)

        # ## READ IN TLD FOR THIS SEGMENT ## #
        target_tld = tld_utils.get_trip_length_distributions(
            import_dir=target_tld_dir,
            segment_params=segment_params,
            trip_origin=trip_origin,
        )

        # Convert to expected format
        rename = {'lower': 'min', 'upper': 'max'}
        target_tld = target_tld.rename(columns=rename)
        target_tld['min'] *= constants.MILES_TO_KM
        target_tld['max'] *= constants.MILES_TO_KM

        # ## GET THE COSTS FOR THIS SEGMENT ## #
        self._logger.debug("Getting costs from: %s" % costs_path)

        int_costs, cost_name = cost_utils.get_costs(
            costs_path,
            segment_params,
            iz_infill=intrazonal_cost_infill,
            replace_nhb_with_hb=(trip_origin == 'nhb'),
        )

        # Translate costs to wide - filter to only internal
        costs = pd_utils.long_to_wide_infill(
            df=int_costs,
            index_col='p_zone',
            columns_col='a_zone',
            values_col='cost',
            index_vals=self.zoning_system.unique_zones,
            column_vals=self.zoning_system.unique_zones,
            infill=0,
        )

        # ## SET UP LOG AND RUN ## #
        # Logging set up
        log_fname = running_segmentation.generate_file_name(
            trip_origin=trip_origin,
            file_desc='gravity_log',
            segment_params=segment_params,
            csv=True,
        )
        log_path = os.path.join(self.report_paths.model_log_dir, log_fname)

        # Need to convert into numpy vectors to work with old code
        seg_productions = seg_productions[self._pa_val_col].values
        seg_attractions = seg_attractions[self._pa_val_col].values
        costs = costs.values

        # Replace the log if it already exists
        if os.path.isfile(log_path):
            os.remove(log_path)

        calib = gravity_model.GravityModelCalibrator(
            row_targets=seg_productions,
            col_targets=seg_attractions,
            cost_function=cost.BuiltInCostFunction.LOG_NORMAL.get_cost_function(),
            costs=costs,
            target_cost_distribution=target_tld,
            target_convergence=convergence_target,
            furness_max_iters=furness_max_iters,
            furness_tol=furness_tol,
            running_log_path=log_path,
        )

        optimal_cost_params = calib.calibrate(
            init_params={'sigma': init_param_a, 'mu': init_param_b},
            max_iters=fitting_loops,
            verbose=2,
        )

        tld_report = target_tld.copy()
        tld_report = tld_report.rename(columns={'band_share': 'target_band_share'})
        tld_report['ach_band_share'] = calib.achieved_band_share
        tld_report['convergence'] = calib.achieved_convergence
        pa_mat = calib.achieved_distribution

        print(optimal_cost_params)
        print(tld_report)
        print(pa_mat)

        exit()

        internal_pa_mat, tld_report = calibrate_gravity_model(
            init_param_a=init_param_a,
            init_param_b=init_param_b,
            productions=seg_productions,
            attractions=seg_attractions,
            costs=costs,
            zones=self.zoning_system.unique_zones,
            target_tld=target_tld,
            log_path=log_path,
            cost_function=cost_function,
            apply_k_factoring=apply_k_factoring,
            furness_loops=furness_max_iters,
            fitting_loops=fitting_loops,
            bs_con_target=convergence_target,
            target_r_gap=furness_tol,
        )

        # ## WRITE OUT GRAVITY MODEL OUTPUTS ## #
        # Write out tld report
        fname = running_segmentation.generate_file_name(
            trip_origin=trip_origin,
            year=str(self.year),
            file_desc='tld_report',
            segment_params=segment_params,
            csv=True,
        )
        path = os.path.join(self.report_paths.tld_report_dir, fname)
        tld_report.to_csv(path, index=False)

        # ## WRITE DISTRIBUTED DEMAND ## #
        # Generate path and write out
        fname = running_segmentation.generate_file_name(
            trip_origin=trip_origin,
            year=str(self.year),
            file_desc='synthetic_pa',
            segment_params=segment_params,
            suffix=self._internal_only_suffix,
            compressed=True,
        )
        path = os.path.join(self.export_paths.distribution_dir, fname)
        nd.write_df(internal_pa_mat, path)


def calibrate_gravity_model(init_param_a: float,
                            init_param_b: float,
                            productions,
                            attractions,
                            costs,
                            zones,
                            target_tld,
                            log_path,
                            cost_function,
                            apply_k_factoring=True,
                            furness_loops=1999,
                            fitting_loops=100,
                            bs_con_target=.95,
                            target_r_gap=1
                            ):
    # ## VALIDATE INPUTS ## #
    # Make sure the costs and P/A are the same shape
    n_prod = len(productions)
    n_attr = len(attractions)
    n_zones = len(zones)
    if n_prod != n_zones:
        raise ValueError(
            "Productions are not the expected length based on given zones."
            "Got %s productions, expected %s."
            % (n_prod, len(zones))
        )

    if n_attr != n_zones:
        raise ValueError(
            "Attractions are not the expected length based on given zones."
            "Got %s attractions, expected %s."
            % (n_attr, len(zones))
        )

    if (n_zones, n_zones) != costs.shape:
        raise ValueError(
            "Costs are not the expected shape based on given zones. "
            "Got %s costs, expected %s."
            % (costs.shape, (n_zones, n_zones))
        )

    # ## Start of parameter search ## #
    min_dist, max_dist, obs_trip, obs_dist = unpack_tlb(target_tld)

    # Initial Search Loop - looking for OK values
    # Define criteria
    a_search, b_search, m_search, s_search, min_para, max_para = define_search_criteria(
        init_param_a,
        init_param_b,
        cost_function,
    )

    # Initialise, values that will be set in the loop
    max_r_sqr = [a_search[0], b_search[0], m_search[0], s_search[0], 0]
    k_factors = costs ** 0

    out_loop = 0
    out_para = list()
    for asv in a_search:
        for bsv in b_search:
            for msv in m_search:
                for ssv in s_search:
                    # Test we're running a sensible value
                    if param_check(min_para, max_para, asv, bsv, msv, ssv):
                        # Run gravity model
                        out_loop += 1
                        grav_run = old_gravity_model(
                            log_path=log_path,
                            target_tld=target_tld,
                            dist_function=cost_function,
                            par_data=[asv, bsv, msv, ssv],
                            min_para=min_para,
                            max_para=max_para,
                            bs_con_target=bs_con_target,
                            target_r_gap=target_r_gap,
                            furness_target=0.1,
                            productions=productions,
                            attractions=attractions,
                            costs=costs,
                            k_factors=k_factors,  # 1s
                            furness_loops=furness_loops,
                            fitting_loops=fitting_loops,
                            loop_number='1.' + str(out_loop),
                            optimise=True
                        )

                        if max_r_sqr[4] < grav_run[6][4]:
                            max_r_sqr = grav_run[6]
                            # This will pass an out para even if it's not doing a great job
                            # TODO: if it's not doing a good job, search more and better!

                        if max_r_sqr[4] > bs_con_target:
                            out_para, bs_con = grav_run[1], grav_run[6][4]

                        if (check_con_val(grav_run[3], target_r_gap) or
                                # Over 90
                                (grav_run[6][4] >= bs_con_target - .05)):
                            # Assign success values and leave loop - well done!
                            out_para, bs_con = grav_run[1], grav_run[6][4]
                            break

                    if len(out_para) != 0:
                        break
                if len(out_para) != 0:
                    break
            if len(out_para) != 0:
                break

    # We did real bad. Just use the last run and output something
    if len(out_para) == 0:
        out_para, bs_con = grav_run[1], grav_run[6][4]
    internal_pa = grav_run[0]

    # Refine values
    if len(list(set(out_para) - set(max_r_sqr))) > 0:
        # Restore best R-squared loop
        # Run gravity model
        # Set total runs to 1
        grav_run = old_gravity_model(
            log_path=log_path,
            target_tld=target_tld,
            dist_function=cost_function,
            par_data=max_r_sqr[0:4],
            min_para=min_para,
            max_para=max_para,
            bs_con_target=bs_con_target,
            target_r_gap=target_r_gap,
            furness_target=0.1,
            productions=productions,
            attractions=attractions,
            costs=costs,
            k_factors=k_factors,  # 1s
            furness_loops=furness_loops,
            fitting_loops=1,
            loop_number='2.0',
            optimise=True,
        )
        out_para, bs_con, max_r_sqr = grav_run[1], grav_run[5], grav_run[6]
        internal_pa = grav_run[0]

    if param_check(min_para, max_para,
                   max_r_sqr[0], max_r_sqr[1],
                   max_r_sqr[2], max_r_sqr[3]):

        internal_pa = grav_run[0]
        num_band = len(min_dist)

        est_trip, est_dist, cij_freq = [0] * num_band, [0] * num_band, [0] * num_band
        for row in range(num_band):
            # TODO(BT): Can this be replaced with a histogram function?
            est_trip[row] = np.sum(
                np.where((costs >= min_dist[row]) & (costs < max_dist[row]), internal_pa, 0))
            est_dist[row] = np.sum(
                np.where((costs >= min_dist[row]) & (costs < max_dist[row]), costs * internal_pa,
                         0))
            est_dist[row] = np.where(est_trip[row] > 0, est_dist[row] / est_trip[row],
                                     (min_dist[row] + max_dist[row]) / 2)
            obs_dist[row] = np.where(obs_dist[row] > 0, obs_dist[row], est_dist[row])
            est_trip[row] = est_trip[row] / np.sum(internal_pa) * 100
            cij_freq[row] = np.sum(
                np.where((costs >= min_dist[row]) & (costs < max_dist[row]), len(costs), 0))
            cij_freq[row] = cij_freq[row] / np.sum(len(costs)) * 100

        # mean trip length
        est_mean = np.sum(internal_pa * costs) / np.sum(internal_pa)
        est_logm = np.sum(internal_pa * np.log(np.where(costs > 0, costs, 1))) / np.sum(
            internal_pa)
        est_stdv = (np.sum(internal_pa * (costs - est_mean) ** 2) / np.sum(internal_pa)) ** 0.5

        # TODO(BT): Do the same as above, compare to the above results - REPORTING
        obs_mean, obs_logm, obs_stdv = 0, 0, 0

        # Auto-apply k-Factor
        kfc_dist, kfc_trip = [0] * num_band, [0] * num_band
        kfc_mean, kfc_logm, kfc_stdv, kfc_para, k_bs_con = est_mean, est_logm, est_stdv, out_para.copy(), bs_con
        if apply_k_factoring:
            out_loop = out_loop + 1
            k_factors = costs ** 0
            for row in range(num_band):
                kfc_dist[row] = np.where(est_trip[row] > 0,
                                         min(max(obs_trip[row] / est_trip[row], .2), 5), 1)
                k_factors = np.where((costs >= min_dist[row]) & (costs < max_dist[row]),
                                     kfc_dist[row], k_factors)
            grav_run = old_gravity_model(
                log_path=log_path,
                target_tld=target_tld,
                dist_function=cost_function,
                par_data=kfc_para,
                min_para=min_para,
                max_para=max_para,
                bs_con_target=bs_con_target,
                target_r_gap=target_r_gap,
                furness_target=0.1,
                productions=productions,
                attractions=attractions,
                costs=costs,
                k_factors=k_factors,
                furness_loops=furness_loops,
                fitting_loops=1,
                loop_number="3.0",
                optimise=True,
            )

            kfc_para, bs_con, k_r_sqr = grav_run[1], grav_run[5], grav_run[6]

            if param_check(min_para, max_para,
                           kfc_para[0], kfc_para[1],
                           kfc_para[2], kfc_para[3]):
                internal_pa = grav_run[0]

                # TODO(BT): Can this be replaced with a histogram function?
                for row in range(num_band):
                    kfc_trip[row] = np.sum(
                        np.where((costs >= min_dist[row]) & (costs < max_dist[row]), internal_pa,
                                 0))
                    kfc_dist[row] = np.sum(
                        np.where((costs >= min_dist[row]) & (costs < max_dist[row]),
                                 costs * internal_pa, 0))
                    kfc_dist[row] = np.where(kfc_trip[row] > 0, kfc_dist[row] / kfc_trip[row],
                                             (min_dist[row] + max_dist[row]) / 2)
                    kfc_trip[row] = kfc_trip[row] / np.sum(internal_pa) * 100
                kfc_mean = np.sum(internal_pa * costs) / np.sum(internal_pa)
                kfc_logm = np.sum(internal_pa * np.log(np.where(costs > 0, costs, 1))) / np.sum(
                    internal_pa)
                kfc_stdv = (np.sum(internal_pa * (costs - kfc_mean) ** 2) / np.sum(
                    internal_pa)) ** 0.5
    else:
        raise ValueError('Grav model netherworld - what did you do?')

    # ########## End of alpha/beta search ########## #

    # TODO: Add indices, back to pandas
    internal_pa = pd.DataFrame(
        internal_pa,
        index=zones,
        columns=zones,
    )

    # ## GENERATE A TLD REPORT ## #
    # Get distance into the right format
    distance = pd.DataFrame(
        data=costs,
        index=zones,
        columns=zones,
    )

    _, tld_report, _ = tld_utils.get_trip_length_by_band(
        band_atl=target_tld,
        distance=distance,
        internal_pa=internal_pa,
    )

    tld_report['bs_con'] = bs_con

    return internal_pa, tld_report


def old_gravity_model(log_path: nd.PathLike,
                  target_tld: pd.DataFrame,
                  dist_function: str,
                  par_data: list,
                  min_para: list,
                  max_para: list,
                  bs_con_target: float,
                  target_r_gap: float,
                  furness_target: float,
                  productions: np.ndarray,
                  attractions: np.ndarray,
                  costs,
                  k_factors,
                  furness_loops: int,
                  fitting_loops: int,
                  loop_number: str,
                  optimise=True):
    """
    Runs the outer loop of the gravity model, searching for the optimal
    alpha and beta values for trip distribution.

    Parameters
    ----------
    dist_log_path:
        Path to the folder that the log for this distribution should be output.

    dist_log_fname:
        The name of the file that the log for this distribution should be
        output. This is joined to dist_log_path. Calib_params will be used to
        personalise the name to this distribution.

    calib_params:
         Model calibration parameters for model.

    dist_function:
        distribution function, 'tanner' or 'ln'

    par_data:
        Input paramters as [alpha, beta, mu, sigma]

    bs_con_target:
        Target convergence on band share. Ie. r squared.

    target_r_gap:
        The ideal value for a good line fit

    productions:
        Productions as an np vector.

    attractions:
        Attractions as an np vector.

    cost:
        A matrix of cost of travel between zones.

    k_factors:
        K factors

    furness_loops:
        Number of inner loop iterations to do before abandoning.
        5000 should work for everything up to -0.5 beta - left higher.

    fitting_loops:
        Number of outer loop iterations to complete before abandoning.

    loop_number:
        String defining the name of this outer loop.

    verbose:
        Indicates whether to print a log of the process to the terminal.
        Useful to set verbose=False when using multi-threaded loops.

    optimise = True:
        Run the optimisation loop while searching or not

    Returns
    -------
    return_list:
        A list of calculated values in the following order:
        internal_pa:
            Distributed internal pa matrix for a given segment.

        run_log:
            A dict log of performance in each iterations.

        gm_loop_counter:
            A counter showing the number of loops completed

        trip_lengths:
            The achieved trip lengths

        band_share:
            The achieved band shares

        convergence_vals:
            A list of convergence values in the following order:
            [alpha_con, beta_con, bs_con, tl_con]

    """

    # Check input params
    assert dist_function.lower() in ['tanner', 'ln'], 'Not a valid function'

    # Build min max vectors
    # Convert miles from raw NTS to km
    # TODO(BT): Calculate Band share, total trip length, total average
    #  trip length in code
    min_dist, max_dist, obs_trip, obs_dist_o = unpack_tlb(target_tld)

    max_r_sqr, pre_data = [0, 0, 0, 0, 0], [0, 0, 0, 0]
    pre_val1, pre_val2 = 0, 0

    if dist_function.lower() == 'tanner':
        max_r_sqr[0], max_r_sqr[1] = par_data[0], par_data[1]
    else:
        max_r_sqr[2], max_r_sqr[3] = par_data[2], par_data[3]

    # Count bands
    num_band = len(target_tld)
    opt_loop = 0

    # Seed calibration factors
    est_trip, est_dist, obs_dist = [0] * num_band, [0] * num_band, [0] * num_band
    obs_dist, est_trip, est_dist = np.array(obs_dist), np.array(est_trip), np.array(est_dist)

    for ft_loop in range(fitting_loops):
        gm_start = timing.current_milli_time()

        if dist_function.lower() == 'tanner':  # x1, x2 - Tanner
            min_val1, min_val2 = min_para[0], min_para[1]
            max_val1, max_val2 = max_para[0], max_para[1]

        elif dist_function.lower() == 'ln':
            min_val1, min_val2 = min_para[2], min_para[3]
            max_val1, max_val2 = max_para[2], max_para[3]

        # Run furness process
        model_run = run_furness(furness_loops,
                                origin=productions,
                                destination=attractions,
                                par_data=par_data,
                                cost=costs,
                                k_factors=k_factors,
                                min_pa_diff=furness_target)

        gm_time_taken = timing.current_milli_time() - gm_start

        internal_pa, fn_loops, pa_diff = model_run
        del model_run

        # Get rid of any NaNs that might have snuck in
        internal_pa = np.nan_to_num(internal_pa)

        # TODO(BT): Can this be replaced with a histogram function?
        for i in range(num_band):
            # Get trips by band
            est_trip[i] = np.sum(
                np.where((costs >= min_dist[i]) & (costs < max_dist[i]), internal_pa, 0))
            # Get distance by band
            est_dist[i] = np.sum(
                np.where((costs >= min_dist[i]) & (costs < max_dist[i]), costs * internal_pa, 0))
            # Get mean distance by band
            est_dist[i] = np.where(est_trip[i] > 0, est_dist[i] / est_trip[i],
                                   (max_dist[i] + min_dist[i]) / 2)
            # Get observed distance by band
            obs_dist[i] = np.where(obs_dist_o[i] > 0, obs_dist_o[i], est_dist[i])

        # Control observed trips to PA volume
        obs_trip = obs_trip * np.sum(internal_pa) / np.sum(obs_trip)

        abs_diff = np.sum(np.abs(est_trip - obs_trip))
        obj_func = np.sum(est_trip - obs_trip * (
                np.where(est_trip > 0, np.log(
                    est_trip), 0) - np.where(obs_trip > 0, np.log(obs_trip), 0)))
        est_err = np.sum(est_trip * np.where(
            est_trip > 0, np.log(est_trip), 0) ** 2) ** 0.5

        # Figure out how to adjust the par_Data (alpha, beta etc...) for next iteration
        if dist_function.lower() == 'tanner':  # x1, x2 - Tanner
            cst_val1 = [np.where(obs_dist > 0, np.log(obs_dist), 0),
                        np.where(est_dist > 0, np.log(est_dist), 0)]
            cst_val2 = [obs_dist * 1, est_dist * 1]
            par_val1, par_val2 = par_data[0], par_data[1]
            fix_val1, fix_val2 = np.sum(obs_trip * cst_val1[0]), np.sum(obs_trip * cst_val2[0])
            cur_val1, cur_val2 = np.sum(est_trip * cst_val1[1]), np.sum(est_trip * cst_val2[1])
            gra_val1, gra_val2 = np.sum(est_trip * cst_val1[1] - obs_trip * cst_val1[0]), np.sum(
                est_trip * cst_val2[1] - obs_trip * cst_val2[0])

        elif dist_function.lower() == 'ln':  # mu, sigma - LogNormal f(Cij) = (1/(Cij*sigma*(2*np.pi)**0.5))*np.exp(-(np.log(Cij)-mu)**2/(2*sigma**2))
            cst_val1 = [np.where(obs_dist > 0, (-np.log(obs_dist) ** 2 / 2), 0),
                        np.where(est_dist > 0, (-np.log(est_dist) ** 2 / 2), 0)]  # mu
            cst_val2 = [
                np.where(obs_dist > 0, np.log(1 / (obs_dist * (2 * np.pi) ** 0.5)), 0) * cst_val1[
                    0],
                np.where(est_dist > 0, np.log(1 / (est_dist * (2 * np.pi) ** 0.5)), 0) * cst_val1[
                    1]]  # sigma
            par_val1, par_val2 = par_data[2], par_data[3]
            fix_val1, fix_val2 = np.sum(obs_trip * cst_val1[0]), np.sum(obs_trip * cst_val2[0])
            cur_val1, cur_val2 = np.sum(est_trip * cst_val1[1]), np.sum(est_trip * cst_val2[1])
            gra_val1, gra_val2 = np.sum(obs_trip * cst_val1[0] - est_trip * cst_val1[1]), np.sum(
                obs_trip * cst_val2[0] - est_trip * cst_val2[1])

        else:
            raise ValueError

        con_val1 = np.where(fix_val1 != 0, np.abs(gra_val1 / fix_val1) * 100, 100)
        con_val2 = np.where(fix_val2 != 0, np.abs(gra_val2 / fix_val2) * 100, 100)

        # ## Calculate our estimating parameters and convergence factors
        bs_con = max(1 - np.sum((est_trip - obs_trip) ** 2) / np.sum(
            (obs_trip[1:] - np.sum(obs_trip) / (len(obs_trip) - 1)) ** 2), 0)

        # Check if this coinvergence is better than previous best
        if bs_con > max_r_sqr[4]:
            if dist_function.lower() == 'tanner':
                max_r_sqr[0], max_r_sqr[1] = par_val1, par_val2
            else:
                max_r_sqr[2], max_r_sqr[3] = par_val1, par_val2
            max_r_sqr[4] = bs_con

        # Log this iteration
        log_dict = {'loop_number': str(loop_number),
                    'fit_loop': str(ft_loop),
                    'run_time': gm_time_taken,
                    'par_val1': np.round(par_val1, 6),
                    'par_val2': np.round(par_val2, 6),
                    'abs_diff': np.round(abs_diff, 3),
                    'obj_fun': np.round(obj_func, 3),
                    'est_error': np.round(est_err, 3),
                    'gra_val1': np.round(gra_val1, 3),
                    'gra_val2': np.round(gra_val2, 3),
                    'con_val1': np.round(con_val1, 6),
                    'con_val2': np.round(con_val2, 6),
                    'furness_loops': fn_loops,
                    'pa_diff': np.round(pa_diff, 6),
                    'bs_con': np.round(bs_con, 4)
                    }

        # Append this iteration to log file
        file_ops.safe_dataframe_to_csv(pd.DataFrame(log_dict, index=[0]),
                                       log_path,
                                       mode='a',
                                       header=(not os.path.exists(log_path)),
                                       index=False)

        # Break conditions
        if np.isnan(pa_diff):
            break
        elif con_val1 <= target_r_gap and con_val2 <= target_r_gap:
            break
        elif bs_con >= bs_con_target:
            break
        elif par_val1 < min_val1 or par_val1 > max_val1 or par_val2 < min_val2 or par_val2 > max_val2 or np.sum(
                internal_pa) == 0:
            break
        elif ft_loop == fitting_loops - 1:
            break
        else:
            par_temp = [0, 0, 0, 0]
            par_temp[0] = par_data[0] * (1 + min(max(gra_val1 / cur_val1, -0.5), 0.5))
            par_temp[1] = par_data[1] * (1 + min(max(gra_val2 / cur_val2, -0.5), 0.5))
            # sigma
            par_temp[2] = par_data[2] * (1 + min(max(gra_val1 / cur_val1, -0.5), 0.5))
            par_temp[3] = par_data[3] * (1 + min(max(gra_val2 / cur_val2, -0.5), 0.5))

            if optimise:
                opt_loop += 1
                if opt_loop == 25:
                    par_temp[0] = pre_data[0] + (0 - pre_val1) / (gra_val1 - pre_val1) * (
                                par_data[0] - pre_data[0])
                    par_temp[1] = pre_data[1] + (0 - pre_val2) / (gra_val2 - pre_val2) * (
                                par_data[1] - pre_data[1])
                    par_temp[2] = pre_data[2] + (0 - pre_val1) / (gra_val1 - pre_val1) * (
                                par_data[2] - pre_data[2])
                    par_temp[3] = pre_data[3] + (0 - pre_val2) / (gra_val2 - pre_val2) * (
                                par_data[3] - pre_data[3])

                    opt_loop = 0
                if opt_loop == 15:
                    pre_val1, pre_val2 = gra_val1, gra_val2
                    pre_data = par_data * 1
            par_data = par_temp

    return [
        internal_pa,
        par_data,
        [abs(gra_val1), abs(gra_val2)],
        [con_val1, con_val2],
        fn_loops,
        bs_con,
        max_r_sqr,
    ]


def run_furness(furness_loops,
                origin,
                destination,
                par_data,
                cost,
                k_factors,
                min_pa_diff=0.01):
    """
    Parameters
    ----------
    furness_loops:
        Number of loops to run furness for

    origin:
        Vector of origin trips, usually productions for hb.

    destination:
        Vector of destination trips, usually attractions for hb.

    par_data:
        list of parameters in order alpha, beta, mu, sigma

    cost:
        Matrix of cost for distribution.

    k_factors:
        Vector of k factors for optimisation.

    min_pa_diff:
        Acceptable level of furness convergence. Default =0.1

    Returns
    ----------
    mat_est:
        Estimated converged matrix
    fur_loop+1:
        Number of furness loops before convergence
    r_gap:
        Achieved r gap at furness end.
    """
    gravity = True

    # Unpack params
    alpha, beta, mu, sigma = par_data

    if alpha == 0 and beta == 0 and mu == 0 and sigma == 0:
        gravity = False

    # Tanner
    if gravity:
        mat_est = np.where(cost > 0,
                           # Tanner
                           (cost ** alpha) * np.exp(beta * cost) *
                           # Log normal
                           np.where(sigma > 0, (1 / (cost * sigma * (2 * np.pi) ** 0.5)) *
                                    np.exp(-(np.log(cost) - mu) ** 2 / (2 * sigma ** 2)), 1),
                           # K factor
                           0) * k_factors

    # Full furness
    for fur_loop in range(furness_loops):

        fur_loop += 1

        mat_d = np.sum(mat_est, axis=0)
        mat_d[mat_d == 0] = 1
        mat_est = mat_est * destination / mat_d

        mat_o = np.sum(mat_est, axis=1)
        mat_o[mat_o == 0] = 1
        mat_est = (mat_est.T * origin / mat_o).T

        # Get pa diff
        mat_o = np.sum(mat_est, axis=1)
        mat_d = np.sum(mat_est, axis=0)
        pa_diff = math_utils.get_pa_diff(mat_o,
                                  origin,
                                  mat_d,
                                  destination)  # .max()

        if pa_diff < min_pa_diff or np.isnan(np.sum(mat_est)):
            break

    return (mat_est,
            fur_loop + 1,
            pa_diff)


def define_search_criteria(init_param_a,
                           init_param_b,
                           dist_function):
    """
    Sets search criteria for search loop depending on initial value and
    fitting function.

    Parameters
    ----------
    init_param_a:
        First input param, will be alpha or mu.

    init_param_b:
        Second input param, will be beta or sigma.

    dist_function:
        Distribution function being used, should be 'tanner' or 'ln' for now.

    Returns:
    ----------
    [0] alpha_search:
        Vector of search terms for alpha.

    [1] beta_search:
        Vector of search terms for beta.

    [2] mu_search:
        Vector of search terms for mu.

    [3] sigma_search:
        Vector of search terms for param sigma.

    [4] min_para:
        Floor range of sensible terms for param a.

    [5] max_para:
        Ceiling range of sensible terms for param b.

    """
    # Assign blank lists in case nothing returns
    alpha_search, beta_search, mu_search, sigma_search = [[0]] * 4

    if dist_function == 'tanner':
        alpha_search_factors = [1, -1, .5, -.5, 2, -2]
        beta_search_factors = [1, -1]
        # Multiply search range by input param to get search params
        alpha_search = [x * init_param_a for x in alpha_search_factors]
        beta_search = [x * init_param_b for x in beta_search_factors]
        # Hard code min/max on what works
        min_para = [-5, -5, 0, 0]
        max_para = [5, 5, 0, 0]
    elif dist_function == 'ln':
        mu_search_factors = [1, .5, .2, 2, 5]
        sigma_search_factors = [1, .5]
        # Multiply search range by input param to get search params
        mu_search = [x * init_param_a for x in mu_search_factors]
        sigma_search = [x * init_param_b for x in sigma_search_factors]
        # Hard code min/max on what works
        min_para = [0, 0, 0, 0]
        max_para = [0, 0, 9, 3]
    else:
        raise ValueError(
            "Don't know what the dist function is %s"
            % dist_function
        )

    return (
        alpha_search,
        beta_search,
        mu_search,
        sigma_search,
        min_para,
        max_para,
    )


def param_check(min_para,
                max_para,
                alpha=0,
                beta=0,
                mu=0,
                sig=0):
    """
    Checks that distribution params are within given range
    """
    return (
        min_para[0] <= alpha <= max_para[0]
        and min_para[1] <= beta <= max_para[1]
        and min_para[2] <= mu <= max_para[2]
        and min_para[3] <= sig <= max_para[3]
    )


def check_con_val(con_vals, target_r_gap):
    """
    Check convergence values are within acceptable range

    con_vals:
        Convergence values.
    target_r_gap:
        Acceptable gap
    """

    val_1 = False
    val_2 = False

    if 0 < con_vals[0] < max(10, target_r_gap):
        val_1 = True
    if 0 < con_vals[1] < max(10, target_r_gap):
        val_2 = True

    if val_1 and val_2:
        return True
    else:
        return False


def unpack_tlb(tlb):
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
    _M_KM = 1.61

    # Convert miles from raw NTS to km
    min_dist = tlb['lower'].astype('float').to_numpy() * _M_KM
    max_dist = tlb['upper'].astype('float').to_numpy() * _M_KM
    obs_trip = tlb['band_share'].astype('float').to_numpy()
    # TODO: Check that this works!!
    obs_dist = tlb['ave_km'].astype(float).to_numpy()

    return min_dist, max_dist, obs_trip, obs_dist

