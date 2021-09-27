# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 11:02:39 2020

@author: cruella
"""
import os
import time

import numpy as np
import pandas as pd

from normits_demand.utils import utils as nup  # Folder management, reindexing, optimisation
from normits_demand.reports import reports_audits as ra

from normits_demand.utils import pandas_utils as pd_utils

# TODO: Where should this live?
_default_rounding = 3


# TODO: More error handling
# TODO: object orient

def run_gravity_model(zone_col,
                      segment_params: dict,
                      init_param_a: float,
                      init_param_b: float,
                      productions,
                      attractions,
                      internal_zones,
                      model_lookup_path,
                      target_tld,
                      dist_log_path,
                      dist_log_fname,
                      dist_function='tanner',
                      cost_type='24hr',
                      apply_k_factoring=True,
                      furness_loops=1999,
                      fitting_loops=100,
                      bs_con_target=.95,
                      target_r_gap=1,
                      rounding_factor=3,
                      iz_cost_infill=0.5,
                      production_val_col: str = 'val',
                      attraction_val_col: str = 'val',
                      verbose=True):
    """
    Function that filters down productions and attractions to deal only with a
    specific segment and then calls the gravity model to model 24hr PA trips.

    Parameters
    ----------
    ia_name:
        Name of the zone in the internal area.

    calib_params:
        Model calibration parameters for model.

    init_param_a:
        Scalar for power function or mu if ln.

    init_param_b:
        Scalar for exponential function or sigma if ln.

    productions:
        Productions for internal area with full segmentation.
        Will be filtered down.

    attractions:
        Attractions for internal area with full segmentation.
        Will be filtered down.

    model_lookup_path:
        Path to 'Model Zone Lookups' folder containing distance and costs.

    dist_log_path:
        See gravity model

    dist_log_fname:
        See gravity model

    dist_function:
        Function to use for distribution. Tanner or log normal.

    cost_type:
        String defining the type of cost being used.

    furness_loops:
        See gravity model
        
    fitting_loops:
        See gravity model

    bs_con_roof:
        See gravity model

    bs_con_floor:
        See gravity model

    alpha_con_target:
        See gravity model

    beta_con_target:
        See gravity model

    rounding_factor:
        Not currently used. Deprecated?

    iz_cost_infill:
        Factor for how to deal with intrazonal cost infills.

    rounding_factor = 3:
        Number of decimal places to round by. Defines how precise calibration
        will be. 3 is industry standard.

    verbose:
        Indicates whether to print a log of the process to the terminal.
        Useful to set verbose=False when using multi-threaded loops.
        
    verbose_outer_loop_updates:
        See gravity model

    Returns
    ----------
    output_row:
        Description of attained distribution parameters

    internal_pa:
        Distributed internal pa matrix for a given segment.

    d_bin:
        Trip length bin by mile - for distribution histograms.

    new_beta:
        beta to do another distribution with.
    """
    # Filter P/A Vectors
    productions = pd_utils.filter_df(productions, segment_params)
    attractions = pd_utils.filter_df(attractions, segment_params)

    # Balance A to P
    adj_factor = productions[production_val_col].sum() / attractions[attraction_val_col].sum()
    attractions[attraction_val_col] *= adj_factor

    # Productions as numpy
    p = nup.df_to_np(
        productions,
        v_heading=zone_col,
        values=production_val_col,
        unq_internal_zones=internal_zones,
    )
    p = p.astype(np.float64, copy=True)

    # Attractions as numpy
    a = nup.df_to_np(
        attractions,
        v_heading=zone_col,
        values=attraction_val_col,
        unq_internal_zones=internal_zones,
    )
    a = a.astype(np.float64, copy=True)

    # TODO: Pass car ownership params for costs
    # Import costs based on distribution parameters & car availability
    nup.print_w_toggle('Importing costs', verbose=verbose)
    internal_costs = nup.get_costs(model_lookup_path,
                                   segment_params,
                                   tp=cost_type,
                                   iz_infill=iz_cost_infill)
    nup.print_w_toggle('Cost lookup returned ' + internal_costs[1], verbose=verbose)
    internal_costs = internal_costs[0].copy()

    # Translate costs to array
    cost = nup.df_to_np(
        internal_costs,
        v_heading='p_zone',
        h_heading='a_zone',
        values='cost',
        unq_internal_zones=internal_zones,
    )
    cost = cost.astype(np.float64, copy=True)

    # Seed k-factors with 1s for first runs
    k_factors = cost ** 0

    min_dist, max_dist, obs_trip, obs_dist = nup.unpack_tlb(target_tld)

    ### Start of parameter search ###

    # Initial Search Loop - looking for OK values
    # Define criteria
    a_search, b_search, m_search, s_search, min_para, max_para = define_search_criteria(
        init_param_a,
        init_param_b,
        dist_function,
    )

    # Initialise, so something will run if all else fails
    max_r_sqr = [a_search[0], b_search[0], m_search[0], s_search[0], 0]

    out_loop = 0
    out_para = list()
    for asv in a_search:
        for bsv in b_search:
            for msv in m_search:
                for ssv in s_search:
                    print('New search')
                    # Test we're running a sensible value
                    if param_check(min_para, max_para,
                                   asv, bsv, msv, ssv):
                        # Run gravity model
                        out_loop += 1
                        print("Running for loop gravity model")
                        grav_run = gravity_model(
                            dist_log_path=dist_log_path,
                            dist_log_fname=dist_log_fname,
                            calib_params=segment_params,
                            target_tld=target_tld,
                            dist_function=dist_function,
                            par_data=[asv, bsv, msv, ssv],
                            min_para=min_para,
                            max_para=max_para,
                            bs_con_target=bs_con_target,
                            target_r_gap=target_r_gap,
                            furness_target=0.1,
                            productions=p,
                            attractions=a,
                            cost=cost,
                            k_factors=k_factors,  # 1s
                            furness_loops=furness_loops,
                            fitting_loops=fitting_loops,
                            loop_number='1.' + str(out_loop),
                            verbose=verbose,
                            optimise=True
                        )

                        # Check convergence criteria
                        print('achieved bs_con: %s' % grav_run[6][4])
                        print('achieved params: %s' % grav_run[6])
                        print()
                        print('prev best bs_con: %s' % max_r_sqr[4])

                        if max_r_sqr[4] < grav_run[6][4]:
                            print('This is better')
                            max_r_sqr = grav_run[6]
                            # This will pass an out para even if it's not doing a great job
                            # TODO: if it's not doing a good job, search more and better!
                            out_para, bs_con = grav_run[1], grav_run[6][4]
                        if (check_con_val(grav_run[3], target_r_gap) or
                                # Over 90
                                (grav_run[5] >= bs_con_target - .05)):
                            # Assign success values and leave loop - well done!
                            out_para, bs_con = grav_run[1], grav_run[6][4]
                            break

                    else:
                        print("Parameters outside of min max range")

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
    print("Length of out_para:", len(out_para))
    if len(list(set(out_para) - set(max_r_sqr))) > 0:
        # Restore best R-squared loop
        out_loop = out_loop + 1
        # Run gravity model
        # Set total runs to 1
        print("Running len(out_para) != 0 gravity model")
        grav_run = gravity_model(
            dist_log_path=dist_log_path,
            dist_log_fname=dist_log_fname,
            calib_params=segment_params,
            target_tld=target_tld,
            dist_function=dist_function,
            par_data=max_r_sqr[0:4],
            min_para=min_para,
            max_para=max_para,
            bs_con_target=bs_con_target,
            target_r_gap=target_r_gap,
            furness_target=0.1,
            productions=p,
            attractions=a,
            cost=cost,
            k_factors=k_factors,  # 1s
            furness_loops=furness_loops,
            fitting_loops=1,
            loop_number=str(out_loop),
            verbose=verbose,
            optimise=True)
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
            est_trip[row] = np.sum(np.where((cost >= min_dist[row]) & (cost < max_dist[row]), internal_pa, 0))
            est_dist[row] = np.sum(
                np.where((cost >= min_dist[row]) & (cost < max_dist[row]), cost * internal_pa, 0))
            est_dist[row] = np.where(est_trip[row] > 0, est_dist[row] / est_trip[row],
                                     (min_dist[row] + max_dist[row]) / 2)
            obs_dist[row] = np.where(obs_dist[row] > 0, obs_dist[row], est_dist[row])
            est_trip[row] = est_trip[row] / np.sum(internal_pa) * 100
            cij_freq[row] = np.sum(np.where((cost >= min_dist[row]) & (cost < max_dist[row]), len(cost), 0))
            cij_freq[row] = cij_freq[row] / np.sum(len(cost)) * 100

        # mean trip length
        est_mean = np.sum(internal_pa * cost) / np.sum(internal_pa)
        est_logm = np.sum(internal_pa * np.log(np.where(cost > 0, cost, 1))) / np.sum(internal_pa)
        est_stdv = (np.sum(internal_pa * (cost - est_mean) ** 2) / np.sum(internal_pa)) ** 0.5

        # TODO(BT): Do the same as above, compare to the above results - REPORTING
        obs_mean, obs_logm, obs_stdv = 0, 0, 0

        # Auto-apply k-Factor
        kfc_dist, kfc_trip = [0] * num_band, [0] * num_band
        kfc_mean, kfc_logm, kfc_stdv, kfc_para, k_bs_con = est_mean, est_logm, est_stdv, out_para.copy(), bs_con
        if apply_k_factoring:
            out_loop = out_loop + 1
            k_factors = cost ** 0
            for row in range(num_band):
                kfc_dist[row] = np.where(est_trip[row] > 0, min(max(obs_trip[row] / est_trip[row], .2), 5), 1)
                k_factors = np.where((cost >= min_dist[row]) & (cost < max_dist[row]), kfc_dist[row], k_factors)
            print("Running third gravity model")
            grav_run = gravity_model(
                dist_log_path=dist_log_path,
                dist_log_fname=dist_log_fname,
                calib_params=segment_params,
                target_tld=target_tld,
                dist_function=dist_function,
                par_data=kfc_para,
                min_para=min_para,
                max_para=max_para,
                bs_con_target=bs_con_target,
                target_r_gap=target_r_gap,
                furness_target=0.1,
                productions=p,
                attractions=a,
                cost=cost,
                k_factors=k_factors,
                furness_loops=furness_loops,
                fitting_loops=1,
                loop_number=str(out_loop + 1),
                verbose=verbose,
                optimise=True)

            kfc_para, bs_con, k_r_sqr = grav_run[1], grav_run[5], grav_run[6]

            if param_check(min_para, max_para,
                           kfc_para[0], kfc_para[1],
                           kfc_para[2], kfc_para[3]):
                internal_pa = grav_run[0]

                # TODO(BT): Can this be replaced with a histogram function?
                for row in range(num_band):
                    kfc_trip[row] = np.sum(
                        np.where((cost >= min_dist[row]) & (cost < max_dist[row]), internal_pa, 0))
                    kfc_dist[row] = np.sum(
                        np.where((cost >= min_dist[row]) & (cost < max_dist[row]), cost * internal_pa, 0))
                    kfc_dist[row] = np.where(kfc_trip[row] > 0, kfc_dist[row] / kfc_trip[row],
                                             (min_dist[row] + max_dist[row]) / 2)
                    kfc_trip[row] = kfc_trip[row] / np.sum(internal_pa) * 100
                kfc_mean = np.sum(internal_pa * cost) / np.sum(internal_pa)
                kfc_logm = np.sum(internal_pa * np.log(np.where(cost > 0, cost, 1))) / np.sum(internal_pa)
                kfc_stdv = (np.sum(internal_pa * (cost - kfc_mean) ** 2) / np.sum(internal_pa)) ** 0.5
    else:
        print('Grav model netherworld - what did you do?')

    # ########## End of alpha/beta search ########## #

    # TODO: Add indices, back to pandas
    internal_pa = pd.DataFrame(
        internal_pa,
        index=internal_zones,
        columns=internal_zones,
    )

    # ## GENERATE A TLD REPORT ## #
    # Get distance into the right format
    distance = pd.DataFrame(
        data=cost,
        index=internal_zones,
        columns=internal_zones,
    )

    _, d_bin, _ = ra.get_trip_length_by_band(
        band_atl=target_tld,
        distance=distance,
        internal_pa=internal_pa,
    )

    return internal_pa, d_bin


def gravity_model(dist_log_path: str,
                  dist_log_fname: str,
                  calib_params: dict,
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
                  cost,
                  k_factors,
                  furness_loops: int,
                  fitting_loops: int,
                  loop_number: str,
                  verbose: bool = True,
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

    # Create the output path
    dist_log_path = os.path.join(dist_log_path, dist_log_fname)
    dist_log_path = nup.build_path(dist_log_path, calib_params)

    # Replace the log if it already exists
    if os.path.isfile(dist_log_path):
        os.remove(dist_log_path)

    # Build min max vectors
    # Convert miles from raw NTS to km
    # TODO(BT): Calculate Band share, total trip length, total average
    #  trip length in code
    min_dist, max_dist, obs_trip, obs_dist_o = nup.unpack_tlb(target_tld)

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
        print('fit loop ' + str(ft_loop))
        gm_start = time.time()

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
                                cost=cost,
                                k_factors=k_factors,
                                min_pa_diff=furness_target)

        gm_time_taken = time.time() - gm_start

        internal_pa, fn_loops, pa_diff = model_run
        del model_run

        # Get rid of any NaNs that might have snuck in
        internal_pa = np.nan_to_num(internal_pa)

        # TODO(BT): Can this be replaced with a histogram function?
        for i in range(num_band):
            # Get trips by band
            est_trip[i] = np.sum(np.where((cost >= min_dist[i]) & (cost < max_dist[i]), internal_pa, 0))
            # Get distance by band
            est_dist[i] = np.sum(np.where((cost >= min_dist[i]) & (cost < max_dist[i]), cost * internal_pa, 0))
            # Get mean distance by band
            est_dist[i] = np.where(est_trip[i] > 0, est_dist[i] / est_trip[i], (max_dist[i] + min_dist[i]) / 2)
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
            cst_val1 = [np.where(obs_dist > 0, np.log(obs_dist), 0), np.where(est_dist > 0, np.log(est_dist), 0)]
            cst_val2 = [obs_dist * 1, est_dist * 1]
            par_val1, par_val2 = par_data[0], par_data[1]
            fix_val1, fix_val2 = np.sum(obs_trip * cst_val1[0]), np.sum(obs_trip * cst_val2[0])
            cur_val1, cur_val2 = np.sum(est_trip * cst_val1[1]), np.sum(est_trip * cst_val2[1])
            gra_val1, gra_val2 = np.sum(est_trip * cst_val1[1] - obs_trip * cst_val1[0]), np.sum(
                est_trip * cst_val2[1] - obs_trip * cst_val2[0])

        elif dist_function.lower() == 'ln':  # mu, sigma - LogNormal f(Cij) = (1/(Cij*sigma*(2*np.pi)**0.5))*np.exp(-(np.log(Cij)-mu)**2/(2*sigma**2))
            cst_val1 = [np.where(obs_dist > 0, (-np.log(obs_dist) ** 2 / 2), 0),
                        np.where(est_dist > 0, (-np.log(est_dist) ** 2 / 2), 0)]  # mu
            cst_val2 = [np.where(obs_dist > 0, np.log(1 / (obs_dist * (2 * np.pi) ** 0.5)), 0) * cst_val1[0],
                        np.where(est_dist > 0, np.log(1 / (est_dist * (2 * np.pi) ** 0.5)), 0) * cst_val1[1]]  # sigma
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

        print('Achieved Rsqr: ' + str(bs_con))
        print('Achieved PA diff: ' + str(round(pa_diff, 4)))

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
        nup.safe_dataframe_to_csv(pd.DataFrame(log_dict, index=[0]),
                                  dist_log_path,
                                  mode='a',
                                  header=(not os.path.exists(dist_log_path)),
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
                    par_temp[0] = pre_data[0] + (0 - pre_val1) / (gra_val1 - pre_val1) * (par_data[0] - pre_data[0])
                    par_temp[1] = pre_data[1] + (0 - pre_val2) / (gra_val2 - pre_val2) * (par_data[1] - pre_data[1])
                    par_temp[2] = pre_data[2] + (0 - pre_val1) / (gra_val1 - pre_val1) * (par_data[2] - pre_data[2])
                    par_temp[3] = pre_data[3] + (0 - pre_val2) / (gra_val2 - pre_val2) * (par_data[3] - pre_data[3])

                    opt_loop = 0
                if opt_loop == 15:
                    pre_val1, pre_val2 = gra_val1, gra_val2
                    pre_data = par_data * 1
            print("par_temp",par_temp)
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
        pa_diff = nup.get_pa_diff(mat_o,
                                  origin,
                                  mat_d,
                                  destination)  # .max()

        if pa_diff < min_pa_diff or np.isnan(np.sum(mat_est)):
            break

    return (mat_est,
            fur_loop + 1,
            pa_diff)


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

    t = (cost ** alpha) * np.exp(beta * cost)
    dt = balance * constraint * t

    # Log normal
    # Normal start values: mu ~ 5 sigma ~ 2
    # 1/(Cij*sigma*(2pi)**0.5)*exp(-nlog(Cij)-mu)**2/2*(2/sigma**2)
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

    beta:
        Beta to use in the function.

    Returns
    ----------
    dt = Distributed trips for a given interzonal.
    """
    t = (cost ** alpha) * np.exp(beta * cost)
    dt = p * ba * a * bb * t
    return dt


def dt_to_factors(pa, dt_type='new_ba'):
    """
    This function calculates the new pa values of a given distribution.
    It rounds the distributed trips to a given value to allow convergence at
    a lower level than 64bit float.

    Parameters
    ----------
    pa:
        pa matrix

    dt_type:
        which balancing factors have been changed and need to be summed.
        Takes 'new_ba' ie. 'balancing factor a' or
        'new_bb' ie. 'balancing factor b'

    Returns:
    ----------
    [0] new:
        PA matrix with dt reduced to factors for calculation.

    [1] new_col:
        Column name of the new balancing factors.

    [2] zone_col:
        Zone type of the new balancing factors.
    """
    # TODO: Errors if conditions aren't met
    if dt_type == 'new_ba':
        zone_col = 'p_zone'
        new_col = 'ba'
    elif dt_type == 'new_bb':
        zone_col = 'a_zone'
        new_col = 'bb'

    new = pa.reindex([zone_col, 'dt'],
                     axis=1).groupby(zone_col).sum().reset_index()
    # Seed in >0 to avoid div0
    new['dt'] = new['dt'].replace(0, 0.0001)
    new['dt'] = 1 / new['dt']
    new = new.rename(columns={'dt': new_col})

    return (new, new_col, zone_col)


def apply_new_dt(pa, new, new_col, zone_col):
    """
    This function adds new balancing factors in to a matrix. They are returned
    in the dt col and added to whichever col comes through in zone_col
    parameter.

    Parameters
    ----------
    pa:
        Pa matrix.

    new:
        new balancing factors.

    new_col:
        column to replace with new balancing factors.

    zone_col:
        Zone column to join new balancing factors on.

    Returns:
    ----------
    pa:
        PA matrix with new balancing factors added in.
    """

    pa = pa.drop([new_col, 'dt'], axis=1)
    pa = pa.merge(new, how='inner', on=zone_col)

    return (pa)


def get_new_pa(pa_dt, rounding=_default_rounding):
    """
    This function calculates the new pa values of a given distribution.
    It rounds the distributed trips to a given value to allow convergence at
    a lower level than 64bit float.

    Parameters
    ----------
    pa_dt:
        distributed trips

    rounding:
        number of decimal places to round to in distribution comparisons

    Returns:
    ----------
    [0] worked_p:
        total number of productions in new distributed matrix

    [1] worked_a:
        total number of attractions in new distributed matrix
    """
    worked_p = pa_dt.reindex(['p_zone', 'dt'],
                             axis=1).groupby('p_zone').sum().reset_index()
    worked_p = worked_p.rename(columns={'dt': 'p'})
    worked_p['p'] = worked_p['p'].round(rounding)
    worked_p = nup.optimise_data_types(worked_p, verbose=False)
    worked_a = pa_dt.reindex(['a_zone', 'dt'],
                             axis=1).groupby('a_zone').sum().reset_index()
    worked_a = worked_a.rename(columns={'dt': 'a'})
    worked_a['a'] = worked_a['a'].round(rounding)
    worked_a = nup.optimise_data_types(worked_a, verbose=False)

    return (worked_p, worked_a)


def check_new_pa(worked_p,
                 worked_a,
                 distribution_p,
                 distribution_a,
                 rounding=_default_rounding):
    """
    Checks a 24hr PA distribution against the total number
    of productions and attractions for a given internal area.

    Parameters
    ----------
    worked_p:
        total distributed productions

    worked_a:
        total distributed attractions

    distribution_p:
        target distributed productions

    distribution_a:
        target distributed attractions

    Returns:
    ----------
    [0] true_p:
        number of worked producitons which match target

    [1] true_a:
        number of worked attractions which match target
    """
    # Reset indices
    distribution_p = distribution_p.reset_index(drop=True)
    distribution_a = distribution_a.reset_index(drop=True)

    # Change column names for comparison
    distribution_p.columns = list(worked_p)
    distribution_a.columns = list(worked_a)

    # Round target PA
    distribution_p['p'] = distribution_p['p'].round(1)
    distribution_a['a'] = distribution_a['a'].round(1)

    # Round worked PA
    worked_p['p'] = worked_p['p'].round(1)
    worked_a['a'] = worked_a['a'].round(1)

    true_p = distribution_p == worked_p
    true_p = true_p.reindex(['p'], axis=1)
    true_a = distribution_a == worked_a
    true_a = true_a.reindex(['a'], axis=1)

    return (true_p, true_a)


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

    return (alpha_search,
            beta_search,
            mu_search,
            sigma_search,
            min_para,
            max_para)


def param_check(min_para,
                max_para,
                alpha=0,
                beta=0,
                mu=0,
                sig=0):
    """
    Checks that distribution params are within given range
    """
    check = (alpha >= min_para[0] and
             alpha <= max_para[0] and
             beta >= min_para[1] and
             beta <= max_para[1] and
             mu >= min_para[2] and
             mu <= max_para[2] and
             sig >= min_para[3] and
             sig <= max_para[3])
    print(check)
    return check


def check_con_val(con_vals,
                  target_r_gap):
    """
    Check convergence values are within acceptable range
    
    con_vals:
        Convergence values.
    target_r_gap:
        Acceptable gap
    """

    val_1 = False
    val_2 = False

    if con_vals[0] > 0 and con_vals[0] < max(10, target_r_gap):
        val_1 = True
    if con_vals[1] > 0 and con_vals[1] < max(10, target_r_gap):
        val_2 = True

    if val_1 and val_2:
        return True
    else:
        return False
