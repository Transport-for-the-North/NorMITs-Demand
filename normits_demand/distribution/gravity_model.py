# -*- coding: utf-8 -*-
"""
Created on: 05/11/2021
Updated on:

Original author: Ben Taylor
Last update made by: Ben Taylor
Other updates made by:

File purpose:

"""
# Built-Ins
import os
import warnings

# Third Party
import numpy as np
import pandas as pd
from scipy import optimize

from typing import Any
from typing import List
from typing import Dict
from typing import Tuple

# Local Imports
import normits_demand as nd

from normits_demand import cost

from normits_demand.utils import timing
from normits_demand.utils import file_ops
from normits_demand.utils import math_utils
from normits_demand.utils import general as du
from normits_demand.utils import pandas_utils as pd_utils

from normits_demand.distribution import furness
from normits_demand.cost import utils as cost_utils


class GravityModelCalibrator:
    # TODO(BT): Write GravityModelCalibrator docs

    _avg_cost_col = 'ave_km'        # Should be more generic
    _target_cost_distribution_cols = ['min', 'max', 'trips'] + [_avg_cost_col]
    _least_squares_method = 'trf'

    # Cost amplification constants
    _cost_amplify_min_dist = np.inf     # Turn off for now
    _cost_amplify_power = 1

    def __init__(self,
                 row_targets: np.ndarray,
                 col_targets: np.ndarray,
                 cost_function: cost.CostFunction,
                 cost_matrix: np.ndarray,
                 target_cost_distribution: pd.DataFrame,
                 target_convergence: float,
                 furness_max_iters: int,
                 furness_tol: float,
                 running_log_path: nd.PathLike = None,
                 ):
        # TODO(BT): Write GravityModelCalibrator __init__ docs
        # Validate attributes
        target_cost_distribution = pd_utils.reindex_cols(
            target_cost_distribution,
            self._target_cost_distribution_cols,
        )

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

        # Set attributes
        self.row_targets = row_targets
        self.col_targets = col_targets
        self.cost_function = cost_function
        self.cost_matrix = cost_matrix
        self.target_cost_distribution = self._update_tcd(target_cost_distribution)
        self.tcd_bin_edges = self._get_tcd_bin_edges()
        self.furness_max_iters = furness_max_iters
        self.furness_tol = furness_tol
        self.running_log_path = running_log_path

        self.target_convergence = target_convergence

        # Running attributes
        self._loop_num = -1
        self._loop_start_time = None
        self._loop_end_time = None
        self._jacobian_mats = None

        # Additional attributes
        self.initial_cost_params = None
        self.initial_convergence = None
        self.optimal_cost_params = None
        self.achieved_band_share = None
        self.achieved_convergence = None
        self.achieved_residuals = None
        self.achieved_distribution = None

    @staticmethod
    def _update_tcd(tcd: pd.DataFrame) -> pd.DataFrame:
        """Extrapolates data where needed"""
        # Add in ave_km where needed
        tcd['ave_km'] = np.where(
            (tcd['ave_km'] == 0) | np.isnan(tcd['ave_km']),
            tcd['min'],
            tcd['ave_km'],
        )

        # Generate the band shares using the given data
        tcd['band_share'] = tcd['trips'].copy()
        tcd['band_share'] /= tcd['band_share'].values.sum()

        return tcd

    def _get_tcd_bin_edges(self) -> List[float]:
        min_bounds = self.target_cost_distribution['min'].tolist()
        max_bounds = self.target_cost_distribution['max'].tolist()
        return [min_bounds[0]] + max_bounds

    def _order_cost_params(self, params: Dict[str, Any]) -> List[Any]:
        """Order params into a list that self.cost_function expects"""
        ordered_params = [0] * len(self.cost_function.kw_order)
        for name, value in params.items():
            index = self.cost_function.kw_order.index(name)
            ordered_params[index] = value

        return ordered_params

    def _cost_params_to_kwargs(self, args: List[Any]) -> Dict[str, Any]:
        """Converts a list or args into kwargs that self.cost_function expects"""
        if len(args) != len(self.cost_function.kw_order):
            raise ValueError(
                "Received the wrong number of args to convert to cost function "
                "kwargs. Expected %s args, but got %s."
                % (len(self.cost_function.kw_order), len(args))
            )

        return {k: v for k, v in zip(self.cost_function.kw_order, args)}

    def _order_init_params(self, init_params: Dict[str, Any]) -> List[Any]:
        """Order init_params into a list that self.cost_function expects"""
        return self._order_cost_params(init_params)

    def _order_bounds(self) -> Tuple[List[Any], List[Any]]:
        """Order min and max into a tuple of lists that self.cost_function expects"""
        return(
            self._order_cost_params(self.cost_function.param_min),
            self._order_cost_params(self.cost_function.param_max),
        )

    def _cost_amplify(self, cost_matrix: np.ndarray) -> np.ndarray:
        if not (0 < self._cost_amplify_min_dist < np.max(cost_matrix)):
            return cost_matrix

        factor = (cost_matrix / self._cost_amplify_min_dist) ** self._cost_amplify_power
        new_cost = cost_matrix * factor
        return np.where(
            cost_matrix <= self._cost_amplify_min_dist,
            cost_matrix,
            new_cost,
        )

    def _cost_distribution(self, matrix: np.ndarray) -> np.ndarray:
        """Returns the distribution of matrix across self.tcd_bin_edges"""
        return cost_utils.calculate_cost_distribution(
            matrix=matrix,
            cost_matrix=self.cost_matrix,
            bin_edges=self.tcd_bin_edges,
        )

    def _guess_init_params(self, cost_args: List[float]):
        """Guesses what the initial params should be.

        Uses the average cost in each band to estimate what changes in
        the cost_params would do to the final cost distributions. This is a
        very coarse grained estimation, but can be used to guess around about
        where the best init params are.

        Used by the `optimize.least_squares` function.
        """
        # Convert the cost function args back into kwargs
        cost_kwargs = self._cost_params_to_kwargs(cost_args)

        # Used to optionally increase the cost of long distance trips
        avg_cost_vals = self.target_cost_distribution[self._avg_cost_col].values
        avg_cost_vals = self._cost_amplify(avg_cost_vals)

        # Estimate what the cost function will do to the costs - on average
        estimated_cost_vals = self.cost_function.calculate(avg_cost_vals, **cost_kwargs)
        estimated_band_shares = estimated_cost_vals / estimated_cost_vals.sum()

        # return the residuals to the target
        return self.target_cost_distribution['band_share'].values - estimated_band_shares

    def _gravity_function(self, cost_args: List[float], diff_step: float):
        """Returns residuals to target cost distribution

        Runs gravity model with given parameters and converts into achieved
        cost distribution. The residuals are then calculated between the
        achieved and the target.

        Used by the `optimize.least_squares` function.
        """
        # Convert the cost function args back into kwargs
        cost_kwargs = self._cost_params_to_kwargs(cost_args)

        # Used to optionally increase the cost of long distance trips
        cost_matrix = self._cost_amplify(self.cost_matrix)

        # Calculate initial matrix through cost function
        init_matrix = self.cost_function.calculate(cost_matrix, **cost_kwargs)

        # Do some prep for jacobian calculations
        self._jacobian_mats = {'base': init_matrix.copy()}
        for cost_param in self.cost_function.kw_order:
            # Adjust cost slightly
            adj_cost_kwargs = cost_kwargs.copy()
            adj_cost_kwargs[cost_param] += adj_cost_kwargs[cost_param] * diff_step

            # Calculate adjusted cost
            adj_cost = self.cost_function.calculate(cost_matrix, **adj_cost_kwargs)

            self._jacobian_mats[cost_param] = adj_cost

        # Furness trips to trip ends
        matrix, iters, rmse = furness.doubly_constrained_furness(
            seed_vals=init_matrix,
            row_targets=self.row_targets,
            col_targets=self.col_targets,
            tol=self.furness_tol,
            max_iters=self.furness_max_iters,
        )

        # Store for the jacobian calculations
        self._jacobian_mats['final'] = matrix.copy()

        # Convert matrix into an achieved distribution curve
        achieved_band_shares = self._cost_distribution(matrix)

        # Evaluate this run
        target_band_shares = self.target_cost_distribution['band_share'].values
        convergence = math_utils.curve_convergence(target_band_shares, achieved_band_shares)
        achieved_residuals = target_band_shares - achieved_band_shares

        # Calculate the time this loop took
        self._loop_end_time = timing.current_milli_time()
        time_taken = self._loop_end_time - self._loop_start_time

        # ## LOG THIS ITERATION ## #
        log_dict = {
            'loop_number': str(self._loop_num),
            'runtime (s)': time_taken / 1000,
        }
        log_dict.update(cost_kwargs)
        log_dict.update({
            'furness_iters': iters,
            'furness_rmse': np.round(rmse, 6),
            'bs_con': np.round(convergence, 4),
        })

        # Append this iteration to log file
        file_ops.safe_dataframe_to_csv(
                pd.DataFrame(log_dict, index=[0]),
                self.running_log_path,
                mode='a',
                header=(not os.path.exists(self.running_log_path)),
                index=False,
        )

        # Update loop params and return the achieved band shares
        self._loop_num += 1
        self._loop_start_time = timing.current_milli_time()
        self._loop_end_time = None

        # Update performance params
        self.achieved_band_share = achieved_band_shares
        self.achieved_convergence = convergence
        self.achieved_residuals = achieved_residuals
        self.achieved_distribution = matrix

        # Store the initial values to log later
        if self.initial_cost_params is None:
            self.initial_cost_params = cost_kwargs
        if self.initial_convergence is None:
            self.initial_convergence = convergence

        return achieved_residuals

    def _gravity_jacobian(self, cost_args: List[float], diff_step: float):
        """Returns the Jacobian for _gravity_function

        Uses the matrices stored in self._jacobian_mats (which were stored in
        the previous call to self._gravity function) to estimate what a change
        in the cost parameters would do to final furnessed matrix. This is
        then formatted into a Jacobian for optimize.least_squares to use.

        Used by the `optimize.least_squares` function.
        """
        # Initialise the output
        n_bands = len(self.target_cost_distribution['band_share'].values)
        n_cost_params = len(cost_args)
        jacobian = np.zeros((n_bands, n_cost_params))

        # Convert the cost function args back into kwargs
        cost_kwargs = self._cost_params_to_kwargs(cost_args)

        # Estimate what the furness does to the matrix
        furness_factor = np.divide(
            self._jacobian_mats['final'],
            self._jacobian_mats['base'],
            where=self._jacobian_mats['base'] != 0,
            out=np.zeros_like(self._jacobian_mats['base']),
        )

        # Calculate the Jacobian section for each cost param
        for i, cost_param in enumerate(self.cost_function.kw_order):
            # Estimate how the final matrix would be different with a
            # different input cost parameter
            furness_mat = self._jacobian_mats[cost_param] * furness_factor
            adj_weights = furness_mat / furness_mat.sum() if furness_mat.sum() != 0 else 0
            adj_final = self._jacobian_mats['final'].sum() * adj_weights

            # Control to final matrix
            adj_final, _, _ = furness.doubly_constrained_furness(
                seed_vals=adj_final,
                row_targets=self._jacobian_mats['final'].sum(axis=1),
                col_targets=self._jacobian_mats['final'].sum(axis=0),
                tol=1e-6,
                max_iters=20,
                warning=False,
            )

            # Turn into bands
            achieved_band_shares = self._cost_distribution(adj_final)

            # Calculate the Jacobian for this cost param
            # jacobian_residuals = self.achieved_band_share - achieved_band_shares
            jacobian_residuals = self.achieved_residuals - achieved_band_shares
            cost_step = cost_kwargs[cost_param] * diff_step
            cost_jacobian = jacobian_residuals / cost_step

            # Store in the Jacobian
            jacobian[:, i] = cost_jacobian

        return jacobian

    def calibrate(self,
                  init_params: Dict[str, Any],
                  estimate_init_params: bool = False,
                  calibrate_params: bool = True,
                  diff_step: float = 1e-8,
                  ftol: float = 1e-6,
                  xtol: float = 1e-6,
                  grav_max_iters: int = 100,
                  verbose: int = 0,
                  ):
        """Finds the optimal parameters for self.cost_function

        Optimal parameters are found using `scipy.optimize.least_squares`
        to fit the distributed row/col targets to self.target_tld. Once
        the optimal parameters are found, the gravity model is run one last
        time to check the self.target_convergence has been met. This also
        populates a number of attributes with values from the optimal run:
        self.achieved_band_share
        self.achieved_convergence
        self.achieved_residuals
        self.achieved_distribution

        Parameters
        ----------
        init_params:
            A dictionary of {parameter_name: parameter_value} to pass
            into the cost function as initial parameters.

        estimate_init_params:
            Whether to ignore the given init_params and estimate new ones
            using least squares, or just use the given init_params to start
            with.

        calibrate_params:
            Whether to calibrate the cost parameters or not. If not
            calibrating, the given init_params will be assumed to be
            optimal.

        diff_step:
            Copied from scipy.optimize.least_squares documentation, where it
            is passed to:
            Determines the relative step size for the finite difference
            approximation of the Jacobian. The actual step is computed as
            x * diff_step. If None (default), then diff_step is taken to be a
            conventional “optimal” power of machine epsilon for the finite
            difference scheme used

        ftol:
            The tolerance to pass to scipy.optimize.least_squares. The search
            will stop once this tolerance has been met. This is the
            tolerance for termination by the change of the cost function

        xtol:
            The tolerance to pass to scipy.optimize.least_squares. The search
            will stop once this tolerance has been met. This is the
            tolerance for termination by the change of the independent
            variables.

        grav_max_iters:
            The maximum number of calibration iterations to complete before
            termination if the ftol has not been met.

        verbose:
            Copied from scipy.optimize.least_squares documentation, where it
            is passed to:
            Level of algorithm’s verbosity:
            - 0 (default) : work silently.
            - 1 : display a termination report.
            - 2 : display progress during iterations (not supported by ‘lm’ method).

        Returns
        -------
        optimal_cost_params:
            Returns a dictionary of the same shape as init_params. The values
            will be the optimal cost parameters to get the best band share
            convergence.

        Raises
        ------
        ValueError
            If the generated trip matrix contains any
            non-finite values.

        See Also
        --------
        gravity_model
        scipy.optimize.least_squares
        """
        # Validate init_params
        self.cost_function.validate_params(init_params)

        # Estimate what the initial params should be
        if estimate_init_params:
            result = optimize.least_squares(
                self._guess_init_params,
                x0=self._order_init_params(init_params),
                method=self._least_squares_method,
                bounds=self._order_bounds(),
            )
            init_params = self._cost_params_to_kwargs(result.x)

            # TODO(BT): standardise this
            if self.cost_function.name == 'LOG_NORMAL':
                init_params['sigma'] *= 0.8
                init_params['mu'] *= 0.5

        # Initialise running params
        self._loop_num = 1
        self._loop_start_time = timing.current_milli_time()
        self.initial_cost_params = None
        self.initial_convergence = None

        # Calculate the optimal cost parameters if we're calibrating
        if calibrate_params is True:
            result = optimize.least_squares(
                fun=self._gravity_function,
                x0=self._order_init_params(init_params),
                method=self._least_squares_method,
                bounds=self._order_bounds(),
                jac=self._gravity_jacobian,
                verbose=verbose,
                ftol=ftol,
                xtol=xtol,
                max_nfev=grav_max_iters,
                kwargs={'diff_step': diff_step},
            )
            optimal_params = result.x
        else:
            optimal_params = self._order_init_params(init_params)

        # Run an optimal version of the gravity
        self.optimal_cost_params = self._cost_params_to_kwargs(optimal_params)
        self._gravity_function(optimal_params, diff_step=diff_step)

        # Check the performance of the best run
        if self.achieved_convergence < self.target_convergence:
            warnings.warn(
                "Calibration was not able to reach the target_convergence. "
                "Perhaps K-Factors are needed to improve the convergence?\n"
                "Target convergence: %s\n"
                "Achieved convergence: %s"
                % (self.target_convergence, self.achieved_convergence)
            )

        return self.optimal_cost_params


def gravity_model(row_targets: np.ndarray,
                  col_targets: np.ndarray,
                  cost_function: cost.CostFunction,
                  costs: np.ndarray,
                  furness_max_iters: int,
                  furness_tol: float,
                  **cost_params
                  ):
    """
    Runs a gravity model and returns the distributed row/col targets

    Uses the given cost function to generate an initial matrix which is
    used in a double constrained furness to distribute the row and column
    targets across a matrix. The cost_params can be used to achieve different
    results based on the cost function.

    Parameters
    ----------
    row_targets:
        The targets for the rows to sum to. These are usually Productions
        in Trip Ends.

    col_targets:
        The targets for the columns to sum to. These are usually Attractions
        in Trip Ends.

    cost_function:
        A cost function class defining how to calculate the seed matrix based
        on the given cost. cost_params will be passed directly into this
        function.

    costs:
        A matrix of the base costs to use. This will be passed into
        cost_function alongside cost_params. Usually this will need to be
        the same shape as (len(row_targets), len(col_targets)).

    furness_max_iters:
        The maximum number of iterations for the furness to complete before
        giving up and outputting what it has managed to achieve.

    furness_tol:
        The R2 difference to try and achieve between the row/col targets
        and the generated matrix. The smaller the tolerance the closer to the
        targets the return matrix will be.

    cost_params:
        Any additional parameters that should be passed through to the cost
        function.

    Returns
    -------
    distributed_matrix:
        A matrix of the row/col targets distributed into a matrix of shape
        (len(row_targets), len(col_targets))

    completed_iters:
        The number of iterations completed by the doubly constrained furness
        before exiting

    achieved_rmse:
        The Root Mean Squared Error achieved by the doubly constrained furness
        before exiting

    Raises
    ------
    TypeError:
        If some of the cost_params are not valid cost parameters, or not all
        cost parameters have been given.
    """
    # Validate additional arguments passed in
    equal, extra, missing = du.compare_sets(
        set(cost_params.keys()),
        set(cost_function.param_names),
    )

    if not equal:
        raise TypeError(
            "gravity_model() got one or more unexpected keyword arguments.\n"
            "Received the following extra arguments: %s\n"
            "While missing arguments: %s"
            % (extra, missing)
        )

    # Calculate initial matrix through cost function
    init_matrix = cost_function.calculate(costs, **cost_params)

    # Furness trips to trip ends
    matrix, iters, rmse = furness.doubly_constrained_furness(
        seed_vals=init_matrix,
        row_targets=row_targets,
        col_targets=col_targets,
        tol=furness_tol,
        max_iters=furness_max_iters,
    )

    return matrix, iters, rmse

