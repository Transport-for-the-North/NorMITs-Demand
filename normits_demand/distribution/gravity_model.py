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


class GravityModelCalibrator:
    # TODO(BT): Write GravityModelCalibrator docs

    _target_cost_distribution_cols = ['min', 'max', 'band_share']

    def __init__(self,
                 row_targets: np.ndarray,
                 col_targets: np.ndarray,
                 cost_function: cost.CostFunction,
                 costs: np.ndarray,
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
        self.costs = costs
        self.target_cost_distribution = target_cost_distribution
        self.furness_max_iters = furness_max_iters
        self.furness_tol = furness_tol
        self.running_log_path = running_log_path

        self.target_convergence = target_convergence

        # Running attributes
        self._loop_num = -1
        self._loop_start_time = None
        self._loop_end_time = None

        # Additional attributes
        self.optimal_cost_params = None
        self.achieved_band_share = None
        self.achieved_convergence = None
        self.achieved_distribution = None

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

    def _calculate_cost_distribution(self, matrix: np.ndarray) -> np.ndarray:
        """
        Calculates the band share distribution of matrix.

        Uses the bounds supplied in self.target_cost_distribution, and the costs in
        self.costs to calculate the equivalent band shares in matrix.

        Parameters
        ----------
        matrix:
            The matrix to calculate the cost distribution for. This matrix
            should be the same shape as self.costs

        Returns
        -------
        cost_distribution:
            a numpy array of distributed costs, where the bands are equivalent
            to min/max values in self.target_cost_distribution
        """
        # Init
        min_costs = self.target_cost_distribution['min']
        max_costs = self.target_cost_distribution['max']

        total_trips = matrix.sum()

        # Calculate band shares
        distribution = list()
        for min_val, max_val in zip(min_costs, max_costs):
            cost_mask = (self.costs >= min_val) & (self.costs < max_val)
            band_trips = (matrix * cost_mask).sum()
            band_share = band_trips / total_trips
            distribution.append(band_share)

        return np.array(distribution)

    def _gm_distribution(self, _, *cost_args: float) -> np.ndarray:
        """Runs gravity model with given parameters and returns distribution.

        Used by the `optimize.curve_fit` function.
        """
        # Convert the cost function args back into kwargs
        cost_kwargs = self._cost_params_to_kwargs(cost_args)

        # Run gravity model
        matrix, max_iters, r2 = gravity_model(
            row_targets=self.row_targets,
            col_targets=self.col_targets,
            cost_function=self.cost_function,
            costs=self.costs,
            furness_max_iters=self.furness_max_iters,
            furness_tol=self.furness_tol,
            **cost_kwargs,
        )

        # Convert matrix into an achieved distribution curve
        achieved_band_shares = self._calculate_cost_distribution(matrix)
        convergence = math_utils.curve_convergence(
            self.target_cost_distribution['band_share'].values,
            achieved_band_shares,
        )

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
            'furness_iters': max_iters,
            'furness_r2': np.round(r2, 6),
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
        self.achieved_distribution = matrix

        return achieved_band_shares

    def calibrate(self,
                  init_params: Dict[str, Any],
                  diff_step: float = None,
                  ftol: float = 1e-4,
                  max_iters: int = 100,
                  verbose: int = 0,
                  ):
        """Finds the optimal parameters for self.cost_function

        Optimal parameters are found using `scipy.optimize.curve_fit`
        to fit the distributed row/col targets to self.target_tld. Once
        the optimal parameters are found, the gravity model is run one last
        time to check the self.target_convergence has been met. This also
        populates a number of attributes with values from the optimal run:
        self.achieved_band_share
        self.achieved_convergence
        self.achieved_distribution

        Parameters
        ----------
        init_params:
            A dictionary of {parameter_name: parameter_value} to pass
            into the cost function as initial parameters.

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
            will stop once this tolerance has been met. 1e-4 By default,
            however this is far more precise than the convergence
            used in this code to evaluate results. 1e-4 should almost always
            get a band share convergence of >0.99

        max_iters:
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
        scipy.optimize.curve_fit
        scipy.optimize.least_squares
        """
        # Validate init_params
        self.cost_function.validate_params(init_params)

        # Initialise running params
        self._loop_num = 1
        self._loop_start_time = timing.current_milli_time()

        # Calculate the optimal cost parameters
        optimal_params, _ = optimize.curve_fit(
            self._gm_distribution,
            0,              # Doesn't matter what this is - it's ignored
            self.target_cost_distribution['band_share'].values,
            p0=self._order_init_params(init_params),
            bounds=self._order_bounds(),
            verbose=verbose,
            diff_step=diff_step,
            ftol=ftol,
            max_nfev=max_iters,
        )

        # Run an optimal version of the gravity
        self.optimal_cost_params = self._cost_params_to_kwargs(optimal_params)
        self._gm_distribution(0, *optimal_params)

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

    achieved_r2:
        The R-squared difference achieved by the doubly constarined furness
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
    matrix, iters, r2 = furness.doubly_constrained_furness(
        seed_vals=init_matrix,
        row_targets=row_targets,
        col_targets=col_targets,
        tol=furness_tol,
        max_iters=furness_max_iters,
    )

    return matrix, iters, r2

