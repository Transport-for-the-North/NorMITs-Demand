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
                 max_iters: int,
                 furness_max_iters: int,
                 furness_tol: float,
                 ):
        # TODO(BT): Write GravityModelCalibrator __init__ docs
        # Validate attributes
        target_cost_distribution = pd_utils.reindex_cols(
            target_cost_distribution,
            self._target_cost_distribution_cols,
        )

        # Set attributes
        self.row_targets = row_targets
        self.col_targets = col_targets
        self.cost_function = cost_function
        self.costs = costs
        self.target_cost_distribution = target_cost_distribution
        self.target_convergence = target_convergence
        self.max_iters = max_iters
        self.furness_max_iters = furness_max_iters
        self.furness_tol = furness_tol

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
        from normits_demand.utils import math_utils
        convergence = math_utils.curve_convergence(
            self.target_cost_distribution['band_share'].values,
            achieved_band_shares,
        )
        print(achieved_band_shares)
        print(self.target_cost_distribution['band_share'].values)
        print(convergence)
        exit()

        # TODO(BT): Write out GM Logs here!? Convergence?!

        return achieved_band_shares

    def calibrate(self,
                  init_params: Dict[str, Any],
                  diff_step: float = None,
                  verbose: int = 0,
                  ):
        """Finds the optimal parameters for self.cost_function

        Optimal parameters are found using `scipy.optimize.curve_fit`
        to fit the distributed row/col targets to self.target_tld.

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

        verbose:
            Copied from scipy.optimize.least_squares documentation, where it
            is passed to:
            Level of algorithm’s verbosity:
            - 0 (default) : work silently.
            - 1 : display a termination report.
            - 2 : display progress during iterations (not supported by ‘lm’ method).

        Returns
        -------
        # TODO(BT): Determine the returns!!

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
        # TODO!!!!

        # Calculate the optimal cost parameters
        optimal_params, _ = optimize.curve_fit(
            self._gm_distribution,
            0,
            self.target_cost_distribution['band_share'].values,
            p0=self._order_init_params(init_params),
            bounds=self._order_bounds(),
            verbose=verbose,
            diff_step=diff_step,
        )

        # Stick optimal params back in dict??


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

