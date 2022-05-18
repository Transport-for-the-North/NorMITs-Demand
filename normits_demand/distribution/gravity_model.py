# -*- coding: utf-8 -*-
"""
Created on: 05/11/2021
Updated on:

Original author: Ben Taylor
Last update made by: Ben Taylor
Other updates made by:

File purpose:

"""
from __future__ import annotations

# Built-Ins
import os
import abc
import enum
import time
import copy
import queue
import warnings
import operator
import functools
import threading
import contextlib
import dataclasses

from typing import Any
from typing import List
from typing import Dict
from typing import Tuple
from typing import Iterable

# Third Party
import numpy as np
import pandas as pd
from scipy import optimize

# Local Imports
import normits_demand as nd

from normits_demand import cost

from normits_demand.utils import timing
from normits_demand.utils import file_ops
from normits_demand.utils import math_utils
from normits_demand.utils import general as du
from normits_demand.utils import pandas_utils as pd_utils

from normits_demand.validation import checks
from normits_demand.distribution import furness
from normits_demand.cost import utils as cost_utils
from normits_demand.concurrency import multithreading
from normits_demand.concurrency import communication


@dataclasses.dataclass(frozen=True)
class FurnessResults:
    mat: np.ndarray
    completed_iters: int
    achieved_rmse: float


@dataclasses.dataclass(frozen=True)
class GravityResults:
    band_share: np.ndarray
    convergence: float
    residuals: np.ndarray
    distribution: np.ndarray
    completed_iters: int
    achieved_rmse: float


@dataclasses.dataclass()
class PartialFurnessRequest:
    seed_mat: np.ndarray
    row_targets: np.ndarray
    col_targets: np.ndarray
    ignore_result: bool = False


@dataclasses.dataclass(frozen=True)
class FurnessSetup:
    area_mats: Dict[Any, np.ndarray]
    gravity_putter_qs: Dict[Any, queue.Queue]
    gravity_getter_qs: Dict[Any, queue.Queue]
    jacobian_putter_qs: Dict[Any, queue.Queue]
    jacobian_getter_qs: Dict[Any, queue.Queue]

    complete_events: Dict[Any, threading.Event]
    all_complete_event: threading.Event

    furness_interface: multithreading.ReturnOrErrorThread
    gravity_furness: multithreading.ReturnOrErrorThread
    jacobian_furness: multithreading.ReturnOrErrorThread
    all_threads: List[multithreading.ReturnOrErrorThread] = dataclasses.field(init=False)

    def __post_init__(self):
        val = [self.furness_interface, self.gravity_furness, self.jacobian_furness]
        super().__setattr__('all_threads', val)


@dataclasses.dataclass(frozen=True)
class SharedArrays:
    gravity_in: communication.SharedNumpyArrayHelper
    gravity_out: communication.SharedNumpyArrayHelper
    jacobian_in: Dict[str, communication.SharedNumpyArrayHelper]
    jacobian_out: Dict[str, communication.SharedNumpyArrayHelper]


class FurnessThreadBase(abc.ABC, multithreading.ReturnOrErrorThread):
    """Base class for running a threaded furness

    Uses its getter_qs to wait for partial matrix inputs. Waits for all
    partial matrices, adds them together, and runs a furness.
    Splits out the furnessed matrix and returns the partial matrices.
    """
    # TODO(BT): Add functionality to allow this to be manually terminated too.

    def __init__(self,
                 area_mats: Dict[Any, np.ndarray],
                 furness_tol: float,
                 furness_max_iters: int,
                 warning: bool,
                 *args,
                 **kwargs,
                 ):
        """
        Parameters
        ----------
        area_mats:
            A dictionary of boolean matrices indicating where the area of each
            area_id is. Keys are the area_ids.

        furness_tol:
            The maximum difference between the achieved and the target values
            in the furness to tolerate before exiting early. Root mean squared
            area is used to calculate the difference.

        furness_max_iters:
            The maximum number of iterations to complete before exiting
            the furness.

        warning:
            Whether to print a warning or not when the tol cannot be met before
            max_iters.

        furness_dtype:
            The datatype being used by the furness. Assumed np.float64 by
            default.
        """
        multithreading.ReturnOrErrorThread.__init__(self, *args, **kwargs)

        self.area_mats = area_mats
        self.furness_tol = furness_tol
        self.furness_max_iters = furness_max_iters
        self.warning = warning
        self.calib_area_keys = area_mats.keys()

    @abc.abstractmethod
    def run_furness(self):
        """Gets data and runs the furness, returning results

        This function will be run forever, until the thread is terminated
        """
        raise NotImplementedError

    def run_target(self) -> None:
        """Runs a furness once all data received, and passes data back

        Runs forever - therefore needs to be a daemon.
        Overrides parent to run this on thread start.

        Returns
        -------
        None
        """
        # Run until program exit.
        while True:
            self.run_furness()


class GravityModelBase(abc.ABC):
    """
    Base Class for gravity models.

    Contains any shared functionality needed across gravity model
    implementations.
    """

    # Class constants
    _avg_cost_col = 'ave_km'        # Should be more generic
    _target_cost_distribution_cols = ['min', 'max', 'trips'] + [_avg_cost_col]
    _least_squares_method = 'trf'

    def __init__(self,
                 cost_function: cost.CostFunction,
                 cost_matrix: np.ndarray,
                 target_cost_distribution: pd.DataFrame,
                 running_log_path: os.PathLike,
                 cost_min_max_buf: float = 0.1,
                 ):
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
        self.cost_function = cost_function
        self.cost_min_max_buf = cost_min_max_buf
        self.cost_matrix = cost_matrix
        self.target_cost_distribution = self._update_tcd(target_cost_distribution)
        self.tcd_bin_edges = self._get_tcd_bin_edges(target_cost_distribution)
        self.running_log_path = running_log_path

        # Running attributes
        self._loop_num: int = -1
        self._loop_start_time: float = -1.0
        self._loop_end_time: float = -1.0
        self._jacobian_mats: Dict[str, np.ndarray] = dict()
        self._perceived_factors: np.ndarray = np.ones_like(self.cost_matrix)

        # Additional attributes
        self.initial_cost_params: Dict[str, Any] = dict()
        self.optimal_cost_params: Dict[str, Any] = dict()
        self.initial_convergence: float = 0
        self.achieved_convergence: float = 0
        self.achieved_band_share: np.ndarray = np.zeros_like(self.target_band_share)
        self.achieved_residuals: np.ndarray = np.full_like(self.target_band_share, np.inf)
        self.achieved_distribution: np.ndarray = np.zeros_like(cost_matrix)

    @property
    def target_band_share(self) -> np.ndarray:
        """Returns the target band share from target cost distribution"""
        return self.target_cost_distribution['band_share'].values

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

    @staticmethod
    def _get_tcd_bin_edges(target_cost_distribution: pd.DataFrame) -> List[float]:
        min_bounds = target_cost_distribution['min'].tolist()
        max_bounds = target_cost_distribution['max'].tolist()
        return [min_bounds[0]] + max_bounds

    def _initialise_calibrate_params(self) -> None:
        """Sets running params to their default values for a run"""
        self._loop_num = 1
        self._loop_start_time = timing.current_milli_time()
        self.initial_cost_params = dict()
        self.initial_convergence = 0
        self._perceived_factors = np.ones_like(self.cost_matrix)

    def _cost_params_to_kwargs(self, args: List[Any]) -> Dict[str, Any]:
        """Converts a list or args into kwargs that self.cost_function expects"""
        if len(args) != len(self.cost_function.kw_order):
            raise ValueError(
                "Received the wrong number of args to convert to cost function "
                "kwargs. Expected %s args, but got %s."
                % (len(self.cost_function.kw_order), len(args))
            )

        return dict(zip(self.cost_function.kw_order, args))

    def _order_cost_params(self, params: Dict[str, Any]) -> List[Any]:
        """Order params into a list that self.cost_function expects"""
        ordered_params = [0] * len(self.cost_function.kw_order)
        for name, value in params.items():
            index = self.cost_function.kw_order.index(name)
            ordered_params[index] = value

        return ordered_params

    def _order_init_params(self, init_params: Dict[str, Any]) -> List[Any]:
        """Order init_params into a list that self.cost_function expects"""
        return self._order_cost_params(init_params)

    def _order_bounds(self) -> Tuple[List[Any], List[Any]]:
        """Order min and max into a tuple of lists that self.cost_function expects"""
        min_vals = self._order_cost_params(self.cost_function.param_min)
        max_vals = self._order_cost_params(self.cost_function.param_max)

        min_vals = [x + self.cost_min_max_buf for x in min_vals]
        max_vals = [x - self.cost_min_max_buf for x in max_vals]

        return min_vals, max_vals

    def _cost_distribution(self,
                           matrix: np.ndarray,
                           tcd_bin_edges: List[float],
                           ) -> np.ndarray:
        """Returns the distribution of matrix across self.tcd_bin_edges"""
        return cost_utils.calculate_cost_distribution(
            matrix=matrix,
            cost_matrix=self.cost_matrix,
            bin_edges=tcd_bin_edges,
        )

    def _guess_init_params(self,
                           cost_args: List[float],
                           target_cost_distribution: pd.DataFrame,
                           ):
        """Internal function of _estimate_init_params()

        Guesses what the initial params should be.
        Used by the `optimize.least_squares` function.
        """
        # Convert the cost function args back into kwargs
        cost_kwargs = self._cost_params_to_kwargs(cost_args)

        # Used to optionally increase the cost of long distance trips
        avg_cost_vals = target_cost_distribution[self._avg_cost_col].values

        # Estimate what the cost function will do to the costs - on average
        estimated_cost_vals = self.cost_function.calculate(avg_cost_vals, **cost_kwargs)
        estimated_band_shares = estimated_cost_vals / estimated_cost_vals.sum()

        # return the residuals to the target
        return target_cost_distribution['band_share'].values - estimated_band_shares

    def _estimate_init_params(self,
                              init_params: Dict[str, Any],
                              target_cost_distribution: pd.DataFrame,
                              ):
        """Guesses what the initial params should be.

        Uses the average cost in each band to estimate what changes in
        the cost_params would do to the final cost distributions. This is a
        very coarse grained estimation, but can be used to guess around about
        where the best init params are.
        """
        result = optimize.least_squares(
            fun=self._guess_init_params,
            x0=self._order_init_params(init_params),
            method=self._least_squares_method,
            bounds=self._order_bounds(),
            kwargs={'target_cost_distribution': target_cost_distribution},
        )
        init_params = self._cost_params_to_kwargs(result.x)

        # TODO(BT): standardise this
        if self.cost_function.name == 'LOG_NORMAL':
            init_params['sigma'] *= 0.8
            init_params['mu'] *= 0.5

        return init_params

    def _calculate_perceived_factors(self) -> None:
        """Updates the perceived cost class variables

        Compares the latest run of the gravity model (as defined by the
        variables: self.achieved_band_share)
        and generates a perceived cost factor matrix, which will be applied
        on calls to self._cost_amplify() in the gravity model.

        This function updates the _perceived_factors class variable.
        """
        # Init
        target_band_share = self.target_cost_distribution['band_share'].values

        # Calculate the adjustment per band in target band share.
        # Adjustment is clipped between 0.5 and 2 to limit affect
        perc_factors = np.divide(
            self.achieved_band_share,
            target_band_share,
            where=target_band_share > 0,
            out=np.ones_like(self.achieved_band_share),
        ) ** 0.5
        perc_factors = np.clip(perc_factors, 0.5, 2)

        # Initialise loop
        perc_factors_mat = np.ones_like(self.cost_matrix)
        min_vals = self.target_cost_distribution['min']
        max_vals = self.target_cost_distribution['max']

        # Convert into factors for the cost matrix
        for min_val, max_val, factor in zip(min_vals, max_vals, perc_factors):
            # Get proportion of all trips that are in this band
            distance_mask = (
                (self.cost_matrix >= min_val)
                & (self.cost_matrix < max_val)
            )

            perc_factors_mat = np.multiply(
                perc_factors_mat,
                factor,
                where=distance_mask,
                out=perc_factors_mat,
            )

        # Assign to class attribute
        self._perceived_factors = perc_factors_mat

    def _apply_perceived_factors(self, cost_matrix: np.ndarray) -> np.ndarray:
        return cost_matrix * self._perceived_factors

    def _gravity_function(self,
                          cost_args: List[float],
                          diff_step: float,
                          ):
        """Returns residuals to target cost distribution

        Runs gravity model with given parameters and converts into achieved
        cost distribution. The residuals are then calculated between the
        achieved and the target.

        Used by the `optimize.least_squares` function.

        This function will populate and update:
            self.achieved_band_share
            self.achieved_convergence
            self.achieved_residuals
            self.achieved_distribution
            self.optimal_cost_params
        """
        # Convert the cost function args back into kwargs
        cost_kwargs = self._cost_params_to_kwargs(cost_args)

        # Used to optionally adjust the cost of long distance trips
        cost_matrix = self._apply_perceived_factors(self.cost_matrix)

        # Calculate initial matrix through cost function
        init_matrix = self.cost_function.calculate(cost_matrix, **cost_kwargs)

        # Do some prep for jacobian calculations
        # TODO(BT): Move this into the Jacobian function. We don't need it here
        #  and it's just using it memory before we need to. Could single loop it
        #  too, so that only one extra cost matrix is needed. NOT n_cost_params
        self._jacobian_mats = {'base': init_matrix.copy()}
        for cost_param in self.cost_function.kw_order:
            # Adjust cost slightly
            adj_cost_kwargs = cost_kwargs.copy()
            adj_cost_kwargs[cost_param] += adj_cost_kwargs[cost_param] * diff_step

            # Calculate adjusted cost
            adj_cost = self.cost_function.calculate(cost_matrix, **adj_cost_kwargs)

            self._jacobian_mats[cost_param] = adj_cost

        # Furness trips to trip ends
        matrix, iters, rmse = self.gravity_furness(seed_matrix=init_matrix)

        # Store for the jacobian calculations
        self._jacobian_mats['final'] = matrix.copy()

        # Convert matrix into an achieved distribution curve
        achieved_band_shares = self._cost_distribution(matrix, self.tcd_bin_edges)

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
        if self.running_log_path is not None:
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
        self._loop_end_time = -1

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

    def _jacobian_function(
            self,
            cost_args: List[float],
            diff_step: float,
            ignore_result: bool = False,
    ):
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

        # Estimate how the final matrix would be different with
        # different input cost parameters
        estimated_mats = dict.fromkeys(self.cost_function.kw_order)
        for cost_param in self.cost_function.kw_order:
            # Estimate what the furness would have done
            furness_mat = self._jacobian_mats[cost_param] * furness_factor
            adj_weights = furness_mat / furness_mat.sum() if furness_mat.sum() != 0 else 0
            adj_final = self._jacobian_mats['final'].sum() * adj_weights

            # Place in dictionary to send to Jacobian
            estimated_mats[cost_param] = adj_final

        # Control estimated matrices to final matrix
        controlled_mats = self.jacobian_furness(
            seed_matrices=estimated_mats,
            row_targets=self._jacobian_mats['final'].sum(axis=1),
            col_targets=self._jacobian_mats['final'].sum(axis=0),
            ignore_result=ignore_result,
        )

        # Calculate the Jacobian
        for i, cost_param in enumerate(self.cost_function.kw_order):
            # Turn into bands
            achieved_band_shares = self._cost_distribution(
                matrix=controlled_mats[cost_param],
                tcd_bin_edges=self.tcd_bin_edges,
            )

            # Calculate the Jacobian for this cost param
            jacobian_residuals = self.achieved_band_share - achieved_band_shares
            cost_step = cost_kwargs[cost_param] * diff_step
            cost_jacobian = jacobian_residuals / cost_step

            # Store in the Jacobian
            jacobian[:, i] = cost_jacobian

        return jacobian

    def _calibrate(self,
                   init_params: Dict[str, Any],
                   calibrate_params: bool = True,
                   diff_step: float = 1e-8,
                   ftol: float = 1e-4,
                   xtol: float = 1e-4,
                   grav_max_iters: int = 100,
                   failure_tol: float = 0,
                   default_params: Dict[str, Any] = None,
                   verbose: int = 0,
                   ) -> None:
        """Internal function of calibrate.

        Runs the gravity model, and calibrates the optimal cost parameters
        if calibrate params is set to True. Will do a final run of the
        gravity_function with the optimal parameter found before return.
        """
        # Initialise running params
        self._initialise_calibrate_params()

        # Calculate the optimal cost parameters if we're calibrating
        if calibrate_params is True:
            result = optimize.least_squares(
                fun=self._gravity_function,
                x0=self._order_init_params(init_params),
                method=self._least_squares_method,
                bounds=self._order_bounds(),
                jac=self._jacobian_function,
                verbose=verbose,
                ftol=ftol,
                xtol=xtol,
                max_nfev=grav_max_iters,
                kwargs={'diff_step': diff_step},
            )
            optimal_params = result.x

            # If performance was terrible, try again with default params
            if self.achieved_convergence <= failure_tol and default_params is not None:
                result = optimize.least_squares(
                    fun=self._gravity_function,
                    x0=self._order_init_params(default_params),
                    method=self._least_squares_method,
                    bounds=self._order_bounds(),
                    jac=self._jacobian_function,
                    verbose=verbose,
                    ftol=ftol,
                    xtol=xtol,
                    max_nfev=grav_max_iters,
                    kwargs={'diff_step': diff_step},
                )
                optimal_params = result.x

            # BACKLOG: Try random init_params as a final option

        else:
            optimal_params = self._order_init_params(init_params)

        # Run an optimal version of the gravity
        self.optimal_cost_params = self._cost_params_to_kwargs(optimal_params)
        self._gravity_function(optimal_params, diff_step=diff_step)

    @abc.abstractmethod
    def gravity_furness(
        self,
        seed_matrix: np.ndarray,
    ) -> Tuple[np.ndarray, int, float]:
        """Runs a doubly constrained furness on the seed matrix

        Wrapper around furness.doubly_constrained_furness, to be used when
        running the furness withing the gravity model.

        Parameters
        ----------
        seed_matrix:
            Initial values for the furness.

        Returns
        -------
        furnessed_matrix:
            The final furnessed matrix

        completed_iters:
            The number of completed iterations before exiting

        achieved_rmse:
            The Root Mean Squared Error difference achieved before exiting
        """
        raise NotImplementedError

    @abc.abstractmethod
    def jacobian_furness(self,
                         seed_matrices: Dict[str, np.ndarray],
                         row_targets: np.ndarray,
                         col_targets: np.ndarray,
                         ignore_result: bool = False,
                         ) -> Dict[str, np.ndarray]:
        """Runs a doubly constrained furness on the seed matrix

        Wrapper around furness.doubly_constrained_furness, to be used when
        running the furness withing the jacobian calculation.

        Parameters
        ----------
        seed_matrices:
            Dictionary of initial values for the furness.
            Keys are the name of the cost params which has been changed
            to get this new seed matrix.

        row_targets:
            The target values for the sum of each row.
            i.e np.sum(seed_matrix, axis=1)

        col_targets:
            The target values for the sum of each column
            i.e np.sum(seed_matrix, axis=0)

        ignore_result:
            Whether to ignore the return result or not. Useful when a Jacobian
            furness is only being called to satisfy other threads.

        Returns
        -------
        furnessed_matrix:
            The final furnessed matrix

        completed_iters:
            The number of completed iterations before exiting

        achieved_rmse:
            The Root Mean Squared Error difference achieved before exiting
        """
        raise NotImplementedError


class GravityModelCalibrator(GravityModelBase):
    # TODO(BT): Write GravityModelCalibrator docs

    def __init__(self,
                 row_targets: np.ndarray,
                 col_targets: np.ndarray,
                 cost_function: cost.CostFunction,
                 cost_matrix: np.ndarray,
                 target_cost_distribution: pd.DataFrame,
                 target_convergence: float,
                 furness_max_iters: int,
                 furness_tol: float,
                 running_log_path: os.PathLike,
                 use_perceived_factors: bool = True,
                 ):
        # TODO(BT): Write GravityModelCalibrator __init__ docs
        super().__init__(
            cost_function=cost_function,
            cost_matrix=cost_matrix,
            target_cost_distribution=target_cost_distribution,
            running_log_path=running_log_path,
        )

        # Set attributes
        self.row_targets = row_targets
        self.col_targets = col_targets
        self.furness_max_iters = furness_max_iters
        self.furness_tol = furness_tol
        self.use_perceived_factors = use_perceived_factors

        self.target_convergence = target_convergence

    def gravity_furness(self,
                        seed_matrix: np.ndarray,
                        ) -> Tuple[np.ndarray, int, float]:
        """Runs a doubly constrained furness on the seed matrix

        Wrapper around furness.doubly_constrained_furness, using class
        attributes to set up the function call.

        Parameters
        ----------
        seed_matrix:
            Initial values for the furness.

        Returns
        -------
        furnessed_matrix:
            The final furnessed matrix

        completed_iters:
            The number of completed iterations before exiting

        achieved_rmse:
            The Root Mean Squared Error difference achieved before exiting
        """
        return furness.doubly_constrained_furness(
            seed_vals=seed_matrix,
            row_targets=self.row_targets,
            col_targets=self.col_targets,
            tol=self.furness_tol,
            max_iters=self.furness_max_iters,
        )

    def jacobian_furness(self,
                         seed_matrices: Dict[str, np.ndarray],
                         row_targets: np.ndarray,
                         col_targets: np.ndarray,
                         ignore_result: bool = False,
                         ) -> Dict[str, np.ndarray]:
        """Runs a doubly constrained furness on the seed matrix

        Wrapper around furness.doubly_constrained_furness, to be used when
        running the furness withing the jacobian calculation.

        Parameters
        ----------
        seed_matrices:
            Dictionary of initial values for the furness.
            Keys are the name of the cost params which has been changed
            to get this new seed matrix.

        row_targets:
            The target values for the sum of each row.
            i.e np.sum(seed_matrix, axis=1)

        col_targets:
            The target values for the sum of each column
            i.e np.sum(seed_matrix, axis=0)

        ignore_result:
            Whether to ignore the return result or not. Useful when a Jacobian
            furness is only being called to satisfy other threads.

        Returns
        -------
        furnessed_matrix:
            The final furnessed matrix

        completed_iters:
            The number of completed iterations before exiting

        achieved_rmse:
            The Root Mean Squared Error difference achieved before exiting
        """
        return_dict = dict.fromkeys(seed_matrices.keys())
        for cost_param, seed_matrix in seed_matrices.items():
            return_dict[cost_param], *_ = furness.doubly_constrained_furness(
                seed_vals=seed_matrix,
                row_targets=row_targets,
                col_targets=col_targets,
                tol=1e-6,
                max_iters=20,
                warning=False,
            )

        return return_dict

    def calibrate(self,
                  init_params: Dict[str, Any],
                  estimate_init_params: bool = False,
                  calibrate_params: bool = True,
                  diff_step: float = 1e-8,
                  ftol: float = 1e-4,
                  xtol: float = 1e-4,
                  grav_max_iters: int = 100,
                  failure_tol: float = 0,
                  default_init_params: Dict[str, Any] = None,
                  verbose: int = 0,
                  ) -> Dict[str, Any]:
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
            `x * diff_step`. If None (default), then diff_step is taken to be a
            conventional “optimal” power of machine epsilon for the finite
            difference scheme used

        ftol:
            The tolerance to pass to `scipy.optimize.least_squares`. The search
            will stop once this tolerance has been met. This is the
            tolerance for termination by the change of the cost function

        xtol:
            The tolerance to pass to `scipy.optimize.least_squares`. The search
            will stop once this tolerance has been met. This is the
            tolerance for termination by the change of the independent
            variables.

        grav_max_iters:
            The maximum number of calibration iterations to complete before
            termination if the ftol has not been met.

        failure_tol:
            The threshold that a convergence needs to pass to not be
            considered a failure. Any convergence values less than or equal
            to this value will be considered a failure. Used in conjunction
            with `default_init_params`.

        default_init_params:
            The default initial parameters to be used to try and calibrate
            the gravity model if the given initial params in `init_params`
            lead to a failing convergence.

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
            init_params = self._estimate_init_params(
                init_params=init_params,
                target_cost_distribution=self.target_cost_distribution,
            )

        # Figure out the optimal cost params
        self._calibrate(
            init_params=init_params,
            calibrate_params=calibrate_params,
            diff_step=diff_step,
            ftol=ftol,
            xtol=xtol,
            grav_max_iters=grav_max_iters,
            failure_tol=failure_tol,
            default_params=default_init_params,
            verbose=verbose,
        )

        # Just return if not using perceived factors
        if not self.use_perceived_factors:
            return self.optimal_cost_params

        # ## APPLY PERCEIVED FACTORS IF WE CAN ## #
        upper_limit = self.target_convergence + 0.03
        lower_limit = self.target_convergence - 0.15

        # Just return if upper limit has been beaten
        if self.achieved_convergence > upper_limit:
            return self.optimal_cost_params

        # Warn if the lower limit hasn't been reached
        if self.achieved_convergence < lower_limit:
            warnings.warn(
                "Calibration was not able to reach the lower threshold "
                "required to use perceived factors.\n"
                "Target convergence: %s\n"
                "Upper Limit: %s\n"
                "Achieved convergence: %s"
                % (self.target_convergence, upper_limit, self.achieved_convergence)
            )
            return self.optimal_cost_params

        # If here, it's safe to use perceived factors
        self._calculate_perceived_factors()

        # Calibrate again, using the perceived factors
        self._calibrate(
            init_params=self.optimal_cost_params.copy(),
            calibrate_params=calibrate_params,
            diff_step=diff_step,
            ftol=ftol,
            xtol=xtol,
            grav_max_iters=grav_max_iters,
            verbose=verbose,
        )

        if self.achieved_convergence < self.target_convergence:
            warnings.warn(
                "Calibration with perceived factors was not able to reach the "
                "target_convergence.\n"
                "Target convergence: %s\n"
                "Achieved convergence: %s"
                % (self.target_convergence, self.achieved_convergence)
            )

        return self.optimal_cost_params


class FurnessThreadInterfaceBase(abc.ABC, multithreading.ReturnOrErrorThread):
    """Communicates between multiple furness callers and receivers

    Handles multiple furness callers by checking the status of each
    caller and making sure the correct action is taken in each case. An
    internal cache is used to make sure no caller gets stuck waiting for a
    Jacobian call to return results while other threads are waiting for a
    furness. If callers stray out of sync, this interface brings all
    callers back into sync. The cache is used to allow the interface to
    re-send data to allow threads to progress and "catch-up" with
    one another.

    This interface also handles setting the self.all_complete_event once
    all furness callers mark themselves as complete.
    """

    # Class constants
    _default_gravity_furness_key = 'furness'
    _default_jacobian_key = 'jacobian'

    @dataclasses.dataclass(frozen=True)
    class FurnessKeys:
        gravity: str
        jacobian: str
        all: List[str] = dataclasses.field(init=False)

        def __post_init__(self):
            super().__setattr__('all', [self.gravity, self.jacobian])

    @enum.unique
    class ThreadStatus(enum.Enum):
        UNKNOWN = 'unknown'
        DONE = 'done'
        GRAVITY = 'gravity'
        JACOBIAN = 'jacobian'

    def __init__(
        self,
        furness_wait_events: Dict[str, Dict[Any, threading.Event]],
        complete_events: Dict[Any, threading.Event],
        area_mats: Dict[Any, np.ndarray],
        all_complete_event: threading.Event,
        getter_qs: Dict[str, Dict[Any, queue.Queue]],
        putter_qs: Dict[str, Dict[Any, queue.Queue]],
        thread_ids: List[Any] = None,
        gravity_furness_key: str = None,
        jacobian_key: str = None,
        *args,
        **kwargs,
    ):
        """
        Parameters
        ----------
        furness_wait_events:
            A nested dictionary, where the first set of keys are the names of
            the furness threads, as defined in furness_priority. The second
            set of keys are the identifiers for each thread calling the
            furnesses. The values are the events that will be set by a thread
            when it is waiting for that furness to complete - the event should
            also be cleared once a thread has received its data.

        complete_events:
            A dictionary of events. The keys should be the same as thread_ids.
            The values are event indicating if the thread with that key is
            complete or not.

        area_mats:
            A dictionary of boolean numpy arrays. The keys should be the same
            as thread_ids. The values indicating which cells are within
            an area, and which are not.

        getter_qs:
            A dictionary in the same format as those nested in
            furness_wait_events. The keys should correspond to thread_ids, and
            the values the Queues to expect data to come in from each thread.

        putter_qs:
            A dictionary in the same format as those nested in
            furness_wait_events. The keys should correspond to thread_ids, and
            the values the Queues to expect data to come in from each thread.

        thread_ids:
            A list of ids to identify each thread. This should correspond to
            the second key in furness_wait_events. If left as None, this is
            assumed from `furness_wait_events[furness_priority[0]].keys()`

        gravity_furness_key:
            The key name used within furness_wait_events, getter_qs, and
            putter_qs to identify objects relating the gravity furness.
            If left as None, self._default_gravity_furness_key is used.

        jacobian_key:
            The key name used within furness_wait_events, getter_qs, and
            putter_qs to identify objects relating the gravity furness.
            If left as None, self._default_jacobian_key is used.
        """
        multithreading.ReturnOrErrorThread.__init__(self, *args, **kwargs)

        # Set default values if not given
        if gravity_furness_key is None:
            gravity_furness_key = self._default_gravity_furness_key
        if jacobian_key is None:
            jacobian_key = self._default_jacobian_key

        if thread_ids is None:
            thread_ids = list(furness_wait_events[gravity_furness_key].keys())

        # Build keys dataclass. Needed for further validation
        self.furness_keys = self.FurnessKeys(
            gravity=gravity_furness_key,
            jacobian=jacobian_key
        )

        # Validate inputs
        name_dict: Dict[str, Dict[Any, Any]] = {
            'furness_wait_events': furness_wait_events,
            'getter_qs': getter_qs,
            'putter_qs': putter_qs,
        }

        for name, dict_to_check in name_dict.items():
            if not checks.all_keys_exist(dict_to_check, self.furness_keys.all):
                raise ValueError(
                    "%s needs to contain values for all "
                    "furness_priority."
                    "Keys needed: %s\n"
                    "Keys found: %s\n"
                    % (name, self.furness_keys.all, list(dict_to_check.keys()))
                )

            for key, value in dict_to_check.items():
                if not checks.all_keys_exist(value, thread_ids):
                    raise ValueError(
                        "%s[%s] needs to contain values for all "
                        "thread_ids."
                        "Keys needed: %s\n"
                        "Keys found: %s\n"
                        % (name, key, thread_ids, list(value.keys()))
                    )

        # Attributes
        self.area_mats = area_mats
        self.furness_wait_events = furness_wait_events
        self.complete_events = complete_events
        self.all_complete_event = all_complete_event
        self.getter_qs = getter_qs
        self.putter_qs = putter_qs
        self.thread_ids = thread_ids

        # internal running attributes
        self._jacobian_q_cache = dict.fromkeys(thread_ids)

    def _create_thread_status(self) -> Dict[Any, FurnessThreadInterfaceSharedArrays.ThreadStatus]:
        """Builds a dictionary of ThreadStatus set to Unknown"""
        return {x: self.ThreadStatus.UNKNOWN for x in self.thread_ids}

    def _get_thread_status(self, thread_id: Any) -> FurnessThreadInterfaceSharedArrays.ThreadStatus:
        """Gets the status of thread and returns it."""
        # Loop until the status is gotten
        while True:
            # Wait for a little bit to avoid intensive looping
            time.sleep(0.05)

            # Check if waiting for gravity
            if not self.getter_qs[self.furness_keys.gravity][thread_id].empty():
                return self.ThreadStatus.GRAVITY

            # Check if waiting for jacobian
            if not self.getter_qs[self.furness_keys.jacobian][thread_id].empty():
                return self.ThreadStatus.JACOBIAN

    def _all_threads_complete(self) -> bool:
        """Checks whether all threads are complete or not"""
        for thread_id in self.thread_ids:
            if not self.complete_events[thread_id].is_set():
                return False
        return True

    @staticmethod
    def _all_status_same(
        statuses: Iterable[FurnessThreadInterfaceSharedArrays.ThreadStatus],
        check_status: FurnessThreadInterfaceSharedArrays.ThreadStatus,
    ) -> bool:
        """Check if all statuses are set to check_status

        Parameters
        ----------
        statuses:
            An iterable of statuses to check.

        check_status:
            The status to check that all statuses are.

        Returns
        -------
        all_same:
            True if all statuses are set to check_status. Else False
        """
        for status in statuses:
            if status != check_status:
                return False
        return True

    @staticmethod
    def _any_status(
        statuses: Iterable[FurnessThreadInterfaceSharedArrays.ThreadStatus],
        check_status: FurnessThreadInterfaceSharedArrays.ThreadStatus,
    ) -> bool:
        """Check if any statuses are set to check_status

        Parameters
        ----------
        statuses:
            An iterable of statuses to check.

        check_status:
            The status to check that any status is.

        Returns
        -------
        all_same:
            True if any status is set to check_status. Else False
        """
        for status in statuses:
            if status == check_status:
                return True
        return False

    def send_gravity(
        self,
        thread_id: Any,
    ) -> None:
        """Gets the data and sends on to the gravity furness

        Parameters
        ----------
        thread_id:
            The key in the getter and putter queue dictionaries defining
            which thread is being used.
        """
        # Init
        getter_q = self.getter_qs[self.furness_keys.gravity][thread_id]
        putter_q = self.putter_qs[self.furness_keys.gravity][thread_id]

        # Get queue data and pass on
        q_data = multithreading.get_data_from_queue(q=getter_q)
        putter_q.put(q_data)

    @abc.abstractmethod
    def cache_and_send_jacobian(
        self,
        thread_id: Any,
        q_cache: Dict[Any, Any],
    ) -> None:
        """Gets the data, caches it, and then sends a copy on

        Parameters
        ----------
        thread_id:
            The key in the getter and putter queue dictionaries defining
            which thread is being used.

        q_cache:
            A dictionary of thread_id keys, where the data gotten from the
            getter queue will be cached.
        """
        pass

    @abc.abstractmethod
    def retrieve_cache_and_send_jacobian(
        self,
        thread_id: Any,
    ) -> None:
        pass

    def run_target(self):
        """Handles passing data to furnesses to keep everything running.

        Each furness thread does not follow a deterministic path, e.g.
        run a furness, then jacobian, then furness etc...
        It appears the running is more like:
        1. Always run a furness
        2. Run 0 or more Jacobians

        Furthermore, it is unlikely that all gravity threads will finish at
        the same time.

        Therefore some functionality is needed to manage the threads to keep
        them in sync, and all able to complete. That is the aim of this thread.

        Until program termination, this thread will:
        1. Try to find that status of all threads
        2. If a jacobian or furness complete event is set, reset relevant
           thread status
        3. Once all status gathered do one of the following.
            3.1. If all threads complete mark all complete event.
            3.2. If all thread events are the same, do nothing.
            3.3. If some threads are waiting to Jacobian:
                 Send the cached data of the threads which are either done,
                 or waiting to furness.
            3.4. If some threads are waiting to Furness:
                 Should only be done threads left, send their cache.

        NOTE: Do we still need to run the furness / jacobian and evaluate if
        a thread is done?? keeps the jacobian and furness up to date.
        """
        # run until program exit, or all threads complete
        while True:
            # Reset thread statuses
            thread_statuses = self._create_thread_status()

            # Try to get the status of each thread
            for thread_id in thread_statuses:
                thread_statuses[thread_id] = self._get_thread_status(thread_id)

            # Check which threads are complete
            complete = dict.fromkeys(self.thread_ids)
            for thread_id in self.thread_ids:
                complete[thread_id] = self.complete_events[thread_id].is_set()

            # Check if all the threads are complete
            if self._all_threads_complete():
                self.all_complete_event.set()

            # If all waiting for furness, cache data and send
            if self._all_status_same(thread_statuses.values(), self.ThreadStatus.GRAVITY):
                for thread_id in self.thread_ids:
                    self.send_gravity(thread_id=thread_id)

            # If all waiting for jacobian, cache data and send
            elif self._all_status_same(thread_statuses.values(), self.ThreadStatus.JACOBIAN):
                for thread_id in self.thread_ids:
                    self.cache_and_send_jacobian(
                        thread_id=thread_id,
                        q_cache=self._jacobian_q_cache,
                    )

            # If any waiting for the Jacobian, try to move them along first
            elif self._any_status(thread_statuses.values(), self.ThreadStatus.JACOBIAN):
                for thread_id in self.thread_ids:
                    # Cache and send data if waiting for jacobian
                    if thread_statuses[thread_id] == self.ThreadStatus.JACOBIAN:
                        self.cache_and_send_jacobian(
                            thread_id=thread_id,
                            q_cache=self._jacobian_q_cache,
                        )

                    # Otherwise, must be waiting for gravity.
                    # Send the cached data
                    else:
                        self.retrieve_cache_and_send_jacobian(thread_id=thread_id)
            else:
                raise NotImplementedError(
                    "This case shouldn't be able to happen!"
                )


class FurnessThreadInterfaceSharedArrays(FurnessThreadInterfaceBase):
    """Communicates between multiple furness callers and receivers

    Handles multiple furness callers by checking the status of each
    caller and making sure the correct action is taken in each case. An
    internal cache is used to make sure no caller gets stuck waiting for a
    Jacobian call to return results while other threads are waiting for a
    furness. If callers stray out of sync, this interface brings all
    callers back into sync. The cache is used to allow the interface to
    re-send data to allow threads to progress and "catch-up" with
    one another.

    This version of the base class (`FurnessThreadInterfaceBase`) uses
    shared numpy arrays to communicate data between threads. This is the
    most efficient method of communication (in terms of CPU time), but it
    comes at the cost of reserving some memory for the arrays. The bigger the
    zoning system in use, the bigger the memeory cost.

    This interface also handles setting the self.all_complete_event once
    all furness callers mark themselves as complete.
    """

    def __init__(
        self,
        gravity_in_array: communication.SharedNumpyArrayHelper,
        jacobian_in_array: Dict[str, communication.SharedNumpyArrayHelper],
        *args,
        **kwargs,
    ):
        """

        Parameters
        ----------
        gravity_in_array:
            The shared input array for the gravity furness.

        jacobian_in_array:
            A dictionary of the shared input array for the jacobian furness.
            Keys do not matter internally to this class.

        args:
            Compatibility with `FurnessThreadInterfaceBase` and
            `multithreading.ReturnOrErrorThread`

        kwargs:
            Compatibility with `FurnessThreadInterfaceBase` and
            `multithreading.ReturnOrErrorThread`
        """
        super().__init__(*args, **kwargs)

        # Attributes
        self.gravity_in_array = gravity_in_array
        self.jacobian_in_array = jacobian_in_array

        # Internal caches
        self._jacobian_array_cache = dict.fromkeys(
            jacobian_in_array.keys(),
            dict.fromkeys(self.thread_ids),
        )

    def cache_and_send_jacobian(
        self,
        thread_id: Any,
        q_cache: Dict[Any, Any],
    ) -> None:
        """Gets the data, caches it, and then sends a copy on

        Parameters
        ----------
        thread_id:
            The key in the getter and putter queue dictionaries defining
            which thread is being used.

        q_cache:
            A dictionary of thread_id keys, where the data gotten from the
            getter queue will be cached.
        """
        # Init
        getter_q = self.getter_qs[self.furness_keys.jacobian][thread_id]
        putter_q = self.putter_qs[self.furness_keys.jacobian][thread_id]

        # Get and cache the queue data
        q_data = multithreading.get_data_from_queue(q=getter_q)
        q_cache[thread_id] = q_data

        # Get and cache a copy of the array data
        for jac_key, in_array in self.jacobian_in_array.items():
            array_data = in_array.get_local_copy()
            array_data *= self.area_mats[thread_id]
            self._jacobian_array_cache[jac_key][thread_id] = array_data

        # Let the furness know data is ready
        putter_q.put(copy.copy(q_data))

    def retrieve_cache_and_send_jacobian(
        self,
        thread_id: Any,
    ) -> None:
        """Gets the cached data and sends a copy on

        Parameters
        ----------
        thread_id:
            The key in the getter and putter queue dictionaries defining
            which thread is being used.
        """
        # Init
        furness_key = self.furness_keys.jacobian
        q_cache = self._jacobian_q_cache

        # Add the cached matrix data to the shared arrays
        for jac_key, array_in in self.jacobian_in_array.items():
            array_in.apply_local_data(
                data=copy.copy(self._jacobian_array_cache[jac_key][thread_id]),
                operation=operator.add,
            )

        # Send the cached queue data - informs array has data
        q_data = copy.copy(q_cache[thread_id])
        q_data.ignore_result = True
        self.putter_qs[furness_key][thread_id].put(q_data)


class FurnessThreadInterfaceQueues(FurnessThreadInterfaceBase):
    """Communicates between multiple furness callers and receivers

    Handles multiple furness callers by checking the status of each
    caller and making sure the correct action is taken in each case. An
    internal cache is used to make sure no caller gets stuck waiting for a
    Jacobian call to return results while other threads are waiting for a
    furness. If callers stray out of sync, this interface brings all
    callers back into sync. The cache is used to allow the interface to
    re-send data to allow threads to progress and "catch-up" with
    one another.

    This version of the base class (`FurnessThreadInterfaceBase`) uses
    queues to communicate arrays between threads. This method is slow for
    communication, but is memory efficient when compared to
    `FurnessThreadInterfaceSharedArrays`.

    This interface also handles setting the self.all_complete_event once
    all furness callers mark themselves as complete.
    """

    def cache_and_send_jacobian(
        self,
        thread_id: Any,
        q_cache: Dict[Any, Any],
    ) -> None:
        """Gets the data, caches it, and then sends a copy on

        Parameters
        ----------
        thread_id:
            The key in the getter and putter queue dictionaries defining
            which thread is being used.

        q_cache:
            A dictionary of thread_id keys, where the data gotten from the
            getter queue will be cached.
        """
        # Init
        getter_q = self.getter_qs[self.furness_keys.jacobian][thread_id]
        putter_q = self.putter_qs[self.furness_keys.jacobian][thread_id]

        # Get and cache the queue data
        q_data = multithreading.get_data_from_queue(q=getter_q)
        q_cache[thread_id] = q_data

        # Let the furness know data is ready
        putter_q.put(copy.copy(q_data))

    def retrieve_cache_and_send_jacobian(
        self,
        thread_id: Any,
    ) -> None:
        """Gets the cached data and sends a copy on

        Parameters
        ----------
        thread_id:
            The key in the getter and putter queue dictionaries defining
            which thread is being used.
        """
        # Init
        putter_q = self.putter_qs[self.furness_keys.jacobian][thread_id]
        q_cache = self._jacobian_q_cache[thread_id]

        # Send the cached data and mark it so we know not to care about result
        q_data = copy.copy(q_cache)
        q_data.ignore_result = True
        putter_q.put(q_data)


class GravityFurnessThreadSharedArrays(FurnessThreadBase):
    """Collects partial matrices and runs a furness

    Uses its getter_qs to wait for partial matrix inputs. Waits for all
    partial matrices, adds them together, and runs a furness.
    Splits out the furnessed matrix and returns the partial matrices.
    """
    def __init__(self,
                 row_targets: np.ndarray,
                 col_targets: np.ndarray,
                 getter_qs: Dict[Any, queue.Queue],
                 putter_qs: Dict[Any, queue.Queue],
                 getter_array: communication.SharedNumpyArrayHelper,
                 putter_array: communication.SharedNumpyArrayHelper,
                 *args,
                 **kwargs,
                 ):
        """
        Parameters
        ----------
        row_targets:
            The row targets to aim for when running the furness. This should
            be the target when `.sum(axis=1) is applied to the full matrix.

        col_targets:
            The row targets to aim for when running the furness. This should
            be the target when `.sum(axis=1) is applied to the full matrix.

        getter_qs:
            A dictionary of Queues for each area_id. Furness will be run once
            data has been received from all queues.

        putter_qs:
            A dictionary of Queues for each area_id. Queues are used to pass
            info back on the completed furness.

        getter_array:
            A shared array where the furness should get its input seed matrix
            from.

        putter_array:
            A shared array where the furness will place its completed furnessed
            matrix.

        *args, **kwargs:
            Arguments to be passed to parent FurnessThreadBase class

        See Also
        --------
        `FurnessThreadBase`
        """
        super().__init__(
            *args,
            **kwargs,
        )

        # Set attributes
        self.row_targets = row_targets
        self.col_targets = col_targets
        self.getter_qs = getter_qs
        self.putter_qs = putter_qs
        self._array_in = getter_array
        self._array_out = putter_array

        # Make sure the in and out arrays are initialised to 0
        self._array_in.reset_zeros()
        self._array_out.reset_zeros()

    @property
    def array_in(self):
        return self._array_in

    @property
    def array_out(self):
        return self._array_out

    def run_furness(self) -> None:
        """Runs a furness once all data received, and passes data back

        Returns
        -------
        None
        """
        # ## GET SEED MAT DATA ## #
        # Wait for queues. Tells us when shared array is read
        multithreading.get_data_from_queue_dict(self.getter_qs)

        # Get the shared data, and reset the shared memory
        seed_mat = self._array_in.get_local_copy()
        self._array_in.reset_zeros()

        # ## FURNESS ## #
        furnessed_mat, iters, rmse = furness.doubly_constrained_furness(
            seed_vals=seed_mat,
            row_targets=self.row_targets,
            col_targets=self.col_targets,
            tol=self.furness_tol,
            max_iters=self.furness_max_iters,
            warning=self.warning,
        )

        # ## RETURN RESULTS ## #
        # Put the furnessed matrix in the return array
        self._array_out.write_local_data(furnessed_mat)

        # Put the data back on the queues
        # Also lets threads know data is waiting
        for area_id in self.calib_area_keys:
            data = FurnessResults(
                mat=self.area_mats[area_id],  # Return area_mat
                completed_iters=iters,
                achieved_rmse=rmse,
            )
            self.putter_qs[area_id].put(data)


class GravityFurnessThreadQueues(FurnessThreadBase):
    """Collects partial matrices and runs a furness

    Uses its getter_qs to wait for partial matrix inputs. Waits for all
    partial matrices, adds them together, and runs a furness.
    Splits out the furnessed matrix and returns the partial matrices.
    """
    def __init__(
        self,
        row_targets: np.ndarray,
        col_targets: np.ndarray,
        getter_qs: Dict[Any, queue.Queue],
        putter_qs: Dict[Any, queue.Queue],
        *args,
        **kwargs,
    ):
        """
        Parameters
        ----------
        row_targets:
            The row targets to aim for when running the furness. This should
            be the target when `.sum(axis=1) is applied to the full matrix.

        col_targets:
            The row targets to aim for when running the furness. This should
            be the target when `.sum(axis=1) is applied to the full matrix.

        getter_qs:
            A dictionary of Queues for each area_id. Furness will be run once
            data has been received from all queues.

        putter_qs:
            A dictionary of Queues for each area_id. Queues are used to pass
            info back on the completed furness.

        *args, **kwargs:
            Arguments to be passed to parent FurnessThreadBase class

        See Also
        --------
        `FurnessThreadBase`
        """
        super().__init__(
            *args,
            **kwargs,
        )

        # Set attributes
        self.row_targets = row_targets
        self.col_targets = col_targets
        self.getter_qs = getter_qs
        self.putter_qs = putter_qs

    def run_furness(self) -> None:
        """Runs a furness once all data received, and passes data back

        Runs forever - therefore needs to be a daemon.
        Overrides parent to run this on thread start.

        Returns
        -------
        None
        """
        # ## GET SEED MAT DATA ## #
        seed_mat_dict = multithreading.get_data_from_queue_dict(self.getter_qs)
        seed_mat = functools.reduce(operator.add, seed_mat_dict.values())

        # ## FURNESS ## #
        furnessed_mat, iters, rmse = furness.doubly_constrained_furness(
            seed_vals=seed_mat,
            row_targets=self.row_targets,
            col_targets=self.col_targets,
            tol=self.furness_tol,
            max_iters=self.furness_max_iters,
            warning=self.warning,
        )

        # ## RETURN RESULTS ## #
        # Split back out into areas and return
        for area_id in self.calib_area_keys:
            data = FurnessResults(
                mat=furnessed_mat * self.area_mats[area_id],
                completed_iters=iters,
                achieved_rmse=rmse,
            )
            self.putter_qs[area_id].put(data)


class JacobianFurnessThreadSharedArrays(FurnessThreadBase):
    """Collects partial matrices and runs a furness

    Uses its getter_qs to wait for partial matrix inputs. Waits for all
    partial matrices, adds them together, and runs a furness.
    Splits out the furnessed matrix and returns the partial matrices.
    """
    def __init__(self,
                 getter_qs: Dict[Any, queue.Queue],
                 putter_qs: Dict[Any, queue.Queue],
                 getter_arrays: Dict[str, communication.SharedNumpyArrayHelper],
                 putter_arrays: Dict[str, communication.SharedNumpyArrayHelper],
                 *args,
                 **kwargs,
                 ):
        """
        Parameters
        ----------
        getter_qs:
            A dictionary of Queues for each area_id. Furness will be run once
            data has been received from all queues.

        putter_qs:
            A dictionary of Queues for each area_id. Queues are used to pass
            info back on the completed furness.

        getter_array:
            A dictionary of shared arrays where the furness should get its
            input seed matrices from. Keys are the unique ids for each
            jacobian furness that needs to be completed.

        putter_array:
            A dictionary of shared arrays where the furness will place its
            completed furnessed matrices. Keys are the unique ids for each
            jacobian furness that needs to be completed.

        *args, **kwargs:
            Arguments to be passed to parent FurnessThreadBase class

        See Also
        --------
        `FurnessThreadBase`
        """
        super().__init__(
            *args,
            **kwargs,
        )

        # Assign attributes
        self.getter_qs = getter_qs
        self.putter_qs = putter_qs
        self.getter_arrays = getter_arrays
        self.putter_arrays = putter_arrays
        self._jac_keys = getter_arrays.keys()

        # Make sure the in and out arrays are initialised to 0
        [x.reset_zeros() for x in self.getter_arrays.values()]
        [x.reset_zeros() for x in self.putter_arrays.values()]

    def get_furness_data(self):
        """Grabs the needed data for the furness to run

        Returns
        -------
        seed_mats:
            A dictionary of retrieved partial seed matrices that need to be
            combined to create the full seed matrix for the furness.

        row_targets:
            The row targets to be used for the furness.
            i.e the target of np.sum(furnessed_matrix, axis=1)

        col_targets:
            The col targets to be used for the furness.
            i.e the target of np.sum(furnessed_matrix, axis=0)

        all_ignore:
            Boolean value. True if all callers want to ignore this run of the
            Jacobian. Used to optimise runs as the Jacobian will not be run
            if none of the callers care about the result.
        """
        # Init
        seed_mats = dict.fromkeys(self._jac_keys)

        # Get all the data
        partial_furness_requests = multithreading.get_data_from_queue_dict(self.getter_qs)

        # Get the shared data, and reset the shared memory
        for key in self._jac_keys:
            seed_mats[key] = self.getter_arrays[key].get_local_copy()
            self.getter_arrays[key].reset_zeros()

        # Get the row and col targets
        row_targets_list = list()
        col_targets_list = list()
        ignore_list = list()
        for key, request in partial_furness_requests.items():
            row_targets_list.append(request.row_targets)
            col_targets_list.append(request.col_targets)
            ignore_list.append(request.ignore_result)

        # Combine individual items
        row_targets = functools.reduce(operator.add, row_targets_list)
        col_targets = functools.reduce(operator.add, col_targets_list)
        all_ignore = all(ignore_list)

        return seed_mats, row_targets, col_targets, all_ignore

    def run_furness(self) -> None:
        """Runs a furness once all data received, and passes data back

        Runs forever - therefore needs to be a daemon.
        Overrides parent to run this on thread start.

        Returns
        -------
        None
        """
        # Get seed mats
        seed_mats, row_targets, col_targets, all_ignore = self.get_furness_data()

        # Only run and return data if any threads care about the result
        if not all_ignore:
            # ## FURNESS ## #
            for key, seed_matrix in seed_mats.items():
                furnessed_mat, *_ = furness.doubly_constrained_furness(
                    seed_vals=seed_matrix,
                    row_targets=row_targets,
                    col_targets=col_targets,
                    tol=self.furness_tol,
                    max_iters=self.furness_max_iters,
                    warning=self.warning,
                )

                # ## RETURN RESULTS ## #
                # Put the furnessed matrix in the return array
                self.putter_arrays[key].write_local_data(furnessed_mat)

            # Put the data back on the queues
            # Also lets threads know data is waiting
            for area_id in self.calib_area_keys:
                data = FurnessResults(
                    mat=self.area_mats[area_id],  # Return area_mat
                    # Not used anyway
                    completed_iters=0,
                    achieved_rmse=np.inf,
                )
                self.putter_qs[area_id].put(data)

        else:
            # Make sure the putter queues are empty
            for q in self.putter_qs.values():
                multithreading.empty_queue(q)


class JacobianFurnessThreadQueues(FurnessThreadBase):
    """Collects partial matrices and runs a furness

    Uses its getter_qs to wait for partial matrix inputs. Waits for all
    partial matrices, adds them together, and runs a furness.
    Splits out the furnessed matrix and returns the partial matrices.
    """
    def __init__(self,
                 getter_qs: Dict[Any, queue.Queue],
                 putter_qs: Dict[Any, queue.Queue],
                 *args,
                 **kwargs,
                 ):
        """
        Parameters
        ----------
        getter_qs:
            A nested dictionary of Queues for each area_id.
            Furness will be run once data has been received from all queues.

        putter_qs:
            A dictionary of Queues for each area_id. Queues are used to pass
            info back on the completed furness.

        *args, **kwargs:
            Arguments to be passed to parent FurnessThreadBase class

        See Also
        --------
        `FurnessThreadBase`
        """
        super().__init__(
            *args,
            **kwargs,
        )

        # Assign attributes
        self.getter_qs = getter_qs
        self.putter_qs = putter_qs

    def get_furness_data(self):
        """Grabs the needed data for the furness to run

        Returns
        -------
        seed_mats:
            A dictionary of retrieved partial seed matrices that need to be
            combined to create the full seed matrix for the furness.

        row_targets:
            The row targets to be used for the furness.
            i.e the target of np.sum(furnessed_matrix, axis=1)

        col_targets:
            The col targets to be used for the furness.
            i.e the target of np.sum(furnessed_matrix, axis=0)

        ignore_threads:
            Dictionary of booleans. True if callers wants to ignore this
            run of the Jacobian. Used to optimise runs as the
            Jacobian will not be run if none of the callers
            care about the result.
        """
        # Get all the data
        partial_furness_requests = multithreading.get_data_from_queue_dict(self.getter_qs)

        # Combine individual requests
        seed_mat_list = list()
        row_targets_list = list()
        col_targets_list = list()
        ignore_threads = dict.fromkeys(self.getter_qs.keys())
        for thread_id, request in partial_furness_requests.items():
            seed_mat_list.append(request.seed_mat)
            row_targets_list.append(request.row_targets)
            col_targets_list.append(request.col_targets)
            ignore_threads[thread_id] = request.ignore_result

        # Combine individual items
        seed_mat = functools.reduce(operator.add, seed_mat_list)
        row_targets = functools.reduce(operator.add, row_targets_list)
        col_targets = functools.reduce(operator.add, col_targets_list)

        return seed_mat, row_targets, col_targets, ignore_threads

    def run_furness(self) -> None:
        """Runs a furness once all data received, and passes data back

        Runs forever - therefore needs to be a daemon.
        Overrides parent to run this on thread start.

        Returns
        -------
        None
        """
        # ## GET DATA ## #
        seed_mat, row_targets, col_targets, ignore_threads = self.get_furness_data()

        # Only run and return data if any threads care about the result
        if not all(ignore_threads.values()):
            # ## FURNESS ## #
            furnessed_mat, iters, rmse = furness.doubly_constrained_furness(
                seed_vals=seed_mat,
                row_targets=row_targets,
                col_targets=col_targets,
                tol=self.furness_tol,
                max_iters=self.furness_max_iters,
                warning=self.warning,
            )

            # ## RETURN RESULTS ## #
            # Split back out into areas and return
            for area_id in self.calib_area_keys:
                data = FurnessResults(
                    mat=furnessed_mat * self.area_mats[area_id],
                    completed_iters=iters,
                    achieved_rmse=rmse,
                )
                self.putter_qs[area_id].put(data)

        # Empty out any queues that we are ignoring - stop build up of items
        for area_id, ignore in ignore_threads.items():
            if ignore:
                multithreading.empty_queue(self.putter_qs[area_id])


class SingleTLDCalibratorThreadBase(multithreading.ReturnOrErrorThread, GravityModelBase):
    """Calibrate Gravity Model params for a single TLD

    Used internally in MultiAreaGravityModelCalibrator. Each TLD is split out
    with its data, then handed over to one of these threads to find the
    optimal params alongside one another.
    """
    def __init__(self,
                 cost_function: cost.CostFunction,
                 cost_matrix: np.ndarray,
                 target_cost_distribution: pd.DataFrame,
                 target_convergence: float,
                 init_params: Dict[str, Any],
                 thread_complete_event: threading.Event,
                 all_done_event: threading.Event,
                 running_log_path: os.PathLike,
                 *args,
                 use_perceived_factors: bool = True,
                 estimate_init_params: bool = False,
                 calibrate_params: bool = True,
                 diff_step: float = 1e-8,
                 ftol: float = 1e-4,
                 xtol: float = 1e-4,
                 grav_max_iters: int = 100,
                 verbose: int = 0,
                 thread_name: str = None,
                 **kwargs,
                 ):
        # Call parent classes
        multithreading.ReturnOrErrorThread.__init__(  # type: ignore
            self,
            name=thread_name,
            *args,
            **kwargs,
        )
        # row and col targets aren't used in this implementation. See
        # self.gravity_furness for more info.
        GravityModelBase.__init__(
            self,
            cost_function=cost_function,
            cost_matrix=cost_matrix,
            target_cost_distribution=target_cost_distribution,
            running_log_path=running_log_path,
        )

        # Assign other attributes
        self.target_convergence = target_convergence
        self.init_params = init_params
        self.use_perceived_factors = use_perceived_factors
        self.estimate_init_params = estimate_init_params

        # optimize_params
        self.calibrate_params = calibrate_params
        self.diff_step = diff_step
        self.ftol = ftol
        self.xtol = xtol
        self.grav_max_iters = grav_max_iters
        self.verbose = verbose

        # Threading attributes
        self.thread_complete_event = thread_complete_event
        self.all_done_event = all_done_event

    def run_target(self) -> Dict[str, Any]:
        """Finds the optimal parameters for self.cost_function

        Optimal parameters are found using `scipy.optimize.least_squares`
        to fit the distributed row/col targets to
        self.target_cost_distribution. Once the optimal parameters are found,
        the gravity model is run one last time to check the
        self.target_convergence has been met.
        Overrides parent to run this on thread start.

        Returns
        -------
        optimal_params:
            Returns a dictionary of the same shape as self.init_params. The
            values will be the optimal cost parameters to get the best band
            share convergence for self.target_cost_distribution
        """
        # Validate init_params
        self.cost_function.validate_params(self.init_params)

        # Estimate what the initial params should be
        if self.estimate_init_params:
            init_params = self._estimate_init_params(
                init_params=self.init_params,
                target_cost_distribution=self.target_cost_distribution,
            )
        else:
            init_params = self.init_params

        # Figure out the optimal cost params
        self._calibrate(
            init_params=init_params,
            calibrate_params=self.calibrate_params,
            diff_step=self.diff_step,
            ftol=self.ftol,
            xtol=self.xtol,
            grav_max_iters=self.grav_max_iters,
            verbose=self.verbose,
        )
        self.thread_complete_event.set()

        # Once we have the optimal cost params, keep running until all
        # threads are complete
        while not self.all_done_event.is_set():
            self._jacobian_function(
                cost_args=self._order_cost_params(self.optimal_cost_params),
                diff_step=self.diff_step,
                ignore_result=True,
            )

            self._gravity_function(
                cost_args=self._order_cost_params(self.optimal_cost_params),
                diff_step=self.diff_step,
            )

        return self.optimal_cost_params

    def gravity_furness(self,
                        seed_matrix: np.ndarray,
                        ) -> Tuple[np.ndarray, int, float]:
        raise NotImplementedError(
            "When a class inherits from %s it needs to implement a method "
            "for gravity_furness()"
            % self.__class__.__name__
        )

    def jacobian_furness(self,
                         seed_matrices: Dict[str, np.ndarray],
                         row_targets: np.ndarray,
                         col_targets: np.ndarray,
                         ignore_result: bool = False,
                         ) -> Dict[str, np.ndarray]:
        raise NotImplementedError(
            "When a class inherits from %s it needs to implement a method "
            "for jacobian_furness()"
            % self.__class__.__name__
        )


class SingleTLDCalibratorThreadSharedArrays(SingleTLDCalibratorThreadBase):

    def __init__(
        self,
        gravity_putter_q: queue.Queue,
        gravity_getter_q: queue.Queue,
        gravity_putter_array: communication.SharedNumpyArrayHelper,
        gravity_getter_array: communication.SharedNumpyArrayHelper,
        jacobian_putter_q: queue.Queue,
        jacobian_getter_q: queue.Queue,
        jacobian_putter_array: Dict[str, communication.SharedNumpyArrayHelper],
        jacobian_getter_array: Dict[str, communication.SharedNumpyArrayHelper],
        *args,
        **kwargs,
    ):
        """
        Parameters
        ----------
        gravity_putter_q:
            A queue object that the gravity furness will use to let the
            furness handler know there is data in gravity_putter_array to collect.

        gravity_getter_q:
            A queue object that the furness handler will use to let the
            gravity furness know there is data in gravity_getter_array to collect.

        gravity_putter_array:
            A shared array between processes that will be used by the gravity
            furness to pass matrices over to be furnessed.

        gravity_getter_array:
            A shared array between processes that will be used by the gravity
            furness to receive furnessed matrices.

        jacobian_putter_q:
            A queue object that the jacobian furness will use to let the
            furness handler know there is data in jacobian_putter_array to collect.

        jacobian_getter_q:
            A queue object that the jacobian handler will use to let the
            gravity furness know there is data in jacobian_getter_array to collect.

        jacobian_putter_array:
            A shared array between processes that will be used by the jacobian
            furness to pass matrices over to be furnessed.

        jacobian_getter_array:
            A shared array between processes that will be used by the jacobian
            furness to receive furnessed matrices.

        args:
            Used to pass further arguments to `SingleTLDCalibratorThreadBase`

        kwargs:
            Used to pass further arguments to `SingleTLDCalibratorThreadBase`

        See Also
        --------
        `SingleTLDCalibratorThreadBase`
        """
        SingleTLDCalibratorThreadBase.__init__(self, *args, **kwargs)

        # Shared array specific arguments
        self.gravity_putter_q = gravity_putter_q
        self.gravity_getter_q = gravity_getter_q
        self.gravity_putter_array = gravity_putter_array
        self.gravity_getter_array = gravity_getter_array
        self.jacobian_putter_q = jacobian_putter_q
        self.jacobian_getter_q = jacobian_getter_q
        self.jacobian_putter_array = jacobian_putter_array
        self.jacobian_getter_array = jacobian_getter_array

    def gravity_furness(self,
                        seed_matrix: np.ndarray,
                        ) -> Tuple[np.ndarray, int, float]:
        """Runs a doubly constrained furness on the seed matrix

        Wrapper around furness.doubly_constrained_furness, using class
        attributes to set up the function call.

        Parameters
        ----------
        seed_matrix:
            Initial values for the furness.

        Returns
        -------
        furnessed_matrix:
            The final furnessed matrix

        completed_iters:
            The number of completed iterations before exiting

        achieved_rmse:
            The Root Mean Squared Error difference achieved before exiting
        """
        # Make sure receive is empty before sending
        if not self.gravity_getter_q.empty():
            raise nd.NormitsDemandError(
                "The gravity furness receive queue contained data before any "
                "data had been sent. "
                "The gravity threads must have come out of sync somewhere."
            )

        # ## SEND ## #
        # Add the data to the shared array - and mark queue
        self.gravity_putter_array.apply_local_data(seed_matrix, operator.add)
        self.gravity_putter_q.put(1)

        # ## RECEIVE ## #
        # Wait until we're told there is data to collect
        furness_data = multithreading.get_data_from_queue(self.gravity_getter_q)

        # Extract our chunk of the matrix
        furnessed_mat = self.gravity_getter_array.get_local_copy()
        furnessed_mat *= furness_data.mat

        return (
            furnessed_mat,
            furness_data.completed_iters,
            furness_data.achieved_rmse,
        )

    def jacobian_furness(self,
                         seed_matrices: Dict[str, np.ndarray],
                         row_targets: np.ndarray,
                         col_targets: np.ndarray,
                         ignore_result: bool = False,
                         ) -> Dict[str, np.ndarray]:
        """Runs a doubly constrained furness on the seed matrix

        Wrapper around furness.doubly_constrained_furness, to be used when
        running the furness withing the jacobian calculation.

        Parameters
        ----------
        seed_matrices:
            Dictionary of initial values for the furness.
            Keys are the name of the cost params which has been changed
            to get this new seed matrix.

        row_targets:
            The target values for the sum of each row.
            i.e np.sum(matrix, axis=1)

        col_targets:
            The target values for the sum of each column
            i.e np.sum(matrix, axis=0)

        ignore_result:
            Whether to ignore the return result or not. Useful when a Jacobian
            furness is only being called to satisfy other threads.

        Returns
        -------
        furnessed_matrix:
            The final furnessed matrix

        completed_iters:
            The number of completed iterations before exiting

        achieved_rmse:
            The Root Mean Squared Error difference achieved before exiting
        """
        # Make sure the receive queue is empty - might be data left
        # In there from other threads needing Jacobian
        multithreading.empty_queue(
            q=self.jacobian_getter_q,
            wait_for_items=True,
            wait_time=0.3,
        )

        # ## SEND ## #
        # Add this seeds to shared memory
        for cost_param, seed_matrix in seed_matrices.items():
            self.jacobian_putter_array[cost_param].apply_local_data(
                data=seed_matrix,
                operation=operator.add,
            )

        # Create a request and place on the queue
        request = PartialFurnessRequest(
            seed_mat=np.array([0]),
            row_targets=row_targets,
            col_targets=col_targets,
            ignore_result=ignore_result,
        )
        self.jacobian_putter_q.put(request)

        # ## RECEIVE ## #
        return_mats = dict.fromkeys(seed_matrices.keys())
        if not ignore_result:
            # Wait until we're told there is data to collect
            furness_data = multithreading.get_data_from_queue(self.jacobian_getter_q)

            # Extract our chunk of the matrix
            for cost_param, getter_array in self.jacobian_getter_array.items():
                furnessed_mat = getter_array.get_local_copy()
                furnessed_mat *= furness_data.mat
                return_mats[cost_param] = furnessed_mat
        else:
            # Just fill with 0s. Data doesn't matter
            for cost_param in return_mats.keys():
                return_mats[cost_param] = np.zeros_like(seed_matrices[cost_param])

        return return_mats


class SingleTLDCalibratorThreadQueues(SingleTLDCalibratorThreadBase):

    def __init__(
            self,
            gravity_putter_q: queue.Queue,
            gravity_getter_q: queue.Queue,
            jacobian_putter_q: queue.Queue,
            jacobian_getter_q: queue.Queue,
            *args,
            **kwargs,
    ):
        """
        Parameters
        ----------
        gravity_putter_q:
            A queue object that the gravity furness will use to pass the partial
            arrays over to be furnessed.

        gravity_getter_q:
            A queue object that the gravity furness will use to return the partial
            arrays once they have been furnessed.

        jacobian_putter_qs:
            A dictionary of queue objects that the jacobian furness will use
            to pass partial jacobian arrays over to be estimated. The keys are
            cost parameters.

        jacobian_getter_qs:
            A dictionary of queue objects that the jacobian furness will use
            to receive partial jacobian arrays once they have been estimated.
            The keys are cost parameters.

        args:
            Used to pass further arguments to `SingleTLDCalibratorThreadBase`

        kwargs:
            Used to pass further arguments to `SingleTLDCalibratorThreadBase`

        See Also
        --------
        `SingleTLDCalibratorThreadBase`
        """
        SingleTLDCalibratorThreadBase.__init__(self, *args, **kwargs)

        # Shared array specific arguments
        self.gravity_putter_q = gravity_putter_q
        self.gravity_getter_q = gravity_getter_q
        self.jacobian_putter_q = jacobian_putter_q
        self.jacobian_getter_q = jacobian_getter_q

    def gravity_furness(self,
                        seed_matrix: np.ndarray,
                        ) -> Tuple[np.ndarray, int, float]:
        """Runs a doubly constrained furness on the seed matrix

        Wrapper around furness.doubly_constrained_furness, using class
        attributes to set up the function call.

        Parameters
        ----------
        seed_matrix:
            Initial values for the furness.

        Returns
        -------
        furnessed_matrix:
            The final furnessed matrix

        completed_iters:
            The number of completed iterations before exiting

        achieved_rmse:
            The Root Mean Squared Error difference achieved before exiting
        """
        # Make sure receive is empty before sending
        if not self.gravity_getter_q.empty():
            raise nd.NormitsDemandError(
                "The gravity furness receive queue contained data before any "
                "data had been sent. "
                "The gravity threads must have come out of sync somewhere."
            )

        # Empty out the Jacobian queue if it has data
        # Would be from other threads needing Jacobian
        multithreading.empty_queue(
            q=self.jacobian_getter_q,
            wait_for_items=True,
            wait_time=0.3,
        )

        # ## SEND ## #
        # Add the data to the shared array - and mark queue
        self.gravity_putter_q.put(seed_matrix)

        # ## RECEIVE ## #
        # Collect the return data
        furness_data = multithreading.get_data_from_queue(self.gravity_getter_q)

        return (
            furness_data.mat,
            furness_data.completed_iters,
            furness_data.achieved_rmse,
        )

    def jacobian_furness(self,
                         seed_matrices: Dict[str, np.ndarray],
                         row_targets: np.ndarray,
                         col_targets: np.ndarray,
                         ignore_result: bool = False,
                         ) -> Dict[str, np.ndarray]:
        """Runs a doubly constrained furness on the seed matrix

        Wrapper around furness.doubly_constrained_furness, to be used when
        running the furness withing the jacobian calculation.

        Parameters
        ----------
        seed_matrices:
            Dictionary of initial values for the furness.
            Keys are the name of the cost params which has been changed
            to get this new seed matrix.

        row_targets:
            The target values for the sum of each row.
            i.e np.sum(matrix, axis=1)

        col_targets:
            The target values for the sum of each column
            i.e np.sum(matrix, axis=0)

        ignore_result:
            Whether to ignore the return result or not. Useful when a Jacobian
            furness is only being called to satisfy other threads.

        Returns
        -------
        furnessed_matrix:
            The final furnessed matrix

        completed_iters:
            The number of completed iterations before exiting

        achieved_rmse:
            The Root Mean Squared Error difference achieved before exiting
        """
        # Make sure the receive queue is empty - might be data left
        # In there from other threads needing Jacobian
        multithreading.empty_queue(
            q=self.jacobian_getter_q,
            wait_for_items=True,
            wait_time=0.3,
        )

        # Send data for each request individually
        return_mats = dict.fromkeys(seed_matrices.keys())
        for cost_param, seed_matrix in seed_matrices.items():
            # ## SEND ## #
            # Create a request and place on the queue
            request = PartialFurnessRequest(
                seed_mat=seed_matrix,
                row_targets=row_targets,
                col_targets=col_targets,
                ignore_result=ignore_result,
            )
            self.jacobian_putter_q.put(request)

            # ## RECEIVE ## #
            if not ignore_result:
                # Wait for data and collect
                furness_data = multithreading.get_data_from_queue(self.jacobian_getter_q)
                return_mats[cost_param] = furness_data.mat
            else:
                # Just return 0s. Values don't matter
                return_mats[cost_param] = np.zeros_like(seed_matrix)

        return return_mats


class MultiAreaGravityModelCalibrator:
    # TODO(BT): Write MultiAreaGravityModelCalibrator docs

    _ignore_calib_area_value = -1

    def __init__(self,
                 row_targets: np.ndarray,
                 col_targets: np.ndarray,
                 calibration_matrix: np.ndarray,
                 cost_function: cost.CostFunction,
                 cost_matrix: np.ndarray,
                 target_cost_distributions: Dict[Any, pd.DataFrame],
                 calibration_naming: Dict[Any, Any],
                 target_convergence: float,
                 furness_max_iters: int,
                 furness_tol: float,
                 running_log_path: os.PathLike,
                 use_perceived_factors: bool = True,
                 memory_optimised: bool = True,
                 ):
        # TODO(BT): Write MultiAreaGravityModelCalibrator __init__ docs
        # Set up logging
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
        self.furness_max_iters = furness_max_iters
        self.furness_tol = furness_tol
        self.use_perceived_factors = use_perceived_factors
        self.running_log_path = running_log_path
        self.memory_optimised = memory_optimised

        self.target_convergence = target_convergence

        # Ensure the calibration stuff was passed in correctly
        self.calibration_matrix = calibration_matrix
        calib_areas = list(np.unique(calibration_matrix))
        self.calib_areas = du.list_safe_remove(calib_areas, [self._ignore_calib_area_value])

        if not checks.all_keys_exist(calibration_naming, self.calib_areas):
            raise ValueError(
                "Calibration matrix needs to calibrate on %s\n"
                "However, names were only given for %s"
                % (self.calib_areas, calibration_naming.keys())
            )

        if not checks.all_keys_exist(target_cost_distributions, self.calib_areas):
            raise ValueError(
                "Calibration matrix needs to calibrate on %s\n"
                "However, target_cost_distributions were only given for %s"
                % (self.calib_areas, calibration_naming.keys())
            )

        self.calibration_naming = calibration_naming
        self.target_cost_distributions = self._update_tcds(target_cost_distributions)
        self.tcd_bin_edges = self._get_tcd_bin_edges(target_cost_distributions)

        # Additional attributes
        self.initial_cost_params = dict.fromkeys(self.calib_areas)
        self.initial_convergence = dict.fromkeys(self.calib_areas)
        self.optimal_cost_params = dict.fromkeys(self.calib_areas)
        self.achieved_convergence = dict.fromkeys(self.calib_areas)
        self.achieved_residuals = dict.fromkeys(self.calib_areas)
        self.achieved_full_distribution: np.ndarray = np.full_like(self.cost_matrix, -1)
        self.achieved_distribution = dict.fromkeys(self.calib_areas)

        # Attributes to store from runs
        self.achieved_band_share = dict.fromkeys(self.calib_areas)
        self.perceived_factors = dict.fromkeys(self.calib_areas)

    @staticmethod
    def _update_tcds(
        target_cost_distributions: Dict[Any, pd.DataFrame],
    ) -> Dict[Any, pd.DataFrame]:
        """Extrapolates data where needed"""
        # Init
        tcd_dict = dict.fromkeys(target_cost_distributions.keys())

        # Get the edges for each bin
        for key, tcd in target_cost_distributions.items():
            tcd_dict[key] = GravityModelBase._update_tcd(tcd)

        return tcd_dict

    @staticmethod
    def _get_tcd_bin_edges(
        target_cost_distributions: Dict[Any, pd.DataFrame],
    ) -> Dict[Any, pd.DataFrame]:
        """Gets the edges of each TCD band as a list"""
        # Init
        bin_edges = dict.fromkeys(target_cost_distributions.keys())

        # Get the edges for each bin
        for key, tcd in target_cost_distributions.items():
            bin_edges[key] = GravityModelBase._get_tcd_bin_edges(tcd)

        return bin_edges

    def _cost_params_to_kwargs(self, args: List[Any]) -> Dict[str, Any]:
        """Converts a list or args into kwargs that self.cost_function expects"""
        if len(args) != len(self.cost_function.kw_order):
            raise ValueError(
                "Received the wrong number of args to convert to cost function "
                "kwargs. Expected %s args, but got %s."
                % (len(self.cost_function.kw_order), len(args))
            )

        return dict(zip(self.cost_function.kw_order, args))

    def _order_cost_params(self, params: Dict[str, Any]) -> List[Any]:
        """Order params into a list that self.cost_function expects"""
        ordered_params = [0] * len(self.cost_function.kw_order)
        for name, value in params.items():
            index = self.cost_function.kw_order.index(name)
            ordered_params[index] = value

        return ordered_params

    def _setup_furness_threads_shared_arrays(
        self,
        shared_arrays: SharedArrays,
    ) -> FurnessSetup:
        """Sets up all the furness/jacobian threads for calibration runs"""
        # Init
        gravity_furness_key = 'furness'
        jacobian_key = 'jacobian'

        # Function to construct FurnessThreadInterface objects
        def create_interface_input(constructor):
            ret_val = dict()
            for name in [gravity_furness_key, jacobian_key]:
                nested_dict = dict.fromkeys(self.calib_areas)
                for key in nested_dict:
                    nested_dict[key] = constructor()
                ret_val[name] = nested_dict
            return ret_val

        # Use above function to create objects
        interface_putter_qs = create_interface_input(lambda: queue.Queue(1))
        interface_getter_qs = create_interface_input(lambda: queue.Queue(1))
        furness_return_qs = create_interface_input(lambda: queue.Queue(5))
        furness_wait_events = create_interface_input(lambda: threading.Event())

        # Generate the complete event
        all_complete_event = threading.Event()

        # Generate the area mats and complete events for each thread
        area_mats = dict.fromkeys(self.calib_areas)
        complete_events = dict.fromkeys(self.calib_areas)
        for area_id in self.calib_areas:
            area_mats[area_id] = self.calibration_matrix == area_id
            complete_events[area_id] = threading.Event()

        # Initialise the interface between gravity and furnesses
        furness_interface = FurnessThreadInterfaceSharedArrays(
            name='FurnessInterface',
            daemon=True,
            furness_wait_events=furness_wait_events,
            area_mats=area_mats,
            getter_qs=interface_getter_qs,
            putter_qs=interface_putter_qs,
            gravity_in_array=shared_arrays.gravity_in,
            jacobian_in_array=shared_arrays.jacobian_in,
            complete_events=complete_events,
            all_complete_event=all_complete_event,
            thread_ids=self.calib_areas,
            gravity_furness_key=gravity_furness_key,
            jacobian_key=jacobian_key,
        )
        furness_interface.start()

        # Initialise the central gravity furness thread
        gravity_furness = GravityFurnessThreadSharedArrays(
            name='Furness',
            daemon=True,
            row_targets=self.row_targets,
            col_targets=self.col_targets,
            getter_qs=interface_putter_qs[gravity_furness_key],
            putter_qs=furness_return_qs[gravity_furness_key],
            getter_array=shared_arrays.gravity_in,
            putter_array=shared_arrays.gravity_out,
            area_mats=area_mats,
            furness_tol=self.furness_tol,
            furness_max_iters=self.furness_max_iters,
            warning=True,
        )
        gravity_furness.start()

        # Initialise the central jacobian furness thread
        jacobian_furness = JacobianFurnessThreadSharedArrays(
            name='Jacobian',
            daemon=True,
            getter_qs=interface_putter_qs[jacobian_key],
            putter_qs=furness_return_qs[jacobian_key],
            getter_arrays=shared_arrays.jacobian_in,
            putter_arrays=shared_arrays.jacobian_out,
            area_mats=area_mats,
            furness_tol=1e-6,
            furness_max_iters=50,
            warning=False,
        )
        jacobian_furness.start()

        return FurnessSetup(
            area_mats=area_mats,
            gravity_putter_qs=interface_getter_qs[gravity_furness_key],
            gravity_getter_qs=furness_return_qs[gravity_furness_key],
            jacobian_putter_qs=interface_getter_qs[jacobian_key],
            jacobian_getter_qs=furness_return_qs[jacobian_key],
            complete_events=complete_events,
            all_complete_event=all_complete_event,
            furness_interface=furness_interface,
            gravity_furness=gravity_furness,
            jacobian_furness=jacobian_furness,
        )

    def _setup_furness_threads_queues(self) -> FurnessSetup:
        """Sets up all the furness/jacobian threads for calibration runs"""
        # Init
        gravity_furness_key = 'furness'
        jacobian_key = 'jacobian'

        # Function to construct FurnessThreadInterface objects
        def create_interface_input(constructor):
            ret_val = dict()
            for name in [gravity_furness_key, jacobian_key]:
                nested_dict = dict.fromkeys(self.calib_areas)
                for key in nested_dict:
                    nested_dict[key] = constructor()
                ret_val[name] = nested_dict
            return ret_val

        # Use above function to create objects
        interface_putter_qs = create_interface_input(lambda: queue.Queue(1))
        interface_getter_qs = create_interface_input(lambda: queue.Queue(1))
        furness_return_qs = create_interface_input(lambda: queue.Queue(5))
        furness_wait_events = create_interface_input(lambda: threading.Event())

        # Generate the complete event
        all_complete_event = threading.Event()

        # Generate the area mats and complete events for each thread
        area_mats = dict.fromkeys(self.calib_areas)
        complete_events = dict.fromkeys(self.calib_areas)
        for area_id in self.calib_areas:
            area_mats[area_id] = self.calibration_matrix == area_id
            complete_events[area_id] = threading.Event()

        # Initialise the interface between gravity and furnesses
        furness_interface = FurnessThreadInterfaceQueues(
            name='FurnessInterface',
            daemon=True,
            furness_wait_events=furness_wait_events,
            area_mats=area_mats,
            getter_qs=interface_getter_qs,
            putter_qs=interface_putter_qs,
            complete_events=complete_events,
            all_complete_event=all_complete_event,
            thread_ids=self.calib_areas,
            gravity_furness_key=gravity_furness_key,
            jacobian_key=jacobian_key,
        )
        furness_interface.start()

        # Initialise the central gravity furness thread
        gravity_furness = GravityFurnessThreadQueues(
            name='Furness',
            daemon=True,
            row_targets=self.row_targets,
            col_targets=self.col_targets,
            getter_qs=interface_putter_qs[gravity_furness_key],
            putter_qs=furness_return_qs[gravity_furness_key],
            area_mats=area_mats,
            furness_tol=self.furness_tol,
            furness_max_iters=self.furness_max_iters,
            warning=True,
        )
        gravity_furness.start()

        # Initialise the central jacobian furness thread
        jacobian_furness = JacobianFurnessThreadQueues(
            name='Jacobian',
            daemon=True,
            getter_qs=interface_putter_qs[jacobian_key],
            putter_qs=furness_return_qs[jacobian_key],
            area_mats=area_mats,
            furness_tol=1e-6,
            furness_max_iters=50,
            warning=False,
        )
        jacobian_furness.start()

        return FurnessSetup(
            area_mats=area_mats,
            gravity_putter_qs=interface_getter_qs[gravity_furness_key],
            gravity_getter_qs=furness_return_qs[gravity_furness_key],
            jacobian_putter_qs=interface_getter_qs[jacobian_key],
            jacobian_getter_qs=furness_return_qs[jacobian_key],
            complete_events=complete_events,
            all_complete_event=all_complete_event,
            furness_interface=furness_interface,
            gravity_furness=gravity_furness,
            jacobian_furness=jacobian_furness,
        )

    def _setup_shared_arrays(self,
                             init_mat: np.ndarray,
                             ctx_manager: contextlib.ExitStack,
                             ) -> SharedArrays:
        """Sets up the needed shared arrays"""
        # Init
        jac_base_name = 'Jac_%s_%s_%s' % ('%s', '%s', os.getpid())

        # Simplify creation
        def enter_array_context(name: str):
            array = communication.SharedNumpyArrayHelper(
                name=name,
                data=copy.copy(init_mat),
                dtype=np.float32,
            )
            return ctx_manager.enter_context(array)

            # Create the gravity furness arrays
        gravity_in = enter_array_context('Grav_in_%s' % os.getpid())
        gravity_out = enter_array_context('Grav_out_%s' % os.getpid())

        # Create the Jacobian furness objects - need one for each cost param
        jacobian_in = dict.fromkeys(self.cost_function.kw_order)
        jacobian_out = dict.fromkeys(self.cost_function.kw_order)
        for cost_param in self.cost_function.kw_order:
            jacobian_in[cost_param] = enter_array_context(jac_base_name % ('in', cost_param))
            jacobian_out[cost_param] = enter_array_context(jac_base_name % ('out', cost_param))

        return SharedArrays(
            gravity_in=gravity_in,
            gravity_out=gravity_out,
            jacobian_in=jacobian_in,
            jacobian_out=jacobian_out,
        )

    def _gravity_function(
        self,
        cost_param_dict: Dict[Any, Dict[str, Any]],
        perceived_factors: Dict[Any, np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[Any, GravityResults]]:
        """Runs a multi-TLD gravity model and returns the results

        Parameters
        ----------
        cost_param_dict:
            A dictionary of cost kwargs. The first key should be one of the
            values in self.calib_areas, and the values should be a kwargs
            dictionary to pass into self.cost_function.

        perceived_factors:
            A dictionary of perceived factors. These factors will be multiplied
            by the cost matrix for each zone to produce a matrix of
            "perceived costs" which are then handed over to the cost function
            in place of the raw cost matrix.

        Returns
        -------
        achieved_distribution:
            A full matrix of the achieved distribution across all areas.

        gravity_results_dict:
            A dictionary of GravityResults objects. The keys are the same as
            those passed in to cost_param_dict.
        """
        # Init
        seed_matrix = np.zeros_like(self.cost_matrix, dtype=float)

        # Check the given keys are valid
        given_keys = set(cost_param_dict.keys())
        valid_keys = set(self.calib_areas)
        invalid_keys = given_keys - valid_keys
        if len(invalid_keys) > 0:
            raise ValueError(
                "Invalid keys given in the cost_param_dict. The following "
                "keys have no area ID defined in "
                "self.calibration_matrix:\n%s"
                % invalid_keys
            )

        # Build the seed matrix
        for area_id, cost_params in cost_param_dict.items():
            # Extract the relevant cost
            area_bool = self.calibration_matrix == area_id
            area_cost = self.cost_matrix * area_bool

            # Calculate the seed matrix
            if perceived_factors is not None:
                area_cost *= perceived_factors[area_id]
            seed_matrix += self.cost_function.calculate(area_cost, **cost_params)

        # Furness trips to trip ends
        furnessed_matrix, iters, rmse = furness.doubly_constrained_furness(
            seed_vals=seed_matrix,
            row_targets=self.row_targets,
            col_targets=self.col_targets,
            tol=self.furness_tol,
            max_iters=self.furness_max_iters,
        )

        # Calculate the achieved results by each area
        results = dict.fromkeys(cost_param_dict.keys())
        for area_id in results:
            # Extract this area
            area_bool = self.calibration_matrix == area_id
            area_matrix = furnessed_matrix * area_bool
            area_cost = self.cost_matrix * area_bool

            # Convert matrix into an achieved distribution curve
            achieved_band_shares = cost_utils.calculate_cost_distribution(
                matrix=area_matrix,
                cost_matrix=area_cost,
                bin_edges=self.tcd_bin_edges[area_id],
            )

            # Evaluate this run
            target_band_shares = self.target_cost_distributions[area_id]['band_share'].values
            convergence = math_utils.curve_convergence(
                target=target_band_shares,
                achieved=achieved_band_shares,
            )
            achieved_residuals = target_band_shares - achieved_band_shares

            results[area_id] = GravityResults(
                band_share=achieved_band_shares,
                convergence=convergence,
                residuals=achieved_residuals,
                distribution=area_matrix,
                completed_iters=iters,
                achieved_rmse=rmse,
            )

        return furnessed_matrix, results

    def _calibrate_cpu_optimised(
        self,
        init_params: Dict[str, Any],
        estimate_init_params: bool,
        calibrate_params: bool,
        diff_step: float,
        ftol: float,
        xtol: float,
        grav_max_iters: int,
        verbose: int,
    ) -> Dict[Any, SingleTLDCalibratorThreadBase]:
        """Internal cpu optimised function of `self.calibrate()`

        See Also
        --------
        `self.calibrate()`
        """
        # Create the shared arrays for all to communicate
        init_mat = np.zeros_like(self.cost_matrix)
        with contextlib.ExitStack() as ctx_manager:
            shared_arrays = self._setup_shared_arrays(
                init_mat=init_mat,
                ctx_manager=ctx_manager,
            )

            # Set up the furness threads for gravity threads
            furness_setup = self._setup_furness_threads_shared_arrays(shared_arrays)

            # Start the gravity processes
            calibrator_threads = dict.fromkeys(self.calib_areas)
            for area_id in self.calib_areas:
                # Get just the costs for this area
                area_cost = self.cost_matrix * furness_setup.area_mats[area_id]

                # Set up where to put the logs
                dir_name, fname = os.path.split(self.running_log_path)
                area_dir_name = os.path.join(dir_name, self.calibration_naming[area_id])
                file_ops.create_folder(area_dir_name)
                area_running_log_path = os.path.join(area_dir_name, fname)

                # Replace the log if it already exists
                if os.path.isfile(area_running_log_path):
                    os.remove(area_running_log_path)

                # Start a thread to calibrate each area
                # TODO(BT): pass in objects rather than individual
                calibrator_threads[area_id] = SingleTLDCalibratorThreadSharedArrays(
                    thread_name=self.calibration_naming[area_id],
                    cost_function=self.cost_function,
                    cost_matrix=area_cost,
                    init_params=init_params,
                    estimate_init_params=estimate_init_params,
                    target_cost_distribution=self.target_cost_distributions[area_id],
                    target_convergence=self.target_convergence,
                    running_log_path=area_running_log_path,
                    gravity_putter_q=furness_setup.gravity_putter_qs[area_id],
                    gravity_getter_q=furness_setup.gravity_getter_qs[area_id],
                    gravity_putter_array=shared_arrays.gravity_in,
                    gravity_getter_array=shared_arrays.gravity_out,
                    jacobian_putter_array=shared_arrays.jacobian_in,
                    jacobian_getter_array=shared_arrays.jacobian_out,
                    jacobian_putter_q=furness_setup.jacobian_putter_qs[area_id],
                    jacobian_getter_q=furness_setup.jacobian_getter_qs[area_id],
                    thread_complete_event=furness_setup.complete_events[area_id],
                    all_done_event=furness_setup.all_complete_event,
                    calibrate_params=calibrate_params,
                    diff_step=diff_step,
                    ftol=ftol,
                    xtol=xtol,
                    grav_max_iters=grav_max_iters,
                    verbose=verbose,
                )
                calibrator_threads[area_id].start()

            multithreading.wait_for_thread_dict_return_or_error(
                return_threads=calibrator_threads,
                error_threads_list=furness_setup.all_threads,
            )

        return calibrator_threads

    def _calibrate_memory_optimised(
        self,
        init_params: Dict[str, Any],
        estimate_init_params: bool,
        calibrate_params: bool,
        diff_step: float,
        ftol: float,
        xtol: float,
        grav_max_iters: int,
        verbose: int,
    ) -> Dict[Any, SingleTLDCalibratorThreadBase]:
        """Internal memory optimised function of `self.calibrate()`

        See Also
        --------
        `self.calibrate()`
        """
        # ## SETUP FOR THREADS ## #
        # Set up the furness threads for gravity threads
        furness_setup = self._setup_furness_threads_queues()

        # ## START EACH THREAD ## #
        # Start the gravity processes
        calibrator_threads = dict.fromkeys(self.calib_areas)
        for area_id in self.calib_areas:
            # Get just the costs for this area
            area_cost = self.cost_matrix * furness_setup.area_mats[area_id]

            # Set up where to put the logs
            dir_name, fname = os.path.split(self.running_log_path)
            area_dir_name = os.path.join(dir_name, self.calibration_naming[area_id])
            file_ops.create_folder(area_dir_name)
            area_running_log_path = os.path.join(area_dir_name, fname)

            # Replace the log if it already exists
            if os.path.isfile(area_running_log_path):
                os.remove(area_running_log_path)

            # Start a thread to calibrate each area
            # TODO(BT): pass in objects rather than individual
            calibrator_threads[area_id] = SingleTLDCalibratorThreadQueues(
                thread_name=self.calibration_naming[area_id],
                cost_function=self.cost_function,
                cost_matrix=area_cost,
                init_params=init_params,
                estimate_init_params=estimate_init_params,
                target_cost_distribution=self.target_cost_distributions[area_id],
                target_convergence=self.target_convergence,
                running_log_path=area_running_log_path,
                gravity_putter_q=furness_setup.gravity_putter_qs[area_id],
                gravity_getter_q=furness_setup.gravity_getter_qs[area_id],
                jacobian_putter_q=furness_setup.jacobian_putter_qs[area_id],
                jacobian_getter_q=furness_setup.jacobian_getter_qs[area_id],
                thread_complete_event=furness_setup.complete_events[area_id],
                all_done_event=furness_setup.all_complete_event,
                calibrate_params=calibrate_params,
                diff_step=diff_step,
                ftol=ftol,
                xtol=xtol,
                grav_max_iters=grav_max_iters,
                verbose=verbose,
            )
            calibrator_threads[area_id].start()

        multithreading.wait_for_thread_dict_return_or_error(
            return_threads=calibrator_threads,
            error_threads_list=furness_setup.all_threads,
        )

        return calibrator_threads

    def calibrate(
        self,
        init_params: Dict[str, Any],
        estimate_init_params: bool = False,
        calibrate_params: bool = True,
        diff_step: float = 1e-8,
        ftol: float = 1e-4,
        xtol: float = 1e-4,
        grav_max_iters: int = 100,
        verbose: int = 0,
    ) -> Dict[Any, Dict[str, Any]]:
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
        self.optimal_cost_params

        As this is a multiple-area gravity model, multiple gravity model
        calibrations are used. One for each area in self.calibration_matrix.
        The optimal cost parameters are calibrated for each area
        simultaneously, and they all share a furness during calibration.

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
            Returns a dictionary of dictionaries. The keys are the unique
            values from self.calibration matrix, and the values are of the
            same shape as init_params. The values of the nested dictionaries
            will be the optimal cost parameters to get the best band share
            convergence. (With perceived factors applied, if they are being
            used)

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

        # Assign the initial cost params
        for key in self.initial_cost_params:
            self.initial_cost_params[key] = init_params.copy()

        # Choose whether to run memory or CPU optimised
        if self.memory_optimised:
            calibrators = self._calibrate_memory_optimised(
                init_params=init_params,
                estimate_init_params=estimate_init_params,
                calibrate_params=calibrate_params,
                diff_step=diff_step,
                ftol=ftol,
                xtol=xtol,
                grav_max_iters=grav_max_iters,
                verbose=verbose,
            )
        else:
            calibrators = self._calibrate_cpu_optimised(
                init_params=init_params,
                estimate_init_params=estimate_init_params,
                calibrate_params=calibrate_params,
                diff_step=diff_step,
                ftol=ftol,
                xtol=xtol,
                grav_max_iters=grav_max_iters,
                verbose=verbose,
            )

        # Save the optimal cost params for each area
        for area_id in self.calib_areas:
            optimal_params = calibrators[area_id].optimal_cost_params
            perceived_factors = calibrators[area_id]._perceived_factors
            self.optimal_cost_params[area_id] = optimal_params
            self.perceived_factors[area_id] = perceived_factors

        # Run an optimal version of the gravity - store convergences
        matrix, results = self._gravity_function(self.initial_cost_params)
        for area_id in self.initial_convergence:
            self.initial_convergence[area_id] = results[area_id].convergence

        # Do a run with optimal_params, so we know what we achieved
        # TODO(BT): Add a log of this final run to each log - or new log?
        matrix, results = self._gravity_function(
            cost_param_dict=self.optimal_cost_params,
            perceived_factors=self.perceived_factors,
        )

        # Assign all of the achieved results
        self.achieved_full_distribution = matrix
        for area_id in self.calib_areas:
            self.achieved_band_share[area_id] = results[area_id].band_share
            self.achieved_convergence[area_id] = results[area_id].convergence
            self.achieved_residuals[area_id] = results[area_id].residuals
            self.achieved_distribution[area_id] = results[area_id].distribution

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
