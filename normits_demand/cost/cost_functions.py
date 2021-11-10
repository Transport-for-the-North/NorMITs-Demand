# -*- coding: utf-8 -*-
"""
Created on: 05/11/2021
Updated on:

Original author: Ben Taylor
Last update made by:
Other updates made by:

File purpose:

"""
# Built-Ins
import enum
import inspect

from typing import Any
from typing import Dict
from typing import Tuple
from typing import Callable

# Third Party
import numpy as np

# Local Imports
import normits_demand as nd

from normits_demand.utils import math_utils


@enum.unique
class BuiltInCostFunction(enum.Enum):
    TANNER = 'tanner'
    LOG_NORMAL = 'log_normal'

    def get_cost_function(self):

        if self == BuiltInCostFunction.TANNER:
            params = {'alpha': [-5, 5], 'beta': [-5, 5]}
            function = tanner

        elif self == BuiltInCostFunction.LOG_NORMAL:
            params = {'sigma': [0, 5], 'mu': [0, 10]}
            function = log_normal

        else:
            raise nd.NormitsDemandError(
                "No definition exists for %s built in cost function"
                % self
            )

        return CostFunction(
            name=self.name,
            params=params,
            function=function,
        )


class CostFunction:
    """Abstract Class defining how cost function classes should look.

    If a new cost function is needed, then a new class needs to be made
    which inherits this abstract class.
    """

    def __init__(self,
                 name: str,
                 params: Dict[str, Tuple[float, float]],
                 function: Callable,
                 ):
        self.name = name
        self.function = function

        # Split params
        self.param_names = list(params.keys())
        self.param_min = {k: min(v) for k, v in params.items()}
        self.param_max = {k: max(v) for k, v in params.items()}

        self.kw_order = list(inspect.signature(self.function).parameters.keys())[1:]

        # Validate the params and cost function
        try:
            self.function(np.array(1e-2), **self.param_max)
        except TypeError:
            raise ValueError(
                "Received a TypeError while testing the given params "
                "definition and cost function will work together. Have the "
                "params been defined correctly for the given function?\n"
                "Tried passing in '%s' to function %s."
                % (self.param_names, self.function)
            )

    def validate_params(self, param_dict: Dict[str, Any]) -> None:
        """
        Validates that the given values are valid and within min/max ranges

        Validates that the param dictionary given contains only and all
        expected parameter names as keys, and that the values for each key
        fall within the acceptable parameter ranges.

        Parameters
        ----------
        param_dict:
            A dictionary of values to validate. Should be in
            {param_name: param_value} format.
            
        Raises
        ------
        ValueError:
            If any of the given params do not have valid name, or their values
            fall outside the min/max range defined in this class.
        """
        # Init
        # math_utils.check_numeric(param_dict)

        # Validate
        for name, value in param_dict.items():
            # Check name is valid
            if name not in self.param_names:
                raise ValueError(
                    "Parameter '%s' is not a valid parameter for "
                    "CostFunction %s"
                    % (name, self.name)
                )

            # Check values are valid
            min_val = self.param_min[name]
            max_val = self.param_max[name]

            if value < min_val or value > max_val:
                raise ValueError(
                    "Parameter '%s' falls outside the acceptable range of "
                    "values. Value must be between %s and %s. Got %s."
                    % (name, min_val, max_val, value)
                )

            if value > self.param_max[name]:
                raise ValueError()

    def calculate(self, base_cost: np.ndarray, **kwargs) -> np.ndarray:
        """
        Calculates the actual cost using self.function

        Before calling the cost function the given cost function params will
        be checked that they are within the min and max values passed in when
        creating the object. The cost function will then be called and the
        value returned.

        Parameters
        ----------
        base_cost:
            Array of the base costs.

        kwargs:
        Parameters of the cost function to pass to self.function.

        Returns
        -------
        costs:
            Output from self.function, same shape as `base_cost`.

        Raises
        ------
        ValueError:
            If the given cost function params are outside the min/max range
            for this class.
        """
        self.validate_params(kwargs)
        return self.function(base_cost, **kwargs)


def tanner(base_cost: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    r"""Implementation of the tanner cost function.

    Parameters
    ----------
    base_cost : np.ndarray
        Array of the base costs.

    alpha, beta : float
        Parameters of the tanner cost function, see Notes.

    Returns
    -------
    tanner_costs:
        Output from the tanner equation, same shape as `base_cost`.

    Notes
    -----
    Formula used for this function is:

    .. math:: f(C_{ij}) = C_{ij}^\alpha \cdot \exp(\beta C_{ij})

    where:

    - :math:`C_{ij}`: cost from i to k.
    - :math:`\alpha, \beta`: calibration parameters.
    """
    math_utils.check_numeric({'alpha': alpha, 'beta': beta})

    # Don't do 0 to the power in case alpha is negative
    # 0^x where x is anything (other than 0) is always 0
    power = np.float_power(
        base_cost,
        alpha,
        out=np.zeros_like(base_cost, dtype=float),
        where=base_cost != 0,
    )
    exp = np.exp(beta * base_cost)
    return power * exp


def log_normal(base_cost: np.ndarray, sigma: float, mu: float) -> np.ndarray:
    r"""Implementation of the log normal cost function.

    Parameters
    ----------
    base_cost : np.ndarray
        Array of the base costs.

    sigma, mu : float
        Parameters of the log normal cost function, see Notes.

    Returns
    -------
    log_normal_costs:
        Output from the log normal equation, same shape as `base_cost`.

    Notes
    -----
    Formula used for this function is:

    .. math::

        f(C_{ij}) = \frac{1}{C_{ij} \cdot \sigma \cdot \sqrt{2\pi}}
        \cdot \exp\left(-\frac{(\ln C_{ij}-\mu)^2}{2\sigma^2}\right)

    where:

    - :math:`C_{ij}`: cost from i to j.
    - :math:`\sigma, \mu`: calibration parameters.
    """
    # Init
    math_utils.check_numeric({'sigma': sigma, 'mu': mu})
    sigma = float(sigma)
    mu = float(mu)

    # We need to be careful to avoid 0 in costs
    # First calculate the fraction
    frac_denominator = (base_cost * sigma * np.sqrt(2 * np.pi))
    frac = np.divide(
        1,
        frac_denominator,
        where=frac_denominator != 0,
        out=np.zeros_like(frac_denominator),
    )

    # Now calculate the exponential
    log = np.log(
        base_cost,
        where=base_cost != 0,
        out=np.zeros_like(base_cost),
    )
    exp_numerator = (log - mu) ** 2
    exp_denominator = 2 * sigma ** 2
    exp = np.exp(exp_numerator / exp_denominator)

    return frac * exp
