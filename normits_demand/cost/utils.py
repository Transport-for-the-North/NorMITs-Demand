# -*- coding: utf-8 -*-
"""
Created on: 10/11/2021
Updated on:

Original author: Ben Taylor
Last update made by:
Other updates made by:

File purpose:

"""
# Built-Ins
from typing import List
from typing import Dict

# Third Party
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter
from matplotlib.ticker import AutoMinorLocator

# Local Imports
import normits_demand as nd


def cells_in_bounds(min_bounds: np.ndarray,
                    max_bounds: np.ndarray,
                    cost: np.ndarray,
                    ) -> np.ndarray:
    cell_counts = list()
    for min_val, max_val in zip(min_bounds, max_bounds):
        band_mask = (cost >= min_val) & (cost < max_val)
        cell_counts.append(band_mask.sum())
    return cell_counts


def iz_infill_costs(cost: pd.DataFrame,
                    iz_infill: float,
                    min_axis: int = 1,
                    ) -> pd.DataFrame:
    """
    Infills the diagonal with iz_infill * min_axis val for each item in axis

    Parameters
    ----------
    cost:
        The cost to infill.

    iz_infill:
        whether to add a value half the minimum
        interzonal value to the intrazonal cells. Currently needed for distance
        but not cost.

    min_axis:
        The axis to get the minimum value across

    Returns
    -------
    infilled_cost:
        cost, but with the diagonal infilled.
    """
    # Init
    infilled_cost = cost.values.copy()

    # Set to inf so we don't pick up 0s or diagonal in min
    infilled_cost = np.where(infilled_cost == 0, np.inf, infilled_cost)
    np.fill_diagonal(infilled_cost, np.inf)

    # Find the min an do infill
    min_vals = infilled_cost.min(axis=min_axis)
    infill = min_vals * iz_infill
    np.fill_diagonal(infilled_cost, infill)

    # Flip all inf back to 0
    infilled_cost = np.where(infilled_cost == np.inf, 0, infilled_cost)

    return pd.DataFrame(
        data=infilled_cost,
        index=cost.index,
        columns=cost.columns,
    )


def calculate_cost_distribution(matrix: np.ndarray,
                                cost_matrix: np.ndarray,
                                min_bounds: List[float] = None,
                                max_bounds: List[float] = None,
                                bin_edges: List[float] = None,
                                ) -> np.ndarray:
    """
    Calculates the band share distribution of matrix.

    Parameters
    ----------
    matrix:
        The matrix to calculate the cost distribution for. This matrix
        should be the same shape as cost_matrix

    cost_matrix:
        A matrix of cost relating to matrix. This matrix
        should be the same shape as matrix

    min_bounds:
        A list of minimum bounds for each edge of a distribution band.
        Corresponds to max_bounds.

    max_bounds:
        A list of maximum bounds for each edge of a distribution band.
        Corresponds to min_bounds.

    bin_edges:
        Defines a monotonically increasing array of bin edges, including the
        rightmost edge, allowing for non-uniform bin widths. This argument
        is passed straight into `numpy.histogram`

    Returns
    -------
    cost_distribution:
        a numpy array of distributed costs, where the bands are equivalent
        to min/max values in self.target_cost_distribution

    See Also
    --------
    `numpy.histogram`
    """
    # Use bounds to calculate bin edges
    if bin_edges is None:
        if min_bounds is None or max_bounds is None:
            raise ValueError(
                "Either bin_edges needs to be set, or both min_bounds and "
                "max_bounds needs to be set."
            )

        bin_edges = [min_bounds[0]] + max_bounds

    # Sort into bins
    distribution, _ = np.histogram(
        a=cost_matrix,
        bins=bin_edges,
        weights=matrix,
    )

    # Normalise
    if distribution.sum() == 0:
        return np.zeros_like(distribution)
    else:
        return distribution / distribution.sum()


def plot_cost_distribution(target_x: List[float],
                           target_y: List[float],
                           achieved_x: List[float],
                           achieved_y: List[float],
                           convergence: float,
                           cost_params: Dict[str, float],
                           plot_title: str,
                           path: nd.PathLike = None,
                           close_plot: bool = True,
                           **save_kwargs,
                           ):

    # Init
    figure, axis = plt.subplots()

    # Plot the target data
    label = 'Target | [R2=%.4f]' % convergence
    axis.plot(target_x, target_y, 'r--', label=label)

    # Plot the achieved data
    label = 'Achieved |'
    for name, value in cost_params.items():
        label += ' %s=%.2f' % (name, value)

    axis.plot(achieved_x, achieved_y, 'b-', label=label)

    # Label the plot
    axis.set_xlabel('Distance (km)')
    axis.set_ylabel('Band Share (%)')
    axis.set_title(plot_title)
    axis.legend()

    # Format the plot
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.set_ylim(0, None)
    axis.set_xlim(0, None)
    axis.yaxis.set_minor_locator(AutoMinorLocator())
    axis.xaxis.set_minor_locator(AutoMinorLocator())
    axis.grid(which="major")
    axis.grid(which="minor", ls=":")

    # Save if a path is given
    if path is not None:
        # plt.savefig(path, **save_kwargs)
        plt.show()

    # Return plot if not closing
    if not close_plot:
        return figure, axis
    else:
        plt.close(figure)
        return None, None


def calculate_average_cost_in_bounds(min_bounds: np.ndarray,
                                     max_bounds: np.ndarray,
                                     cost_matrix: np.ndarray,
                                     trips: np.ndarray,
                                     ) -> np.ndarray:
    """Calculates the average cost between each bounds pair

    Parameters
    ----------
    min_bounds:
        The minimum bounds for each cost band. Corresponds to max_bounds.

    max_bounds:
        The maximum bounds for each cost band. Corresponds to min_bounds.

    cost_matrix:
        A matrix of costs from each point to point. Corresponds to trips.

    trips:
        A matrix of trip counts from each point to point. Corresponds to
        cost_matrix.

    Returns
    -------
    average_costs:
         An array of the average cost between each bounds pair
    """
    average_costs = list()
    for min_val, max_val in zip(min_bounds, max_bounds):
        band_mask = (cost_matrix >= min_val) & (cost_matrix < max_val)
        band_distance = (trips * band_mask * cost_matrix).sum()
        band_trips = (trips * band_mask).sum()

        if band_trips == 0:
            band_average = min_val
        else:
            band_average = band_distance / band_trips

        average_costs.append(band_average)

    return np.array(average_costs)
