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
import os
import dataclasses

from typing import List
from typing import Tuple
from typing import Union

# Third Party
import numpy as np
import pandas as pd

import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import AutoMinorLocator

# Local Imports
import normits_demand as nd

sns.set_theme(style="darkgrid")


@dataclasses.dataclass
class DistributionReportCols:
    cost_units: str

    min: str = "Min"
    max: str = "Max"
    mid: str = "Mid"
    target_ave_cost: str = "Target Average Cost"
    achieved_ave_cost: str = "Achieved Average Cost"
    target_band_share: str = "Target Band Share"
    achieved_band_share: str = "Achieved Band Share"

    achieved_band_count: str = "Achieved Band Count"
    cell_count: str = "Cell Count"
    cell_proportion: str = "Cell Proportion"
    convergence: str = "Convergence"

    def __post_init__(self):
        # Attach cost_units where relevant
        if self.cost_units is None:
            return

        relevant_cols = [
            self.min,
            self.max,
            self.mid,
            self.target_ave_cost,
            self.achieved_ave_cost,
        ]

        for col in relevant_cols:
            col += f" ({self.cost_units})"

    def col_order(self):
        """Return a list of columns, in their expected output order"""
        order = list(dataclasses.astuple(self))
        order.remove(self.cost_units)
        return order


@dataclasses.dataclass
class PlotData:
    """Packages up plot data for easy access"""
    x_values: np.ndarray
    y_values: np.ndarray
    label: str


def cells_in_bounds(
    min_bounds: np.ndarray,
    max_bounds: np.ndarray,
    cost: np.ndarray,
) -> np.ndarray:
    cell_counts = list()
    for min_val, max_val in zip(min_bounds, max_bounds):
        band_mask = (cost >= min_val) & (cost < max_val)
        cell_counts.append(band_mask.sum())
    return np.array(cell_counts)


def iz_infill_costs(
    cost: pd.DataFrame,
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


def normalised_cost_distribution(
    matrix: np.ndarray,
    cost_matrix: np.ndarray,
    min_bounds: Union[List[float], np.ndarray] = None,
    max_bounds: Union[List[float], np.ndarray] = None,
    bin_edges: Union[List[float], np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the normalised distribution of costs across a matrix.

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

    normalised_cost_distribution:
        Similar to `cost_distribution`, however the values in each band
        have been normalised to sum to 1.

    See Also
    --------
    `numpy.histogram`
    """
    distribution = cost_distribution(
        matrix=matrix,
        cost_matrix=cost_matrix,
        min_bounds=min_bounds,
        max_bounds=max_bounds,
        bin_edges=bin_edges,
    )

    # Normalise
    if distribution.sum() == 0:
        normalised = np.zeros_like(distribution)
    else:
        normalised = distribution / distribution.sum()

    return distribution, normalised


def cost_distribution(
    matrix: np.ndarray,
    cost_matrix: np.ndarray,
    min_bounds: Union[List[float], np.ndarray] = None,
    max_bounds: Union[List[float], np.ndarray] = None,
    bin_edges: Union[List[float], np.ndarray] = None,
) -> np.ndarray:
    """
    Calculates the distribution of costs across a matrix.

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
    distribution:
        sum of trips by distance band

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

    return distribution


def _get_cutoff_idx(lst: np.ndarray, cutoff: float) -> int:
    """Get the index of the cutoff point in lst

    Returns the index of the first value in lst that is
    less than or equal to cutoff.

    If no values are less then the cutoff then the index of the final
    value in lst is returned
    """
    # Init
    i = 0

    # Loop until we pass the cutoff
    for i, item in enumerate(reversed(lst)):  # type: ignore
        if item > cutoff:
            break

    # Be careful when flipping edges
    if i == 0:
        return -1

    if i == len(lst) - 1:
        return 0

    # Flip the index to the forwards list
    return -i


def plot_cost_distributions(
    plot_data: Union[PlotData, List[PlotData]],
    plot_title: str,
    band_share_cutoff: float = 0,
    aspect_ratio: float = 9 / 16,
    path: os.PathLike = None,
    close_plot: bool = True,
    x_axis_label: str = None,
    y_axis_label: str = None,
    percent_data: bool = False,
    **save_kwargs,
) -> None:
    """Plots cost distributions onto a graph

    Parameters
    ----------
    plot_data:
        A list of the data to plot

    plot_title:
        Title to label the plot with

    band_share_cutoff:
        The value on the y-axis that all data points must hit before no more
        data is shown on the x-axis.

    aspect_ratio:
        Aspect ratio to give the output graph. This is the ratio of the height
        of the graph to its width. i.e. A value of 2 means the graph will be
        twice as tall as it is wide.

    path:
        The path to output the graph to. Passed straight to matplotlib
        to save. For more detail:
        https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html?highlight=savefig#matplotlib-pyplot-savefig

    close_plot:
        Whether to close the plot before returning. If True,
        `matplotlib.pyplot.close()` is called.

    x_axis_label:
        The label to give to the x axis in the output graph

    y_axis_label:
        The label to give to the y axis in the output graph

    percent_data:
        Whether the y-axis values are percentages or not. If True, a percentage
        formatting is added to the y axis

    save_kwargs:
        Any additional kwargs to pass through when saving the graph. See:
        https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html?highlight=savefig#matplotlib-pyplot-savefig

    Returns
    -------
    None

    See Also
    --------
    `matplotlib.pyplot.savefig`
    """
    # Init
    plt.clf()
    plot_data = [plot_data] if isinstance(plot_data, PlotData) else plot_data

    if len(plot_data) < 0:
        return
    
    # Plot each data chunk in turn
    cutoff_vals = list()
    for data in plot_data:
        # Plot data
        axis = sns.lineplot(x=data.x_values, y=data.y_values, label=data.label)

        # Figure out graph cutoff
        if band_share_cutoff > 0:
            cutoff_idx = _get_cutoff_idx(data.y_values, band_share_cutoff)
            cutoff_vals.append(data.x_values[cutoff_idx + 1])
        else:
            cutoff_vals.append(np.nan)

    # Finalise cutoff point
    upper_x_lim = max(cutoff_vals)
    if np.isnan(upper_x_lim) or np.isinf(upper_x_lim):
        upper_x_lim = None      # type: ignore

    # Label the plot
    if x_axis_label is not None:
        axis.set_xlabel(x_axis_label)
    if y_axis_label is not None:
        axis.set_ylabel(y_axis_label)
    axis.set_title(plot_title)

    # Format the plot
    if percent_data:
        axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    else:
        axis.yaxis.set_major_formatter(ScalarFormatter())

    plt.legend(loc="upper right")
    axis.set_ylim(0, None)
    axis.set_xlim(0, upper_x_lim)
    axis.yaxis.set_minor_locator(AutoMinorLocator())
    axis.xaxis.set_minor_locator(AutoMinorLocator())
    axis.tick_params(which="both", bottom=True)
    axis.grid(which="minor", ls=":")

    # Set the aspect ratio
    x_left, x_right = axis.get_xlim()
    y_low, y_high = axis.get_ylim()
    axis.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * aspect_ratio)

    # Save if a path is given
    if path is not None:
        plt.savefig(path, **save_kwargs)

    # Clear plot, unless told otherwise
    if close_plot:
        plt.clf()


def calculate_average_cost_in_bounds(
    min_bounds: np.ndarray,
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


def get_band_mid_points(
    min_bounds: np.ndarray,
    max_bounds: np.ndarray,
) -> np.ndarray:
    """Calculates the mid point in each of the band bounds

    Parameters
    ----------
    min_bounds:
        The minimum bounds for each cost band. Corresponds to max_bounds.

    max_bounds:
        The maximum bounds for each cost band. Corresponds to min_bounds.

    Returns
    -------
    bound_mid_points:
        The mid points of each of the bands defined by min_bounds
        and max_bounds.
    """
    mid_points = list()
    for min_val, max_val in zip(min_bounds, max_bounds):
        mid_points.append((min_val + max_val) / 2)
    return np.array(mid_points)
