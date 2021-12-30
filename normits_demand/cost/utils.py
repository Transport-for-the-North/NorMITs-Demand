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

# Third Party
import numpy as np
import pandas as pd

# Local Imports


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
                                min_bounds: List[float],
                                max_bounds: List[float],
                                ) -> np.ndarray:
    """
    Calculates the band share distribution of matrix.

    Uses the bounds supplied in target_cost_distribution, and the costs in
    self.costs to calculate the equivalent band shares in matrix.

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

    Returns
    -------
    cost_distribution:
        a numpy array of distributed costs, where the bands are equivalent
        to min/max values in self.target_cost_distribution
    """
    # Init
    total_trips = matrix.sum()

    # Calculate band shares
    distribution = list()
    for min_val, max_val in zip(min_bounds, max_bounds):
        cost_mask = (cost_matrix >= min_val) & (cost_matrix < max_val)
        band_trips = (matrix * cost_mask).sum()
        band_share = band_trips / total_trips
        distribution.append(band_share)

    return np.array(distribution)

