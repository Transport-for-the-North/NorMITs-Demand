# -*- coding: utf-8 -*-
"""
Created on: 21/06/2022
Updated on:

Original author: Ben Taylor
Last update made by:
Other updates made by:

File purpose:

"""
from __future__ import annotations

# Built-Ins
import os

from typing import Any

# Third Party
import numpy as np
import pandas as pd

# Local Imports
# pylint: disable=import-error,wrong-import-position
from normits_demand import core as nd_core
from normits_demand.utils import file_ops
from normits_demand.cost import utils as cost_utils
# pylint: enable=import-error,wrong-import-position

# TODO(BT): Build class to handle a segmentationLevel collection of these


class CostDistribution:
    """Stores and manipulates cost distributions"""

    def __init__(
        self,
        edges: np.ndarray,
        band_trips: np.ndarray,
        cost_units: nd_core.CostUnits,
        band_mean_cost: np.ndarray = None,
    ):
        """
        Parameters
        ----------
        edges:
            The edges to use for each band in the distribution. E.g.
            `[1, 2, 3]` defines 2 bands: 1->2 and 2->3

        band_trips:
            The number of trips in each band. This array should be one shorter
            than `edges`. E.g. `[10, 15]` means 10 trips in the band 1->2 and
            15 trips in the band 2->3.

        cost_units:
            The cost units being used in `edges`.

        band_mean_cost:
            Similar to `band_trips`. The mean cost in each band, as defined by
            `edges`. If left as None, this defaults to all -1 values.
        """
        # Set initial values
        if band_mean_cost is None:
            band_mean_cost = np.full(band_trips.shape, -1)

        # Set attributes
        self._cost_units = cost_units

        self.edges = edges
        self._sample_size = band_trips.sum()
        self.min_bounds = edges[:-1]
        self.max_bounds = edges[1:]
        self.mid_bounds = (self.min_bounds + self.max_bounds) / 2
        self.band_means = band_mean_cost
        self.band_trips = band_trips

        # Band means to use when plotting - can't be -1
        self._plot_band_means = np.where(
            self.band_means > 0,
            self.band_means,
            self.mid_bounds,
        )

        self._set_col_names()

    def __str__(self) -> str:
        df = self.to_df()
        return str(df)

    def __repr__(self) -> str:
        df = self.to_df()
        return repr(df)

    def is_empty(self) -> bool:
        """Check if this CostDistribution is empty

        A CostDistribution is empty when `self.band_trips` and
        `self.band_means` are set to arrays of 0s
        """
        # Check if arrays of 0s
        for array in (self.band_trips, self.band_means):
            if (array == 0).sum() != len(array):
                return False

        return True

    @property
    def cost_units(self) -> nd_core.CostUnits:
        """The cost units of this distribution"""
        return self._cost_units

    @cost_units.setter
    def cost_units(self, new_units: nd_core.CostUnits) -> None:
        """Sets the new cost units and updates column names"""
        self._cost_units = new_units
        self._set_col_names()

    @property
    def sample_size(self) -> float:
        """The number of trips in this distribution"""
        return self._sample_size

    @sample_size.setter
    def sample_size(self, new_sample_size: float) -> None:
        """Updates the sample size and adjusts self.band_trips proportionally"""
        adj_factor = new_sample_size / self.sample_size
        self.band_trips *= adj_factor
        self._sample_size = new_sample_size

    @property
    def band_shares(self) -> np.ndarray:
        """An array of band shares that corresponds to self.edges"""
        if self.sample_size > 0:
            return self.band_trips / self.sample_size
        else:
            return np.zeros_like(self.band_trips)


    @staticmethod
    def _get_col_names(cost_units: nd_core.CostUnits) -> tuple[str, str, str, str, str, str]:
        """Generates the default DataFrame column names

        Parameters
        ----------
        cost_units:
            The cost units being used in the cost distribution

        Returns
        -------
        cost_tuple:
            A tuple of columns names in the following order:
                min_bounds_col
                max_bounds_col
                mid_bounds_col
                mean_col
                trips_col
                shares_col
        """
        return (
            f"min ({cost_units.value})",
            f"max ({cost_units.value})",
            f"mid ({cost_units.value})",
            f"mean ({cost_units.value})",
            "trips",
            "share",
        )

    def _set_col_names(self) -> None:
        """Sets the column names to be used when converting to a DF"""
        col_names = self._get_col_names(self._cost_units)
        self.min_bounds_col = col_names[0]
        self.max_bounds_col = col_names[1]
        self.mid_bounds_col = col_names[2]
        self.mean_col = col_names[3]
        self.trips_col = col_names[4]
        self.shares_col = col_names[5]

    def to_df(self, additional_cols: dict[str, Any] = None) -> pd.DataFrame:
        """Builds a pandas DataFrame of this cost distribution

        Parameters
        ----------
        additional_cols:
            A dictionary of columns to additionally add to the generated
            DataFrame. Dictionary should be {col_name: col_value}.

        Returns
        -------
        cost_distribution_df:
            A pandas DataFrame of the cost distribution data being used. If
            `additional_cols` is set, those columns are also included.
            Columns are named after internal variables:
                self.min_bounds_col
                self.max_bounds_col
                self.mid_bounds_col
                self.mean_col
                self.trips_col
                self.shares_col
        """
        # Init
        df_dict = dict()

        # Add in additional cols if defined
        if additional_cols is not None:
            df_dict.update(additional_cols)

        # Add in the defined columns
        df_dict.update(
            {
                self.min_bounds_col: self.min_bounds,
                self.max_bounds_col: self.max_bounds,
                self.mid_bounds_col: self.mid_bounds,
                self.mean_col: self.band_means,
                self.trips_col: self.band_trips,
                self.shares_col: self.band_shares,
            }
        )

        return pd.DataFrame(df_dict)

    def to_graph(
        self,
        path: os.PathLike,
        band_shares: bool = False,
        label: str = "Cost Distribution",
        **graph_kwargs,
    ) -> None:
        """Writes a graph of this CostDistribution to disk

        Parameters
        ----------
        path:
            The path to write the generated graph out to.

        band_shares:
            Whether to output the graph as band shares or not. If False, graph
            will use band trips.

        label:
            The label to put on the legend of the generated graph.

        graph_kwargs:
            Any other kwargs to pass through to
            `normits_demand.cost.utils.plot_cost_distributions()`

        Returns
        -------
        None

        See Also
        --------
        `normits_demand.cost.utils.plot_cost_distributions()`
        """
        # Set default values
        y_values = self.band_trips
        y_axis_label = "Band Trips"
        if band_shares:
            y_values = self.band_shares
            y_axis_label = "Band Share (%)"

        # Infill any missing kwargs with defaults
        default_graph_kwargs = {
            "plot_title": "Cost Distribution",
            "x_axis_label": f"Distance ({self.cost_units.value})",
            "y_axis_label": y_axis_label,
            "band_share_cutoff": 0,
            "aspect_ratio": 9 / 16,
            "dpi": 300,
        }
        default_graph_kwargs.update(graph_kwargs)
        graph_kwargs = default_graph_kwargs

        # Build the output label
        if self.sample_size > 0:
            label = f"{label} | n={self.sample_size:.2f}"

        # Gather plotting data
        plot_data = cost_utils.PlotData(
            x_values=self._plot_band_means,
            y_values=y_values,
            label=label,
        )

        # Plot
        cost_utils.plot_cost_distributions(
            plot_data=plot_data,
            path=path,
            **graph_kwargs,
        )

    def to_csv(self, path: os.PathLike, **kwargs) -> None:
        """Builds a csv of this cost distribution and writes to disk

        Internally, translates into a pandas DataFrame using `self.to_df()`,
        then uses pandas functionality to write to disk.
        **kwargs are used to pass any additional arguments to the
        `df.to_csv()` call.

        Parameters
        ----------
        path:
            The path to write this cost distribution out to.

        Returns
        -------
        None
        """
        file_ops.safe_dataframe_to_csv(
            df=self.to_df(),
            out_path=path,
            **dict(kwargs, index=False),
        )

    @staticmethod
    def from_csv(
        path: os.PathLike,
        cost_units: nd_core.CostUnits,
        min_bounds_col: str = None,
        max_bounds_col: str = None,
        trips_col: str = None,
        mean_col: str = None,
    ) -> CostDistribution:
        """Reads in data from a csv to build a CostDistribution

         `pandas.read_csv()` is used to read in the csv. The data is then
         accessed via the column names given, before being handed over to the
         constructor to make a `CostDistribution` object

        Parameters
        ----------
        path:
            The path to the csv to read in

        cost_units:
            The cost units used in the cost distribution being read in from
            `path`

        min_bounds_col:
            The name of the column containing the minimum bounds of each band

        max_bounds_col:
            The name of the column containing the maximum bounds of each band

        trips_col:
            The name of the column containing the number of trips in each band

        mean_col:
            The name of the column containing the mean distance of the trips
            in each band

        Returns
        -------
        cost_distribution:
            The generated CostDistribution object
        """
        # Get the default columns names
        default_col_names = CostDistribution._get_col_names(cost_units)
        min_bounds_col = default_col_names[0] if min_bounds_col is None else min_bounds_col
        max_bounds_col = default_col_names[1] if max_bounds_col is None else max_bounds_col
        mean_col = default_col_names[3] if mean_col is None else mean_col
        trips_col = default_col_names[4] if trips_col is None else trips_col

        needed_cols = [min_bounds_col, max_bounds_col, trips_col]

        # Read in the data and validate
        df = pd.read_csv(path)
        assert isinstance(df, pd.DataFrame)
        missing_cols = set(needed_cols) - set(df.columns)
        if len(missing_cols) > 0:
            raise ValueError(
                "Missing some needed columns when reading in a cost "
                "distribution from disk. Missing columns:\n"
                f"{missing_cols}"
            )

        # Extract the needed data
        min_bounds = df[min_bounds_col].values
        max_bounds = df[max_bounds_col].values
        edges = np.array([min_bounds[0]] + max_bounds.tolist())
        band_trips = df[trips_col].values

        band_mean_cost = None
        if mean_col in df.columns:
            band_mean_cost = df[mean_col].values

        return CostDistribution(
            edges=edges,
            band_trips=band_trips,
            cost_units=cost_units,
            band_mean_cost=band_mean_cost,
        )

    @staticmethod
    def build_empty(edges: np.ndarray, cost_units: nd_core.CostUnits) -> CostDistribution:
        """Build an empty CostDistribution

        When `self.is_empty()` is checks, this class will return True.
        A CostDistribution is empty when `self.band_trips` and
        `self.band_means` are set to arrays of 0s.

        Parameters
        ----------
        edges:
            The band edges that would be used, if the data existed. E.g.
            `[1, 2, 3]` defines 2 bands: 1->2 and 2->3

        cost_units:
            The cost units being used in `edges`.

        Returns
        -------
        cost_distribution:
            The generated empty CostDistribution object
        """
        return CostDistribution(
            edges=edges,
            band_trips=np.zeros((len(edges) - 1, )),
            cost_units=cost_units,
            band_mean_cost=np.zeros((len(edges) - 1, )),
        )
