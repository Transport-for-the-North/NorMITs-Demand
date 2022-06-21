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

# Third Party
import numpy as np
import pandas as pd

# Local Imports
from normits_demand import core as nd_core

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
        self.min_bounds = edges[:-1]
        self.max_bounds = edges[1:]
        self.mid_bounds = (self.min_bounds + self.max_bounds) / 2
        self.band_means = band_mean_cost
        self.band_trips = band_trips
        self.band_shares = band_trips / np.sum(band_trips)

        self._set_col_names()

    def __str__(self) -> str:
        df = self.to_df()
        return str(df)

    def __repr__(self) -> str:
        df = self.to_df()
        return repr(df)

    @property
    def cost_units(self) -> nd_core.CostUnits:
        """The cost units of this distribution"""
        return self._cost_units

    @cost_units.setter
    def cost_units(self, new_units: nd_core.CostUnits) -> None:
        """Sets the new cost units and updates column names"""
        self._cost_units = new_units
        self._set_col_names()

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

    def to_df(self) -> pd.DataFrame:
        """Builds a pandas DataFrame of this cost distribution

        Returns
        -------
        cost_distribution_df:
            A pandas DataFrame of the cost distribution data being used.
            Columns are named after internal variables:
                self.min_bounds_col
                self.max_bounds_col
                self.mid_bounds_col
                self.mean_col
                self.trips_col
                self.shares_col
        """
        return pd.DataFrame(
            {
                self.min_bounds_col: self.min_bounds,
                self.max_bounds_col: self.max_bounds,
                self.mid_bounds_col: self.mid_bounds,
                self.mean_col: self.band_means,
                self.trips_col: self.band_trips,
                self.shares_col: self.band_shares,
            }
        )

    def to_csv(self, path: os.PathLike) -> None:
        """Builds a csv of this cost distribution and writes to disk

        Parameters
        ----------
        path:
            The path to write this cost distribution out to.

        Returns
        -------
        None
        """
        self.to_df().to_csv(path, index=False)

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
