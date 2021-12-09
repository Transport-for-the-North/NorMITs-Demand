# -*- coding: utf-8 -*-
"""
    Module for producing forecast demand constrained to NTEM.
"""

##### IMPORTS #####
# Standard imports
from pathlib import Path
from typing import List

# Third party imports
import pandas as pd

# Local imports
from normits_demand.utils import file_ops
from normits_demand import logging as nd_log

##### CONSTANTS #####
LOG = nd_log.get_logger(__name__)


##### CLASSES #####
class TEMProData:
    """Class for reading and filtering TEMPro trip end data.

    Parameters
    ----------
    years : List[int]
        List of year columns to read from data file.

    Raises
    ------
    ValueError
        If `years` isn't (or cannot be converted to)
        a list of integers.
    """

    DATA_PATH = Path(
        r"I:\NorMITs Demand\import\ntem_forecast\tempro_pa_data.csv"
    )
    _columns = {
        "msoa_zone_id": str,
        "trip_end_type": str,
        "Purpose": int,
        "Mode": int,
        "TimePeriod": int,
    }

    def __init__(self, years: List[int]) -> None:
        try:
            self._years = [int(i) for i in years]
        except (ValueError, TypeError) as err:
            raise ValueError("years should be a list of integers") from err
        self._columns.update({str(y): float for y in self._years})
        self._data = None
        if not self.DATA_PATH.exists():
            raise FileNotFoundError(
                f"TEMPro data file cannot be found: {self.DATA_PATH}"
            )
        # Read top 5 rows to check file format
        try:
            file_ops.read_df(
                self.DATA_PATH,
                usecols=self._columns.keys(),
                dtype=self._columns,
                nrows=5,
            )
        except ValueError as err:
            raise ValueError(f"error reading TEMPro data - {err}") from err

    @property
    def data(self) -> pd.DataFrame:
        """pd.DataFrame:
            TEMPro data for all years selected, contains columns:
            - msoa_zone_id
            - trip_end_type
            - purpose
            - mode
            - time_period
            - {year}: separate column for each year selected
              e.g. '2018', '2020'
        """
        if self._data is None:
            LOG.info("Reading TEMPro data: %s", self.DATA_PATH)
            self._data = file_ops.read_df(
                self.DATA_PATH,
                usecols=self._columns.keys(),
                dtype=self._columns,
            )
            self._data.columns = self._data.columns.str.strip().str.lower()
            self._data.rename(
                columns={"timeperiod": "time_period"}, inplace=True
            )
        return self._data.copy()

    def get(
        self,
        purposes: List[int] = None,
        modes: List[int] = None,
        time_periods: List[int] = None
    ) -> pd.DataFrame:
        """Returns filtered version of the TEMPro data for selected years.

        Parameters
        ----------
        purposes : List[int], optional
            List of purposes to include in filter
        modes : List[int], optional
            List of modes to include in filter
        time_periods : List[int], optional
            List of time periods to include in filter

        Returns
        -------
        pd.DataFrame
            Filtered version of the TEMPro data can access
            all data with `TEMProData().data`.

        Raises
        ------
        ValueError
            If any of the parameters provided aren't (or
            cannot be converted to) a list of integers.
        """
        filtered = self.data
        filter_cols = ("purpose", "mode", "time_period")
        filter_lists = (purposes, modes, time_periods)
        for c, ls in zip(filter_cols, filter_lists):
            if ls is not None:
                try:
                    ls = [int(i) for i in ls]
                except (ValueError, TypeError) as err:
                    raise ValueError(
                        f"{c} should be a list of integers"
                    ) from err
                filtered = filtered.loc[filtered[c].isin(ls)]
        return filtered


##### FUNCTIONS #####
