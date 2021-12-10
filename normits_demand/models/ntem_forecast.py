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
from normits_demand import core as nd_core
from normits_demand.utils import timing

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
    SEGMENTATION = {"hb": "hb_p_m_tp_wday", "nhb": "tms_nhb_p_m_tp_wday"}
    SEGMENTATION_COLUMNS = {"p": "purpose", "m": "mode", "tp": "time_period"}
    ZONE_SYSTEM = "msoa"
    TIME_FORMAT = "avg_day"

    def __init__(self, years: List[int]) -> None:
        try:
            self._years = [int(i) for i in years]
        except (ValueError, TypeError) as err:
            raise ValueError("years should be a list of integers") from err
        self._columns.update({str(y): float for y in self._years})
        self._data = None
        self._hb_attractions = None
        self._hb_productions = None
        self._nhb_attractions = None
        self._nhb_productions = None
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

    def _segment_dvector(self, seg: str, pa: str, year: int) -> nd_core.DVector:
        """Create a segment `DVector` from the TEMPro data.

        Parameters
        ----------
        seg : str
            The name of the segmentation to use, should be
            'hb' or 'nhb'.
        pa : str
            'productions' or 'attractions' filter for
            'trip_end_type' column.
        year : int
            Which year column to use.

        Returns
        -------
        nd_core.DVector
            DVector containing trip ends for given
            `segmentation` and `year`.

        Raises
        ------
        ValueError
            If an incorrect value is given for `segmentation`
            or `pa` parameters.
        """
        # Create boolean mask for hb/nhb purposes
        seg = seg.lower()
        if seg == "hb":
            purp_mask = self.data["purpose"] <= 8
        elif seg == "nhb":
            purp_mask = self.data["purpose"] > 8
        else:
            raise ValueError(
                "segmentation should be one of "
                f"{self.SEGMENTATION.values()} not {seg!r}"
            )
        # Create boolean mask for productions/attractions
        pa_options = ["attractions", "productions"]
        if pa not in pa_options:
            raise ValueError(f"pa should be one of {pa_options} not {pa!r}")
        pa_mask = self.data["trip_end_type"] == pa

        cols = ["msoa_zone_id", *self.SEGMENTATION_COLUMNS.values(), str(year)]
        return nd_core.DVector(
            nd_core.get_segmentation_level(self.SEGMENTATION[seg]),
            self.data.loc[purp_mask & pa_mask, cols],
            nd_core.get_zoning_system(self.ZONE_SYSTEM),
            time_format=self.TIME_FORMAT,
            zone_col=cols[0],
            val_col=str(year),
            df_naming_conversion=self.SEGMENTATION_COLUMNS,
        )

    @property
    def hb_attractions(self) -> dict[int, nd_core.DVector]:
        """dict[int, nd_core.DVector]
            Home-based attraction trip ends for all years (keys).
        """
        if self._hb_attractions is None:
            d = {}
            for yr in self._years:
                d[yr] = self._segment_dvector("hb", "attractions", yr)
            self._hb_attractions = d
        return self._hb_attractions

    @property
    def hb_productions(self) -> dict[int, nd_core.DVector]:
        """dict[int, nd_core.DVector]
            Home-based production trip ends for all years (keys).
        """
        if self._hb_productions is None:
            d = {}
            for yr in self._years:
                d[yr] = self._segment_dvector("hb", "productions", yr)
            self._hb_productions = d
        return self._hb_productions

    @property
    def nhb_attractions(self) -> dict[int, nd_core.DVector]:
        """dict[int, nd_core.DVector]
            Non-home-based attraction trip ends for all years (keys).
        """
        if self._nhb_attractions is None:
            d = {}
            for yr in self._years:
                d[yr] = self._segment_dvector("nhb", "attractions", yr)
            self._nhb_attractions = d
        return self._nhb_attractions

    @property
    def nhb_productions(self) -> dict[int, nd_core.DVector]:
        """dict[int, nd_core.DVector]
            Non-home-based production trip ends for all years (keys).
        """
        if self._nhb_productions is None:
            d = {}
            for yr in self._years:
                d[yr] = self._segment_dvector("nhb", "productions", yr)
            self._nhb_productions = d
        return self._nhb_productions

    def produce_dvectors(self) -> None:
        """Produce all of the different DVector properties for TEMPro data."""
        LOG.debug("Producing TEMPro DVectors for %s", self._years)
        start = timing.current_milli_time()
        self.hb_attractions
        self.hb_productions
        self.nhb_attractions
        self.nhb_attractions
        LOG.debug(
            "Done in %s",
            timing.time_taken(start, timing.current_milli_time()),
        )

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
