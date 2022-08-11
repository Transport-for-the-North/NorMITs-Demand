# -*- coding: utf-8 -*-
"""
    Module containing functionality for reading, and handling,
    trip end data from TEMPro.
"""

##### IMPORTS #####
# Standard imports
import dataclasses
from pathlib import Path
from typing import List, Dict

# Third party imports
import pandas as pd

# Local imports
import normits_demand
from normits_demand.utils import file_ops, ntem_extractor
from normits_demand import logging as nd_log
from normits_demand import core as nd_core
from normits_demand.utils import timing

##### CONSTANTS #####
LOG = nd_log.get_logger(__name__)


##### CLASSES #####
class NTEMForecastError(normits_demand.NormitsDemandError):
    """Exception for errors caused during NTEM forecasting."""


class TEMProData:
    """Class for reading and filtering TEMPro trip end data.

    If a folder is passed for `data_path` then the
    `TemproParser` class is used to get the TEMPro data.

    Parameters
    ----------
    data_path : Path
        Path to the TEMPro data CSV or to the folder
        containing the TEMPro databases.
    years : List[int]
        List of year columns to read from data file.

    Raises
    ------
    FileNotFoundError
        If `data_path` isn't an existing file.
    NTEMForecastError
        If `years` isn't (or cannot be converted to)
        a list of integers.

    See Also
    --------
    `tempro_extractor.TemproParser`: extracts TEMPro data from the databases.
    """

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

    def __init__(self, data_path: Path, years: List[int]) -> None:
        self.data_path = Path(data_path)
        self.use_tempro_extractor = False

        # If data path is a folder then assumes this contains the TEMPro
        # databases, otherwise assumes the file is a CSV containing the data
        if self.data_path.is_dir():
            self.use_tempro_extractor = True
        elif not self.data_path.is_file():
            raise FileNotFoundError(f"cannot find TEMPro data: {self.data_path}")

        try:
            self._years = [int(i) for i in years]
        except (ValueError, TypeError) as err:
            raise NTEMForecastError("years should be a list of integers") from err
        self._columns.update({str(y): float for y in self._years})
        self._data = None
        self._dvectors = None

        if not self.use_tempro_extractor:
            # Read top 5 rows to check file format
            try:
                file_ops.read_df(
                    self.data_path,
                    usecols=self._columns.keys(),
                    dtype=self._columns,
                    nrows=5,
                )
            except ValueError as err:
                raise NTEMForecastError(f"error reading TEMPro data - {err}") from err

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(years={self._years})"

    def __repr__(self) -> str:
        return f"{self.__module__}.{self!s}"

    def _read_tempro_csv(self) -> pd.DataFrame:
        LOG.info("Reading TEMPro data from CSV: %s", self.data_path)
        return file_ops.read_df(
            self.data_path,
            usecols=self._columns.keys(),
            dtype=self._columns,
        )

    def _extract_tempro_database(self) -> pd.DataFrame:
        LOG.info("Extracting data from TEMPro databases in: %s", self.data_path)
        parser = ntem_extractor.TemproParser(
            output_years=self._years, data_source=str(self.data_path)
        )
        return parser.get_trip_ends(
            trip_type="pa", all_commute_hb=True, aggregate_car=False, average_weekday=False
        )

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
            if self.use_tempro_extractor:
                data = self._extract_tempro_database()
            else:
                data = self._read_tempro_csv()

            data.columns = data.columns.str.strip().str.lower()
            data.rename(columns={"timeperiod": "time_period"}, inplace=True)

            # Drop mode 4 and tp > 4 to match segmentation used for DVectors
            mask = (data["mode"] != 4) & (data["time_period"] <= 4)
            data = data.loc[mask]
            LOG.debug(
                "Dropping mode 4 and time periods > 4 from "
                "TEMPro data, %s rows dropped (%s remaining)",
                mask.sum(),
                len(data),
            )

            self._data = data

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
            `seg` and `year`.

        Raises
        ------
        NTEMForecastError
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
            raise NTEMForecastError(
                "segmentation should be one of " f"{self.SEGMENTATION.values()} not {seg!r}"
            )
        # Create boolean mask for productions/attractions
        pa_options = ["attractions", "productions"]
        if pa not in pa_options:
            raise NTEMForecastError(f"pa should be one of {pa_options} not {pa!r}")
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

    def produce_dvectors(self):
        """Produce all of the different DVector properties for TEMPro data.

        Returns
        -------
        TEMProTripEnds
            Trip end DVectors for all years.
        """
        if self._dvectors is None:
            LOG.debug("Producing TEMPro DVectors for %s", self._years)
            start = timing.current_milli_time()
            # Produce DVectors for all segments
            hb_attr = {}
            hb_prod = {}
            nhb_attr = {}
            nhb_prod = {}
            for yr in self._years:
                hb_attr[yr] = self._segment_dvector("hb", "attractions", yr)
                hb_prod[yr] = self._segment_dvector("hb", "productions", yr)
                nhb_attr[yr] = self._segment_dvector("nhb", "attractions", yr)
                nhb_prod[yr] = self._segment_dvector("nhb", "productions", yr)
            LOG.debug(
                "Done in %s",
                timing.time_taken(start, timing.current_milli_time()),
            )
            self._dvectors = {
                "hb_attractions": hb_attr,
                "hb_productions": hb_prod,
                "nhb_attractions": nhb_attr,
                "nhb_productions": nhb_prod,
            }
        return TEMProTripEnds(**self._dvectors)

    def get(
        self,
        purposes: List[int] = None,
        modes: List[int] = None,
        time_periods: List[int] = None,
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
        NTEMForecastError
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
                    raise NTEMForecastError(f"{c} should be a list of integers") from err
                filtered = filtered.loc[filtered[c].isin(ls)]
        return filtered


@dataclasses.dataclass
class TEMProTripEnds:
    """TEMPro trip end data for all years.

    Attributes
    ----------
    hb_attractions: Dict[int, nd_core.DVector]
        Home-based attractions trip end vectors,
        dictionary keys are the year.
    hb_productions: Dict[int, nd_core.DVector]
        Home-based productions trip end vectors,
        dictionary keys are the year.
    nhb_attractions: Dict[int, nd_core.DVector]
        Non-home-based attractions trip end vectors,
        dictionary keys are the year.
    nhb_productions: Dict[int, nd_core.DVector]
        Non-home-based productions trip end vectors,
        dictionary keys are the year.

    Raises
    ------
    NTEMForecastError
        If the dictionary keys in all of the attributes
        aren't the same.
    """

    hb_attractions: Dict[int, nd_core.DVector]
    hb_productions: Dict[int, nd_core.DVector]
    nhb_attractions: Dict[int, nd_core.DVector]
    nhb_productions: Dict[int, nd_core.DVector]

    def __post_init__(self):
        """Check all attributes have the same keys."""
        for name, field in dataclasses.asdict(self).items():
            if field.keys() != self.hb_attractions.keys():
                raise NTEMForecastError(f"years (keys) differ for attribute {name!r}")

    def save(self, folder: Path):
        """Save all DVectors to `folder`.

        Saved using `DVector.compress_out` method with name
        in the format "{nhb|hb}_{attractions|productions}-{year}".

        Parameters
        ----------
        folder : Path
            Path to folder to save outputs, will
            be created if it doesn't already exist.
        """
        folder.mkdir(exist_ok=True, parents=True)
        LOG.info("Writing TEMProForecasts to %s", folder)
        years: Dict[int, nd_core.DVector]
        for name, years in dataclasses.asdict(self).items():
            for yr, dvec in years.items():
                dvec.save(folder / f"{name}-{yr}.pkl")

    def translate_zoning(
        self,
        zone_system: str,
        weighting: Dict[str, str] = None,
        **kwargs,
    ):
        """Translates all DVectors into new `zone_system`.

        Translations are done using `DVector.translate_zoning`
        and a new instance of `TEMProForecasts` is returned.
        This does not update the current instance.

        Parameters
        ----------
        zone_system : str
            Name of the zone system to translate to.
        weighting : Dict[str, str]
            Name of the weighting (value) to be passed to
            `DVector.translate_zoning`, weighting is defined for each
            attribute separately (keys should be attribute names).
        kwargs : Dict[str, Any]
            Keyword arguments passed to `DVector.translate_zoning`.

        Returns
        -------
        TEMProTripEnds
            New instance of this class with the DVectors
            all translated into the new `zone_system`.
        """
        if weighting is None:
            weighting = {}
        LOG.info("Translating TEMProTripEnds zone system to %s", zone_system)
        zoning = nd_core.get_zoning_system(zone_system)
        new_data = {}
        for name, attribute in dataclasses.asdict(self).items():
            weight = weighting.get(name, None)
            new_data[name] = {
                yr: dvec.translate_zoning(zoning, weight, **kwargs)
                for yr, dvec in attribute.items()
            }
        return TEMProTripEnds(**new_data)

    def _get_segmentation(
        self, segmentation: Dict[str, str]
    ) -> Dict[str, nd_core.SegmentationLevel]:
        """Get `SegmentationLevel` for each segment name given.

        SegmentationLevel instances are created using
        `nd_core.get_segmentation_level`.

        Parameters
        ----------
        segmentation : Dict[str, str]
            Dictionary containing the attribute name (keys) and
            the name of the segmentation to get (values). Should
            contain keys for every attribute in this class.

        Returns
        -------
        Dict[str, nd_core.SegmentationLevel]
            SegmentationLevel instances for each attribute of
            the class.

        Raises
        ------
        NTEMForecastError
            If there are any attributes in the class which
            aren't in `segmentation`.
        """
        segments = {}
        missing = []
        for field in dataclasses.fields(self):
            try:
                segments[field.name] = nd_core.get_segmentation_level(segmentation[field.name])
            except KeyError:
                missing.append(field.name)
        if missing:
            raise NTEMForecastError(f"trip end attributes expected, but not found: {missing}")
        return segments

    def aggregate(self, segmentation: Dict[str, str], **kwargs):
        """Aggregates all DVectors to new segmentation.

        Parameters
        ----------
        segmentation : Dict[str, str]
            Dictionary containing the attribute name (keys) and
            the name of the segmentation to get (values). Should
            contain keys for every attribute in this class.

        Returns
        -------
        TEMProTripEnds
            A new instance of this class with the new
            segmentation
        """
        new_data = {}
        for attr_name, segment in self._get_segmentation(segmentation).items():
            new_data[attr_name] = {
                yr: dvec.aggregate(segment, **kwargs)
                for yr, dvec in getattr(self, attr_name).items()
            }
        return TEMProTripEnds(**new_data)

    def subset(self, segmentation: Dict[str, str], **kwargs):
        """Gets a subset of the DVectors at the new segmentation.

        Parameters
        ----------
        segmentation : Dict[str, str]
            Dictionary containing the attribute name (keys) and
            the name of the segmentation to get (values). Should
            contain keys for every attribute in this class.

        Returns
        -------
        TEMProTripEnds
            A new instance of this class with the new
            segmentation
        """
        new_data = {}
        for attr_name, segment in self._get_segmentation(segmentation).items():
            new_data[attr_name] = {
                yr: dvec.subset(segment, **kwargs)
                for yr, dvec in getattr(self, attr_name).items()
            }
        return TEMProTripEnds(**new_data)
