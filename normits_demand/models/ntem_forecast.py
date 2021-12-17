# -*- coding: utf-8 -*-
"""
    Module for producing forecast demand constrained to NTEM.
"""

##### IMPORTS #####
# Standard imports
from pathlib import Path
from typing import List, Dict

# Third party imports
import numpy as np
import pandas as pd

# Local imports
from normits_demand.utils import file_ops
from normits_demand import logging as nd_log
from normits_demand import core as nd_core
from normits_demand.utils import timing
from normits_demand import efs_constants as efs_consts
from normits_demand import constants as consts

##### CONSTANTS #####
LOG = nd_log.get_logger(__name__)
LAD_ZONE_SYSTEM = "lad_2020"


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

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(years={self._years})"

    def __repr__(self) -> str:
        return f"{self.__module__}.{self!s}"

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
    def hb_attractions(self) -> Dict[int, nd_core.DVector]:
        """Dict[int, nd_core.DVector]
            Home-based attraction trip ends for all years (keys).
        """
        if self._hb_attractions is None:
            d = {}
            for yr in self._years:
                d[yr] = self._segment_dvector("hb", "attractions", yr)
            self._hb_attractions = d
        return self._hb_attractions

    @property
    def hb_productions(self) -> Dict[int, nd_core.DVector]:
        """Dict[int, nd_core.DVector]
            Home-based production trip ends for all years (keys).
        """
        if self._hb_productions is None:
            d = {}
            for yr in self._years:
                d[yr] = self._segment_dvector("hb", "productions", yr)
            self._hb_productions = d
        return self._hb_productions

    @property
    def nhb_attractions(self) -> Dict[int, nd_core.DVector]:
        """Dict[int, nd_core.DVector]
            Non-home-based attraction trip ends for all years (keys).
        """
        if self._nhb_attractions is None:
            d = {}
            for yr in self._years:
                d[yr] = self._segment_dvector("nhb", "attractions", yr)
            self._nhb_attractions = d
        return self._nhb_attractions

    @property
    def nhb_productions(self) -> Dict[int, nd_core.DVector]:
        """Dict[int, nd_core.DVector]
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
        # Call each property to produce the DVectors
        # pylint: disable=pointless-statement
        self.hb_attractions
        self.hb_productions
        self.nhb_attractions
        self.nhb_attractions
        # pylint: enable=pointless-statement
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


class NTEMImportMatrices:
    """Generates paths to base PostME matrices.

    These matrices are used as the base for the NTEM forecasting.

    Parameters
    ----------
    import_folder : Path
        Path to the import folder, should contain
        the matrices in a sub-path (`MATRIX_FOLDER`).
    year : int
        Model base year.
    model_name : str
        Name of the model to get inputs from, currently
        only works with 'noham'.

    Raises
    ------
    NotImplementedError
        This class only handles the noham model and
        one mode per model.
    """

    MATRIX_FOLDER = "{name}/post_me/tms_seg_pa"
    _MATRIX_FILENAME = "{hb}_pa_yr{yr}_p{p}_m{m}.csv"

    def __init__(self, import_folder: Path, year: int, model_name: str) -> None:
        file_ops.check_path_exists(import_folder)
        self.year = int(year)
        self.model_name = model_name.lower().strip()
        if self.model_name != "noham":
            raise NotImplementedError(
                "this class currently only works for 'noham' model"
            )
        self.matrix_folder = import_folder / self.MATRIX_FOLDER.format(
            name=self.model_name
        )
        file_ops.check_path_exists(self.matrix_folder)
        self.mode = efs_consts.MODEL_MODES[self.model_name]
        if len(self.mode) == 1:
            self.mode = self.mode[0]
        else:
            raise NotImplementedError(
                "cannot handle models with more than one mode, "
                f"this model ({self.model_name}) has {len(self.mode)} modes"
            )
        self._hb_paths = None
        self._nhb_paths = None

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}(matrix_folder={self.matrix_folder}, "
            f"year={self.year}, model_name={self.model_name})"
        )

    def __repr__(self) -> str:
        return f"{self.__module__}.{self!s}"

    def _check_path(self, hb: str, purpose: int) -> Path:
        """Checks file exists for specific purpose.

        Parameters
        ----------
        hb : str {'hb', 'nhb'}
            Whether using home-based (hb) or non-home-based (nhb).
        purpose : int
            Number of the specific purpose to get path for.

        Returns
        -------
        Path
            Path to the matrix file.
        """
        path = self.matrix_folder / self._MATRIX_FILENAME.format(
            hb=hb, yr=self.year, p=purpose, m=self.mode
        )
        file_ops.check_file_exists(path, find_similar=True)
        return path

    @property
    def hb_paths(self) -> Dict[int, Path]:
        """Dict[int, Path]
            Paths to home-based matrices for each
            purpose (keys) for the given year.

        See Also
        --------
        normits_demand.constants.ALL_HB_P:
            for a list of all home-based purposes
        """
        if self._hb_paths is None:
            self._hb_paths = {}
            for p in consts.ALL_HB_P:
                self._hb_paths[p] = self._check_path("hb", p)
        return self._hb_paths.copy()

    @property
    def nhb_paths(self) -> Dict[int, Path]:
        """Dict[int, Path]
            Paths to non-home-based matrices for each
            purpose (keys) for the given year.

        See Also
        --------
        normits_demand.constants.ALL_NHB_P:
            for a list of all non-home-based purposes
        """
        if self._nhb_paths is None:
            self._nhb_paths = {}
            for p in consts.ALL_NHB_P:
                self._nhb_paths[p] = self._check_path("nhb", p)
        return self._nhb_paths.copy()


##### FUNCTIONS #####
def trip_end_growth(
    tempro_vectors: Dict[int, nd_core.DVector]
) -> Dict[int, nd_core.DVector]:
    """Calculate growth at LAD level and return it a `tempro_vectors` zone system.

    The trip ends are translated to `LAD_ZONE_SYSTEM` to
    calculate growth factors then translated back to the
    original zone system before returning.

    Parameters
    ----------
    tempro_vectors : Dict[int, nd_core.DVector]
        Trip end vectors from TEMPro for all study years,
        keys should be years and must include
        `normits_demand.efs_constants.BASE_YEAR`.

    Returns
    -------
    Dict[int, nd_core.DVector]
        Trip end growth factors in same zone system as
        `tempro_vectors` base year, contains all years
        from `tempro_vectors` except the base year.

    Raises
    ------
    ValueError
        If `normits_demand.efs_constants.BASE_YEAR` is not
        in `tempro_vectors`.
    """
    if efs_consts.BASE_YEAR not in tempro_vectors:
        raise ValueError(f"base year ({efs_consts.BASE_YEAR}) data not given")
    old_zone = tempro_vectors[efs_consts.BASE_YEAR].zoning_system
    growth_zone = nd_core.get_zoning_system(LAD_ZONE_SYSTEM)
    base_data = tempro_vectors[efs_consts.BASE_YEAR
                              ].translate_zoning(growth_zone)
    # Convert to LADs and calculate growth from base year
    growth = {}
    # Ignore divide by zero warnings and fill with zeros
    with np.errstate(divide="ignore", invalid="ignore"):
        for yr, data in tempro_vectors.items():
            if yr == efs_consts.BASE_YEAR:
                continue
            data = data.translate_zoning(growth_zone) / base_data
            # Set any nan or inf values created by dividing by 0 to 0 growth
            data = data.segment_apply(
                np.nan_to_num, nan=0.0, posinf=0.0, neginf=0.0
            )
            # Translate back to original zone system
            growth[yr] = data.translate_zoning(old_zone)
    return growth


def grow_trip_ends(
    tempro_vectors: Dict[int, nd_core.DVector]
) -> Dict[int, nd_core.DVector]:
    """Grow TEMPro trip ends based on LAD growth.

    Growth factors are calculated at `LAD_ZONE_SYSTEM`
    level but applied at the original zone system.

    Parameters
    ----------
    tempro_vectors : Dict[int, nd_core.DVector]
        Trip end vectors from TEMPro for all study years,
        keys should be years and must include
        `normits_demand.efs_constants.BASE_YEAR`.

    Returns
    -------
    Dict[int, nd_core.DVector]
        Future trip ends in same zone system as `tempro_vectors`
        base year, contains all years from `tempro_vectors`
        except the base year.

    Raises
    ------
    ValueError
        If `normits_demand.efs_constants.BASE_YEAR` is not
        in `tempro_vectors`.

    See Also
    --------
    trip_end_growth : for growth factor calculation
    """
    # Calculate growth at LAD level
    te_growth = trip_end_growth(tempro_vectors)
    base_data = tempro_vectors[efs_consts.BASE_YEAR]
    future = {}
    for yr, growth in te_growth.items():
        future[yr] = base_data * growth
    return future
