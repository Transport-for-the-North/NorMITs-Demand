# -*- coding: utf-8 -*-
"""
Created on: 11/04/2022
Updated on:

Original author: Ben Taylor
Last update made by:
Other updates made by:

File purpose:
Holds the ZoningSystem Class which stores all information on different zoning
systems
"""
# Built-Ins
import os
import abc
import enum
import pathlib
import shutil
import dataclasses

from typing import List
from typing import Optional
from dataclasses import InitVar


# Third Party
import pandas as pd

# Local Imports
# pylint: disable=import-error
from normits_demand.utils import general as du

# pylint: enable=import-error


@enum.unique
class ReportTemplates(enum.Enum):
    """Enum to reference different Excel report templates"""

    # Define enums here
    DISTRIBUTION_MODEL_MATRIX = "distribution_model_report.xlsx"

    @staticmethod
    def _get_templates_dir() -> pathlib.Path:
        return pathlib.Path(os.path.dirname(os.path.abspath(__file__))) / "templates"

    def copy_report_template(self, out_path: pathlib.Path) -> None:
        """Copy this report template to the given path.

        Parameters
        ----------
        out_path:
            The path to copy the report template to. If not filename is given,
            the default `self.value` filename is used.

        Returns
        -------
        None
        """
        # Make sure the src exists
        src = self._get_templates_dir() / self.value
        if not os.path.exists(src):
            raise du.NormitsDemandError(
                f"We don't seem to have a report template for {self}.\n"
                f"Tried looking for the data here: {src}"
            )

        # Assume no filename given if no suffix
        dst = out_path
        if dst.suffix == "":
            dst = dst / self.value

        shutil.copy(src=src, dst=dst)


class DistributionModelMatrixBase(abc.ABC):
    """Base class to store and manipulate sector output cols

    To be used in conjunction with
    ReportTemplates.DISTRIBUTION_MODEL_MATRIX

    Attributes
    ----------
    data:
        A pandas DataFrame containing all the data to output.

    val_cols:
        The name of the column in `sector_data` which contains the data values.

    p_col:
        The name of the column in `sector_data` which contains the purpose
        segment value.

    m_col:
        The name of the column in `sector_data` which contains the mode
        segment value.

    ca_col:
        The name of the column in `sector_data` which contains the
        car availablity segment value.

    tp_col:
        The name of the column in `sector_data` which contains the time period
        segment value.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        val_cols: List[str],
        p_col: str = "p",
        m_col: str = "m",
        ca_col: str = "ca",
        tp_col: str = "tp",
        segment_default_value: int = -1,
    ):
        # Assign attributes
        self.data = data
        self.val_cols = val_cols
        self.p_col = p_col
        self.m_col = m_col
        self.ca_col = ca_col
        self.tp_col = tp_col
        self.segment_default_value = segment_default_value

        # Add in any columns that don't exist
        for col_name in self.segment_cols:
            if col_name not in self.data.columns:
                self.data[col_name] = segment_default_value

        # Force sector data into the right order
        self.data = self.data.reindex(columns=self.col_order)

    @property
    @abc.abstractmethod
    def unique_cols(self) -> List[str]:
        """A list of the unique_cols in their output order"""
        return list()

    @property
    def segment_cols(self) -> list[str]:
        """A list of the segment columns in their output order"""
        return [
            self.p_col,
            self.m_col,
            self.ca_col,
            self.tp_col,
        ]

    @property
    def col_order(self) -> List[str]:
        """A list of the columns in their required output order"""
        return self.unique_cols + self.segment_cols + self.val_cols


class DistributionModelMatrixReportSectorData(DistributionModelMatrixBase):
    """Helper class to store and manipulate sector output cols

    To be used in conjunction with
    ReportTemplates.DISTRIBUTION_MODEL_MATRIX

    Attributes
    ----------
    data:
        A pandas DataFrame containing all the sector data.

    from_zone_col:
        The name of the column in `sector_data` which contains the from zone
        name.

    to_zone_col:
        The name of the column in `sector_data` which contains the to zone
        name.

    val_col:
        The name of the column in `sector_data` which contains the data values.

    p_col:
        The name of the column in `sector_data` which contains the purpose
        segment value.

    m_col:
        The name of the column in `sector_data` which contains the purpose
        segment value.

    ca_col:
        The name of the column in `sector_data` which contains the purpose
        segment value.

    tp_col:
        The name of the column in `sector_data` which contains the time period
        segment value.
    """

    def __init__(self, from_zone_col: str, to_zone_col: str, *args, **kwargs):
        self.from_zone_col = from_zone_col
        self.to_zone_col = to_zone_col

        super().__init__(*args, **kwargs)

    @property
    def unique_cols(self) -> List[str]:
        """A list of the unique_cols in their output order"""
        return [self.from_zone_col, self.to_zone_col]


class DistributionModelMatrixReportTripEndData(DistributionModelMatrixBase):
    """Helper class to store and manipulate sector output cols

    To be used in conjunction with
    ReportTemplates.DISTRIBUTION_MODEL_MATRIX

    Attributes
    ----------
    data:
        A pandas DataFrame containing all the trip end sector data.

    from_zone_col:
        The name of the column in `sector_data` which contains the from zone
        name.

    val_col:
        The name of the column in `sector_data` which contains the data values.

    p_col:
        The name of the column in `sector_data` which contains the purpose
        segment value.

    m_col:
        The name of the column in `sector_data` which contains the purpose
        segment value.

    ca_col:
        The name of the column in `sector_data` which contains the purpose
        segment value.

    tp_col:
        The name of the column in `sector_data` which contains the time period
        segment value.
    """

    def __init__(self, from_zone_col: str, *args, **kwargs):
        self.from_zone_col = from_zone_col

        super().__init__(*args, **kwargs)

    @property
    def unique_cols(self) -> List[str]:
        """A list of the unique_cols in their output order"""
        return [self.from_zone_col]


class DistributionModelMatrixReportCostDistributionData(DistributionModelMatrixBase):
    """Helper class to store and manipulate sector output cols

    To be used in conjunction with
    ReportTemplates.DISTRIBUTION_MODEL_MATRIX

    Attributes
    ----------
    data:
        A pandas DataFrame containing the distance bands, trip total and distribution.

    distribution_col:
        The name of the column in `tld_data` which contains the data totals.

    distribution_pct_col:
        The name of the column in `tld_data` which contains the band percentages.

    lower_col:
        The name of the column in `tld_data` which contains the lower boundary of
        the distance band.

    upper_col:
        The name of the column in `tld_data` which contains the upper boundary of
        the distance band.

    p_col:
        The name of the column in `sector_data` which contains the purpose
        segment value.

    m_col:
        The name of the column in `sector_data` which contains the purpose
        segment value.

    ca_col:
        The name of the column in `sector_data` which contains the purpose
        segment value.

    tp_col:
        The name of the column in `sector_data` which contains the time period
        segment value.
    """

    def __init__(self, lower_col: str, upper_col: str, *args, **kwargs):
        self.lower_col = lower_col
        self.upper_col = upper_col

        super().__init__(*args, **kwargs)

    @property
    def unique_cols(self) -> List[str]:
        """A list of the unique_cols in their output order"""
        return [self.lower_col, self.upper_col]

    @property
    def col_order(self) -> List[str]:
        """A list of the columns in their required output order"""
        return self.segment_cols + self.unique_cols + self.val_cols







@dataclasses.dataclass
class DistributionModelTripEndReportData:
    """Helper class to store and manipulate sector output cols

    To be used in conjunction with
    ReportTemplates.DISTRIBUTION_MODEL_MATRIX

    Attributes
    ----------
    sector_data:
        A pandas DataFrame containing all the trip end sector data.
        Must contain the required columns, as defined in
        `self.required_cols()`

    from_zone_col:
        The name of the column in `sector_data` which contains the from zone
        name.

    val_col:
        The name of the column in `sector_data` which contains the data values.

    p_col:
        The name of the column in `sector_data` which contains the purpose
        segment value.

    m_col:
        The name of the column in `sector_data` which contains the purpose
        segment value.

    ca_col:
        The name of the column in `sector_data` which contains the purpose
        segment value.

    tp_col:
        The name of the column in `sector_data` which contains the time period
        segment value.
    """

    sector_data: pd.DataFrame
    from_zone_col: str
    val_col: str

    p_col: Optional[str] = "p"
    m_col: Optional[str] = "m"
    ca_col: Optional[str] = "ca"
    tp_col: Optional[str] = "tp"
    segment_default_value: InitVar[int] = -1

    def __post_init__(self, segment_default_value: int):
        # Add in any columns that don't exist
        for col_name in [self.p_col, self.m_col, self.ca_col, self.tp_col]:
            if col_name not in self.sector_data.columns:
                self.sector_data[col_name] = segment_default_value

        # Force sector data into the right order
        self.sector_data = self.sector_data.reindex(columns=self.col_order())

    def col_order(self):
        """Get a list of the columns in their required output order"""
        return [
            self.from_zone_col,
            self.p_col,
            self.m_col,
            self.ca_col,
            self.tp_col,
            self.val_col,
        ]







@dataclasses.dataclass
class DistributionModelReportTLDData:
    """Helper class to store and manipulate sector output cols

    To be used in conjunction with
    ReportTemplates.DISTRIBUTION_MODEL_MATRIX

    Attributes
    ----------
    tld_data:
        A pandas DataFrame containing the distance bands, trip total and distribution.
        Must contain the required columns, as defined in
        `self.required_cols()`

    distribution_col:
        The name of the column in `tld_data` which contains the data totals.

    distribution_pct_col:
        The name of the column in `tld_data` which contains the band percentages.

    lower_col:
        The name of the column in `tld_data` which contains the lower boundary of
        the distance band.

    upper_col:
        The name of the column in `tld_data` which contains the upper boundary of
        the distance band.

    p_col:
        The name of the column in `sector_data` which contains the purpose
        segment value.

    m_col:
        The name of the column in `sector_data` which contains the purpose
        segment value.

    ca_col:
        The name of the column in `sector_data` which contains the purpose
        segment value.

    tp_col:
        The name of the column in `sector_data` which contains the time period
        segment value.
    """

    tld_data: pd.DataFrame
    lower_col: str
    upper_col: str
    distribution_col: str
    distribution_pct_col: str

    p_col: Optional[str] = "p"
    m_col: Optional[str] = "m"
    ca_col: Optional[str] = "ca"
    tp_col: Optional[str] = "tp"
    segment_default_value: InitVar[int] = -1

    def __post_init__(self, segment_default_value: int):
        # Add in any columns that don't exist
        for col_name in [self.p_col, self.m_col, self.ca_col, self.tp_col]:
            if col_name not in self.tld_data.columns:
                self.tld_data[col_name] = segment_default_value

        # Force sector data into the right order
        self.tld_data = self.tld_data.reindex(columns=self.col_order())

    def col_order(self):
        """Get a list of the columns in their required output order"""
        return [
            self.p_col,
            self.m_col,
            self.ca_col,
            self.tp_col,
            self.lower_col,
            self.upper_col,
            self.distribution_col,
            self.distribution_pct_col,
        ]
