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
import enum
import pathlib
import shutil
import dataclasses

from typing import Optional
from dataclasses import InitVar


# Third Party
import pandas as pd

# Local Imports
# pylint: disable=import-error
from normits_demand import core as nd_core
from normits_demand import logging as nd_log
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
        if dst.suffix == '':
            dst = dst / self.value

        shutil.copy(src=src, dst=dst)


@dataclasses.dataclass
class DistributionModelMatrixReportSectorData:
    """Helper class to store and manipulate sector output cols

    To be used in conjunction with
    ReportTemplates.DISTRIBUTION_MODEL_MATRIX

    Attributes
    ----------
    sector_data:
        A pandas DataFrame containing all the sector data.
        Must contain the required columns, as defined in
        `self.required_cols()`

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
    """
    sector_data: pd.DataFrame
    from_zone_col: str
    to_zone_col: str
    val_col: str

    p_col: Optional[str] = 'p'
    m_col: Optional[str] = 'm'
    ca_col: Optional[str] = 'ca'
    segment_default_value: InitVar[int] = -1

    def __post_init__(self, segment_default_value: int):
        # Add in any columns that don't exist
        for col_name in [self.p_col, self.m_col, self.ca_col]:
            if col_name not in self.sector_data.columns:
                self.sector_data[col_name] = segment_default_value

        # Force sector data into the right order
        self.sector_data = self.sector_data.reindex(columns=self.col_order())

    def col_order(self):
        """Get a list of the columns in their required output order"""
        return [
            self.from_zone_col,
            self.to_zone_col,
            self.p_col,
            self.m_col,
            self.ca_col,
            self.val_col,
        ]

    def required_cols(self):
        """Get a list of the required columns for the sector report"""
        return [
            self.from_zone_col,
            self.to_zone_col,
            self.val_col,
        ]
