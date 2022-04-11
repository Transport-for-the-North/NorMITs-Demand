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

# Third Party

# Local Imports
from normits_demand.utils import general as du


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

