# -*- coding: utf-8 -*-
"""
    Module for producing forecast demand constrained to MiTEM.
"""

##### IMPORTS #####
# Standard imports
from pathlib import Path

# Third party imports

# Local imports
from normits_demand.utils import file_ops
from normits_demand import logging as nd_log
from normits_demand import core as nd_core
from normits_demand.models import ntem_forecast

##### CONSTANTS #####
LOG = nd_log.get_logger(__name__)
MIDMITS_MODEL_MODES = {"miham": [3]}

##### CLASSES #####
class MiTEMImportMatrices(ntem_forecast.NTEMImportMatrices):
    """Generates paths to base PostME matrices.

    These matrices are used as the base for the NTEM forecasting.

    Parameters
    ----------
    import_folder : Path
        Path to the import folder containing the matrices.
    year : int
        Model base year.
    model_name : str
        Name of the model to get inputs from, currently
        only works with 'miham'.

    Raises
    ------
    NotImplementedError
        This class only handles the MiHAM model and
        one mode per model.
    """

    MATRIX_FOLDER = None  # Not used in subclass

    def __init__(self, import_folder: Path, year: int, model_name: str) -> None:
        file_ops.check_path_exists(import_folder)
        self.matrix_folder = import_folder
        self.year = int(year)
        self.model_name = model_name.lower().strip()
        if self.model_name != "miham":
            raise NotImplementedError("this class currently only works for 'noham' model")
        self.mode = MIDMITS_MODEL_MODES[self.model_name]
        if len(self.mode) == 1:
            self.mode = self.mode[0]
        else:
            raise NotImplementedError(
                "cannot handle models with more than one mode, "
                f"this model ({self.model_name}) has {len(self.mode)} modes"
            )
        self.segmentation = {
            k: nd_core.get_segmentation_level(s) for k, s in self.SEGMENTATION.items()
        }
        self._hb_paths = None
        self._nhb_paths = None
