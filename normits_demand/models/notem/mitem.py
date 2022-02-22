# -*- coding: utf-8 -*-
"""
    MiTEM frontend for calling all production and attraction models,
    built on top of `NoTEM` class.

    See Also
    --------
    `normits_demand.models.notem.notem`
"""

##### IMPORTS #####
# Standard imports

# Third party imports

# Local imports
from normits_demand.models.notem import NoTEM
from normits_demand.pathing import MiTEMExportPaths


##### CLASSES #####
class MiTEM(NoTEM):
    """MiTEM frontend for calling all production and attraction models.

    See Also
    --------
    `normits_demand.models.notem.notem.NoTEM`
    """
    EXPORT_PATHS_CLASS = MiTEMExportPaths
    _log_fname = "MiTEM_log.log"
