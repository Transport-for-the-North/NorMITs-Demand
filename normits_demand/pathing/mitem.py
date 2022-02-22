# -*- coding: utf-8 -*-
"""
    Classes which build the paths for the MiTEM model inputs
    and outputs, all classes are based on NoTEM versions.

    See Also
    --------
    `normits_demand.pathing.notem`
"""

##### IMPORTS #####
# Standard imports

# Third party imports

# Local imports
import normits_demand.pathing.notem as notem_paths


##### CLASSES #####
class MiTEMImportPaths(notem_paths.NoTEMImportPaths):
    """The default MiTEM import paths class.

    Defines the default input paths for MiTEM.
    Currently doesn't change anything from `NoTEMImportPaths`.

    See Also
    --------
    `normits_demand.pathing.notem.NoTEMImportPaths`
    """


class MiTEMExportPaths(notem_paths.NoTEMExportPaths):
    """Export paths class for the MiTEM model.

    See Also
    --------
    `normits_demand.pathing.notem.NoTEMExportPaths`
    """
