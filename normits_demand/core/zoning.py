# -*- coding: utf-8 -*-
"""
Created on: Tues May 25th 15:04:32 2021
Updated on:

Original author: Ben Taylor
Last update made by:
Other updates made by:

File purpose:
Holds the ZoningSystem Class which stores all information on different zoning
systems
"""
# Builtins
import os

# Third Party
import numpy as np
import pandas as pd

# Local Imports
import normits_demand as nd


class ZoningSystem:

    _normits_demand_name = "NorMITs Demand"
    _core_subpath = os.path.join("import", "core_dtypes")
    _zoning_system_import_fname = "zoning_systems"
    _zones_fname = "zones.csv"

    def __init__(self,
                 name: str,
                 unique_zones: np.ndarray,
                 ):
        # Init
        self.name = name
        self.unique_zones = np.sort(unique_zones)


# ## FUNCTIONS ##
def _get_unique_zones(name: str, import_drive: nd.PathLike) -> np.ndarray:
    """
    Finds and reads in the unique zone data for zoning system with name
    """
    # ## DETERMINE THE IMPORT LOCATION ## #
    import_home = os.path.join(
        import_drive,
        ZoningSystem._normits_demand_name,
        ZoningSystem._core_subpath,
        ZoningSystem._zoning_system_import_fname,
        name,
    )

    # Make sure the import location exists
    if not os.path.exists(import_home):
        raise nd.NormitsDemandError(
            "We don't seem to have any data for the zoning system %s.\n"
            "Tried looking for the data here: %s"
            % (name, import_home)
        )

    # ## READ IN THE UNIQUE ZONES ## #
    # Init
    file_path = os.path.join(import_home, ZoningSystem._zones_fname)

    # Check the file exists
    if not os.path.isfile(file_path):
        raise nd.NormitsDemandError(
            "We don't seem to have any zone data for the zoning system %s.\n"
            "Tried looking for the data here: %s"
            % (name, file_path)
        )

    # Read in the file
    # TODO(BT): Might need some more error checking on this read in
    df = pd.read_csv(file_path, usecols=['zone_name'])

    # Sort the return to make sure it's always the same order
    return np.sort(df['zone_name'].values)


def get_zoning_system(name: str, import_drive: nd.PathLike) -> ZoningSystem:
    # TODO(BT): Write docs!
    # TODO(BT): Add some validation on the zone name
    # TODO(BT): Instantiate import drive for these on module import!
    # TODO(BT): Add some caching to this function!

    # Create the ZoningSystem object and return
    return ZoningSystem(
        name=name,
        unique_zones=_get_unique_zones(name, import_drive)
    )
