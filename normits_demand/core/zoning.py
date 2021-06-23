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
# Allow class self hinting
from __future__ import annotations

# Builtins
import os

# Third Party
import numpy as np
import pandas as pd

# Local Imports
import normits_demand as nd

from normits_demand.utils import file_ops
from normits_demand.utils import pandas_utils as pd_utils


class ZoningSystem:

    _zoning_system_import_fname = "zoning_systems"
    _zones_csv_fname = "zones.csv"
    _zones_compress_fname = "zones.pbz2"

    _zoning_definitions_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "definitions",
        "zoning_systems",
    )

    def __init__(self,
                 name: str,
                 unique_zones: np.ndarray,
                 ) -> ZoningSystem:
        # Init
        self.name = name
        self.unique_zones = np.sort(unique_zones)
        self.n_zones = len(self.unique_zones)

    def __eq__(self, other) -> bool:
        """Overrides the default implementation"""
        # May need to update in future, but assume they are equal if names match
        if isinstance(other, ZoningSystem):
            return self.name == other.name
        return False

    def __ne__(self, other) -> bool:
        """Overrides the default implementation"""
        return not self.__eq__(other)


class ZoningError(nd.NormitsDemandError):
    """
    Exception for all errors that occur around zone management
    """
    def __init__(self, message=None):
        self.message = message
        super().__init__(self.message)


# ## FUNCTIONS ##
def _get_unique_zones(name: str) -> np.ndarray:
    """
    Finds and reads in the unique zone data for zoning system with name
    """
    # ## DETERMINE THE IMPORT LOCATION ## #
    import_home = os.path.join(ZoningSystem._zoning_definitions_path, name)

    # Make sure the import location exists
    if not os.path.exists(import_home):
        raise nd.NormitsDemandError(
            "We don't seem to have any data for the zoning system %s.\n"
            "Tried looking for the data here: %s"
            % (name, import_home)
        )

    # ## READ IN THE UNIQUE ZONES ## #
    # Build the two possible paths
    compress_fname = ZoningSystem._zones_compress_fname
    csv_fname = ZoningSystem._zones_csv_fname

    compress_path = os.path.join(import_home, compress_fname)
    csv_path = os.path.join(import_home, csv_fname)

    # Determine which path to use
    file_path = compress_path
    if not os.path.isfile(compress_path):
        file_path = csv_path
        if not os.path.isfile(csv_path):
            # Can't find either!
            raise nd.NormitsDemandError(
                "We don't seem to have any zone data for the zoning system %s.\n"
                "Tried looking for the data here:"
                "%s\n"
                "%s"
                % (name, compress_path, csv_path)
            )

    # Read in the file
    df = file_ops.read_df(file_path)
    df = pd_utils.reindex_cols(df, columns=['zone_name'])

    # Sort the return to make sure it's always the same order
    return np.sort(df['zone_name'].values)


def get_zoning_system(name: str) -> ZoningSystem:
    # TODO(BT): Write docs!
    # TODO(BT): Add some validation on the zone name
    # TODO(BT): Instantiate import drive for these on module import!
    # TODO(BT): Add some caching to this function!

    # Create the ZoningSystem object and return
    return ZoningSystem(
        name=name,
        unique_zones=_get_unique_zones(name)
    )
