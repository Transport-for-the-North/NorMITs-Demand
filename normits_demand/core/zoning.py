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

    _translation_dir = os.path.join(
        _zoning_definitions_path,
        '_translations'
    )

    _translate_infill = 0
    _translate_base_zone_col = "%s_zone_id"
    _translate_base_trans_col = "%s_to_%s"

    _default_weighting_suffix = 'correspondence'
    _weighting_suffix = {
        'population': 'lsoa_population_weight',
        'employment': 'lsoa_employment_weight',
    }

    possible_weightings = list(_weighting_suffix.keys()) + [None]

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

    def _get_weighting_suffix(self, weighting: str) -> str:
        """
        Takes a weighting name and converts it into a file suffix
        """
        if weighting is None:
            return self._default_weighting_suffix
        return self._weighting_suffix[weighting]

    def _get_translation_definition(self,
                                    other: ZoningSystem,
                                    weighting: str = None,
                                    ) -> pd.DataFrame:
        """
        Returns a long dataframe defining how to translate from self to other.
        """
        # Init
        home_dir = self._translation_dir
        base_fname = '%s_to_%s_%s.csv'
        weight_name = self._get_weighting_suffix(weighting)

        # Try find a translation
        fname = base_fname % (self.name, other.name, weight_name)
        try:
            file_path = file_ops.find_filename(os.path.join(home_dir, fname))
        except FileNotFoundError:
            file_path = None

        # If not found yet, try flipping columns
        if file_path is None:
            fname = base_fname % (other.name, self.name, weight_name)
            try:
                file_path = file_ops.find_filename(os.path.join(home_dir, fname))
            except FileNotFoundError:
                file_path = None

        # If not found again, we don't know what to do
        if file_path is None:
            raise ZoningError(
                "Cannot translate '%s' into '%s' as no definition for the "
                "translation exists."
                % (self.name, other.name)
            )

        # Must exist if we are here, read in and validate
        df = file_ops.read_df(file_path)
        index_cols = [
            self._translate_base_zone_col % self.name,
            self._translate_base_zone_col % other.name,
            self._translate_base_trans_col % (self.name, other.name)
        ]
        return pd_utils.reindex_cols(df, index_cols)

    def translate(self,
                  other: ZoningSystem,
                  weighting: str = None,
                  ) -> np.array:
        """
        Returns a numpy array defining the translation of self to other

        Parameters
        ----------
        other:
            The zoning system to translate this zoning system into

        weighting:
            The weighting to use when building the translation. Must be None,
            or one of ZoningSystem.possible_weightings

        Returns
        -------
        translations_array:
            A numpy array defining the weights to use for the translation.
            The rows correspond to self.unique_zones
            The columns correspond to other.unique_zones

        Raises
        ------
        ZoningError:
            If a translation definition between self and other cannot be found
        """
        # Validate input
        if not isinstance(other, ZoningSystem):
            raise ValueError(
                "other is not the correct type. "
                "Expected ZoningSystem, got %s"
                % type(other)
            )

        if weighting not in self.possible_weightings:
            raise ValueError(
                "%s is not a valid weighting for a translation. "
                "Expected one of: %s"
                % (weighting, self.possible_weightings)
            )

        # Get a numpy array to define the translation
        translation_df = self._get_translation_definition(other, weighting)
        translation = pd_utils.long_to_wide_infill(
            df=translation_df,
            index_col=self._translate_base_zone_col % self.name,
            columns_col=self._translate_base_zone_col % other.name,
            values_col=self._translate_base_trans_col % (self.name, other.name),
            index_vals=self.unique_zones,
            column_vals=other.unique_zones,
            infill=self._translate_infill
        )

        return translation.values


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
