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
import warnings

from typing import Tuple
from typing import Optional

# Third Party
import numpy as np
import pandas as pd

# Local Imports
import normits_demand as nd

from normits_demand.utils import file_ops
from normits_demand.utils import pandas_utils as pd_utils


class ZoningSystem:
    """Zoning definitions to provide common interface

    Attributes
    ----------
    name:
        The name of the zoning system. This will be the same as the name in
        the definitions folder

    col_name:
        The default naming that should be given to this zoning system if
        defined to a pandas.DataFrame

    unique_zones:
        A sorted numpy array of unique zone names for this zoning system.

    n_zones:
        The number of zones in this zoning system
    """
    # Constants
    __version__ = nd.__version__

    _zoning_system_import_fname = "zoning_systems"
    _base_col_name = "%s_zone_id"

    # File names
    _valid_ftypes = ['.csv', '.pbz2']
    _zones_csv_fname = "zones.csv"
    _zones_compress_fname = "zones.pbz2"
    _internal_zones_fname = "internal_zones.csv"
    _external_zones_fname = "external_zones.csv"

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
        'population': 'population_weight',
        'employment': 'employment_weight',
        'no_weight': 'no_weighting',
        'average': 'weighted_average',
    }

    possible_weightings = list(_weighting_suffix.keys()) + [None]

    def __init__(self,
                 name: str,
                 unique_zones: np.ndarray,
                 internal_zones: Optional[np.ndarray] = None,
                 external_zones: Optional[np.ndarray] = None,
                 ) -> ZoningSystem:
        """Builds a ZoningSystem

        This class should almost never be constructed directly. If an
        instance of ZoningSystem is needed, the helper function
        `get_zoning_system()` should be used instead.

        Parameters
        ----------
        name:
            The name of the zoning system to create.

        unique_zones:
            A numpy array of unique zone names for this zoning system.

        internal_zones:
            A numpy array of unique zone names that make up the "internal"
            area of this zoning system. Every value in this array must also
            be contained in unique_zones.

        external_zones:
            A numpy array of unique zone names that make up the "external"
            area of this zoning system. Every value in this array must also
            be contained in unique_zones.
        """
        # Init
        self._name = name
        self._col_name = self._base_col_name % name
        self._unique_zones = np.sort(unique_zones)
        self._n_zones = len(self.unique_zones)

        # Validate and assign the internal and external zones
        self._internal_zones = None
        self._external_zones = None

        if internal_zones is not None:
            extra_internal_zones = set(internal_zones) - set(unique_zones)
            if len(extra_internal_zones) > 0:
                raise ValueError(
                    "Not all of the given values for internal zones are also "
                    "defined in the zoning system unique zones. Check the zones "
                    "definition file for the following zones:\n%s"
                    % extra_internal_zones
                )
            self._internal_zones = internal_zones

        if external_zones is not None:
            extra_external_zones = set(external_zones) - set(unique_zones)
            if len(extra_external_zones) > 0:
                raise ValueError(
                    "Not all of the given values for internal zones are also "
                    "defined in the zoning system unique zones. Check the zones "
                    "definition file for the following zones:\n%s"
                    % extra_external_zones
                )
            self._external_zones = external_zones

    @property
    def name(self):
        return self._name

    @property
    def col_name(self):
        return self._col_name

    @property
    def unique_zones(self):
        return self._unique_zones

    @property
    def n_zones(self):
        return self._n_zones

    @property
    def internal_zones(self):
        if self._internal_zones is None:
            raise ZoningError(
                "No definition for internal zones has been set for this "
                "zoning system. Name: %s"
                % self.name
            )
        return self._internal_zones

    @property
    def external_zones(self):
        if self._external_zones is None:
            raise ZoningError(
                "No definition for external zones has been set for this "
                "zoning system. Name: %s"
                % self.name
            )
        return self._external_zones

    def __copy__(self):
        """Returns a copy of this class"""
        return self.copy()

    def __eq__(self, other) -> bool:
        """Overrides the default implementation"""
        # May need to update in future, but assume they are equal if names match
        if not isinstance(other, ZoningSystem):
            return False

        # Make sure names, unique zones, and n_zones are all the same
        if self.name != other.name:
            return False

        if set(self.unique_zones) != set(other.unique_zones):
            return False

        if self.n_zones != other.n_zones:
            return False

        return True

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

        # Must exist if we are here, read in
        df = file_ops.read_df(file_path)

        # Assume index col is weights
        trans_col = self._translate_base_trans_col % (self.name, other.name)
        df[trans_col] = df[trans_col].astype(np.float64)

        # Keep only the columns we need
        index_cols = [
            self._translate_base_zone_col % self.name,
            self._translate_base_zone_col % other.name,
            trans_col
        ]
        return pd_utils.reindex_cols(df, index_cols)

    def copy(self):
        """Returns a copy of this class"""
        return ZoningSystem(
            name=self.name,
            unique_zones=self.unique_zones.copy(),
        )

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
def _get_zones(name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    # Sort to make sure it's always the same order
    unique_zones = np.sort(df['zone_name'].values)

    # ## READ IN THE INTERNAL AND EXTERNAL ZONES ## #
    internal_zones = None
    external_zones = None

    # Read in the internal zones
    try:
        file_path = os.path.join(import_home, ZoningSystem._internal_zones_fname)
        file_path = file_ops.find_filename(file_path, alt_types=ZoningSystem._valid_ftypes)
        if os.path.isfile(file_path):
            df = file_ops.read_df(file_path)
            internal_zones = np.sort(df['zone_name'].values)
    except FileNotFoundError:
        warn_msg = (
            "No internal zones definition found for zoning system '%s'"
            % name
        )
        warnings.warn(warn_msg, UserWarning, stacklevel=3)

    # Read in the external zones
    try:
        file_path = os.path.join(import_home, ZoningSystem._external_zones_fname)
        file_path = file_ops.find_filename(file_path, alt_types=ZoningSystem._valid_ftypes)
        if os.path.isfile(file_path):
            df = file_ops.read_df(file_path)
            external_zones = np.sort(df['zone_name'].values)
    except FileNotFoundError:
        warn_msg = (
            "No external zones definition found for zoning system '%s'"
            % name
        )
        warnings.warn(warn_msg, UserWarning, stacklevel=3)

    return unique_zones, internal_zones, external_zones


def get_zoning_system(name: str) -> ZoningSystem:
    """
    Creates a ZoningSystem for zoning with name.

    Parameters
    ----------
    name:
        The name of the zoning system to get a ZoningSystem object for.

    Returns
    -------
    zoning_system:
        A ZoningSystem object for zoning system with name
    """
    # TODO(BT): Add some validation on the zone name
    # TODO(BT): Add some caching to this function!
    # Look for zone definitions
    unique, internal, external = _get_zones(name)

    # Create the ZoningSystem object and return
    return ZoningSystem(
        name=name,
        unique_zones=unique,
        internal_zones=internal,
        external_zones=external,
    )
