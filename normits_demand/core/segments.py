# -*- coding: utf-8 -*-
"""
Created on: Tues May 25th 15:04:32 2021
Updated on:

Original author: Ben Taylor
Last update made by:
Other updates made by:

File purpose:
Holds the SegmentationLevel Class which stores all information on segmentations
"""
# Allow class self hinting
from __future__ import annotations

# Builtins
import os

from typing import List
from typing import Dict
from typing import Tuple

# Third Party
import pandas as pd

# Local Imports
import normits_demand as nd

from normits_demand.utils import pandas_utils as pd_utils


# ## CLASSES ## #
class SegmentationLevel:

    # TODO(BT): Move some of these into constants. Maybe core consts?
    _normits_demand_name = "NorMITs Demand"
    _core_subpath = os.path.join("import", "core_dtypes")
    _segmentation_import_fname = "segmentations"
    _unique_segments_fname = "unique_segments.csv"
    _naming_order_fname = "naming_order.csv"

    _multiply_definitions_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "definitions",
        "segmentations",
        "multiply.csv",
    )
    _join_separator = ';'

    def __init__(self,
                 name: str,
                 naming_order: List[str],
                 valid_segments: pd.DataFrame,
                 ):
        # Init
        self.name = name
        self.naming_order = naming_order

        # Validate that naming order is in df
        for col in self.naming_order:
            if col not in valid_segments:
                raise nd.NormitsDemandError(
                    "Error while instantiating a SegmentationLevel object."
                    "Cannot find column '%s' of naming order in the given "
                    "valid segments dataframe!"
                    % col
                )

        # Make sure the df is just the segment columns
        self.segments = pd_utils.reindex_cols(valid_segments, self.naming_order)

        # ## BUILD SEGMENT NAMING ## #
        segments_and_names = self.segments.copy()
        segments_and_names['name'] = pd_utils.str_join_cols(
            df=segments_and_names,
            columns=self.naming_order,
        )
        self.segments_and_names = segments_and_names
        self.segment_names = segments_and_names['name']

    def __eq__(self, other) -> bool:
        """Overrides the default implementation"""
        # May need to update in future, but assume they are equal if names match
        if isinstance(other, SegmentationLevel):
            return self.name == other.name
        return False

    def __ne__(self, other) -> bool:
        """Overrides the default implementation"""
        return not self.__eq__(other)

    def __mul__(self, other) -> Tuple[nd.SegmentMultiplyDict, SegmentationLevel]:
        """
        Multiply two SegmentationLevel objects

        Returns a dictionary defining how to multiply the two
        SegmentationLevel objects, along with a SegmentationLevel defining
        the return segmentation of the multiplication.

        Returns
        -------
        SegmentMultiplyDict:
            A dictionary where they keys are the resultant segment names,
            and the value is a tuple where the first value
            corresponds to the segment_name in this SegmentationLevel to use,
            and the second value is the other SegmentationLevel to use.

        SegmentationLevel:
            A SegmentationLevel object defining what the return segmentation
            would be if two Dvectors with the corresponding SegmentationLevels
            were multiplied.
        """
        # Check we can do it
        if not isinstance(other, SegmentationLevel):
            raise SegmentationError(
                "The __mul__ operator can only be used with."
                "SegmentationLevel objects on both sides."
            )

        # Get the multiplication definition
        return_seg_name, join_cols = self._get_multiply_definition(other)

        # Build the return segmentation
        if return_seg_name == self.name:
            return_seg = self
        elif return_seg_name == other.name:
            return_seg = other
        else:
            # TODO(BT): Move definitions into codebase
            return_seg = get_segmentation_level(return_seg_name, "I:/")

        # ## FIGURE OUT HOW TO MULTIPLY BASED ON DEFINITION ## #
        # Merge, so we know how these segments combine
        seg_mult = pd.merge(
            left=self.segments,
            right=other.segments,
            on=join_cols
        )

        # Extract the segment names for returning, self, and other
        seg_mult['return_name'] = return_seg.create_segment_col(seg_mult)
        seg_mult['self_name'] = self.create_segment_col(seg_mult)
        seg_mult['other_name'] = other.create_segment_col(seg_mult)

        # Convert into the multiply dict
        r = seg_mult['return_name']
        s = seg_mult['self_name']
        o = seg_mult['other_name']
        multiply_dict = dict(zip(r, zip(s, o)))

        return multiply_dict, return_seg

    def _get_multiply_definitions(self) -> pd.DataFrame:
        """
        Returns the multiplication definitions for segments as a pd.DataFrame
        """
        return pd.read_csv(self._multiply_definitions_path)

    def _get_multiply_definition(self,
                                 other: SegmentationLevel,
                                 ) -> Tuple[str, List[str]]:
        """
        Returns the return_seg_name and join cols for multiplying
        self and other
        """
        # Init
        mult_def = self._get_multiply_definitions()

        # Try find a definition
        df_filter = {
            'a': self.name,
            'b': other.name,
        }
        definition = pd_utils.filter_df(mult_def, df_filter)

        # If none found, try flipping a and b
        if definition.empty:
            df_filter = {
                'b': self.name,
                'a': other.name,
            }
            definition = pd_utils.filter_df(mult_def, df_filter)

            # If empty again, we don't know what to do
            if definition.empty:
                raise SegmentationError(
                    "Got no definition for multiplying '%s' by '%s'.\n"
                    "If there should be a definition, please add one in "
                    "at: %s"
                    % (self.name, other.name, self._multiply_definitions_path)
                )

        return (
            definition['out'].squeeze(),
            self._parse_join_cols(definition['join'].squeeze())
        )

    def _parse_join_cols(self, join_cols: str) -> List[str]:
        """
        Parses join cols (from multiply_definitions) into a list.
        """
        return [x.strip() for x in join_cols.split(self._join_separator)]

    def create_segment_col(self,
                           df: pd.DataFrame,
                           naming_conversion: Dict[str, str] = None,
                           ) -> pd.Series:
        """
        TODO(BT): Properly document this function!!

        Parameters
        ----------
        df:
        naming_conversion:

        Returns
        -------

        """
        # Init
        if naming_conversion is None:
            naming_conversion = {x: x for x in self.naming_order}

        # ## VALIDATE ARGS ## #
        # Check the keys are valid segments
        for key in naming_conversion.keys():
            if key not in self.naming_order:
                raise ValueError(
                    "Key '%s' in naming conversion is not a valid segment "
                    "name for this segmentation. Expecting one of: %s"
                    % (key, self.naming_order)
                )

        # Check that the values given are in df
        for value in naming_conversion.values():
            if value not in df:
                raise ValueError(
                    "No column named '%s' in the given dataframe."
                    "Found columns: %s"
                    % (value, list(df))
                )

        # Rename columns as needed
        df = df.rename(columns={v: k for k, v in naming_conversion.items()})

        # Generate the naming column, and return
        return pd_utils.str_join_cols(df, self.naming_order)


class SegmentationError(nd.NormitsDemandError):
    """
    Exception for all errors that occur around zone management
    """
    def __init__(self, message=None):
        self.message = message
        super().__init__(self.message)


# ## FUNCTIONS ## #
def _read_in_and_validate_naming_order(path: nd.PathLike, name: str) -> List[str]:
    """
    Converts the given csv file into a list of column names
    """
    # Check the file exists
    if not os.path.isfile(path):
        raise nd.NormitsDemandError(
            "We don't seem to have any naming order data for the segmentation %s.\n"
            "Tried looking for the data here: %s"
            % (name, path)
        )

    # Read in and validate each row
    order = list()
    with open(path) as f:
        for i, line in enumerate(f):

            # Make sure there is only one value on this line
            if ',' in line:
                raise SegmentationError(
                    "Error while reading in the segmentation naming order at: %s\n"
                    "There appears to be more than one name on line: %s"
                    % (path, i)
                )

            # Clean up value, add to list
            order.append(line.strip().lower())

    if order == list():
        raise SegmentationError(
            "Error while reading in the segmentation naming order at: %s\n"
            "There does not appear to be any names in this file!"
            % path
        )

    return order


def _get_valid_segments(name: str,
                        import_drive: nd.PathLike,
                        ) -> pd.DataFrame:
    """
    Finds and reads in the valid segments data for segmentation with name
    """
    # ## DETERMINE THE IMPORT LOCATION ## #
    import_home = os.path.join(
        import_drive,
        SegmentationLevel._normits_demand_name,
        SegmentationLevel._core_subpath,
        SegmentationLevel._segmentation_import_fname,
        name,
    )

    # Make sure the import location exists
    if not os.path.exists(import_home):
        raise nd.NormitsDemandError(
            "We don't seem to have any data for the segmentation %s.\n"
            "Tried looking for the data here: %s"
            % (name, import_home)
        )

    # ## READ IN THE NAMING ORDER ## #
    file_path = os.path.join(import_home, SegmentationLevel._naming_order_fname)
    naming_order = _read_in_and_validate_naming_order(file_path, name)

    # ## READ IN THE UNIQUE SEGMENTS ## #
    # Init
    file_path = os.path.join(import_home, SegmentationLevel._unique_segments_fname)

    # Check the file exists
    if not os.path.isfile(file_path):
        raise nd.NormitsDemandError(
            "We don't seem to have any valid segment data for the segmentation %s.\n"
            "Tried looking for the data here: %s"
            % (name, file_path)
        )

    # Read in the file
    # TODO(BT): Might need some more error checking on this read in
    df = pd.read_csv(file_path)

    # Tidy up the column names to match the naming_order
    rename_cols = {c: c.lower() for c in list(df)}
    df = df.rename(columns=rename_cols)
    df = pd_utils.reindex_cols(df, naming_order)

    # Sort the return to make sure it's always the same order
    return df, naming_order


def get_segmentation_level(name: str, import_drive: nd.PathLike) -> SegmentationLevel:
    # TODO(BT): Write docs!
    # TODO(BT): Add some validation on the segmentation name
    # TODO(BT): Instantiate import drive for these on module import!
    # TODO(BT): Add some caching to this function!

    valid_segments, naming_order = _get_valid_segments(name, import_drive)

    # Create the ZoningSystem object and return
    return SegmentationLevel(
        name=name,
        naming_order=naming_order,
        valid_segments=valid_segments,
    )
