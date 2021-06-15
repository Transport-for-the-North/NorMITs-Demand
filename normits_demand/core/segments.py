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
# Builtins
import os

from typing import List
from typing import Dict

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
                raise nd.NormitsDemandError(
                    "Error while reading in the segmentation naming order at: %s\n"
                    "There appears to be more than one name on line: %s"
                    % (path, i)
                )

            # Clean up value, add to list
            order.append(line.strip().lower())

    if order == list():
        raise nd.NormitsDemandError(
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
