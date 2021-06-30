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
import collections

from typing import Any
from typing import List
from typing import Dict
from typing import Tuple

# Third Party
import pandas as pd

# Local Imports
import normits_demand as nd

from normits_demand import constants as consts

from normits_demand.utils import file_ops
from normits_demand.utils import general as du
from normits_demand.utils import pandas_utils as pd_utils


# ## CLASSES ## #
class SegmentationLevel:

    _segmentation_import_fname = "segmentations"
    _unique_segments_csv_fname = "unique_segments.csv"
    _unique_segments_compress_fname = "unique_segments.pbz2"
    _naming_order_fname = "naming_order.csv"

    _segment_definitions_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "definitions",
        "segmentations",
    )
    _multiply_definitions_path = os.path.join(
        _segment_definitions_path,
        "multiply.csv",
    )
    _aggregation_definitions_path = os.path.join(
        _segment_definitions_path,
        "aggregate.csv",
    )

    _segment_translation_dir = os.path.join(
        _segment_definitions_path,
        '_translations',
    )

    _tfn_tt_expansion_path = os.path.join(
        _segment_translation_dir,
        'tfn_tt_splits.pbz2',
    )

    _list_separator = ';'
    _translate_separator = ':'
    _segment_name_separator = '_'

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
        self.segment_names = segments_and_names['name'].to_list()

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
            return_seg = get_segmentation_level(return_seg_name)

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

        # Check that the output segmentation has been created properly
        if not other.is_correct_naming(list(multiply_dict.keys())):
            raise SegmentationError(
                "Some segment names seem to have gone missing during"
                "multiplication.\n"
                "Expected %s segments.\n"
                "Found %s segments."
                % (len(other.segment_names), len(set(multiply_dict.keys())))
            )

        return multiply_dict, return_seg

    def _read_multiply_definitions(self) -> pd.DataFrame:
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
        mult_def = self._read_multiply_definitions()

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

    def _read_aggregation_definitions(self) -> pd.DataFrame:
        """
        Returns the multiplication definitions for segments as a pd.DataFrame
        """
        return pd.read_csv(self._aggregation_definitions_path)

    def _get_tfn_tt_expansion(self) -> pd.DataFrame:
        """
        Returns the definition for expanding tfn_tt into its components.
        """
        return file_ops.read_df(self._tfn_tt_expansion_path)

    def _get_aggregation_definition(self,
                                    other: SegmentationLevel,
                                    ) -> Tuple[str, List[str]]:
        """
        Returns the common cols and any translations that are needed
        for aggregating self into other
        """
        # Init
        mult_def = self._read_aggregation_definitions()

        # Try find a definition
        df_filter = {
            'in': self.name,
            'out': other.name,
        }
        definition = pd_utils.filter_df(mult_def, df_filter)

        # If empty, we don't know what to do
        if definition.empty:
            raise SegmentationError(
                "Got no definition for aggregating '%s' into '%s'.\n"
                "If there should be a definition, please add one in "
                "at: %s"
                % (self.name, other.name, self._aggregation_definitions_path)
            )

        return (
            self._parse_join_cols(definition['common'].squeeze()),
            self._parse_translate_cols(definition['translate'].squeeze())
        )

    def _get_segment_translation(self, col1: str, col2: str) -> pd.DataFrame:
        """
        Returns the dataframe defining how to translate col1 into col2
        """
        # Init
        home_dir = self._segment_translation_dir
        base_fname = '%s_to_%s.csv'

        # Try find a translation
        fname = base_fname % (col1, col2)
        try:
            file_path = file_ops.find_filename(os.path.join(home_dir, fname))
        except FileNotFoundError:
            file_path = None

        # If not found yet, try flipping columns
        if file_path is None:
            fname = base_fname % (col2, col1)
            try:
                file_path = file_ops.find_filename(os.path.join(home_dir, fname))
            except FileNotFoundError:
                file_path = None

        # If not found again, we don't know what to do
        if file_path is None:
            raise SegmentationError(
                "Cannot translate '%s' into '%s' as no definition for the "
                "translation exists."
                % (col1, col2)
            )

        # Must exist if we are here, read in and validate
        df = file_ops.read_df(file_path)
        return pd_utils.reindex_cols(df, [col1, col2])

    def _parse_join_cols(self, join_cols: str) -> List[str]:
        """
        Parses join cols (from multiply/aggregate.csv) into a list.
        """
        lst = [x.strip() for x in join_cols.split(self._list_separator)]
        lst = du.list_safe_remove(lst, [''])
        return lst

    def _parse_translate_cols(self, translate_cols: str) -> List[Tuple[str, str]]:
        """
        Parses the translate col (from aggregate.csv) into a list.
        """
        # If not string, assume no value given
        if not isinstance(translate_cols, str):
            return None

        # Otherwise, parse
        translate_cols = str(translate_cols)
        translate_pairs = [x.strip() for x in translate_cols.split(self._list_separator)]
        translate_pairs = du.list_safe_remove(translate_pairs, [''])
        return [tuple(x.split(self._translate_separator)) for x in translate_pairs]

    def create_segment_col(self,
                           df: pd.DataFrame,
                           naming_conversion: Dict[str, str] = None,
                           ) -> pd.Series:
        """
        Creates a pd.Series of segment names based on columns in df

        Expects to find self.naming_order column names in df and builds
        the segment name based off of that. If these columns do not exist,
        then naming_conversion needs to be supplied in order to convert
        the current df columns into naming_conversion names.

        Parameters
        ----------
        df:
            The dataframe containing the segmentation columns to use when
            generating the segmentation names.

        naming_conversion:
            A dictionary mapping segment names in self.naming order into
            df columns names. e.g.
            {segment_name: column_name}

        Returns
        -------
        segment_col:
            A pandas.Series containing segment names, indexed the same as
            df. i.e. it can be added to df as an extra column and it will be
            in the correct order.

        Raises
        ------
        ValueError:
            If any of keys of naming_conversion are not a valid segment_name.

        ValueError:
            If any of values of naming_conversion are not a valid column
            name in df.

        ValueError:
            If naming_conversion is not given, and all segment names cannot
            be found in df.
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

    def get_seg_dict(self, segment_name: str) -> Dict[str, Any]:
        """
        Converts the given segment_name into a {seg_name: seg_val} dict
        """
        if not self.is_valid_segment_name(segment_name):
            raise ValueError(
                "'%s' is not a valid segment name for segmentation level %s."
                % (segment_name, self.name)
            )

        name_parts = segment_name.split(self._segment_name_separator)
        return {n: s for n, s in zip(self.naming_order, name_parts)}

    def is_valid_segment_name(self, segment_name: str) -> bool:
        """
        Checks whether the given segment_name is a valid name for this
        SegmentationLevel
        """
        return segment_name in self.segment_names

    def aggregate(self,
                  other: SegmentationLevel
                  ) -> Dict[str, List[str]]:
        """
        Generates a dict defining how to aggregate this segmentation into other.

        Parameters
        ----------
        other:
            The SegmentationLevel to aggregate this segmentation into.

        Returns
        -------
        aggregation_dict:
            A dictionary defining how to aggregate self into out_segmentation.
            Will be in the form of {out_seg: [in_seg]}.
            Where out seg is a segment name of out_segmentation, and in_seg
            is a list of segment names from self that should be summed to
            generate out_seg.
        """
        # Validate input
        if not isinstance(other, SegmentationLevel):
            raise ValueError(
                "out_segmentation is not the correct type. "
                "Expected SegmentationLevel, got %s"
                % type(other)
            )
        
        join_cols, translate_cols = self._get_aggregation_definition(other)

        # Translate any columns we need to in order to join
        self_segments = self.segments.copy()
        if translate_cols is not None:
            for in_col, out_col in translate_cols:
                translation = self._get_segment_translation(in_col, out_col)

                # Translate
                self_segments = pd.merge(
                    self_segments,
                    translation,
                    how='left',
                    on=[in_col],
                )

                # We now need to join on this translated column
                join_cols += [out_col]

        # ## FIGURE OUT HOW TO AGGREGATE ## #
        # Merge, so we know how these segments combine
        seg_agg = pd.merge(
            left=self_segments,
            right=other.segments,
            on=join_cols
        )

        # Extract the segment names for self and other
        seg_agg['self_name'] = self.create_segment_col(seg_agg)
        seg_agg['other_name'] = other.create_segment_col(seg_agg)

        # Convert into the aggregation dict
        agg_dict = collections.defaultdict(list)
        for o, s in zip(seg_agg['other_name'], seg_agg['self_name']):
            agg_dict[o].append(s)

        # Check that the output segmentation has been created properly
        if not other.is_correct_naming(list(agg_dict.keys())):
            raise SegmentationError(
                "Some segment names seem to have gone missing during"
                "aggregation.\n"
                "Expected %s segments.\n"
                "Found %s segments."
                % (len(other.segment_names), len(set(agg_dict.keys())))
            )

        return agg_dict

    def split_tfntt_segmentation(self,
                                 other: SegmentationLevel
                                 ) -> Dict[str, List[str]]:
        """
        Generates a dict defining how to aggregate this segmentation into other.

        Splits the tfn_tt segment into it's components and aggregates up to
        out_segmentation. The DVector needs to be using a segmentation that
        contains tfn_tt and p in order for this to work.

        Parameters
        ----------
        other:
            The segmentation level the output vector should be in.

        Returns
        -------
        new_dvector:
            A new DVector containing the same data but in using
            out_segmentation as its segmentation level.

        Raises
        ------
        SegmentationError:
            If the DVector does not contain both tfn_tt and p segments.
        """
        # Validate inputs
        if not isinstance(other, SegmentationLevel):
            raise ValueError(
                "out_segmentation is not the correct type. "
                "Expected SegmentationLevel, got %s"
                % type(other)
            )

        if 'tfn_tt' not in self.naming_order:
            raise nd.SegmentationError(
                "This Dvector is not using a segmentation which contains "
                "'tfn_tt'. Need to be using a segmentation using both 'tfn_tt' "
                "and 'p' segments in order to split tfn_tt."
                "Current segmentation uses: %s"
                % self.naming_order
            )

        if 'p' not in self.naming_order:
            raise nd.SegmentationError(
                "This Dvector is not using a segmentation which contains "
                "'p'. Need to be using a segmentation using both 'tfn_tt' "
                "and 'p' segments in order to split tfn_tt."
                "Current segmentation uses: %s"
                % self.naming_order
            )

        # Expand tfn_tt
        tfn_tt_expansion = self._get_tfn_tt_expansion()
        full_segmentation = pd.merge(
            self.segments,
            tfn_tt_expansion,
            how='left',
            on='tfn_tt',
        )
        full_segmentation['self_name'] = self.create_segment_col(full_segmentation)

        # Aggregate soc and ns depending on p segment
        soc_mask = full_segmentation['p'].isin(consts.SOC_P)
        soc_df = full_segmentation[soc_mask]
        ns_df = full_segmentation[~soc_mask]
        del full_segmentation

        soc_df['ns'] = 0
        ns_df['soc'] = 0
        seg_agg = pd.concat([soc_df, ns_df])

        # Create the new segment names
        seg_agg['other_name'] = other.create_segment_col(seg_agg)

        # Convert into the aggregation dict
        agg_dict = collections.defaultdict(list)
        for o, s in zip(seg_agg['other_name'], seg_agg['self_name']):
            agg_dict[o].append(s)

        # Check that the output segmentation has been created properly
        if not other.is_correct_naming(list(agg_dict.keys())):
            raise SegmentationError(
                "Some segment names seem to have gone missing during"
                "aggregation.\n"
                "Expected %s segments.\n"
                "Found %s segments."
                % (len(other.segment_names), len(set(agg_dict.keys())))
            )

        return agg_dict

    def is_correct_naming(self, lst: List[str]) -> bool:
        """
        Checks whether lst is a complete list of all names of this segmentation.

        Parameters
        ----------
        lst:
            A list of segment names to check

        Returns
        -------
        is_correct_naming:
            True if all names in lst are valid, and lst contains all segment
            names. Otherwise False.
        """
        # Init
        names = set(lst)

        # Return False if not all names are valid
        if not all([x in self.segment_names for x in names]):
            return False

        # If the lists are the same length, we can imply all names are contained
        if not len(names) == len(self.segment_names):
            return False

        # If we are here, then must be True
        return True


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


def _get_valid_segments(name: str) -> pd.DataFrame:
    """
    Finds and reads in the valid segments data for segmentation with name
    """
    # ## DETERMINE THE IMPORT LOCATION ## #
    import_home = os.path.join(SegmentationLevel._segment_definitions_path,  name)

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
    # Build the two possible paths
    compress_fname = SegmentationLevel._unique_segments_compress_fname
    csv_fname = SegmentationLevel._unique_segments_csv_fname

    compress_path = os.path.join(import_home, compress_fname)
    csv_path = os.path.join(import_home, csv_fname)

    # Determine which path to use
    file_path = compress_path
    if not os.path.isfile(compress_path):
        file_path = csv_path
        if not os.path.isfile(csv_path):
            # Can't find either!
            raise nd.NormitsDemandError(
                "We don't seem to have any valid segment data for the segmentation %s.\n"
                "Tried looking for the data here:"
                "%s\n"
                "%s"
                % (name, compress_path, csv_path)
            )

    # Read in the file
    df = file_ops.read_df(file_path, find_similar=True)

    # Tidy up the column names to match the naming_order
    rename_cols = {c: c.lower() for c in list(df)}
    df = df.rename(columns=rename_cols)
    df = pd_utils.reindex_cols(df, naming_order)

    return df, naming_order


def get_segmentation_level(name: str) -> SegmentationLevel:
    # TODO(BT): Write docs!
    # TODO(BT): Add some validation on the segmentation name
    # TODO(BT): Instantiate import drive for these on module import!
    # TODO(BT): Add some caching to this function!

    valid_segments, naming_order = _get_valid_segments(name)

    # Create the ZoningSystem object and return
    return SegmentationLevel(
        name=name,
        naming_order=naming_order,
        valid_segments=valid_segments,
    )
