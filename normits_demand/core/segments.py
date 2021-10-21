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
import math
import itertools
import collections

from typing import Any
from typing import List
from typing import Dict
from typing import Tuple
from typing import Optional

# Third Party
import pandas as pd
import numpy as np

# Local Imports
import normits_demand as nd

from normits_demand import constants as consts
from normits_demand.concurrency import multiprocessing

from normits_demand.utils import file_ops
from normits_demand.utils import general as du
from normits_demand.utils import pandas_utils as pd_utils


# ## CLASSES ## #
class SegmentationLevel:
    """Segmentation definitions to provide common interface

    Attributes
    ----------
    name:
        The name of the segmentation. This will be the same as the name in
        the definitions folder

    naming_order:
        A list of segment names, defining the order that the segments should
        be named in when assigned a segment name. E.g. a naming order of
        ['p', 'm'], and the segment where p=1, and m=3, would be named
        '1_3'.

    segments:
        A pandas.DataFrame listing every single valid combination of segment
        values for this segmentation. Often this is simply to product of all
        possible values, but that is not always the case. Columns will be
        named after the segment they relate to.

    segment_names:
        A list of valid segment names, as defined by the valid segments in
        segments, and the naming order.

    segments_and_names:
        The pandas.DataFrame from segments, with segment names attached on
        the relevant segments. An additional column will be added titled
        'name' with the segment names in.
    """
    # Constants
    __version__ = nd.__version__

    # Special segment names
    _time_period_segment_name = 'tp'

    _weekday_time_periods = [1, 2, 3, 4]
    _weekend_time_periods = [5, 6]

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
    _expand_definitions_path = os.path.join(
        _segment_definitions_path,
        "expand.csv",
    )
    _aggregation_definitions_path = os.path.join(
        _segment_definitions_path,
        "aggregate.csv",
    )
    _subset_definitions_path = os.path.join(
        _segment_definitions_path,
        "subset.csv",
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
    _drop_splitter = ':'
    _segment_name_separator = '_'

    def __init__(self,
                 name: str,
                 naming_order: List[str],
                 valid_segments: pd.DataFrame,
                 ):
        """Builds a SegmentationLevel

        This class should almost never be constructed directly. If an
        instance of SegmentationLevel is needed, the helper function
        `get_segmentation_level()` should be used instead.

        Parameters
        ----------
        name:
            The name of the segmentation to create.

        naming_order:
            The order the segments should be in when creating segment names.

        valid_segments:
            A pandas.DataFrame listing all the valid segments of this
            segmentation. The columns should be named after the segments
            they represent, and should correspond to naming_order
        """
        # TODO: Validate this is a valid segment name
        # Init
        self._name = name
        self._naming_order = naming_order

        # Retain this for copying later
        self._valid_segments = valid_segments

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
        self._segments = pd_utils.reindex_cols(valid_segments, self.naming_order)

        # ## BUILD SEGMENT NAMING ## #
        segments_and_names = self.segments.copy()
        segments_and_names['name'] = pd_utils.str_join_cols(
            df=segments_and_names,
            columns=self.naming_order,
        )
        self._segments_and_names = segments_and_names
        self._segment_names = segments_and_names['name'].to_list()

    @property
    def name(self):
        return self._name

    @property
    def naming_order(self):
        return self._naming_order

    @property
    def segments(self):
        return self._segments

    @property
    def segment_names(self):
        return self._segment_names

    @property
    def segments_and_names(self):
        return self._segments_and_names

    def __copy__(self):
        """Returns a copy of this class"""
        return self.copy()

    def __eq__(self, other) -> bool:
        """Overrides the default implementation"""
        if not isinstance(other, SegmentationLevel):
            return False

        # Make sure names, naming order, and segment names are all the same
        if self.name != other.name:
            return False

        if set(self.naming_order) != set(other.naming_order):
            return False

        if set(self.segment_names) != set(other.segment_names):
            return False

        return True

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
        if not return_seg.is_correct_naming(list(multiply_dict.keys())):
            raise SegmentationError(
                "Some segment names seem to have gone missing during "
                "multiplication.\n"
                "Expected %s segments.\n"
                "Found %s segments."
                % (len(return_seg.segment_names), len(set(multiply_dict.keys())))
            )

        return multiply_dict, return_seg

    def __iter__(self):
        """Overrides the default implementation"""
        return self._segments.to_dict(orient='records').__iter__()

    def _read_multiply_definitions(self) -> pd.DataFrame:
        """
        Returns the multiplication definitions for segments as a pd.DataFrame
        """
        return pd.read_csv(self._multiply_definitions_path)

    def _read_expand_definitions(self) -> pd.DataFrame:
        """
        Returns the expansion definitions for segments as a pd.DataFrame
        """
        return pd.read_csv(self._expand_definitions_path)

    def _read_subset_definitions(self) -> pd.DataFrame:
        """
        Returns the expansion definitions for segments as a pd.DataFrame
        """
        return pd.read_csv(self._subset_definitions_path)

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

    def _get_expansion_definition(self,
                                  other: SegmentationLevel,
                                  ) -> Tuple[str, List[str]]:
        """
        Returns the return_seg_name for expanding self with other
        """
        # Init
        expand_def = self._read_expand_definitions()

        # Try find a definition
        df_filter = {
            'a': self.name,
            'b': other.name,
        }
        definition = pd_utils.filter_df(expand_def, df_filter)

        # If none found, try flipping a and b
        if definition.empty:
            df_filter = {
                'b': self.name,
                'a': other.name,
            }
            definition = pd_utils.filter_df(expand_def, df_filter)

            # If empty again, we don't know what to do
            if definition.empty:
                raise SegmentationError(
                    "Got no definition for expanding '%s' with '%s'.\n"
                    "If there should be a definition, please add one in "
                    "at: %s"
                    % (self.name, other.name, self._expand_definitions_path)
                )

        return definition['out'].squeeze()

    def _get_subset_definition(self,
                               other: SegmentationLevel,
                               ) -> Dict[str, Any]:
        """
        Returns the drop cols of subset-ing this segmentation into other.
        """
        # Init
        subset_def = self._read_subset_definitions()

        # Try find a definition
        df_filter = {
            'in': self.name,
            'out': other.name,
        }
        definition = pd_utils.filter_df(subset_def, df_filter)

        # If empty, we don't know what to do
        if definition.empty:
            raise SegmentationError(
                "Got no definition for aggregating '%s' into '%s'.\n"
                "If there should be a definition, please add one in "
                "at: %s"
                % (self.name, other.name, self._subset_definitions_path)
            )

        return self._parse_drop_cols(definition['drop'].squeeze())

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

    def _parse_drop_cols(self, drop_cols: str) -> Dict[str, Any]:
        """Parses the drop col (from subset.csv) into a Dictionary"""
        # If not string, throw error
        if not isinstance(drop_cols, str):
            raise SegmentationError(
                "No drop cols found. Not sure how to proceed. Got %s"
                % drop_cols
            )

        # Parse into dictionary
        drop_vals = [x.strip() for x in drop_cols.split(self._list_separator)]
        drop_vals = [x.split(self._drop_splitter) for x in drop_vals]

        drop_dict = collections.defaultdict(list)
        for seg, val in drop_vals:
            drop_dict[seg].append(val)

        return drop_dict

    def rename_segment_cols(self,
                            df: pd.DataFrame,
                            naming_conversion: Dict[str, str],
                            inplace: bool = False,
                            ) -> pd.DataFrame:
        """
        Renames the columns of df to the correct segment names.

        Similar to doing a rename on df, however there is added error
        checking in this function to make sure the columns we're renaming
        actually exist, and all the names we're renaming too are actually
        valid segments for this segmentation.

        Parameters
        ----------
        df:
            The dataframe to convert

        naming_conversion:
            A dictionary mapping segment names in self.naming order into
            df columns names. e.g.
            {segment_name: column_name}

        inplace:
            Whether to return a new DataFrame.

        Returns
        -------
        DataFrame or None:
            DataFrame with the renamed columns or None if inplace=True.

        Raises
        ------
        ValueError:
            If any of keys of naming_conversion are not a valid segment_name.

        ValueError:
            If any of values of naming_conversion are not a valid column
            name in df.
        """
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
        rename_dict = {v: k for k, v in naming_conversion.items()}
        return df.rename(columns=rename_dict, inplace=inplace)

    def copy(self) -> SegmentationLevel:
        """Returns a copy of this class"""
        return SegmentationLevel(
            name=self.name,
            naming_order=self.naming_order.copy(),
            valid_segments=self._valid_segments.copy()
        )

    def has_time_period_segments(self) -> bool:
        """Checks whether this segmentation has time period segmentation

        Returns
        -------
        has_time_period_segments:
            True if there is a time_period segment in this segmentation,
            False otherwise
        """
        return self._time_period_segment_name in self.naming_order

    def create_segment_col(self,
                           df: pd.DataFrame,
                           naming_conversion: Dict[str, str] = None,
                           chunk_size: int = int(3e6),
                           process_count: int = consts.PROCESS_COUNT,
                           check_all_segments: bool = False,
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

        chunk_size:
            How big each chunk should be when joining the columns.
            This is used to prevent hitting memory limits when joining the cols,
            which can happen for large dataframes.

        process_count:
            The number of processes to use when splitting df into chunk_size
            chunks. Negative numbers are that name less than all cores, and
            0 can be used to determine that no multiprocessing should be used.

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

        # Rename columns as needed
        df = self.rename_segment_cols(df, naming_conversion, inplace=False)

        # Create the segment col in chunks to avoid memory issues
        total = math.ceil(len(df) / chunk_size)
        pbar_kwargs = {
            'desc': 'Creating Segment Cols',
            'total': total,
            'disable': True,
        }

        kwarg_list = list()
        for df_chunk in pd_utils.chunk_df(df, chunk_size):
            kwarg_list.append({
                'df': df_chunk,
                'columns': self.naming_order.copy(),
            })

        ph = multiprocessing.multiprocess(
            fn=pd_utils.str_join_cols,
            kwargs=kwarg_list,
            process_count=process_count,
            pbar_kwargs=pbar_kwargs,
        )

        # Generate the naming column
        segment_col = pd.concat(ph, ignore_index=False)

        if check_all_segments:
            if not self.is_correct_naming(segment_col.to_list()):
                raise SegmentationError(
                    "Some segment names seem to have gone missing during "
                    "while generating the segment col.\n"
                    "Expected %s segments.\n"
                    "Found %s segments."
                    % (len(self.segment_names), len(segment_col))
                )

        return segment_col

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
            on=join_cols,
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
                "Some segment names seem to have gone missing during "
                "aggregation.\n"
                "Expected %s segments.\n"
                "Found %s segments."
                % (len(other.segment_names), len(set(agg_dict.keys())))
            )

        return agg_dict

    def aggregate_soc_ns_by_p(self,
                              other: SegmentationLevel,
                              ) -> Dict[str, List[str]]:
        """
        Returns an aggregation dictionary, defining how to aggregate soc/ns by
        purpose.

        Soc and Ns are only relevant for certain purposes when distributing
        demand.
        Purposes 1, 2, 12 should all have soc segments, but no ns segments.
        Purposes 3-8, 13-18 should all have ns segments, but no soc segments.
        Where full soc/ns segments exist for all purposes, this function
        aggregates the unneeded segmentation away.

        Parameters
        ----------
        other:
            The SegmentationLevel to aggregate to.

        Returns
        -------
        aggregation_dict:
            A dictionary defining how to aggregate self into other.
            Will be in the form of {out_seg: [in_seg]}.
            Where out seg is a segment name of out_segmentation, and in_seg
            is a list of segment names from self that should be summed to
            generate out_seg.
        """
        # Init
        error_message = (
            "This segmentation does not contain %s. "
            "Need to be using a segmentation using 'p', 'soc', and 'ns' "
            "in order to aggregate soc and ns by tp."
            "Current segmentation uses: %s"
        )

        # Validate input
        if 'p' not in self.naming_order:
            raise nd.SegmentationError(error_message % ('p', self.naming_order))

        if 'soc' not in self.naming_order:
            raise nd.SegmentationError(error_message % ('soc', self.naming_order))

        if 'ns' not in self.naming_order:
            raise nd.SegmentationError(error_message % ('ns', self.naming_order))

        # Add in the names of the original segmentation
        full_segmentation = self.segments.copy()
        full_segmentation['self_name'] = self.create_segment_col(full_segmentation)

        # Aggregate soc and ns depending on p segment
        soc_mask = full_segmentation['p'].isin(consts.SOC_P)
        soc_df = full_segmentation[soc_mask]
        ns_df = full_segmentation[~soc_mask]

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

    def split_tfntt_segmentation(self,
                                 other: SegmentationLevel
                                 ) -> Dict[str, List[str]]:
        """
        Generates a dict defining how to aggregate this segmentation into other.

        Splits the tfn_tt segment into it's components and aggregates up to
        out_segmentation. The DVector needs to be using a segmentation that
        contains tfn_tt in order for this to work.

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

        # Expand tfn_tt
        tfn_tt_expansion = self._get_tfn_tt_expansion()
        seg_agg = pd.merge(
            self.segments,
            tfn_tt_expansion,
            how='left',
            on='tfn_tt',
        )

        # Create the new segment names
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

    def split(self, other: SegmentationLevel) -> Dict[str, List[str]]:
        """
        Generates a dict defining how to split this segmentation into other.

        Splits the tfn_tt segment into it's components and aggregates up to
        out_segmentation. The DVector needs to be using a segmentation that
        contains tfn_tt and p in order for this to work.

        Parameters
        ----------
        other:
            The segmentation level to split into.

        Returns
        -------
        split_dict:
            A dictionary defining how to split into other segmentation.
            Keys will be names of this segmentation, and values will be a
            list of the segments in other that it breaks into.

        Raises
        ------
        ValueError:
            If the given parameters are not the correct types

        SegmentationError:
            If the segmentation cannot be split. This DVector must be
            in a segmentation that is a subset of other.segmentation.
        """
        # Validate inputs
        if not isinstance(other, SegmentationLevel):
            raise ValueError(
                "other is not the correct type. "
                "Expected SegmentationLevel, got %s"
                % type(other)
            )

        # ## MAKE SURE SEGMENTATION IS SUBSET ## #
        # Format self segmentation for comparison
        self_cols = self.naming_order
        self_segs = self.segments
        self_segs = self_segs.sort_values(self_cols)

        # Format self segmentation for comparison
        other_segs = other.segments
        other_segs = other_segs.reindex(columns=self_cols).drop_duplicates()
        other_segs = other_segs.sort_values(self_cols)

        # Check if self is subset of other
        if not np.all(self_segs.values == other_segs.values):
            raise nd.SegmentationError(
                "Cannot split this Segmentation. "
                "%s is not a subset segmentation of %s"
                % (self.name, other.name)
            )

        # ## GENERATE THE SPLITTING DICT ## #
        # Generate the segment names
        split_df = other.segments.copy()
        split_df['self_name'] = self.create_segment_col(split_df)
        split_df['other_name'] = other.create_segment_col(split_df)

        # Convert into the splitting dict
        split_dict = collections.defaultdict(list)
        for s, o in zip(split_df['self_name'], split_df['other_name']):
            split_dict[s].append(o)

        # Check that the output segmentation has been created properly
        other_segments = itertools.chain.from_iterable(split_dict.values())
        if not other.is_correct_naming(other_segments):
            raise SegmentationError(
                "Some segment names seem to have gone missing during"
                "aggregation.\n"
                "Expected %s segments.\n"
                "Found %s segments."
                % (len(other.segment_names), len(set(other_segments)))
            )

        return split_dict

    def expand(self,
               other: SegmentationLevel,
               ) -> Tuple[nd.SegmentMultiplyDict, SegmentationLevel]:
        """Generates a dict defining how to expand this segmentation with other

        Every combination of this segmentation is combined with every
        combination of other segmentation to create a new segmentation.

        Parameters
        ----------
        other:
            The segmentation to expand this segmentation with.

        Returns
        -------
        expand_dict:
            A dictionary defining how to combine self and other in order
            to create a new segmentation.

        Raises
        ------
        ValueError:
            If the given parameters are not the correct types

        SegmentationError:
            If some of the segments overlap in self and other

        SegmentationError:
            If not all of the output segments can be found in the generated
            expansion dictionary.
        """
        # Validate inputs
        if not isinstance(other, SegmentationLevel):
            raise ValueError(
                "other is not the correct type. "
                "Expected SegmentationLevel, got %s"
                % type(other)
            )

        # Check there are no overlapping segments
        overlapping_segs = False
        self_segs = set(self.naming_order)
        other_segs = set(other.naming_order)

        if len(self_segs - other_segs) != len(self_segs):
            overlapping_segs = True
        if len(other_segs - self_segs) != len(other_segs):
            overlapping_segs = True

        if overlapping_segs:
            raise SegmentationError(
                "There are some overlapping segments between self and other. "
                "Cannot expand self with other.\n"
                "self segments: %s\n"
                "other segments: %s\n"
                % (self_segs, other_segs)
            )

        # ## DEFINE THE EXPANSION ## #
        # Get the multiplication definition
        return_seg_name = self._get_expansion_definition(other)

        # Build the return segmentation
        if return_seg_name == self.name:
            return_seg = self
        elif return_seg_name == other.name:
            return_seg = other
        else:
            return_seg = get_segmentation_level(return_seg_name)

        # ## EXPAND THE SEGMENTATIONS ## #
        self_vals = self.segments.values
        other_vals = other.segments.values

        # Repeat values to get all combos
        self_repeat = other_vals.shape[0]
        other_repeat = self_vals.shape[0]

        self_vals = np.tile(self_vals.T, self_repeat).T
        other_vals = np.repeat(other_vals, other_repeat, axis=0)

        # Stick it all together
        new_seg = pd.DataFrame(
            data=np.hstack([self_vals, other_vals]),
            columns=list(self.segments.columns) + list(other.segments.columns)
        )

        # ## GENERATE EXPANSION DICT ## #
        new_seg['self_name'] = self.create_segment_col(new_seg)
        new_seg['other_name'] = other.create_segment_col(new_seg)
        new_seg['return_name'] = return_seg.create_segment_col(new_seg)

        # Convert to expand dict
        r = new_seg['return_name']
        s = new_seg['self_name']
        o = new_seg['other_name']
        expand_dict = dict(zip(r, zip(s, o)))

        # Check that the output segmentation has been created properly
        if not return_seg.is_correct_naming(list(expand_dict.keys())):
            raise SegmentationError(
                "Some segment names seem to have gone missing during "
                "expansion.\n"
                "Expected %s segments.\n"
                "Found %s segments."
                % (len(return_seg.segment_names), len(set(expand_dict.keys())))
            )

        return expand_dict, return_seg

    def subset(self, other: SegmentationLevel) -> List[str]:
        """Generates a list defining which segments to keep when subset-ing to other

        Parameters
        ----------
        other:
            The segmentation to subset to

        Returns
        -------
        subset_list:
            A list defining which segments to keep when subset-ing from this
            segmentation to other.

        Raises
        ------
        ValueError:
            If the given parameters are not the correct types

        SegmentationError:
            If not all of the output segments can be found in the generated
            expansion dictionary.
        """
        # Validate inputs
        if not isinstance(other, SegmentationLevel):
            raise ValueError(
                "other is not the correct type. "
                "Expected SegmentationLevel, got %s"
                % type(other)
            )

        # Find out which segments we should be dropping
        subset_def = self._get_subset_definition(other)

        # ## FIGURE OUT THE SUBSET ## #
        # Need to cast to match subset_def
        segments_and_names = self.segments_and_names.copy()
        for seg, vals in subset_def.items():
            segments_and_names[seg] = segments_and_names[seg].astype(type(vals[0]))

        # Get the rows that match the filter
        drop_mask = segments_and_names.isin(subset_def).any(axis='columns')

        # Get a list of the segments we should keep
        keep_segments = self.segments_and_names['name'][~drop_mask].to_list()

        # Check that the output segmentation has been created properly
        if not other.is_correct_naming(keep_segments):
            raise SegmentationError(
                "Some segment names seem to have gone missing during "
                "expansion.\n"
                "Expected %s segments.\n"
                "Found %s segments."
                % (len(other.segment_names), len(set(keep_segments)))
            )

        return keep_segments

    def is_correct_naming(self, lst: List[str]) -> bool:
        """
        Checks whether lst is a complete list of all names of this segmentation.

        Parameters
        ----------
        lst:
            A list of segmentation names to check

        Returns
        -------
        is_correct_naming:
            True if all names in lst are valid, and lst contains all segmentation
            names. Otherwise False.
        """
        # Init
        names = set(lst)

        # Return False if not all names are valid
        if len(names - set(self.segment_names)) > 0:
            return False

        # If the lists are the same length, we can imply all names are contained
        if not len(names) == len(self.segment_names):
            return False

        # If we are here, then must be True
        return True

    def contains_all_segments(self, lst: List[str]) -> bool:
        """Checks whether lst is a complete list of all segments in this segmentation.

        Parameters
        ----------
        lst:
            A list of segmentation names to check.

        Returns
        -------
        contains_all_segments:
            True if all names in lst are valid, and lst contains all segment
            names. Otherwise False.
        """
        # Init
        names = set(lst)

        # Return False if not all names are valid
        if len(names - set(self.naming_order)) > 0:
            return False

        # Return False if some names are missing
        if len(set(self.naming_order) - names) > 0:
            return False

        # If we are here, then must be True
        return True

    def validate_contains_all_segments(self, lst: List[str]) -> None:
        """Raises an error is lst is not valid

        Raises an error if lst is not a complete list of all segments in
        this segmentation.

        Parameters
        ----------
        lst:
            A list of segmentation names to check.

        Returns
        -------
        None

        Raises
        ------
        ValueError:
            If self.contains_all_segments(lst) returns False.
        """
        if not self.contains_all_segments(lst):
            additional = set(lst) - set(self.naming_order)
            missing = set(self.naming_order) - set(lst)

            raise ValueError(
                "Not all segments for this segmentation are contained in "
                "segment_params.\n"
                "\tAdditional segments: %s\n"
                "\tMissing segments: %s"
                % (additional, missing)
            )

    def get_grouped_weekday_segments(self) -> List[List[str]]:
        """
        Get a nested list of segments, grouped by weekday time periods

        Returns
        -------
        nested_list:
            A nested list, where each nested list is a group of
            segments that only differ on the time period segment. All
            time periods will be weekday only.

        Raises
        ------
        ValueError:
            If this segmentation does not have a time period segment
        """
        # Init
        tp_segment = self._time_period_segment_name

        # Validate arguments
        if tp_segment not in self.naming_order:
            raise ValueError(
                "SegmentationLevel does not have a time period segment (tp). "
                "Cannot get segments by weekday without a time period"
                "segment."
            )

        # Init
        no_tp_naming = self.naming_order.copy()
        no_tp_naming.remove(tp_segment)

        # Filter down to just the weekday time periods
        segments = self.segments_and_names.copy()
        segments = segments[segments[tp_segment].isin(self._weekday_time_periods)]

        # Generate no tp segment name
        segments['no_tp_name'] = pd_utils.str_join_cols(segments, no_tp_naming)

        # Group names with a dict
        temp_dict = collections.defaultdict(list)
        for tp_name, no_tp_name in zip(segments['name'], segments['no_tp_name']):
            temp_dict[no_tp_name].append(tp_name)

        # Grab just the values for returning
        return list(temp_dict.values())

    def get_grouped_weekend_segments(self) -> List[List[str]]:
        """
        Get a nested list of segments, grouped by weekend time periods

        Returns
        -------
        nested_list:
            A nested list, where each nested list is a group of
            segments that only differ on the time period segment. All
            time periods will be weekend only.

        Raises
        ------
        ValueError:
            If this segmentation does not have a time period segment
        """
        # Init
        tp_segment = self._time_period_segment_name

        # Validate arguments
        if tp_segment not in self.naming_order:
            raise ValueError(
                "SegmentationLevel does not have a time period segment (tp). "
                "Cannot get segments by weekday without a time period"
                "segment."
            )

        no_tp_naming = self.naming_order.copy()
        no_tp_naming.remove(tp_segment)

        # Filter down to just the weekday time periods
        segments = self.segments_and_names.copy()
        segments = segments[segments[tp_segment].isin(self._weekend_time_periods)]

        # Generate no tp segment name
        segments['no_tp_name'] = pd_utils.str_join_cols(segments, no_tp_naming)

        # Group names with a dict
        temp_dict = collections.defaultdict(list)
        for tp_name, no_tp_name in zip(segments['name'], segments['no_tp_name']):
            temp_dict[no_tp_name].append(tp_name)

        # Grab just the values for returning
        return list(temp_dict.values())

    def get_time_period_groups(self):
        """
        Get a dictionary of {time_period: segments} in this segmentation.

        Returns
        -------
        time_period_dict:
            A dictionary of {time_period: segments}, where segments is a list
            of all the segments in that time_period.

        Raises
        ------
        ValueError:
            If this segmentation does not have a time period segment
        """
        # Init
        tp_segment_name = self._time_period_segment_name

        # Validate arguments
        if tp_segment_name not in self.naming_order:
            raise ValueError(
                "SegmentationLevel does not have a time period segment (tp). "
                "Cannot get segments by weekday without a time period"
                "segment."
            )

        # Find out which time_periods are in this segmentation
        unique_tp = self.segments_and_names[tp_segment_name].unique()

        # Generate the dictionary
        tp_dict = dict.fromkeys(unique_tp)
        segments = self.segments_and_names.copy()
        for tp in unique_tp:
            tp_segments = segments[segments[tp_segment_name].isin([tp])]
            tp_dict[tp] = tp_segments['name'].to_list()

        return tp_dict

    def generate_file_name(self,
                           segment_params: Dict[str, Any],
                           file_desc: Optional[str] = None,
                           trip_origin: Optional[str] = None,
                           year: Optional[str] = None,
                           suffix: Optional[str] = None,
                           csv: Optional[bool] = False,
                           compressed: Optional[bool] = False,
                           ) -> str:
        """Generate a file name from segment_params

        Builds a underscore separated file name based on the segments
        passed in via segment params. Filename is built in the following
        order, missing any arguments that haven't been defined.
        trip_origin, file_desc, year, segment_params (in naming_order order),
        suffix.

        Parameters
        ----------
        file_desc:
            A string describing the file. For matrices, this is usually 'pa'
            or 'od'. For other files it is a description of their contents.

        segment_params:
            A dictionary of {segment_name: segment_value}. All segment_names
            from this segmentation must be contained in segment_params. An
            error will be thrown if any are missing.

        trip_origin:
            The trip origin to add to the filename. Usually 'hb' or 'nhb'.

        year:
            The year to add to the filename.

        suffix:
            An optional suffix to add to the end of the filename. Could be
            something like 'internal' or 'external'. Optionally can be used
            to add a custom filetype suffix. The dot would need to be passed
            in too.

        csv:
            Whether the return should be a csv filetype or not.

        compressed:
            Whether the return should be a compressed filetype or not.

        Returns
        -------
        file_name:
            The generated file_name for this segmentation.
        """
        # Make sure all segments are in segment_params
        self.validate_contains_all_segments(segment_params.keys())

        # Build the filename in order, and store in list
        name_parts = list()
        if trip_origin is not None:
            name_parts += [trip_origin]

        if file_desc is not None:
            name_parts += [file_desc]

        if year is not None:
            name_parts += ["yr%s" % year]

        for segment_name in self.naming_order:
            name_parts += ["%s%s" % (segment_name, segment_params[segment_name])]

        if suffix is not None:
            name_parts += [suffix]

        # Create name string
        final_name = '_'.join(name_parts)

        # Optionally add on a file_type
        if csv:
            final_name += '.csv'
        elif compressed:
            final_name += consts.COMPRESSION_SUFFIX

        return final_name


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
    import_home = os.path.join(SegmentationLevel._segment_definitions_path, name)

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
    """
    Creates a SegmentationLevel for segmentation with name.

    Parameters
    ----------
    name:
        The name of the segmentation to get a SegmentationLevel for.

    Returns
    -------
    segmentation_level:
        A SegmentationLevel object for segmentation with name
    """
    # TODO(BT): Add some validation on the segmentation name
    # TODO(BT): Add some caching to this function!
    valid_segments, naming_order = _get_valid_segments(name)

    # Create the SegmentationLevel object and return
    return SegmentationLevel(
        name=name,
        naming_order=naming_order,
        valid_segments=valid_segments,
    )
