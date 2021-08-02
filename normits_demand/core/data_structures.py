# -*- coding: utf-8 -*-
"""
Created on: Tues May 25th 15:04:32 2021
Updated on:

Original author: Ben Taylor
Last update made by:
Other updates made by:

File purpose:
Holds custom normits_demand objects, such as DVector and its functions
"""
# Allow class self type hinting
from __future__ import annotations

# Builtins
import os
import math
import pickle
import pathlib
import itertools
import copy

from typing import Any
from typing import Dict
from typing import List
from typing import Union
from typing import Callable

# Third Party
import numpy as np
import pandas as pd

# Local Imports
import normits_demand as nd
from normits_demand import constants as consts
from normits_demand import core

from normits_demand.utils import general as du
from normits_demand.utils import pandas_utils as pd_utils

from normits_demand.utils import file_ops
from normits_demand.utils import compress

from normits_demand.concurrency import multiprocessing


# ## CLASSES ## #
class DVector:

    _zone_col = 'zone'
    _segment_col = 'segment'
    _val_col = 'val'
    _zero_infill = 1e-10

    _dvec_suffix = '_dvec%s' % consts.COMPRESSION_SUFFIX

    # Default chunk sizes for multiprocessing
    # Chosen through best guesses and tests
    _chunk_size = 100000
    _to_df_min_chunk_size = 400
    _translate_zoning_min_chunk_size = 700

    # Use for getting a bunch of progress bars for mp code
    _debugging_mp_code = False

    def __init__(self,
                 zoning_system: core.ZoningSystem,
                 segmentation: core.SegmentationLevel,
                 import_data: Union[pd.DataFrame, nd.DVectorData],
                 zone_col: str = None,
                 val_col: str = None,
                 df_naming_conversion: str = None,
                 df_chunk_size: int = None,
                 infill: Any = 0,
                 process_count: int = consts.PROCESS_COUNT,
                 verbose: bool = False,
                 ) -> DVector:
        """
        Validates the input arguments and creates a DVector

        Parameters
        ----------
        zoning_system:
            An nd.core.ZoningSystem object defining the zoning system this
            DVector is using.

        segmentation:
            An nd.core.ZoningSystem object defining the zoning system this
            DVector is using.

        import_data:
            The data to become the DVector data. Can take either a
            dictionary in the DVector.data format (usually used internally),
            or a pandas DataFrame.

        zone_col:
            Only used when import_data is a pandas.DataFrame. This is the name
            of the column in import_data containing the zones names.

        val_col:
            Only used when import_data is a pandas.DataFrame. This is the name
            of the column in import_data containing the data values.

        df_naming_conversion:
            Only used when import_data is a pandas.DataFrame.
            A dictionary mapping segment names in segmentation.naming_order
            into df columns names. e.g.
            {segment_name: column_name}

        df_chunk_size:
            Only used when import_data is a pandas.DataFrame.
            The size of each chunk when processing a large pandas.DataFrame
            into a DVector. Often, the size of the DataFrame can be the
            primary cause of slowdown. By processing in chunks, the conversion
            runs much faster. By default, set to DVector._chunk_size. Play
            with different values to get speedup.

        infill:
            If there are any missing segmentation/zone combinations this value
            will be used to infill. By default, set to 0.

        process_count:
            The number of processes to create in the Pool. Typically this
            should not exceed the number of cores available.
            Negative numbers mean that amount less than all cores e.g. -2
            would be os.cpu_count() - 2. If set to zero, multiprocessing
            will not be used.
            Defaults to consts.PROCESS_COUNT.

        verbose:
            How chatty to be while processing things. Mostly used in debugging
            and should be set to False usually.
        """
        # Validate arguments
        if zoning_system is not None:
            if not isinstance(zoning_system, nd.core.ZoningSystem):
                raise ValueError(
                    "Given zoning_system is not a nd.core.ZoningSystem object."
                )

        if not isinstance(segmentation, nd.core.SegmentationLevel):
            raise ValueError(
                "Given segmentation is not a nd.core.SegmentationLevel object."
            )

        # Init
        self.zoning_system = zoning_system
        self.segmentation = segmentation
        self.verbose = verbose
        self.df_chunk_size = self._chunk_size if df_chunk_size is None else df_chunk_size

        # Define multiprocessing arguments
        if process_count < 0:
            process_count = os.cpu_count() + process_count
        self.process_count = process_count

        if process_count == 0:
            self.chunk_divider = 1
        else:
            self.chunk_divider = self.process_count * 3

        # Set defaults if args not set
        zone_col = self._zone_col if zone_col is None else zone_col
        val_col = self._val_col if val_col is None else val_col

        # Try to convert the given data into DVector format
        if isinstance(import_data, pd.DataFrame):
            self.data = self._dataframe_to_dvec(
                df=import_data,
                zone_col=zone_col,
                val_col=val_col,
                segment_naming_conversion=df_naming_conversion,
                infill=infill,
            )
        elif isinstance(import_data, dict):
            self.data = self._dict_to_dvec(
                import_data=import_data,
                infill=infill,
            )
        else:
            raise NotImplementedError(
                "Don't know how to deal with anything other than: "
                "pandas DF, or dict"
            )

    def __mul__(self: DVector, other: DVector) -> DVector:
        """
        Builds a new Dvec by multiplying a and b together.

        How to join the two Dvectors is defined by the segmentation of each
        Dvector.

        Retains process_count, df_chunk_size, and verbose params from a.

        Parameters
        ----------
        self:
            The first DVector to multiply

        other:
            The second DVector to multiply

        Returns
        -------
        c:
            A new DVector which is the product of multiplying a and b.
        """
        # We can only multiply against other DVectors
        if not isinstance(other, DVector):
            raise nd.NormitsDemandError(
                "The __mul__ operator can only be used with."
                "a DVector objects on each side. Got %s and %s."
                % (type(self), type(other))
            )

        # ## CHECK WE CAN MULTIPLY a AND b ## #
        if self.zoning_system == other.zoning_system:
            return_zoning_system = self.zoning_system
        elif self.zoning_system is None:
            return_zoning_system = other.zoning_system
        elif other.zoning_system is None:
            return_zoning_system = self.zoning_system
        else:
            raise nd.ZoningError(
                "Cannot multiply two Dvectors using different zoning systems.\n"
                "zoning system of a: %s\n"
                "zoning system of b: %s\n"
                % (self.zoning_system.name, other.zoning_system.name)
            )

        # ## DO MULTIPLICATION ## #
        # Use the segmentations to figure out what to multiply
        multiply_dict, return_segmentation = self.segmentation * other.segmentation

        # Build the dvec data here with multiplication
        dvec_data = dict.fromkeys(multiply_dict.keys())
        for final_seg, (self_key, other_key) in multiply_dict.items():
            dvec_data[final_seg] = self.data[self_key] * other.data[other_key]

        return DVector(
            zoning_system=return_zoning_system,
            segmentation=return_segmentation,
            import_data=dvec_data,
            process_count=self.process_count,
            verbose=self.verbose,
        )

    def _dict_to_dvec(self,
                      import_data: nd.DVectorData,
                      infill: Any
                      ) -> nd.DVectorData:
        """
        Validates a given DVector.data dictionary.

        This function should only really be used by the __init__ when
        class functions are creating new DVector dictionaries.

        While converting, will:
        - Makes sure that all segments in the dictionary are valid segments
          for this DVector's segmentation
        - Makes sure that the given dictionary contains ONLY valid segments.
          An error is raised if any extra segments are in the dictionary.
        """
        # TODO(BT): Make sure all values are the correct size of the zoning
        #  system
        # Init

        # ## MAKE SURE DATA CONTAINS ALL SEGMENTS ##
        # Figure out what the default value should be
        if self.zoning_system is None:
            default_val = infill
        else:
            default_val = np.array([infill] * self.zoning_system.n_zones)

        # Find the missing segments and infill
        not_in = set(self.segmentation.segment_names) - import_data.keys()
        for name in not_in:
            import_data[name] = default_val.copy()

        # Double check that all segment names are valid
        if not self.segmentation.is_correct_naming(list(import_data.keys())):
            raise core.SegmentationError(
                "There are additional segment names in the given DVector data "
                "dictionary. Additional names: %s"
                % (set(import_data.keys()) - self.segmentation.segment_names)
            )

        # Make sure all values are the correct shape?
        # Probably want to multiprocess this for large values!
        # Is it worth adding an option to skip some of these steps if
        #  the dictionary is given from a internal DVector build?
        #  We can trust these conditions are met in this environment.
        return import_data

    def _dataframe_to_dvec_internal(self,
                                    df_chunk,
                                    ) -> nd.DVectorData:
        """
        The internal function of _dataframe_to_dvec - for multiprocessing
        """
        if self.zoning_system is None:
            # Make sure only one value exists for each segment
            segments = df_chunk[self._segment_col].tolist()
            segment_set = set(segments)
            if len(segment_set) != len(segments):
                raise ValueError(
                    "The given DataFrame has one or more repeated values "
                    "for some of the segments. Found %s segments, but only "
                    "%s of them are unique."
                    % (len(segments), len(segment_set))
                )

            # Can use 1 to 1 connection to speed this up
            vals = df_chunk[self._val_col].to_list()
            dvec_chunk = {s: v for s, v, in zip(segments, vals)}

        else:
            # Generate the data on a per segment basis
            dvec_chunk = dict()

            for segment in df_chunk['segment'].unique():
                # Check that it's a valid segment_name
                if segment not in self.segmentation.segment_names:
                    raise ValueError(
                        "%s is not a valid segment name for a Dvector using %s "
                        "segmentation" % (segment, self.segmentation.name)
                    )

                # Get all available pop for this segment
                seg_data = df_chunk[df_chunk[self._segment_col] == segment].copy()

                # TODO(BT): There's a VERY slight chance that duplicate zones
                #  could be split across processes. Need to add a check for
                #  this on the calling function.
                # Make sure there are no duplicate zones
                seg_zones = seg_data[self._zone_col].tolist()
                seg_zones_set = set(seg_zones)
                if len(seg_zones_set) != len(seg_zones):
                    raise ValueError(
                        "The given DataFrame has one or more repeated values "
                        "for some of the zones in segment %s. Found %s "
                        "segments, but only %s of them are unique."
                        % (segment, len(seg_zones), len(seg_zones_set))
                    )

                # Make sure zones that don't exist in this zoning system are found
                zoning_system_zones = set(self.zoning_system.unique_zones)
                extra_zones = seg_zones_set - zoning_system_zones
                if len(extra_zones) > 0:
                    raise ValueError(
                        "Found zones that don't exist in %s zoning in the "
                        "given DataFrame. For segment %s, the following "
                        "zones do not belong to this zoning system:\n%s"
                        % (self.zoning_system.name, segment, extra_zones)
                    )

                # Filter down to just data as values, and zoning system as the index
                seg_data = seg_data.reindex(columns=[self._zone_col, self._val_col])
                seg_data = seg_data.set_index(self._zone_col)

                # Infill any missing zones as 0
                seg_data = seg_data.reindex(self.zoning_system.unique_zones, fill_value=0)
                dvec_chunk[segment] = seg_data.values.flatten()

        return dvec_chunk

    def _dataframe_to_dvec(self,
                           df: pd.DataFrame,
                           zone_col: str,
                           val_col: str,
                           segment_naming_conversion: str,
                           infill: Any,
                           ) -> nd.DVectorData:
        """
        Converts a pandas dataframe into dvec.data internal structure

        While converting, will:
        - Make sure that any missing segment/zone combinations are infilled
          with infill
        - Make sure only one value exist for each segment/zone combination
        """
        # Init columns depending on if we have zones
        required_cols = self.segmentation.naming_order + [self._val_col]
        sort_cols = [self._segment_col]

        # Add zoning if we need it
        if self.zoning_system is not None:
            required_cols += [self._zone_col]
            sort_cols += [self._zone_col]

        # ## VALIDATE AND CONVERT THE GIVEN DATAFRAME ## #
        # Rename import_data columns to internal names
        rename_dict = {zone_col: self._zone_col, val_col: self._val_col}
        df = df.rename(columns=rename_dict)

        # Rename the segment columns if needed
        if segment_naming_conversion is not None:
            df = self.segmentation.rename_segment_cols(df, segment_naming_conversion)

        # Make sure we don't have any extra columns
        extra_cols = set(list(df)) - set(required_cols)
        if len(extra_cols) > 0:
            raise ValueError(
                "Found extra columns in the given DataFrame than needed. The "
                "given DataFrame should only contain val_col, "
                "segmentation_cols, and the zone_col (where applicable). "
                "Found the following extra columns: %s"
                % extra_cols
            )

        # Add the segment column - drop the individual cols
        df[self._segment_col] = self.segmentation.create_segment_col(
            df=df,
            naming_conversion=segment_naming_conversion
        )
        df = df.drop(columns=self.segmentation.naming_order)

        # Sort by the segment columns for MP speed
        df = df.sort_values(by=sort_cols)

        # ## MULTIPROCESSING SETUP ## #
        # If the dataframe is smaller than the chunk size, evenly split across cores
        if len(df) < self.df_chunk_size * self.process_count:
            chunk_size = math.ceil(len(df) / self.process_count)
        else:
            chunk_size = self.df_chunk_size

        # setup a pbar
        pbar_kwargs = {
            'desc': "Converting df to dvec",
            'unit': "segment",
            'disable': (not self.verbose),
            'total': math.ceil(len(df) / chunk_size)
        }

        # ## MULTIPROCESS THE DATA CONVERSION ## #
        # Build a list of arguments
        kwarg_list = list()
        for df_chunk in pd_utils.chunk_df(df, chunk_size):
            kwarg_list.append({'df_chunk': df_chunk})

        # Call across multiple threads
        data_chunks = multiprocessing.multiprocess(
            fn=self._dataframe_to_dvec_internal,
            kwargs=kwarg_list,
            process_count=self.process_count,
            pbar_kwargs=pbar_kwargs,
        )
        data = du.sum_dict_list(data_chunks)

        # ## MAKE SURE DATA CONTAINS ALL SEGMENTS ##
        # find the segments which arent in there
        not_in = set(self.segmentation.segment_names) - data.keys()

        # Figure out what the default value should be
        if self.zoning_system is None:
            default_val = infill
        else:
            default_val = np.array([infill] * self.zoning_system.n_zones)

        # Infill the missing segments
        for name in not_in:
            data[name] = copy.copy(default_val)

        return data

    def get_segment_data(self,
                         segment_name: str = None,
                         segment_dict: Dict[str, Any] = None,
                         ) -> Union[np.array, int, float]:
        """
        Gets the data for the given segment from the Dvector

        If no data for the given segment exists, then returns a np.array
        of the length of this DVectors zoning system.

        Parameters
        ----------
        segment_name:
            The name of the segment to get. Can only set either this or
            segment_dict. 
        
        segment_dict:
            A dictionary of a segment to get. Should be as
            {segment_name: segment_value} pairs.
            Can only set this or segment_name. If this value is set, it will
            internally be converted into a segment_name.

        Returns
        -------
        segment_data:
            The data for segment_name

        """
        # Make sure only one argument is set
        if not du.xor(segment_name is None, segment_dict is None):
            raise ValueError(
                "Need to set either segment_name or segment_dict in order to "
                "get the data of a segment.\n"
                "Both values cannot be set, neither can both be left as None."
            )

        # Build the segment_name if we don't have it
        if segment_dict is not None:
            raise NotImplemented(
                "Need to write code to convert a segment_dict into a valid "
                "segment_name"
            )

        if segment_name not in self.segmentation.segment_names:
            raise ValueError(
                "%s is not a valid segment name for a Dvector using %s "
                "segmentation." % self.segmentation.name
            )

        # Get data and covert to zoning system
        return self.data[segment_name]

    @staticmethod
    def _to_df_internal(self_data: nd.DVectorData,
                        self_zoning_system: core.ZoningSystem,
                        self_segmentation: core.SegmentationLevel,
                        col_names: List[str],
                        val_col: str,
                        zone_col: str,
                        ) -> pd.DataFrame:
        """
        Internal function of self.to_df(). For multiprocessing
        """
        # Init
        index_cols = du.list_safe_remove(col_names, [val_col])
        concat_ph = list()

        # Convert all given data into dataframes
        for segment_name, data in self_data.items():
            # Add the zoning system back in
            if self_zoning_system is None:
                df = pd.DataFrame([{val_col: data}])
            else:
                index = pd.Index(self_zoning_system.unique_zones, name=zone_col)
                data = {val_col: data.flatten()}
                df = pd.DataFrame(index=index, data=data).reset_index()

            # Add all segments into the df
            seg_dict = self_segmentation.get_seg_dict(segment_name)
            for col_name, col_val in seg_dict.items():
                df[col_name] = col_val

            # Make sure all dfs are in the same format
            df = df.reindex(columns=col_names)
            concat_ph.append(df)

        return pd.concat(concat_ph, ignore_index=True)

    def to_df(self) -> pd.DataFrame:
        """
        Convert this DVector into a pandas dataframe with the segmentation
        as the index
        """
        # Init
        col_names = list(self.segmentation.get_seg_dict(list(self.data.keys())[0]).keys())
        col_names = col_names + [self._val_col]
        if self.zoning_system is not None:
            col_names = [self._zone_col] + col_names

        # ## MULTIPROCESS ## #
        # Define chunk size
        total = len(self.data)
        chunk_size = math.ceil(total / self.chunk_divider)

        # Make sure the chunks aren't too small
        if chunk_size < self._to_df_min_chunk_size:
            chunk_size = self._to_df_min_chunk_size

        # Define the kwargs
        kwarg_list = list()
        for keys_chunk in du.chunk_list(self.data.keys(), chunk_size):
            # Calculate subsets of self.data to avoid locks between processes
            self_data_subset = {k: self.data[k] for k in keys_chunk}

            if self.zoning_system is not None:
                self_zoning_system = self.zoning_system.copy()
            else:
                self_zoning_system = None

            # Assign to a process
            kwarg_list.append({
                'self_data': self_data_subset,
                'self_zoning_system': self_zoning_system,
                'self_segmentation': self.segmentation.copy(),
                'col_names': col_names.copy(),
                'val_col': self._val_col,
                'zone_col': self._zone_col,
            })

        # Define pbar
        pbar_kwargs = {
            'desc': "Converting DVector to dataframe",
            'disable': not self._debugging_mp_code,
        }

        # Run across processes
        dataframe_chunks = multiprocessing.multiprocess(
            fn=self._to_df_internal,
            kwargs=kwarg_list,
            process_count=self.process_count,
            pbar_kwargs=pbar_kwargs,
        )

        # Join all the dataframe chunks together
        return pd.concat(dataframe_chunks, ignore_index=True)

    def compress_out(self, path: nd.PathLike) -> pathlib.Path:
        """
        Writes this DVector to disk at path.

        Parameters
        ----------
        path:
            The path to write this object out to.
            Conventionally should end in .dvec.pbz2.
            If it does not, the suffix will be added, and the new
            path returned.

        Returns
        -------
        path:
            The path that this object was written out to. If path ended in the
            correct suffix, then this will be exactly the same as input path.

        Raises
        ------
        IOError:
            If the path cannot be found.
        """
        # Init
        path = file_ops.cast_to_pathlib_path(path)

        if path.suffix != self._dvec_suffix:
            path = path.parent / (path.stem + self._dvec_suffix)

        return compress.write_out(self, path, overwrite_suffix=False)

    def to_pickle(self, path: nd.PathLike) -> None:
        """
        Pickle (serialize) object to file.

        Parameters
        ----------
        path:
            Filepath to store the pickled object

        """
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def _multiply_and_aggregate_internal(aggregation_keys_chunk,
                                         aggregation_dict,
                                         multiply_dict,
                                         self_data,
                                         other_data,
                                         ):
        """
        Internal function of self.multiply_and_aggregate. For multiprocessing
        """
        # Init
        dvec_data = dict.fromkeys(aggregation_keys_chunk)

        # Multiply and aggregate in chunks
        for out_seg_name in aggregation_keys_chunk:
            # Calculate all the segments to aggregate
            inter_segs = list()
            for segment_name in aggregation_dict[out_seg_name]:
                self_key, other_key = multiply_dict[segment_name]
                result = (self_data[self_key] * other_data[other_key]).flatten()
                inter_segs.append(result)

            # Aggregate!
            dvec_data[out_seg_name] = np.sum(inter_segs, axis=0)

        return dvec_data

    def aggregate(self,
                  out_segmentation: core.SegmentationLevel,
                  split_tfntt_segmentation: bool = False,
                  ) -> DVector:
        """
        Aggregates (by summing) this Dvector into out_segmentation.

        A definition of how to aggregate from self.segmentation to
        out_segmentation must exist, otherwise a SegmentationError will be
        thrown

        Parameters
        ----------
        out_segmentation:
            The segmentation to aggregate into.

        split_tfntt_segmentation:
            If converting from the current segmentation to out_segmentation
            requires the splitting of the tfn_tt segmentation, mark this as
            True - a special type of aggregation is needed underneath.

        Returns
        -------
        aggregated_DVector:
            a new dvector containing the same data, but aggregated to
            out_segmentation.
        """
        # Validate inputs
        if not isinstance(out_segmentation, core.SegmentationLevel):
            raise ValueError(
                "out_segmentation is not the correct type. "
                "Expected SegmentationLevel, got %s"
                % type(out_segmentation)
            )

        # Get the aggregation dict
        if split_tfntt_segmentation:
            aggregation_dict = self.segmentation.split_tfntt_segmentation(out_segmentation)
        else:
            aggregation_dict = self.segmentation.aggregate(out_segmentation)

        # Aggregate!
        # TODO(BT): Add optional multiprocessing if aggregation_dict is big enough
        dvec_data = dict()
        for out_seg_name, in_seg_names in aggregation_dict.items():
            in_lst = [self.data[x].flatten() for x in in_seg_names]
            dvec_data[out_seg_name] = np.sum(in_lst, axis=0)

        return DVector(
            zoning_system=self.zoning_system,
            segmentation=out_segmentation,
            import_data=dvec_data,
            process_count=self.process_count,
            verbose=self.verbose,
        )

    def multiply_and_aggregate(self: DVector,
                               other: DVector,
                               out_segmentation: core.SegmentationLevel,
                               ) -> DVector:
        """
        Multiply with other DVector, and aggregates as it goes.

        Useful when the output segmentation of multiplying self and other
        would be massive. Multiplication is done in chunks, and aggregated in
        out_segmentation periodically.

        Parameters
        ----------
        other:
            The DVector to multiply self with.

        out_segmentation:
            The segmentation to use in the outputs DVector

        Returns
        -------
        DVector:
            The result of (self * other).aggregate(out_segmentation)
        """
        # Validate inputs
        if not isinstance(other, DVector):
            raise ValueError(
                "b is not the correct type. Expected Dvector, got %s"
                % type(other)
            )

        if not isinstance(out_segmentation, core.SegmentationLevel):
            raise ValueError(
                "out_segmentation is not the correct type. "
                "Expected SegmentationLevel, got %s"
                % type(out_segmentation)
            )

        # Get the aggregation and multiplication dictionaries
        multiply_dict, mult_return_seg = self.segmentation * other.segmentation
        aggregation_dict = mult_return_seg.aggregate(out_segmentation)

        # ## MULTIPROCESS ## #
        # Define the chunk size
        total = len(aggregation_dict)
        chunk_size = math.ceil(total / self.chunk_divider)

        # Define the kwargs
        kwarg_list = list()
        for keys_chunk in du.chunk_list(aggregation_dict.keys(), chunk_size):
            # Calculate subsets of keys to avoid locks between processes
            agg_dict_subset = {k: aggregation_dict[k] for k in keys_chunk}
            
            key_subset = itertools.chain.from_iterable(agg_dict_subset.values())
            mult_dict_subset = {k: multiply_dict[k] for k in key_subset}

            self_keys, other_keys = zip(*mult_dict_subset.values())
            self_data = {k: self.data[k] for k in self_keys}
            other_data = {k: other.data[k] for k in other_keys}

            # Assign to a process
            kwarg_list.append({
                'aggregation_keys_chunk': keys_chunk,
                'aggregation_dict': agg_dict_subset,
                'multiply_dict': mult_dict_subset,
                'self_data': self_data,
                'other_data': other_data,
            })

        # Define pbar
        pbar_kwargs = {
            'desc': "Multiplying and aggregating",
            'disable': not self._debugging_mp_code,
        }

        # Run across processes
        data_chunks = multiprocessing.multiprocess(
            fn=self._multiply_and_aggregate_internal,
            kwargs=kwarg_list,
            process_count=self.process_count,
            pbar_kwargs=pbar_kwargs,
        )

        # Combine all computation chunks into one
        dvec_data = dict.fromkeys(aggregation_dict.keys())
        for chunk in data_chunks:
            dvec_data.update(chunk)

        return DVector(
            zoning_system=self.zoning_system,
            segmentation=out_segmentation,
            import_data=dvec_data,
            process_count=self.process_count,
            verbose=self.verbose,
        )

    def sum(self) -> float:
        """
        Sums all values within the Dvector and returns the total

        Returns
        -------
        sum:
            The total sum of all values
        """
        return np.sum([x.flatten() for x in self.data.values()])

    @staticmethod
    def _translate_zoning_internal(self_data: nd.DVectorData,
                                   translation: np.array,
                                   ) -> nd.DVectorData:
        """
        Internal function of self.translate_zoning. For multiprocessing
        """
        # Init
        dvec_data = dict.fromkeys(self_data.keys())

        # Translate zoning in chunks
        for key, value in self_data.items():
            value = value.flatten()
            temp = np.broadcast_to(np.expand_dims(value, axis=1), translation.shape)
            temp = temp * translation
            dvec_data[key] = temp.sum(axis=0)

        return dvec_data

    def translate_zoning(self,
                         new_zoning: core.ZoningSystem,
                         weighting: str = None,
                         ) -> DVector:
        """
        Translates this DVector into another zoning system and returns a new
        DVector.

        Parameters
        ----------
        new_zoning:
            The zoning system to translate into.

        weighting:
            The weighting to use when building the translation. Must be None,
            or one of ZoningSystem.possible_weightings

        Returns
        -------
        translated_dvector:
            This DVector translated into new_new_zoning zoning system

        """
        # Validate inputs
        if not isinstance(new_zoning, core.ZoningSystem):
            raise ValueError(
                "new_zoning is not the correct type. "
                "Expected ZoningSystem, got %s"
                % type(new_zoning)
            )

        if self.zoning_system.name is None:
            raise nd.NormitsDemandError(
                "Cannot translate the zoning system of a DVector that does "
                "not have a zoning system to begin with."
            )

        # Get translation
        translation = self.zoning_system.translate(new_zoning, weighting)

        # ## MULTIPROCESS ## #
        # Define the chunk size
        total = len(self.data)
        chunk_size = math.ceil(total / self.chunk_divider)

        # Make sure the chunks aren't too small
        if chunk_size < self._translate_zoning_min_chunk_size:
            chunk_size = self._translate_zoning_min_chunk_size

        # Define the kwargs
        kwarg_list = list()
        for keys_chunk in du.chunk_list(self.data.keys(), chunk_size):
            # Calculate subsets of self.data to avoid locks between processes
            self_data_subset = {k: self.data[k] for k in keys_chunk}

            # Assign to a process
            kwarg_list.append({
                'self_data': self_data_subset,
                'translation': translation.copy(),
            })

        # Define pbar
        pbar_kwargs = {
            'desc': "Translating",
            'disable': not self._debugging_mp_code,
        }

        # Run across processes
        data_chunks = multiprocessing.multiprocess(
            fn=self._translate_zoning_internal,
            kwargs=kwarg_list,
            process_count=self.process_count,
            pbar_kwargs=pbar_kwargs,
        )

        # Combine all computation chunks into one
        dvec_data = dict.fromkeys(self.data.keys())
        for chunk in data_chunks:
            dvec_data.update(chunk)

        return DVector(
            zoning_system=new_zoning,
            segmentation=self.segmentation,
            import_data=dvec_data,
            process_count=self.process_count,
            verbose=self.verbose,
        )

    def split_tfntt_segmentation(self,
                                 out_segmentation: core.SegmentationLevel
                                 ) -> DVector:
        """
        Converts a DVector with p and tfn_tt segmentation in out_segmentation

        Splits the tfn_tt segment into it's components and aggregates up to
        out_segmentation. The DVector needs to be using a segmentation that
        contains tfn_tt and p in order for this to work.

        Parameters
        ----------
        out_segmentation:
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
        if not isinstance(out_segmentation, core.SegmentationLevel):
            raise ValueError(
                "out_segmentation is not the correct type. "
                "Expected SegmentationLevel, got %s"
                % type(out_segmentation)
            )

        if 'tfn_tt' not in self.segmentation.naming_order:
            raise nd.SegmentationError(
                "This Dvector is not using a segmentation which contains "
                "'tfn_tt'. Need to be using a segmentation using both 'tfn_tt' "
                "and 'p' segments in order to split tfn_tt."
                "Current segmentation uses: %s"
                % self.segmentation.naming_order
            )

        if 'p' not in self.segmentation.naming_order:
            raise nd.SegmentationError(
                "This Dvector is not using a segmentation which contains "
                "'p'. Need to be using a segmentation using both 'tfn_tt' "
                "and 'p' segments in order to split tfn_tt."
                "Current segmentation uses: %s"
                % self.segmentation.naming_order
            )

        # Get the aggregation dict
        aggregation_dict = self.segmentation.split_tfntt_segmentation(out_segmentation)

        # Aggregate!
        # TODO(BT): Add optional multiprocessing if aggregation_dict is big enough
        dvec_data = dict()
        for out_seg_name, in_seg_names in aggregation_dict.items():
            in_lst = [self.data[x].flatten() for x in in_seg_names]
            dvec_data[out_seg_name] = np.sum(in_lst, axis=0)

        return DVector(
            zoning_system=self.zoning_system,
            segmentation=out_segmentation,
            import_data=dvec_data,
            process_count=self.process_count,
            verbose=self.verbose,
        )

    def split_segmentation_like(self,
                                other: DVector,
                                ) -> DVector:
        """
        Splits this DVector segmentation into other.segmentation

        Using other to derive the splitting factors, this DVector is split
        into the same segmentation as other.segmentation. Splits are derived
        as an average across all zones for each segment in other, resulting in
        a single splitting factor for each segment. The splitting factor is
        then applied to this DVector, equally across all zones.

        Parameters
        ----------
        other:
            The DVector to use to determine the segmentation to split in to,
            as well as the weights to use for the splits.

        Returns
        -------
        new_dvector:
            A new DVector containing the same total as values, but split
            into other.segmentation segmentation.

        Raises
        ------
        ValueError:
            If the given parameters are not the correct types

        SegmentationError:
            If the segmentation cannot be split. This DVector must be
            in a segmentation that is a subset of other.segmentation.
        """
        # Validate inputs
        if not isinstance(other, DVector):
            raise ValueError(
                "other is not the correct type. "
                "Expected DVector, got %s"
                % type(other)
            )

        # Get the dictionary defining how to split
        split_dict = self.segmentation.split(other.segmentation)

        # Split!
        # TODO(BT): Add optional multiprocessing if split_dict is big enough
        dvec_data = dict.fromkeys(other.segmentation.segment_names)
        for in_seg_name, out_seg_names in split_dict.items():
            # Calculate the splitting factors
            other_segs = [np.mean(other.data[s]) for s in out_seg_names]
            split_factors = other_segs / np.sum(other_segs)

            # Get the original value
            self_seg = self.data[in_seg_name]

            # Split
            for name, factor in zip(out_seg_names, split_factors):
                dvec_data[name] = self_seg * factor

        return DVector(
            zoning_system=self.zoning_system,
            segmentation=other.segmentation,
            import_data=dvec_data,
            process_count=self.process_count,
            verbose=self.verbose,
        )

    def balance_at_segments(self,
                            other: DVector,
                            split_weekday_weekend: bool = False,
                            ) -> DVector:
        """
        Balance segment totals to other, ignoring zoning splits.

        Essentially does
        self[segment] *= other[segment].sum() / self[segment].sum()
        for all segments.

        Parameters
        ----------
        other:
            The DVector to control this one to. Must have the same segmentation
            as this DVector

        split_weekday_weekend:
            Whether to control the time periods as weekday and weekend splits
            instead of each individual time period. If set to True,
            each DVector must be at a segmentation with a 'tp' segment.

        Returns
        -------
        controlled_dvector:
            A copy of this DVector, controlled to other. The total of each
            segment should be equal across self and other.

        Raises
        ------
        ValueError:
            If the given parameters are not the correct types.

        ValueError:
            If self and other do not have the same segmentation.
        """
        # Init
        infill = self._zero_infill

        # Validate inputs
        if not isinstance(other, DVector):
            raise ValueError(
                "other is not the correct type. "
                "Expected DVector, got %s"
                % type(other)
            )

        if self.segmentation.name != other.segmentation.name:
            raise ValueError(
                "Segmentation of both DVectors does not match! "
                "Perhaps you need to call "
                "self.split_segmentation_like(other) to bring them into "
                "alignment?\n"
                "self segmentation: %s\n"
                "other segmentation: %s"
                % (self.segmentation.name, other.segmentation.name)
            )

        # Loop through each segment and control
        dvec_data = dict.fromkeys(self.segmentation.segment_names)

        if split_weekday_weekend:
            # Get the grouped segment lists
            wk_day_segs = self.segmentation.get_grouped_weekday_segments()
            wk_end_segs = self.segmentation.get_grouped_weekend_segments()

            # Control by weekday and weekend separately
            for split_wk_segs in [wk_day_segs, wk_end_segs]:
                for segment_group in split_wk_segs:
                    # Get data and infill zeros
                    self_data_lst = list()
                    other_data_lst = list()
                    for segment in segment_group:
                        # Get data
                        self_data = self.data[segment]
                        other_data = other.data[segment]

                        # Infill zeros
                        self_data = np.where(self_data <= 0, infill, self_data)
                        other_data = np.where(other_data <= 0, infill, other_data)

                        # Append
                        self_data_lst.append(self_data)
                        other_data_lst.append(other_data)

                    # Get the control factor
                    factor = np.sum(other_data_lst) / np.sum(self_data_lst)

                    # Balance each segment
                    for segment, self_data in zip(segment_group, self_data_lst):
                        dvec_data[segment] = self_data * factor

        else:
            # Control all segments as normal
            for segment in self.segmentation.segment_names:
                # Get data
                self_data = self.data[segment]
                other_data = other.data[segment]

                # Infill zeros
                self_data = np.where(self_data <= 0, infill, self_data)
                other_data = np.where(other_data <= 0, infill, other_data)

                # Balance
                dvec_data[segment] = self_data * (np.sum(other_data) / np.sum(self_data))

        return DVector(
            zoning_system=self.zoning_system,
            segmentation=other.segmentation,
            import_data=dvec_data,
            process_count=self.process_count,
            verbose=self.verbose,
        )

    def sum_zoning(self) -> DVector:
        """
        Sums all the zone values in DVector into a single value.

        Returns a copy of Dvector. This function is equivalent to calling:
        self.remove_zoning(fn=np.sum)

        Returns
        -------
        summed_dvector:
            A copy of DVector, without any zoning.
        """
        return self.remove_zoning(fn=np.sum)

    def remove_zoning(self, fn: Callable) -> DVector:
        """
        Aggregates all the zone values in DVector into a single value using fn.

        Returns a copy of Dvector.

        Parameters
        ----------
        fn:
            The function to use when aggregating all zone values. fn must
            be able to take a np.array of values and return a single value
            in order for this to work.

        Returns
        -------
        summed_dvector:
            A copy of DVector, without any zoning.
        """
        # Validate fn
        if not callable(fn):
            raise ValueError(
                "fn is not callable. fn must be a function that "
                "takes an np.array of values and return a single value."
            )

        # Aggregate all the data
        keys = self.data.keys()
        values = [fn(x) for x in self.data.values()]

        return DVector(
            zoning_system=None,
            segmentation=self.segmentation,
            import_data=dict(zip(keys, values)),
            process_count=self.process_count,
            verbose=self.verbose,
        )


class DVectorError(nd.NormitsDemandError):
    """
    Exception for all errors that occur around DVector management
    """
    def __init__(self, message=None):
        self.message = message
        super().__init__(self.message)


# ## FUNCTIONS ## #
def multiply_and_aggregate_dvectors(a: DVector,
                                    b: DVector,
                                    out_segmentation: core.SegmentationLevel,
                                    ) -> DVector:
    """
    Multiplies a with b, and aggregates as it goes.

    Useful when the output segmentation of multiplying a and b
    would be massive. Multiplication is done in chunks, and aggregated in
    out_segmentation periodically.

    Parameters
    ----------
    a:
        The first DVector to multiply.

    b:
        The second DVector to multiply.

    out_segmentation:
        The segmentation to use in the outputs DVector

    Returns
    -------
    DVector:
        The result of (a * b).aggregate(out_segmentation)
    """
    # Validate arguments
    if not isinstance(a, DVector):
        raise ValueError(
            "a is not the correct type. Expected Dvector, got %s"
            % type(a)
        )

    if not isinstance(b, DVector):
        raise ValueError(
            "b is not the correct type. Expected Dvector, got %s"
            % type(b)
        )

    return a.multiply_and_aggregate(other=b, out_segmentation=out_segmentation)


def read_compressed_dvector(path: nd.PathLike) -> Union[DVector, Any]:
    """
    Load pickled and compressed DVector object (or any object) from file.

    Parameters
    ----------
    path:
        The full path to the object to read.

    Returns
    -------
    object:
        The object that was read in from disk. Will be the
        same type as object stored in file.
    """
    # File validation
    file_ops.check_file_exists(path)

    return compress.read_in(path)


def from_pickle(path: nd.PathLike) -> Union[DVector, Any]:
    """
    Load pickled DVector object (or any object) from file.

    Parameters
    ----------
    path:
        Filepath to the object to read in and unpickle

    Returns
    -------
    unpickled:
        Same type as object stored in file.
    """
    # TODO(BT): VALIDATE PATH
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj



