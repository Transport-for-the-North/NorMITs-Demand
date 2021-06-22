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
import pathlib
import itertools

from typing import Any
from typing import Dict
from typing import Union

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
    _chunk_size = 100000

    _compress_suffix = '.dvec_%s' % consts.COMPRESSION_SUFFIX.split('.')[-1]

    def __init__(self,
                 zoning_system: core.ZoningSystem,
                 segmentation: core.SegmentationLevel,
                 import_data: Union[pd.DataFrame, nd.DVectorData],
                 zone_col: str = None,
                 val_col: str = None,
                 df_naming_conversion: str = None,
                 infill: Any = 0,
                 chunk_size: int = None,
                 process_count: int = consts.PROCESS_COUNT,
                 verbose: bool = False,
                 ) -> DVector:
        # Init
        self.zoning_system = zoning_system
        self.segmentation = segmentation
        self.chunk_size = chunk_size
        self.verbose = verbose
        self.chunk_size = self._chunk_size if chunk_size is None else chunk_size

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
            self.data = self._dict_to_dvec(import_data)
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

        Retains process_count, chunk_size, and verbose params from a.

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
        dvec_data = dict()
        for final_seg, (self_key, other_key) in multiply_dict.items():
            dvec_data[final_seg] = self.data[self_key] * other.data[other_key]

        return DVector(
            zoning_system=return_zoning_system,
            segmentation=return_segmentation,
            import_data=dvec_data,
            process_count=self.process_count,
            verbose=self.verbose,
        )

    def _dict_to_dvec(self, import_data) -> nd.DVectorData:
        # TODO(BT): Add some error checking to make sure this is
        #  actually a valid dict
        return import_data

    def _dataframe_to_dvec_internal(self,
                                    df_chunk,
                                    ) -> nd.DVectorData:
        """
        The internal function of _dataframe_to_dvec - for multiprocessing
        """
        if self.zoning_system is None:
            # Can use 1 to 1 connection to speed this up
            segments = df_chunk[self._segment_col].tolist()
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

                # Filter down to just data as values, and zoning system as the index
                seg_data = seg_data.reindex(columns=[self._zone_col, self._val_col])
                seg_data = seg_data.set_index(self._zone_col)

                # Infill any missing zones as 0
                seg_data = seg_data.reindex(self.zoning_system.unique_zones, fill_value=0)
                dvec_chunk[segment] = seg_data.values

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
        """
        # ## VALIDATE AND CONVERT THE GIVEN DATAFRAME ## #
        # Rename import_data columns to internal names
        rename_dict = {zone_col: self._zone_col, val_col: self._val_col}
        df = df.rename(columns=rename_dict)

        # Add the segment column
        df[self._segment_col] = self.segmentation.create_segment_col(
            df=df,
            naming_conversion=segment_naming_conversion
        )

        # Remove anything else that isn't needed
        if self.zoning_system is None:
            needed_cols = [self._segment_col, self._val_col]
        else:
            needed_cols = [self._segment_col, self._zone_col, self._val_col]
        df = pd_utils.reindex_and_groupby(
            df=df,
            index_cols=needed_cols,
            value_cols=[self._val_col],
        )

        # ## MULTIPROCESSING SETUP ## #
        # If the dataframe is smaller than the chunk size, evenly split across cores
        if len(df) < self.chunk_size * self.process_count:
            chunk_size = math.ceil(len(df) / self.process_count)
        else:
            chunk_size = self.chunk_size

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
            data[name] = default_val.copy()

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
            segment_name = None
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

    def to_df(self) -> pd.DataFrame:
        """
        Convert this DVector into a pandas dataframe with the segmentation
        as the index
        """
        # Init
        concat_ph = list()
        col_names = list(self.segmentation.get_seg_dict(list(self.data.keys())[0]).keys())
        col_names = [self._zone_col] + col_names + [self._val_col]

        # TODO(BT): Multiprocess
        # Convert each segment into a part of the df
        for segment_name, data in self.data.items():
            # Add the zoning system back in
            if self.zoning_system is None:
                df = pd.DataFrame([{self._val_col: data}])
            else:
                index = pd.Index(self.zoning_system.unique_zones, name=self._zone_col)
                data = {self._val_col: data.flatten()}
                df = pd.DataFrame(index=index, data=data).reset_index()

            # Add all segments into the df
            seg_dict = self.segmentation.get_seg_dict(segment_name)
            for col_name, col_val in seg_dict.items():
                df[col_name] = col_val

            # Make sure all dfs are in the same format
            df = df.reindex(columns=col_names)
            concat_ph.append(df)

        return pd.concat(concat_ph).reset_index(drop=True)

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

        if path.suffix != self._compress_suffix:
            path = path.parent / (path.stem + self._compress_suffix)

        compress.write_out(self, path, overwrite_suffix=False)
        return path

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
            # Calculate subsets of keys to avoid copying
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
            'disable': not self.verbose,
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


def read_compressed_dvector(path: nd.PathLike) -> DVector:
    """

    Parameters
    ----------
    path

    Returns
    -------

    """
    # TODO(BT): VALIDATE PATH
    return compress.read_in(path)



