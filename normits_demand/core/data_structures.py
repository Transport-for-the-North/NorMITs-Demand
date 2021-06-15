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
# Builtins
import os
import math

from typing import List
from typing import Dict

# Third Party
import numpy as np
import pandas as pd

import tqdm

# Local Imports
import normits_demand as nd
from normits_demand import constants as consts
from normits_demand import core

from normits_demand.utils import general as du
from normits_demand.utils import pandas_utils as pd_utils

from normits_demand.concurrency import multiprocessing


# ## CLASSES ## #
class DVector:

    _zone_col = 'zone_col'
    _segment_col = 'segment'
    _val_col = 'val'
    _chunk_size = 100000

    def __init__(self,
                 zoning_system: core.ZoningSystem,
                 segmentation: core.SegmentationLevel,
                 import_data: pd.DataFrame,
                 zone_col: str = None,
                 segment_col: str = None,
                 val_col: str = None,
                 chunk_size: int = None,
                 process_count: int = consts.PROCESS_COUNT,
                 verbose: bool = False,
                 ):
        # Init
        self.zoning_system = zoning_system
        self.segmentation = segmentation

        if process_count < 0:
            process_count = os.cpu_count() + process_count
        self.process_count = process_count

        # Set defaults if args not set
        zone_col = self._zone_col if zone_col is None else zone_col
        segment_col = self._segment_col if segment_col is None else segment_col
        val_col = self._val_col if val_col is None else val_col
        chunk_size = self._chunk_size if chunk_size is None else chunk_size

        # Try to convert the given data into DVector format
        if isinstance(import_data, pd.DataFrame):
            self.data = self._dataframe_to_dvec(
                import_data,
                zone_col,
                segment_col,
                val_col,
                chunk_size,
                verbose,
            )
        else:
            raise NotImplementedError(
                "Don't know how to deal with anything other than a pandas DF"
            )

    def _dataframe_to_dvec_internal(self,
                                    df_chunk,
                                    zone_col,
                                    segment_col,
                                    val_col,
                                    ):
        """
        The internal function of _dataframe_to_dvec - for multiprocessing
        """
        # ## VALIDATE AND CONVERT THE GIVEN DATAFRAME ## #
        if self.zoning_system is None:
            needed_cols = [segment_col, val_col]
        else:
            needed_cols = [zone_col, segment_col, val_col]
        df_chunk = pd_utils.reindex_cols(df_chunk, needed_cols)

        # Rename import_data columns to internal names
        rename_dict = {
            zone_col: self._zone_col,
            segment_col: self._segment_col,
            val_col: self._val_col
        }
        df_chunk = df_chunk.rename(columns=rename_dict)

        # Generate the data on a per segment basis
        dvec_chunk = dict()
        for segment in df_chunk['segment'].unique():
            # Get all available pop for this segment
            seg_data = df_chunk[df_chunk[self._segment_col] == segment].copy()

            # Normalise the values depending on the zoning system
            if self.zoning_system is None:
                # No zoning system, should only have a single value
                if len(seg_data) > 1:
                    raise nd.NormitsDemandError(
                        "While instantiating a DVec object without a "
                        "zoning system, found more than one value for segment "
                        "'%s'. Should only be one value per segment if "
                        "zoning_system is set to None"
                        % segment
                    )

                dvec_chunk[segment] = seg_data[self._val_col].squeeze()

            else:
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
                           segment_col: str,
                           val_col: str,
                           chunk_size: int,
                           verbose: bool = False,
                           ) -> Dict[str, np.ndarray]:
        """
        Converts a pandas dataframe into dvec.data internal structure
        """
        # Init

        # TODO(BT): Once the segmentation object is properly implemented
        #  some validation needs adding to make sure every value in the
        #  segment column is a valid segment.

        # setup a pbar
        pbar_kwargs = {
            'desc': "Converting df to dvec",
            'unit': "segment",
            'disable': (not verbose),
            'total': round(len(df) / chunk_size)
        }

        # If the dataframe is smaller than the chunk size, evenly split across cores
        if len(df) < chunk_size * self.process_count:
            chunk_size = math.ceil(len(df) / self.process_count)

        # ## MULTIPROCESS THE DATA CONVERSION ## #
        # Build a list of arguments
        kwarg_list = list()
        for df_chunk in pd_utils.chunk_df(df, chunk_size):
            kwarg_list.append({
                'df_chunk': df_chunk,
                'zone_col': zone_col,
                'segment_col': segment_col,
                'val_col': val_col,
            })

        # Call across multiple threads
        data_chunks = multiprocessing.multiprocess(
            fn=self._dataframe_to_dvec_internal,
            kwargs=kwarg_list,
            process_count=self.process_count,
            # NEED TO PULL IN CHANGES
            # pbar_kwargs=pbar_kwargs,
        )

        return du.sum_dict_list(data_chunks)


# ## FUNCTIONS ## #
def multiply_dvecs(a: DVector,
                   b: DVector,
                   join_on: List[str],
                   ) -> DVector:
    raise NotImplementedError()
