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

from typing import List
from typing import Dict
from typing import Tuple
from typing import Union

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
                 import_data: Union[pd.DataFrame, nd.DVectorData],
                 zone_col: str = None,
                 segment_col: str = None,
                 val_col: str = None,
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

        if process_count < 0:
            process_count = os.cpu_count() + process_count
        self.process_count = process_count

        # Set defaults if args not set
        zone_col = self._zone_col if zone_col is None else zone_col
        segment_col = self._segment_col if segment_col is None else segment_col
        val_col = self._val_col if val_col is None else val_col

        # Try to convert the given data into DVector format
        if isinstance(import_data, pd.DataFrame):
            self.data = self._dataframe_to_dvec(
                import_data,
                zone_col,
                segment_col,
                val_col,
            )
        elif isinstance(import_data, dict):
            self.data = self._dict_to_dvec(import_data)
        else:
            raise NotImplementedError(
                "Don't know how to deal with anything other than: "
                "pandas DF, or dict"
            )

    def _dict_to_dvec(self, import_data) -> nd.DVectorData:
        # TODO(BT): Add some error checking to make sure this is
        #  actually a valid dict
        return import_data

    def _dataframe_to_dvec_internal(self,
                                    df_chunk,
                                    zone_col,
                                    segment_col,
                                    val_col,
                                    ) -> nd.DVectorData:
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
                           ) -> nd.DVectorData:
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
            'disable': (not self.verbose),
            'total': round(len(df) / self.chunk_size)
        }

        # If the dataframe is smaller than the chunk size, evenly split across cores
        if len(df) < self.chunk_size * self.process_count:
            chunk_size = math.ceil(len(df) / self.process_count)
        else:
            chunk_size = self.chunk_size

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

    @staticmethod
    def _multiply(a: DVector,
                  b: DVector,
                  ) -> DVector:
        """
        Builds a new Dvec by multiplying a and b together.

        How to join the two Dvectors is defined by the segmentation of each
        Dvector.

        Retains process_count, chunk_size, and verbose params from a.

        Parameters
        ----------
        a:
            The first DVector to multiply
        b:
            The second DVector to multiply

        Returns
        -------
        c:
            A new DVector which is the product of multiplying a and b.
        """
        # ## CHECK WE CAN MULTIPLY a AND b ## #
        if a.zoning_system == b.zoning_system:
            return_zoning_system = a.zoning_system
        elif a.zoning_system is None:
            return_zoning_system = b.zoning_system
        elif b.zoning_system is None:
            return_zoning_system = a.zoning_system
        else:
            raise nd.ZoningError(
                "Cannot multiply two Dvectors using different zoning systems.\n"
                "zoning system of a: %s\n"
                "zoning system of b: %s\n"
                % (a.zoning_system.name, b.zoning_system.name)
            )

        # ## DO MULTIPLICATION ## #
        # Use the segmentations to figure out what to multiply
        multiply_dict, return_segmentation = a.segmentation * b.segmentation

        # Build the dvec data here with multiplication
        # TODO(NK): Translate your multiplication code to build the dvec_data
        #  here, using the multiply_dict above. The multiply_dict defines
        #  How the multiplication of the Dvecs should be done.
        #  Key = Returning Dvec segment name
        #  Values = Tuple[a_segment_name, b_segment_name]
        dvec_data = dict()

        return DVector(
            zoning_system=return_zoning_system,
            segmentation=return_segmentation,
            import_data=dvec_data,
            process_count=a.process_count,
            verbose=a.verbose,
        )


# ## FUNCTIONS ## #
def multiply_dvecs(a: DVector,
                   b: DVector,
                   ) -> DVector:
    """
    Builds a new Dvec by multiplying a and b together.

    How to join the two Dvectors is defined by the segmentation of each
    Dvector.

    Parameters
    ----------
    a:
        The first DVector to multiply
    b:
        The second DVector to multiply

    Returns
    -------
    c:
        A new DVector which is the product of multiplying a and b.
    """
    return DVector._multiply(a, b)

    mult_dvec = dict()

    for k in b:
        k1 = str(k).split("_", 1)[1]
        for l in a:
            if k1 == l:
                mult_dvec[k] = np.multiply(a[l], b[k])
    """
    dvec_trips = dict()
    c = 1
    needed_cols = ['p', 'tfn_tt']
    p_tfntt = pd_utils.str_join_cols(b, needed_cols)
    
    for m in p_tfntt:
        for o in mult_dvec:
            o1 = o.rsplit("_", 1)[0]
            if o1 == m and c == 1:
                dvec_trips[m] = mult_dvec[o]
                c += 1
            elif o1 == m and c > 1:
                dvec_trips[m] = np.add(dvec_trips[m], mult_dvec[o])
        c = 1
    """
    return mult_dvec



