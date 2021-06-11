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

from typing import Dict

# Third Party
import numpy as np
import pandas as pd

import tqdm

# Local Imports
from normits_demand import core


# ## CLASSES ## #
class DVector:

    _zone_col = 'zone_col'
    _segment_col = 'segment'
    _val_col = 'val'

    def __init__(self,
                 zoning_system: core.ZoningSystem,
                 segmentation: core.SegmentationLevel,
                 import_data: pd.DataFrame,
                 zone_col: str = None,
                 segment_col: str = None,
                 val_col: str = None,
                 verbose: bool = False,
                 ):
        # Init
        self.zoning_system = zoning_system
        self.segmentation = segmentation

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
                verbose,
            )
        else:
            raise NotImplementedError(
                "Don't know how to deal with anything other than a pandas DF"
            )

    def _dataframe_to_dvec(self,
                           df: pd.DataFrame,
                           zone_col: str,
                           segment_col: str,
                           val_col: str,
                           verbose: bool = False,
                           ) -> Dict[str, np.ndarray]:
        """
        Converts a pandas dataframe into dvec.data internal structure
        """
        # Init
        needed_cols = [zone_col, segment_col, val_col]

        # TODO(BT): Once the segmentation object is properly implemented this
        #  iterator should be determined from that object!
        unique_segments = df['segment'].unique()

        # ## VALIDATE AND CONVERT THE GIVEN DATAFRAME ## #
        # TODO(BT): Make this an error throwing utils function!
        # Check that all the given columns actually exist in the data
        for col in needed_cols:
            if col not in df:
                raise ValueError(
                    "No columns named '%s' in the given dataframe.\n"
                    "Only found the following columns: %s"
                    % (col, list(df))
                )

        df = df.reindex(columns=needed_cols)

        # Rename import_data columns to internal names
        rename_dict = {
            zone_col: self._zone_col,
            segment_col: self._segment_col,
            val_col: self._val_col
        }
        df = df.rename(columns=rename_dict)

        # ## CONVERT THE DATAFRAME INTO A DVEC DATA ## #
        # setup a pbar
        pbar_kwargs = {
            'desc': "Converting df to dvec",
            'unit': "segment",
            'disable': (not verbose),
        }

        # Generate the data on a per segment basis
        dvec_data = dict()
        for segment in tqdm.tqdm(unique_segments, **pbar_kwargs):
            # Get all available pop for this segment
            seg_data = df[df[self._segment_col] == segment].copy()

            # Filter down to just data as values, and zoning system as the index
            seg_data = seg_data.reindex(columns=[self._zone_col, self._val_col])
            seg_data = seg_data.set_index(self._zone_col)

            # Infill any missing zones as 0
            seg_data = seg_data.reindex(self.zoning_system.unique_zones, fill_value=0)

            # Assign to dict for storage
            dvec_data[segment] = seg_data.values

        return dvec_data

# ## FUNCTIONS ## #
