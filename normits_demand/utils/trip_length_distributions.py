# -*- coding: utf-8 -*-
"""
Created on: 29/09/2021
Updated on:

Original author: Ben Taylor
Last update made by: Ben Taylor
Other updates made by: Chris Storey

File purpose:
Collection of utility functions for dealing with trip distributions
"""
# Built-Ins
import os

from typing import Any
from typing import Optional

# Third Party
import pandas as pd

# Local Imports
import normits_demand as nd


def get_trip_length_distributions(import_dir: nd.PathLike,
                                  segment_params: nd.SegmentParams,
                                  trip_origin: str,
                                  replace_nan: Optional[Any] = None,
                                  ) -> pd.DataFrame:
    """Returns the trip length distribution for segment params in import_folder

    Parameters
    ----------
    import_dir:
        The directory to look for trip length distributions

    segment_params:
        A dictionary where the keys are the names of the segments and
        the values are the segment values to search for. This is used
        to select the correct trip length distribution.

    trip_origin:
        Where the trip originated. Usually either 'hb' or 'nhb'

    replace_nan:
        Whether to replace any NaN values found in the trip length
        distribution with 0 values or not

    Returns
    -------

    """
    # TODO(BT): Tidy up get_trip_length_distributions
    # Append name of tlb area

    # Index folder
    target_files = os.listdir(import_dir)
    # Define file contents, should just be target files - should fix.
    import_files = target_files.copy()

    for key, value in segment_params.items():
        # Don't want empty segments, don't want ca
        if value != 'none' and key != 'mat_type':
            # print_w_toggle(key + str(value), echo=echo)
            import_files = [x for x in import_files if
                            ('_' + key + str(value)) in x]

    if trip_origin == 'hb':
        import_files = [x for x in import_files if 'nhb' not in x]
    elif trip_origin == 'nhb':
        import_files = [x for x in import_files if 'nhb' in x]
    else:
        raise ValueError('Trip length band import failed,' +
                         'provide valid trip origin')

    if len(import_files) <= 0:
        raise IOError(
            "Cannot find any %s trip length bands.\n"
            "import folder: %s"
            % (trip_origin, import_dir)
        )

    for key, value in segment_params.items():
        # Don't want empty segments, don't want ca
        if value != 'none' and key != 'mat_type':
            # print_w_toggle(key + str(value), echo=echo)
            import_files = [x for x in import_files if
                            ('_' + key + str(value)) in x]

    if len(import_files) <= 0:
        raise IOError(
            "Cannot find any import files matching the given criteria.\n"
            'Import folder: %s\n'
            'Search criteria: %s'
            % (import_dir, segment_params)
        )

    if len(import_files) > 1:
        raise Warning(
            'Found multiple viable files. Cannot pick one.\n'
            'Search criteria: %s\n'
            'Import folder: %s\n'
            'Viable files: %s'
            % (segment_params, import_dir, import_files)
        )

    # Import
    tlb = pd.read_csv(os.path.join(import_dir, import_files[0]))

    if replace_nan is not None:
        for col_name in list(tlb):
            tlb[col_name] = tlb[col_name].fillna(replace_nan)

    return tlb
