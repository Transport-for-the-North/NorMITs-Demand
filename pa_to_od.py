# -*- coding: utf-8 -*-
"""
Created on: Fri September 11 12:46:25 2020
Updated on:

Original author: Ben Taylor
Last update made by:
Other updates made by:

File purpose:
Collection of functions for translating PA matrices into OD matrices.
TODO: After integrations with TMS, combine with pa_to_od.py
  to create a general pa_to_od.py file

"""

import numpy as np
import pandas as pd

# Can call tms pa_to_od.py functions from here
from old_tms.pa_to_od import *


def simplify_time_period_splits(time_period_splits: pd.DataFrame):
    """
    Simplifies time_period_splits to a case where the purpose_from_home
    is always the same as the purpose_to_home

    Parameters
    ----------
    time_period_splits:
        A time_period_splits dataframe extracted using get_time_period_splits()

    Returns
    -------
    time_period_splits only where the purpose_from_home
    is the same as purpose_to_home
    """
    time_period_splits = time_period_splits.copy()

    # Problem column doesn't exist in this case
    if 'purpose_to_home' not in time_period_splits.columns:
        return time_period_splits

    # Build a mask where purposes match
    unq_purpose = time_period_splits['purpose_from_home'].drop_duplicates()
    keep_rows = np.array([False] * len(time_period_splits))
    for p in unq_purpose:
        purpose_mask = (
            (time_period_splits['purpose_from_home'] == p)
            & (time_period_splits['purpose_to_home'] == p)
        )
        keep_rows = keep_rows | purpose_mask

    time_period_splits = time_period_splits.loc[keep_rows]

    # Filter down to just the needed col and return
    needed_cols = [
        'purpose_from_home',
        'time_from_home',
        'time_to_home',
        'direction_factor']
    return time_period_splits.reindex(needed_cols, axis='columns')
