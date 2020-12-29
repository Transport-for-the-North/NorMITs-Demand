# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 11:48:55 2020

@author: Sneezy
"""

from typing import List

import pandas as pd
import numpy as np

from audits import AuditError
from demand_utilities import utils as du


class ZoneTranslator:
    def run(self,
            dataframe: pd.DataFrame,
            translation_df: pd.DataFrame,
            from_zoning: str,
            to_zoning: str,
            non_split_cols: List[str],
            needs_zone_id_rename: bool = False,
            tolerance: float = 0.005
            ) -> pd.DataFrame:
        """
        Calculates the sector totals off the given parameters.

        Parameters
        ----------
        dataframe:
            The dataframe which will have the zone translation performed on it.
            No default input.
            Possible input is any Pandas Dataframe with a "model_zone_id" or
            (start_zoning_system_name + "_zone_id") column.

        translation_df:
            The dataframe to be used for translating the zoning system.
            No default input.
            Possible input is any Pandas Dataframe with the columns: [
            (start_zoning_system_name + "_zone_id"),
            (end_zoning_system_name + "_zone_id"),
            (start_zoning_system_name + "_to_" + end_zoning_system_name)
            ]

        from_zoning:
            The name of the starting zoning system.
            No default input.
            Possible input is any string.

        to_zoning:
            The name of the end zoning system.
            No default input.
            Possible input is any string.

        non_split_cols:
            The columns that are *not* to be split. Must include "model_zone_id"
            or similar. To be used when, for example, a column is a string or
            categorical value.
            No default input.
            Possible input is any list of strings including "model_zone_id" or
            similar.

        needs_zone_id_rename:
            Whether we need to rename "model_zone_id" to the old zoning system
            and then to the new one.
            Default input is: True
            Possible inputs are: True, False

        Return
        ----------
        new_dataframe:
            The returned totals for the new zoning system in a Pandas dataframe
            format.

        Future Improvements
        ----------
        None yet.
        """
        # copy
        dataframe = dataframe.copy()
        translation_df = translation_df.copy()

        # avoid case problems
        from_zoning = from_zoning.lower()
        to_zoning = to_zoning.lower()

        # TODO: Add check to make sure non_split_columns are in dataframe
        # Set up columns
        from_zone_col = from_zoning + "_zone_id"
        to_zone_col = to_zoning + "_zone_id"
        switch_col = "%s_to_%s" % (from_zoning, to_zoning)

        split_cols = list(set(dataframe.columns) - set(non_split_cols))

        # Remove the zone columns if in there
        zone_cols = [from_zone_col, to_zone_col]
        non_split_cols = du.list_safe_remove(non_split_cols, zone_cols)
        split_cols = du.list_safe_remove(split_cols, zone_cols)

        # Get total for the splitting columns
        split_totals = dict()
        for col in split_cols:
            split_totals[col] = dataframe[col].sum()

        if needs_zone_id_rename:
            dataframe = dataframe.rename(columns={"model_zone_id": from_zone_col})

        # Just grab the columns we need
        needed_cols = [from_zone_col, to_zone_col, switch_col]
        translation_df = translation_df.reindex(columns=needed_cols)

        new_dataframe = pd.merge(
            dataframe,
            translation_df,
            on=from_zone_col
        )

        if needs_zone_id_rename:
            new_dataframe = new_dataframe.rename(columns={to_zone_col: "model_zone_id"})

        for split_column in split_cols:
            new_dataframe[split_column] *= new_dataframe[switch_col]

        # Extract just the required columns
        group_cols = [to_zone_col] + non_split_cols.copy()
        index_cols = group_cols.copy() + split_cols.copy()

        new_dataframe = new_dataframe.reindex(columns=index_cols)
        new_dataframe = new_dataframe.groupby(group_cols).sum().reset_index()

        # Audit what comes out the other side
        for col, val in split_totals.items():
            lower = val - (val*tolerance)
            upper = val + (val*tolerance)

            if not (lower < new_dataframe[col].sum() < upper):
                raise AuditError(
                    "More than the tolerance of demand was dropped during zone "
                    "translation.\n"
                    "Column: %s\n"
                    "Demand before: %f\n"
                    "Demand after: %f\n"
                    % (col, val, new_dataframe[col].sum())
                )

        return new_dataframe
