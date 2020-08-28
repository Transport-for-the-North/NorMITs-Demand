# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 11:48:55 2020

@author: Sneezy
"""

from typing import List

import pandas as pd
import numpy as np


class ZoneTranslator:
    def run(self,
            dataframe: pd.DataFrame,
            translation_dataframe: pd.DataFrame,
            start_zoning_system_name: str,
            end_zoning_system_name: str,
            non_split_columns: List[str],
            needs_zone_id_rename: bool = True
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

        translation_dataframe:
            The dataframe to be used for translating the zoning system.
            No default input.
            Possible input is any Pandas Dataframe with the columns: [
            (start_zoning_system_name + "_zone_id"),
            (end_zoning_system_name + "_zone_id"),
            (start_zoning_system_name + "_to_" + end_zoning_system_name)
            ]

        start_zoning_system_name:
            The name of the starting zoning system.
            No default input.
            Possible input is any string.

        end_zoning_system_name:
            The name of the end zoning system.
            No default input.
            Possible input is any string.

        non_split_columns:
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
        translation_dataframe = translation_dataframe.copy()

        # avoid case problems
        start_zoning_system_name = start_zoning_system_name.lower()
        end_zoning_system_name = end_zoning_system_name.lower()

        # TODO: Add check to make sure non_split_columns are in dataframe
        # set up columns
        start_zoning_column = start_zoning_system_name + "_zone_id"
        end_zoning_column = end_zoning_system_name + "_zone_id"
        switch_column = '_'.join([
            start_zoning_system_name,
            "to",
            end_zoning_system_name
        ])

        split_columns = list(set(dataframe.columns) - set(non_split_columns))

        if needs_zone_id_rename:
            dataframe = dataframe.rename(columns={"model_zone_id": start_zoning_column})

        translation_dataframe = translation_dataframe[[
            start_zoning_column,
            end_zoning_column,
            switch_column
        ]]

        new_dataframe = pd.merge(
            dataframe,
            translation_dataframe,
            on=[start_zoning_column]
        )

        if needs_zone_id_rename:
            new_dataframe = new_dataframe.rename(columns={end_zoning_column: "model_zone_id"})

        for split_column in split_columns:
            new_dataframe.loc[:, split_column] = (
                new_dataframe.loc[:, split_column]
                *
                new_dataframe.loc[:, switch_column]
            )

        # Extract just the required columns
        needed_columns = non_split_columns.copy()
        needed_columns.extend(split_columns)

        new_dataframe = new_dataframe[needed_columns]
        new_dataframe = new_dataframe.groupby(
            by=non_split_columns,
            as_index=False
        ).sum()

        return new_dataframe
