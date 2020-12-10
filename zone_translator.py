# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 11:48:55 2020

@author: Sneezy
"""

from typing import List, Tuple

import pandas as pd
import numpy as np

from demand_utilities.utils import zone_translation_df

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


##### FUNCTIONS #####
def translate_matrix(
    matrix: pd.DataFrame,
    lookup: pd.DataFrame,
    lookup_cols: Tuple[str, str],
    square_format: bool = True,
    zone_cols: Tuple[str, str] = None,
    split_column: str = None,
) -> pd.DataFrame:
    """Convert a matrix to a new zone system using given lookup.

    Based on the `translate_matrices.translate_matrices` function from TMS,
    simplied for use here.

    Parameters
    ----------
    matrix : pd.DataFrame
        Matrix DataFrame in either square format or list format.
        - square: zones should be defined in the index and column headers.
        - list: should contain 2 zone columns and a column with the values.
    lookup : pd.DataFrame
        Lookup between the desired zone systems, with optional splitting factor
        column.
    lookup_cols : Tuple[str, str]
        Names of the lookup columns containing the current zone system then
        the new zone system.
    square_format : bool, optional
        Whether the provided matrix is in square (True) or list format (False),
        by default True
    zone_cols : Tuple[str, str], optional
        The names of the zone columns in the matrix only required for list format
        matrices, by default None
    split_column : str, optional
        Name of the split column in the lookup, by default None

    Returns
    -------
    pd.DataFrame
        Matrix with the new zone system.
    """
    level_names = [matrix.columns.name, matrix.index.name]
    # Make sure the matrix is in list format
    if square_format:
        matrix = matrix.melt(ignore_index=False, var_name="d")
        matrix.index.name = "o"
        matrix.reset_index(inplace=True)
        splitting_cols = ["value"]
    else:
        original_columns = matrix.columns.tolist()
        rename = {zone_cols[0]: "o", zone_cols[1]: "d"}
        matrix = matrix.rename(columns=rename)
        splitting_cols = set(original_columns) - set(zone_cols)
    zone_cols = ["o", "d"]

    # Convert both columns to new zone system
    for c in zone_cols:
        matrix = matrix.merge(lookup, left_on=c, right_on=lookup_cols[0], how="left")
        matrix.drop(columns=[c, lookup_cols[0]], inplace=True)
        matrix.rename(columns={lookup_cols[1]: c}, inplace=True)
        # Multiply columns by splitting factors if given
        if split_column is not None:
            for s in splitting_cols:
                matrix[s] = matrix[s] * matrix[split_column]
            matrix.drop(columns=split_column, inplace=True)
    # Aggregate rows together
    matrix = matrix.groupby(zone_cols, as_index=False).sum()

    # Convert back to input format and reset column/index names
    if square_format:
        matrix = matrix.pivot(zone_cols[0], zone_cols[1], "value")
    else:
        matrix.rename(columns={v: k for k, v in rename.items()}, inplace=True)
        matrix = matrix[original_columns]
    matrix.columns.name = level_names[0]
    matrix.index.name = level_names[1]
    return matrix
