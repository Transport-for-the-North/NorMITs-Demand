# -*- coding: utf-8 -*-
"""
Created on: Mon June 08:12:21 2021
Updated on:

Original author: Ben Taylor
Last update made by:
Other updates made by:

File purpose:
Collection of utility functions specifically for manipulating pandas
"""
# Builtins
import functools

from typing import Any
from typing import Dict
from typing import List
from typing import Generator

# Third Party
import pandas as pd

# Local
from normits_demand.utils import general as du


def reindex_cols(df: pd.DataFrame,
                 columns: List[str],
                 throw_error: bool = True,
                 **kwargs,
                 ) -> pd.DataFrame:
    """
    Wrapper around df.reindex. Will throw error if columns aren't in df

    Parameters
    ----------
    df:
        THe pandas.DataFrame that should be reindexed

    columns:
        The columns to reindex df to.

    throw_error:
        Whether to throw and error or not if the given columns don't exist in
        df. If False, then operates exactly like calling df.reindex directly.

    kwargs:
        Any extra arguments to pass into df.reindex

    Returns
    -------
    reindexed_df:
        The given df, reindexed to only have columns as column names.

    Raises
    ------
    ValueError:
        If any of the given columns don't exists within df and throw_error is
        True.
    """
    if throw_error:
        # Check that all columns actually exist in df
        for col in columns:
            if col not in df:
                raise ValueError(
                    "No columns named '%s' in the given dataframe.\n"
                    "Only found the following columns: %s"
                    % (col, list(df))
                )

    return df.reindex(columns=columns, **kwargs)


def reindex_and_groupby(df: pd.DataFrame,
                        index_cols: List[str],
                        value_cols: List[str],
                        throw_error: bool = True,
                        **kwargs,
                        ) -> pd.DataFrame:
    """
    Wrapper around df.reindex() and df.groupby().

    Optionally throws an error if index_cols aren't in df. Will throw an
    error by default

    Parameters
    ----------
    df:
        The pandas.DataFrame that should be reindexed and grouped.

    index_cols:
        List of column names to reindex to.

    value_cols:
        List of column names that contain values. Groupby will be performed
        on any columns that are in value_cols, but not index_cols.

    throw_error:
        Whether to throw an error if not all index_cols are in the df.

    Returns
    -------
    new_df:
        A copy of df that has been reindexed and grouped.
    """
    # ## VALIDATE INPUTS ## #
    if throw_error:
        for col in index_cols:
            if col not in df:
                raise ValueError(
                    "No columns named '%s' in the given dataframe.\n"
                    "Only found the following columns: %s"
                    % (col, list(df))
                )

    for col in value_cols:
        if col not in index_cols:
            raise ValueError(
                "Value '%s' from value_cols is not in index_cols."
                "Can only accept value_cols that are in index_cols."
                % col
            )
    
    # Generate the group cols
    group_cols = du.list_safe_remove(index_cols, value_cols)

    df = df.reindex(columns=index_cols, **kwargs)
    return df.groupby(group_cols).sum().reset_index()


def filter_df(df: pd.DataFrame,
              df_filter: Dict[str, Any],
              ) -> pd.DataFrame:
    """
    Filters a DataFrame by df_filter.

    Parameters
    ----------
    df:
        The pandas.Dataframe to filter.

    df_filter:
        Dictionary of {column: valid_value} pairs to define the filter to be
        applied. Will return only where all column conditions are met.

    Returns
    -------
    filtered_df:
        A copy of df, filtered down to df_filter.

    """
    # Wrap each item if a list to avoid errors
    for k, v in df_filter.items():
        if not pd.api.types.is_list_like(v):
            df_filter[k] = [v]

    needed_cols = list(df_filter.keys())
    mask = df[needed_cols].isin(df_filter).all(axis='columns')
    return df[mask]


def str_join_cols(df: pd.DataFrame,
                  columns: List[str],
                  separator: str = '_',
                  ) -> pd.Series:
    """
    Equivalent to separator.join(columns) for all rows of df

    Joins the given columns together using separator. Returns a pandas Series
    with the return value in.

    Parameters
    ----------
    df:
        The dataframe containing the columns to join

    columns:
        The columns in df to concatenate together

    separator:
        The separator to use when joining columns together.

    Returns
    -------
    joined_column:
        a Pandas.Series containing all columns joined together using separator
    """
    # Define the accumulator function
    def reducer(accumulator, item):
        return accumulator + separator + item

    # Join the cols together
    join_cols = [df[x].astype(str) for x in columns]
    return functools.reduce(reducer, join_cols)


def chunk_df(df: pd.DataFrame,
             chunk_size: int,
             ) -> Generator[pd.DataFrame, None, None]:
    """
    Yields chunk_size chunks of df

    Parameters
    ----------
    df:
        the pandas.DataFrame to chunk.

    chunk_size:
        The size of the chunks to use, in terms of rows.

    Yields
    ------
    df_chunk:
        A chunk of the given df of size chunk_size
    """
    for i in range(0, len(df), chunk_size):
        chunk_end = i + chunk_size
        yield df[i:chunk_end]


def long_to_wide_infill(df: pd.DataFrame,
                        index_col: str,
                        columns_col: str,
                        values_col: str,
                        index_vals: List[str] = None,
                        column_vals: List[str] = None,
                        infill: Any = 0,
                        ) -> pd.DataFrame:
    """
    Converts a DataFrame from long to wide format, infilling missing values.

    Parameters
    ----------
    df:
        The dataframe, in long format, to convert to wide.

    index_col:
        The column of df to use as the index of the wide return DataFrame

    columns_col:
        The column of df to use as the columns of the wide return DataFrame

    values_col:
        The column of df to use as the values of the wide return DataFrame

    index_vals:
        The unique values to use as the index of the wide return DataFrame.
        If left as None, df[index_col].unique() will be used.

    column_vals:
        The unique values to use as the columns of the wide return DataFrame.
        If left as None, df[columns_col].unique() will be used.

    infill:
        The value to use to infill any missing cells in the wide DataFrame.

    Returns
    -------
    wide_df:
        A copy of df, in wide format, with index_col as the index,
        columns_col as the column names, and values_col as the values.
    """
    # Init
    index_vals = df[index_col].unique() if index_vals is None else index_vals
    column_vals = df[columns_col].unique() if column_vals is None else column_vals
    df = reindex_cols(df, [index_col, columns_col, values_col])

    # Make sure every possible combination exists
    new_index = pd.MultiIndex.from_product(
        [index_vals, column_vals],
        names=[index_col, columns_col]
    )

    df = df.set_index([index_col, columns_col])
    df = df.reindex(new_index, fill_value=infill).reset_index()

    # Convert to wide
    df = df.pivot(
        index=index_col,
        columns=columns_col,
        values=values_col,
    )

    return df
