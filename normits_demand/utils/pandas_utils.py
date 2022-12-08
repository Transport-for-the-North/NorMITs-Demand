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
import os
import operator
import re
import warnings
import functools

from typing import Any
from typing import Dict
from typing import List
from typing import Callable
from typing import Iterable
from typing import Generator

# Third Party
import numpy as np
import pandas as pd
import openpyxl

from openpyxl.utils import dataframe as openpyxl_dataframe

# Local
from normits_demand.utils import general as du
from normits_demand.utils import math_utils


def reindex_cols(df: pd.DataFrame,
                 columns: List[str],
                 throw_error: bool = True,
                 dataframe_name: str = None,
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

    dataframe_name:
        The name to give to the dataframe in the error message being thrown.
        If left as none "the given dataframe" is used instead.

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
    # Init
    df = df.copy()

    if dataframe_name is None:
        dataframe_name = 'the given dataframe'

    if throw_error:
        # Check that all columns actually exist in df
        for col in columns:
            if col not in df:
                raise ValueError(
                    "No columns named '%s' in %s.\n"
                    "Only found the following columns: %s"
                    % (col, dataframe_name, list(df))
                )

    return df.reindex(columns=columns, **kwargs)


def reindex_rows_and_cols(
    df: pd.DataFrame,
    index: List[Any],
    columns: List[Any],
    fill_value: Any = np.nan,
    **kwargs,
) -> pd.DataFrame:
    """
    Wrapper around `df.reindex()` to make sure index/col types don't clash

    If the type of the index of the column names in df does not match the
    types given in index or columns, the index types will be cast to the
    desired types before calling the reindex.

    Parameters
    ----------
    df:
        The pandas.DataFrame that should be reindexed

    index:
        The index to reindex df to.

    columns:
        The columns to reindex df to.

    fill_value:
        Value to use for missing values. Defaults to NaN, but can be
        any “compatible” value.

    kwargs:
        Any extra arguments to pass into df.reindex

    Returns
    -------
    reindexed_df:
        The given df, reindexed to the index and columns given.
    """
    # Init
    idx_dtype = type(index[0])
    col_dtype = type(columns[0])

    # Cast types if needed
    if df.index.dtype != idx_dtype:
        df.index = df.index.astype(idx_dtype)

    if df.columns.dtype != col_dtype:
        df.columns = df.columns.astype(idx_dtype)

    return df.reindex(
        columns=columns,
        index=index,
        fill_value=fill_value,
        **kwargs
    )


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


def filter_df_mask(df: pd.DataFrame,
                   df_filter: Dict[str, Any],
                   ) -> pd.DataFrame:
    """
    Generates a mask for filtering df by df_filter.

    Parameters
    ----------
    df:
        The pandas.Dataframe to filter.

    df_filter:
        Dictionary of {column: valid_value} pairs to define the filter to be
        applied. Will return only where all column conditions are met.

    Returns
    -------
    filter_mask:
        A mask, which when applied, will filter df down to df_filter.
    """
    # Init
    df_filter = df_filter.copy()

    # Wrap each item if a list to avoid errors
    for k, v in df_filter.items():
        if not pd.api.types.is_list_like(v):
            df_filter[k] = [v]

    needed_cols = list(df_filter.keys())
    mask = df[needed_cols].isin(df_filter).all(axis='columns')

    return mask


def filter_df(df: pd.DataFrame,
              df_filter: Dict[str, Any],
              throw_error: bool = False,
              ) -> pd.DataFrame:
    """
    Filters a DataFrame by df_filter.

    Parameters
    ----------
    df:
        The pandas.Dataframe to filter.

    df_filter:
        Dictionary of `{column: valid_values}` pairs to define the filter to be
        applied. `valid_values` can be a single value or a list of values.
        Will return only where all column conditions are met.

    throw_error:
        Whether to throw an error if the filtered dataframe has no
        rows left

    Returns
    -------
    filtered_df:
        A copy of df, filtered down to df_filter.

    """
    # Generate and apply mask
    mask = filter_df_mask(df=df, df_filter=df_filter)
    return_df = df[mask].copy()

    if throw_error:
        if return_df.empty:
            raise ValueError(
                "An empty dataframe was returned after applying the filter. "
                "Are you sure the correct data was passed in?\n"
                f"Given filter: {df_filter}"
            )

    return return_df


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
    Yields df_chunk_size chunks of df

    Parameters
    ----------
    df:
        the pandas.DataFrame to chunk.

    chunk_size:
        The size of the chunks to use, in terms of rows.

    Yields
    ------
    df_chunk:
        A chunk of the given df of size df_chunk_size
    """
    for i in range(0, len(df), chunk_size):
        chunk_end = i + chunk_size
        yield df[i:chunk_end]


def long_product_infill(df: pd.DataFrame,
                        index_col_1: str,
                        index_col_2: str,
                        index_col_1_vals: List[Any] = None,
                        index_col_2_vals: List[Any] = None,
                        infill: Any = 0,
                        ) -> pd.DataFrame:
    """
    Infills missing values of col_1 and col_2 using a product of given values.

    Parameters
    ----------
    df:
        The dataframe, in long format, to infill.

    index_col_1:
        The first column of df to infill. Will be infilled with
        index_col_1_vals, repeated len(index_col_2_vals) times.

    index_col_2:
        The second column of df to infill. Will be infilled with
        index_col_2_vals, repeated len(index_col_1_vals) times.

    index_col_1_vals:
        The unique values to use as the first index of the return dataframe.
        These unique values will be combined with every combination of
        index_col_2_vals to create the full index.
        If left as None, df[index_col].unique() will be used.

    index_col_2_vals:
        The unique values to use as the first index of the return dataframe.
        These unique values will be combined with every combination of
        index_col_1_vals to create the full index.
        If left as None, df[columns_col].unique() will be used.

    infill:
        The value to use to infill any missing cells in the return DataFrame.

    Returns
    -------
    infilled_df:
        A copy of df, in wide format, with index_col as the index,
        columns_col as the column names, and values_col as the values.
    """
    # Init
    index_col_1_vals = df[index_col_1].unique() if index_col_1_vals is None else index_col_1_vals
    index_col_2_vals = df[index_col_2].unique() if index_col_2_vals is None else index_col_2_vals

    # Make sure were not dropping too much. Indication of problems in arguments
    missing_idx = set(index_col_1_vals) - set(df[index_col_1].unique().tolist())
    if len(missing_idx) >= len(set(index_col_1_vals)) * 0.9:
        warnings.warn(
            "Almost all index_col_1_vals do not exist in df[index_col_1]. Are the "
            "given data types matching?\n"
            "There are %s missing values."
            % len(missing_idx)
        )

    missing_cols = set(index_col_2_vals) - set(df[index_col_2].unique().tolist())
    if len(missing_cols) >= len(set(index_col_2_vals)) * 0.9:
        warnings.warn(
            "Almost all index_col_2_vals do not exist in df[index_col_2]. Are the "
            "given data types matching?\n"
            "There are %s missing values."
            % len(missing_cols)
        )

    # Make sure every possible combination exists
    new_index = pd.MultiIndex.from_product(
        [index_col_1_vals, index_col_2_vals],
        names=[index_col_1, index_col_2]
    )
    df = df.set_index([index_col_1, index_col_2])
    df = df.reindex(index=new_index, fill_value=infill).reset_index()

    return df


def long_to_wide_infill(df: pd.DataFrame,
                        index_col: str,
                        columns_col: str,
                        values_col: str,
                        index_vals: List[Any] = None,
                        column_vals: List[Any] = None,
                        infill: Any = 0,
                        check_totals: bool = False,
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

    check_totals:
        Whether to check if the totals are almost equal before and after the
        conversion.

    Returns
    -------
    wide_df:
        A copy of df, in wide format, with index_col as the index,
        columns_col as the column names, and values_col as the values.
    """
    # Init
    index_vals = df[index_col].unique() if index_vals is None else index_vals
    column_vals = df[columns_col].unique() if column_vals is None else column_vals
    orig_total = df[values_col].values.sum()
    df = reindex_cols(df, [index_col, columns_col, values_col])

    # Make sure were not dropping too much. Indication of problems in arguments
    missing_idx = set(index_vals) - set(df[index_col].unique().tolist())
    if len(missing_idx) >= len(set(index_vals)) * 0.9:
        warnings.warn(
            "Almost all index_vals do not exist in df[index_col]. Are the "
            "given data types matching?\n"
            "There are %s missing values."
            % len(missing_idx)
        )

    missing_cols = set(column_vals) - set(df[columns_col].unique().tolist())
    if len(missing_cols) >= len(set(column_vals)) * 0.9:
        warnings.warn(
            "Almost all column_vals do not exist in df[columns_col]. Are the "
            "given data types matching?\n"
            "There are %s missing values."
            % len(missing_cols)
        )

    df = long_product_infill(
        df=df,
        index_col_1=index_col,
        index_col_2=columns_col,
        index_col_1_vals=index_vals,
        index_col_2_vals=column_vals,
        infill=infill,
    )

    # Convert to wide
    df = df.pivot(
        index=index_col,
        columns=columns_col,
        values=values_col,
    )

    if not check_totals:
        return df

    # Make sure nothing was dropped
    after_total = df.values.sum()
    if not math_utils.is_almost_equal(after_total, orig_total):
        raise ValueError(
            "Values have been dropped when reindexing the given dataframe.\n"
            "Starting total: %s\n"
            "Ending total: %s."
            % (orig_total, after_total)
        )

    return df


def wide_to_long_infill(df: pd.DataFrame,
                        index_col_1_name: str,
                        index_col_2_name: str,
                        value_col_name: str,
                        index_col_1_vals: List[Any] = None,
                        index_col_2_vals: List[Any] = None,
                        infill: Any = 0,
                        check_totals: bool = False,
                        ) -> pd.DataFrame:
    """Convert a matrix from wide to long format

    Converts a DataFrame from wide to long format, infilling missing values.

    Parameters
    ----------
    df:
        The dataframe, in wide format, to convert to long. The index of df
        must be the values that are to become index_col_1_name, and the
        columns of df will be melted to become index_col_2_name.

    index_col_1_name:
        The name to give to the column that was the index in the wide df

    index_col_2_name:
        The name to give to the column that was the column names in the wide df

    value_col_name:
        The name to give to the column that was the values in the wide df

    index_col_1_vals:
        The unique values to use as the first index of the return dataframe.
        These unique values will be combined with every combination of
        index_col_2_vals to create the full index.
        If left as None, melted_df[index_col_1_name].unique() will be used.

    index_col_2_vals:
        The unique values to use as the second index of the return dataframe.
        These unique values will be combined with every combination of
        index_col_1_vals to create the full index.
        If left as None, melted_df[index_col_2_name].unique() will be used.

    infill:
        The value to use to infill any missing cells in the return DataFrame.

    check_totals:
        Whether to check if the totals are almost equal before and after the
        conversion.

    Returns
    -------
    long_df:
        A copy of df, in wide format, with index_col as the index,
        columns_col as the column names, and values_col as the values.
    """
    # Init
    orig_total = df.values.sum()

    # Assume the index is the first ID
    df = df.reset_index()
    df = df.rename(columns={df.columns[0]: index_col_1_name})

    # Convert to long
    df = df.melt(
        id_vars=index_col_1_name,
        var_name=index_col_2_name,
        value_name=value_col_name,
    )

    # Infill anything that's missing
    df = long_product_infill(
        df=df,
        index_col_1=index_col_1_name,
        index_col_2=index_col_2_name,
        index_col_1_vals=index_col_1_vals,
        index_col_2_vals=index_col_2_vals,
        infill=infill,
    )

    if not check_totals:
        return df

    # Make sure nothing was dropped
    after_total = df[value_col_name].values.sum()
    if not math_utils.is_almost_equal(after_total, orig_total):
        raise ValueError(
            "Values have been dropped when reindexing the given dataframe.\n"
            "Starting total: %s\n"
            "Ending total: %s."
            % (orig_total, after_total)
        )

    return df


def long_df_to_wide_ndarray(df: pd.DataFrame,
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
        The dataframe, in long format, to convert to wide ndarray ndarray.

    index_col:
        The column of df to use as the index of the return ndarray.

    columns_col:
        The column of df to use as the columns of the return ndarray.

    values_col:
        The column of df to use as the values of the return ndarray.

    index_vals:
        The unique values, and their order, to use as the implicit index
        of the wide return ndarray.
        If left as None, df[columns_col].unique() will be used.

    column_vals:
        The unique values, and their order, to use as the implicit column
        headers of the wide return ndarray.
        If left as None, df[columns_col].unique() will be used.

    infill:
        The value to use to infill any missing cells in the wide DataFrame.

    Returns
    -------
    wide_ndarray:
        An ndarray, in wide format, with index_col as the index,
        columns_col as the column names, and values_col as the values.
    """
    df = long_to_wide_infill(
        df=df,
        index_col=index_col,
        columns_col=columns_col,
        values_col=values_col,
        index_vals=index_vals,
        column_vals=column_vals,
        infill=infill,
    )
    return df.values


def get_wide_mask(df: pd.DataFrame,
                  zones: List[Any] = None,
                  col_zones: List[Any] = None,
                  index_zones: List[Any] = None,
                  join_fn: Callable = operator.and_
                  ) -> np.ndarray:
    """
    Generates a mask for a wide matrix. Returned mask will be same shape as df

    The zones the set the mask for can be set individually with col_zones and
    index_zones, or to the same value with zones.


    Parameters
    ----------
    df:
        The dataframe to generate the mask for

    zones:
        The zones to match to in both the columns and index. If this value
        is set it will overwrite anything passed into col_zones and
        index_zones.

    col_zones:
        The zones to match to in the columns. This value is ignored if
        zones is set.

    index_zones:
        The zones to match to in the index. This value is ignored if
        zones is set.

    join_fn:
        The function to call on the column and index masks to join them.
        By default, a bitwise and is used. See pythons builtin operator
        library for more options.

    Returns
    -------
    mask:
        A mask of true and false values. Will be the same shape as df.
    """
    # Validate input args
    if zones is None:
        if col_zones is None or index_zones is None:
            raise ValueError(
                "If zones is not set, both col_zones and row_zones need "
                "to be set."
            )
    else:
        col_zones = zones
        index_zones = zones

    # Try and cast to the correct types for rows/cols
    try:
        # Assume columns are strings if they are an object
        col_dtype = df.columns.dtype
        col_dtype = str if col_dtype == object else col_dtype
        col_zones = np.array(col_zones, col_dtype)
    except ValueError:
        raise ValueError(
            "Cannot cast the col_zones to the required dtype to match the "
            "dtype of the given df columns. Tried to cast to: %s"
            % str(df.columns.dtype)
        )

    try:
        index_zones = np.array(index_zones, df.index.dtype)
    except ValueError:
        raise ValueError(
            "Cannot cast the index_zones to the required dtype to match the "
            "dtype of the given df index. Tried to cast to: %s"
            % str(df.index.dtype)
        )

    # Create square masks for the rows and cols
    col_mask = np.broadcast_to(df.columns.isin(col_zones), df.shape)
    index_mask = np.broadcast_to(df.index.isin(index_zones), df.shape).T

    # Combine together to get the full mask
    return join_fn(col_mask, index_mask)


def get_internal_mask(df: pd.DataFrame,
                      zones: List[Any],
                      ) -> np.ndarray:
    """
    Generates a mask for a wide matrix. Returned mask will be same shape as df

    Parameters
    ----------
    df:
        The dataframe to generate the mask for

    zones:
        A list of zone numbers that make up the internal zones

    Returns
    -------
    mask:
        A mask of true and false values. Will be the same shape as df.
    """
    return get_wide_mask(df=df, zones=zones, join_fn=operator.and_)


def get_external_mask(df: pd.DataFrame,
                      zones: List[Any],
                      ) -> np.ndarray:
    """
    Generates a mask for a wide matrix. Returned mask will be same shape as df

    Parameters
    ----------
    df:
        The dataframe to generate the mask for

    zones:
        A list of zone numbers that make up the external zones

    Returns
    -------
    mask:
        A mask of true and false values. Will be the same shape as df.
    """
    return get_wide_mask(df=df, zones=zones, join_fn=operator.or_)


def get_external_values(df: pd.DataFrame,
                        zones: List[Any],
                        ) -> pd.DataFrame:
    """Get only the external values in df

    External values contains internal-external, external-internal, and
    external-external. All values not meeting this criteria will be set
    to 0.

    Parameters
    ----------
    df:
        The dataframe to get the external values from

    zones:
        A list of zone numbers that make up the external zones

    Returns
    -------
    external_df:
        A dataframe containing only the external demand from df.
        Will be the same shape as df.
    """
    return df * get_external_mask(df, zones)


def get_internal_values(df: pd.DataFrame,
                        zones: List[Any],
                        ) -> pd.DataFrame:
    """Get only the internal values in df

    Internal values contains internal-internal. All values not
    meeting this criteria will be set to 0.

    Parameters
    ----------
    df:
        The dataframe to get the external values from

    zones:
        A list of zone numbers that make up the internal zones

    Returns
    -------
    internal_df:
        A dataframe containing only the internal demand from df.
        Will be the same shape as df.
    """
    return df * get_internal_mask(df, zones)


def internal_external_report(df: pd.DataFrame,
                             internal_zones: List[Any],
                             external_zones: List[Any],
                             ) -> pd.DataFrame:
    """Generates a report df of values in internal/external zones

    Generates a dataframe with 4 rows, each showing the total across
    that portion of the matrix. The dataframe is split into:
    internal-internal
    internal-external
    external-internal
    external-external

    Parameters
    ----------
    df:
        The dataframe to generate the report on.

    internal_zones:
        A list of the internal zones of the zoning system used by df

    external_zones
        A list of the external zones of the zoning system used by df

    Returns
    -------
    report:
        A report of internal and external demand in df.
    """
    # Build the initial report
    index = pd.Index(['internal', 'external'])
    report = pd.DataFrame(
        index=index,
        columns=index,
        data=np.zeros((len(index), len(index)))
    )

    # Build the kwargs to iterate over
    report_kwargs = {
        ('internal', 'internal'): {'index_zones': internal_zones, 'col_zones': internal_zones},
        ('internal', 'external'): {'index_zones': internal_zones, 'col_zones': external_zones},
        ('external', 'internal'): {'index_zones': external_zones, 'col_zones': internal_zones},
        ('external', 'external'): {'index_zones': external_zones, 'col_zones': external_zones},
    }

    # Build the report from the kwargs
    for (row_idx, col_idx), kwargs in report_kwargs.items():
        # Pull out just the trips for this section
        mask = get_wide_mask(
            df=df,
            join_fn=operator.and_,
            **kwargs,
        )
        total = (df * mask).values.sum()

        # Feel like this indexing is backwards...
        report[col_idx][row_idx] = total

    # Add a total row and column
    report['total'] = report.values.sum(axis=1)
    report.loc['total'] = report.values.sum(axis=0)

    return report


def _openpyxl_df_to_excel(
    df: pd.DataFrame,
    path: os.PathLike,
    sheet_name: str = "Sheet1",
    header: bool = True,
    index: bool = True,
    start_row: int = 0,
    start_col: int = 0,
) -> None:
    """Append a DataFrame to existing Excel workbook

    Append `df` to existing Excel file `path` into `sheet_name` Sheet.
    `start_row` and `start_col` are used to define the cell that the top left
    corner of `df` will begin.
    If `path` doesn't exist, then it is created.

    Parameters
    ----------
    df:
        DataFrame to save to workbook

    path:
        Path to the Excel workbook to write to.

    sheet_name:
        Name of the sheet to save `df`.

    header:
        Whether to write out the column names.

    index:
        Whether to write out the index (row) names.

    start_row:
        Upper left cell row to dump `df`.

    start_col:
        Upper left cell col to dump `df`.

    Returns
    -------
    None
    """
    # Create the workbook if it doesn't exist
    if os.path.isfile(path):
        workbook = openpyxl.load_workbook(path)
    else:
        workbook = openpyxl.Workbook()

    # Grab the sheet to write to
    if sheet_name in workbook.sheetnames:
        sheet = workbook[sheet_name]
    else:
        sheet = workbook.create_sheet(sheet_name)

    # Write each cell of df into individual cells
    df_rows = openpyxl_dataframe.dataframe_to_rows(df, index=index, header=header)
    for r_idx, row in enumerate(df_rows, 1):
        for c_idx, value in enumerate(row, 1):
            sheet.cell(
                row=start_row + r_idx,
                column=start_col + c_idx,
                value=value,
            )

    workbook.save(filename=path)


def append_df_to_excel(
    df: pd.DataFrame,
    path: os.PathLike,
    sheet_name: str = "Sheet1",
    header: bool = True,
    index: bool = True,
    start_row: int = 0,
    start_col: int = 0,
    keep_data_validation: bool = False,
    **to_excel_kwargs,
) -> None:
    """Append a DataFrame to existing Excel workbook

    Append `df` to existing Excel file `path` into `sheet_name` Sheet.
    `start_row` and `start_col` are used to define the cell that the top left
    corner of `df` will begin.
    If `path` doesn't exist, then it is created.

    Parameters
    ----------
    df:
        DataFrame to save to workbook

    path:
        Path to the Excel workbook to write to.

    sheet_name:
        Name of the sheet to save `df`.

    header:
        Whether to write out the column names.

    index:
        Whether to write out the index (row) names.

    start_row:
        Upper left cell row to dump `df`.

    start_col:
        Upper left cell col to dump `df`.

    keep_data_validation:
        If left as False, `df.to_excel()` is used to write out df. However,
        this can cause compatibility issues with data validation in the Excel
        workbook. If this is the cases, set to True. If True, uses openpyxl
        functionality to write out `df` instead, reducing functionality of the
        write, but keeping the data validation.

    to_excel_kwargs:
        Any further kwargs to pass to `df.to_excel()`
        Ignored if `keep_data_validation` is True.

    Returns
    -------
    None
    """
    if keep_data_validation:
        return _openpyxl_df_to_excel(
            df=df,
            path=path,
            sheet_name=sheet_name,
            header=header,
            index=index,
            start_row=start_row,
            start_col=start_col,
        )

    # Excel file doesn't exist - saving and exiting
    if not os.path.isfile(path):
        df.to_excel(
            path,
            header=header,
            index=index,
            startrow=start_row,
            startcol=start_col,
            **to_excel_kwargs,
        )
        return

    # ignore engine parameter if it was passed
    if 'engine' in to_excel_kwargs:
        to_excel_kwargs.pop('engine')

    with pd.ExcelWriter(    # type: ignore[abstract]
        path,
        engine='openpyxl',
        mode='a',
        if_sheet_exists='overlay',
    ) as writer:
        df.to_excel(
            writer,
            sheet_name=sheet_name,
            header=header,
            index=index,
            startrow=start_row,
            startcol=start_col,
            **to_excel_kwargs,
        )


def prepend_cols(
    df: pd.DataFrame,
    col_names: Iterable[Any],
    col_vals: Iterable[Any],
    allow_duplicates: bool = False,
) -> pd.DataFrame:
    """Prepend the given columns to the dataframe

    Adds the given columns to the start of the dataframe, in the order given.
    That is, if a dataframe has columns ["d", "e", "f"], and the `col_names`
    given are ["a", "b", "c"], the resultant dataframe would have columns in
    the following order:["a", "b", "c", "d", "e", "f"].

    Parameters
    ----------
    df:
        The original dataframe to prepend `col_names` to with `col_values`.

    col_names:
        The names to give to the columns being prepended to `df`.

    col_vals:
        The values to give to each `col_names` being added to `df`. The index
        if each value should match that in `col_names`. That is:
        `df[col_name[idx]] = col_vals[idx]`

    allow_duplicates:
        Whether to allow this function to add a column to df if one with the
        same name already exists.

    Returns
    -------
    prepended_df:
        The original `df` with the given columns prepended with their values.
    """
    df = df.copy()

    # Validate inputs
    if len(col_names) != len(col_vals):
        raise ValueError(
            "col_names and col_vals need to be the same length. Got lengths:\n"
            f"col_names: {len(col_names)}\n"
            f"col_vals: {len(col_vals)}"
        )

    # Attach all columns in the given order
    for name, val in zip(col_names, col_vals):
        df.insert(loc=0, column=name, value=val, allow_duplicates=allow_duplicates)

    return df


def column_name_tidy(df: pd.DataFrame) -> pd.DataFrame:
    """Convert column names to lowercase and replace spaces with '_'.

    Parameters
    ----------
    df : pd.DataFrame
        Pandas DataFrame to be updated, will be edited inplace.

    Returns
    -------
    pd.DataFrame
        `df` with updated column names.
    """

    def rename(col) -> str:
        return re.sub(r"\s+", "_", str(col).lower().strip())

    df.columns = [rename(c) for c in df.columns]
    return df


def tidy_dataframe(
    df: pd.DataFrame,
    rename: bool = True,
    drop_unnamed: bool = True,
    nan_columns: bool = True,
    nan_rows: bool = True,
    nan_index: bool = True
) -> pd.DataFrame:
    """Drop Nans and normalise column names.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to update, will not be edited inplace.
    rename : bool, default True
        Whether to run `column_name_tidy`.
    drop_unnamed : bool, default True
        Whether to drop columns starting with 'unnamed'.
    nan_columns : bool, default True
        Whether to drop columns with all Nans.
    nan_rows : bool, default True
        Whether to drop rows with all Nans.
    nan_index : bool, default True
        Whether to drop rows with Nan index value.

    Returns
    -------
    pd.DataFrame
        Copy of `df` with filtering or renaming done.
    """
    if rename:
        df = column_name_tidy(df.copy())

    if drop_unnamed:
        unnamed = [c for c in df.columns if c.startswith("unnamed:")]
        df = df.drop(columns=unnamed)

    if nan_columns:
        df = df.dropna(axis=1, how="all")

    if nan_rows:
        df = df.dropna(axis=0, how="all")

    if nan_index:
        df = df.loc[~df.index.isna()]

    return df
