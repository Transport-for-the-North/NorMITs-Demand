# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 11:48:55 2020

@author: Sneezy
"""

from typing import List, Tuple

import pandas as pd
import numpy as np

# Local imports
from normits_demand import AuditError

from normits_demand.utils import general as du


# BACKLOG: Replace efs_zone_translator with the zone translation in TMS
#  labels: EFS, demand merge

class ZoneTranslator:
    def run(self,
            dataframe: pd.DataFrame,
            translation_df: pd.DataFrame,
            from_zoning: str,
            to_zoning: str,
            non_split_cols: List[str],
            needs_zone_id_rename: bool = False,
            tolerance: float = 0.005,
            aggregate_method: str = 'sum',
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
        aggregate_method = aggregate_method.strip().lower()

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

        # Check that all the columns we need actually exist
        translation_cols = [from_zone_col, to_zone_col, switch_col]
        for col in translation_cols:
            if col not in translation_df:
                raise ValueError(
                    "Cannot find all of the needed columns in the translation "
                    "df. Cannot complete translation. Cannot find the "
                    "following column: %s" % str(col)
                )

        if from_zone_col not in translation_df:
            raise ValueError(
                "Cannot find all of the needed columns in the given dataframe. "
                "Cannot complete translation. Cannot find the "
                "following column: %s" % str(from_zone_col)
            )

        # Just grab the columns we need
        translation_df = translation_df.reindex(columns=translation_cols)

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

        if aggregate_method == 'sum':
            new_dataframe = new_dataframe.reindex(columns=index_cols)
            new_dataframe = new_dataframe.groupby(group_cols).sum().reset_index()

        elif aggregate_method == 'mean':
            new_dataframe = new_dataframe.reindex(columns=index_cols)
            new_dataframe = new_dataframe.groupby(group_cols).mean().reset_index()

            # we can't audit when we aggregate using mean
            return new_dataframe

        else:
            raise ValueError(
                "I don't know what aggregate method '%s' is!"
                % str(aggregate_method)
            )

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


class MatrixTotalError(ValueError):
    """Error for when matrix totals are different in `translate_matrix`."""


##### FUNCTIONS #####
def translate_matrix(matrix: pd.DataFrame,
                     lookup: pd.DataFrame,
                     lookup_cols: Tuple[str, str],
                     square_format: bool = True,
                     zone_cols: Tuple[str, str] = None,
                     split_column: str = None,
                     aggregation_method: str = "sum",
                     weights: pd.DataFrame = None,
                     check_total: bool = True,
                     ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Convert a matrix to a new zone system using given lookup.

    Based on the `translate_matrices.translate_matrices` function
    from TMS, simplified for use here.

    Parameters
    ----------
    matrix : pd.DataFrame
        Matrix DataFrame in either square format or list format.
        - square: zones should be defined in the index and column headers.
        - list: should contain 2 zone columns and a column with the values.

    lookup : pd.DataFrame
        Lookup between the desired zone systems, with optional splitting
        factor column.

    lookup_cols : Tuple[str, str]
        Names of the lookup columns containing the current zone system then
        the new zone system.

    square_format : bool, optional
        Whether the provided matrix is in square (True) or list format
        (False), by default True

    zone_cols : Tuple[str, str], optional
        The names of the zone columns in the matrix only required for list
        format matrices, by default None

    split_column : str, optional
        Name of the split column in the lookup, by default None

    aggregation_method : str, optional
        Name of the aggregation method to use, default "sum".

    weights : pd.DataFrame, optional
        Weights for "weighted_average" aggregation method,
        default None. The format should be [o, d, value]
        or square format if `square_format` is true.

    check_total : bool, optional
        Whether or not the matrix total before and after the translation
        should be checked. If True (default) ValueError will be raised
        if the totals differ by more than 0.1. Always False when using
        "weighted_average" aggregation method.

    Returns
    -------
    pd.DataFrame
        Matrix with the new zone system.

    pd.DataFrame
        Matrix of splitting factors for converting back to the old zone
        system.
    """
    avg_method = "weighted_average"
    agg_methods = ["sum", "mean", avg_method]
    check_total = False if aggregation_method == avg_method else check_total
    if aggregation_method == avg_method and not isinstance(weights, pd.DataFrame):
        raise ValueError(
            f"'weights' should be 'DataFrame' when 'weighted_average' "
            f"is chosen not '{type(weights).__name__}'"
        )
    if aggregation_method not in agg_methods:
        raise ValueError(
            "'aggregation_method' should be one of "
            f"{agg_methods}, not '{aggregation_method}'"
        )

    level_names = [matrix.columns.name, matrix.index.name]
    # Make sure the matrix is in list format
    if square_format:
        matrix_total = np.sum(matrix.values)
        matrix = square_to_list(matrix)
        splitting_cols = ["value"]
    else:
        original_columns = matrix.columns.tolist()
        rename = {zone_cols[0]: "o", zone_cols[1]: "d"}
        matrix = matrix.rename(columns=rename)
        splitting_cols = set(original_columns) - set(zone_cols)
        matrix_total = matrix[splitting_cols].sum()
    zone_cols = ["o", "d"]

    if split_column is None:
        split_column = "split"
        lookup[split_column] = 1.0
    lookup, lookup_cols = _lookup_matrix(lookup, lookup_cols, split_column)

    if aggregation_method == avg_method:
        if square_format:
            weights = square_to_list(weights)
        weights = weights.loc[:, [*zone_cols, "value"]].rename(
            columns={"value": "weights"}
        )
        if len(weights) != len(matrix):
            raise ValueError("'weights' and 'matrix' are not the same lengths")

        # TODO(BT): Figure out why these are changing to the wrong types
        for col in ['o', 'd']:
            weights[col] = weights[col].astype(int)

        # Calculate splitting factor based on weights
        lookup = lookup.merge(
            weights,
            left_on=lookup_cols[:2],
            right_on=zone_cols,
            how="left",
            validate="m:1",
        )
        lookup.drop(columns=zone_cols, inplace=True)
        totals = (
            lookup[lookup_cols[:2] + ["weights"]]
            .groupby(lookup_cols[:2], as_index=False)
            .sum()
        )
        lookup = lookup.merge(
            totals,
            on=lookup_cols[:2],
            how="left",
            validate="m:1",
            suffixes=("", "_total"),
        )
        del totals
        lookup["split"] = lookup["weights"] / lookup["weights_total"]
        lookup.drop(columns=["weights", "weights_total"], inplace=True)

    # Check missing zones in lookup
    for col in zone_cols:
        matrix[col] = matrix[col].astype(int)

    missing = np.isin(
        np.unique(matrix[zone_cols].values),
        np.unique(lookup[lookup_cols[:2]].values),
        assume_unique=True,
        invert=True
    )
    if sum(missing) > 0:
        raise ValueError(
            f"{sum(missing)} zones found in matrix columns {zone_cols} which "
            f"aren't present in lookup columns {lookup_cols[:2]}"
        )

    # Convert both columns to new zone system
    matrix = matrix.merge(
        lookup,
        left_on=zone_cols,
        right_on=lookup_cols[:2],
        how="left",
        validate="1:m",
    ).drop(columns=zone_cols)

    for s in splitting_cols:
        matrix[s] = matrix[s] * matrix[split_column]
    reverse = matrix.copy()

    if aggregation_method == avg_method:
        aggregation_method = "sum"
    matrix = (
        matrix[lookup_cols[2:] + list(splitting_cols)]
        .groupby(lookup_cols[2:], as_index=False)
        .agg(aggregation_method)
    )

    # Calculating splitting factors for reversing translation
    reverse = reverse.merge(
        matrix,
        on=lookup_cols[2:],
        how="left",
        validate="m:1",
        suffixes=("", "_total"),
    )
    for s in splitting_cols:
        reverse["split"] = reverse[f"{s}"] / reverse[f"{s}_total"]

    # Convert back to input format and reset column/index names
    matrix = matrix.rename(columns=dict(zip(lookup_cols[2:], zone_cols)))
    if square_format:
        matrix = matrix.pivot(zone_cols[0], zone_cols[1], "value")
        new_total = np.sum(matrix.values)
    else:
        matrix.rename(columns={v: k for k, v in rename.items()}, inplace=True)
        matrix = matrix[original_columns]
        new_total = matrix[splitting_cols].sum()
    matrix.columns.name = level_names[0]
    matrix.index.name = level_names[1]

    # Check matrix totals
    if (check_total and
        not np.allclose(matrix_total, new_total, rtol=0, atol=0.1)):
        diff = abs(matrix_total - new_total)
        raise MatrixTotalError(
            "The matrix total after translation differs "
            f"from input matrix by: {diff:.1E}"
        )
    return matrix, reverse[lookup_cols + ["split"]]


def _lookup_matrix(lookup: pd.DataFrame,
                   lookup_cols: List[str],
                   split: str,
                   ) -> Tuple[pd.DataFrame, List[str]]:
    """Convert a zone corrsepondence lookup to an OD pair lookup.

    Convert from format [old zone, new zone, splitting factor] to
    [old zone - origin, old zone - destination, new zone - origin,
    new zone - destination, OD pair splitting factor].

    Parameters
    ----------
    lookup : pd.DataFrame
        Zone correspondence with 2 lookup columns and a single split
        column.
    lookup_cols : List[str]
        List of the 2 columns used for lookup where first element
        is the old zone system column and the second in the new
        zone system.
    split : str
        The name of the column containing the splitting factors.

    Returns
    -------
    pd.DataFrame
        OD pair lookup with 4 lookup columns based on `lookup_cols`
        values but with '-o' or '-d' appended and updated splitting
        factor column.
    List[str]
        List of the lookup column names, 4 elements.
    """
    od = ("o", "d")
    matrix = {}
    for col in lookup_cols:
        matrix[f"{col}-{od[0]}"] = np.repeat(lookup[col].values, len(lookup))
        matrix[f"{col}-{od[1]}"] = np.tile(lookup[col].values, len(lookup))
    matrix = pd.DataFrame(matrix)

    # Calculate splitting factors
    for col in od:
        matrix = matrix.merge(
            lookup.rename(columns={split: f"{split}-{col}"}),
            how="left",
            left_on=[f"{i}-{col}" for i in lookup_cols],
            right_on=lookup_cols,
            validate="m:1",
        )
    matrix[split] = matrix[f"{split}-{od[0]}"] * matrix[f"{split}-{od[1]}"]
    cols = [f"{c}-{d}" for c in lookup_cols for d in od]
    return matrix[cols + [split]], cols


def square_to_list(matrix: pd.DataFrame) -> pd.DataFrame:
    """Converts square matrix to list format.

    Parameters
    ----------
    matrix : pd.DataFrame
        Matrix in square format.

    Returns
    -------
    pd.DataFrame
        Matrix in list format with the following
        columns: o, d, value
    """
    matrix = matrix.melt(ignore_index=False, var_name="d")
    matrix.index.name = "o"
    matrix.reset_index(inplace=True)
    return matrix
