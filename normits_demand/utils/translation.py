# -*- coding: utf-8 -*-
"""
Created on: 07/09/2021
Updated on:

Original author: Ben Taylor
Last update made by:
Other updates made by:

File purpose:

"""
# Built-Ins

from typing import List

# Third Party
import numpy as np
import pandas as pd

# Local Imports
from normits_demand.utils import pandas_utils as pd_utils


def numpy_matrix_zone_translation(matrix: np.array,
                                  translation: np.array,
                                  check_shapes: bool = True,
                                  ) -> np.array:
    """Translates matrix with translation

    Pure numpy operations. Should be super fast!!

    Parameters
    ----------
    matrix:
        The matrix to translate. Needs to be square!
        e.g. (n_in, n_in)

    translation:
        The matrix defining the factors to use to translate matrix. Should
        be of shape (n_in, n_out), where the output matrix shape will be
        (n_out, n_out).

    check_shapes:
        Whether to check that the input and translation shapes look correct.
        Will raise an error if matrix is not a square array, or if translation
        does not have the same number of rows as matrix.
        Optionally set to False if checks have been done externally to speed
        up runtime.

    Returns
    -------
    translated_matrix:
        matrix, translated into (n_out, n_out) shape via translation.

    Raises
    ------
    ValueError:
        Will raise an error if matrix is not a square array, or if translation
        does not have the same number of rows as matrix.
    """
    # ## OPTIONALLY CHECK INPUT SHAPES ## #
    if check_shapes:
        # Check matrix is square
        mat_rows, mat_columns = matrix.shape
        if mat_rows != mat_columns:
            raise ValueError(
                "The given matrix is not square. Matrix needs to be square "
                "for the numpy zone translations to work.\n"
                "Given matrix shape: %s"
                % matrix.shape
            )

        # Check translation has the right number of rows
        n_zones_in, _ = translation.shape
        if n_zones_in != mat_rows:
            raise ValueError(
                "The given translation does not have the correct number of "
                "rows. Translation rows needs to match matrix rows for the "
                "numpy zone translations to work.\n"
                "Given matrix shape: %s\n"
                "Given translation shape: %s"
                % (matrix.shape, translation.shape)
            )

    # ## DO THE TRANSLATION ## #
    # Get the input and output shapes
    n_in, n_out = translation.shape

    # Translate rows
    mult_shape = (n_in, n_in, n_out)
    a = np.broadcast_to(np.expand_dims(matrix, axis=2), mult_shape)
    trans_a = np.broadcast_to(np.expand_dims(translation, axis=1), mult_shape)
    temp = a * trans_a

    # mat is transposed, but we need it this way
    out_mat = temp.sum(axis=0)

    # Translate cols
    mult_shape = (n_in, n_out, n_out)
    b = np.broadcast_to(np.expand_dims(out_mat, axis=2), mult_shape)
    trans_b = np.broadcast_to(np.expand_dims(translation, axis=1), mult_shape)
    temp = b * trans_b
    out_mat_2 = temp.sum(axis=0)

    return out_mat_2


def numpy_vector_zone_translation(vector: np.array,
                                  translation: np.array,
                                  ) -> np.array:
    pass


def pandas_matrix_zone_translation(matrix: pd.DataFrame,
                                   translation: pd.DataFrame,
                                   from_zone_col: str,
                                   to_zone_col: str,
                                   factors_col: str,
                                   from_unique_zones: List[str],
                                   to_unique_zones: List[str],
                                   translate_infill: float = 0.0,
                                   ) -> pd.DataFrame:
    """Translates a Pandas DataFrame from one zoning system to another


    Parameters
    ----------
    matrix:
        The matrix to translate. The index and columns need to be the
        from_zone_system ID

    translation:
        A pandas dataframe with at least 3 columns, defining how the
        factor to translate from from_zone to to_zone.
        Needs to contain columns [from_zone_col, to_zone_col, factors_col].

    from_zone_col:
        The name of the column in translation containing the from_zone system
        ID. Values should be in the same format as matrix index and columns.

    to_zone_col:
        The name of the column in translation containing the to_zone system
        ID. Values should be in the same format as expected in the output.

    factors_col:
        The name of the column in translation containing the translation
        factors between from_zone and to_zone. Where zone pairs do not exist,
        they will be infilled with translate_infill.

    from_unique_zones:
        A list of all the unique zones in the from_zone system. Used to know
        where an infill is needed for missing zones in translation.

    to_unique_zones:
        A list of all the unique zones in the to_zone system. Used to know
        where an infill is needed for missing zones in translation.

    translate_infill:
        The value to use to infill any missing translation factors.

    Returns
    -------
    translated_matrix:
        matrix, translated into to_zone system.
    """
    # TODO (BT): Add a check to make sure no demand is being dropped
    # ## CHECK ZONE NAME DTYPES ## #
    # Check the matrix index and column dtypes match
    if matrix.columns.dtype != matrix.index.dtype:
        raise ValueError(
            "The datatype of the index and columns in matrix must be the same "
            "for the zone translation to work.\n"
            "Index Dtype: %s\n"
            "Column Dtype: %s"
            % (matrix.index.dtype, matrix.columns.dtype)
        )

    # Check the matrix and translation dtypes match
    if matrix.index.dtype != translation[from_zone_col].dtype:
        raise ValueError(
            "The datatype of the matrix index and columns must be the same "
            "as the translation datatype in from_zone_col for the zone "
            "translation to work.\n"
            "matrix index Dtype: %s\n"
            "translation[from_zone_col] Dtype: %s"
            % (matrix.index.dtype, translation[from_zone_col].dtype)
        )


    # Check all values in matrix are in from zone col
    row_zones = matrix.index.to_list()
    col_zones = matrix.columns.to_list()
    mat_zones = set(row_zones + col_zones)

    trans_from_zones = set(translation[from_zone_col].unique())

    missing_zones = (mat_zones - trans_from_zones)
    if len(missing_zones) != 0:
        print(
            "Some zones in the matrix are missing in the translation!\n"
            "Missing zones count: %s"
            % len(missing_zones)
        )

    # Square the translation
    translation = pd_utils.long_to_wide_infill(
        df=translation,
        index_col=from_zone_col,
        columns_col=to_zone_col,
        values_col=factors_col,
        index_vals=from_unique_zones,
        column_vals=to_unique_zones,
        infill=translate_infill
    )

    # Translate
    translated = numpy_matrix_zone_translation(
        matrix=matrix.values,
        translation=translation.values,
    )

    # Stick into pandas
    return pd.DataFrame(
        data=translated,
        index=to_unique_zones,
        columns=to_unique_zones,
    )

