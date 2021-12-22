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
import warnings

from typing import Any
from typing import List

# Third Party
import numpy as np
import pandas as pd

# Local Imports
import normits_demand as nd
from normits_demand.utils import math_utils
from normits_demand.utils import pandas_utils as pd_utils
from normits_demand.concurrency import multiprocessing


def _lower_memory_row_translation(vector_chunk: np.array,
                                  row_translation: np.array,
                                  ) -> np.array:
    """Internal MP functionality of _lower_memory_matrix_zone_translation()"""
    # Translate the rows
    row_translated = list()
    for vector in np.hsplit(vector_chunk, vector_chunk.shape[1]):
        vector = vector.flatten()
        vector = np.broadcast_to(np.expand_dims(vector, axis=1), row_translation.shape)
        vector = vector * row_translation
        vector = vector.sum(axis=0)
        row_translated.append(vector)

    return row_translated


def _lower_memory_col_translation(vector_chunk: np.array,
                                  col_translation: np.array,
                                  ) -> np.array:
    """Internal MP functionality of _lower_memory_matrix_zone_translation()"""
    # Translate the columns
    col_translated = list()
    for vector in np.vsplit(vector_chunk, vector_chunk.shape[0]):
        vector = vector.flatten()
        vector = np.broadcast_to(np.expand_dims(vector, axis=1), col_translation.shape)
        vector = vector * col_translation
        vector = vector.sum(axis=0)
        col_translated.append(vector)

    return col_translated


def _lower_memory_matrix_zone_translation(matrix: np.array,
                                          row_translation: np.array,
                                          col_translation: np.array,
                                          chunk_size: int,
                                          ) -> np.array:
    # Translate the rows
    kwargs = list()
    n_splits = matrix.shape[1] / chunk_size
    for vector_chunk in np.array_split(matrix, n_splits, axis=1):
        kwargs.append({
            'vector_chunk': vector_chunk,
            'row_translation': row_translation,
        })

    row_translated = multiprocessing.multiprocess(
        fn=_lower_memory_row_translation,
        kwargs=kwargs,
        process_count=nd.constants.PROCESS_COUNT,
        pbar_kwargs={'desc': 'doing rows'},
    )

    row_translated = np.vstack(row_translated).T

    # Translate the rows
    kwargs = list()
    n_splits = row_translated.shape[0] / chunk_size
    for vector_chunk in np.array_split(row_translated, n_splits, axis=0):
        kwargs.append({
            'vector_chunk': vector_chunk,
            'col_translation': col_translation,
        })

    full_translated = multiprocessing.multiprocess(
        fn=_lower_memory_col_translation,
        kwargs=kwargs,
        process_count=nd.constants.PROCESS_COUNT,
        pbar_kwargs={'desc': 'doing cols'},
    )

    return np.vstack(full_translated)


def numpy_matrix_zone_translation(matrix: np.array,
                                  translation: np.array = None,
                                  row_translation: np.array = None,
                                  col_translation: np.array = None,
                                  translation_dtype: np.dtype = None,
                                  check_shapes: bool = True,
                                  check_totals: bool = False,
                                  chunk_size: int = 100,
                                  ) -> np.array:
    """Translates matrix with translation

    Pure numpy operations. Should be super fast!!

    Parameters
    ----------
    matrix:
        The matrix to translate. Needs to be square!
        e.g. (n_in, n_in)

    translation:
        The matrix defining the factors to use to translate the rows and the
        columns of matrix. Setting this argument would be the same as setting
        row_translation and col_translation to the same matrix. If defined,
        row_translation and col_translation will be ignored.
        Should be of shape (n_in, n_out), where the output
        matrix shape will be (n_out, n_out).

    row_translation:
        The matrix defining the factors to use to translate the rows of
        matrix. If defined, then col_translation needs to be defined too.
        Should be of shape (n_in, n_out), where the output
        matrix shape will be (n_out, n_out).

    col_translation:
        The matrix defining the factors to use to translate the rows of
        matrix. If defined, then row_translation needs to be defined too.
        Should be of shape (n_in, n_out), where the output
        matrix shape will be (n_out, n_out).

    translation_dtype:
        The numpy datatype to use to do the translation. If None, then the
        dtype of the matrix is used. As this is a 3d translation, it can
        use a lot of memory if float64 is used. Where such high precision
        isn't needed, a more memory efficient dtype can be passed in instead
        and the translation will be carried out in it.

    check_shapes:
        Whether to check that the input and translation shapes look correct.
        Will raise an error if matrix is not a square array, or if translation
        does not have the same number of rows as matrix.
        Optionally set to False if checks have been done externally to speed
        up runtime.

    check_totals:
        Whether to check that the input and output matrices sum to the same
        total.

    chunk_size:
        The size of the chunks to use if there isn't enough memory to do a
        3D translation. Most of the time this can be ignored unless
        translating between two massive zoning systems.

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
    if translation is not None:
        row_translation = translation.copy()
        col_translation = translation.copy()

    if row_translation is None or col_translation is None:
        raise ValueError(
            "No translations have been given. Don't know how to translate. "
            "Either translation needs to be set, or row_translation AND "
            "col_translations needs to be set."
        )

    # ## OPTIONALLY CHECK INPUT SHAPES ## #
    if check_shapes:
        # Check matrix is square
        mat_rows, mat_columns = matrix.shape
        if mat_rows != mat_columns:
            raise ValueError(
                "The given matrix is not square. Matrix needs to be square "
                "for the numpy zone translations to work.\n"
                "Given matrix shape: %s"
                % str(matrix.shape)
            )

        # Check translations are the same shape
        if row_translation.shape != col_translation.shape:
            raise ValueError(
                "Row and column translations are not the same shape. "
                "Both need to be (n_in, n_out) shape for "
                "numpy zone translations to work.\n"
                "Row shape: %s\n"
                "Column shape: %s"
                % (row_translation.shape, col_translation.shape)
            )

        # Check translation has the right number of rows
        n_zones_in, _ = row_translation.shape
        if n_zones_in != mat_rows:
            raise ValueError(
                "The given translation does not have the correct number of "
                "rows. Translation rows needs to match matrix rows for the "
                "numpy zone translations to work.\n"
                "Given matrix shape: %s\n"
                "Given translation shape: %s"
                % (matrix.shape, row_translation.shape)
            )

    # ## CONVERT DTYPES ## #
    # If not given, assumed based on the input precision
    if translation_dtype is None:
        translation_dtype = matrix.dtype

    # Make sure we're not gonna introduce infs...
    mat_max = np.max(matrix)
    mat_min = np.min(matrix)
    dtype_max = np.finfo(translation_dtype).max
    dtype_min = np.finfo(translation_dtype).min

    if mat_max > dtype_max:
        raise ValueError(
            "Maximum value in matrix is greater than the given "
            "translation_dtype can handle.\n"
            "Maximum dtype value: %s\n"
            "Maximum matrix value: %s"
            % (dtype_max, mat_max)
        )

    if mat_min < dtype_min:
        raise ValueError(
            "Minimum value in matrix is less than the given "
            "translation_dtype can handle.\n"
            "Minimum dtype value: %s\n"
            "Minimum matrix value: %s"
            % (dtype_min, mat_min)
        )

    # Now we know it's safe. Translate
    matrix = matrix.astype(translation_dtype)
    row_translation = row_translation.astype(translation_dtype)
    col_translation = col_translation.astype(translation_dtype)

    # ## DO THE TRANSLATION ## #
    # Get the input and output shapes
    n_in, n_out = row_translation.shape

    # We might run out of memory doing it this way...
    not_enough_memory = False

    try:
        # Translate rows
        mult_shape = (n_in, n_in, n_out)
        a = np.broadcast_to(np.expand_dims(matrix, axis=2), mult_shape)
        trans_a = np.broadcast_to(np.expand_dims(row_translation, axis=1), mult_shape)
        temp = a * trans_a

        # mat is transposed, but we need it this way
        rows_done = temp.sum(axis=0)

        # Translate cols
        mult_shape = (n_in, n_out, n_out)
        b = np.broadcast_to(np.expand_dims(rows_done, axis=2), mult_shape)
        trans_b = np.broadcast_to(np.expand_dims(col_translation, axis=1), mult_shape)
        temp = b * trans_b
        translated_matrix = temp.sum(axis=0)
    except MemoryError:
        not_enough_memory = True

    # Try translation again, a slower way.
    if not_enough_memory:
        translated_matrix = _lower_memory_matrix_zone_translation(
            matrix=matrix,
            row_translation=row_translation,
            col_translation=col_translation,
            chunk_size=chunk_size,
        )

    if not check_totals:
        return translated_matrix

    if not math_utils.is_almost_equal(matrix.sum(), translated_matrix.sum()):
        raise ValueError(
            "Some values seem to have been dropped during the translation. "
            "Check the given translation matrix isn't unintentionally dropping "
            "values. If the difference is small, it's likely a rounding error.\n"
            "Before: %s\n"
            "After: %s"
            % (matrix.sum(), translated_matrix.sum())
        )

    return translated_matrix


def numpy_vector_zone_translation(vector: np.array,
                                  translation: np.array,
                                  translation_dtype: np.dtype = None,
                                  check_shapes: bool = True,
                                  check_totals: bool = False,
                                  ) -> np.array:
    """Translates vector with translation

    Pure numpy operations. Should be super fast!!

    Parameters
    ----------
    vector:
        The Vector to translate. Needs to be one dimensional!
        e.g. (n_in, )

    translation:
        The matrix defining the factors to use to translate matrix. Should
        be of shape (n_in, n_out), where the output vector shape will be
        (n_out, ).

    translation_dtype:
        The numpy datatype to use to do the translation. If None, then the
        dtype of the vector is used. As this is a 2d translation, it can
        use a lot of memory if float64 is used. Where such high precision
        isn't needed, a more memory efficient dtype can be passed in instead
        and the translation will be carried out in it.

    check_shapes:
        Whether to check that the input and translation shapes look correct.
        Will raise an error if vector is not a 1d array, or if translation
        does not have the same number of rows as vector.
        Optionally set to False if checks have been done externally to speed
        up runtime.

    check_totals:
        Whether to check that the input and output vector sum to the same
        total.

    Returns
    -------
    translated_vector:
        vector, translated into (n_out, ) shape via translation.

    Raises
    ------
    ValueError:
        Will raise an error if vector is not a square array, or if translation
        does not have the same number of rows as vector.
    """
    # ## OPTIONALLY CHECK INPUT SHAPES ## #
    if check_shapes:
        # Check that vector is 1D
        if len(vector.shape) > 1:
            if vector.shape[1] == 1:
                vector = vector.flatten()
            else:
                raise ValueError(
                    "The given vector is not a vector. Expected a np.ndarray with "
                    "only one dimension, but got %s dimensions instead."
                    % len(vector.shape)
                )

        # Check translation has the right number of rows
        n_zones_in, _ = translation.shape
        if n_zones_in != len(vector):
            raise ValueError(
                "The given translation does not have the correct number of "
                "rows. Translation rows needs to match vector rows for the "
                "numpy zone translations to work.\n"
                "Given vector shape: %s\n"
                "Given translation shape: %s"
                % (vector.shape, translation.shape)
            )

    # ## CONVERT DTYPES ## #
    # If not given, assumed based on the input precision
    if translation_dtype is None:
        translation_dtype = vector.dtype

    # Make sure we're not gonna introduce infs...
    mat_max = np.max(vector)
    mat_min = np.min(vector)
    dtype_max = np.finfo(translation_dtype).max
    dtype_min = np.finfo(translation_dtype).min

    if mat_max > dtype_max:
        raise ValueError(
            "Maximum value in vector is greater than the given "
            "translation_dtype can handle.\n"
            "Maximum dtype value: %s\n"
            "Maximum vector value: %s"
            % (dtype_max, mat_max)
        )

    if mat_min < dtype_min:
        raise ValueError(
            "Minimum value in vector is less than the given "
            "translation_dtype can handle.\n"
            "Minimum dtype value: %s\n"
            "Minimum vector value: %s"
            % (dtype_min, mat_min)
        )

    # Now we know it's safe. Translate
    vector = vector.astype(translation_dtype)
    translation = translation.astype(translation_dtype)

    # ## TRANSLATE ## #
    out_vector = np.broadcast_to(np.expand_dims(vector, axis=1), translation.shape)
    out_vector = out_vector * translation
    out_vector = out_vector.sum(axis=0)

    if not check_totals:
        return out_vector

    if not math_utils.is_almost_equal(vector.sum(), out_vector.sum()):
        raise ValueError(
            "Some values seem to have been dropped during the translation. "
            "Check the given translation matrix isn't unintentionally dropping "
            "values. If the difference is small, it's likely a rounding error.\n"
            "Before: %s\n"
            "After: %s"
            % (vector.sum(), out_vector.sum())
        )

    return out_vector


def pandas_matrix_zone_translation(matrix: pd.DataFrame,
                                   from_zone_col: str,
                                   to_zone_col: str,
                                   factors_col: str,
                                   from_unique_zones: List[Any],
                                   to_unique_zones: List[Any],
                                   translation: pd.DataFrame = None,
                                   row_translation: pd.DataFrame = None,
                                   col_translation: pd.DataFrame = None,
                                   translation_dtype: np.dtype = None,
                                   matrix_infill: float = 0.0,
                                   translate_infill: float = 0.0,
                                   check_totals: bool = False,
                                   chunk_size: int = 100,
                                   ) -> pd.DataFrame:
    """Translates a Pandas DataFrame from one zoning system to another

    Parameters
    ----------
    matrix:
        The matrix to translate. The index and columns need to be the
        from_zone_system ID

    translation:
        A pandas dataframe with at least 3 columns, defining how the
        factor to translate from from_zone to to_zone. Setting this argument
        would be the same as setting row_translation and col_translation to
        the same value. If defined, row_translation and col_translation will
        be ignored.
        Needs to contain columns [from_zone_col, to_zone_col, factors_col].

    row_translation:
        A pandas dataframe with at least 3 columns, defining how the
        factor to translate from from_zone to to_zone for the rows of matrix.
        If defined, then col_translation needs to be defined too.
        Needs to contain columns [from_zone_col, to_zone_col, factors_col].

    col_translation:
        A pandas dataframe with at least 3 columns, defining how the
        factor to translate from from_zone to to_zone for the rows of matrix.
        If defined, then row_translation needs to be defined too.
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

    translation_dtype:
        The numpy datatype to use to do the translation. If None, then the
        dtype of the translation is used. As this is a 3d translation, it can
        use a lot of memory if float64 is used. Where such high precision
        isn't needed, a more memory efficient dtype can be passed in instead
        and the translation will be carried out in it.

    matrix_infill:
        The value to use to infill any missing matrix values.

    translate_infill:
        The value to use to infill any missing translation factors.

    check_totals:
        Whether to check that the input and output matrices sum to the same
        total.

    chunk_size:
        The size of the chunks to use if there isn't enough memory to do a
        3D translation. Most of the time this can be ignored unless
        translating between two massive zoning systems.

    Returns
    -------
    translated_matrix:
        matrix, translated into to_zone system.
    """
    # Init
    if translation is not None:
        row_translation = translation.copy()
        col_translation = translation.copy()

    if row_translation is None or col_translation is None:
        raise ValueError(
            "No translations have been given. Don't know how to translate. "
            "Either translation needs to be set, or row_translation AND "
            "col_translations needs to be set."
        )

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
    if matrix.index.dtype != row_translation[from_zone_col].dtype:
        raise ValueError(
            "The datatype of the matrix index and columns must be the same "
            "as the translation datatype in from_zone_col for the zone "
            "translation to work.\n"
            "matrix index Dtype: %s\n"
            "row_translation[from_zone_col] Dtype: %s"
            % (matrix.index.dtype, row_translation[from_zone_col].dtype)
        )

    if matrix.index.dtype != col_translation[from_zone_col].dtype:
        raise ValueError(
            "The datatype of the matrix index and columns must be the same "
            "as the translation datatype in from_zone_col for the zone "
            "translation to work.\n"
            "matrix index Dtype: %s\n"
            "col_translation[from_zone_col] Dtype: %s"
            % (matrix.index.dtype, col_translation[from_zone_col].dtype)
        )

    # ## CHECK THE PASSED IN ARGUMENTS ARE VALID ## #
    # Make sure there are no duplicates in from_unique_zones or to_unique_zones
    if len(from_unique_zones) < len(set(from_unique_zones)):
        raise ValueError(
            "Some zones went missing when converting from_unique_zones into "
            "a set. This must mean there are some duplicate zones."
        )

    if len(to_unique_zones) < len(set(to_unique_zones)):
        raise ValueError(
            "Some zones went missing when converting to_unique_zones into "
            "a set. This must mean there are some duplicate zones."
        )

    # Make sure the matrix only has the zones defined in from_unique_zones
    missing_rows = set(matrix.index.to_list()) - set(from_unique_zones)
    if len(missing_rows) > 0:
        warnings.warn(
            "There are some zones in matrix.index that have not been defined in "
            "from_unique_zones. These zones will be dropped before the "
            "translation!\n"
            "Additional rows count: %s"
            % len(missing_rows)
        )

    missing_cols = set(matrix.columns.to_list()) - set(from_unique_zones)
    if len(missing_cols) > 0:
        warnings.warn(
            "There are some zones in matrix.columns that have not been defined in "
            "from_unique_zones. These zones will be dropped before the "
            "translation!"
            "Additional cols count: %s"
            % len(missing_cols)
        )

    # Check all needed values are in from_zone_col zone col
    trans_from_zones = set(row_translation[from_zone_col].unique())
    missing_zones = (set(from_unique_zones) - trans_from_zones)
    if len(missing_zones) != 0:
        warnings.warn(
            "Some zones in the matrix are missing in the row translation! "
            "Infilling missing zones with translation infill.\n"
            "Missing zones count: %s"
            % len(missing_zones)
        )

    trans_from_zones = set(col_translation[from_zone_col].unique())
    missing_zones = (set(from_unique_zones) - trans_from_zones)
    if len(missing_zones) != 0:
        warnings.warn(
            "Some zones in the matrix are missing in the col translation! "
            "Infilling missing zones with translation infill.\n"
            "Missing zones count: %s"
            % len(missing_zones)
        )

    # ## PREP AND TRANSLATE ## #
    # Square the translation
    row_translation = pd_utils.long_to_wide_infill(
        df=row_translation,
        index_col=from_zone_col,
        columns_col=to_zone_col,
        values_col=factors_col,
        index_vals=from_unique_zones,
        column_vals=to_unique_zones,
        infill=translate_infill
    )

    # Can speed up with translation is not None - row and col translations are same
    if translation is not None:
        col_translation = row_translation.copy()
    else:
        col_translation = pd_utils.long_to_wide_infill(
            df=col_translation,
            index_col=from_zone_col,
            columns_col=to_zone_col,
            values_col=factors_col,
            index_vals=from_unique_zones,
            column_vals=to_unique_zones,
            infill=translate_infill
        )

    # Make sure all zones are in the matrix and infill 0s
    matrix = matrix.reindex(
        index=from_unique_zones,
        columns=from_unique_zones,
        fill_value=matrix_infill,
    )

    # Translate
    translated = numpy_matrix_zone_translation(
        matrix=matrix.values,
        row_translation=row_translation.values,
        col_translation=col_translation.values,
        translation_dtype=translation_dtype,
        check_totals=check_totals,
        chunk_size=chunk_size,
    )

    # Stick into pandas
    return pd.DataFrame(
        data=translated,
        index=to_unique_zones,
        columns=to_unique_zones,
    )


def pandas_vector_zone_translation(vector: pd.DataFrame,
                                   translation: pd.DataFrame,
                                   from_zone_col: str,
                                   to_zone_col: str,
                                   factors_col: str,
                                   from_unique_zones: List[Any],
                                   to_unique_zones: List[Any],
                                   translation_dtype: np.dtype = None,
                                   vector_infill: float = 0.0,
                                   translate_infill: float = 0.0,
                                   check_totals: bool = False,
                                   ) -> pd.DataFrame:
    """Translates a Pandas DataFrame from one zoning system to another

    Parameters
    ----------
    vector:
        The vector to translate. The index needs to be the
        from_zone_system ID

    translation:
        A pandas dataframe with at least 3 columns, defining how the
        factor to translate from from_zone to to_zone.
        Needs to contain columns [from_zone_col, to_zone_col, factors_col].

    from_zone_col:
        The name of the column in translation containing the from_zone system
        ID. Values should be in the same format as vector index.

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

    translation_dtype:
        The numpy datatype to use to do the translation. If None, then the
        dtype of the vector is used. As this is a 2d translation, it can
        use a lot of memory if float64 is used. Where such high precision
        isn't needed, a more memory efficient dtype can be passed in instead
        and the translation will be carried out in it.

    vector_infill:
        The value to use to infill any missing matrix values.

    translate_infill:
        The value to use to infill any missing translation factors.

    check_totals:
        Whether to check that the input and output matrices sum to the same
        total.

    Returns
    -------
    translated_vector:
        vector, translated into to_zone system.
    """
    # ## CHECK ZONE NAME DTYPES ## #
    # Check the matrix and translation dtypes match
    if vector.index.dtype != translation[from_zone_col].dtype:
        raise ValueError(
            "The datatype of the vector index must be the same "
            "as the translation datatype in from_zone_col for the zone "
            "translation to work.\n"
            "vector index Dtype: %s\n"
            "translation[from_zone_col] Dtype: %s"
            % (vector.index.dtype, translation[from_zone_col].dtype)
        )

    # ## CHECK THE PASSED IN ARGUMENTS ARE VALID ## #
    # Make sure there are no duplicates in from_unique_zones or to_unique_zones
    if len(from_unique_zones) < len(set(from_unique_zones)):
        raise ValueError(
            "Some zones went missing when converting from_unique_zones into "
            "a set. This must mean there are some duplicate zones."
        )

    if len(to_unique_zones) < len(set(to_unique_zones)):
        raise ValueError(
            "Some zones went missing when converting to_unique_zones into "
            "a set. This must mean there are some duplicate zones."
        )

    # Make sure the vector only has the zones defined in from_unique_zones
    missing_rows = set(vector.index.to_list()) - set(from_unique_zones)
    if len(missing_rows) > 0:
        warnings.warn(
            "There are some zones in vector.index that have not been defined in "
            "from_unique_zones. These zones will be dropped before the "
            "translation!\n"
            "Additional rows count: %s"
            % len(missing_rows)
        )

    # Check all needed values are in from_zone_col zone col
    trans_from_zones = set(translation[from_zone_col].unique())
    missing_zones = (set(from_unique_zones) - trans_from_zones)
    if len(missing_zones) != 0:
        warnings.warn(
            "Some zones in the vector are missing in the translation! "
            "Infilling missing zones with translation infill.\n"
            "Missing zones count: %s"
            % len(missing_zones)
        )

    # ## PREP AND TRANSLATE ## #
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

    # Make sure all zones are in the vector and infill 0s
    vector = vector.reindex(
        index=from_unique_zones,
        fill_value=vector_infill,
    )

    # Translate
    translated = numpy_vector_zone_translation(
        vector=vector.values,
        translation=translation.values,
        translation_dtype=translation_dtype,
        check_totals=check_totals,
    )

    # Stick into pandas
    return pd.DataFrame(
        data=translated,
        index=to_unique_zones,
        columns=vector.columns,
    )
