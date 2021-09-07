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


def numpy_zone_translation(matrix: np.array,
                           translation: np.array,
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

    Returns
    -------
    translated_matrix:
        matrix, translated into (n_out, n_out) shape via translation.

    """
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
    translated = numpy_zone_translation(
        matrix=matrix.values,
        translation=translation.values,
    )

    # Stick into pandas
    return pd.DataFrame(
        data=translated,
        index=to_unique_zones,
        columns=to_unique_zones,
    )

