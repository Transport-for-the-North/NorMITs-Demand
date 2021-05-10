# -*- coding: utf-8 -*-
"""
Created on: Mon February 15 10:36:32 2020
Updated on:

Original author: Ben Taylor
Last update made by:
Other updates made by:

File purpose:
A collection of functions for validating function arguments are one of
the possible selections.
"""
# Builtins
import warnings

from typing import Any
from typing import List

# Local imports
import normits_demand as nd

from normits_demand import constants as consts
from normits_demand import efs_constants as efs_consts

# TODO: Functions to move into here
from normits_demand.utils.general import validate_model_name
from normits_demand.utils.general import validate_zoning_system
from normits_demand.utils.general import validate_seg_level
from normits_demand.utils.general import validate_scenario_name
from normits_demand.utils.general import validate_model_name_and_mode
from normits_demand.utils.general import validate_user_class
from normits_demand.utils.general import validate_vdm_seg_params


def validate_vector_type(vector_type: str) -> str:
    """
    Tidies up vector_type and raises an exception if not valid

    Parameters
    ----------
    vector_type:
        The name of the vector type to validate

    Returns
    -------
    vector_type:
        vector_type with both strip and lower applied to remove any whitespace
        and make it all lowercase

    Raises
    -------
    ValueError:
        If vector_type is not a valid name for a type of vector
    """
    # Init
    vector_type = vector_type.strip().lower()

    if vector_type not in consts.VECTOR_TYPES:
        raise ValueError(
            "%s is not a valid vector type. It should be one of: %s"
            % (vector_type, str(consts.VECTOR_TYPES))
        )
    return vector_type


def validate_trip_origin(trip_origin: str) -> str:
    """
    Tidies up trip_origin and raises an exception if not valid

    Parameters
    ----------
    trip_origin:
        The name of the trip origin to validate

    Returns
    -------
    trip_origin:
        trip_origin with both strip and lower applied to remove any whitespace
        and make it all lowercase

    Raises
    -------
    ValueError:
        If trip_origin is not a valid name for a trip origin
    """
    # Init
    trip_origin = trip_origin.strip().lower()

    if trip_origin not in efs_consts.TRIP_ORIGINS:
        raise ValueError(
            "%s is not a valid trip origin. It should be one of: %s"
            % (trip_origin, str(efs_consts.TRIP_ORIGINS))
        )
    return trip_origin


def validate_matrix_format(matrix_format: str) -> str:
    """
    Tidies up matrix_type and raises an exception if not valid

    Parameters
    ----------
    matrix_format:
        The name of the matrix format to validate

    Returns
    -------
    matrix_format:
        matrix_format with both strip and lower applied to remove any whitespace
        and make it all lowercase

    Raises
    -------
    ValueError:
        If matrix_format is not a valid name for a type of matrix
    """
    # Init
    matrix_format = matrix_format.strip().lower()

    if matrix_format not in efs_consts.VALID_MATRIX_FORMATS:
        raise ValueError(
            "%s is not a valid matrix format. It should be one of: %s"
            % (matrix_format, str(efs_consts.VALID_MATRIX_FORMATS))
        )
    return matrix_format


def all_values_set(values: List[Any],
                   msg: str,
                   default_values: List[Any] = None,
                   error: bool = False,
                   warn: bool = False,
                   ) -> None:
    """
    Checks if all the values

    Parameters
    ----------
    values:
        A list of the values to validate.

    msg:
        The message to print/warn/error if all of the values are not set.

    default_values:
        A list if the default values for values. If left as None, all
        default values are assumed to be None.

    error:
        Whether to throw an error if the check does not pass. If True, only
        an error is thrown and no warning can be printed.

    warn:
        Whether to print a warning from warnings.warn if the check does
        not pass. If error is also True, only error is thrown and
        no warning can be printed.

    Returns
    -------
        None
    """
    # Init
    if default_values is None:
        default_values = [None] * len(values)

    # Check if any values are set
    unset_count = 0
    for val, default_val in zip(values, default_values):
        if val == default_val:
            unset_count += 1

    # If all, or None, of the values are set we're good
    if unset_count == 0 or unset_count == len(values):
        return

    # Let the user know if any were set
    if error:
        raise ValueError(msg)

    if warn:
        warnings.warn(msg)
        return

    # If we're still here, just print the message
    if msg is not None:
        print(msg)
