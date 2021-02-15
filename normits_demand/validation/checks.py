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

# BACKLOG: Re-organise EFS Constants. Make a generic constants module
#  labels: EFS, demand merge
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


def validate_matrix_type(matrix_type: str) -> str:
    """
    Tidies up matrix_type and raises an exception if not valid

    Parameters
    ----------
    matrix_type:
        The name of the matrix type to validate

    Returns
    -------
    matrix_type:
        matrix_type with both strip and lower applied to remove any whitespace
        and make it all lowercase

    Raises
    -------
    ValueError:
        If matrix_type is not a valid name for a type of matrix
    """
    # Init
    matrix_type = matrix_type.strip().lower()

    if matrix_type not in efs_consts.VALID_MATRIX_FORMATS:
        raise ValueError(
            "%s is not a valid vector type. It should be one of: %s"
            % (matrix_type, str(efs_consts.VALID_MATRIX_FORMATS))
        )
    return matrix_type
