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

from normits_demand import constants as consts


def validate_vector_type(vector_type: str) -> str:
    """
    Tidies up seg_level and raises an exception if not a valid name

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
