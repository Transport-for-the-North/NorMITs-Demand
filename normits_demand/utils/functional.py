# -*- coding: utf-8 -*-
"""
Created on: 22/06/2022
Updated on:

Original author: Ben Taylor
Last update made by:
Other updates made by:

File purpose:

"""
# Built-Ins
import itertools

from typing import Any

# Third Party

# Local Imports


def pairwise(iterable):
    # Taken directly from: https://docs.python.org/3/library/itertools.html#itertools.pairwise
    # Will be built in from python 3.10 onwards
    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def list_safe_remove(lst: list[Any],
                     remove: list[Any],
                     raise_error: bool = False,
                     inplace: bool = False
                     ) -> list[Any]:
    """
    Removes remove items from lst without raising an error

    Parameters
    ----------
    lst:
        The list to remove items from

    remove:
        The items to remove from lst

    raise_error:
        Whether to raise and error or not when an item is not contained in
        lst

    inplace:
        Whether to remove the items in-place, or return a copy of lst

    Returns
    -------
    lst:
        lst with removed items removed from it
    """
    # Init
    if not inplace:
        lst = lst.copy()

    for item in remove:
        try:
            lst.remove(item)
        except ValueError as e:
            if raise_error:
                raise e

    return lst
