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

# Third Party

# Local Imports


def pairwise(iterable):
    # Taken directly from: https://docs.python.org/3/library/itertools.html#itertools.pairwise
    # Will be built in from python 3.10 onwards
    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)
