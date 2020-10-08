# -*- coding: utf-8 -*-
"""
Created on: Tues September 29 09:06:31 2020
Updated on:

Original author: Ben Taylor
Last update made by:
Other updates made by:

File purpose:
Wrappers around python time libraries to make timing easier
"""

import time
import functools


def current_milli_time():
    return int(time.time() * 1000)


def timer_decorator(func):
    """
    Times and prints out the time taken for the function being decorated
    """
    # Makes sure func metadata is preserved on errors
    @functools.wraps(func)
    def timer_wrapper(*args, **kwargs):
        # Do before func
        start_time = current_milli_time()

        # Do func
        return_val = func(*args, **kwargs)

        # Do after func
        end_time = current_milli_time()
        time_taken = end_time - start_time
        print("%s took:\t %dms" % (func.__name__, time_taken))
        return return_val
    return timer_wrapper
