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

from math import floor
from datetime import datetime


def current_milli_time():
    return int(time.time() * 1000)


def get_datetime(time_format: str = "%d-%m-%Y  %H:%M:%S.%f",
                 precision: str = None
                 ) -> str:
    # Just return if ignoring precision
    if precision is None:
        return datetime.now().strftime(time_format)

    # Init
    valid_precision = ['millisecond', 'microsecond']
    precision = precision.strip().lower()

    # Validate
    if precision not in valid_precision:
        raise ValueError("%s is not a valid precision. Can only accept: %s"
                         % (precision, str(valid_precision)))

    # Get the correct format - assumes time ends in microseconds
    if precision == 'microsecond':
        return datetime.now().strftime(time_format)

    if precision == 'millisecond':
        return datetime.now().strftime(time_format)[:-3]

    raise ValueError("%s seems to be a valid precision - but I don't know "
                     "how to format it. There must be a missing if statement!")


def time_taken(start_time: int,
               end_time: int,
               ) -> str:
    """
    Formats the time taken into hours, minutes and seconds

    Parameters
    ----------
    start_time:
        The start time, in milliseconds. Can be gotten from current_milli_time()

    end_time:
        The end time, in milliseconds. Can be gotten from current_milli_time()

    Returns
    -------
    time_taken:
        The time passed between start and end time in format:
        xxhrs xxm xx.xxs. Where x is replaced with actual values

    Raises
    ------
    ValueError:
        If end_time - start_time is less than, or equal to, 0
    """
    # Init
    elapsed_time = end_time - start_time
    elapsed_secs = elapsed_time / 1000

    # Validate
    if elapsed_time <= 0:
        raise ValueError("Elapsed time is 0, or negative! Was the start_time "
                         "and end_time given the wrong way around?")

    # Split into minutes and seconds
    seconds = elapsed_secs % 60
    minutes = floor(elapsed_secs / 60)

    # If no minutes passed, just return seconds
    if minutes == 0:
        return "%.2fs" % seconds

    # Split into hours and minutes
    res_mins = minutes % 60
    hours = floor(minutes / 60)

    # If no hours passed, return minutes and seconds
    if hours == 0:
        return "%dm %.2fs" % (res_mins, seconds)

    # Otherwise return the full format
    return "%dhrs %dm %.2fs" % (hours, res_mins, seconds)


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
