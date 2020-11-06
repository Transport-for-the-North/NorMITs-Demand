# -*- coding: utf-8 -*-
"""
Created on: Tues September 29 16:22:45 2020
Updated on: Fri November 6 11:27:43 2020

Original author: Ben Taylor
Last update made by:
Other updates made by:

File purpose:
Wrapper around pythons built in libraries to use concurrency in NorMITs Demand
TODO: After integrations with TMS, combine with
  old_tms.multiprocessing_wrapper.py
"""
import os
import warnings

from typing import Any
from typing import List
from typing import Callable

from old_tms.multiprocessing_wrapper import *


def multiprocess(fn: Callable,
                 args: List[Any] = None,
                 kwargs: List[Any] = None,
                 process_count: int = os.cpu_count()-1,
                 pool_maxtasksperchild: int = 4,
                 in_order: bool = False,
                 result_timeout: int = 86400
                 ) -> Any:
    """
    Runs the given function with the arguments given in a multiprocessing.Pool,
    returning the function output.

    Deals with various process_count values:
        - If negative, `os.cpu_count() - process_count` processes will be used
        - If 0, no multiprocessing will be used. THe code will be ran in a
          a single process loop.
        - If positive, process_count processes will be used. If process_count
          is greater than `os.cpu_count() - 1`, a warning will be raised.

    Parameters
    ----------
    fn:
        The name of the function to call.

    args:
        A list of iterables e.g. tuples/lists. len(args) matches the number of
        times fn should be called. Each tuple contains a full set of non-
        keyword arguments to be passed to a single call of fn.
        Defaults to None.

    kwargs:
        A list of dictionaries. The keys are the keyword argument names, and
        the values are the keyword argument values. len(kwargs) matches the
        number of times fn should be called, and should directly correspond to
        args. Each dictionary contains a full set of keyword arguments to be
        passed to a single call of fn.
        Defaults to None.

    process_count:
        The number of processes to create in the Pool. Typically this
        should not exceed the number of cores available.
        Defaults to os.cpu_count().

    pool_maxtasksperchild:
        Passed into the created Pool as maxtaskperchild=pool_maxtaskperchild.
        It is the number of tasks a worker process can complete before it will
        exit and be replaced with a fresh worker process, to enable unused
        resources to be freed.
        Defaults to 4.

    in_order:
        Boolean. Whether the return values need to be in the same order they
        were given. in_order=True is slightly slower due to sorting the results.
        Defaults to False.

    result_timeout:
        Int. How long to wait for each process before throwing an exception
        because the results have taken too long to return
        Defaults to 86400 seconds, (24 hours).

    Examples
    --------
    The following three function calls:
    >>> a = sorted(range(10))
    >>> b = sorted(range(100))
    >>> c = sorted(range(20), reverse=True)

    Would be called, using this function, like this:
    >>> # Note the use of a tuple to make sure a single argument is still
    >>> # iterable
    >>> a_args = (range(10), )
    >>> b_args = (range(100), )
    >>> c_args = (range(20 ), )
    >>>
    >>> # Need to use an empty dict where arguments are not given
    >>> a_kwargs = dict()
    >>> b_kwargs = dict()
    >>> c_kwargs = {'reverse': True}

    >>> args = [a_args, b_args, c_args]
    >>> kwargs = [a_kwargs, b_kwargs, c_kwargs]
    >>> a, b, c = process_pool_wrapper(sorted, args, kwargs)
    """
    # Validate process_count
    if process_count < -os.cpu_count():
        raise ValueError(
            "Negative process_count given is too small. Cannot run %d less "
            "processes than cpu count as only %d cpu have been found by python."
            % (process_count, os.cpu_count())
        )

    if process_count > os.cpu_count()-1:
        warnings.warn("process_count given is too high! It is higher than the "
                      "cpu count - 1  This might cause your system to run "
                      "really slow!")

    # Determine the number of processes to use
    if process_count < 0:
        if process_count < -os.cpu_count():
            raise ValueError(
                "Negative process_count given is too small. Cannot run %d less "
                "processes than cpu count as only %d cpu can be used."
                % (process_count, os.cpu_count())
            )
        else:
            process_count = os.cpu_count() + process_count

    # If the process count is 0, run as a normal for loop
    if process_count == 0:
        return [fn(*a, **k) for a, k in zip(args, kwargs)]

    # If we get here, the process count must be > 0 and valid
    return process_pool_wrapper(
        fn,
        args=args,
        kwargs=kwargs,
        process_count=process_count,
        pool_maxtasksperchild=pool_maxtasksperchild,
        in_order=in_order,
        result_timeout=result_timeout
    )
