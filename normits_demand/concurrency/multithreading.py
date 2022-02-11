# -*- coding: utf-8 -*-
"""
Created on: 03/02/2022
Updated on:

Original author: Ben Taylor
Last update made by:
Other updates made by:

File purpose:
Wrapper around pythons built in libraries to use multiple Threads in
NorMITs Demand.
Attempts to wrap threads so they will be killed when the program exits

Uses `threading` tools instead of `multiprocessing.ThreadPool` as ThreadPool
code is unfinished and therefore less stable
See: https://stackoverflow.com/a/46049195/4323989
This does mean going round the houses a little bit, but it's worth it for the
sake of more stable code.
"""
# Built-Ins
import time
import queue
import threading

from typing import Any
from typing import List
from typing import Dict

# Third Party
import tqdm

# Local Imports
from normits_demand.utils import general as du


class MultithreadingError(Exception):
    """
    Custom Error Wrapper to throw in this module
    """
    def __init__(self, message):
        super().__init__(message)


class ReturnOrErrorThread(threading.Thread):
    """
    Wrapper around threading.Thread to handle errors and return values

    Adds error_event, error_q, and return_val properties to threading.Thread.
    If an error occurs during the call of `run()`, that error will be
    caught and placed onto error_q. error_event will also be set.
    If no error occurs and the thread return successfully, the return item
    of self.run_target will be placed into self.return_val
    """
    def __init__(
            self,
            error_event: threading.Event = None,
            error_q: queue.Queue = None,
            daemon: bool = True,
            *args,
            **kwargs,
    ):
        threading.Thread.__init__(self, daemon=daemon, *args, **kwargs)

        if error_event is None:
            error_event = threading.Event()

        if error_q is None:
            error_q = queue.Queue(1)

        self._error_event = error_event
        self._error_q = error_q
        self._return_val = None

    @property
    def error_event(self):
        return self._error_event

    @property
    def error_q(self):
        return self._error_q

    @property
    def return_val(self):
        return self._return_val

    def run(self) -> None:
        """Deals with catching errors and returns. Calls `self.run_target()`

        This method should NOT be overwritten if this class is being used.
        Instead, overwrite `self.run_target()`, which will be used to control
        the thread's activity.
        """
        try:
            self._return_val = self.run_target()
        except BaseException as e:
            # Mark that there has been an error, and re-raise
            self._error_event.set()
            self._error_q.put(e)
            
    def run_target(self) -> Any:
        """Method representing the thread's activity.

        You may override this method in a subclass. The standard run() method
        invokes the callable object passed to the object's constructor as the
        target argument, if any, with sequential and keyword arguments taken
        from the args and kwargs arguments, respectively.
        """
        return threading.Thread.run(self)


def wait_for_thread_dict_return_or_error(
        threads: Dict[Any, ReturnOrErrorThread],
        *args,
        **kwargs,
) -> Dict[Any, Any]:
    """Wrapper around to allow a dictionary of threads instead.

    See `wait_for_thread_return_or_error()` for full documentation.

    Parameters
    ----------
    threads:
        A dictionary of the threads to attempt to retrieve the return
        values from.

    See Also
    --------
    `wait_for_thread_return_or_error()`
    """
    # Convert dict to list
    key_order = list(threads.keys())
    thread_list = [threads[x] for x in key_order]

    # Call original function
    results_list = wait_for_thread_return_or_error(
        threads=thread_list,
        *args,
        **kwargs,
    )

    # Convert back to dict
    return_dict = dict.fromkeys(key_order)
    for key, value in zip(key_order, results_list):
        return_dict[key] = value
    return return_dict


def wait_for_thread_return_or_error(
        threads: List[ReturnOrErrorThread],
        stop_event: threading.Event = None,
        thread_timeout: float = 0.01,
        total_timeout: float = 86400,
        pbar_kwargs: Dict[str, Any] = None,
        default_return_val: Any = None,
) -> List[Any]:
    """Retrieves the return values of threads, or throws a thread error

    Repeatedly checks all threads to see if they are done. Once a thread
    is complete, it will no longer be checked. the return_val of all threads
    will be returned. Can be exited early using stop_event.

    Parameters
    ----------
    threads:
        A list of the threads to attempt to retrieve the return values from.

    stop_event:
        An event telling this function to stop waiting for return values.

    thread_timeout:
        How long to wait for each thread in threads to terminate. This is the
        amount of time to wait for a single thread before moving on to the
        next one.

    total_timeout:
        The total amount of time to wait before terminating.

    pbar_kwargs:
        A dictionary of keyword arguments to pass into a progress bar.

    default_return_val:
        The default value to place into the list of returns. This normally
        won't be seen, as it should be overwritten. However, if the stop_event
        is set before all results are collected, the returned list will contain
        this values in place of the uncollected return values.

    Returns
    -------
    A list of return values, collected from threads. The return values will
    be returned in the same order as the order of threads.

    Raises
    ------
    TimeoutError:
        If total_timeout is exceeded when waiting for the threads to return.

    MultithreadingError:
        If an error has occurred in one of the threads.
    """
    # init
    start_time = time.time()
    got_all_results = False
    results = [default_return_val] * len(threads)

    if stop_event is None:
        stop_event = threading.Event()

    # Add index to save original order
    idx_threads = list(enumerate(threads))

    # If not given any kwargs, assume no pbar wanted
    if pbar_kwargs is None:
        pbar_kwargs = {'disable': True}

    # Context is meant to keep the pbar tidy
    with du.std_out_err_redirect_tqdm() as orig_stdout:
        # Additional args for context
        pbar_kwargs['file'] = orig_stdout
        pbar_kwargs['dynamic_ncols'] = True
        # Improves time prediction guessing
        pbar_kwargs['smoothing'] = 0

        # If no total given, we can add one!
        if 'total' not in pbar_kwargs or pbar_kwargs['total'] == 0:
            pbar_kwargs['total'] = len(threads)

        # Finally, make to pbar!
        pbar = tqdm.tqdm(**pbar_kwargs)

        # Grab all the results as they come in
        while not got_all_results:
            # Wait for a little bit to avoid intensive looping
            time.sleep(0.05)

            # If told to stop, just return what we have so far
            if stop_event.is_set():
                return results

            # Check if we've ran out of time
            if (time.time() - start_time) > total_timeout:
                raise TimeoutError("Ran out of time while waiting for results.")

            # Check if any results are ready
            done_threads = list()
            for cur_idx, (orig_idx, thread) in enumerate(idx_threads):
                thread.join(timeout=thread_timeout)
                if thread.is_alive():
                    continue

                # If here, thread must be finished. Check for errors
                if thread.error_event.is_set():
                    msg = "Error occurred in thread: %s" % thread.name
                    raise MultithreadingError(msg) from thread.error_q.get()

                # Finally, get the return value. Mark as done
                results[orig_idx] = thread.return_val
                done_threads.append(cur_idx)

            # Update the progress bar with the number of results we just got
            if len(done_threads) > 0:
                pbar.update(len(done_threads))

            # Remove results we've got
            for i in sorted(done_threads, reverse=True):
                del idx_threads[i]

            # Check if we have all results
            if len(idx_threads) == 0:
                got_all_results = True

    # Tidy up before we leave
    pbar.close()

    return results
