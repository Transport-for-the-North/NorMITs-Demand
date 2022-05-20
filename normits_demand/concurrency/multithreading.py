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
import copy
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
        *args,
        name: str = None,
        error_event: threading.Event = None,
        error_q: queue.Queue = None,
        daemon: bool = True,
        **kwargs,
    ):
        # Add Thread to name if given
        if name is not None and name.lower()[:6] != "thread":
            name = f"Thread-{name}"

        threading.Thread.__init__(
            self,
            *args,
            name=name,
            daemon=daemon,
            **kwargs,
        )

        if error_event is None:
            error_event = threading.Event()

        if error_q is None:
            error_q = queue.Queue(1)

        self._error_event = error_event
        self._error_q = error_q
        self._return_val = None

    @property
    def error_event(self) -> threading.Event:
        """The Event that is triggered on an error in this thread"""
        return self._error_event

    @property
    def error_q(self) -> queue.Queue:
        """The Queue of errors triggered in this thread"""
        return self._error_q

    @property
    def return_val(self) -> Any:
        """The return value of the `self.run` of this thread"""
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
    return_threads: Dict[Any, ReturnOrErrorThread],
    *args,
    error_threads: Dict[Any, ReturnOrErrorThread] = None,
    error_threads_list: List[ReturnOrErrorThread] = None,
    **kwargs,
) -> Dict[Any, Any]:
    """Wrapper around to allow a dictionary of threads instead.

    See `wait_for_thread_return_or_error()` for full documentation.

    Parameters
    ----------
    return_threads:
        A dictionary of the threads to attempt to retrieve the return
        values from.

    error_threads:
        A dictionary of the threads to check there have been no errors in,
        but no return values need to be retrieved.

    error_threads_list:
        A list of the threads to check there have been no errors in,
        but no return values need to be retrieved.

    See Also
    --------
    `wait_for_thread_return_or_error()`
    """
    # Init
    error_threads = dict() if error_threads is None else error_threads
    error_threads_list = list() if error_threads_list is None else error_threads_list

    # Convert dict to list
    key_order = list(return_threads.keys())
    return_thread_list = [return_threads[x] for x in key_order]

    # Call original function
    results_list = wait_for_thread_return_or_error(
        *args,
        return_threads=return_thread_list,
        error_threads=list(error_threads.values()) + error_threads_list,
        **kwargs,
    )

    # Convert back to dict
    return_dict = dict.fromkeys(key_order)
    for key, value in zip(key_order, results_list):
        return_dict[key] = value
    return return_dict


def wait_for_thread_return_or_error(
    return_threads: List[ReturnOrErrorThread],
    error_threads: List[ReturnOrErrorThread] = None,
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
    return_threads:
        A list of the threads to attempt to retrieve the return values from.

    error_threads:
        A list of the threads to check there have been no errors in, but no
        return values need to be retrieved.

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
    got_all_results = False
    results = [default_return_val] * len(return_threads)
    error_threads = list() if error_threads is None else error_threads
    start_time = time.time()

    if stop_event is None:
        stop_event = threading.Event()

    # Add index to save original order
    idx_threads = list(enumerate(return_threads))

    # If not given any kwargs, assume no pbar wanted
    if pbar_kwargs is None:
        pbar_kwargs = {"disable": True}

    # Context is meant to keep the pbar tidy
    with du.std_out_err_redirect_tqdm() as orig_stdout:
        # Additional args for context
        pbar_kwargs["file"] = orig_stdout
        pbar_kwargs["dynamic_ncols"] = True
        # Improves time prediction guessing
        pbar_kwargs["smoothing"] = 0

        # If no total given, we can add one!
        if "total" not in pbar_kwargs or pbar_kwargs["total"] == 0:
            pbar_kwargs["total"] = len(return_threads)

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

            # Check if there's been any errors in the error_threads
            for thread in error_threads:
                thread.join(timeout=thread_timeout)
                if thread.is_alive():
                    continue

                # If here, thread must be finished. Check for errors
                if thread.error_event.is_set():
                    msg = f"Error occurred in thread: {thread.name}"
                    raise MultithreadingError(msg) from thread.error_q.get()

            # Check if any results are ready
            done_threads = list()
            for cur_idx, (orig_idx, thread) in enumerate(idx_threads):
                thread.join(timeout=thread_timeout)
                if thread.is_alive():
                    continue

                # If here, thread must be finished. Check for errors
                if thread.error_event.is_set():
                    msg = f"Error occurred in thread: {thread.name}"
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


def get_data_from_queue(
    q: queue.Queue,
    stop_event: threading.Event = None,
    total_timeout: float = 86400,
    default_return_val: Any = None,
) -> Any:
    """Retrieves data from the given queue.

    Waits until data can be retrieved from q. If total_timeout is reached, a
    Timeout error is thrown.

    Parameters
    ----------
    q:
        The queue to retrieve data from.

    stop_event:
        An event telling this function to stop waiting for return values.

    total_timeout:
        The total amount of time to wait before terminating.

    default_return_val:
        The default value to return if the stop event is set before the data
        is able to be retrieved from the queue.

    Returns
    -------
    q_data:
        The data retrieved from queue

    Raises
    ------
    TimeoutError:
        If total_timeout is exceeded while waiting to get data from q.
    """
    # Init
    start_time = time.time()
    got_data = False
    data = copy.copy(default_return_val)

    if stop_event is None:
        stop_event = threading.Event()

    # Loop while waiting for data
    while not got_data:
        # Wait for a little bit to avoid intensive looping
        time.sleep(0.01)

        # If told to stop, just return what we have so far
        if stop_event.is_set():
            return data

        # Check if we've ran out of time
        if (time.time() - start_time) > total_timeout:
            raise TimeoutError("Ran out of time while waiting to retrieve data from queue.")

        # Try get the data, catch the error if not available
        try:
            data = q.get_nowait()
            got_data = True
        except queue.Empty:
            pass

    return data


def get_data_from_queue_list(
    q_list: List[queue.Queue],
    stop_event: threading.Event = None,
    total_timeout: float = 86400,
    default_return_val: Any = None,
) -> List[Any]:
    """Retrieves data from the given list of queues.

    Waits until data can be retrieved from each queue. If total_timeout
    is reached, a Timeout error is thrown.

    Parameters
    ----------
    q_list:
        The list of queues to retrieve data from.

    stop_event:
        An event telling this function to stop waiting for return values.

    total_timeout:
        The total amount of time to wait before terminating.

    default_return_val:
        The default value to return if the stop event is set before the data
        is able to be retrieved from the queue.

    Returns
    -------
    q_data:
        A list of the data retrieved from queues. The order of the queues will
        be used to order the return. i.e. q_list[0] returned the data in
        q_data[0].

    Raises
    ------
    TimeoutError:
        If total_timeout is exceeded while waiting to get data from q_list.
    """
    # Init
    got_all_results = False
    results = [default_return_val] * len(q_list)
    start_time = time.time()

    if stop_event is None:
        stop_event = threading.Event()

    # Add index to save original order
    idx_qs = list(enumerate(q_list))

    # Now try get all the results
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
        for cur_idx, (orig_idx, q) in enumerate(idx_qs):
            # Try get the data, catch the error if not available
            try:
                results[orig_idx] = q.get_nowait()
                done_threads.append(cur_idx)
            except queue.Empty:
                pass

        # Remove results we've got
        for i in sorted(done_threads, reverse=True):
            del idx_qs[i]

        # Check if we have all results
        if len(idx_qs) == 0:
            got_all_results = True

    return results


def get_data_from_queue_dict(
    q_dict: Dict[Any, queue.Queue],
    stop_event: threading.Event = None,
    total_timeout: float = 86400,
    default_return_val: Any = None,
) -> Dict[Any, Any]:
    """Retrieves data from the given dictionary of queues.

    Waits until data can be retrieved from each queue. If total_timeout
    is reached, a Timeout error is thrown.

    Parameters
    ----------
    q_dict:
        The dictionary of queues to retrieve data from. It doesn't matter
        what the keys are. Keys will be used in the return to define
        which data came from which queue.

    stop_event:
        An event telling this function to stop waiting for return values.

    total_timeout:
        The total amount of time to wait before terminating.

    default_return_val:
        The default value to return if the stop event is set before the data
        is able to be retrieved from the queue.

    Returns
    -------
    q_data:
        A dictionary of the data retrieved from queue

    Raises
    ------
    TimeoutError:
        If total_timeout is exceeded while waiting to get data from q.
    """
    # Convert dict to list
    key_order = list(q_dict.keys())
    q_list = [q_dict[x] for x in key_order]

    # Call list based function
    results_list = get_data_from_queue_list(
        q_list=q_list,
        stop_event=stop_event,
        total_timeout=total_timeout,
        default_return_val=default_return_val,
    )

    # Convert back to dict
    return_dict = dict.fromkeys(q_dict.keys())
    for key, value in zip(key_order, results_list):
        return_dict[key] = value
    return return_dict


def empty_queue(
    q: queue.Queue,
    wait_for_items: bool = False,
    wait_time: float = 0.1,
) -> List[Any]:
    """Empties q by getting and discarding all items

    Parameters
    ----------
    q:
        The Queue to empty out.

    wait_for_items:
        Whether to wait for new items to be added once the queue has been
        emptied. Useful when a small queue is used and threads are waiting to
        put items on the queue as soon as there is space.

    wait_time:
        Only used when wait_for_items is True. The amount of time to wait
        while items are added to the queue before trying to empty again.
        This value will be passed to time.sleep()

    Returns
    -------
    discarded_items:
        A list of all items that were collected from the queue.
    """
    # init
    items = list()

    # Empty queue and wait until queue is empty
    while True:

        # If queue is still empty, we are done
        if q.empty():
            break

        # Try and get all data from the queue
        while True:
            # Try get the data
            try:
                items.append(q.get_nowait())
            except queue.Empty:
                break

        # Just leave if we're not waiting
        if not wait_for_items:
            break

        # Otherwise, wait and loop again
        time.sleep(wait_time)

    return items
