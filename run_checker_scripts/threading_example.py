# -*- coding: utf-8 -*-
"""
Created on: 03/02/2022
Updated on:

Original author: Ben Taylor
Last update made by:
Other updates made by:

File purpose:
Testing on how to implement threading for the gravity model
"""
# Built-Ins
import time
import queue
import threading

from typing import List

# Third Party
import numpy as np

# Local Imports
from normits_demand.cost import cost_functions
from normits_demand.distribution import furness


def loops_furness(
        area_mats,
        seed_mats,
        row_targets,
        col_targets,
):
    unique_area_codes = area_mats.keys()

    print(seed_mats[1])
    
    seed_mat = np.zeros((len(row_targets), len(col_targets)))
    for area_code in unique_area_codes:
        seed_mat += seed_mats[area_code]

    furnessed_mat, _, _ = furness.doubly_constrained_furness(
        seed_vals=seed_mat,
        row_targets=row_targets,
        col_targets=col_targets,
        tol=0.1,
        max_iters=200,
    )

    furnessed_mats = dict.fromkeys(unique_area_codes)
    for area_code in furnessed_mats:
        furnessed_mats[area_code] = furnessed_mat * area_mats[area_code]

    return furnessed_mats


def loops(
        area_matrix,
        cost_matrix,
        row_targets,
        col_targets,
):
    unique_area_codes = np.unique(area_matrix)

    seed_mats = dict.fromkeys(unique_area_codes)
    area_mats = dict.fromkeys(unique_area_codes)
    for area_code in seed_mats:
        area_bool = area_matrix == area_code
        area_data = cost_matrix * area_bool

        area_mats[area_code] = area_bool
        seed_mats[area_code] = cost_functions.log_normal(area_data, sigma=1, mu=2)

    furnessed_mats = loops_furness(area_mats, seed_mats, row_targets, col_targets)

    for i in unique_area_codes[:1]:
        print(seed_mats[i])
        print(furnessed_mats[i])
        print()


class FurnessThread(threading.Thread):

    def __init__(
            self,
            row_targets,
            col_targets,
            getter_qs,
            putter_qs,
            area_mats,
            *args,
            **kwargs,
    ):
        threading.Thread.__init__(self, *args, **kwargs)

        self.row_targets = row_targets
        self.col_targets = col_targets
        self.getter_qs = getter_qs
        self.putter_qs = putter_qs
        self.area_mats = area_mats

        self.unique_area_keys = area_mats.keys()

        self._stop_running = False

    def _get_gravity_queue_data(self, waiting_threads: List[int]):
        # init
        queue_data = dict.fromkeys(self.unique_area_keys)

        while len(waiting_threads) > 0:

            # Try and get the data
            done_threads = list()
            for area_id in waiting_threads:
                try:
                    data = self.getter_qs[area_id].get(block=False)
                    queue_data[area_id] = data
                    done_threads.append(area_id)
                except queue.Empty:
                    pass

            # Remove ids for data we have
            for item in done_threads:
                waiting_threads.remove(item)

            # Wait for a bit so we don't hammer CPU
            time.sleep(1)

        return queue_data

    def run(self):
        # Run until told not to
        while not self._stop_running:

            # Wait for threads to hand over seed mats
            waiting_threads = list(self.unique_area_keys)
            seed_mats = self._get_gravity_queue_data(waiting_threads)

            break

            # Furness, then split back out
            furnessed_mats = loops_furness(
                area_mats=self.area_mats,
                seed_mats=seed_mats,
                row_targets=self.row_targets,
                col_targets=self.col_targets,
            )

            for area_id in self.unique_area_keys:
                self.putter_qs[area_id].put(furnessed_mats[area_id])

            print("Furness done one iter")

        print("here")
        exit()

    def kill(self):
        print("Killing")
        self._stop_running = True


class GravityThread(threading.Thread):

    def __init__(
        self,
        cost_matrix,
        mu,
        sigma,
        putter_q,
        getter_q,
        *args,
        **kwargs,
    ):
        threading.Thread.__init__(self, *args, **kwargs)

        self.cost_matrix = cost_matrix
        self.mu = mu
        self.sigma = sigma
        self.putter_q = putter_q
        self.getter_q = getter_q

        self._return_val = None

    def run(self):
        seed_matrix = cost_functions.log_normal(
            self.cost_matrix,
            sigma=self.sigma,
            mu=self.mu,
        )

        # Send data to furness thread, get data back
        self.putter_q.put(seed_matrix)
        furnessed_matrix = get_data_from_queue(self.getter_q)

        self._return_val = furnessed_matrix

    def join(self, *args, **kwargs):
        threading.Thread.join(self, *args, **kwargs)
        return self._return_val


def get_data_from_queue(q):
    got_data = False
    while not got_data:
        try:
            data = q.get(block=False)
            got_data = True
        except queue.Empty:
            pass

        time.sleep(1)

    return data


def fake_scipy(
        area_matrix,
        cost_matrix,
        row_targets,
        col_targets,
):
    # Init
    unique_area_codes = np.unique(area_matrix)

    # Initialise area stuff
    area_mats = dict.fromkeys(unique_area_codes)
    gravity_putter_qs = dict.fromkeys(unique_area_codes)
    gravity_getter_qs = dict.fromkeys(unique_area_codes)
    for area_code in unique_area_codes:
        area_mats[area_code] = area_matrix == area_code
        gravity_putter_qs[area_code] = queue.Queue(1)
        gravity_getter_qs[area_code] = queue.Queue(1)

    # Initialise the central furness thread
    furness_thread = FurnessThread(
        daemon=True,
        row_targets=row_targets,
        col_targets=col_targets,
        getter_qs=gravity_putter_qs,
        putter_qs=gravity_getter_qs,
        area_mats=area_mats,
    )
    furness_thread.start()

    # Start the gravity processes
    seed_mats = dict.fromkeys(unique_area_codes)
    threads = dict.fromkeys(unique_area_codes)
    for area_code in unique_area_codes:
        area_cost = cost_matrix * area_mats[area_code]
        seed_mats[area_code] = cost_functions.log_normal(area_cost, sigma=1, mu=2)

        # Need the thread these too!!!
        threads[area_code] = GravityThread(
            cost_matrix=area_cost,
            sigma=1,
            mu=2,
            putter_q=gravity_putter_qs[area_code],
            getter_q=gravity_getter_qs[area_code],
        )
        threads[area_code].start()

    furnessed_mats = dict.fromkeys(unique_area_codes)
    for area_code in unique_area_codes:
        furnessed_mats[area_code] = threads[area_code].join()

    for i in unique_area_codes[:1]:
        print(seed_mats[i])
        print(furnessed_mats[i])
        print()


def main():
    # Initialise
    area_matrix = np.array([
      [1, 1, 2, 2],
      [1, 1, 2, 2],
      [3, 3, 4, 4],
      [3, 3, 4, 4],
    ])
    
    cost_matrix = np.random.randint(10, size=area_matrix.shape)
    row_targets = np.random.randint(100, 200, size=area_matrix.shape[0])
    col_targets = np.random.randint(100, 200, size=area_matrix.shape[1])

    loops(
        area_matrix=area_matrix,
        cost_matrix=cost_matrix,
        row_targets=row_targets,
        col_targets=col_targets,
    )

    fake_scipy(
        area_matrix=area_matrix,
        cost_matrix=cost_matrix,
        row_targets=row_targets,
        col_targets=col_targets,
    )





if __name__ == '__main__':
    main()
