# -*- coding: utf-8 -*-
"""
Created on: 22/02/2022
Updated on:

Original author: Ben Taylor
Last update made by:
Other updates made by:

File purpose:

"""
# Built-Ins
from multiprocessing import shared_memory

# Third Party
import numpy as np

# Local Imports


class SharedNumpyArrayHelper:
    """Shared numpy array to allow threads/processes to communicate"""

    def __init__(self, name: str, data: np.ndarray):
        # Store the name
        self._name = name

        # Store the info we need about the data
        self._dtype = data.dtype
        self._nbytes = data.nbytes
        self._shape = data.shape

    @property
    def name(self):
        return self._name

    @property
    def dtype(self):
        return self._dtype

    @property
    def nbytes(self):
        return self._nbytes

    @property
    def shape(self):
        return self._shape

    def create_shared_array(self, data: np.ndarray):
        shm = shared_memory.SharedMemory(create=True, name=self.name, size=self._nbytes)
        shared_array = np.ndarray(self._shape, dtype=self._dtype, buffer=shm.buf)
        shared_array[:] = data[:]
        return shared_array

    def get_shared_array(self):
        shm = shared_memory.SharedMemory(name=self.name)
        shared_array = np.ndarray(self._shape, dtype=self._dtype, buffer=shm.buf)
        return shared_array


