# -*- coding: utf-8 -*-
"""
Created on: 22/02/2022
Updated on:

Original author: Ben Taylor
Last update made by:
Other updates made by:

File purpose:

"""
from __future__ import annotations

# Built-Ins
import threading
from multiprocessing import shared_memory

from typing import Callable

# Third Party
import numpy as np

# Local Imports


class SharedNumpyArrayHelper:
    """Shared numpy array to allow threads/processes to communicate"""

    def __init__(
        self,
        name: str,
        data: np.ndarray,
        create: bool = True,
        lock: threading.Lock = None,
        dtype: np.dtype = None,
    ):
        # Set defaults
        if lock is None:
            lock = threading.Lock()

        if dtype is not None:
            data = data.astype(dtype)

        # Assign attributes
        self._name = name
        self._lock = lock

        # Store the info we need about the data
        self._dtype = data.dtype
        self._nbytes = data.nbytes
        self._shape = data.shape

        self._shm = shared_memory.SharedMemory(
            create=create,
            name=self._name,
            size=self._nbytes,
        )

        self._create_shared_array(data)

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

    @property
    def shared_memory(self):
        return self._shm

    def __enter__(self) -> SharedNumpyArrayHelper:
        return self

    def __exit__(self, *args, **kwargs) -> None:
        self.unlink_shared_array()

    def _create_shared_array(self, data: np.ndarray) -> np.ndarray:
        shared_array = np.ndarray(self._shape, dtype=self._dtype, buffer=self._shm.buf)
        shared_array[:] = data[:]
        return shared_array

    def get_shared_array(self) -> np.ndarray:
        return np.ndarray(self._shape, dtype=self._dtype, buffer=self._shm.buf)

    def get_local_copy(self) -> np.ndarray:
        shared_array = self.get_shared_array()
        local_copy = np.empty(shape=self._shape, dtype=self._dtype)
        local_copy[:] = shared_array[:]
        return local_copy

    def write_local_data(self, data: np.ndarray) -> None:
        with self._lock:
            shared_array = self.get_shared_array()
            shared_array[:] = data[:]

    def apply_local_data(self, data: np.ndarray, operation: Callable) -> None:
        with self._lock:
            shared_array = self.get_shared_array()
            temp = operation(shared_array, data)
            shared_array[:] = temp[:]

    def close_shared_array(self) -> None:
        self._shm.close()

    def unlink_shared_array(self) -> None:
        self._shm.close()
        self._shm.unlink()

    def reset_zeros(self) -> np.ndarray:
        return self.reset_value(0)

    def reset_ones(self) -> np.ndarray:
        return self.reset_value(1)

    def reset_value(self, fill_value) -> None:
        with self._lock:
            shared_array = self.get_shared_array()
            shared_array[:] = np.full_like(shared_array, fill_value=fill_value)
