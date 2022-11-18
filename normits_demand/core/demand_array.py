# -*- coding: utf-8 -*-
"""
    Module containing functionality for the demand array base class.
"""

##### IMPORTS #####
# Standard imports
from __future__ import annotations
import abc
import logging
import pathlib
from typing import Union, Optional

# Third party imports
import numpy as np

# Local imports
import normits_demand as nd
from normits_demand import constants
from normits_demand.core import data_structures, segments, zoning


##### CONSTANTS #####
LOG = logging.getLogger(__name__)

##### CLASSES #####
DArrayData = Union[dict[str, np.ndarray], pathlib.Path]
"""Data for a demand array can either be in-memory dictionary of
arrays, or a HDF5 file containing the arrays.
"""


class DArrayError(nd.NormitsDemandError):
    """Exception for any demand array related errors."""


class DArray:

    __version__ = nd.__version__

    def __init__(
        self,
        segmentation: segments.SegmentationLevel,
        data: DArrayData,
        array_dimensions: int,
        zoning_system: Optional[zoning.ZoningSystem] = None,
        time_format: Optional[data_structures.TimeFormat] = None,
        process_count: int = constants.PROCESS_COUNT,
    ) -> None:
        if zoning_system is not None:
            if not isinstance(zoning_system, zoning.ZoningSystem):
                raise DArrayError(
                    "Given zoning_system is not a nd.core.ZoningSystem object."
                    f"Got a {type(zoning_system)} object instead."
                )

        if not isinstance(segmentation, segments.SegmentationLevel):
            raise DArrayError(
                "Given segmentation is not a nd.core.SegmentationLevel object."
                f"Got a {type(segmentation)} object instead."
            )

        self._zoning_system = zoning_system
        self._segmentation = segmentation
        self._time_format = self._validate_time_format(time_format)
        self.process_count = process_count

        if array_dimensions < 1:
            raise ValueError(f"array dimensions should be >= 1 not {array_dimensions}")
        self._array_dimensions = array_dimensions

        self._init_data(data)
        self._validate_data_arrays()

    def _init_data(self, data: DArrayData) -> None:
        if isinstance(data, dict):
            self._data_path = None
            self._data = data

        elif isinstance(data, (pathlib.Path, str)):
            self._data_path = pathlib.Path(data)
            # TODO(MB) Implement `_init_data` for HDF5 files
            raise NotImplementedError(
                "Functionality to handle DArray HDF5 files is not yet implemented"
            )

        else:
            raise DArrayError(f"DArray data should be {DArrayData} not {type(self._data)}")

    def _validate_array(self, segment_name: str, array: np.ndarray) -> None:
        if not isinstance(array, np.ndarray):
            raise TypeError(f"segment {segment_name} should be np.ndarray not {type(array)}")

        if self.zoning_system is None:
            expected_shape = (1,) * self.array_dimensions
        else:
            expected_shape = (len(self.zoning_system),) * self.array_dimensions

        if array.shape != expected_shape:
            raise DArrayError(
                f"segment {segment_name} array should have "
                f"shape {expected_shape} not {array.shape}"
            )

    def get_array(self, segment_name: str) -> np.ndarray:
        if isinstance(self._data, dict):
            if segment_name not in self._data:
                raise DArrayError(f"segment {segment_name} missing from data")

            return self._data.get(segment_name)

        # TODO(MB) Implement `get_array` for HDF5 files
        raise TypeError(f"not implemented get_array for {type(self._data)}")

    def set_array(self, segment_name: str, array: np.ndarray) -> None:
        self._validate_array(segment_name, array)

        if isinstance(self._data, dict):
            if segment_name not in self.segmentation.segment_names:
                raise DArrayError(
                    f"segment {segment_name} doesn't exist "
                    f"in segmentation {self.segmentation.name}"
                )

            self._data[segment_name] = array

        # TODO(MB) Implement `set_array` for HDF5 files
        else:
            raise TypeError(f"not implemented set_array for {type(self._data)}")

    def _validate_data_arrays(self) -> None:
        missing_segments = []
        invalid_types = []
        incorrect_shape = []
        for seg_name in self.segmentation.segment_names:
            try:
                array = self.get_array(seg_name)
            except DArrayError:
                missing_segments.append(seg_name)
                continue

            try:
                self._validate_array(seg_name, array)
            except TypeError:
                invalid_types.append(seg_name)
            except DArrayError:
                incorrect_shape.append(seg_name)

        messages = []
        if missing_segments:
            messages.append(f"{len(missing_segments)} segments are missing")
        if invalid_types:
            messages.append(f"{len(invalid_types)} segments have invalid types")
        if incorrect_shape:
            messages.append(f"{len(incorrect_shape)} segments are incorrect shape")
        if messages:
            raise DArrayError(f"Invalid array data: {', '.join(messages)}")

    @property
    def zoning_system(self) -> zoning.ZoningSystem:
        return self._zoning_system

    @zoning_system.setter
    def zoning_system(self, _):
        raise DArrayError(
            "Zoning system cannot be changed for an "
            f"already created {self.__class__.__name__}."
        )

    @property
    def segmentation(self) -> segments.SegmentationLevel:
        return self._segmentation

    @segmentation.setter
    def segmentation(self, _):
        raise DArrayError(
            "Segmentation cannot be changed for an "
            f"already created {self.__class__.__name__}."
        )

    @property
    def array_dimensions(self) -> int:
        return self._array_dimensions

    @array_dimensions.setter
    def array_dimensions(self, _):
        raise DArrayError(
            "Array dimensions cannot be changed for an "
            f"already created {self.__class__.__name__}."
        )

    @property
    def time_format(self) -> data_structures.TimeFormat:
        if self._time_format is None:
            return None
        return self._time_format

    @staticmethod
    def _valid_time_formats() -> list[str]:
        """
        Returns a list of valid strings to pass for time_format
        """
        return [x.value for x in data_structures.TimeFormat]

    def _validate_time_format(
        self,
        time_format: Union[str, data_structures.TimeFormat],
    ) -> data_structures.TimeFormat:
        """Validate the time format is a valid value

        Parameters
        ----------
        time_format:
            The name of the time format name to validate

        Returns
        -------
        time_format: str | data_structures.TimeFormat
            Returns a tidied up version of the passed in time_format.

        Raises
        ------
        ValueError
            If the given time_format is not on of self._valid_time_formats
        """
        # Time period format only matters if it's in the segmentation
        if self.segmentation.has_time_period_segments():
            if time_format is None:
                raise ValueError(
                    "The given segmentation level has time periods in its "
                    "segmentation, but the format of this time period has "
                    "not been defined.\n"
                    "\tTime periods segment name: "
                    f"{self.segmentation._time_period_segment_name}\n"
                    "\tValid time_format values: {self._valid_time_formats()}"
                )

        # If None or TimeFormat, that's fine
        if time_format is None or isinstance(time_format, data_structures.TimeFormat):
            return time_format

        # Check we've got a valid value
        time_format = time_format.strip().lower()
        if time_format not in self._valid_time_formats():
            raise ValueError(
                "The given time_format is not valid.\n"
                f"\tGot: {time_format}\n"
                "\tExpected one of: {self._valid_time_formats()}"
            )

        return data_structures.TimeFormat(time_format)

    def _check_other(self, other: DArray, method: str) -> zoning.ZoningSystem:
        """Check `other` is a `DVector` with the same zoning system."""
        # We can only multiply, or divide, with other DVectors
        if not isinstance(other, self.__class__):
            raise DArrayError(
                f"The {method} operator can only be used with the same "
                f"objects on each side. Got {type(self)} and {type(other)}."
            )

        if self.zoning_system == other.zoning_system:
            return self.zoning_system
        if self.zoning_system is None:
            return other.zoning_system
        if other.zoning_system is None:
            return self.zoning_system
        raise nd.ZoningError(
            f"Cannot {method} two {self.__class__.__name__}s using different "
            f"zoning systems.\n"
            f"zoning system of a: {self.zoning_system}\n"
            f"zoning system of b: {other.zoning_system}\n"
        )


##### FUNCTIONS #####
def test_init():
    segmentation = nd.get_segmentation_level("hb_p_m")
    zone_system = nd.get_zoning_system("lad_2020")

    rng = np.random.default_rng()
    for dim in range(4):
        data = {s: rng.random((len(zone_system),) * dim) for s in segmentation.segment_names}
        print(list(data.values())[0].shape)
        try:
            darr = DArray(segmentation, data, dim, zoning_system=zone_system)
            print(f"{dim} {darr}")
        except Exception as e:
            print(f"{dim} {e.__class__.__name__}: {e}")


if __name__ == "__main__":
    test_init()
