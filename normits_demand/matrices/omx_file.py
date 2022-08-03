# -*- coding: utf-8 -*-
"""
    Module containing functionality for reading and writing to
    OMX files.
"""

##### IMPORTS #####
# Standard imports
from pathlib import Path
import warnings

# Third party imports
import numpy as np
import tables

# Local imports
from normits_demand import logging as nd_log

##### CONSTANTS #####
LOG = nd_log.get_logger(__name__)

##### CLASSES #####
class OMXFile(tables.File):
    """Reading and writing data to OMX files, which use the HDF5 format.

    This class wraps pytables `File` object and provides some helper
    methods and checks for working with OMX files specifically.

    Parameters
    ----------
    filename : str
        The name of the file (supports environment variable expansion).
        It is suggested that file names have any of the .h5, .hdf or
        .hdf5 extensions, although this is not mandatory.
    mode : str, default 'r'
        The mode to open the file. It can be one of the
        following:
            * *'r'*: Read-only; no data can be modified.
            * *'w'*: Write; a new file is created (an existing file
            with the same name would be deleted).
            * *'a'*: Append; an existing file is opened for reading
            and writing, and if the file does not exist it is created.
            * *'r+'*: It is similar to 'a', but the file must already
            exist.
    omx_version : str, optional
        Version of the OMX file, this option is ignored unless
        mode = 'w', when it is mandatory. Expected OMX version is '0.2'.
    shape : str, optional
        Shape of the matrices in the OMX file, this option is ignored
        unless mode = 'w', when it is mandatory.
    kwargs : Keyword arguments, optional
        All other keyword arguments are passed to `tables.File`.

    Raises
    ------
    ValueError
        If a `mode` other than those defined above is provided or
        `omx_version` and `shape` aren't provided in `mode` 'w'.
    """

    _EXPECTED_OMX_VERSION = "0.2"

    def __init__(
        self,
        filename: Path,
        mode: str = "r",
        omx_version: str = None,
        shape: str = None,
        **kwargs,
    ) -> None:
        self.mode = str(mode).strip().lower()
        super().__init__(filename, mode=self.mode, **kwargs)

        self._path = Path(filename)
        self._zones = None
        self._matrix_levels = None
        self._omx_version = None
        self._shape = None

        if self.mode in ("r", "a", "r+"):
            self._omx_version = self._check_omx_version(
                self.root._v_attrs["OMX_VERSION"].decode()
            )
            self._shape = self._check_shape(self.root._v_attrs["SHAPE"])
            self._zones = self._check_zones(self.get_node("/lookup", "ZoneNames"))
            self.get_node("/data")
        elif self.mode == "w":
            if omx_version is None or shape is None:
                raise ValueError(
                    "omx_version and shape keyword arguments "
                    f"should be provided if mode is '{mode}'"
                )
            self.omx_version = omx_version
            self.shape = shape
        else:
            raise ValueError(f"unknown mode '{mode}' should be one of 'r', 'a', 'r+' or 'w'")

    def _can_write(self, name: str) -> None:
        """Raises ValueError if not in a writing mode."""
        if self.mode == "r":
            raise ValueError(f"cannot set {name} in mode = {self.mode}")

    def _check_omx_version(self, value: str) -> str:
        """Warns user if OMX version isn't the expected one."""
        value = str(value).strip()
        if value != self._EXPECTED_OMX_VERSION:
            warnings.warn(
                f"OMXFile expects OMX version {self._EXPECTED_OMX_VERSION} "
                f"but got {value}, which may be incompatible"
            )
        return value

    @staticmethod
    def _check_shape(value: tuple[int, int]) -> tuple[int, int]:
        """Raises ValueError if shape isn't valid."""
        value = tuple(value)
        if len(value) != 2:
            raise ValueError(f"shape should be a tuple of lenght 2 not length {len(value)}")
        if value[0] != value[1]:
            raise ValueError(
                f"matrix should have the same number of rows and columns not {value}"
            )
        return value

    def _check_zones(self, value: np.ndarray) -> np.ndarray:
        """Raises ValueError if zones aren't the correct shape."""
        value = np.array(value)
        if value.shape != (self.shape[0],):
            raise ValueError(
                f"zones should be a 1D array with length {self.shape[0]} not {value.shape}"
            )
        return value

    @property
    def omx_version(self) -> str:
        """OMX version of the current file."""
        if self._omx_version is None:
            raise ValueError("unknown OMX version")
        return self._omx_version

    @omx_version.setter
    def omx_version(self, value: str) -> None:
        self._can_write("omx_version")
        value = self._check_omx_version(value)
        if value != self._omx_version:
            self._omx_version = value
            self.root._v_attrs["OMX_VERSION"] = self._omx_version

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the matrices in the current OMX file."""
        if self._shape is None:
            raise ValueError("unknown matrix shape")
        return self._shape

    @shape.setter
    def shape(self, value: tuple[int, int]) -> None:
        self._can_write("shape")
        value = self._check_shape(value)
        if value != self._shape:
            self._shape = value
            self.root._v_attrs["SHAPE"] = self._shape

    @property
    def zones(self) -> np.ndarray:
        """Array of zone names for the current OMX file."""
        return self._zones

    @zones.setter
    def zones(self, value: np.ndarray) -> None:
        self._can_write("zones")
        value = self._check_zones(value)
        if np.any(value != self._zones):
            self._zones = value
            self.create_array("/data", "ZoneNames", self._zones)

    @property
    def matrix_levels(self) -> tuple[str]:
        """Names of all the matrix levels in the OMX file."""
        return [n.name for n in self.list_nodes("/data", "Array")]

    def get_matrix_level(self, level_name: str) -> np.ndarray:
        """Returns a single matrix level as an array.

        Parameters
        ----------
        level_name : str
            Name of the matrix level to return.

        Returns
        -------
        np.ndarray
            2D square matrix for a single level.
        """
        return self.get_node("/data", level_name).read()

    def set_matrix_level(self, level_name: str, matrix: np.ndarray) -> None:
        """Sets matrix level in OMX file to given array.

        Parameters
        ----------
        level_name : str
            Name of matrix level to set.
        matrix : np.ndarray
            Square array of matrix values.

        Raises
        ------
        ValueError
            If `matrix.shape` isn't equal to `self.shape`.
        """
        self._can_write("matrix level")
        if matrix.shape != self.shape:
            raise ValueError(f"matrix shape should be {self.shape} no {matrix.shape}")
        self.create_array("/data", str(level_name), matrix)
