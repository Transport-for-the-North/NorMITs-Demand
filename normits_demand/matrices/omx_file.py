# -*- coding: utf-8 -*-
"""
    Module containing functionality for reading and writing to
    OMX files.
"""

##### IMPORTS #####
# Standard imports
from pathlib import Path
from typing import Optional
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
    shape : tuple[int, int], optional
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

    _allowed_omx_versions = ("0.2", "0.3")
    _zone_data_paths = {
        "0.2": ("/lookup", "ZoneNames"),
        "0.3": ("/zonalReferences", "zone_labels"),
    }
    _matrix_folder = {"0.2": "/data", "0.3": "/matrices"}
    _shape_attribute = {"0.2": "SHAPE", "0.3": "OMX_ZONES"}

    def __init__(
        self,
        filename: Path,
        mode: str = "r",
        omx_version: Optional[str] = None,
        shape: Optional[tuple[int, int]] = None,
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
            self._shape = self._check_shape(self._get_shape())
            self._zones = self._get_zones()

            nlevels = len(self.matrix_levels)
            if nlevels == 0:
                warnings.warn(f"No matrix levels found in file: {self._path.name}")
            else:
                LOG.debug("Found %s levels in matrix file: %s", nlevels, self._path.name)

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
        if value not in self._allowed_omx_versions:
            raise NotImplementedError(
                f"Unexpected OMX version {value} should "
                f"be one of {self._allowed_omx_versions}"
            )

        if value not in self._zone_data_paths:
            raise NotImplementedError(f"unknown zone location for OMX version {value}")
        if value not in self._matrix_folder:
            raise ValueError(f"unknown matrix location for OMX version {value}")
        if value not in self._shape_attribute:
            raise ValueError(f"unknown shape attribute for OMX version {value}")

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
            self.root._v_attrs["OMX_VERSION"] = self._omx_version.encode()

    def _get_shape(self) -> tuple[int, int]:
        """Get the shape of the matrices."""
        value = self.root._v_attrs[self._shape_attribute[self.omx_version]]
        if self.omx_version == "0.3":
            # Version 0.3 has a single value for number of zones
            shape = (value, value)
        else:
            shape = tuple(value)
        return shape

    @staticmethod
    def _check_shape(value: tuple[int, int]) -> tuple[int, int]:
        """Raises ValueError if shape isn't valid."""
        value = tuple(value)
        if len(value) != 2:
            raise ValueError(f"shape should be a tuple of length 2 not length {len(value)}")

        if value[0] != value[1]:
            raise ValueError(
                f"matrix should have the same number of rows and columns not {value}"
            )

        return value

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
            attr = self._shape_attribute[self.omx_version]

            if self.omx_version == "0.3":
                # Version 0.3 has a single value for number of zones
                self.root._v_attrs[attr] = self._shape[0]
            else:
                self.root._v_attrs[attr] = np.array(self._shape)

    def _check_zones(self, value: np.ndarray) -> np.ndarray:
        """Raises ValueError if zones aren't the correct shape."""
        value = np.array(value)
        if value.shape != (self.shape[0],):
            raise ValueError(
                f"zones should be a 1D array with length {self.shape[0]} not {value.shape}"
            )
        return value

    def _get_zones(self) -> np.ndarray:
        """Attempt to read ZoneNames from file, otherwise uses sequential zones from 1."""
        path, node = self._zone_data_paths[self.omx_version]

        try:
            zones = self.get_node(path, node)
        except tables.NoSuchNodeError:
            zones = np.arange(1, self.shape[0] + 1)

        return self._check_zones(zones)

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
            path, node = self._zone_data_paths[self.omx_version]
            self.create_array(path, node, self._zones, createparents=True)

    @property
    def matrix_levels(self) -> tuple[str]:
        """Names of all the matrix levels in the OMX file."""
        path = self._matrix_folder[self.omx_version]
        return [n.name for n in self.list_nodes(path, "Array")]

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
        path = self._matrix_folder[self.omx_version]
        return self.get_node(path, level_name).read()

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
            raise ValueError(f"matrix shape should be {self.shape} not {matrix.shape}")

        path = self._matrix_folder[self.omx_version]
        self.create_array(path, str(level_name), matrix, createparents=True)
