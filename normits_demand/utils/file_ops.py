# -*- coding: utf-8 -*-
"""
Created on: Thur February 11 15:59:21 2020
Updated on:

Original author: Ben Taylor
Last update made by:
Other updates made by:

File purpose:
A collections of utility functions for file operations
"""
# builtins
import os
import pathlib

# Third Party
import pandas as pd

# Local imports
import normits_demand as nd
from normits_demand import constants as consts
from normits_demand.utils import compress

# Imports that need moving into here
from normits_demand.utils.utils import create_folder
from normits_demand.utils.general import list_files


def file_exists(file_path: nd.PathLike) -> bool:
    """
    Checks if a file exists at the given path.

    Parameters
    ----------
    file_path:
        path to the file to check.

    Returns
    -------
    file_exists:
        True if a file exists, else False
    """
    if not os.path.exists(file_path):
        return False

    if not os.path.isfile(file_path):
        raise IOError(
            "The given path exists, but does not point to a file. "
            "Given path: %s" % str(file_path)
        )

    return True


def check_file_exists(file_path: nd.PathLike) -> None:
    """
    Checks if a file exists at the given path. Throws an error if not.

    Parameters
    ----------
    file_path:
        path to the file to check.

    Returns
    -------
    None
    """
    if not file_exists(file_path):
        raise IOError(
            "Cannot find a path to: %s" % str(file_path)
        )


def is_csv(file_path: nd.PathLike) -> bool:
    """
    Returns True if given file path points to a csv, else False

    Parameters
    ----------
    file_path:
        path to the file to check

    Returns
    -------
    boolean:
        True if given file path points to a csv, else False
    """
    # Try to extract the filename extension
    filename_parts = str(file_path).split('.')

    # File doesn't seem to have an file_extension?
    if len(filename_parts) < 1:
        return False

    file_extension = filename_parts[-1].lower()
    return file_extension == 'csv'


def maybe_add_suffix(path: nd.PathLike,
                     suffix: str,
                     overwrite: bool = False,
                     ) -> pathlib.Path:
    """
    Adds suffix to path if no suffix already exists.

    Will overwrite any existing suffix if overwrite is set to True

    Parameters
    ----------
    path:
        The path to maybe add the suffix to.

    suffix:
        The suffix to add onto path.

    overwrite:
        If set to True, will overwrite any suffix that already exists in path.

    Returns
    -------
    path:
        The original passed in path with an updated suffix.
    """
    # Init
    if not isinstance(path, pathlib.Path):
        path = pathlib.Path(path)

    # Remove current suffix if we're overwriting
    if overwrite:
        path = path.parent / path.stem

    # Add suffix if not there
    if path.suffix == '':
        path = path.parent / (path.name + suffix)

    return path


def read_df(path: nd.PathLike, index_col=None, **kwargs) -> pd.DataFrame:
    """
    Reads in the dataframe at path. Decompresses the df if needed.
    
    Parameters
    ----------
    path:
        The full path to the dataframe to read in

    index_col:
        Will set this column as the index if reading from a compressed
        file, and the index is not already set.
        If reading from a csv, this is passed straight to pd.read_csv()

    Returns
    -------
    df:
        The read in df at path.
    """
    # Determine how to read in df
    if pathlib.Path(path).suffix == consts.COMPRESSION_SUFFIX:
        df = compress.read_in(path)

        # Optionally try and set the index
        if index_col is not None and df.index.name is None:
            df = df.set_index(list(df)[index_col])
        return df

    elif pathlib.Path(path).suffix == '.csv':
        return pd.read_csv(path, index_col=index_col, **kwargs)

    else:
        raise ValueError(
            "Cannot determine the filetype of the given path. Expected "
            "either '.csv' or '%s'" % consts.COMPRESSION_SUFFIX
        )
