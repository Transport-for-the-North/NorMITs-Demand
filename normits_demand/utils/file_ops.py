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

from typing import List

# Third Party
import pandas as pd

# Local imports
import normits_demand as nd
from normits_demand import constants as consts
from normits_demand.utils import compress
from normits_demand.utils import general as du

# Imports that need moving into here
from normits_demand.utils.utils import create_folder
from normits_demand.utils.general import list_files


def cast_to_pathlib_path(path: nd.PathLike) -> pathlib.Path:
    """
    Tries to cast path to pathlib.Path

    Parameters
    ----------
    path:
        The path to convert

    Returns
    -------
    path:
        path, converted to a pathlib.Path object
    """
    if isinstance(path, pathlib.Path):
        return path

    return pathlib.Path(path)


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


def write_df(df: pd.DataFrame, path: nd.PathLike, **kwargs) -> pd.DataFrame:
    """
    Reads in the dataframe at path. Decompresses the df if needed.

    Parameters
    ----------
    df:
        The dataframe to write to disk

    path:
        The full path to the dataframe to read in

    **kwargs:
        Any arguments to pass to the underlying write function.

    Returns
    -------
    df:
        The read in df at path.
    """
    # Init
    path = cast_to_pathlib_path(path)

    # Determine how to read in df
    if pathlib.Path(path).suffix == consts.COMPRESSION_SUFFIX:
        compress.write_out(df, path)

    elif pathlib.Path(path).suffix == '.csv':
        df.to_csv(path, **kwargs)

    else:
        raise ValueError(
            "Cannot determine the filetype of the given path. Expected "
            "either '.csv' or '%s'" % consts.COMPRESSION_SUFFIX
        )


def find_filename(path: nd.PathLike,
                  alt_types: List[str] = None,
                  return_full_path: bool = True,
                  ) -> pathlib.Path:
    """
    Checks if the file at path exists under a different file extension.

    If path ends in a file extension, will try find that file first. If
    that doesn't exist, it will look for a compressed, or '.csv' version.

    Parameters
    ----------
    path:
        The path to the file to try and find

    alt_types:
        A list of alternate filetypes to consider. By default, will be:
        ['.pbz2', '.csv']

    return_full_path:
        If False, will only return the name of the file, and not the full path

    Returns
    -------
    path:
        The path to a matching, or closely matching (differing only on
        filetype extension) file.
    """
    # Init
    path = cast_to_pathlib_path(path)

    # Wrapper around return to deal with full path or not
    def return_fn(ret_path):
        if return_full_path:
            return ret_path
        return ret_path.name

    if alt_types is None:
        alt_types = ['.pbz2', '.csv']

    # Make sure they all start with a dot
    temp_alt_types = list()
    for ftype in alt_types:
        if not du.starts_with(ftype, '.'):
            ftype = '.' + ftype
        temp_alt_types.append(ftype)
    alt_types = temp_alt_types.copy()

    # Try to find the path as is
    if path.suffix != '':
        if os.path.exists(path):
            return return_fn(path)

    # Try to find similar paths
    attempted_paths = list()
    for ftype in alt_types:
        path = path.parent / (path.stem + ftype)
        attempted_paths.append(path)
        if os.path.exists(path):
            return return_fn(path)

    # If here, not paths were found!
    raise FileNotFoundError(
        "Cannot find any similar files. Tried all of the following paths: %s"
        % str(attempted_paths)
    )
