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
from __future__ import annotations

import os
import time
import pickle
import pathlib
import warnings
import itertools

from os import PathLike

from typing import Any
from typing import List
from typing import Union
from typing import Iterable

# Third Party
import numpy as np
import pandas as pd

# Local imports
import normits_demand as nd
from normits_demand import constants as consts
from normits_demand.utils import compress
from normits_demand.utils import general as du

from normits_demand.concurrency import multiprocessing as multiprocessing

# Imports that need moving into here
from normits_demand.utils.general import list_files

# CONSTANTS
PD_COMPRESSION = {'.zip', '.gzip', '.bz2', '.zstd', '.csv.bz2'}


def remove_suffixes(path: pathlib.Path) -> pathlib.Path:
    """Removes all suffixes from path

    Parameters
    ----------
    path:
        The path to remove the suffixes from

    Returns
    -------
    path:
        path with all suffixes removed
    """
    # Init
    parent = path.parent
    prev = pathlib.Path(path.name)

    # Remove a suffix then check if all are removed
    while True:
        new = pathlib.Path(prev.stem)

        # No more suffixes to remove
        if new.suffix == '':
            break

        prev = new

    return parent / new


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


def check_file_exists(file_path: nd.PathLike,
                      find_similar: bool = False,
                      ) -> None:
    """
    Checks if a file exists at the given path. Throws an error if not.

    Parameters
    ----------
    file_path:
        path to the file to check.

    find_similar:
        Whether to look for files with the same name, but a different file
        type extension. If True, this will call find_filename() using the
        default alternate file types: ['.pbz2', '.csv']

    Returns
    -------
    None
    """
    if find_similar:
        find_filename(file_path)
        return

    if not file_exists(file_path):
        raise IOError(
            "Cannot find a path to: %s" % str(file_path)
        )


def check_path_exists(path: nd.PathLike) -> None:
    """
    Checks if a path exists. Throws an error if not.

    Parameters
    ----------
    path:
        path to the file to check.

    Returns
    -------
    None
    """
    if not os.path.exists(path):
        raise IOError(
            "The following path does not exist: %s" % str(path)
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
                     overwrite: bool = True,
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


def is_index_set(df: pd.DataFrame):
    """
    Tries to check if the index of df has been set.

    Parameters
    ----------
    df:
        The df to check

    Returns
    -------
    is_index_set:
        True if index is set
    """
    # If name is set, index is probably set
    if df.index.name is not None:
        return True

    # If resetting the index changes it, it was probably set
    pre_index = df.index
    post_index = df.reset_index().index
    if not (pre_index == post_index).all():
        return True

    return False


def read_df(path: nd.PathLike,
            index_col: int = None,
            find_similar: bool = False,
            **kwargs,
            ) -> pd.DataFrame:
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

    find_similar:
        If True and the given file at path cannot be found, files with the
        same name but different extensions will be looked for and read in
        instead. Will check for: '.csv', '.pbz2'

    Returns
    -------
    df:
        The read in df at path.
    """

    # Try and find similar files if we are allowed
    if not os.path.exists(path):
        if not find_similar:
            raise FileNotFoundError(
                "No such file or directory: '%s'" % path
            )
        path = find_filename(path)

    # Determine how to read in df
    if pathlib.Path(path).suffix == '.pbz2':
        df = compress.read_in(path)

        # Optionally try and set the index
        if index_col is not None and not is_index_set(df):
            df = df.set_index(list(df)[index_col])

        # Unset the index col if it is set - this is how pd.read_csv() works
        if index_col is None and df.index.name is not None:
            df = df.reset_index()

        # Make sure no column name is set - this is how pd.read_csv() works
        df.columns.name = None
        return df

    elif pathlib.Path(path).suffix == '.csv':
        return pd.read_csv(path, index_col=index_col, **kwargs)

    elif pathlib.Path(path).suffix in PD_COMPRESSION:
        return pd.read_csv(path, index_col=index_col, **kwargs)

    else:
        raise ValueError(
            "Cannot determine the filetype of the given path. Expected "
            "either '.csv' or '%s'\n"
            "Got path: %s"
            % (consts.COMPRESSION_SUFFIX, path)
        )


def write_df(df: pd.DataFrame, path: nd.PathLike, **kwargs) -> pd.DataFrame:
    """
    Writes the dataframe at path. Decompresses the df if needed.

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
    path = pathlib.Path(path)

    # Determine how to read in df
    if pathlib.Path(path).suffix == '.pbz2':
        compress.write_out(df, path)

    elif pathlib.Path(path).suffix == '.csv':
        df.to_csv(path, **kwargs)

    elif pathlib.Path(path).suffix in PD_COMPRESSION:
        df.to_csv(path, **kwargs)

    else:
        raise ValueError(
            "Cannot determine the filetype of the given path. Expected "
            "either '.csv' or '%s'" % consts.COMPRESSION_SUFFIX
        )


def filename_in_list(filename: nd.PathLike,
                     lst: List[nd.PathLike],
                     ignore_ftype: bool = False,
                     ) -> bool:
    """Returns True if filename exists in lst

    Parameters
    ----------
    filename:
        The filename to search for in lst

    lst:
        The list to search for filename in

    ignore_ftype:
        Whether to ignore the filetypes in both the filename and lst
        when searching

    Returns
    -------
    boolean:
        True if filename is in lst, False otherwise
    """
    # If we're not ignoring ftype, we can do a simple check
    if not ignore_ftype:
        return filename in lst

    # Init
    filename = pathlib.Path(filename)
    lst = [pathlib.Path(x) for x in lst]

    # Compare the names
    for item in lst:
        item_no_ftype = item.parent / item.stem
        filename_no_ftype = filename.parent / filename.stem

        if item_no_ftype == filename_no_ftype:
            return True

    return False


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

    Raises
    ------
    FileNotFoundError:
        If the file cannot be found under any of the given alt_types file
        extensions.
    """
    # Init
    path = pathlib.Path(path)

    # Wrapper around return to deal with full path or not
    def return_fn(ret_path):
        if return_full_path:
            return ret_path
        return ret_path.name

    if alt_types is None:
        alt_types = ['.pbz2', '.csv'] + list(PD_COMPRESSION)

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
    base_path = remove_suffixes(path)
    for ftype in alt_types:
        i_path = base_path.with_suffix(ftype)
        attempted_paths.append(i_path)
        if os.path.exists(i_path):
            return return_fn(i_path)

    # If here, no paths were found!
    raise FileNotFoundError(
        "Cannot find any similar files. Tried all of the following paths: %s"
        % str(attempted_paths)
    )

def similar_file_exists(
    path: nd.PathLike,
    alt_types: List[str] = None,
) -> bool:
    """Checks if the file at path exists under a different file extension.

    If this function return `True`, `file_ops.read_df()` can be called with
    `find_similar=True` without fail.

    Parameters
    ----------
    path:
        The path to the file to try and find

    alt_types:
        A list of alternate filetypes to consider. By default, will be:
        ['.pbz2', '.csv']

    Returns
    -------
    bool:
        True if the file exists, else False.

    """
    does_file_exist = True
    try:
        find_filename(path=path, alt_types=alt_types)
    except FileNotFoundError:
        does_file_exist = False
    return does_file_exist


def _copy_all_files_internal(import_dir: nd.PathLike,
                             export_dir: nd.PathLike,
                             force_csv_out: bool,
                             index_col_out: bool,
                             in_fname: nd.PathLike,
                             ) -> None:
    """
    internal function of copy_all_files
    """
    in_fname = pathlib.Path(in_fname)

    # Do we need to convert the file? We do with
    # if not(force_csv_out and in_fname.suffix != '.csv'):
    #     # If not, we can just copy over as is
    #     du.copy_and_rename(
    #         src=import_dir / in_fname,
    #         dst=export_dir / in_fname,
    #     )
    #     return

    # Only get here if we do need to convert the file type
    in_path = import_dir / in_fname
    out_path = export_dir / (in_fname.stem + '.csv')

    # Read in, then write out as csv
    df = read_df(in_path)
    write_df(df, out_path, index=index_col_out)


def copy_all_files(import_dir: nd.PathLike,
                   export_dir: nd.PathLike,
                   force_csv_out: bool = False,
                   index_col_out: bool = True,
                   process_count: int = consts.PROCESS_COUNT,
                   ) -> None:
    """
    Copies all of the files from import_dir into export_dir

    Will attempt to read in any format of file, but the output can be
    forced into csv format using force_csv_out.

    Parameters
    ----------
    import_dir:
        Path to the directory containing all the files to copy

    export_dir:
        Path to the directory where all the copied files should be placed

    force_csv_out:
        If True, the copied files will be translated into .csv files.


    index_col_out:
        If True, will write the index column out as well.

    process_count:
        THe number of processes to use when copying the data over.
        0 - use no multiprocessing, run as a loop.
        +ve value - the number of processes to use.
        -ve value - the number of processes less than the cpu count to use.

    Returns
    -------
    None
    """
    # Init
    fnames = du.list_files(import_dir)
    import_dir = pathlib.Path(import_dir)
    export_dir = pathlib.Path(export_dir)

    # ## MULTIPROCESS THE COPY ## #
    unchanging_kwargs = {
        'import_dir': import_dir,
        'export_dir': export_dir,
        'index_col_out': index_col_out,
        'force_csv_out': force_csv_out,
    }

    kwarg_list = list()
    for in_fname in fnames:
        kwargs = unchanging_kwargs.copy()
        kwargs['in_fname'] = in_fname
        kwarg_list.append(kwargs)

    multiprocessing.multiprocess(
        fn=_copy_all_files_internal,
        kwargs=kwarg_list,
        process_count=process_count,
    )


def remove_from_fname(path: nd.PathLike,
                      to_remove: str,
                      ) -> pathlib.Path:
    """
    Returns path without to_remove in it

    Parameters
    ----------
    path:
        The path to edit

    to_remove:
        The string to remove from path.

    Returns
    -------
    path:
        path without to_remove in it
    """
    # Init
    path = pathlib.Path(path)

    # Get a version of the filename without the suffix
    new_fname = path.stem.replace(to_remove, '')

    return path.parent / (new_fname + path.suffix)


def add_to_fname(path: nd.PathLike,
                 to_add: str,
                 ) -> pathlib.Path:
    """
    Returns path with to_add in it

    Parameters
    ----------
    path:
        The path to edit

    to_add:
        The string to add to the end of path (before the file type extension).

    Returns
    -------
    path:
        path with to_add in it
    """
    # Init
    path = pathlib.Path(path)

    # Get a version of the filename without the suffix
    new_fname = path.stem + to_add

    return path.parent / (new_fname + path.suffix)


def remove_internal_suffix(path: nd.PathLike) -> pathlib.Path:
    """
    Returns path without the internal suffix in it.

    The internal suffix comes from efs_consts.INTERNAL_SUFFIX

    Parameters
    ----------
    path:
        The path to remove the internal suffix from

    Returns
    -------
    path:
        path without the internal suffix in it
    """
    return remove_from_fname(path, consts.INTERNAL_SUFFIX)


def add_external_suffix(path: nd.PathLike) -> pathlib.Path:
    """
    Returns path with the external suffix added to it

    The external suffix comes from efs_consts.EXTERNAL_SUFFIX

    Parameters
    ----------
    path:
        The path to edit

    Returns
    -------
    path:
        path with the external suffix added
    """
    return add_to_fname(path, consts.EXTERNAL_SUFFIX)


def copy_segment_files(src_dir: nd.PathLike,
                       dst_dir: nd.PathLike,
                       segmentation: nd.SegmentationLevel,
                       process_count: int = consts.PROCESS_COUNT,
                       **filename_kwargs
                       ) -> None:
    """Copy segment files from src_dir to dst_dir

    Parameters
    ----------
    src_dir
    dst_dir
    segmentation
    process_count
    filename_kwargs
    """
    # Generate all the filenames
    filenames = list()
    for segment_params in segmentation:
        filenames.append(segmentation.generate_file_name(
            segment_params=segment_params,
            **filename_kwargs,
        ))

    copy_files(
        src_dir=src_dir,
        dst_dir=dst_dir,
        filenames=filenames,
        process_count=process_count,
    )


def copy_files(src_dir: nd.PathLike,
               dst_dir: nd.PathLike,
               filenames: List[str],
               process_count: int = consts.PROCESS_COUNT,
               ) -> None:
    """Copy files from src_dir to dst_dir

    Copies all files in `filenames` from `src_dir` into `dst_dir`. Internally
    uses multiprocessing to do the copy to make it really fast.

    Parameters
    ----------
    src_dir:
        The directory to copy `filenames` from.

    dst_dir:
        The directory to copy `filenames` to.

    filenames:
        A list of the filenames to copy.

    process_count:
        The number of processes to use when copying files. By default, uses
        the module default process count.

    Returns
    -------
    None
    """
    # Setup kwargs
    kwarg_list = list()
    for fname in filenames:
        kwarg_list.append({
            'src': os.path.join(src_dir, fname),
            'dst': os.path.join(dst_dir, fname),
        })

    multiprocessing.multiprocess(
        fn=du.copy_and_rename,
        kwargs=kwarg_list,
        process_count=process_count,
        pbar_kwargs={'disable': False},
    )


def create_folder(folder_path: nd.PathLike,
                  verbose_create: bool = True,
                  verbose_exists: bool = False,
                  ) -> None:
    """
    Create a new folder at desired location

    Parameters
    ----------
    folder_path:
        Path to the folder to create

    verbose_create:
        Whether to print a message when creating the path

    verbose_exists:
        Whether to print a message when the path already exists

    """
    # Check if path exists
    if os.path.exists(folder_path):
        du.print_w_toggle('Folder already exists', verbose=verbose_exists)
        return

    os.makedirs(folder_path, exist_ok=True)
    du.print_w_toggle(
        f"New project folder created at {folder_path}",
        verbose=verbose_create,
    )


def write_pickle(obj: object,
                 path: nd.PathLike,
                 protocol: int = pickle.HIGHEST_PROTOCOL,
                 **kwargs,
                 ) -> None:
    """Load any pickled object from disk at path.

    Parameters
    ----------
    obj:
        The object to pickle and write to disk

    path:
        Filepath to write obj to

    protocol:
        The pickle protocol to use when dumping to disk

    **kwargs:
        Any additional arguments to pass to pickle.dump()

    Returns
    -------
    None
    """
    with open(path, 'wb') as file:
        pickle.dump(obj, file, protocol=protocol, **kwargs)


def read_pickle(path: nd.PathLike) -> Any:
    """Load any pickled object from disk at path.

    Parameters
    ----------
    path:
        Filepath to the object to read in and unpickle

    Returns
    -------
    unpickled:
        Same type as object stored in file.
    """
    # Validate path
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No file to read in found at {path}")

    # Read in
    with open(path, 'rb') as file:
        obj = pickle.load(file)

    # If its a DVector, reset the process count
    if isinstance(obj, nd.core.data_structures.DVector):
        obj._process_count = nd.constants.PROCESS_COUNT

    # If no version, return now
    if not hasattr(obj, '__version__'):
        return obj

    # Check if class definition has a version (should do!)
    if not hasattr(obj.__class__, '__version__'):
        warn_msg = (
            f"The object loaded from '{path}' has a version, but the class "
            "definition in the code does not. Aborting version check!\n"
            f"Loaded object is version {obj.__version__}"
        )
        warnings.warn(warn_msg, UserWarning, stacklevel=2)

    # Throw warning if versions don't match
    if obj.__version__ != obj.__class__.__version__:
        warn_msg = (
            f"The object loaded from '{path}' is not the same version as the "
            "class definition in the code. This might cause some unexpected "
            "problems.\n"
            f"Object Version: {obj.__version__}\n"
            f"Class Version: {obj.__class__.__version__}"
        )
        warnings.warn(warn_msg, UserWarning, stacklevel=2)

    return obj


def safe_dataframe_to_csv(df, out_path, **to_csv_kwargs):
    """
    Wrapper around df.to_csv. Gives the user a chance to close the open file.

    Parameters
    ----------
    df:
        pandas.DataFrame to write to call to_csv on

    out_path:
        Where to write the file to. TO first argument to df.to_csv()

    to_csv_kwargs:
        Any other kwargs to be passed straight to df.to_csv()

    Returns
    -------
        None
    """
    written_to_file = False
    waiting = False
    while not written_to_file:
        try:
            df.to_csv(out_path, **to_csv_kwargs)
            written_to_file = True
        except PermissionError:
            if not waiting:
                print(
                    f"Cannot write to file at {out_path}.\n"
                    "Please ensure it is not open anywhere.\n"
                    "Waiting for permission to write...\n"
                )
                waiting = True
            time.sleep(1)


def get_latest_modified_time(paths: Iterable[PathLike]) -> float:
    """Get the latest modified time of all files

    Parameters
    ----------
    paths:
        An iterable of paths to check.

    Returns
    -------
    latest_modified_time:
        The latest modified time of all paths.
        If paths is an empty iterable, -1.0 is returned.
    """
    # init
    latest_time = -1.0

    # Check the latest time of all paths
    for path in paths:
        # Keep the latest time
        modified_time = os.path.getmtime(path)
        if modified_time > latest_time:
            latest_time = modified_time

    return latest_time


def get_oldest_modified_time(paths: Iterable[PathLike]) -> float:
    """Get the oldest modified time of all files

    Parameters
    ----------
    paths:
        An iterable of paths to check.

    Returns
    -------
    oldest_modified_time:
        The oldest modified time of all paths.
        If paths is an empty iterable, np.inf is returned.
    """
    # init
    oldest_time = np.inf

    # Check the latest time of all paths
    for path in paths:
        # Keep the latest time
        modified_time = os.path.getmtime(path)
        if modified_time < oldest_time:
            oldest_time = modified_time

    return oldest_time


def _convert_to_path_list(
    to_convert: Union[pathlib.Path, Iterable[pathlib.Path]],
) -> List[pathlib.Path]:
    """Convert the input into a list of paths

    Takes either a directory, file path, or list of file paths and converts
    into a list of file paths. If a list of file paths is given, this is
    returned. If a list of directories is given, all files in all directories
    are returned as a list. If a single file path is given, it is converted into a list
    with a single item. If a directory is given, all files in the directory
    are returned as a list.

    Parameters
    ----------
    to_convert:
        The directory, file path, or list of file paths to convert.

    Returns
    -------
    file_path_list:
        A list of file paths.
    """
    # If list, try convert each item
    if isinstance(to_convert, list):
        path_lists = [_convert_to_path_list(x) for x in to_convert]
        return list(itertools.chain.from_iterable(path_lists))

    # Otherwise, should be a Path
    if not isinstance(to_convert, pathlib.Path):
        raise ValueError(f"Expected a pathlib.Path. Got {type(to_convert)}")

    # If a file, convert to single item list
    if to_convert.is_file():
        return [to_convert]

    # Must be a directory, get all filenames
    return [x for x in to_convert.iterdir() if x.is_file()]


def is_cache_older(
    original: Union[pathlib.Path, Iterable[pathlib.Path]],
    cache: Union[pathlib.Path, Iterable[pathlib.Path]],
    ignore_cache: bool = False,
) -> bool:
    """Check if the newest original file is newer than the oldest cache.

    Loops though all files in `original` files (checks with `Path.isfile()`)
    and get the latest modified time of the newest file. Then gets the oldest
    modified time of the oldest file in `cache`. Only returns True if
    the oldest cache is still older than the newest original file.

    Parameters
    ----------
    original:
        The directory, file path, or list of file paths or directories of
        the original files to check.

    cache:
        The directory, file path, or list of file paths or directories of
        the cache files to check.

    ignore_cache:
        Whether to completely ignore the check and ignore the cache no
         matter what. If set to True, this function is short circuited and
         will immediately return False.

    Returns
    -------
    is_cache_older:
        A boolean stating whether the cache is older or not.
    """
    # Short circuit if ignoring the cache
    if ignore_cache:
        return False

    # Make sure we've just got lists if paths
    original = _convert_to_path_list(original)
    cache = _convert_to_path_list(cache)

    return get_oldest_modified_time(cache) > get_latest_modified_time(original)
