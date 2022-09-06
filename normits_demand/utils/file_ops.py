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

import itertools
import os
import pathlib
import pickle
import re
import shutil
import time
import warnings
from os import PathLike
from packaging import version
from typing import Any, Collection, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

# Third Party
import numpy as np
import pandas as pd

from caf.toolkit import pandas_utils as caf_pd_utils

# Local imports
import normits_demand as nd
from normits_demand import constants as consts
from normits_demand.concurrency import multiprocessing, multithreading
from normits_demand.utils import compress
from normits_demand.utils import general as du

# Imports that need moving into here
from normits_demand.utils.general import list_files

# CONSTANTS
PD_COMPRESSION = {".zip", ".gzip", ".bz2", ".zstd", ".csv.bz2"}


class WriteDfThread(multithreading.ReturnOrErrorThread):
    """Simple Thread for writing to disk using a thread"""

    def __init__(self, df: pd.DataFrame, path: nd.PathLike, **kwargs) -> None:
        multithreading.ReturnOrErrorThread.__init__(self)
        self.df = df
        self.path = path
        self.kwargs = kwargs

    def run_target(self) -> None:
        """Runs a furness once all data received, and passes data back

        Runs forever - therefore needs to be a daemon.
        Overrides parent to run this on thread start.

        Returns
        -------
        None
        """
        write_df(self.df, self.path, **self.kwargs)


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
        if new.suffix == "":
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


def check_file_exists(
    file_path: nd.PathLike,
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
        raise IOError("Cannot find a path to: %s" % str(file_path))


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
        raise IOError("The following path does not exist: %s" % str(path))


def folder_exists(folder: nd.PathLike) -> pathlib.Path:
    """Raise an error if `folder` doesn't exist or isn't a folder.

    Parameters
    ----------
    folder : nd.PathLike
        Path to check.

    Returns
    -------
    pathlib.Path
        `folder` after conversion to Path.

    Raises
    ------
    NotADirectoryError
        If `folder` isn't a path to a folder or it doesn't exist.
    """
    folder = pathlib.Path(folder)
    if folder.is_dir():
        return folder

    if folder.exists():
        raise NotADirectoryError(f"not a folder: '{folder}'")
    raise NotADirectoryError(f"folder doesn't exist: '{folder}'")


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
    filename_parts = str(file_path).split(".")

    # File doesn't seem to have an file_extension?
    if len(filename_parts) < 1:
        return False

    file_extension = filename_parts[-1].lower()
    return file_extension == "csv"


def maybe_add_suffix(
    path: nd.PathLike,
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
    if path.suffix == "":
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


def read_df(
    path: os.PathLike,
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
            raise FileNotFoundError(f"No such file or directory: '{path}'")
        path = find_filename(path)

    # Determine how to read in df
    if pathlib.Path(path).suffix == ".pbz2":
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

    if pathlib.Path(path).suffix == ".csv":
        return pd.read_csv(path, index_col=index_col, **kwargs)

    if pathlib.Path(path).suffix in PD_COMPRESSION:
        return pd.read_csv(path, index_col=index_col, **kwargs)

    raise ValueError(
        f"Cannot determine the filetype of the given path. "
        f"Expected either '.csv' or '{consts.COMPRESSION_SUFFIX}'\n"
        f"Got path: {path}"
    )


def write_df(df: pd.DataFrame, path: nd.PathLike, **kwargs) -> None:
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
    if pathlib.Path(path).suffix == ".pbz2":
        compress.write_out(df, path)

    elif pathlib.Path(path).suffix == ".csv":
        df.to_csv(path, **kwargs)

    elif pathlib.Path(path).suffix in PD_COMPRESSION:
        df.to_csv(path, **kwargs)

    else:
        raise ValueError(
            f"Cannot determine the filetype of the given path. Expected "
            f"either '.csv' or '{consts.COMPRESSION_SUFFIX}'"
        )


def write_df_threaded(*args, **kwargs) -> WriteDfThread:
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
    thread = WriteDfThread(*args, **kwargs)
    thread.start()
    return thread


def filename_in_list(
    filename: nd.PathLike,
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


def find_filename(
    path: nd.PathLike,
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
        alt_types = [".pbz2", ".csv"] + list(PD_COMPRESSION)

    # Make sure they all start with a dot
    temp_alt_types = list()
    for ftype in alt_types:
        if not du.starts_with(ftype, "."):
            ftype = "." + ftype
        temp_alt_types.append(ftype)
    alt_types = temp_alt_types.copy()

    # Try to find the path as is
    if path.suffix != "":
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


def _copy_all_files_internal(
    import_dir: nd.PathLike,
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
    out_path = export_dir / (in_fname.stem + ".csv")

    # Read in, then write out as csv
    df = read_df(in_path)
    write_df(df, out_path, index=index_col_out)


def copy_all_files(
    import_dir: nd.PathLike,
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
        "import_dir": import_dir,
        "export_dir": export_dir,
        "index_col_out": index_col_out,
        "force_csv_out": force_csv_out,
    }

    kwarg_list = list()
    for in_fname in fnames:
        kwargs = unchanging_kwargs.copy()
        kwargs["in_fname"] = in_fname
        kwarg_list.append(kwargs)

    multiprocessing.multiprocess(
        fn=_copy_all_files_internal,
        kwargs=kwarg_list,
        process_count=process_count,
    )


def _copy_files_internal(
    src: os.PathLike,
    dst: os.PathLike,
) -> None:
    """Copy a file from one location to another"""
    shutil.copy(
        src=src,
        dst=dst,
    )


def copy_and_rename_files(
    files: Sequence[Tuple[os.PathLike, os.PathLike]],
    process_count: int = consts.PROCESS_COUNT,
) -> None:
    """
    Copy files from one location to another

    Takes a list of tuples and copies the first item of each tuple into
    the location given by the second item in a tuple

    Parameters
    ----------
    files:
        A list of Tuples of files to copy from and to. `(src, dst)` Tuples

    process_count:
        THe number of processes to use when copying the data over.
        0 - use no multiprocessing, run as a loop.
        +ve value - the number of processes to use.
        -ve value - the number of processes less than the cpu count to use.

    Returns
    -------
    None
    """
    # Convert into a list of kwargs
    keys = [("src", "dst")] * len(files)
    kwarg_list = [dict(zip(ks, vs)) for ks, vs in zip(keys, files)]

    multiprocessing.multiprocess(
        fn=_copy_files_internal,
        kwargs=kwarg_list,
        process_count=process_count,
    )


def copy_defined_files(
    copy_definition: pd.DataFrame,
    src_dir: os.PathLike,
    dst_dir: os.PathLike,
    src_col: str = "input",
    dst_col: str = "output",
    process_count: int = consts.PROCESS_COUNT,
) -> None:
    """Copy files from source to destination as defined in a pandas DataFrame

    Parameters
    ----------
    copy_definition:
        A Pandas DataFrame defining how to copy the files. The dataframe MUST
        contain at least 2 columns: `src_col` and `dst_col`

    src_dir:
        The directory to find the files in `copy_definition[src_col]`

    dst_dir:
        The directory to write the files in `copy_definition[dst_col]` to

    src_col:
        The name of the column in `copy_definition` containing the names of
        the source files to copy.

    dst_col:
        The name of the column in `copy_definition` containing the names to
        give to the copied files.

    process_count:
        The number of processes to use when copying the data over.
        0 - use no multiprocessing, run as a loop.
        +ve value - the number of processes to use.
        -ve value - the number of processes less than the cpu count to use.

    Returns
    -------
    None

    Raises
    ------
    ValueError:
        If `src_col` or `dst_col` don't exist within copy_definition
    """
    # Convert paths
    src_dir = pathlib.Path(src_dir)
    dst_dir = pathlib.Path(dst_dir)

    # Make sure the columns we need exist
    df = caf_pd_utils.reindex_cols(
        df=copy_definition,
        columns=[src_col, dst_col],
        dataframe_name="copy_definition",
    )

    # Convert to a dictionary of kwargs
    df = df.rename(
        columns={
            src_col: "src",
            dst_col: "dst",
        }
    )
    kwarg_list = df.to_dict(orient="records")

    # Attach directories to paths
    for kwargs in kwarg_list:
        kwargs["src"] = src_dir / kwargs["src"]
        kwargs["dst"] = dst_dir / kwargs["dst"]

    # Multiprocess the copy
    multiprocessing.multiprocess(
        fn=_copy_files_internal,
        kwargs=kwarg_list,
        process_count=process_count,
    )


def remove_from_fname(
    path: nd.PathLike,
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
    new_fname = path.stem.replace(to_remove, "")

    return path.parent / (new_fname + path.suffix)


def add_to_fname(
    path: nd.PathLike,
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


def copy_segment_files(
    src_dir: nd.PathLike,
    dst_dir: nd.PathLike,
    segmentation: nd.SegmentationLevel,
    process_count: int = consts.PROCESS_COUNT,
    **filename_kwargs,
) -> None:
    """Copy segment files from src_dir to dst_dir

    Parameters
    ----------
    src_dir:
        The directory to copy files from.

    dst_dir:
        The directory to copy files to.

    segmentation:
        The segmentation to use to generate the filenames to copy from `src`
        to `dst`

    process_count:
        The number of processes to use when copying files. By default, uses
        the module default process count.

    filename_kwargs:
        Any further kwargs to pass into `segmentation.generate_file_name()`

    Returns
    -------
    None
    """
    # Generate all the filenames
    filenames = list()
    for segment_params in segmentation:
        filenames.append(
            segmentation.generate_file_name(
                segment_params=segment_params,
                **filename_kwargs,
            )
        )

    copy_files(
        src_dir=src_dir,
        dst_dir=dst_dir,
        filenames=filenames,
        process_count=process_count,
    )


def copy_template_segment_files(
    src_dir: nd.PathLike,
    dst_dir: nd.PathLike,
    segmentation: nd.SegmentationLevel,
    input_template_filename: str,
    output_template_filename: str = None,
    process_count: int = consts.PROCESS_COUNT,
) -> None:
    """Copy template segment files from src_dir to dst_dir

    Parameters
    ----------
    src_dir:
        The directory to copy files from.

    dst_dir:
        The directory to copy files to.

    segmentation:
        The segmentation to use to generate the filenames to copy from `src`
        to `dst`. Will be used with `template_segment_filename` to generate
        the filenames.

    input_template_filename:
        The template filename to use for the input filenames.
        Will be used as:
        `segmentation.generate_file_name_from_template(
            input_template_filename, segment_params,
        )`

    output_template_filename:
        The template filename to use for the output filenames. If left as None,
        the same output filename is used as the input.
        Will be used as:
        `segmentation.generate_file_name_from_template(
            output_template_filename, segment_params,
        )`

    process_count:
        The number of processes to use when copying files. By default, uses
        the module default process count.
    """
    # Generate all the filenames
    kwarg_list = list()
    for segment_params in segmentation:
        in_filename = segmentation.generate_file_name_from_template(
            template=input_template_filename,
            segment_params=segment_params,
        )

        if output_template_filename is not None:
            out_filename = segmentation.generate_file_name_from_template(
                template=output_template_filename,
                segment_params=segment_params,
            )
        else:
            out_filename = in_filename

        kwarg_list.append(
            {
                "src": src_dir / in_filename,
                "dst": dst_dir / out_filename,
            }
        )

    # Multiprocess the copy
    multiprocessing.multiprocess(
        fn=_copy_files_internal,
        kwargs=kwarg_list,
        process_count=process_count,
    )


def copy_files(
    src_dir: nd.PathLike,
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
        kwarg_list.append(
            {
                "src": os.path.join(src_dir, fname),
                "dst": os.path.join(dst_dir, fname),
            }
        )

    multiprocessing.multiprocess(
        fn=du.copy_and_rename,
        kwargs=kwarg_list,
        process_count=process_count,
        pbar_kwargs={"disable": False},
    )


def create_folder(
    folder_path: nd.PathLike,
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
        du.print_w_toggle("Folder already exists", verbose=verbose_exists)
        return

    os.makedirs(folder_path, exist_ok=True)
    du.print_w_toggle(
        f"New project folder created at {folder_path}",
        verbose=verbose_create,
    )


def write_pickle(
    obj: object,
    path: os.PathLike,
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
    with open(path, "wb") as file:
        pickle.dump(obj, file, protocol=protocol, **kwargs)


def compare_versions(
    ver1: version.Version,
    ver2: version.Version,
    ver1_name: str,
    ver2_name: str,
    match_str: bool = False,
) -> Optional[str]:
    """
    Compares the versions and generates a message

    Parameters
    ----------
    ver1:
        The first Version object to compare

    ver2:
        The second Version object to compare

    ver1_name:
        The name to give to `ver1` when generating the message string to return

    ver2_name:
        The name to give to `ver2` when generating the message string to return

    match_str:
        Whether to return a message when `ver1` and `ver2` match. If False,
        then `None` is returned when the versions match.

    Returns
    -------
    message:
        A string of a message that changes slightly based on the level of
        difference between `ver1` and `ver2`.

    Raises
    ------
    NotImplementedError:
        When `ver1` and `ver2` have the same base release (major, minor, patch
        numbers), but differ on something other than the post-release.
        Extra code needs adding to solve this error.
    """
    # Compare versions
    if ver1 == ver2:
        if match_str:
            return (
                "Versions match exactly!\n"
                f"{ver1_name} Version: {str(ver1)}\n"
                f"{ver2_name} Version: {str(ver2)}"
            )
        return None

    # Compare base versions
    if version.parse(ver1.base_version) != version.parse(ver2.base_version):
        return (
            "Versions are significantly different\n"
            f"{ver1_name} Version: {str(ver1)}\n"
            f"{ver2_name} Version: {str(ver2)}"
        )

    # Compare post-release
    v1_post = 0 if ver1.post is None else ver1.post
    v2_post = 0 if ver2.post is None else ver2.post
    if v1_post != v2_post:
        return (
            "Versions are similar, but differ on the post-release number\n"
            f"{ver1_name} Version: {str(ver1)}\n"
            f"{ver2_name} Version: {str(ver2)}"
        )

    raise NotImplementedError(
        "Versions are not the same, but differ on something other than the "
        "post-release number. Please implement the code to perform this check."
    )


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
    with open(path, "rb") as file:
        obj = pickle.load(file)

    # If its a DVector, reset the process count
    if isinstance(obj, nd.core.data_structures.DVector):
        obj._process_count = nd.constants.PROCESS_COUNT

    # If no version, return now
    if not hasattr(obj, "__version__"):
        return obj

    # Check if class definition has a version (should do!)
    if not hasattr(obj.__class__, "__version__"):
        warn_msg = (
            f"The object loaded from '{path}' has a version, but the class "
            "definition in the code does not. Aborting version check!\n"
            f"Loaded object is version {obj.__version__}"
        )
        warnings.warn(warn_msg, UserWarning, stacklevel=2)

    # Throw warning if versions don't match
    msg = compare_versions(
        ver1=version.Version(obj.__version__),
        ver2=version.Version(obj.__class__.__version__),
        ver1_name="Object",
        ver2_name="Class",
        match_str=False,
    )
    if msg is not None:
        warn_msg = (
            f"The object loaded from '{path}' is not the same version as the "
            "class definition in the code. This might cause some unexpected "
            f"problems.\n{msg}"
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
        If paths is an empty iterable, np.inf is returned.
    """
    if paths == list():
        return np.inf

    # init
    latest_time = -1

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
        If paths is an empty iterable, -1 is returned.
    """
    if paths == list():
        return -1

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


def is_old_cache(
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

    return get_oldest_modified_time(cache) < get_latest_modified_time(original)


def iterate_files(
    path: PathLike, suffixes: Optional[list[str]] = None, recursive: bool = False
) -> Iterator[pathlib.Path]:
    """Iterate through files in `path` based on optional filtering.

    Parameters
    ----------
    path : PathLike
        Base path all files should start with
    suffixes : list[str], optional
        List of file suffixes to filter by.
    recursive : bool, default False
        Search recursively in sub-folders.

    Yields
    ------
    pathlib.Path
        Path to file matching filter.
    """
    path = pathlib.Path(path)

    path_iter = path.rglob("*") if recursive else path.glob("*")

    for current in path_iter:
        if not current.is_file():
            continue

        if suffixes is not None:
            suffix = "".join(current.suffixes)
            if suffix not in suffixes:
                continue

        yield current


def parse_filename(filename: str) -> dict[str, Union[str, int]]:
    """Extract segmentation parameters from filename.

    Parameters
    ----------
    filename : str
        Name of file following the segmentation naming conventions.

    Returns
    -------
    dict[str, Union[str, int]]
        Dictionary of segment name and value.
    """
    sep = r"(?:\b|[_.])"
    int_values = {"yr", "m", "p", "ca", "soc", "ns", "g"}
    naming_formats = {
        "trip_origin": r"(n?hb)",
        "matrix_type": r"(pa|od_from|od_to)",
        "uc": r"(business|commute|other)",
        "yr": r"yr(\d{4})",
        "m": r"m(\d)",
        "p": r"p(\d{1,2})",
        "ca": r"ca(\d)",
        "soc": r"soc(\d)",
        "ns": r"ns(\d)",
        "g": r"g(\d)",
        "tp": r"tp(\d)",
    }

    data: dict[str, Union[str, int]] = {}
    for nm, pat in naming_formats.items():
        match = re.search(f"{sep}{pat}{sep}", filename, re.I)
        if match is not None:
            if nm in int_values:
                data[nm] = int(match.group(1))
            else:
                data[nm] = match.group(1).lower()

    return data


def parse_folder_files(
    folder: pathlib.Path,
    extension_filter: Collection[str] = None,
    required_data: Collection[str] = None,
) -> Iterator[dict[str, Union[str, int, pathlib.Path]]]:
    """Iterate through files in a folder and return their segmentation parameters.

    Parameters
    ----------
    folder : pathlib.Path
        Folder to iterate through files within.
    extension_filter : Collection[str], optional
        Suffixes to filter files by.
    required_data : Collection[str], optional
        Segments the filenames should contain, files with some missing
        segments will be excluded.

    Yields
    ------
    Iterator[dict[str, Union[str, int, pathlib.Path]]]
        Dictionary containing the file path, with the key 'path' and
        all the segment names and values.
    """
    for file in folder.iterdir():
        if not file.is_file():
            continue

        if extension_filter is not None:
            suffix = "".join(file.suffixes)
            if not suffix in {s.lower() for s in extension_filter}:
                continue
        data = parse_filename(file.name)

        if required_data is not None:
            if any(i not in data for i in required_data):
                continue

        yield {"path": file, **data}


def read_matrix(
    path: os.PathLike,
    format_: Optional[str] = None,
    find_similar: bool = False,
) -> pd.DataFrame:
    """Read matrix CSV in the square or long format.

    Sorts the index and column names and makes sure they're
    the same, doesn't infill any NaNs created when reindexing.

    Parameters
    ----------
    path : os.PathLike
        Path to CSV file
    format_ : str, optional
        Expected format of the matrix 'square' or 'long', if
        not given attempts to figure out the format by reading
        the top few lines of the file.
    find_similar : bool, default False
        If True and the given file at path cannot be found, files with the
        same name but different extensions will be looked for and read in
        instead. Will check for: '.csv', '.pbz2'

    Returns
    -------
    pd.DataFrame
        Matrix file in square format with sorted columns and indices

    Raises
    ------
    ValueError
        If the `format` cannot be determined by reading the file
        or an invalid `format` is given.
    """
    if format_ is None:
        # Determine format by reading top few lines of file
        matrix = read_df(path, nrows=3, find_similar=find_similar)

        if len(matrix.columns) == 3:
            format_ = "long"

            # Check if columns have a header
            if matrix.columns[0].strip().lower() in ("o", "origin", "p", "productions"):
                header = 0
            else:
                header = None

        elif len(matrix.columns) > 3:
            format_ = "square"
            header = 0
        else:
            raise ValueError(f"cannot determine format of matrix {path}")

    format_ = format_.strip().lower()
    if format_ == "square":
        matrix = read_df(path, index_col=0, find_similar=find_similar, header=header)
    elif format_ == "long":
        matrix = read_df(path, index_col=[0, 1], find_similar=True, header=header)

        matrix = matrix.unstack()
        matrix.columns = matrix.columns.droplevel(0)
    else:
        raise ValueError(f"unknown format {format_}")

    # Attempt to convert to integers
    matrix.columns = pd.to_numeric(matrix.columns, errors="ignore", downcast="integer")
    matrix.index = pd.to_numeric(matrix.index, errors="ignore", downcast="integer")

    matrix = matrix.sort_index(axis=0).sort_index(axis=1)
    if (matrix.index != matrix.columns).any():
        # Reindex index to match columns then columns to match index
        matrix = matrix.reindex(matrix.columns, axis=0)
        matrix = matrix.reindex(matrix.index, axis=1)

    return matrix
