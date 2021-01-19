# -*- coding: utf-8 -*-
"""
    Module containing the utility functions for the elasticity calculations.
"""

##### IMPORTS #####
# Standard imports
import sys
import contextlib
from pathlib import Path
from typing import Union, Dict, List, Tuple

# Third party imports
import numpy as np
import pandas as pd
from tqdm.contrib import DummyTqdmFile

# Local imports
from demand_utilities.utils import safe_read_csv
from zone_translator import translate_matrix, MatrixTotalError


##### CONSTANTS #####
COMMON_ZONE_SYSTEM = "norms"
ZONE_LOOKUP_NAME = "{from_zone}_to_{to_zone}.csv"


##### FUNCTIONS #####
def read_segments_file(path: Path) -> pd.DataFrame:
    """Read the segments CSV file containing all the TfN segmentation info.

    Parameters
    ----------
    path : Path
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        The TfN segment information to be used for the
        elasticity calculations.
    """
    dtypes = {
        "EFS_Seg": "int16",
        "EFS_PurpBase": str,
        "EFS_MainPurp": str,
        "EFS_SubPurp": str,
        "EFS_SubPurpID": "int8",
        "EFS_TimePeriod": str,
        "EFS_SkillLevel": "float16",
        "EFS_IncLevel": "float16",
        "Elast_Purp": str,
        "Elast_MarketShare": str,
    }
    return safe_read_csv(path, dtype=dtypes, usecols=dtypes.keys())


def read_elasticity_file(
    data: Union[Path, pd.DataFrame],
    elasticity_type: str = None,
    purpose: str = None,
    market_share: str = None,
) -> pd.DataFrame:
    """Reads and filters the elasticity file.

    Can be given a DataFrame instead of a path for just filtering.

    Parameters
    ----------
    data : Union[Path, pd.DataFrame]
        Path to the elasticity file or a DataFrame containing the information.
    elasticity_type : str
        Value to filter on the 'ElasticityType' column, if None (default) then
        does not filter on this column.
    purpose : str
        Value to filter on the 'Purp' column, if None (default) then
        does not filter on this column.
    market_share : str
        Value to filter on the 'MarketShare' column, if None (default) then
        does not filter on this column.

    Returns
    -------
    pd.DataFrame
        The filtered elasticity data.

    Raises
    ------
    ValueError
        If the combination of filters leads to no remaining rows in
        the DataFrame.
    """
    dtypes = {
        "ElasticityType": str,
        "Purp": str,
        "MarketShare": str,
        "AffectedMode": str,
        "ModeCostChg": str,
        "OwnElast": "float32",
    }
    if isinstance(data, pd.DataFrame):
        df = data.copy()[dtypes.keys()]
    else:
        df = safe_read_csv(data, dtype=dtypes, usecols=dtypes.keys())

    # Filter required values
    for col, val in [
        ("ElasticityType", elasticity_type),
        ("Purp", purpose),
        ("MarketShare", market_share),
    ]:
        if val is None:
            continue
        df = df.loc[df[col] == val]
        if df.empty:
            raise ValueError(f"Value '{val}' not found in column '{col}'")

    return df


def get_constraint_matrices(
    folder: Path, get_files: List[str] = None
) -> Dict[str, Union[Path, np.array]]:
    """Search the given folder for any CSV files.

    All constraint matrices should be in the `COMMON_ZONE_SYSTEM`.

    Parameters
    ----------
    folder : Path
        Folder containing the constraint matrices as CSV files.
    get_files : List[str], optional
        The names of the matrices to read, if None (default) then
        the paths of all CSVs found will be returned.

    Returns
    -------
    Dict[str, Union[Path, np.array]]
        The lowercase name of the file and the absolute path to it
        (if get_files is None). If get_files is a list of names then
        those files will be read and returned.

    Raises
    ------
    FileNotFoundError
        If the folder given doesn't exist or isn't a folder.
    """
    matrices = {}
    if not folder.is_dir():
        raise FileNotFoundError(f"Not a folder, or doesn't exist: {folder}")
    get_files = (
        [i.lower() for i in get_files] if get_files is not None else get_files
    )

    for path in folder.iterdir():
        if path.suffix.lower() != ".csv":
            continue
        nm = path.stem.lower()
        if get_files is None:
            matrices[nm] = path.absolute()
        elif nm in get_files:
            matrices[nm] = np.loadtxt(path, delimiter=",")
    return matrices


def read_demand_matrix(
    path: Path, zone_translation_folder: Path, from_zone: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Reads demand matrix and converts it to `COMMON_ZONE_SYSTEM`.

    Parameters
    ----------
    path : Path
        Path to the demand matrix.
    zone_translation_folder : Path
        Path to the folder contain zone lookups.
    from_zone : str
        The current zone system of the matrix, if this
        isn't `COMMON_ZONE_SYSTEM` then the matrix will
        be converted.

    Returns
    -------
    pd.DataFrame
        The demand matrix in the `COMMON_ZONE_SYSTEM`.
    pd.DataFrame
        Splitting factors for converting back to the old
        zone system.
    pd.DataFrame
        The demand matrix in the `from_zone` zone system.
    """
    demand = pd.read_csv(path, index_col=0)
    # Convert column and index names to int
    demand.columns = pd.to_numeric(demand.columns, downcast="integer")
    demand.index = pd.to_numeric(demand.index, downcast="integer")

    reverse = None
    old_zone = None
    if from_zone != COMMON_ZONE_SYSTEM:
        old_zone = demand.copy()
        lookup_file = zone_translation_folder / ZONE_LOOKUP_NAME.format(
            from_zone=from_zone, to_zone=COMMON_ZONE_SYSTEM
        )
        dtypes = {
            f"{from_zone}_zone_id": int,
            f"{COMMON_ZONE_SYSTEM}_zone_id": int,
            "split": float,
        }
        lookup = pd.read_csv(lookup_file, usecols=dtypes.keys(), dtype=dtypes)
        cols = [f"{from_zone}_zone_id", f"{COMMON_ZONE_SYSTEM}_zone_id"]
        try:
            demand, reverse = translate_matrix(
                demand,
                lookup,
                cols,
                split_column="split",
            )
        except MatrixTotalError as e:
            # Print the error but continue with the translation to still
            # process current segment
            print(f"{path.stem} - {e.__class__.__name__}: {e}")
            demand, reverse = translate_matrix(
                demand,
                lookup,
                cols,
                split_column="split",
                check_total=False
            )

    return demand.sort_index().sort_index(axis=1), reverse, old_zone


@contextlib.contextmanager
def std_out_err_redirect_tqdm():
    """Redirect stdout and stderr to `tqdm.write`.

    Code copied from tqdm documentation:
    https://github.com/tqdm/tqdm#redirecting-writing

    Yields
    -------
    sys.stdout
        Original stdout.
    """
    orig_out_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = map(DummyTqdmFile, orig_out_err)
        yield orig_out_err[0]
    # Relay exceptions
    except Exception as exc:
        raise exc
    # Always restore sys.stdout/err if necessary
    finally:
        sys.stdout, sys.stderr = orig_out_err
