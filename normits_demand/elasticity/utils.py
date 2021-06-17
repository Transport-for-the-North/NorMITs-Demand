# -*- coding: utf-8 -*-
"""
    Module containing the utility functions for the elasticity calculations.
"""

##### IMPORTS #####
# Standard imports
from pathlib import Path
from typing import Union, Dict, List, Tuple

# Third party imports
import numpy as np
import pandas as pd

# Local imports
import normits_demand as nd
from normits_demand.utils import general as du


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
        "trip_origin": str,
        "p": "int8",
        "soc": "float16",
        "ns": "float16",
        "tp": str,
        "elast_p": str,
        "elast_market_share": str,
    }
    return du.safe_read_csv(path, dtype=dtypes, usecols=dtypes.keys())


def read_elasticity_file(data: Union[nd.PathLike, pd.DataFrame],
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
        "type": str,
        "p": str,
        "market_share": str,
        "affected_mode": str,
        "changing_mode": str,
        "elast_value": "float32",
    }

    if isinstance(data, pd.DataFrame):
        df = data.reindex(columns=list((dtypes.keys())))
    else:
        df = du.safe_read_csv(data, dtype=dtypes, usecols=dtypes.keys())

    # Filter required values
    filters = {
        "type": elasticity_type,
        "p": purpose,
        "market_share": market_share,
    }
    df = du.filter_df(df, filters).reset_index(drop=True)
    return df


def get_constraint_mats(folder: nd.PathLike,
                        get_files: List[str] = None,
                        keep_ftype: bool = False,
                        ) -> Dict[str, np.array]:
    """Search the given folder for any CSV files.

    All constraint matrices should be in the `COMMON_ZONE_SYSTEM`.

    Parameters
    ----------
    folder : Path
        Folder containing the constraint matrices as CSV files.

    get_files : List[str], optional
        The names of the matrices to read, if None then all CSVs
        found will be returned.

    keep_ftype:
        Whether the keep the ftype in the name given to the return dictionary.

    Returns
    -------
    constraint_mats:
        The name of the file and the absolute path to it
        (if get_files is None). If get_files is a list of names then
        those files will be read and returned.

    Raises
    ------
    FileNotFoundError
        If the folder given doesn't exist or isn't a folder.

    FileNotFoundError
        If one of the files in get_files cannot be found in folder.
    """
    # Init
    if not folder.is_dir():
        raise FileNotFoundError(f"Not a folder, or doesn't exist: {folder}")

    if get_files is None:
        get_files = du.list_files(folder, ftypes=['.csv'])

    # Load in the matrices
    matrices = dict()
    for fname in get_files:
        # Build the path - Add suffix if not there
        path = folder / fname
        if path.suffix == '':
            path = path.parent / (path.name + '.csv')

        if not path.is_file():
            raise FileNotFoundError(
                "Path not found for constraint_mat %s.\nWas looking in: %s"
                % (fname, folder)
            )

        # Assign to dictionary
        if keep_ftype:
            matrices[path.name] = np.loadtxt(path, delimiter=",")
        else:
            matrices[path.stem] = np.loadtxt(path, delimiter=",")

    return matrices