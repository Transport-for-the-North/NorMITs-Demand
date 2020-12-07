# -*- coding: utf-8 -*-
"""
    Module containing the unit tests for the elasticity_calcs.utils module.
"""

##### IMPORTS #####
# Standard imports
from pathlib import Path
from typing import Tuple

# Third party imports
import pytest
import pandas as pd

# Local imports
from .utils import read_segments_file, read_elasticity_file


##### CONSTANTS #####
SEGMENTS_DATA = {
    "EFS_Seg": [1, 2, 3, 7, 8, 9, 10, 11, 37, 38, 39],
    "EFS_PurpBase": ["hb"] * 8 + ["nhb"] * 3,
    "EFS_MainPurp": ["hb_w"] * 3 + ["hb_o"] * 5 + ["nhb_eb"] * 3,
    "EFS_SubPurp": ["commute"] * 3 + ["edu"] * 5 + ["eb"] * 3,
    "EFS_SubPurpID": [1] * 3 + [3] * 5 + [12] * 3,
    "EFS_TimePeriod": ["24hr"] * 11,
    "EFS_SkillLevel": [1, 2, 3] + [None] * 5 + [1, 2, 3],
    "EFS_IncLevel": [None] * 3 + [1, 2, 3, 4, 5] + [None] * 3,
    "Elast_Purp": ["Commuting"] * 8 + ["Business"] * 3,
    "Elast_MarketShare": ["Car_med"] * 11,
}
ELASTICITY_DATA = {
    "ElasticityType": ["CarFuel"] * 9 + ["CarPark"] * 2,
    "Purp": ["Commuting"] * 5 + ["Leisure"] * 4 + ["Business"] * 2,
    "MarketShare": ["Car_high"] * 11,
    "AffectedMode": ["Car"] * 5 + ["Bus"] * 4 + ["Car"] * 2,
    "ModeCostChg": [
        "Car",
        "Bus",
        "Rail",
        "Active",
        "Not_travel",
        "Car",
        "Bus",
        "Rail",
        "Active",
        "Car",
        "Bus",
    ],
    "OwnElast": [-0.3, 0.06, 0.07, 0.02, 0.01, 0.16, -0.8, 0.17, 0.02, -0.3, 0.06],
    "CstrMatrixName": ["all_trips"] * 11,
}


##### FUNCTIONS #####
@pytest.fixture(name="segments_file", scope="module")
def fixture_segments_file(tmpdir_factory) -> Tuple[Path, pd.DataFrame]:
    """Create segments CSV file for testing.

    Parameters
    ----------
    tmpdir : Path
        Temporary directory provided by pytest.

    Returns
    -------
    Tuple[Path, pd.DataFrame]
        Path to the testing segments file and the DataFrame for comparison.
    """
    df = pd.DataFrame(SEGMENTS_DATA)
    file_path = tmpdir_factory.mktemp("inputs") / "segments_file.csv"
    df.to_csv(file_path, index=False)
    return file_path, df


@pytest.fixture(name="elasticity_file", scope="module")
def fixture_elasticity_file(tmpdir_factory) -> Tuple[Path, pd.DataFrame]:
    """Creates elasticity CSV file for testing.

    Parameters
    ----------
    tmpdir_factory : Path
        Temporary directory provided by pytest.

    Returns
    -------
    Tuple[Path, pd.DataFrame]
        Path to the testing elasticity file and the DataFrame for comparison.
    """
    file_path = tmpdir_factory.mktemp("inputs") / "elasticity_file.csv"
    df = pd.DataFrame(ELASTICITY_DATA)
    df.to_csv(file_path, index=False)
    return file_path, df


def test_read_segments_file(segments_file: Tuple[Path, pd.DataFrame]):
    """Test the `read_segments_file` function reads file correctly.

    Parameters
    ----------
    segments_file : Tuple[Path, pd.DataFrame]
        Path to the testing segments file and the DataFrame for comparison.
    """
    df = read_segments_file(segments_file[0])
    pd.testing.assert_frame_equal(df, segments_file[1], check_dtype=False)


@pytest.mark.parametrize(
    "func, args", [(read_segments_file, []), (read_elasticity_file, ["", "", ""])]
)
def test_missing_file(func, args):
    """Tests that an error is raised if path doesn't exist. """
    path = Path.cwd() / "non-existent_file.csv"
    with pytest.raises(IOError):
        func(path, *args)


def test_read_elasticity_file(elasticity_file: Tuple[Path, pd.DataFrame]):
    """Test the `read_elasticity_file` function correctly reads a file.

    Parameters
    ----------
    elasticity_file : Tuple[Path, pd.DataFrame]
        Path to the testing elasticity file and the DataFrame for comparison.

    See Also
    --------
    `test_filter_elasticity_file` for testing of filtering the file.
    """
    df = read_elasticity_file(elasticity_file[0])
    pd.testing.assert_frame_equal(df, elasticity_file[1], check_dtype=False)


@pytest.mark.parametrize("elasticity_type", ["CarFuel", "missing"])
@pytest.mark.parametrize("purpose", ["Commuting", "missing"])
@pytest.mark.parametrize("market_share", ["Car_high", "missing"])
def test_filter_elasticity_file(
    elasticity_file: Tuple[Path, pd.DataFrame],
    elasticity_type: str,
    purpose: str,
    market_share: str,
):
    """Test the `read_elasticity_file` filtering with a DataFrame.

    Parameters
    ----------
    elasticity_file : Tuple[Path, pd.DataFrame]
        Path to the testing elasticity file and the DataFrame for comparison.
    elasticity_type : str
        Value to filter on the 'ElasticityType' column.
    purpose : str
        Value to filter on the 'Purp' column.
    market_share : str
        Value to filter on the 'MarketShare' column.

    See Also
    --------
    `test_read_elasticity_file` for testing of reading the CSV file.
    """
    # Test for missing values
    err_msg = None
    if elasticity_type == "missing":
        err_msg = "Value 'missing' not found in column 'ElasticityType'"
    elif purpose == "missing":
        err_msg = "Value 'missing' not found in column 'Purp'"
    elif market_share == "missing":
        err_msg = "Value 'missing' not found in column 'MarketShare'"
    if err_msg:
        with pytest.raises(ValueError) as e:
            read_elasticity_file(
                elasticity_file[1], elasticity_type, purpose, market_share
            )
        assert e.value.args[0] == err_msg
    else:
        df = read_elasticity_file(
            elasticity_file[1], elasticity_type, purpose, market_share
        )
        # Filter comparison dataframe for test
        comp_df = elasticity_file[1].copy()
        for col, val in [
            ("ElasticityType", elasticity_type),
            ("Purp", purpose),
            ("MarketShare", market_share),
        ]:
            comp_df = comp_df.loc[comp_df[col] == val]
        pd.testing.assert_frame_equal(df, comp_df, check_dtype=False)
