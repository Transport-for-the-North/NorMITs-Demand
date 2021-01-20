# -*- coding: utf-8 -*-
"""
    Module containing the unit tests for the elasticity_calcs.utils module.
"""

##### IMPORTS #####
# Standard imports
from pathlib import Path
from typing import Tuple, List

# Third party imports
import pytest
import pandas as pd
import numpy as np

# Local imports
from elasticity_calcs import utils as eu


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
        "No_travel",
        "Car",
        "Bus",
        "Rail",
        "Active",
        "Car",
        "Bus",
    ],
    "OwnElast": [
        -0.3,
        0.06,
        0.07,
        0.02,
        0.01,
        0.16,
        -0.8,
        0.17,
        0.02,
        -0.3,
        0.06,
    ],
}


##### CLASSES #####
class TestGetConstraintMatrices:
    """Tests for the `get_constraint_matrices` function. """

    NON_CSV_FILES = [Path("test.txt")]
    CSV_FILES = [Path(f"{i}.csv") for i in range(3)]
    MATRICES = [np.random.rand(3, 3) for i in range(3)]

    @pytest.fixture(name="constraint_folder", scope="class")
    def fixture_constraint_folder(
        self, tmp_path_factory
    ) -> Tuple[Path, List[Path]]:
        """Create matrix CSV files for testing.

        Parameters
        ----------
        tmp_path_factory : Path
            Temporary folder provided by pytest.

        Returns
        -------
        Path
            Path to the folder containing the test files.
        List[Path]
            Absolute paths of all the files that should be found.
        """
        folder = tmp_path_factory.mktemp("constraint_matrices")
        # Write non_csv files
        for path in self.NON_CSV_FILES:
            with open(folder / path, "wt"):
                pass

        # Write matrices for CSV files
        paths = []
        for i, path in enumerate(self.CSV_FILES):
            path = (folder / path).absolute()
            paths.append(path)
            with open(path, "wt") as f:
                np.savetxt(f, self.MATRICES[i], delimiter=",")
        return folder, paths

    def test_get_constraint_matrices(
        self, constraint_folder: Tuple[Path, List[Path]]
    ):
        """Test that `get_constaint_matrices` finds the correct file paths.

        Parameters
        ----------
        constraint_folder : Tuple[Path, List[Path]]
            Folder containing the testing contraint matrices and a list of
            all the expected paths.
        """
        matrices = eu.get_constraint_matrices(constraint_folder[0])
        # Check non_csv files aren't present
        for path in self.NON_CSV_FILES:
            assert path.stem.lower() not in matrices

        # Check csv files are present
        for path in constraint_folder[1]:
            assert matrices[path.stem.lower()] == path

    def test_read_constraint_matrices(
        self, constraint_folder: Tuple[Path, List[Path]]
    ):
        """Test that `get_contraint_matrices` reads the files correctly.

        Parameters
        ----------
        constraint_folder : Tuple[Path, List[Path]]
            Folder containing the testing contraint matrices and a list of
            all the expected paths.
        """
        names = [i.stem for i in self.CSV_FILES[:2]]
        matrices = eu.get_constraint_matrices(constraint_folder[0], names)
        assert names == list(matrices.keys())
        for i, n in enumerate(names):
            np.testing.assert_array_equal(matrices[n], self.MATRICES[i])

    @staticmethod
    @pytest.mark.parametrize("test", ["file", "nothing"])
    def test_get_constraint_matrices_error(tmp_path, test):
        """Test the `get_constraint_matrices` function produces expected errors.

        Parameters
        ----------
        tmp_path : Path
            Temporary folder path provided by pytest.
        test : str
            What test to run.
        """
        if test == "file":
            path = tmp_path / "test.csv"
            with open(path, "wt"):
                pass
        else:
            path = tmp_path / "folder"
        with pytest.raises(FileNotFoundError):
            eu.get_constraint_matrices(path)


class TestReadDemandMatrix:
    """Tests for the `read_demand_matrix` function."""

    DEMAND_NORMS = pd.DataFrame(
        np.full((2, 2), 10.0), columns=[1, 2], index=[1, 2]
    )
    DEMAND_NOHAM = pd.DataFrame(
        np.arange(1, 10, dtype=float).reshape(3, 3),
        columns=[1, 2, 3],
        index=[1, 2, 3],
    )
    LOOKUP = pd.DataFrame(
        {
            "noham_zone_id": [1, 2, 2, 3],
            "norms_zone_id": [1, 1, 2, 2],
            "split": [1, 0.8, 0.2, 1],
        }
    )
    NORMS_NAME = "norms_demand.csv"
    NOHAM_NAME = "noham_demand.csv"

    @pytest.fixture(name="demand_folder", scope="class")
    def fixture_demand_folder(self, tmp_path_factory: Path) -> Path:
        """Create test demand matrix in temporary folder.

        Parameters
        ----------
        tmp_path_factory : Path
            Temporary folder provided by pytest.

        Returns
        -------
        Path
            Folder containing test matrix.
        """
        folder = tmp_path_factory.mktemp("demand")
        self.DEMAND_NORMS.to_csv(folder / self.NORMS_NAME)
        self.DEMAND_NOHAM.to_csv(folder / self.NOHAM_NAME)
        self.LOOKUP.to_csv(folder / "noham_to_norms.csv")
        return folder

    def test_read(self, demand_folder: Path):
        """Test that a NoRMS matrix is read correctly without converting zones.

        Parameters
        ----------
        demand_folder : Path
            The folder containing the test matrix.
        """
        out, _, _ = eu.read_demand_matrix(
            demand_folder / self.NORMS_NAME, demand_folder, "norms"
        )
        pd.testing.assert_frame_equal(out, self.DEMAND_NORMS)

    def test_lookup(self, demand_folder: Path):
        """Test that a NoHAM matrix is read in correctly and converted zones.

        Check that NoHAM demand it is correctly converted to NoRMS zone
        system and check it can be converted back to NoHAM zone system
        correctly.

        Parameters
        ----------
        demand_folder : Path
            The folder containing the test matrix.
        """
        test, reverse, _ = eu.read_demand_matrix(
            demand_folder / self.NOHAM_NAME, demand_folder, "noham"
        )
        answer = pd.DataFrame(
            [[9.0, 9.0], [15.0, 12.0]], index=[1, 2], columns=[1, 2]
        )
        pd.testing.assert_frame_equal(test, answer)

        # Convert back to NoHAM zone system
        od = ["o", "d"]
        lookup_cols = ["noham_zone_id", "norms_zone_id"]
        test.index.name = od[0]
        test.columns.name = od[1]
        test = test.unstack().reset_index().rename(columns={0: "trips"})
        right_on = [f"{lookup_cols[1]}-{i}" for i in od]
        test = test.merge(
            reverse, left_on=od, right_on=right_on, how="left", validate="1:m"
        )
        test["trips"] = test["trips"] * test["split"]
        test = test.drop(columns=od + right_on + ["split"]).rename(
            columns={f"{lookup_cols[0]}-{i}": i for i in od}
        )
        test = test.groupby(od, as_index=False).sum()
        test = test.pivot(index=od[0], columns=od[1], values="trips")
        pd.testing.assert_frame_equal(
            test, self.DEMAND_NOHAM, check_names=False
        )


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
    df = eu.read_segments_file(segments_file[0])
    pd.testing.assert_frame_equal(df, segments_file[1], check_dtype=False)


@pytest.mark.parametrize(
    "func, args",
    [(eu.read_segments_file, []), (eu.read_elasticity_file, ["", "", ""])],
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
    df = eu.read_elasticity_file(elasticity_file[0])
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
            eu.read_elasticity_file(
                elasticity_file[1], elasticity_type, purpose, market_share
            )
        assert e.value.args[0] == err_msg
    else:
        df = eu.read_elasticity_file(
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
