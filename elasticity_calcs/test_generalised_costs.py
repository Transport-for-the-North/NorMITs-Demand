# -*- coding: utf-8 -*-
"""
    Module containing all the unit tests for the elasticty_calcs module.
"""

##### IMPORTS #####
# Standard imports
from typing import Dict, Tuple
from pathlib import Path

# Third party imports
import pytest
import numpy as np
import pandas as pd

# Local imports
from .generalised_costs import (
    _average_matrices,
    _check_matrices,
    gen_cost_car_mins,
    gen_cost_rail_mins,
    RAIL_GC_FACTORS,
    gen_cost_elasticity_mins,
    get_costs,
    read_gc_parameters,
)


##### CLASSES #####
class TestAverageMatrices:
    """Tests for the `_average_matrices` function. """

    MATRIX = np.array([[38, 93, 45], [88, 13, 10], [34, 96, 69]])
    WEIGHTS = np.array([[27, 21, 94], [36, 77, 62], [2, 22, 93]])

    @staticmethod
    def test_missing_matrices():
        """Test that the correct error is raised if matrices are missing when calling. """
        with pytest.raises(KeyError) as e:
            _average_matrices({"test": np.zeros((3, 3))}, ["test", "test2"])
        msg = "The following matrices are missing: ['test2']"
        assert e.value.args[0] == msg

    @pytest.mark.parametrize(
        "weights,answer",
        [
            (None, 54),
            (np.full((3, 3), 1.0), 54),
            (WEIGHTS, 47.453917050691246),
        ],
    )
    def test_average(self, weights: np.array, answer: float):
        """Test the `_average_matrices` function calculates averages correctly.

        Tests for weighted and non-weighted averages.
        """
        averages = _average_matrices(
            {"test": self.MATRIX}, ["test"], weights=weights
        )
        assert averages == {"test": answer}


class TestCheckMatrices:
    """Tests for the `_check_matrices` function. """

    @staticmethod
    def test_missing_matrices():
        """Test that the correct error is raised when matrices are missing. """
        with pytest.raises(KeyError) as e:
            _check_matrices({"test": np.zeros((2, 2))}, ["test", "test2"])
        msg = "The following matrices are missing: ['test2']"
        assert e.value.args[0] == msg

    @staticmethod
    def test_different_shapes():
        """Test that the correct error is raised when matrices are different shapes. """
        with pytest.raises(ValueError) as e:
            _check_matrices(
                {"test": np.zeros((2, 2)), "test2": np.zeros((3, 3))},
                ["test", "test2"],
            )
        msg = "Matrices are not all the same shape: test = (2, 2), test2 = (3, 3)"
        assert e.value.args[0] == msg


class TestGenCostCarMins:
    """Tests for the `gen_costs_car_mins` function. """

    TIME_MAT = np.array([[98.16, 92.38], [5.87, 43.54]])
    DIST_MAT = np.array([[473.99, 517.76], [103.66, 416.25]])
    TOLL_MAT = np.array([[4.53, 4.81], [6.57, 6.56]])
    VC = 7.39
    VT = 6.33

    def test_missing_matrices(self):
        """Test that the correct error is raised if matrices are missing. """
        with pytest.raises(KeyError) as e:
            gen_cost_car_mins({"time": self.TIME_MAT}, self.VC, self.VT)
        msg = "The following matrices are missing: ['dist', 'toll']"
        assert e.value.args[0] == msg

    def test_different_shapes(self):
        """Test that the correct error is raised if matrices aren't the same shape. """
        with pytest.raises(ValueError) as e:
            gen_cost_car_mins(
                {
                    "time": self.TIME_MAT,
                    "dist": self.DIST_MAT,
                    "toll": np.zeros((3, 3)),
                },
                self.VC,
                self.VT,
            )
        msg = (
            "Matrices are not all the same shape: "
            "time = (2, 2), dist = (2, 2), toll = (3, 3)"
        )
        assert e.value.args[0] == msg

    def test_calculation(self):
        """Test that the calculation produces the correct values. """
        test = gen_cost_car_mins(
            {
                "time": self.TIME_MAT,
                "dist": self.DIST_MAT,
                "toll": self.TOLL_MAT,
            },
            self.VC,
            self.VT,
        )
        answer = np.array(
            [
                [2.905002543443918, 2.9040025908372824],
                [1.2567665718799368, 2.2479553712480254],
            ]
        )
        np.testing.assert_array_equal(
            test, answer, err_msg="Incorrect calculation for gen_cost_car_mins"
        )


class TestGenCostRailMins:
    """Tests for the `gen_cost_rail_mins` function. """

    MATRICES = {
        "walk": np.array([[5.13, 5.14], [5.31, 0.80]]),
        "wait": np.array([[7.73, 2.07], [3.83, 2.83]]),
        "ride": np.array([[18.85, 0.59], [72.52, 38.11]]),
        "fare": np.array([[113, 103], [275, 459]]),
        "num_int": np.array([[1, 0], [0, 2]]),
    }
    VT = 83.07
    TEST_FACTORS = [
        None,
        {"walk": 1.75},
        {"walk": 1, "interchange_penalty": 10},
    ]
    TEST_ANSWERS = [
        np.array(
            [
                [48.36529854339714, 13.67991814132659],
                [91.45546105693992, 60.49546045503792],
            ]
        ),
        np.array(
            [
                [49.64779854339714, 14.96491814132659],
                [92.78296105693993, 60.695460455037924],
            ]
        ),
        np.array(
            [
                [50.80029854339713, 11.109918141326592],
                [88.80046105693992, 70.09546045503792],
            ]
        ),
    ]

    def test_missing_matrices(self):
        """Test that the correct error is raised if matrices are missing. """
        with pytest.raises(KeyError) as e:
            gen_cost_rail_mins({}, self.VT)
        msg = (
            "The following matrices are missing: "
            "['walk', 'wait', 'ride', 'fare', 'num_int']"
        )
        assert e.value.args[0] == msg

    def test_different_shapes(self):
        """Test that the correct error is raised if matrices aren't the same shape. """
        matrices = self.MATRICES.copy()
        matrices["fare"] = np.zeros((3, 3))
        with pytest.raises(ValueError) as e:
            gen_cost_rail_mins(matrices, self.VT)
        msg = (
            "Matrices are not all the same shape: "
            "walk = (2, 2), wait = (2, 2), ride = (2, 2), "
            "fare = (3, 3), num_int = (2, 2)"
        )
        assert e.value.args[0] == msg

    @pytest.mark.parametrize("factors,answer", zip(TEST_FACTORS, TEST_ANSWERS))
    def test_calculation(self, factors: Dict[str, float], answer: np.array):
        """Tests the calculation with different weighting factors. """
        factors = RAIL_GC_FACTORS if factors is None else factors
        np.testing.assert_array_equal(
            gen_cost_rail_mins(self.MATRICES, self.VT, factors), answer
        )


class TestGetCosts:
    """Tests for the `get_costs` function. """

    COSTS = {
        "car": pd.DataFrame(
            {
                "from_model_zone_id": [1, 1, 2, 2],
                "to_model_zone_id": [1, 2, 1, 2],
                "time": [0.16, 0.32, 6.55, 9.81],
                "distance": [3.81, 3.9, 7.59, 6.09],
                "toll": [9.34, 4.24, 9.75, 7.94],
            }
        ),
        "rail": pd.DataFrame(
            {
                "from_model_zone_id": range(10),
                "to_model_zone_id": range(10),
                "AE_cost": np.random.rand(10),
                "Wait_Actual_cost": np.random.rand(10),
                "IVT_cost": np.random.rand(10),
                "fare_cost": np.random.rand(10),
                "Interchange_cost": np.random.rand(10),
            }
        ),
    }
    COSTS["missing_car"] = COSTS["car"].drop(columns="toll")
    NOHAM_2_NORMS = pd.DataFrame(
        {"noham_zone_id": [1, 2], "norms_zone_id": [1, 1]}
    )
    NOHAM_DEMAND = pd.DataFrame(
        np.full((2, 2), 10.0), columns=[1, 2], index=[1, 2]
    )
    # Both costs are renamed and car is rezoned
    CONVERTED_COSTS = {
        "car": pd.DataFrame(
            {
                "origin": [1],
                "destination": [1],
                "time": [16.84],
                "dist": [21.39],
                "toll": [31.27],
            }
        ),
        "rail": COSTS["rail"].rename(
            columns={
                "from_model_zone_id": "origin",
                "to_model_zone_id": "destination",
                "AE_cost": "walk",
                "Wait_Actual_cost": "wait",
                "IVT_cost": "ride",
                "fare_cost": "fare",
                "Interchange_cost": "num_int",
            }
        ),
    }

    @pytest.fixture(name="costs", scope="class")
    def fixture_costs(self, tmp_path_factory) -> Dict[str, Path]:
        """Create temporary test folder containing cost files.

        Parameters
        ----------
        tmp_path_factory :
            Temporary test folder provided by pytest.

        Returns
        -------
        Dict[str, Path]
            Paths to the car and rail cost files.
        Path
            Path to the folder containing the cost lookup.
        """
        folder = tmp_path_factory.mktemp("costs")
        paths = {}
        for nm, df in self.COSTS.items():
            paths[nm] = folder / f"{nm}.csv"
            df.to_csv(paths[nm], index=False)

        lookup_file = folder / "noham_to_norms.csv"
        self.NOHAM_2_NORMS.to_csv(lookup_file, index=False)
        return paths, folder

    @pytest.mark.parametrize(
        "mode, zone_system", [("car", "noham"), ("rail", "norms")]
    )
    def test_read(
        self, costs: Tuple[Dict[str, Path], Path], mode: str, zone_system: str
    ):
        """Test that the function reads the costs correctly.

        Parameters
        ----------
        costs : Tuple[Dict[str, Path], Path]
            Paths to the car and rail cost files and the zone translation folder.
        mode : str
            Which mode to test, either car or rail.
        """
        test = get_costs(
            costs[0][mode],
            mode,
            zone_system,
            costs[1],
            self.NOHAM_DEMAND,
        )
        pd.testing.assert_frame_equal(
            test, self.CONVERTED_COSTS[mode], check_dtype=False
        )

    @staticmethod
    def test_missing(costs: Tuple[Dict[str, Path], Path]):
        """Test that a KeyError is raised if a column is missing from file.

        Parameters
        ----------
        costs : Tuple[Dict[str, Path], Path]
            Paths to the car and rail cost files and the zone translation folder.
        """
        with pytest.raises(ValueError) as e:
            get_costs(costs[0]["missing_car"], "car", None, None)
        msg = "Columns missing from car cost, columns expected but not found: ['toll']"
        assert e.value.args[0] == msg


class TestGenCostMode:  # TODO Implement tests for gen_cost_mode
    """Tests for the `gen_cost_mode` function."""

    @staticmethod
    @pytest.mark.skip(reason="Placeholder for `gen_cost_mode` test.")
    def test_calculation():
        pass


class TestCalculateGenCosts:  # TODO implement tests for calculate_gen_costs
    """Tests for the `calculate_gen_costs` function."""

    @staticmethod
    @pytest.mark.skip(reason="Placeholder for the `calculate_gen_costs` test.")
    def test_calculation():
        pass


class TestReadGCParameters:
    """Tests for the `read_gc_parameters` function."""

    PARAM_FILE = pd.DataFrame(
        {
            "year": ["2018", "2018", "2030", "2030"],
            "mode": ["car", "rail"] * 2,
            "vot": [16.2, 16.4, 17.2, 17.4],
            "voc": [9.45, np.nan, 10.45, np.nan],
        }
    )
    PARAMS = {
        "2018": {
            "car": {"vot": 16.2, "voc": 9.45},
            "rail": {"vot": 16.4},
        },
        "2030": {
            "car": {"vot": 17.2, "voc": 10.45},
            "rail": {"vot": 17.4},
        },
    }
    YEARS = ["2018", "2030"]
    MODES = ["car", "rail"]

    @pytest.fixture(name="param_path", scope="class")
    def fixture_param_path(self, tmp_path_factory) -> Tuple[Path, Path]:
        """Write generalised cost parameters test files.

        Parameters
        ----------
        tmp_path_factory : Path
            Temporary folder path provided by pytest.

        Returns
        -------
        Path
            Path to the correct input file.
        Path
            Path to the input file with missing data.
        """
        folder = tmp_path_factory.mktemp("gc_params")
        correct = folder / "correct_gc_params.csv"
        self.PARAM_FILE.to_csv(correct, index=False)
        missing = folder / "missing_gc_params.csv"
        self.PARAM_FILE.iloc[[0]].to_csv(missing, index=False)
        return correct, missing

    def test_correct(self, param_path: Tuple[Path, Path]):
        """Tests that the GC parameters file is read correctly.

        Parameters
        ----------
        param_path : Tuple[Path, Path]
            Paths to the correct and missing files, respectively.
        """
        test = read_gc_parameters(param_path[0], self.YEARS, self.MODES)
        assert test == self.PARAMS, "Correct GC params input file"

    def test_missing(self, param_path: Tuple[Path, Path]):
        """Test that the correct error is raised when data is missing.

        Parameters
        ----------
        param_path : Tuple[Path, Path]
            Paths to the correct and missing files, respectively.
        """
        with pytest.raises(ValueError) as e:
            read_gc_parameters(param_path[1], self.YEARS, self.MODES)
        msg = (
            "Years missing: ['2030'] Year - mode pairs missing: "
            "['2018 - rail'] from: missing_gc_params.csv"
        )
        print(repr(str(e.value.args[0])))
        assert e.value.args[0] == msg, "Missing GC parameters file"


##### FUNCTIONS #####
@pytest.mark.parametrize(
    "cost_factor,answer",
    [(None, 0.26649522280649446), (2.0, 0.13324761140324723)],
)
def test_gen_cost_elasticity_mins(cost_factor: float, answer: float):
    """Test that the `gen_cost_elasticity_mins` calulation is correct with various factors. """
    test_params = {
        "elasticity": 0.8,
        "gen_cost": np.array([[2.8, 1.67], [4.49, 0.1]]),
        "cost": np.array([[7.72, 9.29], [6.92, 5.9]]),
        "demand": np.array([[42.66, 83.66], [77.28, 31.84]]),
    }
    test = gen_cost_elasticity_mins(**test_params, cost_factor=cost_factor)
    np.testing.assert_array_equal(
        test, answer, "Incorrect answer for `gen_cost_elasticity_mins`"
    )
