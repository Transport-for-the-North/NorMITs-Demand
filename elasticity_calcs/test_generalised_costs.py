# -*- coding: utf-8 -*-
"""
    Module containing all the unit tests for the elasticty_calcs module.
"""

##### IMPORTS #####
# Standard imports
from typing import Dict
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
        assert e.value.args[0] == "The following matrices are missing: ['test2']"

    @pytest.mark.parametrize(
        "weights,answer",
        [(None, 54), (np.full((3, 3), 1.0), 54), (WEIGHTS, 47.453917050691246)],
    )
    def test_average(self, weights: np.array, answer: float):
        averages = _average_matrices({"test": self.MATRIX}, ["test"], weights=weights)
        assert averages == {"test": answer}


class TestCheckMatrices:
    """Tests for the `_check_matrices` function. """

    @staticmethod
    def test_missing_matrices():
        """Test that the correct error is raised when matrices are missing. """
        with pytest.raises(KeyError) as e:
            _check_matrices({"test": np.zeros((2, 2))}, ["test", "test2"])
        assert e.value.args[0] == "The following matrices are missing: ['test2']"

    @staticmethod
    def test_different_shapes():
        """Test that the correct error is raised when matrices are different shapes. """
        with pytest.raises(ValueError) as e:
            _check_matrices(
                {"test": np.zeros((2, 2)), "test2": np.zeros((3, 3))}, ["test", "test2"]
            )
        msg = "Matrices are not all the same shape: " "test = (2, 2), test2 = (3, 3)"
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
        assert e.value.args[0] == "The following matrices are missing: ['dist', 'toll']"

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
            {"time": self.TIME_MAT, "dist": self.DIST_MAT, "toll": self.TOLL_MAT},
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
    TEST_FACTORS = [None, {"walk": 1, "interchange_penalty": 10}]
    TEST_ANSWERS = [
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
                "from_model_zone_id": range(10),
                "to_model_zone_id": range(10),
                "time": np.random.rand(10),
                "distance": np.random.rand(10),
                "toll": np.random.rand(10),
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
        """
        folder = tmp_path_factory.mktemp("costs")
        paths = {}
        for nm, df in self.COSTS.items():
            paths[nm] = folder / f"{nm}.csv"
            df.to_csv(paths[nm], index=False)
        return paths

    @pytest.mark.parametrize("mode", ["car", "rail"])
    def test_read(self, costs: Dict[str, Path], mode: str):
        """Test that the function reads the costs correctly.

        Parameters
        ----------
        costs : Dict[str, Path]
            Paths to the car and rail cost files.
        mode : str
            Which mode to test, either car or rail.
        """
        pd.testing.assert_frame_equal(get_costs(costs[mode], mode), self.COSTS[mode])

    @staticmethod
    def test_missing(costs: Dict[str, Path]):
        """Test that a KeyError is raised if a column is missing from file.

        Parameters
        ----------
        costs : Dict[str, Path]
            Paths to the car, rail and missing_car cost files.
        """
        with pytest.raises(ValueError) as e:
            get_costs(costs["missing_car"], "car")
        msg = "Columns missing from car cost, columns expected but not found: ['toll']"
        assert e.value.args[0] == msg


##### FUNCTIONS #####
@pytest.mark.parametrize(
    "cost_factor,answer", [(None, 0.26649522280649446), (2.0, 0.13324761140324723)]
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
