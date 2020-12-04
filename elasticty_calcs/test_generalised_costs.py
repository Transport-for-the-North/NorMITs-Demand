# -*- coding: utf-8 -*-
"""
    Module containing all the unit tests for the elasticty_calcs module.
"""

##### IMPORTS #####
# Standard imports
from typing import Dict

# Third party imports
import pytest
import numpy as np

# Local imports
from generalised_costs import (
    _average_matrices,
    gen_cost_car_mins,
    gen_cost_rail_mins,
    RAIL_GC_FACTORS,
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


class TestGenCostCarMins:
    """Tests for the `gen_costs_car_mins` function. """

    TIME_MAT = np.array(
        [
            [98.16714318, 92.38193922, 69.20872894],
            [5.87348448, 43.54886814, 30.68318242],
            [59.40738931, 89.52624922, 61.89571624],
        ]
    )
    DIST_MAT = np.array(
        [
            [473.99133543, 517.76911753, 358.41564964],
            [103.66007206, 416.25154913, 932.40644355],
            [519.39186653, 391.45504159, 171.02775983],
        ]
    )
    TOLL_MAT = np.array(
        [
            [4.53284724, 4.8140093, 8.47547131],
            [6.57339959, 6.56382552, 3.25431721],
            [3.5610022, 5.39221686, 0.10560566],
        ]
    )
    WEIGHTS = np.array(
        [
            [7.39231584, 62.03557062, 92.54597502],
            [69.86590459, 29.75576107, 0.91098305],
            [89.81080133, 19.12336751, 86.02823059],
        ]
    )
    VC = 7.39
    VT = 6.33

    def test_missing_matrices(self):
        """Test that the correct error is raised if matrices are missing when calling. """
        with pytest.raises(KeyError) as e:
            gen_cost_car_mins({"time": self.TIME_MAT}, self.VC, self.VT)
        assert e.value.args[0] == "The following matrices are missing: ['dist', 'toll']"

    @pytest.mark.parametrize(
        "weights,answer",
        [
            (None, 2.2832413648256646),
            (np.full((3, 3), 1.0), 2.2832413648256646),
            (WEIGHTS, 2.1478463049790033),
        ],
    )
    def test_calculation(self, weights, answer):
        """Test that the calculation produces the correct values.

        Tested with and without weights.

        Parameters
        ----------
        weights : np.array
            Weights to be passed to the calculation function.
        answer : float
            The expected answer from the function.
        """
        test = gen_cost_car_mins(
            {"time": self.TIME_MAT, "dist": self.DIST_MAT, "toll": self.TOLL_MAT},
            self.VC,
            self.VT,
            weights=weights,
        )
        assert test == answer


class TestGenCostRailMins:
    """Tests for the `gen_cost_rail_mins` function. """

    MATRICES = {
        "walk": np.array([[5.13540686, 5.14771908], [5.31881756, 0.80650462]]),
        "wait": np.array([[7.73676407, 2.07717374], [3.83748466, 2.83424161]]),
        "ride": np.array([[18.8577773, 0.59300399], [72.5284892, 38.11644619]]),
        "fare": np.array([[113, 103], [275, 459]]),
    }
    VT = 83.073

    def test_missing_matrices(self):
        """Test that the correct error is raised if matrices are missing when calling. """
        with pytest.raises(KeyError) as e:
            gen_cost_rail_mins({}, self.VT)
        msg = "The following matrices are missing: ['walk', 'wait', 'ride', 'fare']"
        assert e.value.args[0] == msg

    @pytest.mark.parametrize(
        "factors,answer",
        [
            (None, 50.804388563885524),
            ({"walk": 2.5, "wait": 3}, 58.00238860638553),
            ({"walk": 1, "interchange_penalty": 10}, 47.72780454138553),
        ],
    )
    @pytest.mark.parametrize("num_interchanges", [0, 1])
    def test_calculation(
        self, factors: Dict[str, float], answer: float, num_interchanges: int
    ):
        """Tests the calculation with different weighting factors. """
        factors = RAIL_GC_FACTORS if factors is None else factors
        if num_interchanges != 0:
            # Add interchange penalty onto answer
            nm = "interchange_penalty"
            interchange_penalty = (
                RAIL_GC_FACTORS[nm] if nm not in factors.keys() else factors[nm]
            )
            answer += num_interchanges * interchange_penalty
        assert (
            gen_cost_rail_mins(self.MATRICES, self.VT, factors, num_interchanges)
            == answer
        )
