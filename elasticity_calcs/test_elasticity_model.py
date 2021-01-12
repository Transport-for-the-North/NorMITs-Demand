# -*- coding: utf-8 -*-
"""
    Unit tests for the elasticity_model module using pytest.
"""

##### IMPORTS #####
# Standard imports
import shutil
from pathlib import Path
from typing import Tuple, Dict

# Third party imports
import numpy as np
import pandas as pd
import pytest

# Local imports
from .elasticity_model import ElasticityModel


##### CONSTANTS #####
CAR_ZONES = [1, 2, 3]
RAIL_ZONES = [1, 2]
CAR_COSTS = pd.DataFrame(
    {
        "from_model_zone_id": np.repeat(CAR_ZONES, len(CAR_ZONES)),
        "to_model_zone_id": np.tile(CAR_ZONES, len(CAR_ZONES)),
        "time": 3000,
        "toll": 0,
        "distance": 25000,
    }
)
RAIL_COSTS = pd.DataFrame(
    {
        "from_model_zone_id": np.repeat(RAIL_ZONES, len(RAIL_ZONES)),
        "to_model_zone_id": np.tile(RAIL_ZONES, len(RAIL_ZONES)),
        "AE_cost": 5,
        "fare_cost": 420,
        "IVT_cost": 30,
        "Wait_Actual_cost": 10,
        "Interchange_cost": 2,
    }
)
CAR_DEMAND = pd.DataFrame(1, columns=CAR_ZONES, index=CAR_ZONES)
RAIL_DEMAND = pd.DataFrame(1, columns=RAIL_ZONES, index=RAIL_ZONES)
CONSTRAINT = np.full_like(RAIL_DEMAND.values, 1)
ELASTICITIES_FILE = (
    Path(__file__).parent / "test_elastcities_commuting_moderate.csv"
)
YEARS = ["2018"]
# Answers taken from manual calculations within Excel
CAR_ANSWER = pd.DataFrame(
    0.897845275494197, columns=CAR_ZONES, index=CAR_ZONES
)
RAIL_ANSWER = pd.DataFrame(
    0.990900752128458, columns=RAIL_ZONES, index=RAIL_ZONES
)
OTHER_ANSWER = pd.DataFrame(
    {
        "mode": ["bus", "active", "no_travel"],
        "mean_demand_adjustment": [
            1.21590935775892,
            0.946596709199525,
            1.43053565783266,
        ],
    }
)

##### CLASSES #####
class TestElasticityModel:
    """Tests for the `ElasticityModel` class."""

    ELASTICITY_PARAMS = {
        "purpose": "Commuting",
        "market_share": "CarToRail_Moderate",
    }
    DEMAND_PARAMS = {
        "trip_origin": "hb",
        "matrix_format": "pa",
        "year": YEARS[0],
        "purpose": "1",
        "segment": "1",
    }
    APPLY_ELASTICITIES_RETURNS = [
        ("car", CAR_ANSWER),
        ("ca1", RAIL_ANSWER),
        ("ca2", RAIL_ANSWER),
        *[
            (i, OTHER_ANSWER.loc[OTHER_ANSWER["mode"] == i])
            for i in ("bus", "active", "no_travel")
        ],
    ]
    LOOKUP = pd.DataFrame(
        {
            "noham_zone_id": [1, 2, 2, 3],
            "norms_zone_id": [1, 1, 2, 2],
            "split": [1, 0.8, 0.2, 1],
        }
    )
    CHECK_TOLERANCE = 1e-4

    @pytest.fixture(name="folders", scope="class")
    def fixture_folders(self, tmp_path_factory: Path) -> Tuple[Path, Path]:
        """Creates input files in temp folder for testing."""
        folders = [
            "elasticity",
            "translation",
            "rail_demand",
            "car_demand",
            "rail_costs",
            "car_costs",
        ]
        input_folders = {i: tmp_path_factory.mktemp(i) for i in folders}
        # Elasticities and contraint files
        shutil.copy(
            ELASTICITIES_FILE,
            input_folders["elasticity"] / "elasticity_values.csv",
        )
        constraint_folder = input_folders["elasticity"] / "constraint_matrices"
        constraint_folder.mkdir()
        np.savetxt(
            constraint_folder / "all_trips.csv", CONSTRAINT, delimiter=","
        )
        # Demand files
        base_demand = "{trip_origin}_{matrix_format}_yr{year}_p{purpose}_m{mode}_soc{segment}"
        CAR_DEMAND.to_csv(
            input_folders["car_demand"]
            / (base_demand.format(**self.DEMAND_PARAMS, mode=1) + ".csv")
        )
        for i in 1, 2:
            filename = (
                base_demand.format(**self.DEMAND_PARAMS, mode=6)
                + f"_ca{i}.csv"
            )
            RAIL_DEMAND.to_csv(input_folders["rail_demand"] / filename)
        # Cost files
        cost_file = f"{{}}_costs_p{self.DEMAND_PARAMS['purpose']}.csv"
        for df, nm in ((RAIL_COSTS, "rail"), (CAR_COSTS, "car")):
            df.to_csv(
                input_folders[f"{nm}_costs"] / cost_file.format(nm),
                index=False,
            )
        # Lookup file
        lookup_file = "noham_to_norms.csv"
        self.LOOKUP.to_csv(
            input_folders["translation"] / lookup_file, index=False
        )

        output_folder = tmp_path_factory.mktemp("outputs")
        return input_folders, output_folder

    @pytest.fixture(name="output_demand", scope="class")
    def fixture_output_demand(
        self, folders: Tuple[Path, Path]
    ) -> Dict[str, pd.DataFrame]:
        """Returns outputs from `ElasticityModel.apply_elasticities` for multiple tests."""
        elast_model = ElasticityModel(*folders, YEARS)
        return elast_model.apply_elasticities(
            self.DEMAND_PARAMS, self.ELASTICITY_PARAMS
        )

    @pytest.mark.parametrize("mode,answer", APPLY_ELASTICITIES_RETURNS)
    def test_apply_elasticities_return(
        self,
        output_demand: Dict[str, pd.DataFrame],
        mode: str,
        answer: pd.DataFrame,
    ):
        """Tests if `ElasticityModel.apply_elasticities` returns correct values."""
        if mode in ("car", "ca1", "ca2"):
            pd.testing.assert_frame_equal(
                answer, output_demand[mode], atol=self.CHECK_TOLERANCE
            )
        else:
            assert (
                abs(answer.iloc[0, 1] - np.mean(output_demand[mode]))
                < self.CHECK_TOLERANCE
            )

    @pytest.mark.parametrize(
        "mode,answer",
        [("car", CAR_ANSWER), ("rail", RAIL_ANSWER), ("others", OTHER_ANSWER)],
    )
    def test_apply_elasticities_output(
        self,
        folders: Tuple[Path, Path],
        # Argument required to make sure output_demand fixture produces outputs
        output_demand,  # pylint: disable=unused-argument
        mode: str,
        answer: pd.DataFrame,
    ):
        """Tests if `ElasticityModel.apply_elasticities` produces correct files."""
        _, output_folder = folders
        expected = 2 if mode == "rail" else 1
        out_files = list((output_folder / mode).iterdir())
        msg = f"{len(out_files)} output files produced, expected {expected}"
        assert len(out_files) == expected, msg
        for i in out_files:
            ind = None if mode == "others" else 0
            out = pd.read_csv(i, index_col=ind)
            if mode != "others":
                out.columns = out.columns.astype(int)
            pd.testing.assert_frame_equal(
                answer, out, check_dtype=False, atol=self.CHECK_TOLERANCE
            )
