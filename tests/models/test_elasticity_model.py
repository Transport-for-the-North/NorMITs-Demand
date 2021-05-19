# -*- coding: utf-8 -*-
"""
    Unit tests for the elasticity_model module using pytest.
"""
# ## IMPORTS ## #
# Standard imports
import shutil
from pathlib import Path
from typing import Tuple, Dict, List

# Third party imports
import numpy as np
import pandas as pd
import pytest

# Local imports
from normits_demand.elasticity import constants as ec
from normits_demand.models import elasticity_model as em
from normits_demand.elasticity import generalised_costs as gc


# ## CONSTANTS ## #
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
CONSTRAINT_ALL = ("all_trips", np.full_like(RAIL_DEMAND.values, 1))
CONSTRAINT_ZEROS = ("no_trips", np.zeros_like(RAIL_DEMAND.values))
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
COST_CHANGES = pd.DataFrame(
    {
        "year": "2018",
        "elasticity_type": [
            "Car_JourneyTime",
            "Car_JourneyTime",
            "Car_FuelCost",
            "Rail_Fare",
            "Rail_IVTT",
            "Bus_Fare",
            "Bus_IVTT",
            "Car_RUC",
        ],
        "constraint_matrix_name": ["no_trips"] + ["all_trips"] * 7,
        "percentage_change": [10, 0.8, 1.2, 0.8, 0.8, 0.8, 0.8, 1.2],
    }
)

# ## CLASSES ## #
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
    ELASTICITY_SEGMENTS = pd.DataFrame(
        {
            "EFS_Seg": 1,
            "EFS_PurpBase": "hb",
            "EFS_MainPurp": "hb_w",
            "EFS_SubPurp": "commute",
            "EFS_SubPurpID": 1,
            "EFS_TimePeriod": "24hr",
            "EFS_SkillLevel": 1,
            "EFS_IncLevel": None,
            "Elast_Purp": "Commuting",
            "Elast_MarketShare": "CarToRail_Moderate",
        },
        index=[0],
    )
    GC_PARAMETERS = pd.DataFrame(
        {
            "year": ["2018", "2018"],
            "mode": ["car", "rail"],
            "vot": [16.58, 16.6],
            "voc": [9.45, np.nan],
        }
    )
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
    def fixture_folders(self,
                        tmp_path_factory: Path,
                        cost_changes: Path,
                        ) -> Tuple[Dict[str, Path],
                                   Dict[str, Path],
                                   Dict[str, Path]]:
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
        self.ELASTICITY_SEGMENTS.to_csv(
            input_folders["elasticity"] / "elasticity_segments.csv"
        )
        constraint_folder = input_folders["elasticity"] / "constraint_matrices"
        constraint_folder.mkdir()
        for nm, df in (CONSTRAINT_ALL, CONSTRAINT_ZEROS):
            np.savetxt(constraint_folder / f"{nm}.csv", df, delimiter=",")
        # Demand files
        base_demand = (
            "{trip_origin}_{matrix_format}_yr{year}_p{purpose}_m{mode}_soc{segment}"
        )
        CAR_DEMAND.to_csv(
            input_folders["car_demand"]
            / (base_demand.format(**self.DEMAND_PARAMS, mode=3) + ".csv")
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

        input_files = {}
        folder = tmp_path_factory.mktemp("input_files")
        # GC parameters file
        input_files["gc_parameters"] = folder / "gc_parameters.csv"
        self.GC_PARAMETERS.to_csv(input_files["gc_parameters"], index=False)

        input_files["cost_changes"] = cost_changes

        output_folder = tmp_path_factory.mktemp("outputs")
        modes = ("car", "rail", "others")
        output_folders = {m: output_folder / m for m in modes}
        return input_folders, input_files, output_folders

    @pytest.fixture(name="output_demand", scope="class")
    def fixture_output_demand(self,
                              folders: Tuple[Dict[str, Path], Dict[str, Path], Path],
                              ) -> Dict[str, pd.DataFrame]:
        """Returns outputs from `ElasticityModel.apply_elasticities` for multiple tests."""
        elast_model = em.ElasticityModel(*folders, YEARS)
        gc_params = gc.read_gc_parameters(
            folders[1]["gc_parameters"],
            YEARS,
            ["car", "rail"]
        )
        return elast_model.apply_elasticities(
            self.DEMAND_PARAMS,
            self.ELASTICITY_PARAMS,
            gc_params[YEARS[0]],
            em.read_cost_changes(folders[1]["cost_changes"], YEARS),
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
    def test_apply_elasticities_output(self,
                                       folders: Tuple[Path, Path],
                                       # Argument required to make sure output_demand fixture produces outputs
                                       output_demand,  # pylint: disable=unused-argument
                                       mode: str,
                                       answer: pd.DataFrame,
                                       ) -> None:
        """Tests if `ElasticityModel.apply_elasticities` produces correct files."""
        _, _, output_folders = folders
        expected = 2 if mode == "rail" else 1
        out_files = list((output_folders[mode]).iterdir())
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

    def test_apply_all(self,
                       folders: Tuple[Dict[str, Path], Dict[str, Path], Dict[str, Path]]
                       ) -> None:
        """Test `ElasticityModel.apply_all` method with a single segment."""
        elast_model = em.ElasticityModel(*folders, YEARS)
        elast_model.apply_all()

        output_folders = {
            "car": (1, CAR_ANSWER),
            "rail": (2, RAIL_ANSWER),
            "others": (1, OTHER_ANSWER),
        }
        for m, (num, ans) in output_folders.items():
            out_files = list((folders[2][m]).iterdir())
            # Check we got the right number of files back
            err_msg = (
                "The wrong number of files for mode %s were returned. "
                "Expected %d, got %d." % (m, num, len(out_files))
            )
            assert len(out_files) == num, err_msg

            for i in out_files:
                col = 0 if m != "others" else None
                test = pd.read_csv(i, index_col=col)
                if m != "others":
                    test.columns = pd.to_numeric(
                        test.columns, downcast="integer"
                    )
                pd.testing.assert_frame_equal(
                    test, ans, atol=self.CHECK_TOLERANCE
                )


##### FUNCTIONS #####
@pytest.fixture(name="cost_changes", scope="module")
def fixture_cost_changes(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Write cost changes test file.

    Parameters
    ----------
    tmp_path_factory : Path
        Temporary path provided by pytest.

    Returns
    -------
    Path
        Path to the cost changes CSV file.
    """
    folder = tmp_path_factory.mktemp("cost_changes")
    path = folder / "cost_changes.csv"
    COST_CHANGES.to_csv(path, index=False)
    return path


@pytest.mark.parametrize("years", (["2018"], ["2018", "2019"]))
@pytest.mark.parametrize("missing_type", (True, False))
def test_read_cost_changes(monkeypatch,
                           cost_changes: Path,
                           years: List[str],
                           missing_type: bool
                           ) -> None:
    """Test that `read_cost_changes` reads the file and performs checks."""
    if missing_type:
        monkeypatch.delitem(ec.GC_ELASTICITY_TYPES, "Car_RUC")
        with pytest.raises(KeyError) as e:
            em.read_cost_changes(cost_changes, years)
        msg = (
            "Unknown elasticity_type: ['Car_RUC'], available types "
            "are: ['Car_JourneyTime', 'Car_FuelCost', 'Rail_Fare', "
            "'Rail_IVTT', 'Bus_Fare', 'Bus_IVTT']"
        )
        assert e.value.args[0] == msg, "Unknown elasticity type"
    elif years != ["2018"]:
        with pytest.raises(ValueError) as e:
            em.read_cost_changes(cost_changes, years)
        msg = "Cost change not present for years: ['2019']"
        assert e.value.args[0] == msg, "Missing year data"
    else:
        data = em.read_cost_changes(cost_changes, years)
        pd.testing.assert_frame_equal(data, COST_CHANGES)
