"""Config files and options for `run_forecast`."""
import enum
from pathlib import Path
from typing import Any, Optional

import pydantic

import normits_demand as nd
from normits_demand.utils import config_base, ntem_extractor
from normits_demand.core import enumerations as nd_enum


class ForecastModel(nd_enum.IsValidEnumWithAutoNameLower):
    """Forecasting models available."""

    NTEM = enum.auto()
    TRIP_END = enum.auto()
    EFS = enum.auto()


class ForecastParameters(config_base.BaseConfig):
    """Base class for storing the parameters for running forecasting."""

    iteration: str
    import_path: Path
    assignment_model: nd.AssignmentModel
    base_year: int
    future_years: list[int]
    export_folder: Path
    export_path_fmt: str = "{export_folder}/{model_name}/iter{iteration}"
    matrix_import_path: Path
    hb_purposes_needed: list[int]
    nhb_purposes_needed: list[int]
    comparison_zone_systems: dict[str, str]
    time_periods: list[int]
    user_classes: list[str]
    output_trip_end_data: bool = False
    output_trip_end_growth: bool = False

    @property
    def export_path(self) -> Path:
        """Read-only path to export folder.

        This is built from the `export_path_fmt` with variables filled
        in from the class attributes.
        """
        return Path(
            self.export_path_fmt.format(
                model_name=self.assignment_model.get_name(), **self.dict()
            )
        )

    @property
    def pa_to_od_factors(self) -> dict[str, Path]:
        """Dict[str, Path]
        Paths to the PA to OD tour proportions, has
        keys `post_me_tours` and `post_me_fh_th_factors`.
        """
        # TODO(MB) Add flexibility to how it finds the tour proportions
        tour_prop_home = Path(
            self.import_path
            / self.assignment_model.get_name().lower()
            / "post_me_tour_proportions"
        )
        paths = {
            "post_me_tours": tour_prop_home,
            "post_me_fh_th_factors": tour_prop_home / "fh_th_factors",
        }
        for nm, p in paths.items():
            if not p.is_dir():
                raise NotADirectoryError(f"cannot find {nm} folder: {p}")
        return paths

    @property
    def car_occupancies_path(self) -> Path:
        """Path
        Path to the vehicle occupancies CSV file.
        """
        path = self.import_path / "vehicle_occupancies/car_vehicle_occupancies.csv"
        if not path.exists():
            raise FileNotFoundError(f"cannot find vehicle occupancies CSV: {path}")
        return path


class TEMForecastParameters(ForecastParameters):
    """Class for storing parameters for trip end forecasting."""

    base_tripend_path: Path
    tem_scenario: nd.Scenario
    export_path_fmt: str = (
        "{export_folder}/{model_name}/TEM/iter{iteration}/{tem_scenario_name}"
    )

    @pydantic.root_validator(skip_on_failure=True)
    def _check_tripend_path(  # pylint: disable=no-self-argument
        cls, values: dict[str, Any]
    ) -> dict[str, Any]:
        """Check the trip end folder exists."""
        base: Path = values["base_tripend_path"]
        scenario: nd.Scenario = values["tem_scenario"]
        path = base / str(scenario.value)
        if not path.is_dir():
            raise ValueError(
                f"tripend folder doesn't exist: {path}, make sure the "
                "base_tripend_path folder contains a sub-folder with the name "
                f"defined by the tem_scenario parameter ({scenario.value})"
            )

        return values

    @property
    def tripend_path(self) -> Path:
        """Folder containing trip end data."""
        return self.base_tripend_path / self.tem_scenario.value

    @property
    def export_path(self) -> Path:
        """Read-only path to export folder.

        This is built from the `export_path_fmt` with variables filled
        in from the class attributes.
        """
        return Path(
            self.export_path_fmt.format(
                model_name=self.assignment_model.get_name(),
                tem_scenario_name=self.tem_scenario.value,
                **self.dict(),
            )
        )


class NTEMDataParameters(pydantic.BaseModel):
    """Parameters for defining what NTEM data to use."""

    data_path: Path
    version: float = ntem_extractor.TemproParser._ntem_version
    scenario: Optional[str] = None

    @pydantic.validator("scenario")
    def _check_scenario(  # pylint: disable=no-self-argument
        cls, value: Optional[str]
    ) -> Optional[str]:
        if value is not None and value not in ntem_extractor.TemproParser._scenario_list:
            scenarios = ", ".join(f"'{s}'" for s in ntem_extractor.TemproParser._scenario_list)
            raise ValueError(f"scenario should be one of {scenarios}, not '{value}'")

        return value

    @pydantic.root_validator(skip_on_failure=True)
    def _check_version_and_scenario(  # pylint: disable=no-self-argument
        cls, values: dict[str, Any]
    ) -> dict[str, Any]:
        if values["version"] > 7.2 and values.get("scenario", None) is None:
            raise ValueError("scenario required for NTEM versions > 7.2")

        return values


class NTEMForecastParameters(ForecastParameters):
    """Class for storing parameters for NTEM forecasting."""

    ntem_parameters: NTEMDataParameters
    export_path_fmt: str = "{export_folder}/{model_name}/NTEM/{ntem_version}/iter{iteration}"

    @property
    def export_path(self) -> Path:
        """Read-only path to export folder.

        This is built from the `export_path_fmt` with variables filled
        in from the class attributes.
        """
        if self.ntem_parameters.version > 7.2:
            version = f"v{self.ntem_parameters.version}-{self.ntem_parameters.scenario}"
        else:
            version = f"v{self.ntem_parameters.version}"

        return Path(
            self.export_path_fmt.format(
                model_name=self.assignment_model.get_name(),
                ntem_version=version,
                **self.dict(),
            )
        )


# TODO(MB) Function to create an example config file
