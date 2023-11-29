"""Config files and options for `run_forecast`."""
import enum
from pathlib import Path
from typing import Any, Optional, Union

import pydantic

import normits_demand as nd
from normits_demand.utils import config_base, ntem_extractor
from normits_demand.core import enumerations as nd_enum


EXPORT_PATH_FORMAT = (
    "{export_folder}/Forecasting/{forecast_model}/"
    "{forecast_version}/iter{iteration}/{scenario}/{mode}"
)


class ForecastModel(nd_enum.IsValidEnumWithAutoNameLower):
    """Forecasting models available."""

    NTEM = enum.auto()
    TRIP_END = enum.auto()
    EFS = enum.auto()
    EDGE = enum.auto()
    DLOG = enum.auto()


class ForecastParameters(config_base.BaseConfig):
    """Base class for storing the parameters for running forecasting."""

    iteration: str
    import_path: Path
    assignment_model: nd.AssignmentModel
    base_year: int
    future_years: list[int]
    export_folder: Path
    export_path_fmt: str = EXPORT_PATH_FORMAT
    matrix_import_path: Path
    hb_purposes_needed: list[int]
    nhb_purposes_needed: list[int]
    comparison_zone_systems: dict[str, str]
    time_periods: list[int]
    user_classes: list[str]
    output_trip_end_data: bool = False
    output_trip_end_growth: bool = False

    def _build_export_path(
        self, forecast_model: str, forecast_version: str, forecast_scenario: str
    ) -> Path:
        """Build export path from `export_path_fmt`."""
        return Path(
            self.export_path_fmt.format(
                forecast_model=forecast_model,
                forecast_version=forecast_version,
                scenario=forecast_scenario,
                mode=self.assignment_model.get_mode().get_name(),
                **self.dict(),
            )
        )

    @property
    def export_path(self) -> Path:
        """Read-only path to export folder.

        This is built from the `export_path_fmt` with variables filled
        in from the class attributes.
        """
        return self._build_export_path("UNKNOWN", "UNKNOWN", "UNKNOWN")

    @property
    def pa_to_od_factors(self) -> dict[str, Path]:
        """Dict[str, Path]
        Paths to the PA to OD tour proportions, has keys
        `post_me_tours`, `post_me_fh_th_factors`, `pre_me_tours`
        and `pre_me_fh_th_factors`.
        """
        # TODO(MB) Add flexibility to how it finds the tour proportions
        tour_prop_home = (
            self.import_path / self.assignment_model.name.lower() / "post_me_tour_proportions"
        )

        pre_me_tour_home = (
            self.import_path / self.assignment_model.name.lower() / "pre_me_tour_proportions"
        )
        paths = {
            "post_me_tours": tour_prop_home,
            "post_me_fh_th_factors": tour_prop_home / "fh_th_factors",
            "pre_me_tours": pre_me_tour_home,
            "pre_me_fh_th_factors": pre_me_tour_home / "fh_th_factors",
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


class TEMDataParameters(pydantic.BaseModel):  # pylint: disable=no-member
    """Trip end data parameters for forecasting."""

    base_tripend_path: Path
    tem_iteration: str
    tem_scenario: nd.Scenario
    tem_input_zoning: str = "msoa"

    @staticmethod
    def _build_tripend_path(base: Path, tripend_iteration: str, scenario: nd.Scenario) -> Path:
        return base / f"iter{tripend_iteration}/{scenario.value}"

    @pydantic.root_validator(skip_on_failure=True)
    def _check_tripend_path(  # pylint: disable=no-self-argument
        cls, values: dict[str, Any]
    ) -> dict[str, Any]:
        """Check the trip end folder exists."""
        base: Path = values["base_tripend_path"]
        iteration: str = values["tem_iteration"]
        scenario: nd.Scenario = values["tem_scenario"]

        path = cls._build_tripend_path(base, iteration, scenario)
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
        return self._build_tripend_path(
            self.base_tripend_path, self.tem_iteration, self.tem_scenario
        )


class TEMForecastParameters(ForecastParameters):
    """Class for storing parameters for trip end forecasting."""

    data_parameters: TEMDataParameters
    forecasting_model_version: str
    forecasting_model_name: str

    @property
    def export_path(self) -> Path:
        """Read-only path to export folder.

        This is built from the `export_path_fmt` with variables filled
        in from the class attributes.
        """
        return self._build_export_path(
            self.forecasting_model_name, self.forecasting_model_version, self.tem_scenario.name
        )


class NTEMDataParameters(pydantic.BaseModel):  # pylint: disable=no-member
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

    data_parameters: NTEMDataParameters

    @property
    def export_path(self) -> Path:
        """Read-only path to export folder.

        This is built from the `export_path_fmt` with variables filled
        in from the class attributes.
        """
        if self.data_parameters.version <= 7.2:
            scenario = "Core"
        elif self.data_parameters.scenario is not None:
            scenario = self.data_parameters.scenario
        else:
            raise ValueError("expected scenario for NTEM version > 7.2")

        return self._build_export_path("NTEM", str(self.data_parameters.version), scenario)


class DLOGForecastParameters(ForecastParameters):
    """Parameters for DLOG forecasting."""

    background_trip_end_parameters: Union[TEMDataParameters, NTEMDataParameters]
    dlog_trip_end_parameters: TEMDataParameters
    forecasting_model_version: str
    forecasting_model_name: str
    # TODO(MB) replace this with a folder containing cost matrices by purpose and mode
    cost_matrix_path: pydantic.FilePath  # pylint: disable=no-member
    cost_distribution_folder: pydantic.DirectoryPath  # pylint: disable=no-member

    @property
    def scenario_name(self) -> str:
        """Name of background growth scenario."""
        if isinstance(self.background_trip_end_parameters, TEMDataParameters):
            scenario = self.background_trip_end_parameters.tem_scenario.value
        elif isinstance(self.background_trip_end_parameters, NTEMDataParameters):
            scenario = self.background_trip_end_parameters.scenario
            if scenario is None:
                scenario = "NTEM"
        else:
            raise TypeError(
                f"unexpected type ({type(self.background_trip_end_parameters)}) "
                "for background_trip_end_parameters"
            )

        return scenario + "-DLOG"

    @property
    def export_path(self) -> Path:
        """Read-only path to export folder.

        This is built from the `export_path_fmt` with variables filled
        in from the class attributes.
        """
        return self._build_export_path(
            self.forecasting_model_name, self.forecasting_model_version, self.scenario_name
        )


class EDGEParameters(config_base.BaseConfig):
    """Parameters for the EDGE forecasting model."""

    forecast_years: dict[int, Any]

    # Used to build output path
    iteration: str
    forecast_version: str = "1.0"
    forecast_scenario: str
    export_folder: Path
    assignment_model: nd.AssignmentModel = nd.AssignmentModel.NORMS

    # Input folders
    matrices_to_grow_dir: Path

    # Input files
    demand_segments: Path
    flow_cat_path: Path
    ticket_type_splits_path: Path
    norms_to_edge_stns_path: Path
    cube_exe: Path

    # EDGE file
    edge_flows_path: Path

    # EDGE outputs
    edge_growth_dir: Path

    _export_path_fmt: str = pydantic.PrivateAttr(EXPORT_PATH_FORMAT)

    @classmethod
    @pydantic.validator("edge_growth_dir", "matrices_to_grow_dir")
    def _check_folder(cls, value: Path) -> Path:
        if not value.is_dir():
            raise ValueError(f"folder doesn't exist: {value}")
        return value

    @classmethod
    @pydantic.validator(
        "demand_segments",
        "flow_cat_path",
        "ticket_type_splits_path",
        "norms_to_edge_stns_path",
        "edge_flows_path",
        "cube_exe",
    )
    def _check_file(cls, value: Path) -> Path:
        if not value.is_file():
            raise ValueError(f"file doesn't exist: {value}")
        return value

    @pydantic.root_validator(skip_on_failure=True)
    def _check_factors_path(  # pylint: disable=no-self-argument
        cls, values: dict[str, Any]
    ) -> dict[str, Any]:
        """Check the factors Path."""
        path: Path = values["edge_growth_dir"]
        if not path.is_dir():
            raise ValueError(f"EDGE factors folder doesn't exist: {path}")

        return values

    @pydantic.root_validator(skip_on_failure=True)
    def _check_factors_file(  # pylint: disable=no-self-argument
        cls, values: dict[str, Any]
    ) -> dict[str, Any]:
        """Check the factors file."""
        for forecast_year in values["forecast_years"]:
            path: Path = Path(values["edge_growth_dir"]) / values["forecast_years"][
                forecast_year
            ].format(**values)
            if not path.is_file():
                raise ValueError(f"EDGE factors file doesn't exist: {path}")

        return values

    @property
    def export_path(self) -> Path:
        """Build export path from `export_path_fmt`."""
        return Path(
            self._export_path_fmt.format(
                forecast_model=ForecastModel.EDGE.value,
                scenario=self.forecast_scenario,
                mode=self.assignment_model.get_mode().get_name(),
                **self.dict(),
            )
        )


# TODO(MB) Function to create an example config file
