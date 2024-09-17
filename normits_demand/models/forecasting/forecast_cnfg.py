"""Config files and options for `run_forecast`."""
import enum
import os
from pathlib import Path
from typing import Any, Optional, Union
import pydantic
import caf.toolkit

import normits_demand as nd
from normits_demand.utils import ntem_extractor
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


class ForecastParameters(caf.toolkit.BaseConfig):
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
        self,
        forecast_model: str,
        forecast_version: str,
        forecast_scenario: str,
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
                raise NotADirectoryError(
                    f"cannot find {nm} folder: {p}"
                )
        return paths

    @property
    def car_occupancies_path(self) -> Path:
        """Path
        Path to the vehicle occupancies CSV file.
        """
        path = (
            self.import_path
            / "vehicle_occupancies/car_vehicle_occupancies.csv"
        )
        if not path.exists():
            raise FileNotFoundError(
                f"cannot find vehicle occupancies CSV: {path}"
            )
        return path


class TEMForecastParameters(ForecastParameters):
    """Class for storing parameters for trip end forecasting."""

    base_tripend_path: Path
    tem_iteration: str
    tem_scenario: nd.Scenario
    forecasting_model_version: str
    forecasting_model_name: str

    @staticmethod
    def _build_tripend_path(
        base: Path, tripend_iteration: str, scenario: nd.Scenario
    ) -> Path:
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
            self.base_tripend_path,
            self.tem_iteration,
            self.tem_scenario,
        )

    @property
    def export_path(self) -> Path:
        """Read-only path to export folder.

        This is built from the `export_path_fmt` with variables filled
        in from the class attributes.
        """
        return self._build_export_path(
            self.forecasting_model_name,
            self.forecasting_model_version,
            self.tem_scenario.name,
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
        if (
            value is not None
            and value not in ntem_extractor.TemproParser._scenario_list
        ):
            scenarios = ", ".join(
                f"'{s}'"
                for s in ntem_extractor.TemproParser._scenario_list
            )
            raise ValueError(
                f"scenario should be one of {scenarios}, not '{value}'"
            )

        return value

    @pydantic.root_validator(skip_on_failure=True)
    def _check_version_and_scenario(  # pylint: disable=no-self-argument
        cls, values: dict[str, Any]
    ) -> dict[str, Any]:
        if (
            values["version"] > 7.2
            and values.get("scenario", None) is None
        ):
            raise ValueError(
                "scenario required for NTEM versions > 7.2"
            )

        return values


class NTEMForecastParameters(ForecastParameters):
    """Class for storing parameters for NTEM forecasting."""

    ntem_parameters: NTEMDataParameters

    @property
    def export_path(self) -> Path:
        """Read-only path to export folder.

        This is built from the `export_path_fmt` with variables filled
        in from the class attributes.
        """
        if self.ntem_parameters.version <= 7.2:
            scenario = "Core"
        elif self.ntem_parameters.scenario is not None:
            scenario = self.ntem_parameters.scenario
        else:
            raise ValueError("expected scenario for NTEM version > 7.2")

        return self._build_export_path(
            "NTEM", str(self.ntem_parameters.version), scenario
        )


class TicketSplitParams(caf.toolkit.BaseConfig):
    """
    Everything needed to generate ticket splitting factors.

    Parameters
    ----------

    edge_flows_path: Path
        Path to edge flows file
    flow_cat_path: Path
        Path to flow cat file
    splits_path: Path
        Dir containing TicketTypeSplits.csv as well as stn2stn_costs csvs (for each time period). This doubles as the cache
        for processed ticket type splits, which are saved as a pickle file accompanied by a yml of this class.
    info: str
        Hard coded info about the corresponding pickle file. Can be changed by the user if wanted.
    """

    edge_flows_path: Path
    flow_cat_path: Path
    splits_path: Path
    info: Optional[str]
    
    @pydantic.validator("splits_path")
    def _check_contents(cls, value):
        files = os.listdir(value)
        if "TicketTypeSplits.csv" not in files:
            raise ValueError("splits_path must contain a file called TicketTypeSplits.csv, containing ticket type split data.")
        for i in files:
            if '_stn2stn_costs.csv' in i:
                return value
        raise ValueError("splits_path must contain files called {tp}_stn2stn_costs.csv, where tp are time periods (usually 'AM', 'IP', 'PM', 'OP')")


class EDGEParameters(caf.toolkit.BaseConfig):
    """Parameters for the EDGE forecasting model.

    forecast_years: Self explanatory

    iteration: used in naming dir
    forecast_version: used in naming dir
    forecast_scenario: used in naming dir
    export_folder: where export dir is built
    assignment_model: default to NoRMs

    matrices_to_grow_dir: folder containing matrices to be grown

    demand_segments: path to csv containing demand segments
    ticket_type_splits: If this has been generated already provide a path to it,
    if not provide parameters specified in TicketSplitParameters.
    norms_to_edge_stns_path: path to csv containing lookup


    """

    forecast_years: dict[int, Any]

    # Used to build output path
    iteration: str
    forecast_version: str = "1.0"
    forecast_scenario: str
    export_folder: Path
    assignment_model: nd.AssignmentModel = nd.AssignmentModel.NORMS

    # Input folders
    matrices_to_grow_dir: Path
    irsj_props_dir: Path

    # Input files
    demand_segments: Path
    ticket_type_splits: TicketSplitParams
    norms_to_edge_stns_path: Path
    cube_exe: Path

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
            raise ValueError(
                f"EDGE factors folder doesn't exist: {path}"
            )

        return values

    @pydantic.root_validator(skip_on_failure=True)
    def _check_factors_file(  # pylint: disable=no-self-argument
        cls, values: dict[str, Any]
    ) -> dict[str, Any]:
        """Check the factors file."""
        for forecast_year in values["forecast_years"]:
            path: Path = Path(values["edge_growth_dir"]) / values[
                "forecast_years"
            ][forecast_year].format(**values)
            if not path.is_file():
                raise ValueError(
                    f"EDGE factors file doesn't exist: {path}"
                )

        return values

    @property
    def export_path(self) -> Path:
        """Build export path from `export_path_fmt`."""
        path = Path(
            self._export_path_fmt.format(
                forecast_model=ForecastModel.EDGE.value,
                scenario=self.forecast_scenario,
                mode=self.assignment_model.get_mode().get_name(),
                **self.dict(),
            )
        )
        path.mkdir(exist_ok=True, parents=True)
        return path


# TODO(MB) Function to create an example config file
