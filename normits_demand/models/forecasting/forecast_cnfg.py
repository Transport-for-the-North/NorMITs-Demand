"""Config files and options for `run_forecast`."""
import enum
from pathlib import Path
from typing import Dict

from normits_demand.utils import config_base
from normits_demand.core import enumerations as nd_enum


class ForecastModel(nd_enum.IsValidEnumWithAutoNameLower):
    """Forecasting models available."""

    NTEM = enum.auto()
    TRIP_END = enum.auto()
    EFS = enum.auto()


class ForecastParameters(config_base.BaseConfig):
    """Class for storing the parameters for running forecasting.

    Attributes
    ----------
    import_path : Path
        Path to the NorMITs demand imports.
    tempro_data_path : Path
        Path to the CSV containing the TEMPro data.
    model_name : str
        Name of the model.
    iteration : str
        Iteration number.
    export_path_fmt: str
        Format for the export path, used for building
        `export_path` property.
    export_path_params : Dict[str, Any]
        Dictionary containing any additional parameters
        for building the `export_path`.
    _export_path : Path
        Read-only path to export folder, this is built from
        the `export_path_fmt` with variables filled in from
        the class attributes, and additional optional values
        from `export_path_params`.
    tripend_path: Path
        Base path for the process to read trip-end data from for
        growth factors
    """

    iteration: str
    import_path: Path
    model_name: str
    base_year: int
    future_years: list[int]
    export_folder: Path
    export_path_fmt: str = "{export_folder}/{model_name}/iter{iteration}"
    tripend_path: Path
    matrix_import_path: Path
    hb_purposes_needed: list[int]
    nhb_purposes_needed: list[int]
    comparison_zone_systems: Dict[str, str]
    mode: Dict[str, int]
    time_periods: list[int]
    user_classes: list[str]

    @property
    def export_path(self) -> Path:
        """
        Read-only path to export folder, this is built from
        the `export_path_fmt` with variables filled in from
        the class attributes, and additional optional values
        from `export_path_params`.
        """
        return Path(self.export_path_fmt.format(**self.dict()))

    def __repr__(self) -> str:
        params = (
            "import_path",
            "export_path",
            "model_name",
            "iteration",
        )
        msg = f"{self.__class__.__name__}("
        for p in params:
            msg += f"\n\t{p}={getattr(self, p)!s}"
        for p, val in self.pa_to_od_factors.items():
            msg += f"\n\t{p}={val!s}"
        msg += "\n)"
        return msg

    @property
    def pa_to_od_factors(self) -> Dict[str, Path]:
        """Dict[str, Path]
        Paths to the PA to OD tour proportions, has
        keys `post_me_tours` and `post_me_fh_th_factors`.
        """
        tour_prop_home = Path(
            self.import_path / self.model_name / "synthetic_tour_proportions"
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


# TODO(MB) Function to create an example config file
