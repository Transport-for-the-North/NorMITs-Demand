# -*- coding: utf-8 -*-
"""Run script for disaggregating model matrices into other segmentations."""

##### IMPORTS #####
import logging
import pathlib
import re
import sys
from typing import Any, Optional

import pydantic

# Add parent folder to path
sys.path.append("..")
# Allow running from parent folder e.g. python run_tools\run_matrix_decompilation.py
sys.path.append(".")
# pylint: disable=import-error,wrong-import-position
import normits_demand as nd
from normits_demand import logging as nd_log, constants
from normits_demand.converters import traveller_segmentation_trip_ends
from normits_demand.distribution import segment_disaggregator
from normits_demand.matrices import matrix_processing
from normits_demand.utils import config_base, file_ops

# pylint: enable=import-error,wrong-import-position

##### CONSTANTS #####
LOG_FILE = "Traveller_segmentation.log"
LOG = nd_log.get_logger(nd_log.get_package_logger_name() + ".run_traveller_segmentation")
CONFIG_PATH = pathlib.Path("config/Traveller_segmentation_parameters.yml")
MODEL_SEGMENTATIONS = {
    nd.TripOrigin.HB: "notem_hb_output_uc",
    nd.TripOrigin.NHB: "notem_nhb_output_uc",
}


##### CLASSES #####
class TravellerSegmentationParameters(config_base.BaseConfig):
    """Parameters and config for the Traveller Segmentation tool.

    Attributes
    ----------
    iteration : str
        Name of the traveller segmentation iteration being ran,
        used for naming the output folder.
    base_output_folder : pathlib.Path
        Base folder for saving outputs to, a new sub-folder will
        be created using the `iteration` name.
    notem_export_home : pathlib.Path
        Base folder for the trip ends outputs.
    notem_iteration : str
        Iteration of the trip end model to use.
    scenario : nd.Scenario
        Trip end scenario to use.
    matrix_folder : pathlib.Path
        Folder containing the input matrices.
    model : nd.AssignmentModel
        Assignment model of the input matrices.
    matrix_zoning : str
        Zoning system that the matrices are in, usually `model` zone
        system.
    year : int
        Year of trip ends and matrices to use.
    disaggregation_output_segment : DisaggregationOutputSegment
        Additional segment to disaggregate matrices to.
    cost_folder : pathlib.Path
        Folder containing the cost matrices which should be the same
        segmentation as the input matrices.
    trip_length_distribution_folder : pathlib.Path
        Folder containing the TLDs which should be the same segmentation
        as the matrices are being disaggregated to.
    trip_length_distribution_units : nd.CostUnits, default KILOMETERS
        Units the trip length distributions and cost matrices are in.
    aggregate_time_periods : list[int], optional
        List of time periods to aggregate together for the input
        matrices, if not given time periods aren't aggregated.
    disaggregation_settings : DisaggregationSettings, optional
        Custom settings for the disaggregation process.
    """

    iteration: str
    base_output_folder: pathlib.Path
    notem_export_home: pathlib.Path
    notem_iteration: str
    scenario: nd.Scenario
    matrix_folder: pathlib.Path
    model: nd.AssignmentModel
    matrix_zoning: str
    year: int
    disaggregation_output_segment: segment_disaggregator.DisaggregationOutputSegment
    cost_folder: pathlib.Path
    trip_length_distribution_folder: pathlib.Path
    trip_length_distribution_units: nd.CostUnits = nd.CostUnits.KILOMETERS
    aggregate_time_periods: Optional[list[int]] = None
    disaggregation_settings: segment_disaggregator.DisaggregationSettings = (
        segment_disaggregator.DisaggregationSettings()
    )

    @pydantic.validator(
        "matrix_folder",
        "notem_export_home",
        "trip_length_distribution_folder",
        "base_output_folder",
        "cost_folder",
        allow_reuse=True,
    )
    def _folder_exists(cls, value) -> pathlib.Path:  # pylint: disable=no-self-argument
        try:
            return file_ops.folder_exists(value)
        except NotADirectoryError as err:
            raise ValueError(err) from err

    @pydantic.validator("matrix_zoning")
    def _check_zone_system(cls, value: str) -> str:  # pylint: disable=no-self-argument
        value = value.lower()
        try:
            _ = nd.get_zoning_system(value)
        except nd.NormitsDemandError as err:
            raise ValueError(err) from err

        return value

    @pydantic.validator("aggregate_time_periods", pre=True)
    def _check_time_periods(cls, value: str) -> Optional[str]:
        """Convert empty or none / null strings to None."""
        none_str = {"", "none", "null", "no"}
        if isinstance(value, str) and value.strip().lower() in none_str:
            return None
        if isinstance(value, list) and value == []:
            return None

        return value

    @property
    def iteration_folder(self) -> pathlib.Path:
        """Iteration output folder."""
        iteration_folder = self.base_output_folder / f"iter{self.iteration}"
        iteration_folder.mkdir(exist_ok=True)
        return iteration_folder

    @property
    def output_folder(self) -> pathlib.Path:
        """Output folder for specific disaggregation segment."""
        output_folder = self.iteration_folder / str(self.disaggregation_output_segment.value)
        output_folder.mkdir(exist_ok=True)
        return output_folder


##### FUNCTIONS #####
def _compile_params_format(
    compile_params_path: pathlib.Path, aggregate_time_periods: bool
) -> None:
    """Replace ca/nca in compiled matrix names with ca2/ca1, respectively.

    Overwrites existing compilation parameters file with adjusted names.

    Parameters
    ----------
    compile_params_path : pathlib.Path
        Path to compilation parameters file created by
        `matrix_processing.build_compile_params`.
    aggregate_time_periods : bool
        Whether to remote time period for compiled file name and
        combine all time periods together.
    """

    def replace(match: re.Match) -> str:
        for val, i in (("ca", 2), ("nca", 1)):
            if match.group(1).lower() == val:
                return f"_ca{i}{match.group(2)}"

        # This should never occur
        raise ValueError(f"unexpected match value: {match!r}")

    compile_params = file_ops.read_df(compile_params_path)
    compile_params.loc[:, "compilation"] = compile_params["compilation"].str.replace(
        r"_(ca|nca)([_.])", replace, flags=re.I, regex=True
    )

    if aggregate_time_periods:
        compile_params.loc[:, "compilation"] = compile_params["compilation"].str.replace(
            r"_tp\d+", "", regex=True
        )

    file_ops.write_df(compile_params, compile_params_path, index=False)


def aggregate_purposes(
    matrix_folder: pathlib.Path,
    model: nd.AssignmentModel,
    year: int,
    aggregate_time_periods: Optional[list[int]] = None,
) -> pathlib.Path:  # TODO Update docstring with tp parameter
    """Aggregate matrices in NTEM purposes to model user classes.

    Parameters
    ----------
    matrix_folder : pathlib.Path
        Folder containing the matrices by NTEM purpose.
    model : nd.AssignmentModel
        Assignment model for the matrices.
    year : int
        Model year.

    Returns
    -------
    pathlib.Path
        Folder where the aggregated user class matrices are saved,
        creates new sub-folder 'userclasses' inside `matrix_folder`.

    Raises
    ------
    NotImplementedError
        For any model other than NoRMS.
    """
    if model != nd.AssignmentModel.NORMS:
        raise NotImplementedError(f"aggregate_purposes not implemented for {model.get_name()}")

    LOG.info("Compiling %s matrices to userclasses", model.get_name())
    LOG.debug("Input matrices: %s", matrix_folder)

    output_folder = matrix_folder / "userclass"
    output_folder.mkdir(exist_ok=True)

    required = None
    if aggregate_time_periods is not None:
        required = ["tp"]

    # Check whether matrices contain time period and car availability segmentation
    ca_needed = set()
    tp_needed = set()
    for params in file_ops.parse_folder_files(
        matrix_folder, constants.VALID_MAT_FTYPES, required
    ):
        if "tp" in params:
            tp_needed.add(int(params["tp"]))
        if "ca" in params:
            ca_needed.add(int(params["ca"]))
    ca_needed = list(ca_needed) if len(ca_needed) > 0 else None
    tp_needed = list(tp_needed) if len(tp_needed) > 0 else None

    if aggregate_time_periods is not None:
        tp_needed = aggregate_time_periods

    compile_params_path = pathlib.Path(
        matrix_processing.build_compile_params(
            import_dir=matrix_folder,
            export_dir=output_folder,
            matrix_format="pa",
            years_needed=[year],
            m_needed=model.get_mode().get_mode_values(),
            ca_needed=ca_needed,
            tp_needed=tp_needed,
            split_hb_nhb=True,
        )[0]
    )

    _compile_params_format(compile_params_path, aggregate_time_periods is not None)

    matrix_processing.compile_matrices(
        mat_import=matrix_folder,
        mat_export=output_folder,
        compile_params_path=compile_params_path,
        overwrite=False,
    )
    LOG.info("Output user class matrices saved: %s", output_folder)

    return output_folder


def main(params: TravellerSegmentationParameters, init_logger: bool = True) -> None:
    """Run traveller segmentation tool.

    Parameters
    ----------
    params : TravellerSegmentationParameters
        Parameters for running the tool.
    init_logger : bool, default True
        Whether or not to initialise a log file.
    """
    # TODO For now use NoRMS syntheic Full PA aggregating to commute, business and other
    # TODO In future use EFS decompile post ME function to convert from NORMS/NOHAM to TMS segmentation

    if init_logger:
        # Add log file output to main package logger
        nd_log.get_logger(
            nd_log.get_package_logger_name(),
            params.output_folder / LOG_FILE,
            "Running Traveller Segmentation Disaggregator",
            log_version=True,
        )
        nd_log.capture_warnings(
            file_handler_args=dict(log_file=params.output_folder / LOG_FILE)
        )

    LOG.info("Outputs saved to: %s", params.output_folder)
    out = params.output_folder / "Traveller_segmentation_parameters.yml"
    parameters.save_yaml(out)
    LOG.info("Input parameters saved to: %s", out)
    LOG.debug("Input parameters:\n%s", params.to_yaml())

    trip_end_converter = traveller_segmentation_trip_ends.NoTEMToTravellerSegmentation(
        output_zoning=nd.get_zoning_system(params.matrix_zoning),
        base_year=params.year,
        scenario=params.scenario,
        notem_iteration_name=params.notem_iteration,
        export_home=params.notem_export_home,
        cache_dir=params.iteration_folder / ".cache",
    )
    LOG.info(
        "Trip ends used:\nHB productions: %s\nHB attractions: %s\n"
        "NHB productions: %s\nNHB attractions: %s\nZoning system: %s\n"
        "Time format: %s\nHB segmentation: %s\nNHB segmentation: %s",
        trip_end_converter.hb_productions_path,
        trip_end_converter.hb_attractions_path,
        trip_end_converter.nhb_productions_path,
        trip_end_converter.nhb_attractions_path,
        trip_end_converter.output_zoning.name,
        trip_end_converter.time_format.name,
        MODEL_SEGMENTATIONS[nd.TripOrigin.HB],
        MODEL_SEGMENTATIONS[nd.TripOrigin.NHB],
    )
    LOG.info(
        "Other inputs:\nCosts: %s\nTrip length distributions: %s",
        params.cost_folder,
        params.trip_length_distribution_folder,
    )

    matrix_folder = aggregate_purposes(
        params.matrix_folder, params.model, params.year, params.aggregate_time_periods
    )

    for to in nd.TripOrigin:
        LOG.info(
            "Splitting %s matrices to %s segmentation",
            to.value,
            params.disaggregation_output_segment.value,
        )

        productions, attractions = trip_end_converter.get_trip_ends(
            to, nd.get_segmentation_level(MODEL_SEGMENTATIONS[to])
        )

        segment_disaggregator.disaggregate_segments(
            import_folder=matrix_folder,
            target_tld_folder=params.trip_length_distribution_folder,
            tld_units=params.trip_length_distribution_units,
            model=params.model,
            base_productions=productions,
            base_attractions=attractions,
            export_folder=params.output_folder,
            cost_folder=params.cost_folder,
            disaggregation_segment=params.disaggregation_output_segment,
            trip_origin=to,
            settings=params.disaggregation_settings,
        )
        LOG.info("Finished %s", to.value)

    LOG.info("Finished traveller segmentation")


def example_config(path: pathlib.Path) -> None:
    """Writes an example of the input config YAML file to `path`."""

    class ExampleTSP(TravellerSegmentationParameters):
        """New sub-class which turns of path validation for writing example config."""

        @pydantic.validator(
            "matrix_folder",
            "notem_export_home",
            "trip_length_distribution_folder",
            "base_output_folder",
            allow_reuse=True,
        )  # pylint: disable=no-self-argument
        def _folder_exists(cls, value) -> pathlib.Path:
            return value

        @pydantic.root_validator(skip_on_failure=True, allow_reuse=True)
        def _check_cost_folder(cls, values: dict[str, Any]) -> dict[str, Any]:
            return values

    example = ExampleTSP(
        iteration="1",
        base_output_folder="path/to/output/folder",
        notem_export_home="path/to/NoTEM/base/export/folder",
        notem_iteration="1",
        scenario=nd.Scenario.SC01_JAM,
        matrix_folder="path/to/folder/containing/matrices/for/segmentation",
        model=nd.AssignmentModel.NORMS,
        year=2018,
        disaggregation_output_segment=segment_disaggregator.DisaggregationOutputSegment.SOC,
        trip_length_distribution_folder="path/to/tld/folder",
    )

    example.save_yaml(path)
    print(f"Written example config to: {path}")


##### MAIN #####
if __name__ == "__main__":
    try:
        # TODO Add command line argument to specify config path,
        # with default as CONFIG_PATH if no arguments are given
        parameters = TravellerSegmentationParameters.load_yaml(CONFIG_PATH)
    except (pydantic.ValidationError, NotADirectoryError) as err:
        LOG.critical("Config file error: %s", err)
        raise SystemExit(1) from err

    try:
        main(parameters)
    except Exception as err:
        LOG.critical("Traveller segmentation disaggregator error:", exc_info=True)
        raise
