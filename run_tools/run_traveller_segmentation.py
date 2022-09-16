# -*- coding: utf-8 -*-
"""Traveller segmentation tool for disaggregating model matrices into other segmentations."""

##### IMPORTS #####
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import pathlib
import sys
from typing import Iterator, Optional

import pydantic

# Add parent folder to path
sys.path.append("..")
# Allow running from parent folder e.g. python run_tools\run_matrix_decompilation.py
sys.path.append(".")
# pylint: disable=import-error,wrong-import-position
import normits_demand as nd
from normits_demand import logging as nd_log
from normits_demand.converters import traveller_segmentation_trip_ends
from normits_demand.distribution import segment_disaggregator
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
class TSMatrixParameters(pydantic.BaseModel):
    """Parameters for defining the Travaller segmentation tool input matrices.

    Attributes
    ----------
    folder: pathlib.Path
        Folder containing the input matrices.
    zoning_system_name: str
        Name of the zone system the matrices are in.
    segmentation_name: str
        Name of the segmentation that the matrices are in.
    aggregate_segmentation_name: str, optional
        Optional segmentation to aggregate the matrices to before
        dissaggregating.
    zoning_system : ZoningSystem
        Zone system the input matrices are in.
    segmentation : SegmentationLevel
        Segmentation the input matrices are.
    aggregate_segmentation : SegmentationLevel, optional
        Optional segmentation to aggregate the matrices to before
        dissaggregating.
    """

    folder: pathlib.Path
    zoning_system_name: str
    segmentation_name: str
    aggregate_segmentation_name: Optional[str] = None

    # Define private variables for the actual zone system and segmentation instances
    _zoning_system = pydantic.PrivateAttr(None)
    _segmentation = pydantic.PrivateAttr(None)
    _aggregate_segmentation = pydantic.PrivateAttr(None)

    @pydantic.validator("folder", allow_reuse=True)
    def _folder_exists(  # pylint: disable=no-self-argument
        cls, value: pathlib.Path
    ) -> pathlib.Path:
        try:
            return file_ops.folder_exists(value)
        except NotADirectoryError as err:
            raise ValueError(err) from err

    @pydantic.validator("zoning_system_name")
    def _get_zone_system(cls, value: str) -> str:  # pylint: disable=no-self-argument
        try:
            nd.get_zoning_system(value)
        except (FileNotFoundError, nd.NormitsDemandError) as err:
            raise ValueError(err) from err

        return value

    @pydantic.validator("segmentation_name", "aggregate_segmentation_name")
    def _get_segmentation(cls, value: str) -> str:  # pylint: disable=no-self-argument
        try:
            nd.get_segmentation_level(value)
        except (FileNotFoundError, nd.NormitsDemandError) as err:
            raise ValueError(err) from err

        return value

    @property
    def zoning_system(self) -> nd.ZoningSystem:
        """Zone system the input matrices are in."""
        if self._zoning_system is None:
            self._zoning_system = nd.get_zoning_system(self.zoning_system_name)
        return self._zoning_system

    @property
    def segmentation(self) -> nd.SegmentationLevel:
        """Segmentation the input matrices are."""
        if self._segmentation is None:
            self._segmentation = nd.get_segmentation_level(self.segmentation_name)
        return self._segmentation

    @property
    def aggregate_segmentation(self) -> Optional[nd.SegmentationLevel]:
        """Optional segmentation to aggregate the matrices to before dissaggregating."""
        if self.aggregate_segmentation_name is None:
            return None

        if self._aggregate_segmentation is None:
            self._aggregate_segmentation = nd.get_segmentation_level(
                self.aggregate_segmentation_name
            )
        return self._aggregate_segmentation


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
    model : nd.AssignmentModel
        Assignment model of the input matrices.
    year : int
        Year of trip ends and matrices to use.
    matrix_parameters : TSMatrixParameters
        Parameters for defining the input matrices.
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
    disaggregation_settings : DisaggregationSettings, optional
        Custom settings for the disaggregation process.
    """

    iteration: str
    base_output_folder: pathlib.Path
    notem_export_home: pathlib.Path
    notem_iteration: str
    scenario: nd.Scenario
    model: nd.AssignmentModel
    year: int
    matrix_parameters: TSMatrixParameters
    disaggregation_output_segment: segment_disaggregator.DisaggregationOutputSegment
    cost_folder: pathlib.Path
    trip_length_distribution_folder: pathlib.Path
    trip_length_distribution_units: nd.CostUnits = nd.CostUnits.KILOMETERS
    disaggregation_settings: segment_disaggregator.DisaggregationSettings = (
        segment_disaggregator.DisaggregationSettings()
    )

    @pydantic.validator(
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


@dataclasses.dataclass
class TravellerSegmentationArguments:
    """Command line arguments for traveller segmentation tool.

    Attributes
    ----------
    config_path: pathlib.Path, default CONFIG_PATH
        Path to config file.
    example_config: bool, default False
        If True write example config file to `config_path`
        and exit the program.
    """

    config_path: pathlib.Path = CONFIG_PATH
    example_config: bool = False

    @classmethod
    def parse(cls) -> TravellerSegmentationArguments:
        """Parse command line arguments."""
        parser = argparse.ArgumentParser(
            description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        parser.add_argument(
            "config",
            nargs="?",
            default=cls.config_path,
            type=pathlib.Path,
            help="path to config file containing parameters",
        )
        parser.add_argument(
            "-e",
            "--example",
            action="store_true",
            help="flag to create example config file and exit, instead of running the tool",
        )

        parsed_args = parser.parse_args()
        return TravellerSegmentationArguments(parsed_args.config, parsed_args.example)

    def validate(self) -> None:
        """Raise error if any arguments are invalid."""
        if not self.example_config and not self.config_path.is_file():
            raise FileNotFoundError(f"config file doesn't exist: {self.config_path}")

        if self.example_config and self.config_path.is_file():
            raise FileExistsError(
                "cannot create example config because file already "
                f"exists: {self.config_path}.\nPlease provide a new "
                "path to create the example config file at."
            )


##### FUNCTIONS #####
def iterate_matrix_files(
    folder: pathlib.Path,
    segmentation: nd.SegmentationLevel,
    file_desc: str,
    year: int,
) -> Iterator[tuple[pathlib.Path, dict]]:
    """Interate through `segmentation` to find matrices.

    Parameters
    ----------
    folder : pathlib.Path
        Folder to look for matrices in.
    segmentation : nd.SegmentationLevel
        Segmentation level of matrices to find.
    file_desc : str
        Expected description in the matrix filenames.
    year : int
        Year of the matrices to find

    Yields
    ------
    pathlib.Path
        Path to the matrix file
    dict[str, Any]
        Segmentation parameters dictionary.

    Raises
    ------
    FileNotFoundError
        If one of the matrix files don't exist.
    """
    for seg_params in segmentation:
        name = segmentation.generate_file_name(
            seg_params, year=year, file_desc=file_desc, compressed=True
        )

        path = folder / name
        if not path.is_file():
            raise FileNotFoundError(f"cannot find matrix: {path}")

        yield path, seg_params


def aggregate_matrices(
    input_folder: pathlib.Path,
    output_folder: pathlib.Path,
    input_segmentation: nd.SegmentationLevel,
    aggregate_segmentation: nd.SegmentationLevel,
    year: int,
    file_desc: str,
) -> None:
    """Aggregate matrices from one segmentation to another.

    If all the output aggregated matrices exist in
    `output_folder` then returns without reproducing them.

    Parameters
    ----------
    input_folder : pathlib.Path
        Folder containing input matrices.
    output_folder : pathlib.Path
        Folder to save aggregated matrices to.
    input_segmentation : nd.SegmentationLevel
        Segmentation for the input matrices.
    aggregate_segmentation : nd.SegmentationLevel
        Segmentation to aggregate to.
    year : int
        Year for the matrices to load.
    file_desc : str
        Expected description in the matrix filenames.
    """
    # Check if aggregated matrices have already been created and don't recreate
    try:
        list(iterate_matrix_files(output_folder, aggregate_segmentation, file_desc, year))
        LOG.info("Using existing aggregated matrices in %s", output_folder)
        return
    except FileNotFoundError:
        pass

    LOG.info(
        "Aggregating input matrices from '%s' to '%s'",
        input_segmentation.name,
        aggregate_segmentation.name,
    )
    input_matrices = {
        input_segmentation.get_segment_name(s): p
        for p, s in iterate_matrix_files(input_folder, input_segmentation, file_desc, year)
    }

    agg_trans = input_segmentation.aggregate(aggregate_segmentation)

    output_folder.mkdir(exist_ok=True)
    with open(output_folder / "README.txt", "wt", encoding="utf-8") as file:
        file.write(
            f"Matrices aggregated from '{input_segmentation.name}' "
            f"to '{aggregate_segmentation.name}'\non {dt.datetime.now():%c}\n"
            f"Input matrices from: {input_folder}\n\n"
            "Matrix Aggregations\n-------------------\n"
            + "\n".join(f"{k}: {v}" for k, v in agg_trans.items())
            + "\n"
        )

    for out_name, in_names in agg_trans.items():
        out_file = output_folder / aggregate_segmentation.generate_file_name(
            aggregate_segmentation.get_seg_dict(out_name),
            year=year,
            file_desc=file_desc,
            compressed=True,
        )

        agg_mat = file_ops.read_matrix(
            input_matrices[in_names[0]], find_similar=True, format_="square"
        )

        if len(in_names) > 1:
            for nm in in_names[1:]:
                mat = file_ops.read_matrix(
                    input_matrices[nm], find_similar=True, format_="square"
                )
                agg_mat += mat

        file_ops.write_df(agg_mat, out_file, index=True)
        LOG.info("Aggregated to matrix: %s", out_file.name)

    LOG.info("Aggregated matrices saved to %s", output_folder)


def main(params: TravellerSegmentationParameters, init_logger: bool = True) -> None:
    """Run traveller segmentation tool.

    Parameters
    ----------
    params : TravellerSegmentationParameters
        Parameters for running the tool.
    init_logger : bool, default True
        Whether or not to initialise a log file.
    """
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
    params.save_yaml(out)
    LOG.info("Input parameters saved to: %s", out)
    LOG.debug("Input parameters:\n%s", params.to_yaml())

    trip_end_converter = traveller_segmentation_trip_ends.NoTEMToTravellerSegmentation(
        output_zoning=params.matrix_parameters.zoning_system,
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

    LOG.info("Checking input matrices: %s", params.matrix_parameters.folder)
    # TODO(MB) Add matrix file description parameter
    file_desc = "hb_synthetic_pa"
    if params.matrix_parameters.aggregate_segmentation is not None:
        matrix_folder = params.iteration_folder / "Aggregated Matrices"
        matrix_segmentation = params.matrix_parameters.aggregate_segmentation
        aggregate_matrices(
            params.matrix_parameters.folder,
            matrix_folder,
            params.matrix_parameters.segmentation,
            params.matrix_parameters.aggregate_segmentation,
            params.year,
            file_desc,
        )
    else:
        matrix_folder = params.matrix_parameters.folder
        matrix_segmentation = params.matrix_parameters.segmentation

    matrices = list(
        iterate_matrix_files(matrix_folder, matrix_segmentation, file_desc, params.year)
    )

    raise NotImplementedError(
        "Not yet implemented functionality for handling matrix segmentations"
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
    """Write an example of the input config YAML file to `path`."""

    class ExampleTSP(TravellerSegmentationParameters):
        """New sub-class which turns off path validation for writing example config."""

        @pydantic.validator(
            "notem_export_home",
            "trip_length_distribution_folder",
            "base_output_folder",
            "cost_folder",
            allow_reuse=True,
        )  # pylint: disable=no-self-argument
        def _folder_exists(cls, value) -> pathlib.Path:
            return value

    class ExampleTSMP(TSMatrixParameters):
        """New sub-class which turns off path validation for writing example config."""

        @pydantic.validator("folder", allow_reuse=True)  # pylint: disable=no-self-argument
        def _folder_exists(cls, value) -> pathlib.Path:
            return value

    example = ExampleTSP(
        iteration="1",
        base_output_folder="path/to/output/folder",
        notem_export_home="path/to/NoTEM/base/export/folder",
        notem_iteration="1",
        scenario=nd.Scenario.SC01_JAM,
        matrix_parameters=ExampleTSMP(
            folder="path/to/folder/containing/matrices/for/segmentation",
            zoning_system_name="norms",
            segmentation_name="hb_p_m",
            aggregate_segmentation_name="hb_p_m",
        ),
        model=nd.AssignmentModel.NORMS,
        cost_folder="path/to/folder/containing/cost/matrices",
        year=2018,
        disaggregation_output_segment=segment_disaggregator.DisaggregationOutputSegment.SOC,
        trip_length_distribution_folder="path/to/tld/folder",
    )

    example.save_yaml(path)
    print(f"Written example config to: {path}")


##### MAIN #####
if __name__ == "__main__":
    args = TravellerSegmentationArguments.parse()
    args.validate()

    if args.example_config:
        example_config(args.config_path)
        raise SystemExit()

    try:
        parameters = TravellerSegmentationParameters.load_yaml(args.config_path)
    except (pydantic.ValidationError, NotADirectoryError) as error:
        LOG.critical("Config file error: %s", error)
        raise SystemExit(1) from error

    try:
        main(parameters)
    except Exception:
        LOG.critical("Traveller segmentation disaggregator error:", exc_info=True)
        raise
