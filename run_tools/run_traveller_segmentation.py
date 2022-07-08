# -*- coding: utf-8 -*-
"""Run script for disaggregating model matrices into other segmentations."""

##### IMPORTS #####
import logging
import pathlib
import re
import sys
from typing import Any

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
    # TODO Docstring explaining the parameters
    iteration: str
    import_folder: pathlib.Path
    base_output_folder: pathlib.Path
    notem_export_home: pathlib.Path
    notem_iteration: str
    scenario: nd.Scenario
    matrix_folder: pathlib.Path
    model: nd.AssignmentModel
    year: int
    disaggregation_output_segment: segment_disaggregator.DisaggregationOutputSegment
    trip_length_distribution_folder: pathlib.Path
    trip_length_distribution_units: nd.CostUnits = nd.CostUnits.KILOMETERS
    disaggregation_settings: segment_disaggregator.DisaggregationSettings = (
        segment_disaggregator.DisaggregationSettings()
    )

    @pydantic.validator(
        "import_folder",
        "matrix_folder",
        "notem_export_home",
        "trip_length_distribution_folder",
        "base_output_folder",
    )  # pylint: disable=no-self-argument
    def _folder_exists(cls, value) -> pathlib.Path:
        try:
            return file_ops.folder_exists(value)
        except NotADirectoryError as err:
            raise ValueError(err) from err

    @staticmethod
    def _build_cost_folder(
        import_folder: pathlib.Path, model: nd.AssignmentModel
    ) -> pathlib.Path:
        return (
            import_folder / "modal" / model.get_mode().get_name() / "costs" / model.get_name()
        )

    @pydantic.root_validator(skip_on_failure=True)
    def _check_cost_folder(  # pylint: disable=no-self-argument
        cls, values: dict[str, Any]
    ) -> dict[str, Any]:
        cost_folder = cls._build_cost_folder(values.get("import_folder"), values.get("model"))  # type: ignore
        cls._folder_exists(cost_folder)
        return values

    @property
    def cost_folder(self) -> pathlib.Path:
        return self._build_cost_folder(self.import_folder, self.model)

    @property
    def output_folder(self) -> pathlib.Path:
        output_folder = self.base_output_folder / f"iter{self.iteration}"
        output_folder.mkdir(exist_ok=True)
        return output_folder


##### FUNCTIONS #####
def _compile_params_format(compile_params_path: pathlib.Path) -> None:
    """Replace ca/nca in compiled matrix names with ca2/ca1, respectively.

    Overwrites existing compilation parameters file with adjusted names.

    Parameters
    ----------
    compile_params_path : pathlib.Path
        Path to compilation parameters file created by
        `matrix_processing.build_compile_params`.
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
    file_ops.write_df(compile_params, compile_params_path, index=False)


def aggregate_purposes(
    matrix_folder: pathlib.Path, model: nd.AssignmentModel, year: int
) -> pathlib.Path:
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

    compile_params_path = pathlib.Path(
        matrix_processing.build_compile_params(
            import_dir=matrix_folder,
            export_dir=output_folder,
            matrix_format="pa",
            years_needed=[year],
            m_needed=model.get_mode().get_mode_values(),
            ca_needed=[1, 2],
            split_hb_nhb=True,
        )[0]
    )
    # Update compile parameters to use ca1/2 instead of nca/ca in the output names
    _compile_params_format(compile_params_path)

    matrix_processing.compile_matrices(
        mat_import=matrix_folder,
        mat_export=output_folder,
        compile_params_path=compile_params_path,
        overwrite=False,
    )
    LOG.info("Output user class matrices saved: %s", output_folder)

    return output_folder


def main(params: TravellerSegmentationParameters, init_logger: bool = True) -> None:
    # TODO Docstring
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

    LOG.info("Outputs saved to: %s", params.output_folder)
    out = params.output_folder / "Traveller_segmentation_parameters.yml"
    parameters.save_yaml(out)
    LOG.info("Input parameters saved to: %s", out)
    LOG.debug("Input parameters:\n%s", params.to_yaml())

    trip_end_converter = traveller_segmentation_trip_ends.NoTEMToTravellerSegmentation(
        output_zoning=nd.get_zoning_system(params.model.get_name().lower()),
        base_year=params.year,
        scenario=params.scenario,
        notem_iteration_name=params.notem_iteration,
        export_home=params.notem_export_home,
        cache_dir=params.output_folder / ".cache",
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

    matrix_folder = aggregate_purposes(params.matrix_folder, params.model, params.year)

    for to in nd.TripOrigin:
        LOG.info("Decompiling %s matrices", to.value)

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


##### MAIN #####
if __name__ == "__main__":
    logging.captureWarnings(True)

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
