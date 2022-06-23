# -*- coding: utf-8 -*-
"""Run script for disaggregating model matrices into other segmentations."""

##### IMPORTS #####
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
class DisaggregationSettings(pydantic.BaseModel):
    # TODO Docstring explaining the parameters
    aggregate_surplus_segments: bool = True
    export_original: bool = True
    export_furness: bool = False
    rounding: int = 5
    time_period: str = "24hr"
    intrazonal_cost_infill: float = 0.5
    maximum_furness_loops: int = 1999
    pa_furness_convergence: float = 0.1
    bandshare_convergence: float = 0.975
    max_bandshare_loops: int = 200
    multiprocessing_threads: int = -1


class TravellerSegmentationParameters(config_base.BaseConfig):
    # TODO Docstring explaining the parameters
    import_folder: pathlib.Path
    output_folder: pathlib.Path
    notem_export_home: pathlib.Path
    notem_iteration: str
    scenario: nd.Scenario
    matrix_folder: pathlib.Path
    model: nd.AssignmentModel
    year: int
    trip_length_distribution_folder: pathlib.Path
    disaggregation_settings: DisaggregationSettings = DisaggregationSettings()

    @pydantic.validator(
        "import_folder",
        "matrix_folder",
        "notem_export_home",
        "trip_length_distribution_folder",
    )
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
    def _check_cost_folder(cls, values: dict[str, Any]) -> dict[str, Any]:
        cost_folder = cls._build_cost_folder(values.get("import_folder"), values.get("model"))
        cls._folder_exists(cost_folder)
        return values

    @property
    def cost_folder(self) -> pathlib.Path:
        return self._build_cost_folder(self.import_folder, self.model)


##### FUNCTIONS #####
def _compile_params_format(compile_params_path: nd.PathLike) -> None:
    """Replace ca/nca in compiled matrix names with ca2/ca1, respectively.

    Overwrites existing compilation parameters file with adjusted names.

    Parameters
    ----------
    compile_params_path : nd.PathLike
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

    compile_params_path = matrix_processing.build_compile_params(
        import_dir=matrix_folder,
        export_dir=output_folder,
        matrix_format="pa",
        years_needed=[year],
        m_needed=model.get_mode().get_mode_values(),
        ca_needed=[1, 2],
        split_hb_nhb=True,
    )[0]
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

    params.output_folder.mkdir(exist_ok=True, parents=True)

    if init_logger:
        # Add log file output to main package logger
        nd_log.get_logger(
            nd_log.get_package_logger_name(),
            params.output_folder / LOG_FILE,
            "Running Traveller Segmentation Disaggregator",
            log_version=True,
        )

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

    model_folder = params.import_folder / params.model.get_name()
    file_ops.folder_exists(model_folder)
    # TODO Use new cost lookup
    lookup_folder = model_folder / "Model Zone Lookups"

    matrix_folder = aggregate_purposes(params.matrix_folder, params.model, params.year)

    for to in nd.TripOrigin:
        LOG.info("Decompiling %s matrices", to.get_name())

        productions, attractions = trip_end_converter.get_trip_ends(
            to, nd.get_segmentation_level(MODEL_SEGMENTATIONS[to])
        )

        continue  # Skip dissaggregator as it's not yet implemented with the new data
        segment_disaggregator.disaggregate_segments(
            import_folder=matrix_folder,
            # TODO Old TLDs were in miles new are kms and costs and kms so don't need to convert anymore
            # TODO Use TLD enums as parameters for finding TLD path, TLD has a function for finding the path
            target_tld_folder=params.trip_length_distribution_folder,
            model=params.model,
            base_productions=productions,
            base_attractions=attractions,
            export_folder=params.output_folder,
            lookup_folder=lookup_folder,
            trip_origin=to,
            aggregate_surplus_segments=params.disaggregation_settings.aggregate_surplus_segments,
            rounding=params.disaggregation_settings.rounding,
            tp=params.disaggregation_settings.time_period,
            iz_infill=params.disaggregation_settings.intrazonal_cost_infill,
            furness_loops=params.disaggregation_settings.maximum_furness_loops,
            min_pa_diff=params.disaggregation_settings.pa_furness_convergence,
            bs_con_crit=params.disaggregation_settings.bandshare_convergence,
            max_bs_loops=params.disaggregation_settings.max_bandshare_loops,
            mp_threads=params.disaggregation_settings.multiprocessing_threads,
            export_original=params.disaggregation_settings.export_original,
            export_furness=params.disaggregation_settings.export_furness,
        )


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