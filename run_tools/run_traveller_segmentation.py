# -*- coding: utf-8 -*-
"""Run script for disaggregating model matrices into other segmentations."""

##### IMPORTS #####
import pathlib
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
from normits_demand.core import enumerations as nd_enum
from normits_demand.distribution import segment_disaggregator
from normits_demand.matrices import matrix_processing
from normits_demand.utils import config_base, file_ops

# pylint: enable=import-error,wrong-import-position

##### CONSTANTS #####
LOG_FILE = "Traveller_segmentation.log"
LOG = nd_log.get_logger(nd_log.get_package_logger_name() + ".run_traveller_segmentation")
CONFIG_PATH = pathlib.Path("config/Traveller_segmentation_parameters.yml")
MODEL_SEGMENTATIONS = {
    nd_enum.AssignmentModel.NOHAM: (
        (nd.TripOrigin.HB, "commute"),
        (nd.TripOrigin.HB, "business"),
        (nd.TripOrigin.HB, "other"),
        (nd.TripOrigin.NHB, "business"),
        (nd.TripOrigin.NHB, "other"),
    ),
    nd_enum.AssignmentModel.NORMS: (
        (nd.TripOrigin.HB, "commute", "ca"),
        (nd.TripOrigin.HB, "business", "ca"),
        (nd.TripOrigin.HB, "other", "ca"),
        (nd.TripOrigin.NHB, "business", "ca"),
        (nd.TripOrigin.NHB, "other", "ca"),
        (nd.TripOrigin.HB, "commute", "nca"),
        (nd.TripOrigin.HB, "business", "nca"),
        (nd.TripOrigin.HB, "other", "nca"),
        (nd.TripOrigin.NHB, "business", "nca"),
        (nd.TripOrigin.NHB, "other", "nca"),
    ),
}


##### CLASSES #####
class TLDFolder(pydantic.BaseModel):
    # TODO Docstring explaining parameters
    folder: pathlib.Path
    area: str
    segmentation: str

    @pydantic.validator("folder")
    def _folder_exists(cls, value) -> pathlib.Path:
        try:
            return file_ops.folder_exists(value)
        except NotADirectoryError as err:
            raise ValueError(err) from err

    @staticmethod
    def _build_full_folder(folder: pathlib.Path, area: str, segmentation: str) -> pathlib.Path:
        return folder / area / segmentation

    @pydantic.root_validator(skip_on_failure=True)
    def _check_folder(cls, values: dict[str, Any]) -> dict[str, Any]:
        folder = cls._build_full_folder(
            values.get("folder"), values.get("area"), values.get("segmentation")
        )
        cls._folder_exists(folder)
        return values

    @property
    def full_folder(self) -> pathlib.Path:
        """Folder containing the trip length distribution files."""
        return self._build_full_folder(self.folder, self.area, self.segmentation)


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
    notem_iteration: str
    scenario: nd.Scenario
    matrix_folder: pathlib.Path
    model: nd_enum.AssignmentModel
    year: int
    trip_length_distribution: TLDFolder
    disaggregation_settings: DisaggregationSettings = DisaggregationSettings()

    @pydantic.validator("import_folder", "matrix_folder")
    def _folder_exists(cls, value) -> pathlib.Path:
        try:
            return file_ops.folder_exists(value)
        except NotADirectoryError as err:
            raise ValueError(err) from err

    @staticmethod
    def _build_cost_folder(
        import_folder: pathlib.Path, model: nd_enum.AssignmentModel
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
def aggregate_purposes(
    matrix_folder: pathlib.Path, model: nd_enum.AssignmentModel, year: int
) -> pathlib.Path:
    """Aggregate matrices in NTEM purposes to model user classes.

    Parameters
    ----------
    matrix_folder : pathlib.Path
        Folder containing the matrices by NTEM purpose.
    model : nd_enum.AssignmentModel
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
    if model != nd_enum.AssignmentModel.NORMS:
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

    model_folder = params.import_folder / params.model.get_name()
    file_ops.folder_exists(model_folder)

    # TODO Extract trip ends from NoTEM DVectors instead of CSVs
    p_home = model_folder / f"iter{params.notem_iteration}" / "Production Outputs"
    a_home = p_home.with_name("Attraction Outputs")
    if params.model == nd_enum.AssignmentModel.NOHAM:
        raise NotImplementedError(
            "Traveller segmentation tool MVP not implemented for the NoHAM model"
        )
    elif params.model == nd_enum.AssignmentModel.NORMS:
        productions = {
            nd.TripOrigin.HB: p_home / "fake out/hb_productions_norms.csv",
            nd.TripOrigin.NHB: p_home / "fake out/nhb_productions_norms.csv",
        }
        attractions = {
            nd.TripOrigin.HB: a_home / "fake out/ca/norms_hb_attractions.csv",
            nd.TripOrigin.NHB: a_home / "fake out/ca/norms_nhb_attractions.csv",
        }
    else:
        raise NotImplementedError(
            "Traveller segmentation tool functionality "
            f"not implemented for {params.model.get_name()}"
        )

    lookup_folder = model_folder / "Model Zone Lookups"

    matrix_folder = aggregate_purposes(params.matrix_folder, params.model, params.year)

    for to in nd.TripOrigin:
        LOG.info("Decompiling %s matrices", to.get_name())

        segment_disaggregator.disaggregate_segments(
            import_folder=matrix_folder,
            # TODO Old TLDs were in miles new are kms and costs and kms so don't need to convert anymore
            target_tld_folder=params.trip_length_distribution.full_folder,
            model_name=params.model.get_name(),
            base_productions_path=productions[to],
            base_attractions_path=attractions[to],
            export_folder=params.output_folder,
            lookup_folder=lookup_folder,
            trip_origin=to.get_name(),
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
