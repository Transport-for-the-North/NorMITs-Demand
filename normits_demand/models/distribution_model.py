# -*- coding: utf-8 -*-
"""
Created on: 07/12/2021
Updated on:

Original author: Ben Taylor
Last update made by:
Other updates made by:

File purpose:
Home of the NorMITs Distribution Model
"""
# Built-Ins
import os

from typing import Any
from typing import List
from typing import Dict

# Third Party

# Local Imports
import normits_demand as nd

from normits_demand import constants

from normits_demand.utils import timing
from normits_demand.matrices import matrix_processing

from normits_demand.pathing.distribution_model import DistributionModelExportPaths
from normits_demand.pathing.distribution_model import DMArgumentBuilderBase


class DistributionModel(DistributionModelExportPaths):
    # ## Class Constants ## #
    __version__ = nd.__version__

    _running_report_fname = 'running_parameters.txt'
    _log_fname = "Distribution_Model_log.log"

    _dist_overall_log_name = '{trip_origin}_overall_log.csv'

    def __init__(self,
                 year: int,
                 running_mode: nd.Mode,
                 trip_origin: str,
                 running_segmentation: nd.SegmentationLevel,
                 iteration_name: str,
                 arg_builder: DMArgumentBuilderBase,
                 export_home: nd.PathLike,
                 upper_model_method: nd.DistributionMethod,
                 upper_model_zoning: nd.ZoningSystem,
                 upper_running_zones: List[Any],
                 upper_model_kwargs: Dict[str, Any] = None,
                 lower_model_method: nd.DistributionMethod = None,
                 lower_model_zoning: nd.ZoningSystem = None,
                 lower_running_zones: List[Any] = None,
                 lower_model_kwargs: Dict[str, Any] = None,
                 process_count: int = constants.PROCESS_COUNT,
                 ):
        # Make sure all are set if one is
        lower_args = [lower_model_method, lower_model_zoning, lower_running_zones]
        if not all([x is not None for x in lower_args]):
            raise ValueError(
                "Only some of the lower tier model arguments have been set. "
                "Either all of these arguments need to be set, or none of them "
                "do. This applies to the following arguments: "
                "[lower_model_method', 'lower_model_zoning', 'lower_running_zones]"
            )

        # Generate export paths
        super().__init__(
            year=year,
            trip_origin=trip_origin,
            iteration_name=iteration_name,
            running_mode=running_mode,
            upper_model_method=upper_model_method,
            lower_model_method=lower_model_method,
            export_home=export_home,
        )

        # Get default values if set to None
        upper_model_kwargs = dict() if upper_model_kwargs is None else upper_model_kwargs
        lower_model_kwargs = dict() if lower_model_kwargs is None else lower_model_kwargs

        # TODO(BT): Check all lower things are set

        # Assign attributes
        self.running_segmentation = running_segmentation
        self.process_count = process_count

        self.upper_model_zoning = upper_model_zoning
        self.upper_running_zones = upper_running_zones
        self.upper_model_kwargs = upper_model_kwargs
        self.lower_model_zoning = lower_model_zoning
        self.lower_running_zones = lower_running_zones
        self.lower_model_kwargs = lower_model_kwargs

        # TODO(BT): Validate this is correct type
        self.arg_builder = arg_builder

        # Create a logger
        logger_name = "%s.%s" % (nd.get_package_logger_name(), self.__class__.__name__)
        log_file_path = os.path.join(self.export_home, self._log_fname)
        self._logger = nd.get_logger(
            logger_name=logger_name,
            log_file_path=log_file_path,
            instantiate_msg="Initialised new Distribution Model Logger",
        )

        self._write_running_report()

    def _write_running_report(self):
        """
        Outputs a simple report detailing inputs and outputs
        """
        # Define the lines to output
        out_lines = [
            'Code Version: %s' % str(nd.__version__),
            'TMS Iteration: %s' % str(self.iteration_name),
            '',
            # '### Upper Model ###',
            # 'vector_export: %s' % self.upper_model.export_paths.home,
            # 'report_export: %s' % self.external_model.report_paths.home,
            # '',
            # '### Lower Model ###',
            # 'vector_export: %s' % self.gravity_model.export_paths.home,
            # 'report_export: %s' % self.gravity_model.report_paths.home,
            # '',
        ]

        # Write out to disk
        output_path = os.path.join(self.export_home, self._running_report_fname)
        with open(output_path, 'w') as out:
            out.write('\n'.join(out_lines))

    def run(self,
            run_all: bool = False,
            run_upper_model: bool = False,
            run_lower_model: bool = False,
            run_pa_matrix_reports: bool = False,
            run_pa_to_od: bool = False,
            run_od_matrix_reports: bool = False,
            ) -> None:
        """Runs the components of Distribution Model

        Run parameters are based off of the parameters passed into the class
        constructors. Read the documentation of individual run functions to
        see the parameters used in each step.

        Parameters
        ----------
        run_all:
            Whether to run all parts of TMS or not. This argument overwrites
            all others if it is set to True.

        run_upper_model:
            Whether to run the upper model or not. Usually the upper model
            deals with a bigger geographic area than the lower model. The
            upper should always be run first. Only set this to False if a
            previous run of this distribution model has ran the upper model
            and those outputs should be used in the following steps instead.

        run_lower_model:
            Whether to run the lower model or not. The lower model generally
            deals with a smaller geographic area than the upper model.
            The lower model is designed to take some "internal" demand from
            the upper model and more finely rune the outputs. This argument is
            ignored if lower_model_needed is set to False when constructing
            the object.

        run_pa_matrix_reports:
            Whether to run the matrix reports for full PA matrices or not.
            This step depends on at least the upper model being run, and where
            lower_model_needed is set in the constructor, depends on the lower
            model being run too. These steps produce the PA matrices.
            The following reports will be generated:
            Matrix Trip End totals
            Sector Reports - by segment
            TLD curve by segment and in single mile bands.

        run_pa_to_od:
            Whether to run the PA to OD conversion process or not. This step
            depends on the external model and gravity model already being
            run - as these steps produce the PA matrices to convert.

        run_od_matrix_reports:
            Whether to run the matrix reports for full OD matrices or not.
            This step depends on the PA to OD conversion already being run -
            as this step produces the OD matrices. The following reports will
            be generated:
            Matrix Trip End totals
            Sector Reports - by segment
            TLD curve by segment and in single mile bands.

        Returns
        -------
        None
        """
        # TODO(BT): Add checks to make sure input paths exist when models
        #  depend on one another
        start_time = timing.current_milli_time()
        self._logger.info("Starting a new run of TMS")

        # Determine which models to run
        if run_all:
            run_upper_model = True
            run_lower_model = True
            run_pa_matrix_reports = True
            run_pa_to_od = True
            run_od_matrix_reports = True

        self._logger.debug("Running upper model: %s" % run_upper_model)
        self._logger.debug("Running lower model: %s" % run_lower_model)
        self._logger.debug("Running pa matrix reports: %s" % run_pa_matrix_reports)
        self._logger.debug("Running pa to od: %s" % run_pa_to_od)
        self._logger.debug("Running od matrix reports: %s" % run_od_matrix_reports)
        self._logger.debug("")

        # Check that we are actually running something
        if not any([run_upper_model, run_lower_model, run_pa_to_od]):
            self._logger.info(
                "All run args set to False. Not running anything"
            )

        # Run the models
        if run_upper_model:
            self.run_upper_model()

        if run_lower_model:
            self.run_lower_model()

        if run_pa_matrix_reports:
            self.run_pa_matrix_reports()

        if run_pa_to_od:
            self.run_pa_to_od()

        if run_od_matrix_reports:
            self.run_od_matrix_reports()

        # Log the time taken to run
        end_time = timing.current_milli_time()
        time_taken = timing.time_taken(start_time, end_time)
        self._logger.info("Distribution Model run complete! Took %s" % time_taken)

    def run_upper_model(self):

        self._logger.info("Initialising the Upper Model")
        upper_model = self.upper_model_method.get_distributor(
                year=self.year,
                trip_origin=self.trip_origin,
                running_mode=self.running_mode,
                zoning_system=self.upper_model_zoning,
                running_zones=self.upper_running_zones,
                export_home=self.upper_export_home,
                process_count=self.process_count,
                **self.upper_model_kwargs,
        )

        self._logger.info("Building arguments for the Upper Model")
        kwargs = self.arg_builder.build_upper_model_arguments()

        self._logger.info("Running the Upper Model")
        upper_model.distribute(**kwargs)
        self._logger.info("Upper Model Done!")

    def run_lower_model(self):
        self._logger.info("Initialising the Lower Model")
        lower_model = self.lower_model_method.get_distributor(
                year=self.year,
                trip_origin=self.trip_origin,
                running_mode=self.running_mode,
                zoning_system=self.lower_model_zoning,
                running_zones=self.lower_running_zones,
                export_home=self.lower_export_home,
                process_count=self.process_count,
                **self.lower_model_kwargs,
        )

        self._logger.info("Converting Upper Model Outputs for Lower Model")
        productions, attractions = self.arg_builder.read_lower_pa(
            upper_model_matrix_dir=self.upper.export_paths.matrix_dir,
            external_matrix_output_dir=self.export_paths.upper_external_pa,
        )

        self._logger.info("Building arguments for the Lower Model")
        kwargs = self.arg_builder.build_lower_model_arguments()
        kwargs.update({
            'productions': productions,
            'attractions': attractions,
        })

        self._logger.info("Running the Lower Model")
        lower_model.distribute(**kwargs)
        self._logger.info("Lower Model Done!")

    def run_pa_matrix_reports(self):
        # PA RUN REPORTS
        # Matrix Trip ENd totals
        # Sector Reports Dvec style
        # TLD curve
        #   single mile bands - p/m (ca ) segments full matrix
        #   NorMITs Vis

        pass

    def run_pa_to_od(self):
        # TODO(BT): Make sure the internal and external matrices exist!

        # ## COMBINE INTERNAL AND EXTERNAL MATRICES ## #
        if self.lower_model_method is not None:
            # External should be made by  lower tier
            self._logger.info("Recombining internal and external matrices")
            matrix_processing.recombine_internal_external(
                internal_import=self.lower.export_paths.matrix_dir,
                external_import=self.upper.export_paths.matrix_dir,
                full_export=self.export_paths.full_pa_dir,
                force_compress_out=True,
                years=[self.year],
            )
        else:
            # Copy over
            pass

        # ## CONVERT HB PA TO OD ## #
        self._logger.info("Converting HB PA matrices to OD")
        kwargs = self.tms_arg_builder.build_pa_to_od_arguments()
        pa_to_od.build_od_from_fh_th_factors(
            pa_import=self.export_paths.full_pa_dir,
            od_export=self.export_paths.full_od_dir,
            pa_matrix_desc='synthetic_pa',
            od_to_matrix_desc='synthetic_od_to',
            od_from_matrix_desc='synthetic_od_from',
            years_needed=[self.year],
            **kwargs
        )

        # ## MOVE NHB TO OD DIR ## #
        # they're already OD anyway, just need a little name change
        matrix_processing.copy_nhb_matrices(
            import_dir=self.export_paths.full_pa_dir,
            export_dir=self.export_paths.full_od_dir,
            replace_pa_with_od=True,
            pa_matrix_desc='synthetic_pa',
            od_matrix_desc='synthetic_od',
        )

    def run_od_matrix_reports(self):
        # PA RUN REPORTS
        # Matrix Trip ENd totals
        # Sector Reports Dvec style
        # TLD curve
        #   single mile bands - p/m (ca ) segments full matrix
        #   NorMITs Vis

        pass



