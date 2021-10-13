# -*- coding: utf-8 -*-
"""
Created on: 08/09/2021
Updated on:

Original author: Ben Taylor
Last update made by:
Other updates made by:

File purpose:
Home of the NorMITs Travel Market Synthesiser
"""
# Built-Ins
import os

# Third Party
import pandas as pd

# Local Imports
import normits_demand as nd
from normits_demand import constants

from normits_demand import models

from normits_demand.utils import timing
from normits_demand.pathing.travel_market_synthesiser import TMSExportPaths

from normits_demand.matrices import matrix_processing
from normits_demand.matrices import pa_to_od


# Alias for shorter name
tms_arg_builders = nd.pathing.travel_market_synthesiser


class TravelMarketSynthesiser(TMSExportPaths):
    # ## Class Constants ## #
    __version__ = nd.version.__version__

    _running_report_fname = 'running_parameters.txt'
    _log_fname = "TMS_log.log"

    def __init__(self,
                 year: int,
                 running_mode: nd.Mode,
                 hb_running_segmentation: nd.core.segments.SegmentationLevel,
                 nhb_running_segmentation: nd.core.segments.SegmentationLevel,
                 iteration_name: str,
                 zoning_system: nd.core.zoning.ZoningSystem,
                 external_model_arg_builder: tms_arg_builders.ExternalModelArgumentBuilderBase,
                 gravity_model_arg_builder: tms_arg_builders.GravityModelArgumentBuilderBase,
                 export_home: nd.PathLike,
                 process_count: int = constants.PROCESS_COUNT,
                 ):
        # Generate export paths
        super().__init__(
            year=year,
            iteration_name=iteration_name,
            running_mode=running_mode,
            export_home=export_home,
        )

        # Assign attributes
        self.hb_running_segmentation = hb_running_segmentation
        self.nhb_running_segmentation = nhb_running_segmentation
        self.zoning_system = zoning_system
        self.process_count = process_count

        # TODO(BT): Validate this is correct type
        self.external_model_arg_builder = external_model_arg_builder
        self.gravity_model_arg_builder = gravity_model_arg_builder

        # Create a logger
        logger_name = "%s.%s" % (__name__, self.__class__.__name__)
        log_file_path = os.path.join(self.export_home, self._log_fname)
        self._logger = nd.get_logger(
            logger_name=logger_name,
            log_file_path=log_file_path,
            instantiate_msg="Initialised new TMS Logger",
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
            '### External Model ###',
            'vector_export: %s' % self.external_model.export_paths.home,
            'report_export: %s' % self.external_model.report_paths.home,
            '',
            '### Gravity Model ###',
            'vector_export: %s' % self.gravity_model.export_paths.home,
            'report_export: %s' % self.gravity_model.report_paths.home,
            '',
        ]

        # Write out to disk
        output_path = os.path.join(self.export_home, self._running_report_fname)
        with open(output_path, 'w') as out:
            out.write('\n'.join(out_lines))

    def run(self,
            run_all: bool = False,
            run_external_model: bool = False,
            run_gravity_model: bool = False,
            run_pa_matrix_reports: bool = False,
            run_pa_to_od: bool = False,
            run_od_matrix_reports: bool = False,
            ) -> None:
        """Runs the components of TMS

        Run parameters are based off of the parameters passed into the class
        constructors. Read the documentation of individual run functions to
        see the parameters used in each step.

        Parameters
        ----------
        run_all:
            Whether to run all parts of TMS or not. This argument overwrites
            all others if it is set to True.

        run_external_model:
            Whether to run the external model of TMS or not. The external
            model deals with all external demand, including internal to
            external and vice versa

        run_gravity_model:
            Whether to run the gravity model of TMS or not. The gravity model
            deals with internal to internal demand only.

        run_pa_matrix_reports:
            Whether to run the matrix reports for full PA matrices or not.
            This step depends on the external model and gravity model already
            being run - as these steps produce the PA matrices. The following
            reports will be generated:
            Matrix Trip End totals
            Sector Reports - by segment
            TLD curve by segment and in single mile bands.

        run_pa_to_od:
            Whether to run the PA to OD conversion process or not. This step
            depends on the external model and gravity model already being
            run - as these steps produce the PA matrices to convert.

        run_od_matrix_reports:
            Whether to run the matrix reports for full PA matrices or not.
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
            run_external_model = True
            run_gravity_model = True
            run_pa_matrix_reports = True
            run_pa_to_od = True
            run_od_matrix_reports = True

        self._logger.debug("Running external model: %s" % run_external_model)
        self._logger.debug("Running gravity model: %s" % run_gravity_model)
        self._logger.debug("Running pa matrix reports: %s" % run_pa_matrix_reports)
        self._logger.debug("Running pa to od: %s" % run_pa_to_od)
        self._logger.debug("Running od matrix reports: %s" % run_od_matrix_reports)
        self._logger.debug("")

        # Check that we are actually running something
        if not any([run_external_model, run_gravity_model, run_pa_to_od]):
            self._logger.info(
                "All run args set to False. Not running anything"
            )

        # Run the models
        if run_external_model:
            self.run_external_model()

        if run_gravity_model:
            self.run_gravity_model()

        if run_pa_matrix_reports:
            self.run_pa_matrix_reports()

        if run_pa_to_od:
            self.run_pa_to_od()

        if run_od_matrix_reports:
            self.run_od_matrix_reports()

        # Log the time taken to run
        end_time = timing.current_milli_time()
        time_taken = timing.time_taken(start_time, end_time)
        self._logger.info("TMS run complete! Took %s" % time_taken)

    def run_external_model(self):

        self._logger.info("Initialising the External Model")
        external_model = models.ExternalModel(
            year=self.year,
            running_mode=self.running_mode,
            zoning_system=self.zoning_system,
            export_home=self.external_model.export_paths.home,
            process_count=self.process_count,
        )

        self._logger.info("Building home-based arguments for external model")
        args = self.external_model_arg_builder.build_hb_arguments()

        self._logger.info("Executing a home-based run of external model")
        external_model.run(
            trip_origin='hb',
            running_segmentation=self.hb_running_segmentation,
            **args,
        )

        self._logger.info("Building non-home-based arguments for external model")
        args = self.external_model_arg_builder.build_nhb_arguments()

        self._logger.info("Executing a non-home-based run of external model")
        external_model.run(
            trip_origin='nhb',
            running_segmentation=self.nhb_running_segmentation,
            **args,
        )

        self._logger.info("External Model Done!")

    def run_gravity_model(self):
        self._logger.info("Initialising the Gravity Model")
        gravity_model = models.GravityModel(
            year=self.year,
            running_mode=self.running_mode,
            zoning_system=self.zoning_system,
            export_home=self.gravity_model.export_paths.home,
        )

        self._logger.info("Building home-based arguments for gravity model")
        args = self.gravity_model_arg_builder.build_hb_arguments()

        self._logger.info("Executing a home-based run of gravity model")
        gravity_model.run(
            trip_origin='hb',
            running_segmentation=self.hb_running_segmentation,
            **args,
        )

        self._logger.info("Building non-home-based arguments for gravity model")
        args = self.gravity_model_arg_builder.build_nhb_arguments()

        self._logger.info("Executing a non-home-based run of gravity model")
        gravity_model.run(
            trip_origin='nhb',
            running_segmentation=self.nhb_running_segmentation,
            **args,
        )

        self._logger.info("Gravity Model Done!")

    def run_pa_matrix_reports(self):
        # PA RUN REPORTS
        # Matrix Trip ENd totals
        # Sector Reports Dvec style
        # TLD curve
        #   single mile bands - p/m (ca ) segments full matrix
        #   NorMITs Vis

        pass

    def run_pa_to_od(self):
        # Combine internal and external
        self._logger.info("Recombining internal and external matrices")
        matrix_processing.recombine_internal_external(
            internal_import=self.gravity_model.export_paths.distribution_dir,
            external_import=self.external_model.export_paths.external_distribution_dir,
            full_export=self.export_paths.full_pa_dir,
            force_compress_out=True,
            years=[self.year],
        )

        # Convert hb PA to OD matrices
        # Set up the segmentation params
        # TODO(BT): UPDATE build_od_from_fh_th_factors() to use segmentation levels
        seg_level = 'tms'
        seg_params = {
            'p_needed': self.hb_running_segmentation.segments['p'].unique(),
            'm_needed': self.hb_running_segmentation.segments['m'].unique(),
        }
        if 'ca' in self.hb_running_segmentation.segment_names:
            seg_params.update({
                'ca_needed': self.hb_running_segmentation.segments['ca'].unique(),
            })

        # TODO(BT): Build import paths for TMS!
        if self.running_mode in [nd.Mode.CAR, nd.Mode.BUS]:
            fh_th_factors_dir = r'I:\NorMITs Demand\import\noham\post_me_tour_proportions\fh_th_factors'
        elif self.running_mode in [nd.Mode.RAIL]:
            fh_th_factors_dir = r'I:\NorMITs Demand\import\norms\post_me_tour_proportions\fh_th_factors'
        else:
            raise ValueError("Uh Oh!")

        pa_to_od.build_od_from_fh_th_factors(
            pa_import=self.export_paths.full_pa_dir,
            od_export=self.export_paths.full_od_dir,
            fh_th_factors_dir=fh_th_factors_dir,
            years_needed=[self.year],
            seg_level=seg_level,
            seg_params=seg_params
        )

        # Move NHB to OD folder (they're already OD anyway)

        # Compile to output segmentation

        pass

    def run_od_matrix_reports(self):
        # PA RUN REPORTS
        # Matrix Trip ENd totals
        # Sector Reports Dvec style
        # TLD curve
        #   single mile bands - p/m (ca ) segments full matrix
        #   NorMITs Vis

        pass
