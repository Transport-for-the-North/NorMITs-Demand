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

from normits_demand import distribution

from normits_demand.utils import timing
from normits_demand.utils import vehicle_occupancy as vehicle_occupancy_utils
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
                 tms_arg_builder: tms_arg_builders.TMSArgumentBuilderBase,
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
        self.tms_arg_builder = tms_arg_builder
        self.external_model_arg_builder = tms_arg_builder.external_model_arg_builder
        self.gravity_model_arg_builder = tms_arg_builder.gravity_model_arg_builder

        # Create a logger
        logger_name = "%s.%s" % (nd.get_package_logger_name(), self.__class__.__name__)
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
        external_model = distribution.ExternalModel(
            year=self.year,
            running_mode=self.running_mode,
            zoning_system=self.zoning_system,
            export_home=self.external_model.export_paths.home,
            process_count=self.process_count,
        )

        self._logger.info("Building home-based arguments for external model")
        args = self.external_model_arg_builder.build_hb_arguments()

        self._logger.info("Executing a home-based run of external model")
        external_model.run(trip_origin='hb', **args)

        self._logger.info("Building non-home-based arguments for external model")
        args = self.external_model_arg_builder.build_nhb_arguments()

        self._logger.info("Executing a non-home-based run of external model")
        external_model.run(trip_origin='nhb', **args)

        self._logger.info("External Model Done!")

    def run_gravity_model(self):
        self._logger.info("Initialising the Gravity Model")
        gravity_model = distribution.TMSGravityModel(
            year=self.year,
            running_mode=self.running_mode,
            zoning_system=self.zoning_system,
            export_home=self.gravity_model.export_paths.home,
        )

        self._logger.info("Building home-based arguments for gravity model")
        args = self.gravity_model_arg_builder.build_hb_arguments()

        self._logger.info("Executing a home-based run of gravity model")
        gravity_model.run(trip_origin='hb', **args)

        self._logger.info("Building non-home-based arguments for gravity model")
        args = self.gravity_model_arg_builder.build_nhb_arguments()

        self._logger.info("Executing a non-home-based run of gravity model")
        gravity_model.run(trip_origin='nhb', **args)

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
        # TODO(BT): Make sure the internal and external matrices exist!

        # ## COMBINE INTERNAL AND EXTERNAL MATRICES ## #
        self._logger.info("Recombining internal and external matrices")
        matrix_processing.recombine_internal_external(
            internal_import=self.gravity_model.export_paths.distribution_dir,
            external_import=self.external_model.export_paths.external_distribution_dir,
            full_export=self.export_paths.full_pa_dir,
            force_compress_out=True,
            years=[self.year],
        )

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

    def compile_to_assignment_format(self):
        """TfN Specific helper function to compile outputs into assignment format

        This should really be the job of NorMITs Matrix tools! Move there
        once we create an object of it.

        Returns
        -------

        """
        # TODO(BT): NEED TO OUTPUT SPLITTING FACTORS

        # TODO(BT): UPDATE build_compile_params() to use segmentation levels
        # Imply params from hb. Not ideal but right 99% of time
        m_needed = self.hb_running_segmentation.segments['m'].unique()

        # NoHAM should be tp split
        tp_needed = [1, 2, 3, 4]

        if self.running_mode == nd.Mode.CAR:
            # Compile to NoHAM format
            compile_params_paths = matrix_processing.build_compile_params(
                import_dir=self.export_paths.full_od_dir,
                export_dir=self.export_paths.compiled_od_dir,
                matrix_format='synthetic_od',
                years_needed=[self.year],
                m_needed=m_needed,
                tp_needed=tp_needed,
            )

            matrix_processing.compile_matrices(
                mat_import=self.export_paths.full_od_dir,
                mat_export=self.export_paths.compiled_od_dir,
                compile_params_path=compile_params_paths[0],
            )

            # TODO(BT): Build in TMS imports!
            car_occupancies = pd.read_csv(os.path.join(
                r'I:\NorMITs Demand\import',
                'vehicle_occupancies',
                'car_vehicle_occupancies.csv',
            ))

            # Need to convert into hourly average PCU for noham
            vehicle_occupancy_utils.people_vehicle_conversion(
                mat_import=self.export_paths.compiled_od_dir,
                mat_export=self.export_paths.compiled_od_dir_pcu,
                car_occupancies=car_occupancies,
                mode=m_needed[0],
                method='to_vehicles',
                out_format='wide',
                hourly_average=True,
            )

        else:
            raise ValueError(
                "I don't know how to compile mode %s into an assignment model "
                "format :("
                % self.running_mode.value
            )
