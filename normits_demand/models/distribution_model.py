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
import pathlib
import functools

from typing import Any
from typing import List
from typing import Dict

# Third Party
import tqdm
import pandas as pd

# Local Imports
import normits_demand as nd

from normits_demand import constants

from normits_demand.utils import timing
from normits_demand.utils import file_ops
from normits_demand.utils import translation
from normits_demand.utils import vehicle_occupancy
from normits_demand.matrices import matrix_processing
from normits_demand.matrices import pa_to_od
from normits_demand.matrices import utils as mat_utils
from normits_demand.reports import matrix_reports

from normits_demand.pathing.distribution_model import DistributionModelExportPaths
from normits_demand.pathing.distribution_model import DMArgumentBuilderBase


class DistributionModel(DistributionModelExportPaths):
    # ## Class Constants ## #
    __version__ = nd.__version__

    _pa_matrix_desc = 'synthetic_pa'
    _od_matrix_desc = 'synthetic_od'
    _od_to_matrix_desc = 'synthetic_od_to'
    _od_from_matrix_desc = 'synthetic_od_from'

    _translated_dir_name = 'translated'

    _running_report_fname = 'running_parameters.txt'
    _log_fname = "Distribution_Model_log.log"

    _dist_overall_log_name = '{trip_origin}_overall_log.csv'

    # Trip End cache constants

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
                 upper_distributor_kwargs: Dict[str, Any] = None,
                 lower_model_method: nd.DistributionMethod = None,
                 lower_model_zoning: nd.ZoningSystem = None,
                 lower_running_zones: List[Any] = None,
                 lower_distributor_kwargs: Dict[str, Any] = None,
                 compile_zoning_system: nd.ZoningSystem = None,
                 report_lower_vectors: bool = True,
                 process_count: int = constants.PROCESS_COUNT,
                 upper_model_process_count: int = None,
                 lower_model_process_count: int = None,
                 ):
        # Make sure all are set if one is
        lower_args = [lower_model_method, lower_model_zoning, lower_running_zones]
        if not all([x is not None for x in lower_args]):
            # Check they're not all None still
            if not all([x is None for x in lower_args]):
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
        if upper_distributor_kwargs is None:
            upper_distributor_kwargs = dict()
        if lower_distributor_kwargs is None:
            lower_distributor_kwargs = dict()

        if upper_model_process_count is None:
            upper_model_process_count = process_count
        if lower_model_process_count is None:
            lower_model_process_count = process_count

        # Assign attributes
        self.running_segmentation = running_segmentation
        self.upper_model_process_count = upper_model_process_count
        self.lower_model_process_count = lower_model_process_count

        self.upper_model_zoning = upper_model_zoning
        self.upper_running_zones = upper_running_zones
        self.upper_distributor_kwargs = upper_distributor_kwargs
        self.lower_model_zoning = lower_model_zoning
        self.lower_running_zones = lower_running_zones
        self.lower_distributor_kwargs = lower_distributor_kwargs
        self.report_lower_vectors = report_lower_vectors

        # Control output zoning systems depending on what we've been given
        if compile_zoning_system is not None:
            self.compile_zoning_system = compile_zoning_system
        else:
            if lower_model_zoning is not None:
                self.compile_zoning_system = lower_model_zoning
            else:
                self.compile_zoning_system = upper_model_zoning

        # TODO(BT): Integrate code to allow the output_zoning to be different
        self.output_zoning = self.compile_zoning_system

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
            'Distribution Model Iteration: %s' % str(self.iteration_name),
            '',
            '### Upper Model ###',
            'vector_export: %s' % self.upper.export_paths.home,
            'report_export: %s' % self.upper.report_paths.home,
            '',
            '### Lower Model ###',
            'vector_export: %s' % self.lower.export_paths.home,
            'report_export: %s' % self.lower.report_paths.home,
            '',
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
        self._logger.info("Building arguments for the Upper Model")
        kwargs = self.arg_builder.build_upper_model_arguments(
            cache_dir=self.cache_paths.upper_trip_ends,
        )

        self._logger.info("Initialising the Upper Model")
        upper_model = self.upper_model_method.get_distributor(
                year=self.year,
                trip_origin=self.trip_origin,
                running_mode=self.running_mode,
                zoning_system=self.upper_model_zoning,
                running_zones=self.upper_running_zones,
                export_home=self.upper_export_home,
                process_count=self.upper_model_process_count,
                **self.upper_distributor_kwargs,
        )

        self._logger.info("Running the Upper Model")
        upper_model.distribute(**kwargs)
        self._logger.info("Upper Model Done!")

    def run_lower_model(self):
        if self.lower_model_method is None:
            self._logger.info(
                "Cannot run Lower Model as no method has been given to run "
                "this model."
            )
            return

        self._logger.info("Initialising the Lower Model")
        lower_model = self.lower_model_method.get_distributor(
                year=self.year,
                trip_origin=self.trip_origin,
                running_mode=self.running_mode,
                zoning_system=self.lower_model_zoning,
                running_zones=self.lower_running_zones,
                export_home=self.lower_export_home,
                process_count=self.lower_model_process_count,
                **self.lower_distributor_kwargs,
        )

        self._logger.info("Converting Upper Model Outputs for Lower Model")
        productions, attractions = self.arg_builder.read_lower_pa(
            cache_dir=self.cache_paths.lower_trip_ends,
            upper_model_matrix_dir=self.upper.export_paths.matrix_dir,
            external_matrix_output_dir=self.export_paths.upper_external_pa,
            lower_model_vector_report_dir=self.report_paths.lower_vector_reports_dir,
            report_vectors=self.report_lower_vectors,
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
        """Generates a standard set of matrix reports on the PA matrices"""
        # Make sure we have full PA matrices before running
        self._maybe_recombine_pa_matrices()

        # Generate needed arguments
        input_fname_template = self.running_segmentation.generate_template_file_name(
                file_desc="synthetic_pa",
                trip_origin=self.trip_origin,
                year=str(self.year),
                csv=True
            )
        cost_matrices = self.arg_builder.build_pa_report_arguments(
            self.compile_zoning_system,
        )

        matrix_reports.generate_matrix_reports(
            matrix_dir=pathlib.Path(self.export_paths.full_pa_dir),
            report_dir=pathlib.Path(self.report_paths.pa_reports_dir),
            matrix_segmentation=self.running_segmentation,
            matrix_zoning_system=self.compile_zoning_system,
            matrix_fname_template=input_fname_template,
            cost_matrices=cost_matrices,
            row_name='productions',
            col_name='attractions',
            report_prefix=f"{self.trip_origin}_{self.iteration_name}",
        )

    def _maybe_recombine_od_matrices(self) -> None:
        """Combine od-to and od-from matrices if needed"""
        if self.trip_origin == 'nhb':
            # TODO(BT): Make sure the expected OD matrices exist
            return

        # TODO(BT): Doesn't currently work as need tp segments
        in_path = pathlib.Path(self.export_paths.full_od_dir)
        out_path = pathlib.Path(self.export_paths.combined_od_dir)

        if file_ops.is_cache_older(original=in_path, cache=out_path):
            # Generate fname templates
            template = self.running_segmentation.generate_template_file_name(
                file_desc="{matrix_format}",
                trip_origin=self.trip_origin,
                year=str(self.year),
                compressed=True,
            )
            template_fn = functools.partial(template.format, segment_params="{segment_params}")

            mat_utils.combine_od_to_from_matrices(
                import_dir=in_path,
                export_dir=out_path,
                segmentation=self.running_segmentation,
                od_fname_template=template_fn(matrix_format=self._od_matrix_desc),
                od_from_fname_template=template_fn(matrix_format=self._od_from_matrix_desc),
                od_to_fname_template=template_fn(matrix_format=self._od_to_matrix_desc),
            )

    def _maybe_recombine_pa_matrices(self) -> None:
        """Combine pa matrices if it hasn't been done yet"""
        # Build the I/O paths
        in_paths = [
            pathlib.Path(self.lower.export_paths.matrix_dir),
            pathlib.Path(self.export_paths.upper_external_pa),
        ]
        out_path = pathlib.Path(self.export_paths.full_pa_dir)

        # Only recombine if cache is older than original files
        if file_ops.is_cache_older(original=in_paths, cache=out_path):
            self._recombine_pa_matrices()

    def _recombine_pa_matrices(self):
        # ## GET THE FULL PA MATRICES ## #
        if self.lower_model_method is not None:
            # External should be made by  lower tier
            self._logger.info("Combining Upper and Lower Tier Matrices")
            import_dirs = [
                self.lower.export_paths.matrix_dir,
                self.export_paths.upper_external_pa,
            ]
            matrix_processing.combine_partial_matrices(
                import_dirs=import_dirs,
                export_dir=self.export_paths.full_pa_dir,
                segmentation=self.running_segmentation,
                import_suffixes=[None, self.arg_builder._external_suffix],
                trip_origin=self.trip_origin,
                year=str(self.year),
                file_desc=self._pa_matrix_desc,
            )
        else:
            self._logger.info("Copying over Upper Tier Matrices")
            file_ops.copy_segment_files(
                src_dir=self.upper.export_paths.matrix_dir,
                dst_dir=self.export_paths.full_pa_dir,
                segmentation=self.running_segmentation,
                trip_origin=self.trip_origin,
                year=str(self.year),
                file_desc=self._pa_matrix_desc,
                compressed=True,
            )

    def _maybe_translate_matrices_for_compile(self,
                                              matrices_path: nd.PathLike,
                                              matrices_desc: str,
                                              ) -> nd.PathLike:
        """Translates the matrices for compilation if they need it

        Returns the path to the translated matrices if translated, otherwise
        returns the given matrices_path

        Parameters
        ----------
        matrices_path:
            Path the the matrices that might need converting. If matrices
            need converting, a sub folder is made named
            DistributionModel._translated_dir_name -> "translated"

        Returns
        -------
        path:
            The path to the 'output' matrices from this process.
            If nothing needs converting, this path is the same as
            matrices_path.
        """
        # Init
        translation_weight_col = 'weight'

        # Figure out what the current zoning is
        if self.lower_model_zoning is not None:
            current_zoning = self.lower_model_zoning
        else:
            current_zoning = self.upper_model_zoning

        if current_zoning == self.compile_zoning_system:
            return matrices_path

        # If here, a translation needs doing
        out_dir = os.path.join(matrices_path, self._translated_dir_name)
        file_ops.create_folder(out_dir)

        # Get the translations
        pop_trans, emp_trans = translation.get_long_pop_emp_translations(
            from_zoning_system=current_zoning,
            to_zoning_system=self.compile_zoning_system,
            weight_col_name=translation_weight_col
        )

        desc = "Translating matrices for compilation"
        total = len(self.running_segmentation)
        for segment_params in tqdm.tqdm(self.running_segmentation, desc=desc, total=total):
            # Read in DF
            fname = self.running_segmentation.generate_file_name(
                trip_origin=self.trip_origin,
                year=str(self.year),
                file_desc=matrices_desc,
                segment_params=segment_params,
                compressed=True,
            )
            path = os.path.join(matrices_path, fname)
            df = file_ops.read_df(path, index_col=0)

            # Make sure index and columns are the same type
            df.columns = df.columns.astype(df.index.dtype)

            # Translate
            df = translation.pandas_matrix_zone_translation(
                matrix=df,
                row_translation=pop_trans,
                col_translation=emp_trans,
                from_zone_col=current_zoning.col_name,
                to_zone_col=self.compile_zoning_system.col_name,
                factors_col=translation_weight_col,
                from_unique_zones=current_zoning.unique_zones,
                to_unique_zones=self.compile_zoning_system.unique_zones,
            )

            # Write new matrix out
            file_ops.write_df(df, os.path.join(out_dir, fname))

        return out_dir

    def run_pa_to_od(self):
        # TODO(BT): Make sure the upper and lower matrices exist!

        # ## GET THE FULL PA MATRICES ## #
        self._maybe_recombine_pa_matrices()

        # ## CONVERT HB PA TO OD ## #
        if self.trip_origin == 'hb':
            self._logger.info("Converting HB PA matrices to OD")
            kwargs = self.arg_builder.build_pa_to_od_arguments()
            pa_to_od.build_od_from_fh_th_factors(
                pa_import=self.export_paths.full_pa_dir,
                od_export=self.export_paths.full_od_dir,
                pa_matrix_desc=self._pa_matrix_desc,
                od_to_matrix_desc=self._od_to_matrix_desc,
                od_from_matrix_desc=self._od_from_matrix_desc,
                base_year=self.year,
                years_needed=[self.year],
                **kwargs
            )

        # ## MOVE NHB TO OD DIR ## #
        elif self.trip_origin == 'nhb':
            # they're already OD anyway, just need a little name change
            self._logger.info("Copying NHB PA matrices to OD")
            matrix_processing.copy_nhb_matrices(
                import_dir=self.export_paths.full_pa_dir,
                export_dir=self.export_paths.full_od_dir,
                replace_pa_with_od=True,
                pa_matrix_desc=self._pa_matrix_desc,
                od_matrix_desc=self._od_matrix_desc,
            )

        else:
            raise ValueError(
                f"Don't know how to compile PA matrices to OD for trip "
                f"origin'{self.trip_origin}'."
            )

    def run_od_matrix_reports(self):
        """Generates a standard set of matrix reports on the OD matrices"""
        # Make sure we have full OD matrices before running
        self._maybe_recombine_od_matrices()

        print("Combined")
        exit()

        # TODO: OD to and OD from to add (for directional OD) OR just compile to OD?
        #  OD report arguments
        input_fname_template = self.running_segmentation.generate_template_file_name(
            file_desc="synthetic_od",
            trip_origin=self.trip_origin,
            year=str(self.year),
            csv=True
        )
        print(input_fname_template)
        cost_matrices = self.arg_builder.build_od_report_arguments(
            self.compile_zoning_system,
        )

        matrix_reports.generate_matrix_reports(
            matrix_dir=pathlib.Path(self.export_paths.combined_od_dir),
            report_dir=pathlib.Path(self.report_paths.od_reports_dir),
            matrix_segmentation=self.running_segmentation,
            matrix_zoning_system=self.compile_zoning_system,
            matrix_fname_template=input_fname_template,
            cost_matrices=cost_matrices,
            row_name='origins',
            col_name='destinations',
        )

    def compile_to_assignment_format(self):
        """TfN Specific helper function to compile outputs into assignment format

        This should really be the job of NorMITs Matrix tools! Move there
        once we create an object of it.

        Returns
        -------

        """
        # TODO(BT): NEED TO OUTPUT SPLITTING FACTORS

        # TODO(BT): UPDATE build_compile_params() to use segmentation levels
        m_needed = self.running_segmentation.segments['m'].unique()

        # NoHAM should be tp split
        tp_needed = [1, 2, 3, 4]

        if self.running_mode in [nd.Mode.CAR, nd.Mode.BUS]:
            # Compile to NoHAM format
            compile_params_paths = matrix_processing.build_compile_params(
                import_dir=self.export_paths.full_od_dir,
                export_dir=self.export_paths.compiled_od_dir,
                matrix_format=self._od_matrix_desc,
                years_needed=[self.year],
                m_needed=m_needed,
                tp_needed=tp_needed,
            )

            matrix_processing.compile_matrices(
                mat_import=self.export_paths.full_od_dir,
                mat_export=self.export_paths.compiled_od_dir,
                compile_params_path=compile_params_paths[0],
            )

            # TODO(BT): Build in DM imports!
            if self.running_mode == nd.Mode.CAR:
                occupancy_fname = 'car_vehicle_occupancies.csv'

            elif self.running_mode == nd.Mode.BUS:
                occupancy_fname = 'bus_vehicle_occupancies.csv'

            else:
                raise ValueError(
                    "This Error shouldn't be possible. The code must have "
                    "been updated without checking here!"
                )

            occupancies = pd.read_csv(os.path.join(
                r'I:\NorMITs Demand\import',
                'vehicle_occupancies',
                occupancy_fname,
            ))

            # Need to convert into hourly average PCU for noham
            vehicle_occupancy.people_vehicle_conversion(
                mat_import=self.export_paths.compiled_od_dir,
                mat_export=self.export_paths.compiled_od_dir_pcu,
                car_occupancies=occupancies,
                mode=m_needed[0],
                method='to_vehicles',
                out_format='wide',
                hourly_average=True,
            )

        elif self.running_mode == nd.Mode.TRAIN:
            self._maybe_recombine_pa_matrices()

            # Translate matrices if needed
            compile_in_path = self._maybe_translate_matrices_for_compile(
                matrices_path=self.export_paths.full_pa_dir,
                matrices_desc=self._pa_matrix_desc,
            )

            self._logger.info("Compiling NoRMS VDM Format")
            matrix_processing.compile_norms_to_vdm(
                mat_import=compile_in_path,
                mat_export=self.export_paths.compiled_pa_dir,
                params_export=self.export_paths.compiled_pa_dir,
                year=self.year,
                m_needed=m_needed,
                internal_zones=self.output_zoning.internal_zones.tolist(),
                external_zones=self.output_zoning.external_zones.tolist(),
                matrix_format=self._pa_matrix_desc,
            )

        else:
            raise ValueError(
                "I don't know how to compile mode %s into an assignment model "
                "format :("
                % self.running_mode.value
            )
