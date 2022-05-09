# -*- coding: utf-8 -*-
"""
Created on: Tues May 25th 2021
Updated on:

Original author: Ben Taylor
Last update made by: Nirmal Kumar
Other updates made by: Nirmal Kumar

File purpose:
NoTEM Class Frontend for calling all production and attraction models
"""
# Builtins
import os

from typing import List, Optional

# Third Party

# Local Imports
import normits_demand as nd

from normits_demand.models.notem import HBProductionModel
from normits_demand.models.notem import NHBProductionModel
from normits_demand.models.notem import HBAttractionModel
from normits_demand.models.notem import NHBAttractionModel
from normits_demand.models.notem.production_models import TripEndAdjustmentFactors

from normits_demand.pathing import NoTEMExportPaths
from normits_demand.utils import timing


class NoTEM:
    EXPORT_PATHS_CLASS = NoTEMExportPaths
    _running_report_fname = 'running_parameters.txt'
    _log_fname = "NoTEM_log.log"

    def __init__(self,
                 years: List[int],
                 scenario: str,
                 iteration_name: str,
                 import_builder: nd.pathing.NoTEMImportPathsBase,
                 export_home: nd.PathLike,
                 hb_attraction_balance_zoning: nd.BalancingZones = None,
                 nhb_attraction_balance_zoning: nd.BalancingZones = None,
                 trip_end_adjustments: Optional[List[TripEndAdjustmentFactors]] = None,
                 ):
        """
        Assigns the attributes needed for NoTEM model.

        Parameters
        ----------
        years:
            List of years to run NoTEM for. Will assume that the smallest
            year is the base year.

        iteration_name:
            The name of this iteration of the NoTEM models. Will have 'iter'
            prepended to create the folder name. e.g. if iteration_name was
            set to '3i' the iteration folder would be called 'iter3i'.

        scenario:
            The name of the scenario to run for.

        import_builder:
            A subclass of nd.pathing.NoTEMImportPathsBase. This class will
            be called on to build and grab all of the import paths for each
            model. See the aforementioned class for full detail on how an
            implementation of this class should look. Also see
            nd.pathing.NoTEMImportPaths for an example implementation.

        export_home:
            The home where all the export paths should be built from. See
            nd.pathing.NoTEMExportPaths for more info on how these paths
            will be built.

        hb_attraction_balance_zoning:
            The zoning systems to balance the home-based attractions to the productions
            at, for each segment of the attractions segmentation. A translation must exist
            between this and the running zoning system, which is MSOA by default.
            If left as None, then no spatial balance is done, only a segmental balance.

        nhb_attraction_balance_zoning:
            The zoning systems to balance the non-home-based attractions to the productions
            at, for each segment of the attractions segmentation. A translation must exist
            between this and the running zoning system, which is MSOA by default.
            If left as None, then no spatial balance is done, only a segmental balance.

        trip_end_adjustments: List[TripEndAdjustmentFactors], optional
            List of all adjustment factors to apply to the HB productions trip ends.
            Adjustments are applied one after another at to the HB productions.
        """
        # Validate inputs
        if not isinstance(import_builder, nd.pathing.NoTEMImportPathsBase):
            raise ValueError(
                'import_builder is not the correct type. Expected '
                '"nd.pathing.NoTEMImportPathsBase", but got %s'
                % type(import_builder)
            )

        # Assign
        self.years = years
        self.scenario = scenario
        self.import_builder = import_builder
        self.hb_attraction_balance_zoning = hb_attraction_balance_zoning
        self.nhb_attraction_balance_zoning = nhb_attraction_balance_zoning
        self.adjustment_factors = trip_end_adjustments

        # Generate the export paths
        self.exports = self.EXPORT_PATHS_CLASS(
            export_home=export_home,
            path_years=self.years,
            scenario=scenario,
            iteration_name=iteration_name,
        )

        # Create a logger
        logger_name = "%s.%s" % (nd.get_package_logger_name(), self.__class__.__name__)
        log_file_path = os.path.join(self.exports.export_home, self._log_fname)
        self._logger = nd.get_logger(
            logger_name=logger_name,
            log_file_path=log_file_path,
            instantiate_msg=f"Initialised new {self.name} Logger",
        )

        self._write_running_report()

    @property
    def name(self) -> str:
        """Name of the model."""
        return self.__class__.__name__

    def _write_running_report(self):
        """
        Outputs a simple report detailing inputs and outputs
        """
        # Define the lines to output
        out_lines = [
            'Code Version: %s' % str(nd.__version__),
            '%s Iteration: %s' % (self.name, str(self.exports.iteration_name)),
            'Scenario: %s' % str(self.scenario),
            '',
            '### HB Productions ###',
            'import_files: %s' % self.import_builder.generate_hb_production_imports(),
            'vector_export: %s' % self.exports.hb_production.export_paths.home,
            'report_export: %s' % self.exports.hb_production.report_paths.home,
            '',
            '### HB Attractions ###',
            'import_files: %s' % self.import_builder.generate_hb_attraction_imports(),
            'vector_export: %s' % self.exports.hb_attraction.export_paths.home,
            'report_export: %s' % self.exports.hb_attraction.report_paths.home,
            '',
            '### NHB Productions ###',
            'import_files: %s' % self.import_builder.generate_nhb_production_imports(),
            'vector_export: %s' % self.exports.nhb_production.export_paths.home,
            'report_export: %s' % self.exports.nhb_production.report_paths.home,
            '',
            '### NHB Attractions ###',
            'import_files: %s' % self.import_builder.generate_nhb_attraction_imports(),
            'vector_export: %s' % self.exports.nhb_attraction.export_paths.home,
            'report_export: %s' % self.exports.nhb_attraction.report_paths.home,
        ]

        # Write out to disk
        output_path = os.path.join(self.exports.export_home, self._running_report_fname)
        with open(output_path, 'w') as out:
            out.write('\n'.join(out_lines))

    def run(self,
            generate_all: bool = False,
            generate_hb: bool = False,
            generate_hb_production: bool = False,
            generate_hb_attraction: bool = False,
            generate_nhb: bool = False,
            generate_nhb_production: bool = False,
            generate_nhb_attraction: bool = False,
            ) -> None:
        """
        Runs the notem trip end models based on the criteria given.

        Parameters
        ----------
        generate_all:
            Runs both home based and non home based trip end models.

        generate_hb:
            Runs the home based trip end models only.

        generate_hb_production:
            Runs the home based production trip end model only.

        generate_hb_attraction:
            Runs the home based attraction trip end model only.

        generate_nhb:
            Runs the non home based trip end models only.

        generate_nhb_production:
            Runs the non home based production trip end model only.

        generate_nhb_attraction:
            Runs the non home based attraction trip end model only.

        Returns
        -------
        None
        """
        # TODO(BT): Add checks to make sure input paths exist when models
        #  depend on one another
        start_time = timing.current_milli_time()
        self._logger.info("Starting a new run of %s", self.name)

        # Determine which models to run
        if generate_all:
            generate_hb = True
            generate_nhb = True

        if generate_hb:
            generate_hb_production = True
            generate_hb_attraction = True

        if generate_nhb:
            generate_nhb_production = True
            generate_nhb_attraction = True

        self._logger.debug("Running hb productions: %s" % generate_hb_production)
        self._logger.debug("Running nhb productions: %s" % generate_nhb_production)
        self._logger.debug("Running hb attractions: %s" % generate_hb_attraction)
        self._logger.debug("Running nhb attractions: %s" % generate_nhb_attraction)
        self._logger.debug("")

        # Run the models
        if generate_hb_production:
            self._generate_hb_production()

        if generate_hb_attraction:
            self._generate_hb_attraction()

        if generate_nhb_production:
            self._generate_nhb_production()

        if generate_nhb_attraction:
            self._generate_nhb_attraction()

        end_time = timing.current_milli_time()
        time_taken = timing.time_taken(start_time, end_time)
        self._logger.info("%s run complete! Took %s" % (self.name, time_taken))

    def _generate_hb_production(self) -> None:
        """
        Runs home based Production trip end models
        """
        self._logger.debug("Generating Home-Based Production Model imports")
        import_files = self.import_builder.generate_hb_production_imports()

        # Runs the home based Production model
        self._logger.debug("Instantiating Home-Based Production Model")
        hb_prod = HBProductionModel(
            **import_files,
            constraint_paths=None,
            export_home=self.exports.hb_production.export_paths.home,
            trip_end_adjustments=self.adjustment_factors,
        )

        self._logger.info("Running the Home-Based Production Model")
        hb_prod.run(
            export_pure_demand=False,
            export_fully_segmented=False,
            export_notem_segmentation=True,
            export_reports=True,
        )

    def _generate_hb_attraction(self) -> None:
        """
        Runs the home based Attraction trip end model
        """
        self._logger.debug("Generating Home-Based Attraction Model imports")
        # Runs the module to create import dictionary
        imports = self.import_builder.generate_hb_attraction_imports()

        # Get the hb productions
        export_paths = self.exports.hb_production.export_paths
        control_production_paths = {y: export_paths.notem_segmented[y] for y in self.years}

        self._logger.debug("Instantiating Home-Based Attraction Model")
        hb_attr = HBAttractionModel(
            **imports,
            production_balance_paths=control_production_paths,
            constraint_paths=None,
            export_home=self.exports.hb_attraction.export_paths.home,
            balance_zoning=self.hb_attraction_balance_zoning,
        )

        self._logger.info("Running the Home-Based Attraction Model")
        hb_attr.run(
            export_pure_attractions=False,
            export_notem_segmentation=True,
            export_reports=True,
        )

    def _generate_nhb_production(self) -> None:
        """
        Runs the non-home based Production trip end model
        """
        self._logger.debug("Generating Non-Home-Based Production Model imports")
        # Runs the module to create import dictionary
        imports = self.import_builder.generate_nhb_production_imports()

        # Get the hb attractions
        export_paths = self.exports.hb_attraction.export_paths
        hb_attraction_paths = {y: export_paths.notem_segmented[y] for y in self.years}

        self._logger.debug("Instantiating Non-Home-Based Production Model")
        nhb_prod = NHBProductionModel(
            **imports,
            hb_attraction_paths=hb_attraction_paths,
            export_home=self.exports.nhb_production.export_paths.home,
            constraint_paths=None,
        )

        self._logger.info("Running the Non-Home-Based Production Model")
        nhb_prod.run(
            export_nhb_pure_demand=False,
            export_fully_segmented=False,
            export_notem_segmentation=True,
            export_reports=True,
        )

    def _generate_nhb_attraction(self) -> None:
        """
        Runs non home based Attraction trip end models.
        """
        self._logger.debug("Generating Non-Home-Based Attraction Model imports")
        # No Imports currently needed for this model!
        # imports = self.generate_nhb_attraction_imports()

        # Get the hb attractions
        export_paths = self.exports.hb_attraction.export_paths
        hb_attraction_paths = {y: export_paths.notem_segmented[y] for y in self.years}

        # Get the nhb productions
        export_paths = self.exports.nhb_production.export_paths
        nhb_production_paths = {y: export_paths.notem_segmented[y] for y in self.years}

        self._logger.debug("Instantiating Non-Home-Based Attraction Model")
        nhb_attr = NHBAttractionModel(
            hb_attraction_paths=hb_attraction_paths,
            nhb_production_paths=nhb_production_paths,
            export_home=self.exports.nhb_attraction.export_paths.home,
            constraint_paths=None,
            balance_zoning=self.nhb_attraction_balance_zoning,
        )

        self._logger.info("Running the Non-Home-Based Attraction Model")
        nhb_attr.run(
            export_nhb_pure_attractions=False,
            export_notem_segmentation=True,
            export_reports=True,
        )
