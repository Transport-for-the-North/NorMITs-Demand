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
import os.path
from typing import List
from typing import Dict

# Third Party

# Local Imports
import normits_demand as nd

from normits_demand.models import HBProductionModel
from normits_demand.models import NHBProductionModel
from normits_demand.models import HBAttractionModel
from normits_demand.models import NHBAttractionModel

from normits_demand.pathing import NoTEMPaths

from normits_demand.utils import file_ops
from normits_demand.utils import general as du


class NoTEM(NoTEMPaths):

    def __init__(self,
                 years: List[int],
                 scenario: str,
                 import_home: nd.PathLike,
                 export_home: nd.PathLike,

                 *args,
                 **kwargs,
                 ):
        """
        Assigns the attributes needed for NoTEM model.

        Parameters
        ----------
        years:
            List of years to run NoTEM for. Will assume that the smallest
            year is the base year.

        scenario:
            The name of the scenario to run for.

        land_use_import_home:
            Path to the base directory of land use outputs.

        by_land_use_iter:
            String containing base year land use iteration Eg: 'iter3b'.

        fy_land_use_iter:
            String containing future year land use iteration Eg: 'iter3b'.
        """
        # Validate inputs
        file_ops.check_path_exists(import_home)

        # Assign
        self.years = years
        self.scenario = scenario

        # Generate the import and export paths
        super().__init__(
            years=self.years,
            scenario=self.scenario,
            import_home=import_home,
            export_home=export_home,
            *args,
            **kwargs,
        )

    def run(self,
            generate_all: bool = False,
            generate_hb: bool = False,
            generate_hb_production: bool = False,
            generate_hb_attraction: bool = False,
            generate_nhb: bool = False,
            generate_nhb_production: bool = False,
            generate_nhb_attraction: bool = False,
            verbose: bool = True,
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

        verbose:
            Whether to print progress updates to the terminal while running
            or not.

        Returns
        -------
        None
        """
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

        # Run the models
        if generate_hb_production:
            self._generate_hb_production(verbose)

        if generate_hb_attraction:
            self._generate_hb_attraction(verbose)

        if generate_nhb_production:
            self._generate_nhb_production(verbose)

        if generate_nhb_attraction:
            self._generate_nhb_attraction(verbose)

    def _generate_hb_production(self, verbose: bool) -> None:
        """
        Runs home based Production trip end models
        """
        import_files = self.generate_hb_production_imports()

        # Runs the home based Production model
        hb_prod = HBProductionModel(
            **import_files,
            constraint_paths=None,
            export_home=self.hb_production.export_paths.home,
        )

        hb_prod.run(
            export_pure_demand=True,
            export_fully_segmented=True,
            export_notem_segmentation=True,
            export_reports=True,
            verbose=verbose,
        )

    def _generate_hb_attraction(self, verbose: bool) -> None:
        """
        Runs the home based Attraction trip end model
        """
        # ## GENERATE THE NEEDED PATHS ## #
        # Runs the module to create import dictionary
        imports = self.generate_hb_attraction_imports()

        # Get the hb productions
        export_paths = self.hb_production.export_paths
        control_production_paths = {y: export_paths.notem_segmented[y] for y in self.years}

        # ## INSTANTIATE AND RUN THE MODEL ## #
        hb_attr = HBAttractionModel(
            **imports,
            production_balance_paths=control_production_paths,
            constraint_paths=None,
            export_home=self.hb_attraction.export_paths.home,
        )

        hb_attr.run(
            export_pure_attractions=True,
            export_fully_segmented=False,
            export_notem_segmentation=True,
            export_reports=True,
            verbose=verbose,
        )

    def _generate_nhb_production(self, verbose: bool) -> None:
        """
        Runs the non-home based Production trip end model
        """
        # ## GENERATE THE NEEDED PATHS ## #
        # Runs the module to create import dictionary
        imports = self.generate_nhb_production_imports()

        # Get the hb attractions
        export_paths = self.hb_attraction.export_paths
        hb_attraction_paths = {y: export_paths.notem_segmented[y] for y in self.years}

        # ## INSTANTIATE AND RUN THE MODEL ## #
        nhb_prod = NHBProductionModel(
            **imports,
            hb_attraction_paths=hb_attraction_paths,
            export_home=self.nhb_production.export_paths.home,
            constraint_paths=None,
        )

        nhb_prod.run(
            export_nhb_pure_demand=True,
            export_fully_segmented=False,
            export_notem_segmentation=True,
            export_reports=True,
            verbose=verbose,
        )

    def _generate_nhb_attraction(self, verbose: bool) -> None:
        """
        Runs non home based Attraction trip end models.
        """
        # ## GENERATE THE NEEDED PATHS ## #
        # No Imports currently needed for this model!
        # imports = self.generate_nhb_attraction_imports()

        # Get the hb attractions
        export_paths = self.hb_attraction.export_paths
        hb_attraction_paths = {y: export_paths.notem_segmented[y] for y in self.years}

        # Get the nhb productions
        export_paths = self.nhb_production.export_paths
        nhb_production_paths = {y: export_paths.notem_segmented[y] for y in self.years}

        # ## INSTANTIATE AND RUN THE MODEL ## #
        nhb_attr = NHBAttractionModel(
            hb_attraction_paths=hb_attraction_paths,
            nhb_production_paths=nhb_production_paths,
            export_home=self.nhb_attraction.export_paths.home,
            constraint_paths=None
        )

        nhb_attr.run(
            export_nhb_pure_attractions=True,
            export_notem_segmentation=False,
            export_reports=True,
            verbose=verbose,
        )
