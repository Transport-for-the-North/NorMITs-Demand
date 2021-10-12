# -*- coding: utf-8 -*-
"""
Created on: Wednesday Sept 22 2021
Updated on:

Original author: Nirmal Kumar
Last update made by:
Other updates made by:

File purpose:
Master run file to run tram inclusion
"""
import os
from typing import List


import normits_demand as nd
from normits_demand.utils import file_ops
from normits_demand.pathing import NoTEMExportPaths
from normits_demand.models.notem import TramInclusion
from normits_demand.utils import timing


class Tram(NoTEMExportPaths):
    _running_report_fname = 'running_parameters.txt'

    def __init__(self,
                 years: List[int],
                 scenario: str,
                 iteration_name: str,
                 import_home: nd.PathLike,
                 export_home: nd.PathLike,
                 ):
        """
        Assigns the attributes needed for tram inclusion model.

        Parameters
        ----------
        years:
            List of years to run tram inclusion for. Will assume that the smallest
            year is the base year.

        iteration_name:
            The name of this iteration of the NoTEM models. Will have 'iter'
            prepended to create the folder name. e.g. if iteration_name was
            set to '3i' the iteration folder would be called 'iter3i'.

        scenario:
            The name of the scenario to run for.

        import_home:
            The home location where all the import files are located.

        export_home:
            The home where all the export paths should be built from. See
            nd.pathing.NoTEMExportPaths for more info on how these paths
            will be built.
        """
        # Validate inputs
        file_ops.check_path_exists(import_home)


        # Assign
        self.years = years
        self.scenario = scenario
        self.import_home = import_home

        # Generate the export paths
        super().__init__(
            export_home=export_home,
            path_years=self.years,
            scenario=scenario,
            iteration_name=iteration_name,
        )

        # Create a logger
        logger_name = "%s.%s" % (__name__, self.__class__.__name__)
        log_file_path = os.path.join(self.export_home, self._log_fname)
        self._logger = nd.get_logger(
            logger_name=logger_name,
            log_file_path=log_file_path,
            instantiate_msg="Initialised new NoTEM Logger",
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
        Runs tram inclusion for the notem trip end models based on the criteria given.

        Parameters
        ----------
        generate_all:
            Runs tram inclusion for both home based and non home based trip end models.

        generate_hb:
            Runs tram inclusion for the home based trip end models only.

        generate_hb_production:
            Runs tram inclusion for the home based production trip end model only.

        generate_hb_attraction:
            Runs tram inclusion for the home based attraction trip end model only.

        generate_nhb:
            Runs tram inclusion for the non home based trip end models only.

        generate_nhb_production:
            Runs tram inclusion for the non home based production trip end model only.

        generate_nhb_attraction:
            Runs tram inclusion for the non home based attraction trip end model only.

        verbose:
            Whether to print progress updates to the terminal while running
            or not.

        Returns
        -------
        None
        """
        # TODO(BT): Add checks to make sure input paths exist when models
        #  depend on one another
        start_time = timing.current_milli_time()
        self._logger.info("Starting a new run of NoTEM")

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
            self._generate_hb_production(verbose)

        if generate_hb_attraction:
            self._generate_hb_attraction(verbose)

        if generate_nhb_production:
            self._generate_nhb_production(verbose)

        if generate_nhb_attraction:
            self._generate_nhb_attraction(verbose)

        end_time = timing.current_milli_time()
        time_taken = timing.time_taken(start_time, end_time)
        self._logger.info("NoTEM run complete! Took %s" % time_taken)

    def _generate_hb_production(self, verbose: bool) -> None:
        """
        Runs tram inclusion for home based Production trip end models
        """
        self._logger.debug("Generating Home-Based Production Model imports")
        tram_data = os.path.join(self.import_home,"tram_hb_productions.csv")

        export_paths = self.hb_production.export_paths
        hb_production_paths = {y: export_paths.notem_segmented[y] for y in self.years}

        # Runs the home based Production model
        self._logger.debug("Instantiating Home-Based Production Model")
        hb_prod = TramInclusion(
            tram_data=tram_data,
            notem_outputs=hb_production_paths,
            export_home=self.hb_production.export_paths.home,
        )

        self._logger.info("Running the Home-Based Production Model")
        hb_prod.run(
            verbose=verbose,
        )

    def _generate_hb_attraction(self, verbose: bool) -> None:
        """
        Runs the home based Attraction trip end model
        """
        self._logger.debug("Generating Home-Based Attraction Model imports")
        tram_data = os.path.join(self.import_home, "tram_hb_attractions.csv")

        export_paths = self.hb_attraction.export_paths
        hb_attraction_paths = {y: export_paths.notem_segmented[y] for y in self.years}

        self._logger.debug("Instantiating Home-Based Attraction Model")
        hb_attr = TramInclusion(
            tram_data=tram_data,
            notem_outputs=hb_attraction_paths,
            export_home=self.hb_attraction.export_paths.home,
        )

        self._logger.info("Running the Home-Based Attraction Model")
        hb_attr.run(
                   verbose=verbose,
        )

    def _generate_nhb_production(self, verbose: bool) -> None:
        """
        Runs the non-home based Production trip end model
        """
        self._logger.debug("Generating Non-Home-Based Production Model imports")
        tram_data = os.path.join(self.import_home, "tram_nhb_productions.csv")

        export_paths = self.nhb_production.export_paths
        nhb_production_paths = {y: export_paths.notem_segmented[y] for y in self.years}

        self._logger.debug("Instantiating Non-Home-Based Production Model")
        nhb_prod = TramInclusion(
            tram_data=tram_data,
            notem_outputs=nhb_production_paths,
            export_home=self.nhb_production.export_paths.home,
        )

        self._logger.info("Running the Non-Home-Based Production Model")
        nhb_prod.run(
            verbose=verbose,
        )

    def _generate_nhb_attraction(self, verbose: bool) -> None:
        """
        Runs non home based Attraction trip end models.
        """
        self._logger.debug("Generating Non-Home-Based Attraction Model imports")
        tram_data = os.path.join(self.import_home, "tram_hb_attractions.csv")

        export_paths = self.nhb_attraction.export_paths
        nhb_attraction_paths = {y: export_paths.notem_segmented[y] for y in self.years}

        self._logger.debug("Instantiating Non-Home-Based Attraction Model")
        nhb_attr = TramInclusion(
            tram_data=tram_data,
            notem_outputs=nhb_attraction_paths,
            export_home=self.nhb_attraction.export_paths.home,
        )

        self._logger.info("Running the Non-Home-Based Attraction Model")
        nhb_attr.run(
            verbose=verbose,
        )

export_home = r"C:\Data\Nirmal_Atkins"


def main():
    n = TramInclusion(
        tram_data_paths=tram_paths,
        notem_outputs=notem_outputs,
        export_home=export_home,
    )
    n.run(
        verbose=True
    )


if __name__ == '__main__':
    main()
