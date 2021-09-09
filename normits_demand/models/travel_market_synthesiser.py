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

# Third Party

# Local Imports
import normits_demand as nd

from normits_demand.pathing import TMSExportPaths

from normits_demand.utils import timing


class TravelMarketSynthesiser(TMSExportPaths):
    # ## Class Constants ## #
    __version__ = nd.version.__version__
    out_dir = "NorMITs Demand"

    def __init__(self):

        # Generate export paths
        super().__init__()

        # Create a logger
        # TODO (BT): Determine output file path
        logger_name = "%s.%s" % (__name__, self.__class__.__name__)
        # log_file_path = os.path.join(self.export_home, self._log_fname)
        self._logger = nd.get_logger(
            logger_name=logger_name,
            # log_file_path=log_file_path,
            instantiate_msg="Initialised new TMS Logger",
        )

    def run(self,
            run_all: bool = False,
            run_external_model: bool = False,
            run_gravity_model: bool = False,
            run_pa_to_od: bool = False,
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

        run_pa_to_od:
            Whether to run the PA to OD conversion process or not. This step
            depends on the external model and gravity model already being
            run - as these steps produce the PA matrices to convert.

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
            run_pa_to_od = True

        self._logger.debug("Running external model: %s" % run_external_model)
        self._logger.debug("Running gravity model: %s" % run_gravity_model)
        self._logger.debug("Running pa to od: %s" % run_pa_to_od)
        self._logger.debug("")

        # Check that we are actually running something
        if not all([run_external_model, run_gravity_model, run_pa_to_od]):
            self._logger.info(
                "All run args set to False. Not running anything"
            )

        # Run the models
        if run_external_model:
            self._run_external_model()

        if run_gravity_model:
            self._run_gravity_model()

        if run_pa_to_od:
            self._run_pa_to_od()

        # Log the time taken to run
        end_time = timing.current_milli_time()
        time_taken = timing.time_taken(start_time, end_time)
        self._logger.info("TMS run complete! Took %s" % time_taken)

    def _run_external_model(self):
        pass

    def _run_gravity_model(self):
        pass

    def _run_pa_to_od(self):
        pass

