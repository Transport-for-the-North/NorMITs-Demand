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


class TravelMarketSynthesiser:

    def __init__(self):
        pass

    def run(self,
            run_external_model: bool = True,
            run_gravity_model: bool = True,
            run_pa_to_od: bool = True,
            ) -> None:
        """Runs the components of TMS

        Run parameters are based off of the parameters passed into the class
        constructors. Read the documentation of individual run functions to
        see the parameters used in each step.

        Parameters
        ----------
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
        pass