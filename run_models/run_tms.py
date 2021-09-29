# -*- coding: utf-8 -*-
"""
Created on: 09/09/2021
Updated on:

Original author: Ben Taylor
Last update made by:
Other updates made by:

File purpose:

"""
# Built-Ins
import sys

# Third Party

# Local Imports
sys.path.append("..")
from normits_demand import constants as consts
from normits_demand.models import TravelMarketSynthesiser

from normits_demand.pathing import TMSArgumentBuilder

# Constants
base_year = 2018
scenario = consts.SC00_NTEM
notem_iteration_name = '4'
notem_export_home = r"I:\NorMITs Demand\NoTEM\


def main():

    tms_arg_builder = TMSArgumentBuilder(
        base_year=base_year,
        scenario=scenario,
        notem_iteration_name=notem_iteration_name,
        notem_export_home=notem_export_home,
    )

    tms = TravelMarketSynthesiser(
        zoning_system=zoning_system,
        argument_builder=tms_arg_builder,
    )

    tms.run(
        run_all=False,
        run_external_model=True,
        run_gravity_model=False,
        run_pa_to_od=False,
    )


if __name__ == '__main__':
    main()
