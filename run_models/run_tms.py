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
import normits_demand as nd
from normits_demand import constants as consts
from normits_demand.models import TravelMarketSynthesiser

from normits_demand.pathing.travel_market_synthesiser import ExternalModelArgumentBuilder
from normits_demand.pathing.travel_market_synthesiser import GravityModelArgumentBuilder

# Constants
base_year = 2018
scenario = consts.SC00_NTEM
tms_import_home = r"I:\NorMITs Demand\import"
tms_export_home = r"E:\TMS"
notem_iteration_name = '4'
notem_export_home = r"I:\NorMITs Demand\NoTEM"


def main():
    mode = nd.Mode.CAR
    # mode = nd.Mode.BUS

    if mode == nd.Mode.CAR:
        zoning_system = nd.get_zoning_system('noham')
        internal_tld_name = 'p_m_standard_bands'
        external_tld_name = 'p_m_large_bands'
        hb_running_seg = nd.get_segmentation_level('hb_p_m_car')
        nhb_running_seg = nd.get_segmentation_level('nhb_p_m_car')
        intrazonal_cost_infill = 0.5
    elif mode == nd.Mode.BUS:
        zoning_system = nd.get_zoning_system('noham')
        internal_tld_name = 'p_m_standard_bands'
        external_tld_name = 'p_m_large_bands'
        hb_running_seg = nd.get_segmentation_level('hb_p_m_bus')
        nhb_running_seg = nd.get_segmentation_level('nhb_p_m_bus')
        intrazonal_cost_infill = 0.4
    else:
        raise ValueError(
            "Don't know what mode %s is!" % mode
        )

    em_arg_builder = ExternalModelArgumentBuilder(
        import_home=tms_import_home,
        base_year=base_year,
        scenario=scenario,
        zoning_system=zoning_system,
        internal_tld_name=internal_tld_name,
        external_tld_name=external_tld_name,
        intrazonal_cost_infill=intrazonal_cost_infill,
        hb_running_segmentation=hb_running_seg,
        nhb_running_segmentation=nhb_running_seg,
        notem_iteration_name=notem_iteration_name,
        notem_export_home=notem_export_home,
    )

    gm_arg_builder = GravityModelArgumentBuilder()

    tms = TravelMarketSynthesiser(
        year=base_year,
        running_mode=mode,
        zoning_system=zoning_system,
        external_model_arg_builder=em_arg_builder,
        gravity_model_arg_builder=gm_arg_builder,
        export_home=tms_export_home,
    )

    tms.run(
        run_all=False,
        run_external_model=True,
        run_gravity_model=False,
        run_pa_to_od=False,
    )


if __name__ == '__main__':
    main()
