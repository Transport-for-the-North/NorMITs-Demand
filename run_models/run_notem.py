# -*- coding: utf-8 -*-
"""
Created on: Tues August 17 2021
Updated on:

Original author: Ben Taylor
Last update made by:
Other updates made by:

File purpose:
Master run file to run NoTEM
"""
import sys


sys.path.append("..")
# pylint: disable=import-error,wrong-import-position
import normits_demand as nd
from normits_demand import core as nd_core
from normits_demand.models import NoTEM
from normits_demand.pathing import NoTEMImportPaths
# pylint: enable=import-error,wrong-import-position


# GLOBAL VARIABLES
# years = [2018, 2033, 2040, 2050]
YEARS = [2018]
SCENARIO = nd_core.Scenario.SC01_JAM
NOTEM_ITER = '9.10'
LU_DRIVE = "I:/"
LU_BY_ITER = "iter3e"
LU_FY_ITER = "iter3e"
NOTEM_IMPORT_HOME = r"I:\NorMITs Demand\import\NoTEM"
NOTEM_EXPORT_HOME = r"F:\NorMITs Demand\NoTEM"


def main():
    hb_production_import_version = '3.0'
    hb_attraction_import_version = '2.3'
    nhb_production_import_version = '3.0'

    # Define different balancing zones for each mode
    mode_balancing_zones = {5: nd.get_zoning_system("ca_sector_2020")}
    hb_attraction_balance_zoning = nd.BalancingZones.build_single_segment_group(
        nd.get_segmentation_level('notem_hb_output'),
        nd.get_zoning_system('gor'),
        "m",
        mode_balancing_zones,
    )
    nhb_attraction_balance_zoning = nd.BalancingZones.build_single_segment_group(
        nd.get_segmentation_level('notem_nhb_output'),
        nd.get_zoning_system('gor'),
        "m",
        mode_balancing_zones,
    )

    import_builder = NoTEMImportPaths(
        import_home=NOTEM_IMPORT_HOME,
        scenario=SCENARIO,
        years=YEARS,
        land_use_import_home=LU_DRIVE,
        by_land_use_iter=LU_BY_ITER,
        fy_land_use_iter=LU_FY_ITER,
        hb_production_import_version=hb_production_import_version,
        hb_attraction_import_version=hb_attraction_import_version,
        nhb_production_import_version=nhb_production_import_version,
    )

    n = NoTEM(
        years=YEARS,
        scenario=SCENARIO,
        iteration_name=NOTEM_ITER,
        import_builder=import_builder,
        export_home=NOTEM_EXPORT_HOME,
        hb_attraction_balance_zoning=hb_attraction_balance_zoning,
        nhb_attraction_balance_zoning=nhb_attraction_balance_zoning,
    )
    n.run(
        generate_all=True,
        generate_hb=False,
        generate_nhb=False,
        generate_hb_production=False,
        generate_hb_attraction=False,
        generate_nhb_production=False,
        generate_nhb_attraction=False,
    )


if __name__ == '__main__':
    main()
