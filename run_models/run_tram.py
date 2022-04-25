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
import sys

sys.path.append("..")
# pylint: disable=import-error,wrong-import-position
import normits_demand as nd
from normits_demand import core as nd_core
from normits_demand.models.tram_model import TramModel
from normits_demand.pathing import TramImportPaths
# pylint: enable=import-error,wrong-import-position

# GLOBAL VARIABLES
# years = [2018, 2033, 2040, 2050]
YEARS = [2018]
SCENARIO = nd_core.Scenario.SC01_JAM
NOTEM_ITER = '9.7'
NOTEM_EXPORT_HOME = r"E:\NorMITs Demand\NoTEM"

TRAM_IMPORT_HOME = r"I:\NorMITs Demand\import\modal\tram\tram_pa"
TRAM_EXPORT_HOME = r"E:\NorMITs Demand\Tram"


def main():
    # Input versions
    hb_production_data_version = '1.1'
    hb_attraction_data_version = '1.1'
    nhb_production_data_version = '1.1'
    nhb_attraction_data_version = '1.1'

    # Define different balancing zones for each mode
    mode_balancing_zones = {5: nd.get_zoning_system("ca_sector_2020")}
    hb_balance_zoning = nd.BalancingZones.build_single_segment_group(
        nd.get_segmentation_level('tram_hb_output'),
        nd.get_zoning_system('gor'),
        "m",
        mode_balancing_zones,
    )
    nhb_balance_zoning = nd.BalancingZones.build_single_segment_group(
        nd.get_segmentation_level('tram_nhb_output'),
        nd.get_zoning_system('gor'),
        "m",
        mode_balancing_zones,
    )

    # Define which modes compete with tram
    tram_competitors = [nd.Mode.CAR, nd.Mode.BUS, nd.Mode.TRAIN]

    # Generate the imports
    notem_exports = nd.pathing.NoTEMExportPaths(
        path_years=YEARS,
        scenario=SCENARIO,
        iteration_name=NOTEM_ITER,
        export_home=NOTEM_EXPORT_HOME,
    )

    import_builder = TramImportPaths(
        years=YEARS,
        notem_exports=notem_exports,
        tram_import_home=TRAM_IMPORT_HOME,
        hb_production_data_version=hb_production_data_version,
        hb_attraction_data_version=hb_attraction_data_version,
        nhb_production_data_version=nhb_production_data_version,
        nhb_attraction_data_version=nhb_attraction_data_version,
    )

    # Instantiate and run the tram model
    n = TramModel(
        years=YEARS,
        scenario=SCENARIO,
        iteration_name=NOTEM_ITER,
        import_builder=import_builder,
        export_home=TRAM_EXPORT_HOME,
        tram_competitors=tram_competitors,
        hb_balance_zoning=hb_balance_zoning,
        nhb_balance_zoning=nhb_balance_zoning,
    )

    n.run_tram(
        generate_all=True,
        generate_hb=False,
        generate_nhb=False,
        generate_hb_production=False,
        generate_hb_attraction=False,
        generate_nhb_production=False,
        generate_nhb_attraction=False,
        before_after_report=False,
    )


if __name__ == '__main__':
    main()
