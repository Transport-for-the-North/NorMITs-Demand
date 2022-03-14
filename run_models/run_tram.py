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
import normits_demand as nd
from normits_demand.models.tram_model import TramModel
from normits_demand.pathing import TramImportPaths

# GLOBAL VARIABLES
# years = [2018, 2033, 2040, 2050]
years = [2018]
scenario = nd.constants.SC01_JAM
notem_iter = '9.6'
tram_import_home = r"I:\NorMITs Demand\import\modal\tram\tram_pa"
notem_export_home = r"F:\NorMITs Demand\NoTEM"

export_home = r"F:\NorMITs Demand\Tram"


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
        path_years=years,
        scenario=scenario,
        iteration_name=notem_iter,
        export_home=notem_export_home,
    )

    import_builder = TramImportPaths(
        years=years,
        notem_exports=notem_exports,
        tram_import_home=tram_import_home,
        hb_production_data_version=hb_production_data_version,
        hb_attraction_data_version=hb_attraction_data_version,
        nhb_production_data_version=nhb_production_data_version,
        nhb_attraction_data_version=nhb_attraction_data_version,
    )

    # Instantiate and run the tram model
    n = TramModel(
        years=years,
        scenario=scenario,
        iteration_name=notem_iter,
        import_builder=import_builder,
        export_home=export_home,
        tram_competitors=tram_competitors,
        hb_balance_zoning=hb_balance_zoning,
        nhb_balance_zoning=nhb_balance_zoning,
    )

    n.run_tram(
        generate_all=False,
        generate_hb=False,
        generate_nhb=False,
        generate_hb_production=False,
        generate_hb_attraction=True,
        generate_nhb_production=False,
        generate_nhb_attraction=False,
        before_after_report=False,
    )


if __name__ == '__main__':
    main()
