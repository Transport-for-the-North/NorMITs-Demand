# -*- coding: utf-8 -*-
"""
    Master run file for MiTEM.
"""

##### IMPORTS #####
# Standard imports
import sys

# Third party imports

# Local imports
sys.path.append("..")
# pylint: disable=import-error,wrong-import-position
import run_notem
import normits_demand as nd
from normits_demand.models import MiTEM
from normits_demand.pathing import MiTEMImportPaths
# pylint: enable=import-error,wrong-import-position


##### CONSTANTS #####
years = run_notem.years
scenario = run_notem.scenario
mitem_iter = run_notem.notem_iter
lu_drive = run_notem.lu_drive
by_iteration = run_notem.by_iteration
fy_iteration = run_notem.fy_iteration
mitem_import_home = run_notem.notem_import_home
mitem_export_home = r"T:\MidMITs Demand\MiTEM"

##### CLASSES #####

##### FUNCTIONS #####
def main():
    """Run MiTEM."""
    hb_production_import_version = '2.1'
    hb_attraction_import_version = '1.6'
    nhb_production_import_version = '2.0'

    attraction_balance_zoning = nd.get_zoning_system('gor')

    import_builder = MiTEMImportPaths(
        import_home=mitem_import_home,
        scenario=scenario,
        years=years,
        land_use_import_home=lu_drive,
        by_land_use_iter=by_iteration,
        fy_land_use_iter=fy_iteration,
        hb_production_import_version=hb_production_import_version,
        hb_attraction_import_version=hb_attraction_import_version,
        nhb_production_import_version=nhb_production_import_version,
    )

    m = MiTEM(
        years=years,
        scenario=scenario,
        iteration_name=mitem_iter,
        import_builder=import_builder,
        export_home=mitem_export_home,
        attraction_balance_zoning=attraction_balance_zoning,
    )
    m.run(
        generate_all=True,
        generate_hb=False,
        generate_nhb=False,
        generate_hb_production=False,
        generate_hb_attraction=False,
        generate_nhb_production=False,
        generate_nhb_attraction=False,
    )

##### MAIN #####
if __name__ == '__main__':
    main()
