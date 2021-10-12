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
from normits_demand.models import NoTEM
from normits_demand.pathing import NoTEMImportPaths


# GLOBAL VARIABLES
years = [2018]
scenario = "NTEM"
notem_iter = '4'
lu_drive = "I:/"
by_iteration = "iter3d"
fy_iteration = "iter3d"
notem_import_home = r"I:\NorMITs Demand\import\NoTEM"
notem_export_home = r"C:\Data\Nirmal_Atkins"
#notem_export_home = r"E:\NoTEM"


def main():
    hb_production_import_version = '2.0'
    hb_attraction_import_version = '1.5'
    nhb_production_import_version = '1.5'

    import_builder = NoTEMImportPaths(
        import_home=notem_import_home,
        scenario=scenario,
        years=years,
        land_use_import_home=lu_drive,
        by_land_use_iter=by_iteration,
        fy_land_use_iter=fy_iteration,
        hb_production_import_version=hb_production_import_version,
        hb_attraction_import_version=hb_attraction_import_version,
        nhb_production_import_version=nhb_production_import_version,
    )

    n = NoTEM(
        years=years,
        scenario=scenario,
        iteration_name=notem_iter,
        import_builder=import_builder,
        export_home=notem_export_home,
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
