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

from normits_demand.models import Tram
from normits_demand.pathing import TramImportPaths

# GLOBAL VARIABLES
years = [2018, 2033]
years = [2018]
scenario = "SC01_JAM"
notem_iter = '4.1'
tram_import_home = r"I:\NorMITs Demand\import\modal\tram\tram_pa"
hb_prod_tram_import_version = '1.0'
hb_attr_tram_import_version = '1.0'
nhb_prod_tram_import_version = '1.0'
nhb_attr_tram_import_version = '1.0'

export_home = r'E:\NorMITs Demand\NoTEM'


def main():

    import_builder = TramImportPaths(
        import_home=tram_import_home,
        scenario=scenario,
        years=years,
        hb_prod_tram_import_version=hb_prod_tram_import_version,
        hb_attr_tram_import_version=hb_attr_tram_import_version,
        nhb_prod_tram_import_version=nhb_prod_tram_import_version,
        nhb_attr_tram_import_version=nhb_attr_tram_import_version,
    )

    n = Tram(
        years=years,
        scenario=scenario,
        iteration_name=notem_iter,
        import_builder=import_builder,
        export_home=export_home
    )

    n.run_tram(
        generate_all=True,
        generate_hb=False,
        generate_hb_production=False,
        generate_hb_attraction=False,
        generate_nhb=False,
        generate_nhb_production=False,
        generate_nhb_attraction=False,
    )


if __name__ == '__main__':
    main()
