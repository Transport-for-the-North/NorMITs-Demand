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

from normits_demand.models.tram_model import TramModel
from normits_demand.pathing import TramImportPaths

# GLOBAL VARIABLES
years = [2018, 2033]
scenario = "SC01_JAM"
notem_iter = '9.2'
tram_import_home = r"I:\NorMITs Demand\import\modal\tram\tram_pa"
notem_import_home = r"I:\NorMITs Demand\NoTEM"
hb_prod_tram_import_version = '1.0'
hb_attr_tram_import_version = '1.0'
nhb_prod_tram_import_version = '1.0'
nhb_attr_tram_import_version = '1.0'

export_home = r"I:\NorMITs Demand\Tram"


def main():
    import_builder = TramImportPaths(
        tram_import_home=tram_import_home,
        notem_import_home=notem_import_home,
        scenario=scenario,
        iter_name=notem_iter,
        years=years,
        hb_prod_tram_import_version=hb_prod_tram_import_version,
        hb_attr_tram_import_version=hb_attr_tram_import_version,
        nhb_prod_tram_import_version=nhb_prod_tram_import_version,
        nhb_attr_tram_import_version=nhb_attr_tram_import_version,
    )

    n = TramModel(
        years=years,
        scenario=scenario,
        iteration_name=notem_iter,
        import_builder=import_builder,
        export_home=export_home,
    )

    n.run_tram(
        generate_all=False,
        generate_hb=False,
        generate_hb_production=False,
        generate_hb_attraction=False,
        generate_nhb=False,
        generate_nhb_production=False,
        generate_nhb_attraction=True,
    )


if __name__ == '__main__':
    main()
