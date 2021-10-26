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

# GLOBAL VARIABLES
years = [2018, 2033]
years = [2018]
scenario = "SC01_JAM"
notem_iter = '4.1'
tram_import_home = r"I:\NorMITs Demand\import\modal\tram\tram_pa"

export_home = r'E:\NorMITs Demand\NoTEM'


def main():
    n = Tram(
        years=years,
        scenario=scenario,
        iteration_name=notem_iter,
        import_home=tram_import_home,
        export_home=export_home
    )
    n.run_tram(
        verbose=True
    )


if __name__ == '__main__':
    main()
