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
years = [2018, 2027]
scenario = "SC01_JAM"
notem_iter = '4'
tram_import_home = r"I:\Data\Light Rail"

export_home = r"C:\Data\Nirmal_Atkins"


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
