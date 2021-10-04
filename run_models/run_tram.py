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
from normits_demand.models.notem import TramInclusion

# GLOBAL VARIABLES
tram_paths = {'hb_p': r"I:\Data\Light Rail\tram_hb_productions.csv",
              'hb_a': r"I:\Data\Light Rail\tram_hb_attractions.csv",
              'nhb_p': r"I:\Data\Light Rail\tram_nhb_productions.csv",
              'nhb_a': r"I:\Data\Light Rail\tram_nhb_attractions.csv",
              }
notem_outputs = {'hb_p': r"C:\Data\Nirmal_Atkins\iter4\NTEM\hb_productions\hb_msoa_notem_segmented_2018_dvec.pkl",
                 'hb_a': r"I:\NorMITs Demand\NoTEM\iter4\NTEM\hb_attractions\hb_msoa_notem_segmented_2018_dvec.pkl",
                 'nhb_p': r"I:\NorMITs Demand\NoTEM\iter4\NTEM\nhb_productions\nhb_msoa_notem_segmented_2018_dvec.pkl",
                 'nhb_a': r"I:\NorMITs Demand\NoTEM\iter4\NTEM\nhb_attractions\nhb_msoa_notem_segmented_2018_dvec.pkl",
                 }

export_home = r"C:\Data\Nirmal_Atkins"


def main():
    n = TramInclusion(
        tram_data_paths=tram_paths,
        notem_outputs=notem_outputs,
        export_home=export_home,
    )
    n.run(
        verbose=True
    )


if __name__ == '__main__':
    main()
