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
tram_paths = r"I:\Data\Light Rail\tram_hb_productions.csv"
notem_outputs = {'hb_p': r"C:\Data\Nirmal_Atkins\iter4\NTEM\hb_productions\hb_msoa_notem_segmented_2018_dvec.pkl",
                 'hb_a': r"I:\NorMITs Demand\NoTEM\iter4\NTEM\hb_attractions\hb_msoa_notem_segmented_2018_dvec.pkl",
                 'nhb_p': r"I:\NorMITs Demand\NoTEM\iter4\NTEM\nhb_productions\nhb_msoa_notem_segmented_2018_dvec.pkl",
                 'nhb_a': r"I:\NorMITs Demand\NoTEM\iter4\NTEM\nhb_attractions\nhb_msoa_notem_segmented_2018_dvec.pkl",
                 }


def main():
    n = TramInclusion(
        tram_data_paths=tram_paths,
        notem_outputs=notem_outputs
    )
    n.run(
        verbose=True
    )


if __name__ == '__main__':
    main()
