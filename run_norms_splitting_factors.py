# -*- coding: utf-8 -*-
"""
Created on: Mon March 1 16:30:34 2020
Updated on:

Original author: Ben Taylor
Last update made by:
Other updates made by:

File purpose:
WRITE PURPOSE
"""

from normits_demand.matrices import matrix_processing as mat_p

PA_MATRICES_DIR = r'E:\NorMITs Demand\norms\v0.3-EFS_Output\NTEM\iter0\Matrices\24hr PA Matrices'
BASE_YEAR = '2018'


def main():
    mat_p.compile_norms_to_vdm()


if __name__ == '__main__':
    main()
