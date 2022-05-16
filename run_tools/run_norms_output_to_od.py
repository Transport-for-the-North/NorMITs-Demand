# -*- coding: utf-8 -*-
"""
Created on: 10/05/2022
Updated on:

Original author: Ben Taylor
Last update made by:
Other updates made by:

File purpose:

"""
# Built-Ins
import sys
import pathlib

# Third Party

# Local Imports
sys.path.append("..")
# pylint: disable=import-error,wrong-import-position
from normits_demand.tools.norms import matrix_converter as norms_matrix_converter
# pylint: enable=import-error,wrong-import-position

# RUNNING ARGS
TP_PROPORTIONS_DIR = r"E:\temp\cube\2f ILF 2018\source - test"
MATRIX_DIR = r"E:\temp\cube\2f ILF 2018"

MATRIX_OUTPUT_DIRNAME = "converted_matrices"
MATRIX_YEAR = 2018

MATRIX_RENAME_PATH = r"I:\NorMITs Demand\import\norms\post_me_model_data\norms_to_demand_names.csv"
MATRIX_COMPILE_PATH = r"I:\NorMITs Demand\import\norms\post_me_model_data\od_compile_params.csv"
INTERNAL_MATRIX_COMPILE_PATH = r"I:\NorMITs Demand\import\norms\post_me_model_data\od_from_to_internal_compile_params.csv"


def main():
    # Init
    tp_proportions_dir = pathlib.Path(TP_PROPORTIONS_DIR)
    import_matrix_dir = pathlib.Path(MATRIX_DIR)

    # Read in and convert the tp proportions
    tp_proportions = norms_matrix_converter.get_norms_post_me_tp_proportions(
        norms_matrix_converter.NormsTpProportionFiles(tp_proportions_dir)
    )

    # Convert the matrices
    converter = norms_matrix_converter.NormsOutputToOD(
        matrix_dir=import_matrix_dir,
        matrix_year=MATRIX_YEAR,
        time_period_proportions=tp_proportions,
        matrix_renaming=MATRIX_RENAME_PATH,
        output_dir=import_matrix_dir / MATRIX_OUTPUT_DIRNAME
    )

    converter.convert_hb_internal()
    converter.convert_nhb_internal()
    converter.convert_external()
    converter.compile_matrices(MATRIX_COMPILE_PATH)
    converter.compile_internal_matrices(INTERNAL_MATRIX_COMPILE_PATH)


if __name__ == '__main__':
    main()
