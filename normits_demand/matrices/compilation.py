# -*- coding: utf-8 -*-
"""
Created on: Tues April 20 12:29:22 2021
Updated on:

Original author: Ben Taylor
Last update made by:
Other updates made by:

File purpose:
Collection of functions to compile matrices into different formats.
Primarily used to compile ouputs into NoRMS or NoHAM formats
"""
# Built ins
import os

# Third Party
import pandas as pd

# Local
import normits_demand as nd

from normits_demand.utils import file_ops


def convert_efs_to_norms_matrices(mat_import: nd.PathLike,
                                  mat_export: nd.PathLike,
                                  year: int,
                                  from_to_split_factors: nd.FactorsDict,
                                  ) -> None:
    """
    Converts EFS matrices into NoRMS Post-ME format

    Parameters
    ----------
    mat_import:
        Path to the directory containing the matrices to convert

    mat_export:
        Path to the directory to output the converted matrices

    year:
        The year of matrices to perform the conversion on.

    from_to_split_factors:
        A nested dictionary of:
        dict[efs_matrix_name][norms_matrix_name] = matrix of splitting factors

    Returns
    -------
    None
    """
    # Get all the matrix names from imports
    import_mats = file_ops.list_files(mat_import, ftypes=['.csv'])

    # Filter down to the required year
    yr_str = "yr%s" % year
    import_mats = [x for x in import_mats if yr_str in x]

    # Split the matrices according to the split factors
    for efs_mat_name, split_factors_dict in from_to_split_factors.items():
        # Find the efs matrix in the import matrices
        efs_mat = None
        for fname in import_mats:
            if efs_mat_name in fname:
                path = os.path.join(mat_import, fname)
                efs_mat = file_ops.read_df(path, index_col=0)

        if efs_mat is None:
            raise FileNotFoundError(
                "Cannot find a file in the import directory containing: %s"
                % efs_mat_name
            )

        # Decompile into component parts and write out
        for out_mat_name, split_factors in split_factors_dict.items():
            out_mat = efs_mat * split_factors

            # Write out
            fname = file_ops.add_to_fname(out_mat_name, "_%s" % yr_str)
            out_path = os.path.join(mat_export, fname)
            file_ops.write_df(out_mat, out_path)
