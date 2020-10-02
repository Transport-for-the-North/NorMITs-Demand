import os

import pandas as pd

import demand_utilities.utils as du


def decompile_od(od_import: str,
                 od_export: str,
                 year: int,
                 decompile_factors_path: str
                 ) -> None:
    """
    Takes User Class compiled OD matrices and decompiles them down to their
    individual purposes. Uses the stored decompiled factors to split.

    Parameters
    ----------
    od_import:
        Directory containing the user class compiled OD matrices.

    od_export:
        Directory to write the decompiled OD matrices.

    year:
        Which year to decompile the matrices for.

    decompile_factors_path:
        Full path to the pickle file containing the decompile factors to use.

    Returns
    -------
    None
    """
    # Load the factors
    decompile_factors = pd.read_pickle(decompile_factors_path)

    # Loop through the compiled matrices and decompile
    for comp_mat_name in decompile_factors.keys():
        # We need to ignore the year, so break into component parts
        comp_calib_params = du.fname_to_calib_params(comp_mat_name,
                                                     get_user_class=True,
                                                     get_matrix_format=True,
                                                     force_ca_exists=True)

        # Find the matching compiled matrix and load
        mat_name = du.get_compiled_matrix_name(
            matrix_format=comp_calib_params['matrix_format'],
            user_class=comp_calib_params['user_class'],
            year=str(year),
            mode=str(comp_calib_params['m']),
            ca=int(comp_calib_params['ca']),
            tp=str(comp_calib_params['tp']),
            csv=True
        )
        comp_mat = pd.read_csv(os.path.join(od_import, mat_name), index_col=0)
        print("Decompiling matrix: %s" % mat_name)

        # Loop through the factors and decompile the matrix
        for part_mat_name in decompile_factors[comp_mat_name].keys():
            # Decompile the matrix using the factors
            factors = decompile_factors[comp_mat_name][part_mat_name]
            part_mat = comp_mat * factors

            # Generate filename and save the decompiled matrix
            part_calib_params = du.fname_to_calib_params(part_mat_name,
                                                         get_trip_origin=True,
                                                         get_matrix_format=True)
            mat_name = du.calib_params_to_dist_name(
                trip_origin=part_calib_params['trip_origin'],
                matrix_format=part_calib_params['matrix_format'],
                calib_params=part_calib_params,
                csv=True
            )
            part_mat.to_csv(os.path.join(od_export, mat_name))
