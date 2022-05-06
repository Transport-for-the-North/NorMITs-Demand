import os
import pathlib
from typing import List
from functools import reduce

import numpy as np
import pandas as pd


# Self imports
from normits_demand import constants as consts
from normits_demand import efs_constants as efs_consts
from normits_demand.utils import compress, file_ops, general as du
from normits_demand.utils import vehicle_occupancy as vo
from normits_demand import AuditError

from normits_demand.concurrency import multiprocessing


def _decompile_od_internal(od_import,
                           od_export,
                           decompile_factors,
                           comp_mat_name,
                           year,
                           round_dp,
                           audit,
                           audit_tol,
                           ):
    # We need to ignore the year, so break into component parts
    comp_calib_params = du.fname_to_calib_params(comp_mat_name,
                                                 get_user_class=True,
                                                 get_matrix_format=True,
                                                 force_ca_exists=True)

    # Find the matching compiled matrix and load
    in_mat_name = du.get_compiled_matrix_name(
        matrix_format=comp_calib_params['matrix_format'],
        user_class=comp_calib_params['user_class'],
        year=str(year),
        mode=str(comp_calib_params['m']),
        ca=comp_calib_params['ca'],
        tp=str(comp_calib_params['tp']),
        csv=True
    )
    comp_mat = file_ops.read_df(
        os.path.join(od_import, in_mat_name),
        index_col=0,
        find_similar=True,
    )
    # Attempt to convert columns to integer to match decompile factors
    comp_mat.columns = pd.to_numeric(
        comp_mat.columns, downcast="integer", errors="ignore"
    )
    print("Decompiling matrix: %s" % in_mat_name)

    # Loop through the factors and decompile the matrix
    decompiled_mats = list()
    for part_mat_name in decompile_factors.keys():
        # Decompile the matrix using the factors
        factors = decompile_factors[part_mat_name]
        factors.columns = pd.to_numeric(
            factors.columns, downcast="integer", errors="ignore"
        )
        part_mat = comp_mat * factors

        # Generate filename and save the decompiled matrix
        part_calib_params = du.fname_to_calib_params(part_mat_name,
                                                     get_trip_origin=True,
                                                     get_matrix_format=True)

        # If the year has not been found, add it in manually
        if part_calib_params.get('yr') is None:
            part_calib_params['yr'] = str(year)

        mat_name = du.calib_params_to_dist_name(
            trip_origin=part_calib_params['trip_origin'],
            matrix_format=part_calib_params['matrix_format'],
            calib_params=part_calib_params,
            csv=True
        )

        part_mat.round(round_dp).to_csv(os.path.join(od_export, mat_name))

        # Save for audit later
        decompiled_mats.append(part_mat)

    # Check that the output matrices total the input matrices
    if audit:
        # Sum the split matrices
        mats_sum = reduce(lambda x, y: x.add(y, fill_value=0),
                          decompiled_mats)

        # Get the absolute difference between the compiled and decompiled
        abs_diff = np.absolute((mats_sum - comp_mat).values).sum()

        if abs_diff > audit_tol:
            raise AuditError(
                "While decompiling matrices from %s, the absolute "
                "difference between the original and decompiled matrices "
                "exceeded the tolerance. Tolerance: %s, Absolute "
                "Difference: %s"
                % (in_mat_name, str(audit_tol), str(abs_diff)))


def decompile_od(od_import: str,
                 od_export: str,
                 year: int,
                 decompile_factors_path: str,
                 audit: bool = False,
                 round_dp: int = consts.DEFAULT_ROUNDING,
                 audit_tol: float = 0.001,
                 process_count: int = consts.PROCESS_COUNT,
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

    audit:
        Whether to perform print_audits to make sure the decompiled matrices are
        sufficiently similar to the compiled matrices when reversing the
        process.

    audit_tol:
        The tolerance to apply when auditing. If the total absolute difference
        between the original and audit matrices is greater than this value, an
        error will be thrown.

    Returns
    -------
    None
    """
    # Check if file is compressed before loading the factors
    suffix = "".join(pathlib.Path(decompile_factors_path).suffixes)
    if suffix.lower() in (file_ops.PD_COMPRESSION | {consts.COMPRESSION_SUFFIX}):
        decompile_factors = compress.read_in(decompile_factors_path)
    else:
        decompile_factors = file_ops.read_pickle(decompile_factors_path)

    # ## MULTIPROCESS ## #
    unchanging_kwargs = {
        'od_import': od_import,
        'od_export': od_export,
        'year': year,
        'round_dp': round_dp,
        'audit': audit,
        'audit_tol': audit_tol,
    }

    kwarg_list = list()
    for comp_mat_name in decompile_factors.keys():
        kwargs = unchanging_kwargs.copy()
        kwargs.update({
            'decompile_factors': decompile_factors[comp_mat_name],
            'comp_mat_name': comp_mat_name,
        })
        kwarg_list.append(kwargs)

    # Loop through the compiled matrices and decompile
    multiprocessing.multiprocess(
        fn=_decompile_od_internal,
        kwargs=kwarg_list,
        process_count=process_count,
    )


def _convert_to_efs_matrices_user_class(import_path: str,
                                        export_path: str,
                                        matrix_format: str,
                                        to_wide: bool = False,
                                        wide_col_name: str = 'zone_id',
                                        force_year: int = None
                                        ) -> None:
    # Init
    import_files = du.list_files(import_path)

    # Figure out the new filename and copy to export location
    for fname in import_files:
        print("Converting '%s' to EFS matrix format..." % str(fname))

        if '.csv' not in fname or '.pbz2' not in fname:
            continue

        # Try get the calib params from the filename
        calib_params = du.post_me_fname_to_calib_params(fname,
                                                        force_year=force_year)

        # Generate the new filename
        new_fname = du.get_compiled_matrix_name(
            matrix_format=matrix_format,
            user_class=calib_params['user_class'],
            year=str(calib_params['yr']),
            mode=str(calib_params['m']),
            tp=str(calib_params['tp']),
            csv=True
        )

        # Just copy over if we don't need to convert
        if not to_wide:
            du.copy_and_rename(
                src=os.path.join(import_path, fname),
                dst=os.path.join(export_path, new_fname)
            )
            continue

        # Only get here if we need to convert to wide

        # Read in, convert to wide, and save in new location
        mat = pd.read_csv(os.path.join(import_path, fname),
                          names=[wide_col_name, 'col2', 'vals'])

        # Convert from long to wide format and output
        du.long_to_wide_out(
            mat,
            v_heading=wide_col_name,
            h_heading='col2',
            values='vals',
            out_path=os.path.join(export_path, new_fname)
        )

    return


def convert_to_efs_matrices(import_path: str,
                            export_path: str,
                            matrix_format: str,
                            year: int,
                            user_class: bool = True,
                            to_wide: bool = True,
                            wide_col_name: str = 'zone_id',
                            from_pcu: bool = False,
                            vehicle_occupancy_import: str = None,
                            m_needed: List[int] = efs_consts.MODES_NEEDED
                            ) -> None:
    """
    Converts matrices from TfN models into a format that EFS uses.
    This usually means a name conversion, and converting to wide format.

    Parameters
    ----------
    import_path:
        The directory to find the matrices to import and convert.

    export_path:
        The directory to output the converted matrices.

    matrix_format:
        What format the matrices are in. Usually 'pa' or 'od'.

    year:
        The year the compiled matrices have been generated for

    user_class:
        Whether the matrices are aggregated to user class or not.
        Default value is True.

    to_wide:
        Whether the matrices need converting from long to wide format.
        Default value is True.

    wide_col_name:
        If converting to wide format, this name is used as the title for the
        rows/columns of the resulting matrices.

    from_pcu:
        Whether the input matrices need converting from vehicle demand to
        people demand or not

    vehicle_occupancy_import:
        Only needs to be set if from_pcu is True. This is the location to find
        the vehicle occupancy file. The given location is usually the home of
        the imports directory. e.g. "Y:/NorMITs Demand/import"

    m_needed:
        The mode being used when converting from PCU to people.

    Returns
    -------
    None
    """
    # Input checks
    if len(m_needed) > 1:
        raise ValueError("Got more than one mode. convert_to_efs_matrices() "
                         "can only handle one mode at a time.")

    temp_export_path = export_path
    if from_pcu:
        if vehicle_occupancy_import is None:
            raise ValueError("vehicle_occupancy_import needs to be set when"
                             "converting from PCU matrices.")
        temp_export_path = os.path.join(export_path, 'from_pcu')
        du.create_folder(temp_export_path)

    if user_class:
        _convert_to_efs_matrices_user_class(
            import_path=import_path,
            export_path=temp_export_path,
            matrix_format=matrix_format,
            to_wide=to_wide,
            wide_col_name=wide_col_name,
            force_year=year
        )
    else:
        # TODO: Write this functionality
        raise NotImplementedError("Cannot convert naming unless in user class"
                                  "format.")

    if not from_pcu:
        return

    car_occupancies = pd.read_csv(os.path.join(
        vehicle_occupancy_import,
        'vehicle_occupancies',
        'car_vehicle_occupancies.csv',
    ))

    # Only get here if we need to convert from PCU format
    vo.people_vehicle_conversion(
        mat_import=temp_export_path,
        mat_export=export_path,
        car_occupancies=car_occupancies,
        mode=str(m_needed[0]),
        method='to_people',
        out_format='wide'
    )
