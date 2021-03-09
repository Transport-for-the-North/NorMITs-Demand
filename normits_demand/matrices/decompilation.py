# -*- coding: utf-8 -*-
"""
Created on: Thurs March 4 11:59:43 2021
Updated on:

Original author: Ben Taylor
Last update made by:
Other updates made by:

File purpose:
WRITE PURPOSE
"""
# Builtins
import os
import functools
import operator

# 3rd Party
import numpy as np
import pandas as pd

# Local imports
import normits_demand as nd
import normits_demand.constants as consts

from normits_demand.validation import checks

from normits_demand.utils import general as du
from normits_demand.utils import file_ops

from normits_demand.matrices import matrix_processing as mat_p
from normits_demand.matrices import od_to_pa as od2pa


def decompile_noham(year: int,
                    post_me_import: nd.PathLike,
                    post_me_renamed_export: nd.PathLike,
                    od_24hr_export: nd.PathLike,
                    pa_24hr_export: nd.PathLike,
                    zone_translate_dir: nd.PathLike,
                    tour_proportions_export: nd.PathLike,
                    decompile_factors_path: nd.PathLike,
                    vehicle_occupancy_import: nd.PathLike,
                    overwrite_decompiled_od: bool = True,
                    overwrite_tour_proportions: bool = True,
                    ) -> None:
    """

    Performs the following actions:
        - Converts post-ME files into and EFS format as needed. (file name
          changes, converting long to wide as needed.)
        - Decompiles the converted post-ME matrices into purposes (and ca
          when needed) using the split factors produced during pre-me
          OD compilation
        - Generates tour proportions for each OD pair, for each purpose
          (and ca as needed), saving for future year post-ME compilation
          later.
        - Converts OD matrices to PA.

    Parameters
    ----------
    year
    post_me_import
    post_me_renamed_export
    od_24hr_export
    pa_24hr_export
    zone_translate_dir
    tour_proportions_export
    decompile_factors_path
    vehicle_occupancy_import
    overwrite_decompiled_od
    overwrite_tour_proportions

    Returns
    -------

    """
    # TODO: Document and test NOHAM decompile
    model_name = 'noham'

    if overwrite_decompiled_od:
        print("Decompiling OD Matrices into purposes...")
        need_convert = need_to_convert_to_efs_matrices(
            post_me_import=post_me_import,
            converted_export=post_me_renamed_export
        )

        if need_convert:
            od2pa.convert_to_efs_matrices(
                import_path=post_me_import,
                export_path=post_me_renamed_export,
                matrix_format='od',
                year=year,
                user_class=True,
                to_wide=True,
                wide_col_name='%s_zone_id' % model_name,
                from_pcu=True,
                vehicle_occupancy_import=vehicle_occupancy_import
            )

        # TODO: Stop the filename being hardcoded after integration with TMS
        od2pa.decompile_od(
            od_import=post_me_renamed_export,
            od_export=od_24hr_export,
            decompile_factors_path=decompile_factors_path,
            year=year
        )

    if overwrite_tour_proportions:
        print("Converting OD matrices to PA and generating tour "
              "proportions...")
        mat_p.generate_tour_proportions(
            od_import=od_24hr_export,
            zone_translate_dir=zone_translate_dir,
            pa_export=pa_24hr_export,
            tour_proportions_export=tour_proportions_export,
            year=year,
        )


def need_to_convert_to_efs_matrices(post_me_import: str,
                                    converted_export: str
                                    ) -> bool:
    """
    Checks if the matrices stored in model_import need converting into
    efs format.

    At the moment this is just a simple check that matrices exist in
    model_import and not od_import.
    TODO: Update with better checks on NoRMS and NoHAM post-ME matrices
      are more clear

    Parameters
    ----------
    post_me_import:
        Path to the dir containing the post-me matrices.

    converted_export:
        Path to the dir that the converted post-ME matrices should be output
        to

    Returns
    -------
    bool:
        Returns True if the matrices need converting. Otherwise False.
    """
    return (len(du.list_files(converted_export)) == 0 and
            len(du.list_files(post_me_import)) > 0)


def convert_norms_to_efs_matrices(import_dir: nd.PathLike,
                                  export_dir: nd.PathLike,
                                  matrix_format: str,
                                  year: int,
                                  wide_col_name: str = None,
                                  csv_out: bool = False,
                                  compress_out: bool = True,
                                  ) -> None:
    # TODO: Write convert_norms_to_efs_matrices() docs
    # Init
    conversion_dict = consts.NORMS_VDM_SEG_TO_NORMS_POSTME_NAMING

    if len(consts.MODEL_MODES['norms']) != 1:
        raise nd.NormitsDemandError(
            "Somehow gotten more than 1 more for NoRMS! What's gone wrong?"
        )
    mode = consts.MODEL_MODES['norms'][0]

    # ## CHECK POST-ME MATRICES EXIST ## #
    # Build a list of matrix names
    post_me_matrices = conversion_dict.values()
    post_me_matrices = functools.reduce(operator.add, post_me_matrices)
    post_me_matrices = ['%s.csv' % x for x in post_me_matrices]

    for fname in post_me_matrices:
        path = os.path.join(import_dir, fname)
        if not os.path.exists(path):
            raise IOError(
                "Cannot find all the post-me matrices needed to read in. "
                "Specifically, a matrix at '%s' cannot be found."
                % str(path)
            )

    # ## CONVERT TO EFS FORMAT ## #
    for efs_mat_name, post_me_mat_names in conversion_dict.items():
        # Read in and combine matrices if needed
        if len(post_me_mat_names) == 1:
            fname = '%s.csv' % post_me_mat_names[0]
            path = os.path.join(import_dir, fname)
            mat = pd.read_csv(path, index_col=0)
        else:
            mat_list = list()
            for fname in post_me_mat_names:
                fname = '%s.csv' % fname
                path = os.path.join(import_dir, fname)
                mat_list.append(pd.read_csv(path, index_col=0))
            mat = functools.reduce(operator.add, mat_list)

        # Name the index col if we have a name
        if wide_col_name is not None:
            mat.index.name = wide_col_name

        # Generate the output fname
        seg_agg_dict = du.get_norms_vdm_segment_aggregation_dict(efs_mat_name)
        full_efs_mat_name = du.get_compiled_matrix_name(
            matrix_format,
            user_class=seg_agg_dict['uc'],
            year=str(year),
            trip_origin=None,
            mode=str(mode),
            suffix='_%s' % efs_mat_name,
            csv=csv_out,
            compress=compress_out,
        )

        # Write the new matrix to disk
        output_path = os.path.join(export_dir, full_efs_mat_name)
        file_ops.write_df(mat, output_path)


def decompile_matrices(matrix_import: nd.PathLike,
                       matrix_export: nd.PathLike,
                       decompile_factors_path: nd.PathLike,
                       audit: bool = True,
                       audit_tol: float = 0.1
                       ) -> None:
    """
    Decompiles the matrices in matrix_import.

    Decompiled matrices are written to matrix_export, and matrices are
    decompiled using the factors at decompile_factors_path.

    Optionally performs an audit to make sure the resultant matrices
    still sum to their compiled counterpart, within audit_tol.

    Parameters
    ----------
    matrix_import:
        Path to the directory containing the compiled matrices.

    matrix_export:
        Path to the directory to write the decompiled OD matrices.

    decompile_factors_path:
        Full path to the pickle file containing the decompile factors to use.

    audit:
        Whether to perform a check to make sure the decompiled matrices are
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
    # Load the factors
    decompile_factors = pd.read_pickle(decompile_factors_path)

    # Loop through the compiled matrices and decompile
    # TODO: Multiprocess decompile_od()
    for comp_mat_name in decompile_factors.keys():
        # Deal with potentially different suffixes
        in_mat_name = file_ops.find_filename(
            os.path.join(matrix_import, comp_mat_name),
            return_full_path=False,
        )
        path = os.path.join(matrix_import, in_mat_name)
        comp_mat = file_ops.read_df(path, index_col=0)
        print("Decompiling %s..." % in_mat_name)

        # Loop through the factors and decompile the matrix
        decompiled_mats = list()
        for part_mat_name in decompile_factors[comp_mat_name].keys():
            # Decompile the matrix using the factors
            factors = decompile_factors[comp_mat_name][part_mat_name]
            part_mat = comp_mat * factors

            # Write the decompiled matrix to disk
            path = os.path.join(matrix_export, part_mat_name)
            file_ops.write_df(part_mat, path)

            # Save for audit later
            decompiled_mats.append(part_mat)

        # Check that the output matrices total the input matrices
        if audit:
            # Sum the split matrices
            mats_sum = functools.reduce(lambda x, y: x.add(y, fill_value=0),
                                        decompiled_mats)

            # Figure out the upper and lower tolerance limits
            comp_total = comp_mat.to_numpy().sum()
            lower = comp_total - (comp_total * audit_tol)
            upper = comp_total + (comp_total * audit_tol)

            # Calculate our decompiled absolute/percentage differences
            mats_sum_total = mats_sum.to_numpy().sum()
            abs_diff = np.absolute((mats_sum - comp_mat).values).sum()

            # We can skip this check if we're all 0
            if mats_sum_total == comp_total == 0:
                continue

            perc_diff = abs_diff / comp_mat.to_numpy().sum()

            if not (lower < mats_sum_total < upper):
                raise nd.AuditError(
                    "While decompiling matrices from %s, the absolute "
                    "difference between the original and decompiled matrices "
                    "exceeded the tolerance.\n"
                    "Percentage Tolerance: +- %s%%\n"
                    "Percentage Difference: %.4f%%\n"
                    "Absolute Difference: %.4f"
                    % (in_mat_name, str(audit_tol), perc_diff, abs_diff))


def recombine_internal_external(internal_import: nd.PathLike,
                                external_import: nd.PathLike,
                                full_export: nd.PathLike,
                                ) -> None:
    """
    Combines the internal and external split matrices and write out to full_export

    Will warn the user if all matrices from both folders are not used

    Parameters
    ----------
    internal_import:
        Path to the directory containing the segmented internal matrices

    external_import:
        Path to the directory containing the segmented external matrices

    full_export:
        Path to the directory to write out the combined matrices.

    Returns
    -------
    None

    """
    # Init
    all_internal_fnames = file_ops.list_files(internal_import)
    all_external_fnames = file_ops.list_files(external_import)

    # ## BUILD DICTIONARY OF MATRICES TO COMBINE ## #
    comp_dict = dict()
    used_external_fnames = list()
    for int_fname in all_internal_fnames:
        # Determine the related filenames
        full_fname = file_ops.remove_internal_suffix(int_fname)
        ext_fname = file_ops.add_external_suffix(full_fname)

        # Check the external file actually exists
        if not os.path.exists(os.path.join(external_import, ext_fname)):
            raise FileNotFoundError(
                "No external file exists to match the internal file.\n"
                "Internal file location: %s\n"
                "Expected external file location: %s"
                % (os.path.join(internal_import, int_fname),
                   os.path.join(external_import, ext_fname))
            )

        # Make a note of the external files we've used
        used_external_fnames.append(str(ext_fname))

        # Add an entry to the dictionary
        output_path = os.path.join(full_export, full_fname)
        comp_dict[output_path] = [
            os.path.join(internal_import, int_fname),
            os.path.join(external_import, ext_fname),
        ]

    # Make sure we've used all the external matrices
    for ext_fname in all_external_fnames:
        if ext_fname not in used_external_fnames:
            int_fname = ext_fname.replace(consts.EXTERNAL_SUFFIX, consts.INTERNAL_SUFFIX)
            raise FileNotFoundError(
                "No internal file exists to match the external file.\n"
                "External file location: %s\n"
                "Expected internal file location: %s"
                % (os.path.join(external_import, ext_fname),
                   os.path.join(internal_import, int_fname))
            )

    # ## COMPILE THE MATRICES ## #
    for output_path, in_paths in comp_dict.items():
        # Read in the matrices and compile
        partial_mats = [file_ops.read_df(x, index_col=0) for x in in_paths]
        full_mat = functools.reduce(lambda x, y: x.add(y, fill_value=0), partial_mats)

        # Write the complete matrix to disk
        file_ops.write_df(full_mat, output_path)


def decompile_norms(year: int,
                    post_me_import: nd.PathLike,
                    post_me_renamed_export: nd.PathLike,
                    post_me_decompiled_export: nd.PathLike,
                    decompile_factors_dir: nd.PathLike,
                    matrix_format: str = 'pa',
                    overwrite_converted_matrices: bool = True,
                    csv_out: bool = False,
                    compress_out: bool = True,
                    ) -> None:
    # TODO: Write decompile_norms() docs
    # Init
    model_name = 'norms'
    matrix_format = checks.validate_matrix_format(matrix_format)

    # Make intermediate folder for internal and external
    int_dir = os.path.join(post_me_decompiled_export, 'internal')
    ext_dir = os.path.join(post_me_decompiled_export, 'external')

    for path in [int_dir, ext_dir]:
        file_ops.create_folder(path, verbose=False)

    # ## CONVERT MATRICES TO EFS VDM FORMAT ## #
    need_convert = need_to_convert_to_efs_matrices(
        post_me_import=post_me_import,
        converted_export=post_me_renamed_export
    )

    if need_convert or overwrite_converted_matrices:
        convert_norms_to_efs_matrices(
            import_dir=post_me_import,
            export_dir=post_me_renamed_export,
            matrix_format='pa',
            year=year,
            wide_col_name='%s_zone_id' % model_name,
            csv_out=csv_out,
            compress_out=compress_out,
        )

    # ## DECOMPILE THE NORMS MATRICES ## #
    for int_or_ext, out_dir in zip(['internal', 'external'], [int_dir, ext_dir]):
        factors_fname = du.get_split_factors_fname(
            matrix_format=matrix_format,
            year=str(year),
            suffix=int_or_ext,
        )
        decompile_factors_path = os.path.join(decompile_factors_dir, factors_fname)

        decompile_matrices(
            matrix_import=post_me_renamed_export,
            matrix_export=out_dir,
            decompile_factors_path=decompile_factors_path,
        )

    # ## RECOMBINE INTERNAL AND EXTERNAL DEMAND ## #
    recombine_internal_external(
        internal_import=int_dir,
        external_import=ext_dir,
        full_export=post_me_decompiled_export,
    )
