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
import pandas as pd

# Local imports
import normits_demand as nd
import normits_demand.constants as consts

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
        file_ops.write_df(mat, output_path, index=False)


def decompile_norms(year: int,
                    post_me_import: nd.PathLike,
                    post_me_renamed_export: nd.PathLike,
                    decompile_factors_dir: nd.PathLike,
                    overwrite_converted_matrices: bool = True,
                    ) -> None:
    # TODO: Write decompile_norms() docs
    # Init
    model_name = 'norms'

    print(year)
    print(post_me_import)
    print(post_me_renamed_export)
    print(decompile_factors_dir)

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
        )

    raise NotImplementedError
