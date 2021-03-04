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

# 3rd Party

# Local imports
import normits_demand as nd

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
        need_convert = od2pa.need_to_convert_to_efs_matrices(
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
            ca_needed=ca_needed
        )


def convert_norms_to_efs_matrices(import_dir: nd.PathLike,
                                  export_dir: nd.PathLike,
                                  matrix_format: str,
                                  year: int,
                                  wide_col_name: str = None,
                                  ) -> None:
    # TODO: Write convert_norms_to_efs_matrices() docs
    raise NotImplementedError


def decompile_norms(year: int,
                    post_me_import: nd.PathLike,
                    post_me_renamed_export: nd.PathLike,
                    decompile_factors_dir: nd.PathLike,
                    ) -> None:
    # TODO: Write decompile_norms() docs
    # Init
    model_name = 'norms'

    print(year)
    print(post_me_import)
    print(post_me_renamed_export)
    print(decompile_factors_dir)

    need_convert = od2pa.need_to_convert_to_efs_matrices(
        post_me_import=post_me_import,
        converted_export=post_me_renamed_export
    )

    if need_convert:
        rename_norms_to_efs_matrices(
            import_dir=post_me_import,
            export_dir=post_me_renamed_export,
            matrix_format='pa',
            year=year,
            wide_col_name='%s_zone_id' % model_name,
        )

    raise NotImplementedError
