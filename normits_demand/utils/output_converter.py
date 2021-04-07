# -*- coding: utf-8 -*-
"""
Created on: Wed November 18 11:20:36 2020
Updated on:

Original author: Ben Taylor
Last update made by:
Other updates made by:

File purpose:
Collection of methods used to convert EFS files into different formats.
Usually for different programmes e.g. NoHam VDM matrices
"""
import os

import pandas as pd

from typing import Any
from typing import Dict
from typing import Union

from collections import defaultdict

from tqdm import tqdm

from normits_demand.utils import general as du


def noham_vdm_tour_proportions_out(input_path: str,
                                   output_path: str,
                                   year: Union[str, int],
                                   seg_level: str,
                                   seg_params: Dict[str, Any],
                                   ) -> None:
    """
    Converts .pkl tour proportions into .csv format for NoHAM VDM

    Parameters
    ----------
    input_path:
        Directory containing the tour proportions to be converted

    output_path:
        Path (including file name) to write the .csv converted tour proportions.

    year:
        The year the tour proportions were made for

    seg_level:
        The level of segmentation of the tour proportions to convert. This
        should be one of the values in efs_constants.SEG_LEVELS.

    seg_params:
        A dictionary of kwarg: values for the segmentation level chosen. See
        TODO: Create reference for how seg_params work

    Returns
    -------
    None
    """
    # Init
    seg_params = seg_params.copy()

    # Safely ignore time periods
    if 'tp_needed' in seg_params:
        del seg_params['tp_needed']

    # Loop through each segmentation
    for seg_values in du.seg_level_loop_generator(seg_level, seg_params):
        # Read in base TP
        tour_props_name = du.get_seg_level_dist_name(
            seg_level=seg_level,
            seg_values=seg_values,
            matrix_format='%s_tour_proportions' % seg_level,
            year=year,
            suffix='.pkl'
        )
        tour_props = pd.read_pickle(os.path.join(input_path, tour_props_name))

        # Convert to csv
        output_ph = defaultdict(list)
        desc = 'Translating %s' % tour_props_name
        for orig, orig_dict in tqdm(tour_props.items(), desc=desc):
            for dest, tp_array in orig_dict.items():

                # Convert tour proportions
                for tp_num, tp_val in enumerate(tp_array.flatten(), 1):
                    output_ph['Origin'].append(orig)
                    output_ph['Destination'].append(dest)
                    output_ph['TourPropNo'].append(tp_num)
                    output_ph['Factor'].append(tp_val)

        # Write out
        csv_tour_props_name = tour_props_name.replace('pkl', 'csv')
        out_path = os.path.join(output_path, csv_tour_props_name)

        print("Writing %s to disk..." % csv_tour_props_name)
        pd.DataFrame(output_ph).to_csv(out_path, index=False)


def convert_wide_to_long(import_dir: str,
                         export_dir: str,
                         id_vars: str = 'o_zone',
                         var_name: str = 'd_zone',
                         value_name: str = 'trips',
                         echo: bool = True
                         ) -> None:
    # TODO: Write convert_wide_to_long() docs
    desc = 'Converting Matrices'
    for mat_name in tqdm(du.list_files(import_dir), desc=desc, disable=(not echo)):
        in_path = os.path.join(import_dir, mat_name)
        out_path = os.path.join(export_dir, mat_name)

        df = pd.read_csv(in_path)
        du.wide_to_long_out(
            df,
            id_vars,
            var_name,
            value_name,
            out_path
        )