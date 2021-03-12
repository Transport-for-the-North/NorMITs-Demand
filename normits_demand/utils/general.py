# -*- coding: utf-8 -*-
"""
Created on: Fri September 11 12:05:31 2020
Updated on:

Original author: Ben Taylor
Last update made by:
Other updates made by:

File purpose:
General utils for use in EFS.
TODO: After integrations with TMS, combine with old_tms.utils.py
  to create a general utils file
"""

import os
import re
import shutil
import random
import inspect


import pandas as pd
import numpy as np

from typing import Any
from typing import List
from typing import Dict
from typing import Tuple
from typing import Union
from typing import Iterable
from typing import Iterator

from pathlib import Path

from math import isclose

import functools
from tqdm import tqdm
from itertools import product
from collections import defaultdict

# Local imports
from normits_demand import constants as consts
from normits_demand import efs_constants as efs_consts

# Can call tms utils.py functions from here
from normits_demand.utils.utils import *

# TODO: Utils is getting big. Refactor into smaller, more specific modules


class NormitsDemandError(Exception):
    """
    Base Exception for all custom NorMITS demand errors
    """

    def __init__(self, message=None):
        self.message = message
        super().__init__(self.message)


class ExternalForecastSystemError(NormitsDemandError):
    """
    Base Exception for all custom EFS errors
    """
    def __init__(self, message=None):
        self.message = message
        super().__init__(self.message)


class InitialisationError(NormitsDemandError):
    """
    Exception for all errors that occur during normits_demand initialisation
    """
    def __init__(self, message=None):
        self.message = message
        super().__init__(self.message)


def get_seg_level_cols(seg_level: str,
                       keep_ca: bool = True,
                       keep_tp: bool = True,
                       ) -> List[str]:
    """
    Returns the column names used by the given segmentation

    Parameters
    ----------
    seg_level:
        The name of the segmentation level to get columns for

    keep_ca:
        Whether to keep ca segmentation in the return or not. If ca does not
        exist for seg_level, this arg is ignored.

    keep_tp:
        Whether to keep tp segmentation in the return or not. If tp does not
        exist for seg_level, this arg is ignored.


    Returns
    -------
    seg_cols:
        A list of columns names for the given seg_level
    """
    # Init
    seg_level = validate_seg_level(seg_level)
    seg_cols = efs_consts.SEG_LEVEL_COLS[seg_level]

    # Remove cols if asked to and they exist
    if not keep_ca:
        seg_cols = list_safe_remove(seg_cols, ['ca'])

    if not keep_tp:
        seg_cols = list_safe_remove(seg_cols, ['tp'])

    return seg_cols


def validate_seg_level(seg_level: str) -> str:
    """
    Tidies up seg_level and raises an exception if not a valid name

    Parameters
    ----------
    seg_level:
        The name of the segmentation level to validate

    Returns
    -------
    seg_level:
        seg_level with both strip and lower applied to remove any whitespace
        and make it all lowercase

    Raises
    -------
    ValueError:
        If seg_level is not a valid name for a level of segmentation
    """
    # Init
    seg_level = seg_level.strip().lower()

    if seg_level not in efs_consts.SEG_LEVELS:
        raise ValueError("%s is not a valid name for a level of segmentation"
                         % seg_level)
    return seg_level


def validate_zoning_system(zoning_system: str) -> str:
    """
    Tidies up zoning_system and raises an exception if not a valid name

    Parameters
    ----------
    zoning_system:
        The name of the zoning system to validate

    Returns
    -------
    zoning_system:
        zoning_system with both strip and lower applied to remove any
        whitespace and make it all lowercase

    Raises
    -------
    ValueError:
        If zoning_system is not a valid name for a zoning system
    """
    # Init
    zoning_system = zoning_system.strip().lower()

    if zoning_system not in efs_consts.ZONING_SYSTEMS:
        raise ValueError("%s is not a valid name for a zoning system"
                         % zoning_system)
    return zoning_system


def validate_scenario_name(scenario_name: str) -> str:
    """
    Tidies up zoning_system and raises an exception if not a valid name

    Parameters
    ----------
    scenario_name:
        The name of the scenario to validate

    Returns
    -------
    scenario_name:
        scenario_name with both strip and upper applied to remove any
        whitespace and make it all uppercase

    Raises
    -------
    ValueError:
        If scenario_name is not a valid name for a scenario
    """
    # Init
    scenario_name = scenario_name.strip().upper()

    if scenario_name not in efs_consts.SCENARIOS:
        raise ValueError("%s is not a valid name for a scenario."
                         % scenario_name)
    return scenario_name


def validate_model_name(model_name: str) -> str:
    """
    Tidies up model_name and raises an exception if not a valid name

    Parameters
    ----------
    model_name:
        The the model name to validate

    Returns
    -------
    model_name:
        model_name with both strip and lower applied to remove any
        whitespace and make it all lowercase

    Raises
    -------
    ValueError:
        If model_name is not a valid name for a model
    """
    # Init
    model_name = model_name.strip().lower()

    if model_name not in efs_consts.MODEL_NAMES:
        raise ValueError("%s is not a valid name for a model"
                         % model_name)
    return model_name


def validate_model_name_and_mode(model_name: str,
                                 m_needed: List[int] = efs_consts.MODES_NEEDED
                                 ) -> None:
    """
    Checks that the given modes are valid modes for the given model name

    Parameters
    ----------
    model_name:
        The the model name to validate

    m_needed:
        A list of the modes to validate against the model name

    Returns
    -------
        None
    
    Raises
    -------
    ValueError:
        If any of the modes in m_needed are not a valid mode for the given
        model_name
    """
    # Init
    model_name = validate_model_name(model_name)

    for mode in m_needed:
        if mode not in efs_consts.MODEL_MODES[model_name]:
            raise ValueError("%s is not a valid mode for model %s"
                             % (str(mode), model_name))


def validate_user_class(user_class: str) -> str:
    """
    Tidies up user_class and raises an exception if not a valid name

    Parameters
    ----------
    user_class:
        The name of the user class to validate

    Returns
    -------
    seg_level:
        user_class with both strip and lower applied to remove any whitespace
        and make it all lowercase

    Raises
    -------
    ValueError:
        If user_class is not a valid name for a level of segmentation
    """
    # Init
    user_class = user_class.strip().lower()

    if user_class not in efs_consts.USER_CLASSES:
        raise ValueError("%s is not a valid name for user class"
                         % user_class)
    return user_class


def validate_vdm_seg_params(seg_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validates and cleans seg_params ready for a loop_generator

    Parameters
    ----------
    seg_params:
        Dictionary of vdm seg params to be cleaned.

    Returns
    -------
    valid_seg_params:
        cleaned version of the given seg_params. If a key did not exist that
        is needed, [None] is added as the value.
    """
    # Init
    seg_params = seg_params.copy()
    valid_segmentation = [
        'to_needed',
        'uc_needed',
        'm_needed',
        'ca_needed',
        'tp_needed'
    ]

    for seg in valid_segmentation:
        if seg_params.get(seg) is None:
            seg_params[seg] = [None]

    return seg_params


def print_w_toggle(*args, echo, **kwargs):
    """
    Small wrapper to only print when echo=True

    Parameters
    ----------
    *args:
        The text to print - can be passed in the same format as a usual
        print function

    echo:
        Whether to print the text or not

    **kwargs:
        Any other kwargs to pass directly to the print function call
    """
    if echo:
        print(*args, **kwargs)


def build_efs_io_paths(import_location: str,
                       export_location: str,
                       model_name: str,
                       iter_name: str,
                       scenario_name: str,
                       demand_version: str,
                       demand_dir_name: str = 'NorMITs Demand',
                       base_year: str = efs_consts.BASE_YEAR_STR,
                       land_use_iteration: str = None,
                       land_use_drive: str = None,
                       ) -> Tuple[dict, dict, dict]:
    """
    Builds three dictionaries of paths to the locations of all inputs and
    outputs for EFS

    Parameters
    ----------
    import_location:
        The directory the import directory exists - a dir named
        self.out_dir (NorMITs Demand) should exist here. Usually
        a drive name e.g. Y:/

    export_location:
        The directory to create the new output directory in - a dir named
        self.out_dir (NorMITs Demand) should exist here. Usually
        a drive name e.g. Y:/

    model_name:
        TfN model name in use e.g. norms or noham

    iter_name:
        The name of the iteration being run. Usually of the format iterx,
        where x is a number, e.g. iter3

    scenario_name:
        The name of the scenario use to produce outputs. This should be one
        of efs_consts.SCENARIOS

    demand_version:
        Version number of NorMITs Demand being run - this is used to generate
        the correct output path.

    demand_dir_name:
        The name used for the NorMITs Demand input/output directories.


    Returns
    -------
    imports:
        Dictionary of import paths with the following keys:
        imports, lookups, seed_dists, default

    efs_exports:
        Dictionary of export paths with the following keys:
        productions, attractions, pa, od, pa_24, od_24, sectors

    params:
        Dictionary of parameter export paths with the following keys:
        compile, tours
    """
    # TODO: Tidy up Y:/ drive imports/inputs folders after contract
    # BACKLOG: Update I/O paths to point to TMS as needed
    #  labels: demand merge
    # Init
    model_name = validate_model_name(model_name)

    # ## IMPORT PATHS ## #
    # Generate general import paths
    model_home = os.path.join(import_location, demand_dir_name)
    import_home = os.path.join(model_home, 'import')
    input_home = os.path.join(import_home, 'default')

    # Build the longer paths
    path = 'attractions/soc_2_digit_sic_%s.csv' % base_year
    soc_weights_path = os.path.join(import_home, path)

    # Build the zone translation dict
    zt_home = os.path.join(import_home, 'zone_translation')
    zone_translation = {
        'home': zt_home,
        'one_to_one': os.path.join(zt_home, 'one_to_one'),
        'weighted': os.path.join(zt_home, 'weighted'),
        'no_overlap': os.path.join(zt_home, 'no_overlap'),
        'msoa_str_int': os.path.join(zt_home, 'msoa_zones.csv'),
    }

    # BACKLOG: EFS Attraction model needs to make use of HB and NHB
    #  attraction weights. Currently only uses HB for both.
    #  labels: demand merge, EFS

    model_schema_home = os.path.join(import_home, model_name, 'model schema')

    imports = {
        'home': import_home,
        'default_inputs': input_home,
        'zone_translation': zone_translation,
        'tp_splits': os.path.join(import_home, 'tp_splits'),
        'lookups': os.path.join(model_home, 'lookup'),
        # 'seed_dists': os.path.join(import_home, model_name, 'seed_distributions'),
        'scenarios': os.path.join(import_home, 'scenarios'),
        'a_weights': os.path.join(import_home, 'attractions', 'hb_attraction_weights.csv'),
        'soc_weights': soc_weights_path,
        'ntem_control': os.path.join(import_home, 'ntem_constraints'),
        'model_schema': model_schema_home,
        'internal_zones': os.path.join(model_schema_home, consts.INTERNAL_AREA % model_name),
        'external_zones': os.path.join(model_schema_home, consts.EXTERNAL_AREA % model_name),
        'post_me_matrices': os.path.join(import_home, model_name, 'post_me'),
        'decomp_post_me': os.path.join(import_home, model_name, 'decompiled_post_me'),

    }

    # Add Land use import if we have an iteration
    if land_use_drive is not None and land_use_iteration is not None:
        land_use_home = os.path.join(
            land_use_drive,
            'NorMITs Land Use',
            land_use_iteration,
            'outputs',
        )
        land_use_fy = os.path.join(land_use_home, 'scenarios', scenario_name)

        imports['pop_by'] = os.path.join(land_use_home, consts.BASE_YEAR_POP_FNAME)
        imports['emp_by'] = os.path.join(land_use_home, consts.BASE_YEAR_EMP_FNAME)
        imports['land_use_fy_dir'] = land_use_fy

    # ## EXPORT PATHS ## #
    # Create home paths
    fname_parts = [
        export_location,
        demand_dir_name,
        model_name,
        "v%s-EFS_Output" % demand_version,
        scenario_name,
        iter_name,
    ]
    export_home = os.path.join(*fname_parts)
    matrices_home = os.path.join(export_home, 'Matrices')
    post_me_home = os.path.join(matrices_home, 'Post-ME Matrices')

    # Create consistent filenames
    pa = 'PA Matrices'
    pa_24 = '24hr PA Matrices'
    vdm_pa_24 = '24hr VDM PA Matrices'
    od = 'OD Matrices'
    od_24 = '24hr OD Matrices'
    compiled = 'Compiled'
    aggregated = 'Aggregated'
    pa_24_bespoke = '24hr PA Matrices - Bespoke Zones'
    pcu = 'PCU'

    exports = {
        'home': export_home,
        'productions': os.path.join(export_home, 'Productions'),
        'attractions': os.path.join(export_home, 'Attractions'),
        'sectors': os.path.join(export_home, 'Sectors'),
        'audits': os.path.join(export_home, 'Audits'),
        'dist_audits': os.path.join(export_home, 'Audits', 'Matrices'),
        'reports': os.path.join(export_home, 'Reports'),

        # Pre-ME
        'pa': os.path.join(matrices_home, pa),
        'pa_24': os.path.join(matrices_home, pa_24),
        'vdm_pa_24': os.path.join(matrices_home, vdm_pa_24),
        'od': os.path.join(matrices_home, od),
        'od_24': os.path.join(matrices_home, od_24),

        'compiled_od': os.path.join(matrices_home, ' '.join([compiled, od])),
        'compiled_od_pcu': os.path.join(matrices_home, ' '.join([compiled, od, pcu])),

        'aggregated_pa_24': os.path.join(matrices_home, ' '.join([aggregated, pa_24])),
        'aggregated_od': os.path.join(matrices_home, ' '.join([aggregated, od])),

        'pa_24_bespoke': os.path.join(matrices_home, pa_24_bespoke)
    }

    for _, path in exports.items():
        create_folder(path, chDir=False)

    # Post-ME
    compiled_od_path = os.path.join(post_me_home, ' '.join([compiled, od]))
    post_me_exports = {
        'pa': os.path.join(post_me_home, pa),
        'pa_24': os.path.join(post_me_home, pa_24),
        'vdm_pa_24': os.path.join(post_me_home, vdm_pa_24),
        'od': os.path.join(post_me_home, od),
        'od_24': os.path.join(post_me_home, od_24),
        'compiled_od': compiled_od_path,
        'model_output': os.path.join(compiled_od_path, ''.join(['from_', model_name]))
    }

    for _, path in post_me_exports.items():
        create_folder(path, chDir=False)

    # Combine into full export dict
    exports['post_me'] = post_me_exports

    # ## PARAMS OUT ## #
    param_home = os.path.join(export_home, 'Params')

    params = {
        'home': param_home,
        'compile': os.path.join(param_home, 'Compile Params'),
        'tours': os.path.join(param_home, 'Tour Proportions')
    }
    for _, path in params.items():
        create_folder(path, chDir=False)

    return imports, exports, params


def convert_msoa_naming(df: pd.DataFrame,
                        msoa_col_name: str,
                        msoa_path: str,
                        msoa_str_col: str = 'model_zone_code',
                        msoa_int_col: str = 'model_zone_id',
                        to: str = 'string'
                        ) -> pd.DataFrame:
    """
    Returns df with the msoa zoning given converted to either string or int
    names, as requested.

    Parameters
    ----------
    df:
        The dataframe to convert. Must have a column named as msoa_col_name

    msoa_col_name:
        The name of the column in df to convert.

    msoa_path:
        The full path to the file to use to do the conversion.

    msoa_str_col:
        The name of the column in msoa_path file which contains the string
        names for all msoa zones.

    msoa_int_col:
        The name of the column in msoa_path file which contains the integer
        ids for all msoa zones.

    to:
        The format to convert to. Supports either 'int' or 'string'.

    Returns
    -------
    converted_df:
        df, in the same order, but the msoa_col_name has been converted to the
        desired format.
    """
    # Init
    column_order = list(df)
    to = to.strip().lower()

    # Validate
    if msoa_col_name not in df:
        raise KeyError("Column '%s' not in given dataframe to convert."
                       % msoa_col_name)

    # Rename everything to make sure there are no clashes
    df = df.rename(columns={msoa_col_name: 'df_msoa'})

    # Read in MSOA conversion file
    msoa_zones = pd.read_csv(msoa_path).rename(
        columns={
            msoa_str_col: 'msoa_string',
            msoa_int_col: 'msoa_int'
        }
    )

    if to == 'string' or to == 'str':
        merge_col = 'msoa_int'
        keep_col = 'msoa_string'
    elif to == 'integer' or to == 'int':
        merge_col = 'msoa_string'
        keep_col = 'msoa_int'
    else:
        raise ValueError("Invalid value received. Do not know how to convert "
                         "to '%s'" % str(to))

    # Convert MSOA strings to id numbers
    df = pd.merge(df,
                  msoa_zones,
                  left_on='df_msoa',
                  right_on=merge_col)

    # Drop unneeded columns and rename
    df = df.drop(columns=['df_msoa', merge_col])
    df = df.rename(columns={keep_col: msoa_col_name})

    return df.reindex(column_order, axis='columns')



def copy_and_rename(src: str, dst: str) -> None:
    """
    Makes a copy of the src file and saves it at dst with the new filename.

    If no filename is given for dst, then the file will be copied over with
    the same name as used in src.

    Parameters
    ----------
    src:
        Path to the file to be copied.

    dst:
        Path to the new save location.

    Returns
    -------
    None
    """
    if not os.path.exists(src):
        raise IOError("Source file does not exist.\n %s" % src)

    if not os.path.isfile(src):
        raise IOError("The given src file is not a file. Cannot handle "
                      "directories.")

    # If no filename given, don't need to rename - just use src filename
    if '.' not in os.path.basename(dst):
        # Copy over with same filename
        shutil.copy(src, dst)
        return

    # Split paths
    src_head, src_tail = os.path.split(src)
    dst_head, dst_tail = os.path.split(dst)

    # Avoid case where src and dist is same locations
    if dst_head == src_head:
        shutil.copy(src, dst)
        return

    # Copy then rename
    shutil.copy(src, dst_head)
    shutil.move(os.path.join(dst_head, src_tail), dst)


def add_fname_suffix(fname: str, suffix: str):
    """
    Adds suffix to fname - in front of the file type extension

    Parameters
    ----------
    fname:
        The fname to be added to - must have a file type extension
        e.g. .csv
    suffix:
        The string to add between the end of the fname and the file
        type extension

    Returns
    -------
    new_fname:
        fname with suffix added

    """
    f_type = '.' + fname.split('.')[-1]
    new_fname = '.'.join(fname.split('.')[:-1])
    new_fname += suffix + f_type
    return new_fname


def safe_read_csv(file_path: str,
                  print_time: bool = False,
                  **kwargs
                  ) -> pd.DataFrame:
    """
    Reads in the file and performs some simple file checks

    Parameters
    ----------
    file_path:
        Path to the file to read in

    print_time:
        Whether to print out some info on how long the file read has taken

    kwargs:
        ANy kwargs to pass onto pandas.read_csv()

    Returns
    -------
    dataframe:
        The data from file_path
    """
    # Init
    if kwargs is None:
        kwargs = dict()

    # TODO: Add any more error checks here
    # Check file exists
    if not os.path.exists(file_path):
        raise IOError("No file exists at %s" % file_path)

    # Just return the file
    if not print_time:
        return pd.read_csv(file_path, **kwargs)

    # Print out some timing info while reading
    start = time.perf_counter()
    print('\tReading "%s"' % file_path, end="")
    df = pd.read_csv(file_path, **kwargs)
    print(" - Done in %fs" % (time.perf_counter() - start))
    return df


def is_none_like(o) -> bool:
    """
    Checks if o is none-like

    Parameters
    ----------
    o:
        Object to check

    Returns
    -------
    bool:
        True if o is none-like else False
    """
    if o is None:
        return True

    if isinstance(o, str):
        if o.lower().strip() == 'none':
            return True

    if isinstance(o, list):
        return all([is_none_like(x) for x in o])

    return False


def get_vdm_dist_name(trip_origin: str,
                      matrix_format: str,
                      year: Union[int, str],
                      user_class: str,
                      mode: Union[int, str],
                      ca: int = None,
                      tp: Union[int, str] = None,
                      csv: bool = False,
                      suffix: str = None
                      ) -> str:
    """
    Wrapper around get_compiled_matrix_name to deal with different ca naming
    """
    compiled_name = get_compiled_matrix_name(
        matrix_format,
        user_class,
        str(year),
        trip_origin=trip_origin,
        mode=str(mode),
        ca=ca,
        tp=str(tp),
        csv=csv,
        suffix=suffix
    )

    # Need to switch over ca naming
    if ca is not None:
        if 'nca' in compiled_name:
            compiled_name = compiled_name.replace('nca', 'ca1')
        elif 'ca' in compiled_name:
            compiled_name = compiled_name.replace('ca', 'ca2')
        else:
            raise ValueError("Couldn't find ca/nca in name returned from "
                             "get_compiled_matrix_name(). This shouldn't be "
                             "able to happen!")

    return compiled_name


def get_dist_name(trip_origin: str,
                  matrix_format: str,
                  year: str = None,
                  purpose: str = None,
                  mode: str = None,
                  segment: str = None,
                  car_availability: str = None,
                  tp: str = None,
                  csv: bool = False,
                  compressed: bool = False,
                  suffix: str = None,
                  ) -> str:
    """
    Generates the distribution name
    """
    # Generate the base name
    name_parts = [
        trip_origin,
        matrix_format,
    ]

    # Optionally add the extra segmentation
    if not is_none_like(year):
        name_parts += ["yr" + year]

    if not is_none_like(purpose):
        name_parts += ["p" + purpose]

    if not is_none_like(mode):
        name_parts += ["m" + mode]

    if not is_none_like(segment) and not is_none_like(purpose):
        seg_name = "soc" if int(purpose) in efs_consts.SOC_P else "ns"
        name_parts += [seg_name + segment]

    if not is_none_like(car_availability):
        name_parts += ["ca" + car_availability]

    if not is_none_like(tp):
        name_parts += ["tp" + tp]

    # Create name string
    final_name = '_'.join(name_parts)

    # Optionally add a custom f_type suffix
    if suffix is not None:
        final_name += suffix

    # Optionally add on the csv if needed
    if csv:
        final_name += '.csv'
    elif compressed:
        final_name += consts.COMPRESSION_SUFFIX

    return final_name


def calib_params_to_dist_name(trip_origin: str,
                              matrix_format: str,
                              calib_params: Dict[str, int],
                              csv: bool = False,
                              compressed: bool = False,
                              suffix: str = None,
                              ) -> str:
    """
    Wrapper for get_distribution_name() using calib params
    """
    segment_str = 'soc' if calib_params['p'] in efs_consts.SOC_P else 'ns'

    return get_dist_name(
        trip_origin=trip_origin,
        matrix_format=matrix_format,
        year=str(calib_params.get('yr')),
        purpose=str(calib_params.get('p')),
        mode=str(calib_params.get('m')),
        segment=str(calib_params.get(segment_str)),
        car_availability=str(calib_params.get('ca')),
        tp=str(calib_params.get('tp')),
        csv=csv,
        compressed=compressed,
        suffix=suffix,
    )


def get_dist_name_parts(dist_name: str) -> List[str]:
    """
    Splits a full dist name into its individual components


    Parameters
    ----------
    dist_name:
        The dist name to parse

    Returns
    -------
    name_parts:
        dist_name split into parts. Returns in the following order:
        [trip_origin, matrix_format, year, purpose, mode, segment, ca, tp]
    """
    if dist_name[-4:] == '.csv':
        dist_name = dist_name[:-4]

    name_parts = dist_name.split('_')

    # TODO: Can this be done smarter?
    return [
        name_parts[0],
        name_parts[1],
        name_parts[2][-4:],
        name_parts[3][-1:],
        name_parts[4][-1:],
        name_parts[5][-1:],
        name_parts[6][-1:],
        name_parts[7][-1:],
    ]


def get_seg_level_dist_name(seg_level: str,
                            seg_values: Dict[str, Any],
                            matrix_format: str,
                            year: Union[str, int],
                            trip_origin: str = None,
                            csv: bool = False,
                            suffix: str = None
                            ) -> str:
    """
    Generates the distribution name, regardless of segmentation level

    Parameters
    ----------
    seg_level:
        The level of segmentation of the tour proportions to convert. This
        should be one of the values in efs_constants.SEG_LEVELS.

    seg_values:
        A dictionary of {seg_name: value} for the segmentation level chosen.

    matrix_format:
        The format of the matrix. Usually 'pa', 'od', 'od_from', or 'od_to'.

    year:
        The year of the matrix.

    trip_origin:
        Usually 'hb or 'nhb'

    csv:
        Whether to add .csv on the end of the file or not

    suffix:
        Any additional suffix to add to the end of the filename. This comes
        before .csv if csv=True.

    Returns
    -------
    dist_name:
        The generated distribution name
    """
    # Init
    seg_level = validate_seg_level(seg_level)

    if seg_level == 'vdm':
        return get_vdm_dist_name(
            trip_origin=seg_values.get('to'),
            matrix_format=matrix_format,
            year=str(year),
            user_class=seg_values.get('uc'),
            mode=seg_values.get('m'),
            ca=seg_values.get('ca'),
            tp=seg_values.get('tp'),
            csv=csv,
            suffix=suffix
        )

    else:
        raise ValueError("'%s' is a valid seg_level, however, we do not have "
                         "a way of dealing with it right not. You should "
                         "write it!" % seg_level)


def generate_calib_params(year: str = None,
                          purpose: int = None,
                          mode: int = None,
                          segment: int = None,
                          ca: int = None,
                          tp: int = None
                          ) -> dict:
    """
    Returns a TMS style calib_params dict
    """
    # Purpose needs to be set if segment is
    if segment is not None and purpose is None:
        raise ValueError("If segment is set, purpose needs to be set too, "
                         "otherwise segment text cannot be determined.")
    # Init
    segment_str = 'soc' if purpose in efs_consts.SOC_P else 'ns'

    keys = ['yr', 'p', 'm', segment_str, 'ca', 'tp']
    vals = [year, purpose, mode, segment, ca, tp]

    # Add params to dict if they are not None
    return {k: v for k, v in zip(keys, vals) if v is not None}


def create_vdm_seg_values(trip_origin: str = None,
                          user_class: str = None,
                          mode: int = None,
                          ca: int = None,
                          tp: int = None,
                          ) -> Dict[str, Union[str, int]]:
    """
    Returns a TMS style calib_params dict, but for vdm segmentation
    """
    keys = ['to', 'uc', 'm', 'ca', 'tp']
    vals = [trip_origin, user_class, mode, ca, tp]

    # Add params to dict if they are not None
    return {k: v for k, v in zip(keys, vals) if v is not None}


def starts_with(s: str, x: str) -> bool:
    """
    Boolean test to see if string s starts with string x or not.

    Parameters
    ----------
    s:
        The string to test

    x:
        The string to search for

    Returns
    -------
    Bool:
        True if s starts with x, else False.
    """
    search_string = '^' + x
    return re.search(search_string, s) is not None


def post_me_fname_to_calib_params(fname: str,
                                  get_user_class: bool = True,
                                  force_year: int = None
                                  ) -> Dict[str, str]:
    """
    Convert the filename into a calib_params dict, with the following keys
    (if they exist in the filename):
    yr, p, m, soc/ns, ca, tp
    """
    # Init
    calib_params = {}

    # Might need to save or recreate this filename

    # Assume year starts in 20/21
    loc = re.search('2[0-1][0-9]+', fname)
    if loc is not None:
        calib_params['yr'] = int(fname[loc.start():loc.end()])

    # Force the year if we need to
    if force_year is not None and 'yr' not in calib_params.keys():
        calib_params['yr'] = force_year

    # Mode.
    loc = re.search('_m[0-9]+', fname)
    if loc is not None:
        calib_params['m'] = int(fname[loc.start() + 2:loc.end()])
    elif re.search('_Hwy', fname) is not None:
        calib_params['m'] = 3
    else:
        # What is the code for rail?
        Warning("Cannot find a mode in filename. It might be rail, but I "
                "don't know what to search for at the moment.\n"
                "File name: '%s'" % fname)

    # tp
    loc = re.search('_TS[0-9]+', fname)
    loc2 = re.search('_tp[0-9]+', fname)
    if loc is not None:
        calib_params['tp'] = int(fname[loc.start() + 3:loc.end()])
    elif loc2 is not None:
        calib_params['tp'] = int(fname[loc2.start() + 3:loc2.end()])

    # User Class
    if get_user_class:
        if re.search('_commute', fname) is not None:
            calib_params['user_class'] = 'commute'
        elif re.search('_business', fname) is not None:
            calib_params['user_class'] = 'business'
        elif re.search('_other', fname) is not None:
            calib_params['user_class'] = 'other'
        else:
            raise ValueError("Cannot find the user class in filename: %s" %
                             str(fname))

    return calib_params


def fname_to_calib_params(fname: str,
                          get_trip_origin: bool = False,
                          get_matrix_format: bool = False,
                          get_user_class: bool = False,
                          force_ca_exists: bool = False,
                          ) -> Dict[str, Union[str, int]]:
    """
    Convert the filename into a calib_params dict, with the following keys
    (if they exist in the filename):
    yr, p, m, soc/ns, ca, tp
    """
    # Init
    calib_params = dict()

    # Search for each param in fname - store if found
    # year
    loc = re.search('_yr[0-9]+', fname)
    if loc is not None:
        calib_params['yr'] = int(fname[loc.start() + 3:loc.end()])

    # purpose
    loc = re.search('_p[0-9]+', fname)
    if loc is not None:
        calib_params['p'] = int(fname[loc.start() + 2:loc.end()])

    # mode
    loc = re.search('_m[0-9]+', fname)
    if loc is not None:
        calib_params['m'] = int(fname[loc.start() + 2:loc.end()])

    # soc
    loc = re.search('_soc[0-9]+', fname)
    if loc is not None:
        calib_params['soc'] = int(fname[loc.start() + 4:loc.end()])

    # ns
    loc = re.search('_ns[0-9]+', fname)
    if loc is not None:
        calib_params['ns'] = int(fname[loc.start() + 3:loc.end()])

    # ca
    loc = re.search('_ca[0-9]+', fname)
    if loc is not None:
        calib_params['ca'] = int(fname[loc.start() + 3:loc.end()])
    elif re.search('_nca', fname) is not None:
        calib_params['ca'] = 1
    elif re.search('_ca', fname) is not None:
        calib_params['ca'] = 2

    if force_ca_exists:
        if 'ca' not in calib_params:
            calib_params['ca'] = None

    # tp
    loc = re.search('_tp[0-9]+', fname)
    if loc is not None:
        calib_params['tp'] = int(fname[loc.start() + 3:loc.end()])

    # Optionally search for extra params
    if get_trip_origin:
        if re.search('^hb_', fname) is not None:
            calib_params['trip_origin'] = 'hb'
        elif re.search('^nhb_', fname) is not None:
            calib_params['trip_origin'] = 'nhb'
        else:
            raise ValueError("Cannot find the trip origin in filename: %s" %
                             str(fname))

    if get_matrix_format:
        if re.search('od_from_', fname) is not None:
            calib_params['matrix_format'] = 'od_from'
        elif re.search('od_to_', fname) is not None:
            calib_params['matrix_format'] = 'od_to'
        elif re.search('od_', fname) is not None:
            calib_params['matrix_format'] = 'od'
        elif re.search('pa_', fname) is not None:
            calib_params['matrix_format'] = 'pa'
        else:
            raise ValueError("Cannot find the matrix format in filename: %s" %
                             str(fname))

    if get_user_class:
        if re.search('commute_', fname) is not None:
            calib_params['user_class'] = 'commute'
        elif re.search('business_', fname) is not None:
            calib_params['user_class'] = 'business'
        elif re.search('other_', fname) is not None:
            calib_params['user_class'] = 'other'
        else:
            raise ValueError("Cannot find the user class in filename: %s" %
                             str(fname))

    return calib_params


def get_segmentation_mask(df: pd.DataFrame,
                          col_vals: dict,
                          ignore_missing_cols=False
                          ) -> pd.Series:
    """
    Creates a mask on df, optionally skipping non-existent columns

    Parameters
    ----------
    df:
        The dataframe to make the mask from.

    col_vals:
        A dictionary of column names to wanted values.

    ignore_missing_cols:
        If True, and error will not be raised when a given column in
        col_val does not exist.

    Returns
    -------
    segmentation_mask:
        A pandas.Series of boolean values
    """
    # Init Mask
    mask = pd.Series([True] * len(df))

    # Narrow down mask
    for col, val in col_vals.items():
        # Make sure column exists
        if col not in df.columns:
            if ignore_missing_cols:
                continue
            else:
                raise KeyError("'%s' does not exist in DataFrame."
                               % str(col))

        mask &= (df[col] == val)

    return mask


def expand_distribution(dist: pd.DataFrame,
                        year: str,
                        purpose: str,
                        mode: str,
                        segment: str = None,
                        car_availability: str = None,
                        id_vars='p_zone',
                        var_name='a_zone',
                        value_name='trips',
                        year_col: str = 'yr',
                        purpose_col: str = 'p',
                        mode_col: str = 'm',
                        soc_col: str = 'soc',
                        ns_col: str = 'ns',
                        ca_col: str = 'ca',
                        int_conversion: bool = True
                        ) -> pd.DataFrame:
    """
    Returns a converted distribution  - converted from wide to long
    format, adding in a column for each segmentation

    WARNING: This only works with a single id_vars
    """
    dist = dist.copy()

    # Convert from wide to long
    # This way we can avoid the name of the first col
    dist = dist.melt(
        id_vars=dist.columns[:1],
        var_name=var_name,
        value_name=value_name
    )
    id_vars = id_vars[0] if isinstance(id_vars, list) else id_vars
    dist.columns.values[0] = id_vars

    # Convert the melted cols to ints
    # This prevents string/int clashes later
    if int_conversion:
        dist[id_vars] = dist[id_vars].astype(int)
        dist[var_name] = dist[var_name].astype(int)

    # Add new columns
    dist[purpose_col] = purpose
    dist[mode_col] = mode

    # Optionally add other columns
    if not is_none_like(year):
        dist[year_col] = year

    if not is_none_like(car_availability):
        dist[ca_col] = car_availability

    if not is_none_like(segment):
        if purpose in efs_consts.SOC_P:
            dist[soc_col] = segment
            dist[ns_col] = 'none'
        elif purpose in efs_consts.NS_P:
            dist[soc_col] = 'none'
            dist[ns_col] = segment
        else:
            raise ValueError(
                "%s is not a valid HB or NHB purpose" % str(purpose)
            )

    return dist


def ensure_segmentation(df: pd.DataFrame,
                        p_needed: List[int] = None,
                        m_needed: List[int] = None,
                        soc_needed: List[int] = None,
                        ns_needed: List[int] = None,
                        ca_needed: List[int] = None,
                        tp_needed: List[int] = None,
                        p_col: str = 'p',
                        m_col: str = 'm',
                        soc_col: str = 'soc',
                        ns_col: str = 'ns',
                        ca_col: str = 'ca',
                        tp_col: str = 'tp',
                        ignore_ns: bool = False,
                        ignore_ca: bool = False
                        ) -> pd.DataFrame:
    """
    Ensures the required segmentation exists in the given dataframe

    Check is carried out by ensuring a column exists in the dataframe if the
    list of inputs for that segmentation is not None, or only contains one
    item. This function will also try to make sure the columns are in the
    correct data types, and will return a copy of the given df after
    conversion.

    Parameters
    ----------
    df:
        The dataframe to ensure the segmentation exists in.

    p_needed:
        A list of the purposes to segment by, or None if no segmentation
        required by purpose.

    m_needed:
        A list of the modes to segment by, or None if no segmentation
        required by mode.

    soc_needed:
        A list of soc categories to segment by, or None if no segmentation
        required by soc.

    ns_needed:
        A list of ns categories to segment by, or None if no segmentation
        required by ns

    ca_needed:
        A list of ca categories to segment by, or None if no segmentation
        required by car availability

    tp_needed:
        A list of time periods to segment by, or None if no segmentation
        required by time period.

    p_col:
        The name of the column containing purpose data.
    
    m_col:
        The name of the column containing mode data.
    
    soc_col:
        The name of the column containing soc data.

    ns_col:
        The name of the column containing ns data.

    ca_col:
        The name of the column containing car availability data.

    tp_col:
        The name of the column containing time period data.

    ignore_ns:
        If True, the ns segmentation will not be checked regardless of the
        value of ns_needed. This is useful when checking the segmentation of
        attractions which never have ns segmentation.

    ignore_ca:
        If True, the car availability segmentation will not be checked
        regardless of the value of ca_needed. This is useful when checking the
        segmentation of attractions which do not have ca segmentation.

    Returns
    -------
    converted_df:
        The given df with some data type conversions applied to make sure
        columns are using the correct data types.

    Raises
    ------
    NormitsDemandError:
        If the arguments given determine that a segmenation should exist, but
        no segmentation can be found in df.
    """
    # Init
    col_data_dict = {
        p_col: p_needed,
        m_col: m_needed,
        soc_col: soc_needed,
        ns_col: ns_needed,
        ca_col: ca_needed,
        tp_col: tp_needed
    }

    # Remove what we're ignoring
    if ignore_ns:
        del col_data_dict[ns_col]

    if ignore_ca:
        del col_data_dict[ca_col]

    # Sanitise the df. Make sure column names are strings and soc/ns are
    # strings
    df.columns = df.columns.astype(str)

    if soc_col in list(df):
        df[soc_col] = df[soc_col].astype(str)

    if ns_col in list(df):
        df[ns_col] = df[ns_col].astype(str)

    # Build a list of the columns we need to check for
    check_cols = list()
    for col, seg in col_data_dict.items():
        if seg is None or len(seg) <= 1:
            continue

        check_cols.append(col)

    if len(check_cols) == 0:
        raise NormitsDemandError(
            "There doesn't seem to be any segmentation we need to check for."
            "Please make sure the segmentation parameters given are not None, "
            "and longer than 1 item."
        )

    # Check all the columns exist
    for col in check_cols:
        if col not in list(df):
            raise KeyError(
                "No column exists in the given dataframe for %s segmentation."
                % col
            )
        
    return df


def vdm_segment_loop_generator(to_list: Iterable[str],
                               uc_list: Iterable[str],
                               m_list: Iterable[int],
                               ca_list: Iterable[int],
                               tp_list: Iterable[int] = None,
                               ) -> (Union[Iterator[Tuple[str, str, int, int, int]],
                                           Iterator[Tuple[str, str, int, int]]]):
    """
    Simple generator to avoid the need for so many nested loops
    """

    for trip_origin, user_class in product(to_list, uc_list):
        # Not a valid segmentation - skip it
        if trip_origin == 'nhb' and user_class == 'commute':
            continue

        for mode, ca in product(m_list, ca_list):
            if tp_list is None:
                yield (
                    trip_origin,
                    user_class,
                    mode,
                    ca,
                )
            else:
                for time_period in tp_list:
                    yield(
                        trip_origin,
                        user_class,
                        mode,
                        ca,
                        time_period
                    )


def segmentation_loop_generator(p_list: Iterable[int],
                                m_list: Iterable[int],
                                soc_list: Iterable[int],
                                ns_list: Iterable[int],
                                ca_list: Iterable[int],
                                tp_list: Iterable[int] = None
                                ) -> (Union[Iterator[Tuple[int, int, int, int, int]],
                                            Iterator[Tuple[int, int, int, int]]]):
    """
    Simple generator to avoid the need for so many nested loops
    """
    for purpose in p_list:
        if purpose in efs_consts.SOC_P:
            required_segments = soc_list
        elif purpose in efs_consts.NS_P:
            required_segments = ns_list
        else:
            raise ValueError("'%s' does not seem to be a valid soc or ns "
                             "purpose." % str(purpose))
        for mode in m_list:
            for segment in required_segments:
                for car_availability in ca_list:
                    if tp_list is None:
                        yield (
                            purpose,
                            mode,
                            segment,
                            car_availability
                        )
                    else:
                        for tp in tp_list:
                            yield (
                                purpose,
                                mode,
                                segment,
                                car_availability,
                                tp
                            )


def cp_segmentation_loop_generator(p_list: Iterable[int],
                                   m_list: Iterable[int],
                                   soc_list: Iterable[int] = None,
                                   ns_list: Iterable[int] = None,
                                   ca_list: Iterable[int] = None,
                                   tp_list: Iterable[int] = None
                                   ) -> Iterator[Dict[str, int]]:
    """
    Wrapper for segmentation_loop_generator() to return TMS style
    calib params instead of a number of integer
    """
    # Init
    soc_list = [None] if soc_list is None else soc_list
    ns_list = [None] if ns_list is None else ns_list
    ca_list = [None] if ca_list is None else ca_list
    tp_list = [None] if tp_list is None else tp_list

    loop_generator = segmentation_loop_generator(
        p_list=p_list,
        m_list=m_list,
        soc_list=soc_list,
        ns_list=ns_list,
        ca_list=ca_list,
        tp_list=tp_list
    )

    for p, m, seg, ca, tp in loop_generator:
        yield generate_calib_params(
            purpose=p,
            mode=m,
            segment=seg,
            ca=ca,
            tp=tp
        )


def segment_loop_generator(seg_dict: Dict[str, List[Any]],
                           ) -> Iterator[Dict[str, Any]]:
    """
    Yields seg_values dictionary for all unique combinations of seg_dict

    Parameters
    ----------
    seg_dict:
        Dictionary of {seg_names: [seg_vals]}. All possible combinations of
        seg_values will be iterated through

    Returns
    -------
    seg_values:
        A dictionary of {seg_name: seg_value}
    """
    # Separate keys and values
    keys, vals = zip(*seg_dict.items())

    for unq_seg in product(*vals):
        yield {keys[i]: unq_seg[i] for i in range(len(keys))}


def build_seg_params(seg_level: str,
                     df: pd.DataFrame,
                     raise_error: bool = False
                     ) -> Dict[str, Any]:
    """
    Builds a dictionary of seg_params from df for seg_level

    Parameters
    ----------
    seg_level:
        The level of segmentation of the tour proportions to convert. This
        should be one of the values in efs_constants.SEG_LEVELS.

    df:
        The dataframe to pull the unique segments from and build the seg_params

    raise_error:
        Whether to raise an error or not when a segmentation does not exist in
        df.

    Returns
    -------
    seg_params:
        A dictionary of {kwarg: values} for the segmentation level chosen. See...
        Need to Create reference for how seg_params work
    """
    # Init
    seg_level = validate_seg_level(seg_level)

    # Choose the segments to look for
    if seg_level == 'tfn':
        segments = ['p', 'm', 'soc', 'ns', 'ca']
    else:
        raise ValueError(
            "Cannot determine the unique segments for seg_level '%s', we don't "
            "have a branch for it!" % str(seg_level)
        )

    # Get the unique values for each segment
    unq_segs = dict()
    for segment in segments:
        if segment not in df and raise_error:
            raise KeyError(
                "Segment '%s' does not exist in df" % segment
            )

        # Ignore none-like splits
        unique = df[segment].unique().tolist()
        unique = [x for x in unique if not is_none_like(x)]
        unq_segs[segment] = unique

    # Build the seg_params
    if seg_level == 'tfn':
        seg_params = {
            'p_needed': unq_segs.get('p', None),
            'm_needed': unq_segs.get('m', None),
            'soc_needed': unq_segs.get('soc', None),
            'ns_needed': unq_segs.get('ns', None),
            'ca_needed': unq_segs.get('ca', None),
        }
    else:
        raise ValueError(
            "Cannot determine the unique segments for seg_level '%s', we don't "
            "have a branch for it!" % str(seg_level)
        )

    return seg_params


def seg_level_loop_length(seg_level: str,
                          seg_params: Dict[str, Any],
                          ) -> int:
    """
    Returns the length of the generator that would be created if
    seg_level_loop_generator() was called with the same arguments

    Parameters
    ----------
    seg_level:
        The level of segmentation of the tour proportions to convert. This
        should be one of the values in efs_constants.SEG_LEVELS.

    seg_params:
        A dictionary of kwarg: values for the segmentation level chosen. See...
        Need to Create reference for how seg_params work

    Yields
    -------
    generator_length:
        The total number of items that will be yielded from
        seg_level_loop_generator() when called with the same arguments
    """
    return len(list(seg_level_loop_generator(seg_level, seg_params)))


def seg_level_loop_generator(seg_level: str,
                             seg_params: Dict[str, Any],
                             ) -> Iterator[Dict[str, Union[int, str]]]:
    """
    Yields seg_values dictionary for the seg_level given

    Parameters
    ----------
    seg_level:
        The level of segmentation of the tour proportions to convert. This
        should be one of the values in efs_constants.SEG_LEVELS.

    seg_params:
        A dictionary of kwarg: values for the segmentation level chosen. See...
        Need to Create reference for how seg_params work

    Yields
    -------
    seg_values:
        A dictionary of {seg_name: seg_value}
    """
    # Init
    seg_level = validate_seg_level(seg_level)

    if seg_level == 'vdm':
        seg_params = validate_vdm_seg_params(seg_params)

        loop_generator = vdm_segment_loop_generator(
            to_list=seg_params.get('to_needed', [None]),
            uc_list=seg_params.get('uc_needed', [None]),
            m_list=seg_params.get('m_needed', [None]),
            ca_list=seg_params.get('ca_needed', [None]),
            tp_list=seg_params.get('tp_needed', [None]),
        )

        # Convert to dict
        for to, uc, m, ca, tp in loop_generator:
            yield create_vdm_seg_values(
                trip_origin=to,
                user_class=uc,
                mode=m,
                ca=ca,
                tp=tp
            )

    if seg_level == 'tfn':
        loop_gen = cp_segmentation_loop_generator(
            p_list=seg_params.get('p_needed', [None]),
            m_list=seg_params.get('m_needed', [None]),
            soc_list=seg_params.get('soc_needed', [None]),
            ns_list=seg_params.get('ns_needed', [None]),
            ca_list=seg_params.get('ca_needed', [None]),
            tp_list=seg_params.get('tp_needed', [None]),
        )

        for seg_params in loop_gen:
            yield seg_params

    else:
        raise ValueError("'%s' is a valid seg_level, however, we do not have "
                         "a way of dealing with it right not. You should "
                         "write it!" % seg_level)


def segmentation_order(segmentation_lst: List[str]) -> List[str]:
    """
    Returns the segmentation_lst in its hierarchical order for segmentation.

    Any non-segmentation keys will be appended onto the end of the list

    Parameters
    ----------
    segmentation_lst:
        A list of segmentation keys. See efs_consts.SEGMENTATION_ORDER for a list
        of valid values

    Returns
    -------
    list:
        segmentation_lst in its expected order. This is the
        same order as filenames etc.
    """
    # Init
    seg_order = efs_consts.SEGMENTATION_ORDER.copy()

    # Order the segmentation keys, stick non seg back on the end
    non_seg_vals = [x for x in segmentation_lst if x not in seg_order]
    ordered_seg_vals = [x for x in seg_order if x in segmentation_lst]
    return ordered_seg_vals + non_seg_vals


def sort_vector_cols(vector: pd.DataFrame,
                     zone_col: str = None,
                     in_place: bool = False,
                     ) -> pd.DataFrame:
    """
    Re-orders the columns if vector to be in the correct segmentation order

    The zone column will be placed first, and any other non-segment columns
    are appended to the end of the order.

    Parameters
    ----------
    vector:
        The vector to re-order the columns of

    zone_col:
        The name of the column containing the zone_id data. If not given,
        it will be inferred by looking for the first column containing
        "zone_id".

    in_place:
        Whether to re-order in place or return a copy of the given dataframe.

    Returns
    -------
    reindexed_vector:
        The given vector re-indexed to to be in the correct segmentation order,
        as defined by efs_consts.SEGMENTATION_ORDER.
    """
    # init
    columns = list(vector)

    if not in_place:
        vector = vector.copy()

    # Infer the zone col if not given
    if zone_col is None:
        zone_col_candidates = [x for x in columns if '_zone_id' in x]
        if zone_col_candidates == list():
            raise ValueError(
                "No zone_col argument was given. Tried to infer which "
                "column to use, but there were not columns containing "
                "'zone_col'."
            )
        zone_col = zone_col_candidates[0]

    # Build a list of the final output order
    col_order = segmentation_order(list_safe_remove(columns, [zone_col]))
    col_order = [zone_col] + col_order

    # Reindex and return
    return vector.reindex(columns=col_order)


def seg_dict_key_order(segmentation_dict: Dict[str, Any]) -> List[str]:
    """
    Returns the keys of segmentation_dict in their hierarchical order

    Parameters
    ----------
    segmentation_dict:
        A dictionary of segmentation keys and their values. AKA Calib_params

    Returns
    -------
    list:
        A list of segmentation_dict keys in their expected order. This is the
        same order as filenames etc.
    """
    return segmentation_order(list(segmentation_dict.keys()))


def long_to_wide_out(df: pd.DataFrame,
                     v_heading: str,
                     h_heading: str,
                     values: str,
                     out_path: str,
                     unq_zones: List[str] = None,
                     round_dp: int = 12,
                     ) -> None:
    """
    Converts a long format pd.Dataframe, converts it to long and writes
    as a csv to out_path

    Parameters
    ----------
    df:
        The dataframe to convert and output

    v_heading:
        Column name of df to be the vertical heading.

    h_heading:
        Column name of df to be the horizontal heading.

    values:
        Column name of df to be the values.

    out_path:
        Where to write the converted matrix.

    unq_zones:
        A list of all the zone names that should exist in the output matrix.
        If zones in this list are not in the given df, they are infilled with
        values of 0.
        If left as None, it assumes all zones in the range 1 to max zone number
        should exist.

    round_dp:
        The number of decimal places to round the output to

    Returns
    -------
        None
    """
    # Init
    df = df.copy()

    # Get the unique column names
    if unq_zones is None:
        unq_zones = df[v_heading].drop_duplicates().reset_index(drop=True).copy()
        unq_zones = list(range(1, max(unq_zones)+1))

    # Make sure all unq_zones exists in v_heading and h_heading
    df = ensure_multi_index(
        df=df,
        index_dict={v_heading: unq_zones, h_heading: unq_zones},
    )

    # Convert to wide format and round
    df = df.pivot(
        index=v_heading,
        columns=h_heading,
        values=values
    ).round(decimals=round_dp)

    # Finally, write to disk
    df.to_csv(out_path)


def wide_to_long_out(df: pd.DataFrame,
                     id_vars: str,
                     var_name: str,
                     value_name: str,
                     out_path: str
                     ) -> None:
    # TODO: Write wide_to_long_out() docs
    # This way we can avoid the name of the first col
    df = df.melt(
        id_vars=df.columns[:1],
        var_name=var_name,
        value_name=value_name
    )
    id_vars = id_vars[0] if isinstance(id_vars, list) else id_vars
    df.columns.values[0] = id_vars

    df.to_csv(out_path, index=False)


def get_compile_params_name(matrix_format: str,
                            year: str,
                            suffix: str = None
                            ) -> str:
    """
    Generates the compile params filename
    """
    if suffix is None:
        return "%s_yr%s_compile_params.csv" % (matrix_format, year)

    return "%s_yr%s_%s_compile_params.csv" % (matrix_format, year, suffix)


def get_split_factors_fname(matrix_format: str,
                            year: str,
                            suffix: str = None
                            ) -> str:
    """
    Generates the splitting factors filename
    """
    ftype = consts.COMPRESSION_SUFFIX
    if suffix is None:
        return "%s_yr%s_splitting_factors.%s" % (matrix_format, year, ftype)

    return "%s_yr%s_%s_splitting_factors.%s" % (matrix_format, year, suffix, ftype)


def build_full_paths(base_path: str,
                     fnames: Iterable[str]
                     ) -> List[str]:
    """
    Prepends the base_path name to all of the given fnames
    """
    return [os.path.join(base_path, x) for x in fnames]


def list_files(path: str,
               include_path: bool = False
               ) -> List[str]:
    """
    Returns the names of all files (excluding directories) at the given path

    Parameters
    ----------
    path:
        Where to search for the files

    include_path:
        Whether to include the path with the returned filenames

    Returns
    -------
    files:
        Either filenames, or the paths to the found files

    """
    if include_path:
        file_paths = build_full_paths(path, os.listdir(path))
        return [x for x in file_paths if os.path.isfile(x)]
    else:
        fnames = os.listdir(path)
        return [x for x in fnames if os.path.isfile(os.path.join(path, x))]


def is_in_string(vals: Iterable[str],
                 string: str
                 ) -> bool:
    """
    Returns True if any of vals is in string, else False
    """
    for v in vals:
        if v in string:
            return True
    return False


def get_compiled_matrix_name(matrix_format: str,
                             user_class: str,
                             year: str,
                             trip_origin: str = None,
                             mode: str = None,
                             ca: int = None,
                             tp: str = None,
                             csv: bool = False,
                             compress: bool = False,
                             suffix: str = None,
                             ) -> str:

    """
    Generates the compiled matrix name
    """
    # Generate the base name
    name_parts = [
        matrix_format,
        user_class
    ]

    # Optionally add the extra segmentation
    if not is_none_like(trip_origin):
        name_parts = [trip_origin] + name_parts

    if not is_none_like(year):
        name_parts += ["yr" + year]

    if not is_none_like(mode):
        name_parts += ["m" + mode]

    if not is_none_like(ca):
        if ca == 1:
            name_parts += ["nca"]
        elif ca == 2:
            name_parts += ["ca"]
        else:
            raise ValueError("Received an invalid car availability value. "
                             "Got %s, expected either 1 or 2." % str(ca))

    if not is_none_like(tp):
        name_parts += ["tp" + tp]

    # Create name string
    final_name = '_'.join(name_parts)

    # Optionally add a custom f_type suffix
    if suffix is not None:
        final_name += suffix

    # Optionally add on the csv if needed
    if csv:
        final_name += '.csv'
    elif compress:
        final_name += consts.COMPRESSION_SUFFIX

    return final_name


def write_csv(headers: Iterable[str],
              out_lines: List[Iterable[str]],
              out_path: str
              ) -> None:
    """
    Writes the given headers and outlines as a csv to out_path

    Parameters
    ----------
    headers
    out_lines
    out_path

    Returns
    -------
    None
    """
    # Make sure everything is a string
    headers = [str(x) for x in headers]
    out_lines = [[str(x) for x in y] for y in out_lines]

    all_out = [headers] + out_lines
    all_out = [','.join(x) for x in all_out]
    with open(out_path, 'w') as f:
        f.write('\n'.join(all_out))


def check_tour_proportions(tour_props: Dict[int, Dict[int, np.array]],
                           n_tp: int,
                           n_row_col: int,
                           n_tests: int = 10
                           ) -> None:
    """
    Carries out some checks to make sure the tour proportions are in the
    correct format. Will randomly check n_tests vals.

    Parameters
    ----------
    tour_props:
        A loaded tour proportions dictionary to check.

    n_tp:
        The number of time periods to be expected.

    n_row_col:
        Assumes square PA/OD matrices. The number of zones in the matrices.

    n_tests:
        The number of random tests to carry out.

    Returns
    -------
    None
    """
    # Get a list of keys - Assume completely square dict
    first_keys = list(tour_props.keys())
    second_keys = list(tour_props[first_keys[0]].keys())

    # Check dict shape
    if len(first_keys) != n_row_col or len(second_keys) != n_row_col:
        raise ValueError(
            "Tour proportions dictionary is not the expected shape. Expected "
            "a shape of (%d, %d), but got (%d, %d)."
            % (n_row_col, n_row_col, len(first_keys), len(second_keys))
        )

    # Check nested np.array shapes
    for _ in range(n_tests):
        key_1 = random.choice(first_keys)
        key_2 = random.choice(second_keys)

        if tour_props[key_1][key_2].shape != (n_tp, n_tp):
            raise ValueError(
                "Tour proportion matrices are not the expected shape. Expected "
                "a shape of (%d, %d), but found a shape of %s at "
                "tour_props[%s][%s]."
                % (n_tp, n_tp, str(tour_props[key_1][key_2].shape),
                   str(key_1), str(key_2))
            )

    # If here, all checks have passed
    return


def combine_yearly_dfs(year_dfs: Dict[str, pd.DataFrame],
                       unique_col: str,
                       p_col: str = 'p',
                       purposes: List[int] = None
                       ) -> pd.DataFrame:
    """
    Efficiently concatenates the yearly dataframes in year_dfs.

    Parameters
    ----------
    year_dfs:
        Dictionary, with keys as the years, and the productions data for that
        year as a value.

    unique_col:
        The name of the column containing the data - i.e. the productions for
        that year

    p_col:
        The name of the column containing the purpose values.

    purposes:
        A list of the purposes to keep. If left as None, all purposes are
        kept

    Returns
    -------
    combined_productions:
        A dataframe of all combined data in year_dfs. There will be a separate
        column for each year of data.
    """
    # Init
    keys = list(year_dfs.keys())
    merge_cols = list(year_dfs[keys[0]])
    merge_cols.remove(unique_col)

    if purposes is None:
        purposes = year_dfs[keys[0]][p_col].unique()

    # ## SPLIT MATRICES AND JOIN BY PURPOSE ## #
    purpose_ph = list()
    desc = "Merging dataframes by purpose"
    for p in tqdm(purposes, desc=desc):

        # Get all the matrices that belong to this purpose
        yr_p_dfs = list()
        for year, df in year_dfs.items():
            temp_df = df[df[p_col] == p].copy()
            temp_df = temp_df.rename(columns={unique_col: year})
            yr_p_dfs.append(temp_df)

        # Iteratively merge all matrices into one
        merged_df = yr_p_dfs[0]
        for df in yr_p_dfs[1:]:
            merged_df = pd.merge(
                merged_df,
                df,
                on=merge_cols
            )
        purpose_ph.append(merged_df)
        del yr_p_dfs

    # ## CONCATENATE ALL MERGED MATRICES ## #
    return pd.concat(purpose_ph)


def get_mean_tp_splits(tp_split_path: str,
                       p: int,
                       aggregate_to_weekday: bool = True,
                       p_col: str = 'purpose',
                       tp_as: str = 'str'
                       ) -> pd.DataFrame:
    """
    TODO: Write get_mean_tp_splits() doc

    Parameters
    ----------
    tp_split_path
    p
    aggregate_to_weekday
    p_col
    tp_as

    Returns
    -------

    """
    # Init
    tp_splits = pd.read_csv(tp_split_path)
    p_tp_splits = tp_splits[tp_splits[p_col] == p].copy()

    # If more than one row, we have a problem!
    if len(p_tp_splits) > 1:
        raise ValueError("Found more than one row in the mean time period "
                         "splits file.")

    if aggregate_to_weekday:
        tp_needed = ['tp1', 'tp2', 'tp3', 'tp4']

        # Drop all unneeded columns
        p_tp_splits = p_tp_splits.reindex(tp_needed, axis='columns')

        # Aggregate each value
        tp_sum = p_tp_splits.values.sum()
        for tp_col in tp_needed:
            p_tp_splits[tp_col] = p_tp_splits[tp_col] / tp_sum

    tp_as = tp_as.lower()
    if tp_as == 'str' or tp_as == 'string':
        # Don't need to change anything
        pass
    elif tp_as == 'int' or tp_as == 'integer':
        p_tp_splits = p_tp_splits.rename(
            columns={
                'tp1': 1,
                'tp2': 2,
                'tp3': 3,
                'tp4': 4,
                'tp5': 5,
                'tp6': 6,
            }
        )
    else:
        raise ValueError("'%s' is not a valid value for tp_as.")

    return p_tp_splits


def get_zone_translation(import_dir: str,
                         from_zone: str,
                         to_zone: str,
                         return_dataframe: bool = False
                         ) -> Union[Dict[int, int], pd.DataFrame]:
    """
    Reads in the zone translation file from import_dir and converts it into a
    dictionary of from_zone: to_zone numbers

    Note: from_zone must be of a lower aggregation than to_zone, otherwise
    weird things might happen

    Parameters
    ----------
    import_dir:
        The directory to find the zone translation files

    from_zone:
        The name of the zoning system to convert from, e.g. noham

    to_zone
        The name of the zoning system to convert to, e.g. lad

    return_dataframe : bool, optional
        If True returns a DataFrame instead of a dictionary, by default
        False.

    Returns
    -------
    zone_translation:
        A dictionary (or DataFrame depending on `return_dataframe`) of
        from_zone values to to_zone values. Can be used to convert
        a zone number from one zoning system to another.
    """
    # Init
    base_filename = '%s_to_%s.csv'
    base_col_name = '%s_zone_id'

    # Load the file
    path = os.path.join(import_dir, base_filename % (from_zone, to_zone))
    translation = pd.read_csv(path)

    # Make sure we can find the columns
    from_col = base_col_name % from_zone
    to_col = base_col_name % to_zone

    if from_col not in translation.columns:
        raise ValueError("Found the file at '%s', but the columns do not "
                         "match. Cannot find from_zone column '%s'"
                         % (path, from_col))

    if to_col not in translation.columns:
        raise ValueError("Found the file at '%s', but the columns do not "
                         "match. Cannot find to_zone column '%s'"
                         % (path, to_col))

    # Make sure the columns are in the correct format
    translation = translation.reindex([from_col, to_col], axis='columns')
    translation[from_col] = translation[from_col].astype(int)
    translation[to_col] = translation[to_col].astype(int)
    if return_dataframe:
        return translation

    # Convert pandas to a {from_col: to_col} dictionary
    translation = dict(translation.itertuples(index=False, name=None))

    return translation


def defaultdict_to_regular(d):
    """
    Iteratively converts nested default dicts to nested regular dicts.

    Useful for pickling - keeps the unpickling of the dict simple

    Parameters
    ----------
    d:
        The nested defaultdict to convert

    Returns
    -------
    converted_d:
        nested dictionaries with same values
    """
    if isinstance(d, defaultdict):
        d = {k: defaultdict_to_regular(v) for k, v in d.items()}
    return d


def file_write_check(path: Union[str, Path], wait: bool=True) -> Path:
    """Attempts to write to given path to see if file is in use.

    Will either wait for the file to be closed or it will append numbers
    to the file name until and path can be found that isn't in use.

    Parameters
    ----------
    path : str
        Path to the file to check.

    wait : bool, optional
        Whether or not to wait for the file to be closed, by default True.
        If False appends number to the end of file name to find a path that
        isn't in use.

    Returns
    -------
    Path
        Path that isn't currently in use, will be the same as given `path`
        if wait is True.

    Raises
    ------
    ValueError
        When wait is False and a path can't be found, that isn't in use, in less
        than 100 attempts.
    """
    path = Path(path)
    new_path = path
    count = 1
    waiting = False
    while True:
        try:
            with open(new_path, 'wb') as f:
                pass
            return new_path
        except PermissionError:
            if wait:
                if not waiting:
                    print(f"Cannot write to file at {new_path.absolute()}.",
                          "Please ensure it is not open anywhere.",
                          "Waiting for permission to write...", sep='\n')
                    waiting = True
                time.sleep(1)
            else:
                new_path = path.parent / (path.stem + f'_{count}' + path.suffix)
                count += 1
                if count > 100:
                    raise ValueError('Too many files in use!')


def safe_dataframe_to_csv(df: pd.DataFrame,
                          out_path: str,
                          flatten_header: bool = False,
                          **to_csv_kwargs: Any,
                          ) -> None:
    """
    Wrapper around df.to_csv. Gives the user a chance to close the open file.

    Parameters
    ----------
    df:
        pandas.DataFrame to write to call to_csv on

    out_path:
        Where to write the file to. TO first argument to df.to_csv()

    flatten_header: bool, optional
        Whether or not MultiIndex column names should be flattened into a single level,
        default False.

    to_csv_kwargs:
        Any other kwargs to be passed straight to df.to_csv()

    Returns
    -------
        None
    """
    if flatten_header and len(df.columns.names) > 1:
        # Combine multiple columns levels into a single name split by ':'
        df.columns = [' : '.join(str(i) for i in c) for c in df.columns]

    written_to_file = False
    waiting = False
    while not written_to_file:
        try:
            df.to_csv(out_path, **to_csv_kwargs)
            written_to_file = True
        except PermissionError:
            if not waiting:
                print("Cannot write to file at %s.\n" % out_path +
                      "Please ensure it is not open anywhere.\n" +
                      "Waiting for permission to write...\n")
                waiting = True
            time.sleep(1)


def fit_filter(df: pd.DataFrame,
               df_filter: Dict[str, Any],
               raise_error: bool = False
               ) -> Dict[str, Any]:
    """
    Whittles down filter to only include relevant items

    Any columns that do not exits in the dataframe will be removed from the
    filter; optionally raises an error if a filter column is not in the given
    dataframe. Furthermore, any items that are 'none like' as determined by
    is_none_like() will also be removed.

    Parameters
    ----------
    df:
        The dataframe that the filter is to be applied to.

    df_filter:
        The filter dictionary in the format of {df_col_name: filter_values}

    raise_error:
        Whether to raise an error or not when a df_col_name does not exist in
        the given dataframe.

    Returns
    -------
    fitted_filter:
        A filter with non-relevant (as defined in the function description)
        items removed.
    """
    # Init
    fitted_filter = dict()
    df = df.copy()
    df.columns = df.columns.astype(str)

    # Check each item in the given filter
    for col, vals in df_filter.items():

        # Check the column exists
        if col not in df.columns:
            if raise_error:
                raise KeyError("'%s' Column not found in given dataframe"
                               % str(col))
            else:
                continue

        # Check the given value isn't None
        if is_none_like(vals):
            continue

        # Should only get here for valid combinations
        fitted_filter[col] = vals

    return fitted_filter


def remove_none_like_filter(df_filter: Dict[str, Any]) -> Dict[str, Any]:
    """
    Removes all None-like items from df_filter

    Parameters
    ----------
    df_filter:
        The filter dictionary in the format of {df_col_name: filter_values}

    Returns
    -------
    df_filter:
        df_filter with None-like items removed
    """
    # Init
    new_df_filter = dict()

    for k, v in df_filter.items():
        if not is_none_like(v):
            new_df_filter[k] = v

    return new_df_filter


def filter_by_segmentation(df: pd.DataFrame,
                           df_filter: Dict[str, Any],
                           fit: bool = False,
                           **kwargs
                           ) -> pd.DataFrame:
    """
    Filters a dataframe down to a given segmentation

    Can handle flexible segmentation if fit is set to True - all unnecessary
    columns will be removed, and any 'None like' filters will be removed. This
    follows the convention of settings segmentation splits to None when it
    is not needed.

    Parameters
    ----------
    df:
        The dataframe that the filter is to be applied to.

    df_filter:
        The filter dictionary in the format of {df_col_name: filter_values}.

    fit:
        Whether to try and fit the given filter to the dataframe before
        application. If using flexible segmentation and filter has not already
        been fit, set to True.

    kwargs:
        Any additional kwargs that should be passed to fit_filter() if fit is
        set to True.
    Returns
    -------
    filtered_df:
        The original dataframe given, segmented to the given filter level.
    """
    # Init
    df = df.copy()
    df_filter = df_filter.copy()

    # Wrap each item if a list to avoid errors
    for k, v in df_filter.items():
        if not pd.api.types.is_list_like(v):
            df_filter[k] = [v]

    # Ignore none-like filters to avoid dropping trips
    df_filter = remove_none_like_filter(df_filter)

    if fit:
        df_filter = fit_filter(df, df_filter.copy(), **kwargs)

    # Figure out the correct mask
    needed_cols = list(df_filter.keys())
    mask = df[needed_cols].isin(df_filter).all(axis='columns')

    return df[mask]


def intersection(l1: List[Any], l2: List[Any]) -> List[Any]:
    """
    Efficient method to return the intersection between l1 and l2
    """
    # Want to loop through the smaller list for efficiency
    if len(l1) > len(l2):
        big = l1.copy()
        small = l2
    else:
        big = l2
        small = l1

    # Get the intersection
    temp = set(big)
    return [x for x in small if x in temp]


def ensure_index(df: pd.DataFrame,
                 index: List[Any],
                 index_col: str,
                 infill: float = 0.0
                 ) -> pd.DataFrame:
    """
    Ensures every value in index exists in index_col of df.
    Missing values are infilled with infill
    """
    # Make a dataframe with just the new index
    ph = pd.DataFrame({index_col: index})

    # Merge with the given and infill missing
    return ph.merge(df, how='left', on=index_col).fillna(infill)


def ensure_multi_index(df: pd.DataFrame,
                       index_dict: Dict[str, List[Any]],
                       infill: float = 0.0
                       ) -> pd.DataFrame:
    """
    Ensures every combination of values in index_list exists in df.

    This function is useful to ensure a conversion from long to wide will
    happen correctly

    Parameters
    ----------
    df:
        The dataframe to alter.

    index_dict:
        A dictionary of {column_name: index_vals} to ensure exist in all
        combinations within df.

    infill:
        Value to infill any other columns of df where the given indexes
        don't exist.

    Returns
    -------
    df:
        The given df given with all combinations of the index dict values
    """
    # Create a new placeholder df with every combination of the unique columns
    all_combos = zip(*product(*index_dict.values()))
    ph = {col: vals for col, vals in zip(index_dict.keys(), all_combos)}
    ph = pd.DataFrame(ph)

    # Merge with the given and infill missing
    merge_cols = list(index_dict.keys())
    return ph.merge(df, how='left', on=merge_cols).fillna(infill)


def match_pa_zones(productions: pd.DataFrame,
                   attractions: pd.DataFrame,
                   unique_zones: List[Any],
                   zone_col: str = 'model_zone_id',
                   infill: float = 0.0,
                   set_index: bool = False
                   ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Makes sure all unique zones exist in productions and attractions

    Any missing zones will be infilled with infill,

    Parameters
    ----------
    productions:
        Dataframe containing the productions data - must have a zone_col

    attractions:
        Dataframe containing the productions data - must have a zone_col

    unique_zones:
        List of the desired zones to have in productions and attractions

    zone_col:
        Column in productions and attractions that contains the zone data.

    infill:
        Value to infill missing rows with in productions and attractions when
        new zone may be added in.

    set_index:
        Whether to set the zone_col as the index before returning or not.

    Returns
    -------
    (productions, attractions):
        The provided productions and attractions with all zones from
        unique_zones either as the index, or in zone_col
    """
    # Match productions and attractions
    productions = ensure_index(
        df=productions,
        index=unique_zones,
        index_col=zone_col,
        infill=infill
    )

    attractions = ensure_index(
        df=attractions,
        index=unique_zones,
        index_col=zone_col,
        infill=infill
    )

    if set_index:
        productions = productions.set_index(zone_col)
        attractions = attractions.set_index(zone_col)

    return productions, attractions


def balance_a_to_p(productions: pd.DataFrame,
                   attractions: pd.DataFrame,
                   unique_cols: List[str],
                   significant: int = 5
                   ) -> pd.DataFrame:
    """
    Balances the attractions total to productions total, keeping the same
    attraction distribution across segments

    Parameters
    ----------
    productions:
        Dataframe containing the productions data

    attractions:
        Dataframe containing the attractions data

    unique_cols:
        A list of the columns in productions and attractions to balance to
        one another

    significant:
        The number of significant places to check when ensuring the balancing
        has succeeded.

    Returns
    -------
    balanced_attractions

    """
    # Init
    attractions = attractions.copy()

    # Balance Attractions for each year
    for col in unique_cols:

        # Only balance if needed
        if productions[col].sum() == attractions[col].sum():
            continue

        if productions[col].sum() == 0:
            attractions[col] = 0
        else:
            attractions[col] /= attractions[col].sum() / productions[col].sum()

        # Throw an error if we somehow don't match
        np.testing.assert_approx_equal(
            productions[col].sum(),
            attractions[col].sum(),
            significant=significant,
            err_msg="After balancing Attraction to Productions the totals do"
                    "not match somehow. Might need to double check the "
                    "balancing function."
        )

    return attractions


def compile_efficient_df(eff_df: List[Dict[str, Any]],
                         col_names: List[Any]
                         ) -> pd.DataFrame:
    """
    Compiles an 'efficient df' and makes it a full dataframe.

    A efficient dataframe is a list of dictionaries, where each dictionary
    contains a df under the key 'df'. All other keys in this dictionary should
    be in the format {col_name, col_val}. All dataframes are expanded with
    the other columns from the dictionary then concatenated together

    Parameters
    ----------
    eff_df:
        Efficient df structure as described in the function description.

    col_names:
        The name and order of columns in the returned compiled_df

    Returns
    -------
    compiled_df:
        eff_df compiled into a full dataframe
    """
    # Init
    concat_ph = list()

    for part_df in eff_df:
        # Grab the dataframe
        df = part_df.pop('df')

        # Add all segmentation cols back into df
        for col_name, col_val in part_df.items():
            df[col_name] = col_val

        # Make sure all dfs are in the same format
        df = df.reindex(columns=col_names)
        concat_ph.append(df)

    return pd.concat(concat_ph).reset_index(drop=True)


def list_safe_remove(lst: List[Any],
                     remove: List[Any],
                     raise_error: bool = False,
                     inplace: bool = False
                     ) -> List[Any]:
    """
    Removes remove items from lst without raising an error

    Parameters
    ----------
    lst:
        The list to remove items from

    remove:
        The items to remove from lst

    raise_error:
        Whether to raise and error or not when an item is not contained in
        lst

    inplace:
        Whether to remove the items in-place, or return a copy of lst

    Returns
    -------
    lst:
        lst with removed items removed from it
    """
    # Init
    if not inplace:
        lst = lst.copy()

    for item in remove:
        try:
            lst.remove(item)
        except ValueError as e:
            if raise_error:
                raise e

    return lst


def is_almost_equal(v1: float,
                    v2: float,
                    significant: int = 7
                    ) -> bool:
    """
    Checks v1 and v2 are equal to significant places

    Parameters
    ----------
    v1:
        The first value to compare

    v2:
        The second value to compare

    significant:
        The number of significant bits to compare over

    Returns
    -------
    almost_equal:
        True if v1 and v2 are equal to significant bits, else False
    """
    return isclose(v1, v2, abs_tol=10 ** -significant)


def remove_all_commute_cat(df: pd.DataFrame,
                           emp_cat_col: str,
                           all_commute_name: str = 'E01'
                           ) -> pd.DataFrame:
    """
    Removes all_commute_name from emp_cat_col in df.

    df must be in long format, with a column for all employment categories

    Parameters
    ----------
    df:
        The dataframe to remove the all commute category from

    emp_cat_col:
        The name of the column in df that contains the employment category
        data

    all_commute_name:
        The name of the employment category that represents all commute data.
        This is the category that will be removed from df/

    Returns
    -------
    df:
        A copy of the original df with the all commute category removed.
    """
    # Validate input
    if emp_cat_col not in df:
        raise ValueError(
            "Cannot remove all commute category from df, as the emp_cat_col "
            "given does not exist. Given: %s" % str(emp_cat_col)
        )

    return df.loc[df[emp_cat_col] != all_commute_name]


def add_all_commute_cat(df: pd.DataFrame,
                        emp_cat_col: str,
                        unique_data_cols: List[str],
                        all_commute_name: str = 'E01'
                        ) -> pd.DataFrame:
    # Init
    df = df.copy()

    # Validate input
    if emp_cat_col not in df:
        raise ValueError(
            "Cannot add all commute category to df, as the emp_cat_col "
            "given does not exist. Given: %s" % str(emp_cat_col)
        )

    # Set up the group and index columns
    index_cols = list(df)
    index_cols.remove(emp_cat_col)
    group_cols = list_safe_remove(index_cols.copy(), unique_data_cols)

    # Calculate the totals for each zone
    all_commute = df.reindex(columns=index_cols)
    all_commute = all_commute.groupby(group_cols).sum().reset_index()
    all_commute[emp_cat_col] = all_commute_name

    # Append back to the starting dataframe and sort
    df = df.append(all_commute)
    df = df.sort_values(by=group_cols + ["employment_cat"])
    return df.reset_index(drop=True)


def create_iter_name(iter_num: Union[int, str]) -> str:
    return 'iter' + str(iter_num)


def convert_to_weights(df: pd.DataFrame,
                       year_cols: List[str],
                       weight_by_col: str = 'p'
                       ) -> pd.DataFrame:
    """
    TODO: write convert_to_weights() doc
    """
    df = df.copy()
    unq_vals = df[weight_by_col].unique()

    for val in unq_vals:
        mask = (df[weight_by_col] == val)
        for year in year_cols:
            df.loc[mask, year] = (
                df.loc[mask, year]
                /
                df.loc[mask, year].sum()
            )
    return df


def trip_origin_to_purposes(trip_origin: str) -> List[int]:
    """
    Returns a list of purposes for the given trip origin

    Parameters
    ----------
    trip_origin:
        The trip origin to get purposes for.

    Returns
    -------
    purposes:
        A list of integers representing purposes
    """
    # TODO Validate trip origin
    return efs_consts.TRIP_ORIGIN_TO_PURPOSE[trip_origin]


def merge_df_list(df_list, **kwargs):
    """
    Merge all dfs in df_list into a single dataframe

    Parameters
    ----------
    df_list:
        The list of dataframes to merge

    kwargs:
        ANy extra arguments to pass straight to pandas.merge()

    Returns
    -------
    merged_df:
        A single df of all items in df_list merged together
    """
    return functools.reduce(lambda l, r: pd.merge(l, r, **kwargs), df_list)


def split_base_future_years(years: List[int],
                            ) -> Tuple[int, List[int]]:
    """
    Splits years into base and future years.

    The smallest year in the list is assumed to be the base

    Parameters
    ----------
    years:
        A list of years to split

    Returns
    -------
    base_year:
        The base year from years

    future_years:
        A list of all other years than base_year in years.
        This will be returned in order, from lowest to highest.
    """
    # Validate inputs
    if not isinstance(years, list):
        raise TypeError(
            "Expecting a list of years but got %s instead."
            % str(type(years))
        )

    if not all([isinstance(x, int) for x in years]):
        raise TypeError(
            "Expecting a list of integers, but not all items are integers."
        )

    # Find the smallest value
    base_year = years.pop(years.index(min(years)))

    # Sort other items
    years.sort()

    return base_year, years


def split_base_future_years_str(years: List[str],
                                ) -> Tuple[str, List[str]]:
    """
    Splits years into base and future years.

    The smallest year in the list is assumed to be the base.
    A wrapper around split_base_future_years() to handle strings

    Parameters
    ----------
    years:
        A list of years to split

    Returns
    -------
    base_year:
        The base year from years

    future_years:
        A list of all other years than base_year in years.
        This will be returned in order, from lowest to highest.
    """
    base_year, future_years = split_base_future_years([int(x) for x in years])
    return str(base_year), [str(x) for x in future_years]


def get_norms_vdm_segment_aggregation_dict(norms_vdm_seg_name: str
                                           ) -> Dict[str, List[Any]]:
    """
    Returns a dictionary of valid segments for the given name

    Parameters
    ----------
    norms_vdm_seg_name:
        The name of the norms_vdm_matrix to get a dictionary for.
        Should be one of the values in efs_consts.NORMS_VDM_MATRIX_NAMES

    Returns
    -------
    segment_aggregation_dictionary:
        A dictionary of valid segments for norms_vdm_seg_name
    """
    if norms_vdm_seg_name in list(consts.NORMS_VDM_SEG_INTERNAL.keys()):
        return consts.NORMS_VDM_SEG_INTERNAL[norms_vdm_seg_name]
    elif norms_vdm_seg_name in list(consts.NORMS_VDM_SEG_EXTERNAL.keys()):
        return consts.NORMS_VDM_SEG_EXTERNAL[norms_vdm_seg_name]

    raise ValueError(
        "norms_vdm_seg_name does not seem to be a valid name. Expecting "
        "one of the values from efs_consts.NORMS_VDM_MATRIX_NAMES"
    )


def get_default_kwargs(func):
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }
