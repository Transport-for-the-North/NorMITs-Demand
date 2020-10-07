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

import pandas as pd

from typing import List
from typing import Dict
from typing import Iterable

import efs_constants as consts

# Can call tms pa_to_od.py functions from here
from old_tms.utils import *


def copy_and_rename(src: str, dst: str) -> None:
    """
    Makes a copy of the src file and saves it at dst with the new filename.

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
    if not os.path.isfile(src):
        raise ValueError("The given src file is not a file. Cannot handle "
                         "directories.")

    # Only rename if given a filename
    if '.' not in os.path.basename(dst):
        # Copy over with same filename
        shutil.copy(src, dst)
    else:
        # Split paths
        _, src_tail = os.path.split(src)
        dst_head, dst_tail = os.path.split(dst)

        # Copy then rename
        shutil.copy(src, dst_head)
        shutil.move(os.path.join(dst_head, src_tail), dst)


def get_model_name(mode: int) -> str:
    """
    Returns a string of the TfN model name based on the mode given.

    Parameters
    ----------
    mode:
        Mode of transport

    Returns
    -------
    model_name:
        model name string
    """
    mode_to_name = {
        1: None,
        2: None,
        3: 'noham',
        4: None,
        5: None,
        6: 'norms'
    }

    if mode not in mode_to_name:
        raise ValueError("'%s' is not a valid mode." % str(mode))

    if mode_to_name[mode] is None:
        raise ValueError("'%s' is a valid mode, but a model name does not "
                         "exist for it." % str(mode))

    return mode_to_name[mode]


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
                  **kwargs
                  ) -> pd.DataFrame:
    """
    Reads in the file and performs some simple file checks

    Parameters
    ----------
    file_path:
        Path to the file to read in

    kwargs:
        ANy kwargs to pass onto pandas.read_csv()

    Returns
    -------
    dataframe:
        The data from file_path
    """
    # TODO: Add any more error checks here
    # Check file exists
    if not os.path.exists(file_path):
        raise IOError("No file exists at %s" % file_path)

    return pd.read_csv(file_path, **kwargs)


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
        return is_none_like(o[0])

    return False


def get_data_subset(orig_data: pd.DataFrame,
                    split_col_name: str = 'model_zone_id',
                    subset_vals: List[object] = consts.DEFAULT_ZONE_SUBSET
                    ) -> pd.DataFrame:
    """
    Returns a subset of the original data - useful for testing and dev

    Parameters
    ----------
    orig_data:
        The pandas DataFrame containing the starting data

    split_col_name:
        The column of orig_data we will look for subset_vals in

    subset_vals:
        The values to look for and keep in split_col_data

    Returns
    -------
    subset_data:
        A smaller version of orig_data

    """
    subset_mask = orig_data[split_col_name].isin(subset_vals)
    return orig_data.loc[subset_mask]


def get_dist_name(trip_origin: str,
                  matrix_format: str,
                  year: str = None,
                  purpose: str = None,
                  mode: str = None,
                  segment: str = None,
                  car_availability: str = None,
                  tp: str = None,
                  csv: bool = False,
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
        seg_name = "soc" if purpose in ['1', '2'] else "ns"
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

    return final_name


def calib_params_to_dist_name(trip_origin: str,
                              matrix_format: str,
                              calib_params: dict,
                              csv: bool = False
                              ) -> str:
    """
        Wrapper for get_distribution_name() using calib params
    """
    segment_str = 'soc' if calib_params['p'] in [1, 2] else 'ns'

    if 'tp' in calib_params:
        return get_dist_name(
            trip_origin,
            matrix_format,
            str(calib_params.get('yr')),
            str(calib_params.get('p')),
            str(calib_params.get('m')),
            str(calib_params.get(segment_str)),
            str(calib_params.get('ca')),
            tp=str(calib_params.get('tp')),
            csv=csv
        )
    else:
        return get_dist_name(
            trip_origin,
            matrix_format,
            str(calib_params.get('yr')),
            str(calib_params.get('p')),
            str(calib_params.get('m')),
            str(calib_params.get(segment_str)),
            str(calib_params.get('ca')),
            csv=csv
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


def generate_calib_params(year: str,
                          purpose: int,
                          mode: int,
                          segment: int,
                          ca: int
                          ) -> dict:
    """
    Returns a TMS style calib_params dict
    """
    segment_str = 'soc' if purpose in [1, 2] else 'ns'
    return {
        'yr': year,
        'p': purpose,
        'm': mode,
        segment_str: segment,
        'ca': ca
    }


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

    # Mode. What is the code for rail?
    if re.search('_Hwy', fname) is not None:
        calib_params['m'] = 3
    else:
        Warning("Cannot find a mode in filename. It might be rail, but I "
                "don't know what to search for at the moment.\n"
                "File name: '%s'" % fname)

    # tp
    loc = re.search('_TS[0-9]+', fname)
    if loc is not None:
        calib_params['tp'] = int(fname[loc.start() + 3:loc.end()])

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
                          ) -> Dict[str, str]:
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
                        year_col: str = 'year',
                        purpose_col: str = 'purpose_id',
                        mode_col: str = 'mode_id',
                        soc_col: str = 'soc_id',
                        ns_col: str = 'ns_id',
                        ca_col: str = 'car_availability_id',
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
        if purpose in [1, 2]:
            dist[soc_col] = segment
            dist[ns_col] = 'none'
        else:
            dist[soc_col] = 'none'
            dist[ns_col] = segment

    return dist


def segmentation_loop_generator(p_list: Iterable[int],
                                m_list: Iterable[int],
                                soc_list: Iterable[int],
                                ns_list: Iterable[int],
                                ca_list: Iterable[int],
                                tp_list: Iterable[int] = None):
    """
    Simple generator to avoid the need for so many nested loops
    """
    for purpose in p_list:
        if purpose in consts.SOC_P:
            required_segments = soc_list
        elif purpose in consts.NS_P:
            required_segments = ns_list
        elif purpose in consts.ALL_NHB_P:
            required_segments = [None]
        else:
            raise ValueError("'%s' does not seem to be a valid soc, ns, or "
                             "nhb purpose." % str(purpose))
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


def long_to_wide_out(df: pd.DataFrame,
                     v_heading: str,
                     h_heading: str,
                     values: str,
                     out_path: str,
                     echo=False,
                     unq_zones: List[str] = None
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

    echo:
        Indicates whether to print a log of the process to the terminal.

    unq_zones:
        A list of all the zone names that should exist in the output matrix.
        If zones in this list are not in the given df, they are infilled with
        values of 0.
        If left as None, it assumes all zones in the range 1 to max zone number
        should exist.

    Returns
    -------
        None
    """
    # Get the unique column names
    if unq_zones is None:
        unq_zones = df[v_heading].drop_duplicates().reset_index(drop=True).copy()
        unq_zones = list(range(1, max(unq_zones)+1))

    # Convert to wide format and write to file
    wide_mat = df_to_np(
        df=df,
        values=values,
        unq_internal_zones=unq_zones,
        v_heading=v_heading,
        h_heading=h_heading,
        echo=echo
    )
    pd.DataFrame(
        wide_mat,
        index=unq_zones,
        columns=unq_zones
    ).to_csv(out_path)


def get_compile_params_name(matrix_format: str, year: str) -> str:
    """
    Generates the compile params filename
    """
    return "%s_yr%s_compile_params.csv" % (matrix_format, year)


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
    Returns True if any of vals is on string, else False
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
                             csv=False
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

    # Optionally add on the csv if needed
    if csv:
        final_name += '.csv'

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
