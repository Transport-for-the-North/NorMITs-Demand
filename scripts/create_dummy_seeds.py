# -*- coding: utf-8 -*-
"""
Created on: Tue December 8 14:28:33 2020
Updated on:

Original author: Ben Taylor
Last update made by:
Other updates made by:

File purpose:
Creates segmented dummy seed matrices. Useful for testing code - the output
matrices will not lead to accurate results!!

NOTE: A lot of related code in here from demand_utilities.utils - fix this
when demand is a package!
"""

import os
import shutil

from typing import Tuple
from typing import Union
from typing import Iterable
from typing import Iterator

SOC_P = [1, 2, 12]
NS_P = [3, 4, 5, 6, 7, 8, 13, 14, 15, 16, 18]

HB_P = [1, 2, 3, 4, 5, 6, 7, 8]
NHB_P = [12, 13, 14, 15, 16, 18]

INPUT_DIR = r'Y:\NorMITs Demand\import\norms\seed_distributions'
OUTPUT_DIR = r'Y:\NorMITs Demand\import\norms\seed_distributions'

TRIP_ORIGIN = 'nhb'
MATRIX_FORMAT = 'pa'

START_SEG = {
    'p_list': NHB_P,
    'm_list': [6],
}

END_SEG = START_SEG.copy()
END_SEG.update({
    'soc_list': [0, 1, 2, 3],
    'ns_list': [1, 2, 3, 4, 5],
    'ca_list': [1, 2]
})


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
    if not os.path.exists(src):
        raise IOError("Source file does not exist.\n %s" % src)

    if not os.path.isfile(src):
        raise ValueError("The given src file is not a file. Cannot handle "
                         "directories.")

    # Only rename if given a filename
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
        if purpose in SOC_P:
            required_segments = soc_list
        elif purpose in NS_P:
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
        seg_name = "soc" if purpose in [str(x) for x in SOC_P] else "ns"
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


def main():
    # Make sure start and end seg match
    for k, v in START_SEG.items():
        if k not in END_SEG:
            raise ValueError("Cannot find %s in END_SEG, but it is in "
                             "START_SEG" % k)

        if END_SEG[k] != v:
            raise ValueError("Key %s does not match in START_SEG and END_SEG. " % k)

    # Loop through all combinations
    for p, m, seg, ca in segmentation_loop_generator(**END_SEG):
        seg_str = 'soc' if p in SOC_P else 'ns'

        # Figure out the input mat name
        in_p = p if 'p_list' in START_SEG else None
        in_m = m if 'm_list' in START_SEG else None
        in_seg = seg if seg_str + '_list' in START_SEG else None
        in_ca = ca if 'ca_list' in START_SEG else None

        in_dist = get_dist_name(
            trip_origin=TRIP_ORIGIN,
            matrix_format=MATRIX_FORMAT,
            purpose=str(in_p),
            mode=str(in_m),
            segment=str(in_seg),
            car_availability=str(in_ca),
            csv=True
        )

        # Figure out the output mat name
        out_dist = get_dist_name(
            trip_origin=TRIP_ORIGIN,
            matrix_format='enhpa',
            purpose=str(p),
            mode=str(m),
            segment=str(seg),
            car_availability=str(ca),
            csv=True
        )

        # Copy over
        copy_and_rename(
            os.path.join(INPUT_DIR, in_dist),
            os.path.join(OUTPUT_DIR, out_dist),
        )


if __name__ == '__main__':
    main()
