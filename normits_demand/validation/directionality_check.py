# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import os

import itertools
import random

from typing import List

from normits_demand import efs_constants as consts

# TODO: Directionality check w/ threshold
# TODO: Make numeric purpose work properly
# TODO: Get the zone number somewhere useful

# TODO: directionality_check returns report. Make audit based on report like...
# o_from = 'from_o'
# d_to = 'to_d'
# o_from_total = df[df['direction']==o_from]['demand'].sum()
# d_to_total = df[df['direction']==d_to]['demand'].sum()
# o_from_total - d_to_total


def directionality_check(import_dir: str,
                         model_name: str = None,
                         year: int = None,
                         trip_origin: str = 'hb',
                         m_needed: List[int] = None,
                         model_zone_col: str = 'o_zone',
                         n_zones: int = 1,
                         ) -> pd.DataFrame:
    """
    Folder = path
    purpose_type = 'numeric' or 'string'
    subset_zones = list
    purpose_subset = list
    """
    # Init
    time_periods = consts.TIME_PERIOD_STRS
    directions = ['from', 'to']
    model_name = consts.MODEL_NAME if model_name is None else model_name
    m_needed = consts.MODES_NEEDED if m_needed is None else m_needed

    # Randomly pick a zone and mode to test
    if trip_origin == 'hb':
        p_needed = [random.choice(consts.ALL_HB_P)]
    elif trip_origin == 'nhb':
        p_needed = [random.choice(consts.ALL_NHB_P)]
    else:
        raise ValueError(
            "%s is not a valid trip origin."
        )

    zone_count = consts.ZONE_SYSTEM_ZONE_COUNTS[model_name]
    subset_zones = [random.randint(1, zone_count) for _ in range(n_zones)]

    # Grab all csv in dir - assume csv==matrix
    all_mats = [x for x in os.listdir(import_dir) if '.csv' in x]

    # Loop through segmentation
    report_ph = list()
    seg_iterator = itertools.product(p_needed, m_needed, time_periods, directions)
    for p, m, tp, direction in seg_iterator:
        filtered_mats = all_mats.copy()

        # Filter by segmentations
        if year is not None:
            yr_str = '_yr%s_' % str(year)
            filtered_mats = [x for x in filtered_mats if yr_str in x]

        # purpose
        p_str = '_p%s_' % str(p)
        filtered_mats = [x for x in filtered_mats if p_str in x]

        # mode
        m_str = '_m%s_' % str(m)
        filtered_mats = [x for x in filtered_mats if m_str in x]

        # direction
        filtered_mats = [x for x in filtered_mats if direction in x]

        # time periods
        filtered_mats = [x for x in filtered_mats if tp in x]

        # Calculate totals for each given zone
        for zone in subset_zones:
            # Get the total of all the mats in this segmentation
            o_total = 0
            d_total = 0
            for mat_name in filtered_mats:
                mat = pd.read_csv(os.path.join(import_dir, mat_name))

                # Get the total O/D for this zone
                o_subset = mat[mat[model_zone_col] == zone]
                o_subset = o_subset.drop(columns=model_zone_col)
                o_total += o_subset.sum().sum()

                d_subset = mat[str(zone)]
                d_total += d_subset.sum()

            # Initialise the output dictionary
            base_zone_ph = {
                'p': p,
                'm': m,
                'tp': tp,
                'zone': zone,
                'direction': '%s_%s' % (direction, 'o')
            }
            if year is not None:
                base_zone_ph['yr'] = year

            # Add the calculated totals
            for zone_type, zone_total in zip(['o', 'd'], [o_total, d_total]):
                zone_ph = base_zone_ph.copy()
                zone_ph.update({
                    'direction': '%s_%s' % (direction, zone_type),
                    'demand': zone_total,
                })
                report_ph.append(zone_ph)

    return pd.DataFrame(report_ph)
