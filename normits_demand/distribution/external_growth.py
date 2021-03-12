# -*- coding: utf-8 -*-
"""
Created on: Fri March 12 15:37:46 2021
Updated on:

Original author: Ben Taylor
Last update made by:
Other updates made by:

File purpose:
Handles the external distribution of trips. Avoid using a furness.
"""
# Builtins

from typing import List

# Third party
import pandas as pd

# Local imports
from normits_demand import efs_constants as efs_consts

from normits_demand.utils import general as du


def grow_external_pa(productions: pd.DataFrame,
                     seed_dist_dir: str,
                     dist_out: str,
                     external_zones: List[int],
                     years_needed: List[str],
                     p_needed: List[int],
                     m_needed: List[int],
                     soc_needed: List[int] = None,
                     ns_needed: List[int] = None,
                     ca_needed: List[int] = None,
                     tp_needed: List[int] = None,
                     zone_col: str = 'model_zone_id',
                     p_col: str = 'p',
                     m_col: str = 'm',
                     soc_col: str = 'soc',
                     ns_col: str = 'ns',
                     ca_col: str = 'ca',
                     tp_col: str = 'tp',
                     trip_origin: str = 'hb',
                     seed_mat_format: str = 'enhpa',
                     fname_suffix: str = None,
                     csv_out: bool = True,
                     compress_out: bool = True,
                     verbose: bool = False,
                     audit_out: str = None,
                     round_dp: int = efs_consts.DEFAULT_ROUNDING,
                     process_count: int = efs_consts.PROCESS_COUNT
                     ) -> None:
    # TODO: Write grow_external_pa() docs
    # Init
    productions = productions.copy()

    base_year, future_years = du.split_base_future_years_str(years_needed)
    group_cols = du.list_safe_remove(list(productions), years_needed)

    # Calculate the growth factors
    for year in future_years:
        productions[year] /= productions[base_year]
    productions.drop(columns=[base_year], inplace=True)

    print(productions)


    raise NotImplementedError
