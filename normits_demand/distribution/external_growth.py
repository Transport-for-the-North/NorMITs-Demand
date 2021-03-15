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
import os

from typing import List

# Third party
import numpy as np
import pandas as pd

# Local imports
import normits_demand as nd
from normits_demand import constants as consts
from normits_demand import efs_constants as efs_consts

from normits_demand.utils import general as du
from normits_demand.utils import file_ops

from normits_demand.concurrency import multiprocessing


def _grow_external_pa_internal(growth_factors,
                               base_year,
                               import_dir,
                               export_dir,
                               trip_origin,
                               matrix_format,
                               seg_params,
                               zone_col,
                               p_col,
                               m_col,
                               seg_col,
                               ca_col,
                               tp_col,
                               round_dp,
                               fname_suffix,
                               csv_out,
                               compress_out,
                               ):
    """
    The internal function of grow_external_pa, used for multiprocessing
    """
    # Init
    growth_factors = growth_factors.copy()
    audit = dict()

    # Print some progress
    out_dist_name = du.calib_params_to_dist_name(
        trip_origin=trip_origin,
        matrix_format='pa',
        calib_params=seg_params,
        suffix=fname_suffix,
        csv=csv_out,
        compressed=compress_out,
    )
    print("Furnessing %s ..." % out_dist_name)

    # ## READ IN THE BASE YEAR EXTERNAL DEMAND ## #
    # Build seg_params for the seed values
    seed_seg_params = seg_params.copy()
    seed_seg_params['yr'] = base_year

    # Read in the seed distribution
    by_fname = du.calib_params_to_dist_name(
        trip_origin=trip_origin,
        matrix_format=matrix_format,
        calib_params=seed_seg_params,
        suffix=fname_suffix,
        csv=csv_out,
        compressed=compress_out,
    )
    path = file_ops.find_filename(os.path.join(import_dir, by_fname))
    by_mat = file_ops.read_df(path, index_col=0)
    by_mat.columns = by_mat.columns.astype(int)

    # Quick check that seed is valid
    if len(by_mat.index.difference(by_mat.index)) > 0:
        raise ValueError(
            "The index and columns of the base year distribution '%s' "
            "do not match." % by_mat
        )

    base_year_shape = (len(by_mat.index), len(by_mat.index))

    # ## FILTER GROWTH FACTORS TO SEGMENTATION ## #
    if seg_params.get('soc') is not None:
        seg = seg_params.get('soc')
    else:
        seg = seg_params.get('ns')

    base_filter = {
        p_col: seg_params.get('p'),
        m_col: seg_params.get('m'),
        seg_col: str(seg),
        ca_col: seg_params.get('ca'),
        tp_col: seg_params.get('tp')
    }

    growth_factors = du.filter_by_segmentation(
        growth_factors,
        df_filter=base_filter,
        fit=True,
    )

    # ## GROW THE BASE YEAR MATRIX TO FUTURE YEAR ## #
    # Make sure all the rows in the base matrix are in the growth factors
    pre_reindex_length = len(growth_factors)
    growth_factors = growth_factors.set_index(zone_col)
    growth_factors = growth_factors.reindex(by_mat.index).fillna(1)

    # Record the number of missing growth factors infilled with 1
    audit['name'] = out_dist_name
    audit['infilled_gf'] = len(growth_factors) - pre_reindex_length

    # Broadcast the growth factors to same shape as base year mat
    year = str(seg_params['yr'])
    growth_factors = np.broadcast_to(
        np.expand_dims(growth_factors[year].values, 1),
        base_year_shape
    )

    # Multiply to get the future year matrix
    fy_mat = by_mat * growth_factors
    fy_mat = fy_mat.round(round_dp)

    # Write out the new matrix
    out_path = os.path.join(export_dir, out_dist_name)
    file_ops.write_df(fy_mat, out_path)

    return audit


def grow_external_pa(growth_factors: pd.DataFrame,
                     import_dir: nd.PathLike,
                     export_dir: nd.PathLike,
                     base_year: str,
                     future_years: List[str],
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
                     matrix_format: str = 'pa',
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
    growth_factors = growth_factors.copy()
    soc_needed = [None] if soc_needed is None else soc_needed
    ns_needed = [None] if ns_needed is None else ns_needed
    ca_needed = [None] if ca_needed is None else ca_needed
    tp_needed = [None] if tp_needed is None else tp_needed

    group_cols = du.list_safe_remove(list(growth_factors), future_years)

    # Make sure the soc and ns columns are strings
    if 'soc' in list(growth_factors):
        growth_factors['soc'] = growth_factors['soc'].astype(str)
    if 'ns' in list(growth_factors):
        growth_factors['ns'] = growth_factors['ns'].astype(str)

    # Make sure the segmentations we're asking for exist in Productions
    growth_factors = du.ensure_segmentation(
        df=growth_factors,
        p_needed=p_needed,
        m_needed=m_needed,
        soc_needed=soc_needed,
        ns_needed=ns_needed,
        ca_needed=ca_needed,
        tp_needed=tp_needed,
        p_col=p_col,
        m_col=m_col,
        soc_col=soc_col,
        ns_col=ns_col,
        ca_col=ca_col,
        tp_col=tp_col,
    )

    # ## GROW THE SEED BY PRODUCTION GROWTH FOR FUTURE YEARS ## #
    for year in future_years:
        # Filter growth factors
        gf_index = group_cols.copy() + [year]
        yr_gf = growth_factors.reindex(columns=gf_index)

        # Loop through segmentations for this year
        loop_generator = du.cp_segmentation_loop_generator(
            p_list=p_needed,
            m_list=m_needed,
            soc_list=soc_needed,
            ns_list=ns_needed,
            ca_list=ca_needed,
            tp_list=tp_needed,
        )

        # ## MULTIPROCESS ## #
        unchanging_kwargs = {
            'growth_factors': yr_gf,
            'base_year': base_year,
            'import_dir': import_dir,
            'export_dir': export_dir,
            'trip_origin': trip_origin,
            'zone_col': zone_col,
            'p_col': p_col,
            'm_col': m_col,
            'ca_col': ca_col,
            'tp_col': tp_col,
            'matrix_format': matrix_format,
            'round_dp': round_dp,
            'fname_suffix': fname_suffix,
            'csv_out': csv_out,
            'compress_out': compress_out,
        }

        # Build a list of all kw arguments
        kwargs_list = list()
        for seg_params in loop_generator:
            # Set the column name of the ns/soc column
            if seg_params['p'] in efs_consts.SOC_P:
                seg_col = soc_col
            elif seg_params['p'] in efs_consts.NS_P:
                seg_col = ns_col
            else:
                raise ValueError("'%s' does not seem to be a valid soc or ns "
                                 "purpose." % str(seg_params['p']))

            # Add in year
            seg_params['yr'] = int(year)

            kwargs = unchanging_kwargs.copy()
            kwargs.update({
                'seg_params': seg_params,
                'seg_col': seg_col,
            })
            kwargs_list.append(kwargs)

        audits = multiprocessing.multiprocess(
            fn=_grow_external_pa_internal,
            kwargs=kwargs_list,
            process_count=process_count,
        )

        # ## WRITE OUT THE AUDITS ## #
        fname = "%s_%s_external_growth_summary.csv" % (trip_origin, year)
        audit_path = os.path.join(audit_out, fname)
        pd.DataFrame(audits).to_csv(audit_path, index=False)
