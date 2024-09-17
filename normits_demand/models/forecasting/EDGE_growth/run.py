# -*- coding: utf-8 -*-
"""
Created on: 17/05/2023
Updated on:

Original author: Ben Taylor
Last update made by:
Other updates made by:

File purpose:

"""
# Built-Ins
import pathlib
from functools import partial
import logging
import os

# Third Party
import pandas as pd
import numpy as np
from tqdm import tqdm
from caf.toolkit import pandas_utils
from caf.toolkit.concurrency import multiprocess

# Local Imports
# pylint: disable=import-error,wrong-import-position
# Local imports here
from normits_demand.models.forecasting.edge_growth import growth_matrices
from normits_demand.models.forecasting.edge_growth import apply_growth
from normits_demand.models.forecasting.edge_growth import utils
from normits_demand.models.forecasting.edge_growth import loading
from normits_demand.utils import timing
from normits_demand.matrices import omx_file
from normits_demand.utils import file_ops
from normits_demand.models.forecasting import forecast_cnfg

# pylint: enable=import-error,wrong-import-position

# # # CONSTANTS # # #
logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
# # # CLASSES # # #

# # # FUNCTIONS # # #
def _tp_loop(
    matrices_to_grow_dir: pathlib.Path,
    probs_dir: pathlib.Path,
    model_stations_tlcs_len: int,
    forecast_year: int,
    demand_segments: pd.DataFrame,
    filled_growth_matrices,
    time_period,
    splitting_matrices,
):
    """
    Private function only called once for multiprocessing. All args are as named when read in.

    Parameters
    ----------
    matrices_to_grow_dir: Directory containing matrices for factors to be applied to.
    probs_dir: directory containing irsj prorbabilites.
    model_stations_tlcs_len: Passed to zonal_from_to_stations_demand
    forecast_year: Year of the forecast
    demand_segments: See documentation of GlobalVars
    filled_growth_matrices: Return from 'fill_missing_factors'
    time_period: The time period for this loop
    splitting_matrices: Ticket type splits, from GlobalVars
    """
    factored_matrices = {}
    growth_summary = pd.DataFrame(
            {
                "Time_Period": [],
                "Demand_Segment": [],
                "Base_Demand": [],
                f"{forecast_year}_Demand": [],
            }
        )
    overall_not_grown_demand = 0
    overall_base_demand = 0
    LOG.info(
        "-- Processing Time Period %s @ %s",
        time_period,
        timing.get_datetime(),
    )
    # read time period specific files
    LOG.info(
        f"{'Time_Period':>12}{'Demand_Segment':>15}"
        f"{'Base_Demand':>12}{f'{forecast_year}_Demand':>12}"
    )

    with omx_file.OMXFile(
        pathlib.Path(
            matrices_to_grow_dir,
            f"PT_{time_period}.omx",
        )
    ) as omx_mat:
        for row in tqdm(
            demand_segments.itertuples(),
            desc="    Demand Segments Loop ",
            unit=" Segment",
            colour="cyan",
            total=len(demand_segments),
        ):
            # store current segment's details
            segment = row.Segment
            to_home = row.ToHome
            growth_method = row.Growth_Method
            userclass = row.Userclass
            purpose = row.Purpose

            # read demand matrix
            zonal_base_demand_mx = utils.wide_to_long_np(omx_mat.get_matrix_level(segment))

            # If demand is 0, skip the work, keep as is. Can't grow anyway
            if zonal_base_demand_mx["Demand"].sum() == 0:
                cols = zonal_base_demand_mx.columns
                factored_matrices[segment] = pandas_utils.long_to_wide_infill(
                    zonal_base_demand_mx,
                    cols[0],
                    cols[1],
                    cols[2]
                ).values
                LOG.info(
                    f"{time_period:>12}{segment:>15}" f"{0:>12}{0:>12}"
                )
                continue

            # reduce probabilities to current userclass
            hdf_file = probs_dir / f"{time_period}_iRSj_probabilities_split.h5"
            irsj_probs_segment = pd.read_hdf(hdf_file, key=f'userclass_{userclass}')

            # convert matrix to numpy stn2stn and produce a conversion lookup
            if to_home:
                zonal_base_demand_mx = utils.transpose_matrix(zonal_base_demand_mx)
            (
                np_stn2stn_base_demand_mx,
                zonal_from_to_stns,
            ) = utils.zonal_from_to_stations_demand(
                zonal_base_demand_mx,
                irsj_probs_segment,
                model_stations_tlcs_len,
                userclass,
                time_period
            )
            # store matrix total demand
            tot_input_demand = round(np_stn2stn_base_demand_mx.sum())
            # add to overall base demand
            overall_base_demand += tot_input_demand
            # apply growth
            np_stn2stn_grown_demand_mx = (
                apply_growth.apply_demand_growth(
                    np_stn2stn_base_demand_mx,
                    splitting_matrices[purpose],
                    filled_growth_matrices[purpose],
                    len(np_stn2stn_base_demand_mx),
                    growth_method,
                    to_home,
                )
            )

            # Log movements with no growth applied
            mask = (np_stn2stn_base_demand_mx == np_stn2stn_grown_demand_mx)
            no_growth_movements = np_stn2stn_base_demand_mx[mask]
            overall_not_grown_demand += no_growth_movements.sum()

            # Log the iteration into the growth summary
            tot_output_demand = round(np_stn2stn_grown_demand_mx.sum())
            segment_growth_summary = pd.DataFrame(
                {
                    "Time_Period": [time_period],
                    "Demand_Segment": [segment],
                    "Base_Demand": [tot_input_demand],
                    f"{forecast_year}_Demand": [tot_output_demand],
                }
            )
            growth_summary = pd.concat(
                [growth_summary, segment_growth_summary], axis=0
            )
            # convert back to zonal level demand
            zonal_grown_demand_mx = (
                utils.convert_stns_to_zonal_demand(
                    np_stn2stn_grown_demand_mx,
                    zonal_from_to_stns,
                    time_period,
                    to_home,
                )
            )
            # add to grown matrices dictionary
            factored_matrices[segment] = zonal_grown_demand_mx

    return loading.TPLoopOutput(overall_not_grown_demand, overall_base_demand, factored_matrices, growth_summary)


def run_edge_growth(params: forecast_cnfg.EDGEParameters) -> None:
    """Run Growth Process."""
    LOG.info("#" * 80)
    LOG.info(
        "Started Process @ %s",
        timing.get_datetime("%d-%m-%Y  %H:%M:%S"),
    )
    LOG.info("#" * 80)
    LOG.info("Loading global variables for growth process.")
    global_params = loading.load_globals(params)
    # loop over forecast years
    for forecast_year in params.forecast_years:
        LOG.info(
            "**** Applying growth for %s @ %s",
            forecast_year,
            timing.get_datetime(),
        )
        # read input files
        growth_factors = file_ops.read_df(
            params.edge_growth_dir
            / params.forecast_years[forecast_year]
        )
        # produce growth matrices
        growth_matrix_dic = growth_matrices.prepare_growth_matrices(
            global_params.demand_segments,
            growth_factors,
            global_params.station_tlcs,
        )
        # fill growth matrices
        filled_growth_matrices = growth_matrices.fill_missing_factors(global_params.purposes, growth_matrix_dic)
        # create empty dictionary to store matrices
        factored_24hr_matrices = {}
        # declare global demand total variables
        # empty dataframe for growth summary

        # loop over time periods
        for tp in global_params.time_periods:
            hdf_file = params.irsj_props_dir / f"{tp}_iRSj_probabilities_split.h5"
            if not os.path.isfile(hdf_file):
                utils.split_irsj(params.irsj_props_dir, 'userclass', tp)
        tp_looper = partial(
            _tp_loop,
            params.matrices_to_grow_dir,
            params.irsj_props_dir,
            len(global_params.station_tlcs),
            forecast_year,
            global_params.demand_segments,
            filled_growth_matrices,
        )
        args = [
            (time_period, global_params.ticket_type_splits[time_period])
            for time_period in global_params.time_periods
        ]
        # test = tp_looper(*args[0])
        outputs = multiprocess(
            tp_looper, args, in_order=True
        )
        factored = [k.factored for k in outputs]
        factored_matrices = dict(zip(global_params.time_periods, factored))
        overall_not_grown_demand = sum([i.overall_not_grown for i in outputs])
        overall_base_demand = sum([i.overall_base for i in outputs])

        # calculate proportion of not grown demand
        growth_summary = pd.concat([i.summary for i in outputs])
        not_grown_demand_pcent = (
            overall_not_grown_demand / overall_base_demand
        )
        # if proportion is greater than 1%, terminate the program
        if not_grown_demand_pcent > 0.01:
            raise ValueError(f" Percentage of the demand not being grown {round(not_grown_demand_pcent * 100,3)}% is > 1%.\n"
                "User must review growth factors used.")
        LOG.warning(
            f"Percentage of the demand not being grown {round(not_grown_demand_pcent * 100,3)}%."
        )
        # prepare 24Hr level demand matrices
        for row in global_params.demand_segments.itertuples():
            # declare current segment's details
            segment = row.Segment
            # get 24Hr demand matrix
            forecast_matrix_24hr = (
                factored_matrices["AM"][segment]
                + factored_matrices["IP"][segment]
                + factored_matrices["PM"][segment]
                + factored_matrices["OP"][segment]
            )
            # add to 24Hr matrices dict
            factored_24hr_matrices[segment] = forecast_matrix_24hr

        # Combine matrices into NoRMS segments
        norms_matrices = apply_growth.fromto_2_from_by_averaging(
            factored_24hr_matrices,
            global_params.norms_segments,
            global_params.all_segments,
        )

        # export files
        for segment in global_params.norms_segments:
            # write out demand matrix
            file_ops.write_df(
                utils.wide_to_long_np(
                    norms_matrices[segment]
                ).sort_values(
                    by=["from_model_zone_id", "to_model_zone_id"]
                ),
                params.export_path
                / f"{forecast_year}_24Hr_{segment}.csv",
                index=False,
            )
        # convert to Cube .MAT
        utils.convert_csv_2_mat(
            global_params.norms_segments,
            params.cube_exe,
            forecast_year,
            params.export_path,
        )
        # export growth summary
        file_ops.write_df(
            growth_summary,
            params.export_path / f"{forecast_year}_Growth_Summary.csv",
            index=False,
        )
