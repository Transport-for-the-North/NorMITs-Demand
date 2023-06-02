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
import sys
import pickle

sys.path.append(r"E:\NorMITs-Demand")
# Third Party
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
from caf.toolkit import concurrency
from functools import partial

# Local Imports
# pylint: disable=import-error,wrong-import-position
# Local imports here
import growth_matrices
import apply_growth
import utils
import loading
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
    factored_matrices,
    matrices_to_grow_dir,
    model_stations_tlcs,
    forecast_year,
    demand_segments,
    filled_growth_matrices,
    growth_summary,
    time_period,
    splitting_matrices,
):
    """
    Private function only called once for multiprocessing. All args are as named when read in.
    """
    overall_not_grown_demand = 0
    overall_base_demand = 0
    LOG.info(
        "-- Processing Time Period %s @ %s",
        time_period,
        timing.get_datetime(),
    )
    # read time period specific files
    irsj_props = pd.read_hdf(
        matrices_to_grow_dir / f"{time_period}_iRSj_probabilities.h5",
        key="iRSj",
    )
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
            zonal_base_demand_mx = utils.wide_mx_2_long_mx(
                omx_mat.get_matrix_level(segment),
                rows="from_model_zone_id",
                cols="to_model_zone_id",
            )
            # check if matrix has no demand then continue
            if zonal_base_demand_mx["Demand"].sum() == 0:
                # keep matrix as it is, i.e. = 0
                factored_matrices[segment] = utils.long_mx_2_wide_mx(
                    zonal_base_demand_mx
                )
                LOG.info(
                    f"{time_period:>12}{segment:>15}" f"{0:>12}{0:>12}"
                )
                continue
            # reduce probabilities to current userclass
            irsj_probs_segment = irsj_props.loc[
                irsj_props["userclass"] == userclass
            ].reset_index(drop=True)
            # convert matrix to numpy stn2stn and produce a conversion lookup
            (
                np_stn2stn_base_demand_mx,
                zonal_from_to_stns,
            ) = utils.zonal_from_to_stations_demand(
                zonal_base_demand_mx,
                irsj_probs_segment,
                len(model_stations_tlcs),
                userclass,
                to_home,
            )
            # store matrix total demand
            tot_input_demand = round(np_stn2stn_base_demand_mx.sum())
            # add to overall base demand
            overall_base_demand = overall_base_demand + tot_input_demand
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
            # get movements where no growth has been applied
            no_growth_movements = np_stn2stn_base_demand_mx[
                np_stn2stn_base_demand_mx == np_stn2stn_grown_demand_mx
            ]
            # Add to the total not grown demand
            overall_not_grown_demand = (
                overall_not_grown_demand + no_growth_movements.sum()
            )
            # store matrix total demand
            tot_output_demand = round(np_stn2stn_grown_demand_mx.sum())
            LOG.info(
                f"{time_period:>12}{segment:>15}"
                f"{tot_input_demand:>12}{tot_output_demand:>12}"
            )
            # append to growth summary df
            # empty dataframe for growth summary
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
                apply_growth.convert_stns_to_zonal_demand(
                    np_stn2stn_grown_demand_mx,
                    zonal_from_to_stns,
                    time_period,
                    to_home,
                )
            )
            # add to grown matrices dictionary
            factored_matrices[segment] = zonal_grown_demand_mx

    outputs = {}
    outputs["overall_not_grown"] = overall_not_grown_demand
    outputs["overall_base"] = overall_base_demand
    outputs["factored"] = factored_matrices
    outputs["summary"] = growth_summary

    return outputs


def run_edge_growth(params: forecast_cnfg.EDGEParameters) -> None:
    """Run Growth Process."""
    LOG.info("#" * 80)
    LOG.info(
        "Started Process @ %s",
        timing.get_datetime("%d-%m-%Y  %H:%M:%S"),
    )
    LOG.info("#" * 80)
    LOG.info("Loading globals variables for growth process.")
    globals = loading.load_globals(params)
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
            globals.demand_segments,
            growth_factors,
            globals.station_tlcs,
        )
        # fill growth matrices
        (
            filled_growth_matrices,
            missing_factors,
        ) = growth_matrices.fill_missing_factors(
            globals.purposes,
            growth_matrix_dic,
        )
        for i, j in missing_factors.items():
            j.to_csv(
                params.export_folder
                / f"missing_factors_{i}_{forecast_year}.csv"
            )
        # create empty dictionary to store matrices
        factored_matrices = {}
        factored_24hr_matrices = {}
        # declare global demand total variables
        # empty dataframe for growth summary
        growth_summary = pd.DataFrame(
            {
                "Time_Period": [],
                "Demand_Segment": [],
                "Base_Demand": [],
                f"{forecast_year}_Demand": [],
            }
        )
        with open(
            r"E:\NorMITs Demand\Forecasting\edge\1.0\iter1.0\Test\refactor_growth.pkl",
            "wb",
        ) as file:
            pickle.dump(filled_growth_matrices, file)

        # loop over time periods
        tp_looper = partial(
            _tp_loop,
            factored_matrices.copy(),
            params.matrices_to_grow_dir,
            globals.station_tlcs,
            forecast_year,
            globals.demand_segments.copy(),
            filled_growth_matrices.copy(),
            growth_summary,
        )
        args = [
            (time_period, globals.ticket_type_splits[time_period])
            for time_period in globals.time_periods
        ]
        outputs = concurrency.multiprocess(
            tp_looper, args, in_order=True
        )
        factored = [k["factored"] for k in outputs]
        factored_matrices = {
            i: j for i, j in zip(globals.time_periods, factored)
        }
        overall_not_grown_demand = sum(
            [i["overall_not_grown"] for i in outputs]
        )
        overall_base_demand = sum([i["overall_base"] for i in outputs])

        # calculate proportion of not grown demand
        growth_summary = pd.concat([i["summary"] for i in outputs])
        not_grown_demand_pcent = (
            overall_not_grown_demand / overall_base_demand
        )
        # if proportion is greater than 1%, terminate the program
        if not_grown_demand_pcent > 0.01:
            LOG.critical(
                f" Percentage of the demand not being grown {round(not_grown_demand_pcent * 100,3)}% is > 1%.\n"
                "User must review growth factors used."
            )
            sys.exit()
        else:
            LOG.warning(
                f" Percentage of the demand not being grown {round(not_grown_demand_pcent * 100,3)}%."
            )
        # prepare 24Hr level demand matrices
        for row in globals.demand_segments.itertuples():
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
            globals.norms_segments,
            globals.all_segments,
        )

        # export files
        for segment in globals.norms_segments:
            # write out demand matrix
            file_ops.write_df(
                utils.wide_mx_2_long_mx(
                    norms_matrices[segment],
                    rows="from_model_zone_id",
                    cols="to_model_zone_id",
                ).sort_values(
                    by=["from_model_zone_id", "to_model_zone_id"]
                ),
                params.export_path
                / f"{forecast_year}_24Hr_{segment}.csv",
                index=False,
            )
        # convert to Cube .MAT
        utils.convert_csv_2_mat(
            globals.norms_segments,
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
