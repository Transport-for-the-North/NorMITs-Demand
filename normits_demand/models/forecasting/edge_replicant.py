# -*- coding: utf-8 -*-
"""EDGE Replicant process to grow demand."""
# ## IMPORTS ## #
# Standard imports
import sys
import logging
import itertools
import pathlib

# Third party imports
from typing import Tuple
import pandas as pd
import numpy as np
from tqdm import tqdm

# Local imports
from normits_demand.utils import timing
from normits_demand.matrices import omx_file
from normits_demand.utils import file_ops
from normits_demand.models.forecasting import forecast_cnfg
from normits_demand.matrices.cube_mat_converter import CUBEMatConverter

# ## CONSTANTS ## #
LOG = logging.getLogger(__name__)

# ## CLASSES ## #

# ## FUNCTIONS ## #


# TODO (MI): Possibly modularize this into 2 or 3 modules. e.g. growth matrices, splitting matrices, growth application
def run_edge_growth(params: forecast_cnfg.EDGEParameters) -> None:
    """Run Growth Process."""
    LOG.info("#" * 80)
    LOG.info("Started Process @ %s", timing.get_datetime())
    LOG.info("#" * 80)

    # fixed objects
    time_periods = ["AM", "IP", "PM", "OP"]

    # read global input files
    demand_segments = file_ops.read_df(params.demand_segments)
    demand_segments.loc[:, "ToHome"] = demand_segments["ToHome"].astype(
        bool
    )
    model_stations_tlcs = file_ops.read_df(
        params.norms_to_edge_stns_path
    )
    ticket_splits_df = file_ops.read_df(params.ticket_type_splits_path)
    flow_cats = file_ops.read_df(params.flow_cat_path)
    edge_flows = file_ops.read_df(
        params.edge_flows_path, usecols=[0, 2, 5]
    )
    # declare journey purposes
    purposes = demand_segments["Purpose"].drop_duplicates().to_list()

    # demand segment list groups
    # NoRMS demand segments
    norms_segments = (
        demand_segments.loc[demand_segments["ModelSegment"] == 1][
            ["Segment"]
        ]
        .drop_duplicates()
        .values.tolist()
    )
    norms_segments = [
        segment for sublist in norms_segments for segment in sublist
    ]
    # all segments
    all_segments = demand_segments["Segment"].to_list()

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
        growth_matrices = prepare_growth_matrices(
            demand_segments, growth_factors, model_stations_tlcs
        )
        # fill growth matrices
        filled_growth_matrices = fill_missing_factors(
            purposes,
            growth_matrices,
        )
        # TODO (MI): Record where no factor or a different factor is being used and log
        # create empty dictionary to store matrices
        factored_matrices = {}
        factored_24hr_matrices = {}
        # declare global demand total variables
        overall_base_demand = 0
        overall_not_grown_demand = 0
        # empty dataframe for growth summary
        growth_summary = pd.DataFrame(
            {
                "Time_Period": [],
                "Demand_Segment": [],
                "Base_Demand": [],
                f"{forecast_year}_Demand": [],
            }
        )
        # loop over time periods
        for time_period in tqdm(
            time_periods,
            desc="Time Periods Loop ",
            unit=" Period",
            colour="cyan",
            total=len(time_periods),
        ):
            LOG.info(
                "-- Processing Time Period %s @ %s",
                time_period,
                timing.get_datetime(),
            )
            # create dictionary
            factored_matrices[time_period] = {}
            # read time period specific files
            irsj_props = pd.read_hdf(
                params.matrices_to_grow_dir
                / f"{time_period}_iRSj_probabilities.h5",
                key="iRSj",
            )
            dist_mx = pd.read_csv(
                params.matrices_to_grow_dir
                / f"{time_period}_stn2stn_costs.csv",
                usecols=[0, 1, 4],
            )
            # produce ticket type splitting matrices
            splitting_matrices = produce_ticketype_splitting_matrices(
                edge_flows,
                model_stations_tlcs,
                dist_mx,
                flow_cats,
                ticket_splits_df,
            )
            LOG.info(
                f"{'Time_Period':>12}{'Demand_Segment':>15}"
                f"{'Base_Demand':>12}{f'{forecast_year}_Demand':>12}"
            )
            # TODO (MI): Potentially multiprocessing
            # loop over demand segments
            for i, row in tqdm(
                demand_segments.iterrows(),
                desc="    Demand Segments Loop ",
                unit=" Segment",
                colour="cyan",
                total=len(demand_segments),
            ):
                # store current segment's details
                segment = row["Segment"]
                to_home = row["ToHome"]
                growth_method = row["Growth_Method"]
                userclass = row["Userclass"]
                purpose = row["Purpose"]
                # read demand matrix
                with omx_file.OMXFile(
                    pathlib.Path(
                        params.matrices_to_grow_dir,
                        f"PT_{time_period}.omx",
                    )
                ) as omx_mat:
                    # read segment matrix into a dataframe
                    zonal_base_demand_mx = wide_mx_2_long_mx(
                        omx_mat.get_matrix_level(segment),
                        rows="from_model_zone_id",
                        cols="to_model_zone_id",
                    )
                # check if matrix has no demand then continue
                if zonal_base_demand_mx["Demand"].sum() == 0:
                    # keep matrix as it is, i.e. = 0
                    factored_matrices[time_period][
                        segment
                    ] = long_mx_2_wide_mx(zonal_base_demand_mx)
                    LOG.info(
                        f"{time_period:>12}{segment:>15}"
                        f"{0:>12}{0:>12}"
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
                ) = zonal_from_to_stations_demand(
                    zonal_base_demand_mx,
                    irsj_probs_segment,
                    len(model_stations_tlcs),
                    userclass,
                    to_home,
                )
                # store matrix total demand
                tot_input_demand = round(
                    np_stn2stn_base_demand_mx.sum()
                )
                # add to overall base demand
                overall_base_demand = (
                    overall_base_demand + tot_input_demand
                )
                # apply growth
                np_stn2stn_grown_demand_mx = apply_demand_growth(
                    np_stn2stn_base_demand_mx,
                    splitting_matrices,
                    filled_growth_matrices,
                    len(np_stn2stn_base_demand_mx),
                    purpose,
                    growth_method,
                    to_home,
                )
                # get movements where no growth has been applied
                no_growth_movements = np_stn2stn_base_demand_mx[
                    np_stn2stn_base_demand_mx
                    == np_stn2stn_grown_demand_mx
                ]
                # Add to the total not grown demand
                overall_not_grown_demand = (
                    overall_not_grown_demand + no_growth_movements.sum()
                )
                # store matrix total demand
                tot_output_demand = round(
                    np_stn2stn_grown_demand_mx.sum()
                )
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
                zonal_grown_demand_mx = convert_stns_to_zonal_demand(
                    np_stn2stn_grown_demand_mx,
                    zonal_from_to_stns,
                    time_period,
                    to_home,
                )
                # add to grown matrices dictionary
                factored_matrices[time_period][
                    segment
                ] = zonal_grown_demand_mx
        # calculate proportion of not grown demand
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
        for i, row in demand_segments.iterrows():
            # declare current segment's details
            segment = row["Segment"]
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
        norms_matrices = fromto_2_from_by_averaging(
            factored_24hr_matrices, norms_segments, all_segments
        )

        # export files
        # TODO (MI): Export to .OMX and then convert .OMX to .MAT instead of the .CSVs
        for segment in norms_segments:
            # write out demand matrix
            file_ops.write_df(
                wide_mx_2_long_mx(
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
        convert_csv_2_mat(
            norms_segments,
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
