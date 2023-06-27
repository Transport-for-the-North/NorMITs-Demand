# -*- coding: utf-8 -*-
"""
Created on: 26/05/2023
Updated on:

Original author: Ben Taylor
Last update made by:
Other updates made by:

File purpose:

"""
# Built-Ins
import dataclasses
import pickle
from pathlib import Path
from typing import Union

# Third Party
import pandas as pd

# Local Imports
# pylint: disable=import-error,wrong-import-position
from normits_demand.models.forecasting import forecast_cnfg                                                              
from normits_demand.utils import file_ops
from normits_demand.models.forecasting.edge_growth import ticket_splits


# pylint: enable=import-error,wrong-import-position

# # # CONSTANTS # # #

# # # CLASSES # # #
@dataclasses.dataclass
class TPLoopOutput:
    overall_not_grown: int
    overall_base: int
    factored: dict
    summary: pd.DataFrame
    """
    Return from _tp_loop function in 'run'
    
    Parameters
    ----------
    overall_not_grown: Sum of matrix movements not grown by the process.
    overall_base: Sum of base demand.
    factored: Dictionary of grown matrices. Keys are demand segments.
    summary: A summary dataframe with rows for each segment.
    """

@dataclasses.dataclass
class GlobalVars:
    demand_segments: pd.DataFrame
    purposes: list
    norms_segments: list
    all_segments: list
    ticket_type_splits: pd.DataFrame
    station_tlcs: pd.DataFrame
    time_periods: tuple = ("AM", "IP", "PM", "OP")
    """
    Designed as an output from load_globals (below).
    
    Parameters
    ----------
    demand_segments: A dataframe of demand segments simply read in.
    purposes: List of purposes, derived from the 'Purposes' column of demand_segments.
    norms_segments: Norms segments list from demand_segments.
    all_segments: All segments from demand_segments.
    ticket_type_splits: Ticket type splits, either read in or produced by the ticket_splits module.
    station_tlcs: Dataframe loaded from csv.
    time_periods: Tuple of strings. Defaults.
    """

    def keys(self):
        return [field.name for field in dataclasses.fields(self)]


# # # FUNCTIONS # # #
def load_globals(params: forecast_cnfg.EDGEParameters) -> GlobalVars:
    """
    Load in global variables for the process (i.e. variables which don't vary by year/time_period/segment).
    """
    demand_segments = file_ops.read_df(params.demand_segments)
    norms_segments = (
        demand_segments.loc[demand_segments["ModelSegment"] == 1][
            ["Segment"]
        ]
        .drop_duplicates()
        .values.tolist()
    )
    purposes = demand_segments["Purpose"].drop_duplicates().to_list()
    norms_segments = [
        segment for sublist in norms_segments for segment in sublist
    ]
    # all segments
    all_segments = demand_segments["Segment"].to_list()
    demand_segments["ToHome"] = demand_segments["ToHome"].astype(
        bool
    )
    model_stations_tlcs = file_ops.read_df(
        params.norms_to_edge_stns_path
    )
    ticket_type_splits = ticket_splits_logic(params.ticket_type_splits, params.ticket_splits_dir, model_stations_tlcs)
    

    return GlobalVars(demand_segments, purposes, norms_segments, all_segments, ticket_type_splits, model_stations_tlcs)

def ticket_splits_logic(ticket_type_splits: Union[forecast_cnfg.TicketSplitParams, Path], splits_dir: Path, model_stations_tlcs: pd.DataFrame):
    """
    Logic for handling various ways of producing ticket type splits (ultimately either loading from a pickle file or generating)
    Parameters
    ----------
    ticket_type_splits (Union[forecast_cnfg.TicketSplitParams, Path]): Directly from params.
    splits_dir (Path): From params. This will be updated soon to a dedicated cache
    model_stations_tlcs (pd.DataFrame): Loaded in load_globals

    Returns:
        _type_: _description_
    """
    if isinstance(ticket_type_splits, Path):
        with open(ticket_type_splits, "rb") as file:
            return pickle.load(file)
    tick_param_path = splits_dir / "ticket_split_params.yml"
    if tick_param_path.is_file():
        ex_params = forecast_cnfg.TicketSplitParams.load_yaml(tick_param_path)
        if ex_params == ticket_type_splits:
            with open(splits_dir / "splitting_matrices.pkl", "rb") as file:
                return pickle.load(file)
        return ticket_splits.splits_loop(ticket_type_splits, model_stations_tlcs, splits_dir)
    return ticket_splits.splits_loop(ticket_type_splits, model_stations_tlcs, splits_dir)
