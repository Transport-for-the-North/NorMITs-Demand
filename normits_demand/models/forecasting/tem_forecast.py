# -*- coding: utf-8 -*-
"""
    Module for producing forecast demand constrained to MiTEM.
"""

##### IMPORTS #####
# Standard imports
import os
from pathlib import Path

# Third party imports

# Local imports
import normits_demand as nd
from normits_demand import logging as nd_log
from normits_demand.models.forecasting import tempro_trip_ends

##### CONSTANTS #####
LOG = nd_log.get_logger(__name__)


##### CLASSES #####
def read_tripends(
    base_year: int, forecast_years: list[int], tripend_path: Path
) -> tempro_trip_ends.TEMProTripEnds:
    """
    Reads in trip-end dvectors from picklefiles
    Args:
        base_year (int): The base year for the forecast
        forecast_years (list[int]): A list of forecast years
    Returns:
        tempro_trip_ends.TEMProTripEnds: the same trip-ends read in
    """
    SEGMENTATION = {"hb": "hb_p_m", "nhb": "nhb_p_m"}
    dvectors = {
        "hb_attractions": {},
        "hb_productions": {},
        "nhb_attractions": {},
        "nhb_productions": {},
    }
    for i in ["hb", "nhb"]:
        for j in ["productions", "attractions"]:
            years = {}
            key = f"{i}_{j}"
            for year in [base_year] + forecast_years:
                dvec = nd.DVector.load(
                    os.path.join(
                        tripend_path,
                        key,
                        f"{i}_msoa_notem_segmented_{year}_dvec.pkl",
                    )
                )
                if i == "nhb":
                    dvec = dvec.reduce(nd.get_segmentation_level("notem_nhb_output_reduced"))
                years[year] = dvec.aggregate(nd.get_segmentation_level(SEGMENTATION[i]))
            dvectors[key] = years
    return tempro_trip_ends.TEMProTripEnds(**dvectors)
