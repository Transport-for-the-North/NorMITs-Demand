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
# TODO Remove this function when no-longer needed
def _overwrite_notem_segment_name(filename: str, segmentation: nd.SegmentationLevel) -> str:
    """Temporary function to convert to older naming style used by NoTEM.

    This should be removed once NoTEM has been updated to generate file
    names using the DVector method.
    """
    return filename.replace(segmentation.get_name_without_trip_origin(), "notem_segmented")


def read_tripends(
    base_year: int,
    forecast_years: list[int],
    tripend_path: Path,
    zoning_system: nd.ZoningSystem,
) -> tempro_trip_ends.TEMProTripEnds:
    """Load trip-end DVectors from pickled files.

    Parameters
    ----------
    base_year : int
        Base year for the forecast.
    forecast_years : list[int]
        List of forecast years.
    tripend_path : Path
        Path to the base trip end folder, assumed
        to have sub-folders for HB / NHB attractions
        and productions e.g. "hb_productions".
    zoning_system : nd.ZoningSystem
        Zoning system of the trip ends.

    Returns
    -------
    tempro_trip_ends.TEMProTripEnds
        Trip end DVectors for HB / NHB productions
        and attractions.
    """
    LOG.info("Reading trip ends from %s", tripend_path)

    # TODO Check if trip ends are positive

    input_segmentations = {
        "hb_attractions": "notem_hb_output",
        "hb_productions": "notem_hb_output",
        "nhb_attractions": "notem_nhb_output",
        "nhb_productions": "notem_nhb_output",
    }

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
                segmentation = nd.get_segmentation_level(input_segmentations[f"{i}_{j}"])
                filename = nd.DVector.build_filename_from_attributes(
                    segmentation, zoning_system, year
                )
                filename = _overwrite_notem_segment_name(filename, segmentation)

                dvec = nd.DVector.load(os.path.join(tripend_path, key, filename))
                if i == "nhb":
                    dvec = dvec.reduce(nd.get_segmentation_level("notem_nhb_output_reduced"))

                years[year] = dvec.aggregate(nd.get_segmentation_level(SEGMENTATION[i]))

            dvectors[key] = years

    return tempro_trip_ends.TEMProTripEnds(**dvectors)
