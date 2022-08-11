# -*- coding: utf-8 -*-
"""
    Classes for converting trip end DVectors into format for use within the
    traveller segmentation disaggragator tool.
"""

##### IMPORTS #####
from typing import Tuple

import normits_demand as nd
from normits_demand.converters import notem


##### CLASSES #####
class NoTEMToTravellerSegmentation(notem.NoTEMToDistributionModel):
    """Converts NoTEM outputs to a format usable by traveller segmentation tool.

    Wrapper around NoTEMToDistributionModel, with additional function
    `get_trip_ends` which will return the trip ends needed for the
    traveller segmentation tool.
    """

    def get_trip_ends(
        self,
        trip_origin: nd.TripOrigin,
        segmentation: nd.SegmentationLevel,
        ignore_cache: bool = False,
    ) -> Tuple[nd.DVector, nd.DVector]:
        """Get trip end DVectors for use in the traveller segmentation tool.

        Parameters
        ----------
        trip_origin : nd.TripOrigin
            Trip origin type to get trip ends for.
        segmentation : nd.SegmentationLevel
            Segmentation level to convert trip ends to before returning.
        ignore_cache : bool, default False
            Whether to ignore the cache and recreate the cache no matter
            what.

        Returns
        -------
        nd.DVector, nd.DVector
            Productions and attractions trip ends at `segmentation` and
            `self.output_zoning` zoning system.

        Raises
        ------
        ValueError
            If `trip_origin` isn't HB or NHB.
        """
        if trip_origin == nd.TripOrigin.HB:
            productions, attractions = self.convert_hb(
                aggregation_segmentation=segmentation, ignore_cache=ignore_cache
            )
        elif trip_origin == nd.TripOrigin.NHB:
            productions, attractions = self.convert_nhb(
                aggregation_segmentation=segmentation, ignore_cache=ignore_cache
            )
        else:
            raise ValueError(f"Unexpected value for trip_origin: {trip_origin!r}")

        return productions, attractions
