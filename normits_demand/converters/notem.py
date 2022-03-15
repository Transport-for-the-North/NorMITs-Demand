# -*- coding: utf-8 -*-
"""
Created on: 11/03/2022
Updated on:

Original author: Ben Taylor
Last update made by:
Other updates made by:

File purpose:

"""
# Built-Ins
import os
import logging
import pathlib

from typing import Union

# Third Party

# Local Imports
import normits_demand as nd

from normits_demand.pathing import NoTEMExportPaths
from normits_demand import core as nd_core

# Module logger
LOG = logging.getLogger(__name__)


class ToDistributionModel:
    
    def __init__(
        self,
        output_zoning: nd_core.ZoningSystem,
        hb_productions_path: pathlib.Path = None,
        hb_attractions_path: pathlib.Path = None,
        nhb_productions_path: pathlib.Path = None,
        nhb_attractions_path: pathlib.Path = None,
        cache_path: pathlib.Path = None,
        time_format: nd_core.TimeFormat = nd_core.TimeFormat.AVG_DAY,
    ):
        # Assign attributes
        self.hb_productions_path = hb_productions_path
        self.hb_attractions_path = hb_attractions_path
        self.nhb_productions_path = nhb_productions_path
        self.nhb_attractions_path = nhb_attractions_path

        self.output_zoning = output_zoning
        self.cache_path = cache_path
        self.time_format = time_format

    def convert_hb(
        self,
        ignore_cache: bool = False,
        *args,
        **kwargs,
    ) -> nd_core.DVector:
        """Reads in and converts the hb_production and hb_attraction Dvectors

        Converts into the format required by the distribution model, which is
        defined by: reduce_segmentation, subset_segmentation,
        aggregation_segmentation, and modal_segmentation.
        Optionally makes use of a cached version the converted DVector if one
        exists, ignore_cache is False, and self.cache_path is set.

        Parameters
        ----------
        ignore_cache:
            Whether to ignore the cache and recreate the cache no matter
            what.

        args:
            Used for compatibility with `self.maybe_read_and_convert_trip_end()`

        kwargs:
            Used for compatibility with `self.maybe_read_and_convert_trip_end()`

        Returns
        -------
        converted_dvec:
            The convert trip_end_or_path DVector

        See Also
        --------
        `self.convert_hb_productions()`
        `self.convert_hb_attractions()`
        `self.maybe_read_and_convert_trip_end()`
        """
        return (
            self.convert_hb_productions(ignore_cache, *args, **kwargs),
            self.convert_hb_attractions(ignore_cache, *args, **kwargs),
        )

    def convert_nhb(
        self,
        ignore_cache: bool = False,
        *args,
        **kwargs,
    ) -> nd_core.DVector:
        """Reads in and converts the nhb_production and nhb_attraction Dvectors

        Converts into the format required by the distribution model, which is
        defined by: reduce_segmentation, subset_segmentation,
        aggregation_segmentation, and modal_segmentation.
        Optionally makes use of a cached version the converted DVector if one
        exists, ignore_cache is False, and self.cache_path is set.

        Parameters
        ----------
        ignore_cache:
            Whether to ignore the cache and recreate the cache no matter
            what.

        args:
            Used for compatibility with `self.maybe_read_and_convert_trip_end()`

        kwargs:
            Used for compatibility with `self.maybe_read_and_convert_trip_end()`

        Returns
        -------
        converted_dvec:
            The convert trip_end_or_path DVector

        See Also
        --------
        `self.convert_nhb_productions()`
        `self.convert_nhb_attractions()`
        `self.maybe_read_and_convert_trip_end()`
        """
        return (
            self.convert_nhb_productions(ignore_cache, *args, **kwargs),
            self.convert_nhb_attractions(ignore_cache, *args, **kwargs),
        )

    def convert_hb_productions(
        self,
        ignore_cache: bool = False,
        *args,
        **kwargs,
    ) -> nd_core.DVector:
        """Reads in and converts the hb_production Dvector

        Converts into the format required by the distribution model, which is
        defined by: reduce_segmentation, subset_segmentation,
        aggregation_segmentation, and modal_segmentation.
        Optionally makes use of a cached version the converted DVector if one
        exists, ignore_cache is False, and self.cache_path is set.

        Parameters
        ----------
        ignore_cache:
            Whether to ignore the cache and recreate the cache no matter
            what.

        args:
            Used for compatibility with `self.maybe_read_and_convert_trip_end()`

        kwargs:
            Used for compatibility with `self.maybe_read_and_convert_trip_end()`

        Returns
        -------
        converted_dvec:
            The convert trip_end_or_path DVector

        See Also
        --------
        `self.maybe_read_and_convert_trip_end()`
        """

        return self.maybe_read_and_convert_trip_end(
            dvector_path=self.hb_productions_path,
            cache_fname=self.hb_productions_path.name,
            ignore_cache=ignore_cache,
            translation_weighting='population',
            *args,
            **kwargs
        )

    def convert_hb_attractions(
        self,
        ignore_cache: bool = False,
        *args,
        **kwargs,
    ) -> nd_core.DVector:
        """Reads in and converts the hb_attraction Dvector

        Converts into the format required by the distribution model, which is
        defined by: reduce_segmentation, subset_segmentation,
        aggregation_segmentation, and modal_segmentation.
        Optionally makes use of a cached version the converted DVector if one
        exists, ignore_cache is False, and self.cache_path is set.

        Parameters
        ----------
        ignore_cache:
            Whether to ignore the cache and recreate the cache no matter
            what.

        args:
            Used for compatibility with `self.maybe_read_and_convert_trip_end()`

        kwargs:
            Used for compatibility with `self.maybe_read_and_convert_trip_end()`

        Returns
        -------
        converted_dvec:
            The convert trip_end_or_path DVector

        See Also
        --------
        `self.maybe_read_and_convert_trip_end()`
        """

        return self.maybe_read_and_convert_trip_end(
            dvector_path=self.hb_attractions_path,
            cache_fname=self.hb_attractions_path.name,
            ignore_cache=ignore_cache,
            translation_weighting='employment',
            *args,
            **kwargs
        )

    def convert_nhb_productions(
        self,
        ignore_cache: bool = False,
        *args,
        **kwargs,
    ) -> nd_core.DVector:
        """Reads in and converts the nhb_production Dvector

        Converts into the format required by the distribution model, which is
        defined by: reduce_segmentation, subset_segmentation,
        aggregation_segmentation, and modal_segmentation.
        Optionally makes use of a cached version the converted DVector if one
        exists, ignore_cache is False, and self.cache_path is set.

        Parameters
        ----------
        ignore_cache:
            Whether to ignore the cache and recreate the cache no matter
            what.

        args:
            Used for compatibility with `self.maybe_read_and_convert_trip_end()`

        kwargs:
            Used for compatibility with `self.maybe_read_and_convert_trip_end()`

        Returns
        -------
        converted_dvec:
            The convert trip_end_or_path DVector

        See Also
        --------
        `self.maybe_read_and_convert_trip_end()`
        """

        return self.maybe_read_and_convert_trip_end(
            dvector_path=self.nhb_productions_path,
            cache_fname=self.nhb_productions_path.name,
            ignore_cache=ignore_cache,
            translation_weighting='population',
            *args,
            **kwargs
        )

    def convert_nhb_attractions(
        self,
        ignore_cache: bool = False,
        *args,
        **kwargs,
    ) -> nd_core.DVector:
        """Reads in and converts the nhb_attraction Dvector

        Converts into the format required by the distribution model, which is
        defined by: reduce_segmentation, subset_segmentation,
        aggregation_segmentation, and modal_segmentation.
        Optionally makes use of a cached version the converted DVector if one
        exists, ignore_cache is False, and self.cache_path is set.

        Parameters
        ----------
        ignore_cache:
            Whether to ignore the cache and recreate the cache no matter
            what.

        args:
            Used for compatibility with `self.maybe_read_and_convert_trip_end()`

        kwargs:
            Used for compatibility with `self.maybe_read_and_convert_trip_end()`

        Returns
        -------
        converted_dvec:
            The convert trip_end_or_path DVector

        See Also
        --------
        `self.maybe_read_and_convert_trip_end()`
        """

        return self.maybe_read_and_convert_trip_end(
            dvector_path=self.nhb_attractions_path,
            cache_fname=self.nhb_attractions_path.name,
            ignore_cache=ignore_cache,
            translation_weighting='employment',
            *args,
            **kwargs
        )

    def maybe_read_and_convert_trip_end(
        self,
        dvector_path: pathlib.Path,
        cache_fname: pathlib.Path,
        ignore_cache: bool,
        translation_weighting: str,
        reduce_segmentation: nd_core.SegmentationLevel = None,
        subset_segmentation: nd_core.SegmentationLevel = None,
        aggregation_segmentation: nd_core.SegmentationLevel = None,
        modal_segmentation: nd_core.SegmentationLevel = None,
    ) -> nd_core.DVector:
        """Caching wrapper around `read_and_convert_trip_end()`

        Parameters
        ----------
        dvector_path:
            The DVector read in and convert.

        cache_fname:
            The name to give to this trip end when cached to disk.

        ignore_cache:
            Whether to ignore the cache and recreate the cache no matter
            what.

        translation_weighting:
            The weighting to use when translating trip_end_or_path to
            self.output_zoning.

        reduce_segmentation:
            The name of the segmentation to reduce to if some segments in
            trip_end_or_path need to be combined together.

        subset_segmentation:
            The name of the segmentation when time periods 5/6 are dropped from
            the input DVector.

        aggregation_segmentation:
            The name of the segmentation, containing all modes, which will be
            used in the distribution model. This segmentation will be
            modal_segmentation before the unneeded modes have been dropped.

        modal_segmentation:
            This will be the final output segmentation and should be the same
            as the segmentation used in
            `DistributionModel.running_segmentation`.

        Returns
        -------
        converted_dvec:
            The convert trip_end_or_path DVector
        """
        # Ignore cache if no path given
        if self.cache_path is None:
            ignore_cache = True
            cache_path = None

        # Figure out if the cache already exists and is valid
        else:
            cache_path = self.cache_path / cache_fname

            # Ignore cache if DVec was more recently modified
            dvec_modified_time = os.path.getmtime(dvector_path)
            cache_modified_time = os.path.getmtime(cache_path)
            if dvec_modified_time > cache_modified_time:
                ignore_cache = True

        # Return the cache only if it's safe to
        if not ignore_cache and cache_path.is_file():
            return nd_core.DVector.load(cache_path)

        # If here, we need to recreate the cache
        converted_dvec = self.read_and_convert_trip_end(
            trip_end_or_path=dvector_path,
            translation_weighting=translation_weighting,
            reduce_segmentation=reduce_segmentation,
            subset_segmentation=subset_segmentation,
            aggregation_segmentation=aggregation_segmentation,
            modal_segmentation=modal_segmentation,
        )

        # Save into cache and return
        if cache_path is not None:
            converted_dvec.save(cache_path)
        return converted_dvec

    def read_and_convert_trip_end(
        self,
        trip_end_or_path: Union[pathlib.Path, nd_core.DVector],
        translation_weighting: str,
        reduce_segmentation: nd_core.SegmentationLevel = None,
        subset_segmentation: nd_core.SegmentationLevel = None,
        aggregation_segmentation: nd_core.SegmentationLevel = None,
        modal_segmentation: nd_core.SegmentationLevel = None,
    ) -> nd_core.DVector:
        """Reads in and converts trips ends ready for the DistributionModel

        Adds compatibility with NoTEM outputs. Will take the NoTEM output
        trip ends and convert them into a format ready to be run in the
        DistributionModel.
        Also makes use of an internal cache functionality which can be used to
        speed up conversion times if the cache already exists and is newer
        than the input files at the given trip end paths.

        Parameters
        ----------
        trip_end_or_path:
            Either a DVector or a path to DVector saved to disk to read in
            and convert.

        translation_weighting:
            The weighting to use when translating trip_end_or_path to
            self.output_zoning.

        reduce_segmentation:
            The name of the segmentation to reduce to if some segments in
            trip_end_or_path need to be combined together.

        subset_segmentation:
            The name of the segmentation when time periods 5/6 are dropped from
            the input DVector.

        aggregation_segmentation:
            The name of the segmentation, containing all modes, which will be
            used in the distribution model. This segmentation will be
            modal_segmentation before the unneeded modes have been dropped.

        modal_segmentation:
            This will be the final output segmentation and should be the same
            as the segmentation used in
            `DistributionModel.running_segmentation`.

        Returns
        -------
        converted_dvec:
            The convert trip_end_or_path DVector
        """
        # Read in DVector if given path
        if isinstance(trip_end_or_path, nd.core.data_structures.DVector):
            dvec = trip_end_or_path
        else:
            dvec = nd_core.DVector.load(trip_end_or_path)

        # Reduce nhb 11 into 12 if needed
        if reduce_segmentation is not None:
            dvec = dvec.reduce(out_segmentation=reduce_segmentation)

        # Convert from ave_week to ave_day
        dvec = dvec.subset(out_segmentation=subset_segmentation)
        dvec = dvec.convert_time_format(self.time_format)

        # Convert zoning and segmentation to desired
        dvec = dvec.aggregate(aggregation_segmentation)
        dvec = dvec.subset(modal_segmentation)
        dvec = dvec.translate_zoning(self.output_zoning, translation_weighting)

        return dvec
