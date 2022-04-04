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
import pathlib

from typing import Tuple
from typing import Union

# Third Party

# Local Imports
import normits_demand as nd

from normits_demand import core as nd_core
from normits_demand.pathing import NoTEMExportPaths


class ToDistributionModel:
    
    def __init__(
        self,
        output_zoning: nd_core.ZoningSystem,
        hb_productions_path: pathlib.Path = None,
        hb_attractions_path: pathlib.Path = None,
        nhb_productions_path: pathlib.Path = None,
        nhb_attractions_path: pathlib.Path = None,
        cache_dir: pathlib.Path = None,
        time_format: nd_core.TimeFormat = nd_core.TimeFormat.AVG_DAY,
    ):
        """
        Parameters
        ----------
        output_zoning:
            The zoning system the output DVectors should be in

        hb_productions_path:
            The path to the pickled DVector of home-based productions.

        hb_attractions_path:
            The path to the pickled DVector of home-based attractions.

        nhb_productions_path:
            The path to the pickled DVector of non-home-based productions.

        nhb_attractions_path:
            The path to the pickled DVector of non-home-based attractions.

        cache_dir:
            A path to the directory to store the cached DVectors.

        time_format:
            The nd_core.TimeFormat to use in the output DVectors.
        """
        # Assign attributes
        self.hb_productions_path = hb_productions_path
        self.hb_attractions_path = hb_attractions_path
        self.nhb_productions_path = nhb_productions_path
        self.nhb_attractions_path = nhb_attractions_path

        self.output_zoning = output_zoning
        self.cache_dir = cache_dir
        self.time_format = time_format

    def convert(
        self,
        *args,
        trip_origin: nd_core.TripOrigin,
        ignore_cache: bool = False,
        **kwargs,
    ) -> Tuple[nd_core.DVector, nd_core.DVector]:
        """Reads in and converts the trip origin DVectors

        Wrapper class for `self.convert_hb()` or `self.convert_nhb()`.

        Converts into the format required by the distribution model, which is
        defined by: reduce_segmentation, subset_segmentation,
        aggregation_segmentation, and modal_segmentation.
        Optionally makes use of a cached version the converted DVector if one
        exists, ignore_cache is False, and self.cache_dir is set.

        Parameters
        ----------
        trip_origin:
            The trip origin to convert. Used to determine whether to pick the
            `self.convert_hb()` or `self.convert_nhb()` function.

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
        `self.convert_hb()`
        `self.convert_nhb()`
        `self.convert_hb_productions()`
        `self.convert_hb_attractions()`
        `self.maybe_read_and_convert_trip_end()`
        """
        if trip_origin == nd_core.TripOrigin.HB:
            return self.convert_hb(ignore_cache=ignore_cache, *args, **kwargs)

        if trip_origin == nd_core.TripOrigin.NHB:
            return self.convert_nhb(ignore_cache=ignore_cache, *args, **kwargs)

        raise ValueError(
            "Don't know how to convert the trip ends for trip origin: %s"
            % trip_origin.value
        )

    def convert_hb(
        self,
        *args,
        ignore_cache: bool = False,
        **kwargs,
    ) -> Tuple[nd_core.DVector, nd_core.DVector]:
        """Reads in and converts the hb_production and hb_attraction Dvectors

        Converts into the format required by the distribution model, which is
        defined by: reduce_segmentation, subset_segmentation,
        aggregation_segmentation, and modal_segmentation.
        Optionally makes use of a cached version the converted DVector if one
        exists, ignore_cache is False, and self.cache_dir is set.

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
    ) -> Tuple[nd_core.DVector, nd_core.DVector]:
        """Reads in and converts the nhb_production and nhb_attraction Dvectors

        Converts into the format required by the distribution model, which is
        defined by: reduce_segmentation, subset_segmentation,
        aggregation_segmentation, and modal_segmentation.
        Optionally makes use of a cached version the converted DVector if one
        exists, ignore_cache is False, and self.cache_dir is set.

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
        exists, ignore_cache is False, and self.cache_dir is set.

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

        Raises
        ------
        ValueError:
            If self.hb_productions_path has not been set before calling.

        See Also
        --------
        `self.maybe_read_and_convert_trip_end()`
        """
        if self.hb_productions_path is None:
            raise ValueError(
                "Cannot convert the hb_productions as no path was given"
                "to the home-based productions."
            )

        return self.maybe_read_and_convert_trip_end(
            dvector_path=self.hb_productions_path,
            cache_fname='hb_productions_dvec.pkl',
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
        exists, ignore_cache is False, and self.cache_dir is set.

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

        Raises
        ------
        ValueError:
            If self.hb_attractions_path has not been set before calling.

        See Also
        --------
        `self.maybe_read_and_convert_trip_end()`
        """
        if self.hb_attractions_path is None:
            raise ValueError(
                "Cannot convert the hb_attractions as no path was given"
                "to the home-based attractions."
            )

        return self.maybe_read_and_convert_trip_end(
            dvector_path=self.hb_attractions_path,
            cache_fname='hb_attractions_dvec.pkl',
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
        exists, ignore_cache is False, and self.cache_dir is set.

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

        Raises
        ------
        ValueError:
            If self.nhb_productions_path has not been set before calling.

        See Also
        --------
        `self.maybe_read_and_convert_trip_end()`
        """
        if self.nhb_productions_path is None:
            raise ValueError(
                "Cannot convert the nhb_productions as no path was given"
                "to the non-home-based productions."
            )

        return self.maybe_read_and_convert_trip_end(
            dvector_path=self.nhb_productions_path,
            cache_fname='nhb_productions_dvec.pkl',
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
        exists, ignore_cache is False, and self.cache_dir is set.

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

        Raises
        ------
        ValueError:
            If self.nhb_attractions_path has not been set before calling.


        See Also
        --------
        `self.maybe_read_and_convert_trip_end()`
        """
        if self.nhb_attractions_path is None:
            raise ValueError(
                "Cannot convert the nhb_attractions as no path was given"
                "to the non-home-based attractions."
            )

        return self.maybe_read_and_convert_trip_end(
            dvector_path=self.nhb_attractions_path,
            cache_fname='nhb_attractions_dvec.pkl',
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
        if self.cache_dir is None:
            ignore_cache = True
            cache_path = None

        # Figure out if the cache already exists and is valid
        else:
            cache_path = self.cache_dir / cache_fname

            # Ignore cache if DVec was more recently modified
            if cache_path.is_file():
                dvec_modified_time = os.path.getmtime(dvector_path)
                cache_modified_time = os.path.getmtime(cache_path)
                if dvec_modified_time > cache_modified_time:
                    ignore_cache = True

        # Return the cache only if it's safe to
        if not ignore_cache and cache_path.is_file():
            dvec = nd_core.DVector.load(cache_path)

            # Do a few checks
            if dvec.zoning_system == self.output_zoning:
                return dvec

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


class NoTEMToDistributionModel(ToDistributionModel):
    """Helper class to convert NoTEM outputs into Distribution Model inputs"""
    # BACKLOG: Build a tool which guesses internal segmentations to use and
    #  build into the converters
    #  labels: QoL, NoTEM

    def __init__(
        self,
        output_zoning: nd_core.ZoningSystem,
        base_year: int,
        scenario: nd_core.Scenario,
        notem_iteration_name: str,
        export_home: pathlib.Path,
        cache_dir: pathlib.Path = None,
        time_format: nd_core.TimeFormat = nd_core.TimeFormat.AVG_DAY,
    ):
        """
        Parameters
        ----------
        output_zoning:
            The zoning system the output DVectors should be in

        base_year:
            The year the Distribution Model is running for. Needed for
            compatibility with nd.pathing.NoTEMExportPaths

        scenario:
            The name of the scenario to run for. Needed for
            compatibility with nd.pathing.NoTEMExportPaths

        notem_iteration_name:
            The name of this iteration of the NoTEM models. Will be passed to
            nd.pathing.NoTEMExportPaths to generate the NoTEM paths.

        export_home:
            The home directory of all the export paths. Will be passed to
            nd.pathing.NoTEMExportPaths to generate the NoTEM paths.

        cache_dir:
            A path to the directory to store the cached DVectors.

        time_format:
            The nd_core.TimeFormat to use in the output DVectors.
        """
        # Get the path to the NoTEM Exports
        notem_exports = NoTEMExportPaths(
            path_years=[base_year],
            scenario=scenario.value,
            iteration_name=notem_iteration_name,
            export_home=export_home,
        )

        # Extract shorthand paths
        hbp_path = notem_exports.hb_production.export_paths.notem_segmented[base_year]
        hba_path = notem_exports.hb_attraction.export_paths.notem_segmented[base_year]
        nhbp_path = notem_exports.nhb_production.export_paths.notem_segmented[base_year]
        nhba_path = notem_exports.nhb_attraction.export_paths.notem_segmented[base_year]

        # Pass the generated paths over to the converter
        super().__init__(
            output_zoning=output_zoning,
            hb_productions_path=pathlib.Path(hbp_path),
            hb_attractions_path=pathlib.Path(hba_path),
            nhb_productions_path=pathlib.Path(nhbp_path),
            nhb_attractions_path=pathlib.Path(nhba_path),
            cache_dir=cache_dir,
            time_format=time_format,
        )
