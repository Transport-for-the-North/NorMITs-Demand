# -*- coding: utf-8 -*-
"""
Created on: 26/08/2020
Updated on:

Original author: Chris Storey
Last update made by: Ben Taylor
Other updates made by:

File purpose:

"""
# Built-Ins
import os
import pathlib

from typing import Any
from typing import Optional

# Third Party
import numpy as np
import pandas as pd

# Local Imports
# pylint: disable=import-error,wrong-import-position
import normits_demand as nd

from normits_demand import cost
from normits_demand import core as nd_core
from normits_demand import logging as nd_log
from normits_demand import constants as nd_consts
from normits_demand.utils import file_ops
from normits_demand.utils import timing
from normits_demand.utils import math_utils
from normits_demand.utils import functional as func_utils
from normits_demand.utils import string_utils as str_utils
from normits_demand.utils import pandas_utils as pd_utils
from normits_demand.tools.trip_length_distributions import enumerations as tld_enums
# pylint: enable=import-error,wrong-import-position

LOG = nd_log.get_logger(__name__)


class TripLengthDistributionBuilder:
    # Class constants
    _input_params_fname = "1. input_params.txt"
    _running_log_fname = "2. run_log.log"
    _full_export_fname = "3. full_export.csv"
    _seg_report_fname = "4. {seg_name}_report.csv"
    _agg_full_export_fname = "5. aggregated_full_export.csv"
    _agg_seg_report_fname = "6. {seg_name}_aggregated_report.csv"

    _distribution_fname_desc = "cost_distribution"

    # Maps for non-classified categories
    _household_type_to_ca = {
        "hh_type": [1, 2, 3, 4, 5, 6, 7, 8],
        "ca": [1, 2, 1, 2, 2, 1, 2, 2],
    }

    def __init__(
        self,
        tlb_folder: nd.PathLike,
        tlb_version: nd.PathLike,
        output_folder: nd.PathLike,
        bands_definition_dir: str,
        segment_copy_definition_dir: str,
        trip_miles_col: str = "TripDisIncSW",
        trip_count_col: str = "trips",
    ):
        """
        Define the environment for a set of trip length distribution runs.

        Parameters
        ----------
        tlb_folder: pd.DataFrame
            Path to folder containing TLD specific output from
            'NTS Processing' tool
        tlb_version: str
            Which version of the TLD export to pick up
        output_folder:
            NorMITs Demand config folder to export outputs to
        trip_miles_col:
            Which column to use as the trip miles in the import data
        trip_count_col:
            Which column to use as the count of trips in the import data
        """
        # TODO(BT): Pass this in
        self.input_cost_units = nd_core.CostUnits.MILES

        self.bands_definition_dir = bands_definition_dir
        self.segment_copy_definition_dir = segment_copy_definition_dir

        self.tlb_folder = tlb_folder
        self.tlb_version = tlb_version
        self.tlb_import_path = os.path.join(tlb_folder, tlb_version)
        self.output_folder = output_folder

        print(f"Loading processed NTS trip length data from {self.tlb_import_path}...")
        self.nts_import = pd.read_csv(self.tlb_import_path)

        # Validate needed columns exist
        if trip_miles_col not in list(self.nts_import):
            raise ValueError(f"Given trip miles col {trip_miles_col} not in NTS data")
        if trip_count_col not in list(self.nts_import):
            raise ValueError(f"Given trip count col {trip_count_col} not in NTS data")

        self.trip_distance_col = trip_miles_col
        self.trip_count_col = trip_count_col

    def _apply_od_geo_filter(
        self,
        nts_data: pd.DataFrame,
        geo_area: tld_enums.GeoArea,
        trip_filter_type: tld_enums.TripFilter,
    ) -> pd.DataFrame:
        """Internal function of self._apply_geo_filter"""
        # Init
        output_data = nts_data.copy()
        orig_gor_col = "TripOrigGOR_B02ID"
        dest_gor_col = "TripDestGOR_B02ID"

        # Transpose origin and destination for OD_to trip ends (Makes them "PA")
        mask = output_data[self.trip_origin_col] == "hb_to"
        temp_orig = output_data[mask][orig_gor_col].copy()
        temp_dest = output_data[mask][dest_gor_col].copy()
        output_data.loc[mask, orig_gor_col] = temp_dest
        output_data.loc[mask, dest_gor_col] = temp_orig

        # Decide how to filter
        if trip_filter_type == tld_enums.TripFilter.TRIP_OD:
            filter_orig = True
            filter_dest = True
        elif trip_filter_type == tld_enums.TripFilter.TRIP_O:
            filter_orig = True
            filter_dest = False
        elif trip_filter_type == tld_enums.TripFilter.TRIP_D:
            filter_orig = False
            filter_dest = True
        else:
            raise ValueError(f"Don't know how to apply the OD filter {trip_filter_type}")

        # Finally, filter the data
        if filter_orig:
            mask = output_data[orig_gor_col].isin(geo_area.get_gors())
            output_data = output_data[mask].reset_index(drop=True)

        if filter_dest:
            mask = output_data[dest_gor_col].isin(geo_area.get_gors())
            output_data = output_data[mask].reset_index(drop=True)

        return output_data

    def _apply_geo_filter(
        self,
        nts_data: pd.DataFrame,
        geo_area: tld_enums.GeoArea,
        trip_filter_type: tld_enums.TripFilter,
    ) -> pd.DataFrame:
        """Filters the data to a geographical area

        Parameters
        ----------
        nts_data: pd.DataFrame
            Processed NTS data

        trip_filter_type: str
            How to filter the origin/destination of trips.
            this is defined here in the class, so improvement work here.

        geo_area:
            Target regional subset

        Returns
        ----------
        output_dat:
            Input dataframe modified to target geography
        """
        # TODO(CS): Work for household trip_filter_type properly

        if trip_filter_type.is_od_type():
            return self._apply_od_geo_filter(
                nts_data=nts_data,
                geo_area=geo_area,
                trip_filter_type=trip_filter_type,
            )

        raise ValueError(f"Don't know how to apply to filter '{trip_filter_type}'")

    @staticmethod
    def _map_dict(output_dat: pd.DataFrame, map_dict: dict, key: str):

        """
        Analogue of pd.map for filling out a category from a dictionary
        output_dat: a DataFrame of NTS dataset
        map: a dictionary in category: list format
        key: string for join, existing category in output_dat
        """

        map_frame = pd.DataFrame(map_dict)

        output_dat = output_dat.merge(map_frame, how="left", on=key)

        return output_dat

    @staticmethod
    def _distribution_smoothing_interpolation(
        distribution: np.ndarray,
        mean_costs: np.ndarray,
        segment_params: dict[str, Any],
    ) -> np.ndarray:
        """Uses interpolation to smooth gaps in a distribution"""
        # Check for zero values
        if (distribution == 0).sum() <= 0:
            return distribution

        # Get indexes of non-zero values in array
        min_ = np.argwhere(distribution).min().squeeze()
        max_ = np.argwhere(distribution).max().squeeze()

        # Trim away non-zero values - max + 1 one as not inclusive
        trimmed_dist = distribution[min_:max_+1]
        trimmed_cost = mean_costs[min_:max_+1]

        # Check again for 0s within data
        if (trimmed_dist == 0).sum() <= 0:
            return distribution

        # If here, we need to interpolate something
        LOG.info(
            f"Gap found in %s distribution. Linear interpolation "
            "will fill the gap."
            % segment_params
        )

        # Interpolate, and fit back in place
        interpolated = distribution.copy()
        interpolated[min_:max_+1] = math_utils.interpolate_array(trimmed_dist, trimmed_cost)
        return interpolated

    @staticmethod
    def _passing_tld_checks(distribution: cost.CostDistribution) -> bool:
        """Checks if a TLD looks valid"""
        # make sure it's not empty
        if distribution.is_empty():
            return False

        # Trim the 0s off the edges
        trips = distribution.band_trips
        min_ = np.argwhere(trips).min().squeeze()
        max_ = np.argwhere(trips).max().squeeze()
        trimmed_trips = trips[min_:max_+1]     # Plus one as not inclusive

        # Check if the values get smaller across bands
        for prev, cur in func_utils.pairwise(trimmed_trips):
            if prev < cur:
                return False

        return True

    def _build_cost_distribution(
        self,
        data: pd.DataFrame,
        band_edges: np.ndarray,
        output_cost_units: nd_core.CostUnits,
        segment_params: dict[str, Any],
        trip_count_col: str = None,
        trip_dist_col: str = None,
        inter_smoothing: bool = False,
    ) -> cost.CostDistribution:
        """Generates a CostDistribution from a dataframe of data and bands

        Distributes `data` between `band_edges`,
        counting trips per band and mean trip length per band.

        Parameters
        ----------
        data:
            The data to convert into a `CostDistribution`. Must contain the
            columns: `trip_count_col`, `trip_miles_col`

        band_edges:
            The edges to use for each band in the distribution. E.g.
            `[1, 2, 3]` defines 2 bands: 1->2 and 2->3

        output_cost_units:
            The cost units to convert the data to from `self.input_cost_units`

        segment_params:
            A dictionary defining the segmentation used to generate this
            cost distribution.

        trip_count_col:
            The name of the column in `data` containing the count of trips
            being made.

        trip_dist_col:
            The name of the column in `data` containing the distance of trips
            being made. This should be in `self.input_cost_units` units.

        inter_smoothing:
            If set to True, then the generated distribution will be checked
            for gaps. if gaps are found (i.e. 0 for a band where there is data
            later on) then the missing value(s) will be interpolated.

        Returns
        -------
        cost_distribution:
            A `CostDistribution` storing all the cost distribution data
        """
        # Init
        trip_count_col = self.trip_count_col if trip_count_col is None else trip_count_col
        trip_dist_col = self.trip_distance_col if trip_dist_col is None else trip_dist_col
        data = data.copy()

        # Convert distances to output cost
        conv_factor = self.input_cost_units.get_conversion_factor(output_cost_units)
        data[trip_dist_col] *= conv_factor

        # Calculate values for each band
        all_band_trips = list()
        all_band_mean_cost = list()
        for lower, upper in func_utils.pairwise(band_edges):
            # Filter to trips in this band
            mask = (data[trip_dist_col] >= lower) & (data[trip_dist_col] < upper)
            band_data = data[mask].copy()

            # Calculate mean miles and total trips
            band_trips = band_data[trip_count_col].values.sum()
            band_distance = np.sum(
                band_data[trip_count_col].values * band_data[trip_dist_col].values
            )
            if band_distance <= 0:
                band_mean_cost = np.mean([lower, upper])
            else:
                band_mean_cost = band_distance / band_trips

            all_band_trips.append(band_trips)
            all_band_mean_cost.append(band_mean_cost)

        # Convert to numpy arrays
        np_all_band_trips = np.array(all_band_trips)
        np_all_band_mean_cost = np.array(all_band_mean_cost)

        # Check for gaps. If gaps, then interpolate - add this as an option
        if inter_smoothing:
            np_all_band_trips = self._distribution_smoothing_interpolation(
                distribution=np_all_band_trips,
                mean_costs=np_all_band_mean_cost,
                segment_params=segment_params,
            )

        return cost.CostDistribution(
            edges=band_edges,
            band_trips=np_all_band_trips,
            cost_units=output_cost_units,
            band_mean_cost=np_all_band_mean_cost,
        )

    def _filter_nts_data(
        self,
        nts_data: pd.DataFrame,
        segment_params: dict[str, Any],
    ) -> pd.DataFrame:
        """Filters to keep only the segments needed

        Handles all special cases on how to use the `segment_params` to filter
        the `nts_data`.

        Parameters
        ----------
        nts_data:
            A dataframe of the data to filter.

        segment_params:
            The `{col_names: col_value}` pairs to filter `nts_data` with.
            Note that internally, this function handles the following `col_names`
            specially:
            ["trip_origin", "uc", "soc", "ns"]

        Returns
        -------
        filtered_nts_data:
            The original `nts_data` filtered to only include the selected
            segments. Return will be a copied and filtered version of `nts_data`.
        """
        # Init
        to_dict = {"hb": ["hb_to", "hb_from"], "nhb": ["nhb"]}

        # Build the filter
        df_filter = dict()
        for seg_name, seg_val in segment_params.items():

            # Deal with special cases
            if seg_name == "trip_origin":
                df_filter[seg_name] = to_dict[seg_val]

            elif seg_name == "uc":
                uc_enum = nd_core.UserClass(seg_val)
                df_filter[self.purpose_col] = uc_enum.get_purposes()

            elif seg_name in ("soc", "ns"):
                # Don't filter if Nan. Invalid segment
                if not np.isnan(seg_val):
                    df_filter[seg_name] = seg_val

            else:
                # Assume we can filter as normal
                df_filter[seg_name] = seg_val

        # Apply
        return pd_utils.filter_df(df=nts_data, df_filter=df_filter)

    @staticmethod
    def _generate_tld_name(
        segmentation: nd_core.SegmentationLevel,
        segment_params: dict[str, Any],
    ) -> str:
        """Wrapper around `segmentation.generate_file_name()` for consistent filenames"""
        trip_origin = segment_params.get("trip_origin", None)
        return segmentation.generate_file_name(
            segment_params=segment_params,
            trip_origin=trip_origin,
            file_desc=TripLengthDistributionBuilder._distribution_fname_desc,
        )

    @staticmethod
    def _generate_tld_name_template(
        segmentation: nd_core.SegmentationLevel,
        trip_origin: str = None,
    ) -> str:
        """Wrapper around `segmentation.generate_template_file_name()` for consistent filenames"""
        return segmentation.generate_template_file_name(
            trip_origin=trip_origin,
            file_desc=TripLengthDistributionBuilder._distribution_fname_desc,
        )

    def _handle_sample_period(
        self,
        input_dat: pd.DataFrame,
        sample_period: tld_enums.SampleTimePeriods,
    ) -> pd.DataFrame:
        """
        Function to subset whole dataset for time periods
        """
        keep_tps = sample_period.get_time_periods()
        output_dat = input_dat[input_dat[self.tp_col].isin(keep_tps)]
        output_dat = output_dat.reset_index(drop=True)
        return output_dat

    @staticmethod
    def _get_name_and_fname(band_or_seg_name: str) -> tuple[str, str]:
        if ".csv" in band_or_seg_name:
            fname = band_or_seg_name
            name = band_or_seg_name.replace(".csv", "")
        else:
            fname = f"{band_or_seg_name}.csv"
            name = band_or_seg_name
        return fname, name

    @staticmethod
    def _dynamically_pick_log_bins(
        max_value: float,
        n_bin_pow: float = 0.51,
        log_factor: float = 2.2,
        final_val: float = 1500.0,
    ) -> np.ndarray:
        """Dynamically choose the bins based on the maximum possible value

        `n_bins = int(max_value ** n_bin_pow)` Is used to choose the number of bins to use.
        `bins = (np.array(range(2, n_bins)) / n_bins) ** log_factor * max_value` is
        used to determine the bins being used

        Parameters
        ----------
        max_value:
            The maximum value seen in the data, this is used to scale the bins
            appropriately.

        n_bin_pow:
            The power used to determine the number of bins to use, depending
            on the max value. This value should be between 0 and 1.
            `max_value ** n_bin_pow`.

        log_factor:
            The log factor to determine the bin spacing. This should be a
            value greater than 1. Larger numbers mean closer bins

        final_val:
            The final value to append to the end of the bin edges. The second
            to last bin will be less than `max_value`, therefore this number
            needs to be larger than the max value.

        Returns
        -------
        bin_edges:
            A numpy array of bin edges.
        """
        if final_val < max_value:
            raise ValueError("`final_val` is lower than `max_value`.")

        n_bins = int(max_value ** n_bin_pow)
        bins = (np.array(range(2, n_bins + 1)) / n_bins) ** log_factor * max_value
        bins = np.floor(bins)

        # Add the first and last item
        bins = np.insert(bins, 0, 0)
        return np.insert(bins, len(bins), final_val)

    def build_output_path(
        self,
        geo_area: tld_enums.GeoArea,
        trip_filter_type: tld_enums.TripFilter,
        bands_name: str,
        segmentation: nd_core.SegmentationLevel,
        sample_period: tld_enums.SampleTimePeriods,
        cost_units: nd_core.CostUnits,
    ) -> pathlib.Path:
        """Generates the output path for the TLD params

        Parameters
        ----------
        geo_area:
            The geographical area the TLD is constrained to.

        trip_filter_type:
            How to filter the trips into given `geo_area`.

        bands_name:
            The name of the bands being used in the TLD.

        segmentation:
            The segmentation being used in the TLD.

        sample_period:
            Which time periods the TLD is restricted to.

        cost_units:
            The cost units used in the output of the TLDs.

        Returns
        -------
        path_string:
            A string. The full path to a folder where this collection of TLDs
            should be stored.
        """
        # Make sure band and segment names are correct
        _, bands_name = self._get_name_and_fname(bands_name)

        return pathlib.Path(
            os.path.join(
                self.output_folder,
                geo_area.value,
                trip_filter_type.value,
                bands_name,
                segmentation.name,
                sample_period.value,
                cost_units.value,
            )
        )

    def generate_output_paths(
        self,
        geo_area: tld_enums.GeoArea,
        trip_filter_type: tld_enums.TripFilter,
        bands_name: str,
        segmentation: nd_core.SegmentationLevel,
        sample_period: tld_enums.SampleTimePeriods,
        cost_units: nd_core.CostUnits,
    ) -> tuple[pathlib.Path, list[str]]:
        """Generates all the file output paths for the TLD params

        Parameters
        ----------
        geo_area:
            The geographical area the TLD is constrained to.

        trip_filter_type:
            How to filter the trips into given `geo_area`.

        bands_name:
            The name of the bands being used in the TLD.

        segmentation:
            The segmentation to generate a group of TLDs for

        sample_period:
            Which time periods the TLD is restricted to.

        cost_units:
            The cost units used in the output of the TLDs.

        Returns
        -------
        path_string:
            A string. The full path to a folder where this collection of TLDs
            should be stored.
        """
        # Generate the directory all the files are output
        base_path = self.build_output_path(
            geo_area=geo_area,
            trip_filter_type=trip_filter_type,
            bands_name=bands_name,
            segmentation=segmentation,
            sample_period=sample_period,
            cost_units=cost_units,
        )

        # Generate the filenames
        fnames = list()
        for segment_params in segmentation:
            tld_name = self._generate_tld_name(segmentation, segment_params)
            fnames.append(f"{tld_name}.csv")

        return base_path, fnames

    def copy_across_tps(
        self,
        geo_area: tld_enums.GeoArea,
        trip_filter_type: tld_enums.TripFilter,
        bands_name: str,
        segmentation: nd_core.SegmentationLevel,
        sample_period: tld_enums.SampleTimePeriods,
        cost_units: nd_core.CostUnits,
        process_count: int = nd_consts.PROCESS_COUNT,
    ) -> None:
        """Copies generated TLDs across time period segments

        This is useful when segments have had to be aggregated due to sample
        sizes, but other models and tools expect the inputs to be at
        a time period segmentation.

        Parameters
        ----------
        geo_area:
            The geographical area the TLD is constrained to.

        trip_filter_type:
            How to filter the trips into given `geo_area`.

        bands_name:
            The name of the bands being used in the TLD.

        segmentation:
            The segmentation being used in the TLD.

        sample_period:
            Which time periods the TLD is restricted to.

        cost_units:
            The cost units used in the output of the TLDs.

        process_count:
            The number of processes to use when copying the data over.
            0 - use no multiprocessing, run as a loop.
            +ve value - the number of processes to use.
            -ve value - the number of processes less than the cpu count to use.

        Returns
        -------
        None
        """
        # Generate the needed paths
        base_path, fnames = self.generate_output_paths(
            geo_area=geo_area,
            trip_filter_type=trip_filter_type,
            bands_name=bands_name,
            segmentation=segmentation,
            sample_period=sample_period,
            cost_units=cost_units,
        )

        output_path = base_path / "tp_copied"
        file_ops.create_folder(output_path)

        # Copy across time periods
        copy_files = list()
        for fname in fnames:
            for tp in sample_period.get_time_periods():
                out_name = fname.replace(".csv", f"_tp{tp}.csv")

                # Tuple of src, dst files
                copy_files.append(
                    (
                        base_path / fname,
                        output_path / out_name,
                    )
                )

        file_ops.copy_and_rename_files(files=copy_files, process_count=process_count)

        # Write out a log of what happened
        run_log_str = self.generate_run_log(
            geo_area=geo_area,
            trip_filter_type=trip_filter_type,
            bands_name=bands_name,
            segmentation=segmentation,
            sample_period=sample_period,
            cost_units=cost_units,
            tp_copy=True,
        )

        output_path = output_path / self._input_params_fname
        with open(output_path, "w") as f:
            f.write(run_log_str)

    def copy_tlds(
        self,
        copy_definition_name: str,
        geo_area: tld_enums.GeoArea,
        trip_filter_type: tld_enums.TripFilter,
        bands_name: str,
        segmentation: nd_core.SegmentationLevel,
        sample_period: tld_enums.SampleTimePeriods,
        cost_units: nd_core.CostUnits,
        process_count: int = nd_consts.PROCESS_COUNT,
    ) -> None:
        """Copies generated TLDs across multiple segments

        This is useful when segments have had to be aggregated due to sample
        sizes, but other models and tools expect the inputs to be at
        the original segmentation.

        Parameters
        ----------
        copy_definition_name:
            The name of the copy definition to use to copy the files

        geo_area:
            The geographical area the TLD is constrained to.

        trip_filter_type:
            How to filter the trips into given `geo_area`.

        bands_name:
            The name of the bands being used in the TLD.

        segmentation:
            The segmentation being used in the TLD.

        sample_period:
            Which time periods the TLD is restricted to.

        cost_units:
            The cost units used in the output of the TLDs.

        process_count:
            The number of processes to use when copying the data over.
            0 - use no multiprocessing, run as a loop.
            +ve value - the number of processes to use.
            -ve value - the number of processes less than the cpu count to use.

        Returns
        -------
        None
        """
        # Read in the copy definition
        fname, copy_definition_name = self._get_name_and_fname(copy_definition_name)
        copy_def = pd.read_csv(os.path.join(self.segment_copy_definition_dir, fname))

        # Generate the directory all the files are output
        base_path = self.build_output_path(
            geo_area=geo_area,
            trip_filter_type=trip_filter_type,
            bands_name=bands_name,
            segmentation=segmentation,
            sample_period=sample_period,
            cost_units=cost_units,
        )

        # Build the output path
        output_path = base_path / copy_definition_name
        file_ops.create_folder(output_path)

        file_ops.copy_defined_files(
            copy_definition=copy_def,
            src_dir=base_path,
            dst_dir=output_path,
            process_count=process_count,
        )

        # Write out a log of what happened
        run_log_str = self.generate_run_log(
            geo_area=geo_area,
            trip_filter_type=trip_filter_type,
            bands_name=bands_name,
            segmentation=segmentation,
            sample_period=sample_period,
            cost_units=cost_units,
            copy_definition_name=copy_definition_name,
        )

        output_path = output_path / self._input_params_fname
        with open(output_path, "w") as f:
            f.write(run_log_str)

    def build_aggregated_tlds(
        self,
        tld_data: pd.DataFrame,
        bands: pd.DataFrame,
        segmentation: nd_core.SegmentationLevel,
        aggregated_exclude_segments: list[str],
        cost_units: nd_core.CostUnits,
        sample_threshold: int,
        trip_count_col: str = None,
        trip_dist_col: str = None,
        inter_smoothing: bool = False,
    ) -> tuple[dict[str, cost.CostDistribution], dict[str, str], pd.DataFrame, pd.DataFrame]:
        """Build a set of  aggregated trip length distributions

        similar functionality to `self.build_tld`, but generates an aggregated
        TLD instead based off of `segmentation` and `aggregated_exclude_segments`.
        The aggregated TLD is defined by taking the `segmentation` and
        removing the segments defined in `aggregated_exclude_segments`.
        That is, if `segmentation` contains the segments ['p', 'm', 'soc] and
        `aggregated_exclude_segments` is set to `soc` then the `soc` segment will
        be ignored when generating the aggregated TLDs.

        Parameters
        ----------
        tld_data:
            Dataframe of pre-processed trip length distribution data

        bands: pd.DataFrame:
            Dataframe of bands with headings lower, upper

        segmentation:
            The segmentation to generate a group of TLDs for

        aggregated_exclude_segments:
            A list of the segment names to exclude from `segmentation` when
            generated an aggregated segmentation.

        cost_units:
            The cost units to use in the output. The data will be multiplied
            by a constant to convert.

        sample_threshold: int = 10:
            Sample size below which skip allocation to bands and fail out

        trip_count_col:
            The name of the column in `tld_data` containing the count of trips
            being made.

        trip_dist_col:
            The name of the column in `tld_data` containing the distance of trips
            being made. This should be in `self.input_cost_units` units.

        inter_smoothing:
            If set to True, then the generated distribution will be checked
            for gaps. if gaps are found (i.e. 0 for a band where there is data
            later on) then the missing value(s) will be interpolated.


        Returns
        ----------
        agg_name_to_distribution:
            A dictionary with {tld_name: CostDistribution}

        name_to_aggregated_name:
            A dictionary mapping each segment name to an aggregated segment
            name. The aggregated segment names are used in
            `agg_name_to_distribution` to access the distributions.

        sample_size_log:
            input segments with reported number of records and status

        full_export:
            A pandas dataframe containing all the generated TLDs alongside
            their segment_params
        """
        # Init
        trip_count_col = self.trip_count_col if trip_count_col is None else trip_count_col
        trip_dist_col = self.trip_distance_col if trip_dist_col is None else trip_dist_col

        # Calculate band edges
        min_bounds = bands["lower"].values
        max_bounds = bands["upper"].values
        band_edges = np.array([min_bounds[0]] + max_bounds.tolist())

        # Generate a TLD for each segment
        name_to_distribution = dict()
        name_to_agg_name = dict()
        sample_size_log = list()
        full_export = list()
        for segment_params in segmentation:
            # generate template name
            tld_name_template = self._generate_tld_name_template(
                segmentation,
                trip_origin=segment_params.get("trip_origin", None),
            )

            # Generate aggregated names
            agg_segment_params = segment_params.copy()
            agg_segment_types = segmentation.segment_types.copy()
            for exclude in aggregated_exclude_segments:
                agg_segment_params.pop(exclude, None)
                agg_segment_types.pop(exclude, None)

            agg_naming_order = func_utils.list_safe_remove(
                segmentation.naming_order, aggregated_exclude_segments
            )

            # Build the TLD
            tld, tld_name, log_line = self._build_tld_from_segment_params(
                segment_params=agg_segment_params,
                segment_naming_order=agg_naming_order,
                raw_tld_data=tld_data,
                band_edges=band_edges,
                cost_units=cost_units,
                sample_threshold=sample_threshold,
                inter_smoothing=inter_smoothing,
                trip_count_col=trip_count_col,
                trip_dist_col=trip_dist_col,
                tld_name_template=tld_name_template,
                segment_types=agg_segment_types,
            )

            # Build the original segment name and map it
            segment_str = nd_core.SegmentationLevel.generate_template_segment_str(
                naming_order=segmentation.naming_order,
                segment_params=segment_params,
                segment_types=segmentation.segment_types,
            )
            orig_tld_name = tld_name_template.format(segment_params=segment_str)
            name_to_agg_name[orig_tld_name] = tld_name

            # Only store if we don't already have it stored
            if tld_name not in name_to_distribution:
                # Store the generated data
                sample_size_log.append(log_line)
                name_to_distribution.update({tld_name: tld})
                full_export.append(tld.to_df(additional_cols=agg_segment_params))

        # Consolidate reports
        full_export = pd.concat(full_export, ignore_index=True)
        sample_size_log = pd.DataFrame(sample_size_log)

        return name_to_distribution, name_to_agg_name, sample_size_log, full_export

    def _build_tld_from_segment_params(
        self,
        segment_params: dict[str, Any],
        segment_naming_order: list[str],
        raw_tld_data: pd.DataFrame,
        cost_units: nd_core.CostUnits,
        sample_threshold: int,
        inter_smoothing: bool,
        trip_count_col: str,
        trip_dist_col: str,
        tld_name_template: str,
        band_edges: Optional[np.ndarray] = None,
        segment_types: dict[str, type] = None,
    ) -> tuple[cost.CostDistribution, str, dict[str, str]]:
        """Build a single trip length distribution

        Uses the given `raw_tld_data`, filters to the `segment_params` and
        builds a single TLD

        Parameters
        ----------
        segment_params:
            A dictionary defining the segmentation to filter to.
            {segment_name: segment_value}

        segment_naming_order:
            The order to name to segments in the generated name. Passed to
            `SegmentationLevel.generate_template_segment_str()`

        raw_tld_data:
            The data to filter (using `segment_params`) to generate the
            TLD.

        band_edges:
            The edges to use for each band in the distribution. E.g.
            `[1, 2, 3]` defines 2 bands: 1->2 and 2->3. if None, the bands
            are dynamically chosen based on the maximum value.

        cost_units:
            The cost units to use in the output. The data will be multiplied
            by a constant to convert.

        sample_threshold:
            Sample size below which skip allocation to bands and fail out

        inter_smoothing:
            If set to True, then the generated distribution will be checked
            for gaps. if gaps are found (i.e. 0 for a band where there is data
            later on) then the missing value(s) will be interpolated.

        trip_count_col:
            The name of the column in `raw_tld_data` containing the count of trips
            being made.

        trip_dist_col:
            The name of the column in `raw_tld_data` containing the distance of trips
            being made. This should be in `self.input_cost_units` units.

        tld_name_template:
            A template name to use when generating the TLD name. This will
            be formatting with `tld_name_template.format(segment_params=segment_str)`

        segment_types:
            A dictionary of `{segment_name: segment_type}`. Must have all
            segment names in that `segment_params` does, if defined.
            Passed into `SegmentationLevel.generate_template_segment_str()`

        Returns
        -------
        trip_length_distribution:
            A CostDistribution object containing the generated trip length
            distribution.

        trip_length_distribution_name:
            The generated name of the `trip_length_distribution`.

        log_dict:
            A dictionary of names and log values, logging the performance of
            the trip_length_distribution generation
        """

        # Build the band edges if not given
        if band_edges is None:
            conv_factor = self.input_cost_units.get_conversion_factor(cost_units)
            band_edges = self._dynamically_pick_log_bins(
                max_value=np.max(raw_tld_data[self.trip_distance_col]) * conv_factor
            )

        # Filter to data for this segment
        data_subset = raw_tld_data.copy()
        data_subset = self._filter_nts_data(data_subset, segment_params)
        sample_size = data_subset[trip_count_col].values.sum()

        # Build the sample size log
        log_line = segment_params.copy()
        log_line["records"] = sample_size

        # If sample size is too small, set warning to be dealt with later
        if sample_size < sample_threshold:
            log_line["status"] = "Failed"
            LOG.warning(
                "Not enough data was returned to create a TLD for segment "
                "%s. Lower limit set to %s, "
                "but only %.2f were found. No TLD will be generated."
                % (segment_params, sample_threshold, sample_size)
            )

            tld = cost.CostDistribution.build_empty(edges=band_edges, cost_units=cost_units)

        else:
            # Build into a cost distribution
            log_line["status"] = "Passed"
            tld = self._build_cost_distribution(
                data=data_subset,
                band_edges=band_edges,
                output_cost_units=cost_units,
                trip_count_col=trip_count_col,
                trip_dist_col=trip_dist_col,
                inter_smoothing=inter_smoothing,
                segment_params=segment_params,
            )

        segment_str = nd_core.SegmentationLevel.generate_template_segment_str(
            naming_order=segment_naming_order,
            segment_params=segment_params,
            segment_types=segment_types,
        )
        tld_name = tld_name_template.format(segment_params=segment_str)
        log_line["name"] = tld_name

        return tld, tld_name, log_line

    def build_tld(
        self,
        tld_data: pd.DataFrame,
        segmentation: nd_core.SegmentationLevel,
        cost_units: nd_core.CostUnits,
        sample_threshold: int,
        bands: Optional[pd.DataFrame] = None,
        trip_count_col: str = None,
        trip_dist_col: str = None,
        inter_smoothing: bool = False,
    ) -> tuple[dict[str, cost.CostDistribution], pd.DataFrame, pd.DataFrame]:
        """
        Build a set of trip length distributions

        Parameters
        ----------
        tld_data:
            Dataframe of pre-processed trip length distribution data

        bands: pd.DataFrame:
            Dataframe of bands with headings lower, upper. If None, then the
            bands will be dynamically chosen

        segmentation:
            The segmentation to generate a group of TLDs for

        cost_units:
            The cost units to use in the output. The data will be multiplied
            by a constant to convert.

        sample_threshold: int = 10:
            Sample size below which skip allocation to bands and fail out

        trip_count_col:
            The name of the column in `tld_data` containing the count of trips
            being made.

        trip_dist_col:
            The name of the column in `tld_data` containing the distance of trips
            being made. This should be in `self.input_cost_units` units.

        inter_smoothing:
            If set to True, then the generated distribution will be checked
            for gaps. if gaps are found (i.e. 0 for a band where there is data
            later on) then the missing value(s) will be interpolated.

        Returns
        ----------
        name_to_distribution:
            A dictionary with {tld_name: CostDistribution}

        sample_size_log:
            input segments with reported number of records and status

        full_export:
            A pandas dataframe containing all the generated TLDs alongside
            their segment_params
        """
        # Init
        trip_count_col = self.trip_count_col if trip_count_col is None else trip_count_col
        trip_dist_col = self.trip_distance_col if trip_dist_col is None else trip_dist_col

        # Calculate band edges
        if bands is not None:
            min_bounds = bands["lower"].values
            max_bounds = bands["upper"].values
            band_edges = np.array([min_bounds[0]] + max_bounds.tolist())
        else:
            band_edges = None

        # Generate a TLD for each segment
        name_to_distribution = dict()
        sample_size_log = list()
        full_export = list()
        for segment_params in segmentation:
            # generate template name
            tld_name_template = self._generate_tld_name_template(
                segmentation,
                trip_origin=segment_params.get("trip_origin", None),
            )

            # Build the TLD
            tld, tld_name, log_line = self._build_tld_from_segment_params(
                segment_params=segment_params,
                segment_naming_order=segmentation.naming_order,
                raw_tld_data=tld_data,
                band_edges=band_edges,
                cost_units=cost_units,
                sample_threshold=sample_threshold,
                inter_smoothing=inter_smoothing,
                trip_count_col=trip_count_col,
                trip_dist_col=trip_dist_col,
                tld_name_template=tld_name_template,
                segment_types=segmentation.segment_types,
            )

            # Store the generated data
            sample_size_log.append(log_line)
            name_to_distribution.update({tld_name: tld})
            full_export.append(tld.to_df(additional_cols=segment_params))

        # Consolidate reports
        full_export = pd.concat(full_export, ignore_index=True)
        sample_size_log = pd.DataFrame(sample_size_log)

        return name_to_distribution, sample_size_log, full_export

    def generate_run_log(
        self,
        geo_area: tld_enums.GeoArea,
        trip_filter_type: tld_enums.TripFilter,
        bands_name: str,
        segmentation: nd_core.SegmentationLevel,
        sample_period: tld_enums.SampleTimePeriods,
        cost_units: nd_core.CostUnits,
        copy_definition_name: str = None,
        tp_copy: bool = False,
    ) -> str:
        """Writes out a file of the params used to run

        Parameters
        ----------
        geo_area:
            The geographical area the TLD is constrained to.

        trip_filter_type:
            How to filter the trips into given `geo_area`.

        bands_name:
            The name of the bands being used in the TLD.

        segmentation:
            The segmentation being used in the TLD.

        sample_period:
            Which time periods the TLD is restricted to.

        cost_units:
            The cost units used in the output of the TLDs.

        copy_definition_name:
            Used when running `self.copy_tlds()` adds a line detailing the
            copy

        tp_copy:
            Used when running `self.copy_across_tps()`. Adds a line detailing
            the copy.

        Returns
        -------
        None
        """
        # Generate the lines of the log
        lines = list()
        lines.append(f"{str_utils.title_padding('TLD Tool Input Params and Run Log')}\n")
        lines.append(f"Code Version: {nd.__version__}")
        lines.append(f"Ran at: {timing.get_datetime()}\n")
        lines.append(f"Using the Classified Build from:\n{self.tlb_import_path}\n")
        lines.append("Input Params")
        lines.append("-" * 12)
        lines.append(f"Geo Area: {geo_area.value}")
        lines.append(f"Trip Filter Type: {trip_filter_type.value}")
        lines.append(f"Bands Name: {bands_name}")
        lines.append(f"Segmentation Name: {segmentation.name}")
        lines.append(f"Sample Time Periods: {sample_period.value}")
        lines.append(f"Output Cost Units: {cost_units.value}")

        if copy_definition_name is not None:
            fname, copy_definition_name = self._get_name_and_fname(copy_definition_name)
            copy_def_path = os.path.join(self.segment_copy_definition_dir, fname)

            lines.append(
                f"\nFiles were then copied to {copy_definition_name} segmentation "
                "using the definition:"
            )
            lines.append(f"{copy_def_path}")

        if tp_copy:
            lines.append("\nFiles were then copied across all time periods")

        return "\n".join(lines)

    def tld_generator(
        self,
        bands_name: str,
        segmentation: nd_core.SegmentationLevel,
        geo_area: tld_enums.GeoArea,
        sample_period: tld_enums.SampleTimePeriods,
        trip_filter_type: tld_enums.TripFilter,
        cost_units: nd_core.CostUnits = nd_core.CostUnits.MILES,
        aggregated_exclude_segments: Optional[list[str]] = None,
        inter_smoothing: bool = False,
        check_sample_size: int = 400,
        min_sample_size: int = 40,
    ) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
        """Generate a trip length distribution

        Checks that the sample sizes are reasonable (as defined by
        `check_sample_size` and `min_sample_size`) and will default to an
        aggregated TLD if the checks don't pass. The aggregated TLD is defined
        by taking the `segmentation` and removing the segments defined in
        `aggregated_exclude_segments`.
        That is, if `segmentation` contains the segments ['p', 'm', 'soc] and
        `aggregated_exclude_segments` is set to `soc` then the `soc` segment will
        be ignored when generating the aggregated TLDs.

        Parameters
        ----------
        bands_name:
            Name of the bands to use, as named in self.bands_definition_dir
            Set to "dynamic" for dynamic, exponential bands.
            # TODO(BT): Make this an object

        segmentation:
            The segmentation to generate a group of TLDs for

        aggregated_exclude_segments:
            A list of the segment names to exclude from `segmentation` when
            generating an aggregated segmentation.

        geo_area:
            The geographical area to limit the generated TLD to

        sample_period:
            Time period filter for the generated TLD

        trip_filter_type:
            How to filter the trips into the geographical area.
            TRIP_OD will filter the origin and destination of trips into the
            defined `gep_area`.
            TRIP_O will only filter the origins
            TRIP_O will only filter the destinations

        cost_units:
            The cost units to use in the output. The data will be multiplied
            by a constant to convert.

        inter_smoothing:
            If set to True, then the generated distributions will be checked
            for gaps. if gaps are found (i.e. 0 for a band where there is data
            later on) then the missing value(s) will be interpolated.

        check_sample_size:
            When the sample size is below this value, additional checks are
            carried out to ensure the generated TLD is reasonable.
            Will ensure that values continue to decrease in greater bands.
            If not, the aggregated TLD is used.

        min_sample_size:
            The minimum acceptable sample size. If the number of trips left
            to generate a TLD is smaller than this number then the aggregated
            TLD will be used instead.

        Returns
        ----------
        name_to_distribution: dict
            A dictionary with {description of tld: tld DataFrame}

        full_export: pd.DataFrame
            A compiled, concatenated version of the DataFrames in name_to_distribution

        Future Improvements
        ----------
        Add more functionality for time period handling.
        Add better error control and type limiting for inputs.
        """
        # TODO(BT): Pass in - NEED to rewrite class first!
        self.purpose_col = "p"
        self.tp_col = "tp"
        self.trip_origin_col = "trip_origin"
        nts_to_segment_names = {
            "main_mode": "m",
            "p": self.purpose_col,
            "gender": "g",
            "soc": "soc",
            "ns": "ns",
            "start_time": self.tp_col,
            "trip_direction": self.trip_origin_col,
        }

        # Rename the input data to expected column names
        nts_data = self.nts_import.rename(columns=nts_to_segment_names)

        # Build output path
        tld_out_path = self.build_output_path(
            geo_area=geo_area,
            trip_filter_type=trip_filter_type,
            bands_name=bands_name,
            segmentation=segmentation,
            sample_period=sample_period,
            cost_units=cost_units,
        )
        graph_out_path = tld_out_path / "graphs"
        file_ops.create_folder(graph_out_path, verbose_create=False)
        LOG.info(f"Generating TLD at: {tld_out_path}...")

        # Try read in the bands and segmentation
        if bands_name.strip().lower() == "dynamic":
            bands = None
        else:
            fname, bands_name = self._get_name_and_fname(bands_name)
            bands = pd.read_csv(os.path.join(self.bands_definition_dir, fname))

        # Map categories not classified in classified build
        # Car availability
        # TODO: This should be handled upstream, in inputs
        nts_data = self._map_dict(
            output_dat=nts_data,
            map_dict=self._household_type_to_ca,
            key="hh_type",
        )

        # Filter data down to the time periods and geo area
        nts_data = self._handle_sample_period(nts_data, sample_period=sample_period)
        nts_data = self._apply_geo_filter(
            nts_data,
            geo_area=geo_area,
            trip_filter_type=trip_filter_type,
        )

        tld_kwargs = {
            "tld_data": nts_data,
            "bands": bands,
            "segmentation": segmentation,
            "cost_units": cost_units,
            "sample_threshold": min_sample_size,
            "inter_smoothing": inter_smoothing,
        }

        log_path = tld_out_path / self._running_log_fname
        with nd.logging.TemporaryLogFile(LOG, log_path):
            # Build tld dictionary, return a proper name for the distributions
            name_to_distribution, sample_size_log, full_export = self.build_tld(**tld_kwargs)

            # build the aggregated TLDs, and link back to above
            if aggregated_exclude_segments is not None:
                agg_tld_data = self.build_aggregated_tlds(
                    aggregated_exclude_segments=aggregated_exclude_segments,
                    **tld_kwargs
                )
            else:
                agg_tld_data = (None, None, None, None)
            agg_name_to_dist, name_to_agg_name, agg_sample_size_log, agg_full_export = agg_tld_data

            # Are any of the sample sizes small enough for further checks?
            for name, dist in name_to_distribution.items():
                if dist.sample_size > check_sample_size:
                    continue

                LOG.info(f"{name} has a sample size less than {check_sample_size}.")
                if self._passing_tld_checks(dist):
                    LOG.info(f"{name} looks OK. Leaving as is.")
                else:
                    if agg_name_to_dist is not None:
                        LOG.info(
                            f"{name} did not pass further checks. Reverting to aggregated TLD."
                        )
                        # Adjust the aggregated dist to sample size of this segment
                        mask = sample_size_log['name'] == name
                        sample_size = sample_size_log[mask]["records"].squeeze()

                        agg_dist = agg_name_to_dist[name_to_agg_name[name]]
                        agg_dist.sample_size = sample_size
                        name_to_distribution[name] = agg_dist
                    else:
                        LOG.warning(
                            f"{name} did not pass further checks, and there is no "
                            f"aggregated TLD to fall back to. Use {name} with "
                            f"caution! Try setting `aggregated_exclude_segments` "
                            f"to generate aggregated TLDs."
                        )

        # ## WRITE THINGS OUT ## #
        # Write the run log
        run_log_str = self.generate_run_log(
            geo_area=geo_area,
            trip_filter_type=trip_filter_type,
            bands_name=bands_name,
            segmentation=segmentation,
            sample_period=sample_period,
            cost_units=cost_units,
        )

        output_path = tld_out_path / self._input_params_fname
        with open(output_path, "w") as f:
            f.write(run_log_str)

        # Write sample_size_logs
        seg_report_fname = self._seg_report_fname.format(seg_name=segmentation.name)
        report_path = tld_out_path / seg_report_fname
        file_ops.safe_dataframe_to_csv(sample_size_log, report_path, index=False)

        if agg_sample_size_log is not None:
            agg_seg_report_fname = self._agg_seg_report_fname.format(seg_name=segmentation.name)
            report_path = tld_out_path / agg_seg_report_fname
            file_ops.safe_dataframe_to_csv(agg_sample_size_log, report_path, index=False)

        # Write final compiled tlds
        full_export_path = tld_out_path / self._full_export_fname
        file_ops.safe_dataframe_to_csv(full_export, full_export_path, index=False)

        if agg_full_export is not None:
            full_export_path = tld_out_path / self._agg_full_export_fname
            file_ops.safe_dataframe_to_csv(agg_full_export, full_export_path, index=False)

        # Write individual tlds
        for name, tld in name_to_distribution.items():
            # Csv
            path = tld_out_path / f"{name}.csv"
            tld.to_csv(path)

            # Graph
            path = graph_out_path / f"{name}.png"
            tld.to_graph(path, band_shares=True, plot_title=name)

        return name_to_distribution, full_export
