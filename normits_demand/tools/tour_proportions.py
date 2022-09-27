# -*- coding: utf-8 -*-
"""
Created on: 06/05/2022
Updated on:

Original author: Ben Taylor
Last update made by:
Other updates made by:

File purpose:

"""
# Built-Ins
import itertools
import dataclasses
import os

from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

# Third Party
import pandas as pd
import numpy as np

from caf.toolkit import pandas_utils as pd_utils

# Local Imports
from normits_demand import logging as nd_log
from normits_demand import core as nd_core
from normits_demand import types as nd_types
from normits_demand.utils import general as du
from normits_demand.utils import file_ops


LOG = nd_log.get_logger(__name__)

@dataclasses.dataclass
class ReturnHomeFactors:
    """Storage of return home factor data

    Can only handle return home factors that are purpose segmented currently
    """

    dataframe: pd.DataFrame
    time_periods: List[int]
    time_fh_col: str
    time_th_col: str
    purpose_col: str
    factor_col: str

    def __post_init__(self):
        # Init
        needed_cols = [self.purpose_col, self.time_th_col, self.time_fh_col, self.factor_col]
        all_df_cols = self.dataframe.columns.to_list()

        # Make sure self.dataframe actually contains the needed columns
        missing_cols = set(needed_cols) - set(all_df_cols)
        if len(missing_cols) > 0:
            raise ValueError(
                "The given dataframe doesn't contain all the columns it "
                "should. The following given columns names are missing: "
                f"'{missing_cols}'"
            )

        # Drop unneeded tps
        # fmt: off
        mask = (
            self.dataframe[self.time_fh_col].isin(self.time_periods)
            & self.dataframe[self.time_th_col].isin(self.time_periods)
        )
        # fmt: on
        self.dataframe = self.dataframe[mask].copy()

    def get_factors(
        self,
        purpose_col: str = None,
        time_fh_col: str = None,
    ) -> pd.DataFrame:
        """Get a square matrix of return home factors

        filters `self.dataframe` down to only include the time periods
        given in `time_periods` and returns a wide matrix where the columns
        are the `self.time_th_col` values

        Parameters
        ----------
        purpose_col:
            The name to give to the purpose col in the return. If left as
            None, then `self.purpose_col` is used.

        time_fh_col
            The name to give to the time from home col in the return. If left
            as None, then `self.time_fh_col` is used.

        Returns
        -------

        """
        # Rename columns if needed
        purpose_col = self.purpose_col if purpose_col is None else purpose_col
        time_fh_col = self.time_fh_col if time_fh_col is None else time_fh_col

        dataframe = self.dataframe.rename(
            columns={
                self.purpose_col: purpose_col,
                self.time_fh_col: time_fh_col,
            }
        )

        # Return as square matrix
        return dataframe.pivot(
            index=[purpose_col, time_fh_col],
            columns=self.time_th_col,
            values=self.factor_col,
        ).reset_index()


class TimePeriodSplits:
    """Calculation and storage of the zonal time period split data"""

    # Generated attributes
    df: pd.DataFrame = dataclasses.field(init=False)
    output_seg: nd_core.SegmentationLevel = dataclasses.field(init=False)

    # Default col names from DVec Segmentations
    purpose_col = "p"
    mode_col = "m"
    ca_col = "ca"
    tp_col = "tp"
    val_col = "val"

    # Class constants
    factor_col = "factor"
    valid_segment_cols = [purpose_col, mode_col, ca_col, tp_col]

    def __init__(
        self,
        mode: nd_core.Mode,
        zoning_system: nd_core.ZoningSystem,
        dvec: nd_core.DVector,
        time_periods: List[int],
        translation_weighting: str = "population",
    ):
        # Assign attributes
        self.mode = mode
        self.zoning_system = zoning_system
        self.time_periods = time_periods
        self.dataframe = self._generate_dataframe(
            dvec=dvec,
            translation_weighting=translation_weighting,
        )

    def _generate_dataframe(
        self, dvec: nd_core.DVector, translation_weighting: str
    ) -> pd.DataFrame:
        # Init
        zone_col = self.zoning_system.col_name
        aggregate_seg, output_seg = self._segmentation_lookup()
        self.output_seg = output_seg

        # Convert dvec into a dataframe - keeping only needed data
        dvec = dvec.aggregate(aggregate_seg)
        dvec = dvec.translate_zoning(self.zoning_system, weighting=translation_weighting)
        df = dvec.to_df()

        # Filter down by mode
        mask = df[self.mode_col].isin(self.mode.get_mode_values())
        df = df[mask].reset_index(drop=True)

        # Filter to the time periods we want
        mask = df[self.tp_col].isin(self.time_periods)
        df = df[mask].reset_index(drop=True)

        # Adjust factors back to 1
        group_cols = du.list_safe_remove(df.columns.to_list(), [self.tp_col, self.val_col])
        df["sum"] = df.groupby(group_cols)[self.val_col].transform("sum")
        df[self.val_col] /= df["sum"]

        # If val is NaN, assume even split
        df[self.val_col] = df[self.val_col].fillna(1/len(self.time_periods))
        df = df.drop(columns="sum")

        # Double check that we have valid columns names
        self.segment_cols = du.list_safe_remove(
            lst=df.columns.to_list(),
            remove=[zone_col, self.val_col],
        )

        for col_name in self.segment_cols:
            if col_name not in self.valid_segment_cols:
                raise ValueError(
                    "Invalid column name in the generated "
                    "TimePeriodSplits.dataframe.\n"
                    f"Expected one of: {self.valid_segment_cols}\n"
                    f"Got: {col_name}"
                )

        # Rename the value column to something more obvious
        return df.rename(columns={self.val_col: self.factor_col})

    def _segmentation_lookup(
        self,
    ) -> Tuple[nd_core.SegmentationLevel, nd_core.SegmentationLevel]:
        """Lookup for segmentations to be used when calculating factors"""
        if self.mode == nd_core.Mode.TRAIN:
            aggregate_seg = nd_core.get_segmentation_level("hb_p_m_ca_tp_week")
            output_seg = nd_core.get_segmentation_level("hb_p_m_ca_rail")
        elif self.mode == nd_core.Mode.CAR:
            aggregate_seg = nd_core.get_segmentation_level("hb_p_m_tp_week")
            output_seg = nd_core.get_segmentation_level("hb_p_m_car")
        elif self.mode in [nd_core.Mode.WALK, nd_core.Mode.CYCLE, nd_core.Mode.BUS]:
            aggregate_seg = nd_core.get_segmentation_level("hb_p_m_tp_week")
            output_seg = nd_core.get_segmentation_level(f"hb_p_m_{self.mode.value}")
        else:
            raise NotImplementedError(f"segmentation not implemented for mode = {self.mode}")

        return aggregate_seg, output_seg


class TourProportions:
    """Storage and building of custom tour proportion formats

    Attributes
    ----------
    zoning_system:
    segmentation:
    zone_col:
    tp_col:
    time_periods:
    return_home_cols:
    col_order:
    """

    def __init__(
        self,
        tour_props_df: pd.DataFrame,
        zoning_system: nd_core.ZoningSystem,
        segmentation: nd_core.SegmentationLevel,
        zone_col: str,
        segment_cols: List[str],
        tp_col: str,
        time_periods: List[int],
    ):
        # Assign attributes
        self.zoning_system = zoning_system
        self.segmentation = segmentation
        self.zone_col = zone_col
        self.segment_cols = du.list_safe_remove(segment_cols, [tp_col])
        self.tp_col = tp_col
        self.time_periods = time_periods
        self.return_home_cols = [str(x) for x in time_periods]
        self.col_order = [zone_col] + segment_cols + [tp_col] + self.return_home_cols

        self.tour_props_df = self._validate_tour_props_df(tour_props_df)

    def _validate_tour_props_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure df is valid for self.tour_props_df"""
        # Ensure cols have string names
        df.columns = df.columns.astype(str)

        # Check for missing cols
        missing_cols = set(self.col_order) - set(df.columns.to_list())
        if len(missing_cols) > 0:
            raise ValueError(
                "Columns missing from tour_props_df.\n"
                f"Expected: {self.col_order}\n"
                f"Missing: {missing_cols}"
            )

        # Ensure we have all the time periods we need
        if len(set(self.time_periods) - set(df[self.tp_col].unique())) > 0:
            raise ValueError(
                "tour_props_df does not contain all of the time periods "
                "given. This will result in incorrect tour proportions "
                "being generated."
            )

        # Warn if it looks like segments are missing
        n_expected_segments = (
            len(self.zoning_system) * len(self.segmentation) * len(self.time_periods)
        )
        n_missing_segments = n_expected_segments - len(df)
        if n_missing_segments > 0:
            LOG.warning("%s segments missing from tour_props_df", n_missing_segments)

        return df

    def get_tour_proportions_dictionary(
        self,
        segment_params: Dict[str, Any],
        out_dtype: type = None,
    ) -> nd_types.TimePeriodNestedDict:
        """Generate the tour proportion nested dictionary

        Uses internal data to generate a nested dictionary describing the tour
        proportions. The first key is the from-home time period. The second
        key is the to-home time period. The value is a
        `(len(self.zoning_system), len(self.zoning_system))` size array.

        Parameters
        ----------
        segment_params:
            A segment_params dictionary where the keys are the names of
            segments, and the values are the values of the segments.
            This is the format returned when iterating over a
            nd.core.SegmentationLevel

        out_dtype:
            The datatype to cast the matrices to before writing to disk.
            If left as None, no casting is done.

        Returns
        -------
        tour_proportions:
            A nested dictionary of tour proportions.
            The first key is the from-home time period. The second
            key is the to-home time period. The value is a
            `(len(self.zoning_system), len(self.zoning_system))` size array.
        """
        if not self.segmentation.is_valid_segment_params(segment_params):
            raise ValueError(
                f"segment_params are not valid for segmentation " f"{self.segmentation.name}"
            )

        # Filter down to the segment params we care about
        df = pd_utils.filter_df(df=self.tour_props_df, df_filter=segment_params)
        df = df.drop(columns=list(segment_params.keys()))

        # Generate the values for each time period
        return_dict = dict.fromkeys(self.time_periods)
        return_dict = {k: dict.fromkeys(self.time_periods) for k in return_dict}
        for tp_from, tp_to in itertools.product(self.time_periods, self.time_periods):
            # All columns are the same as the return home factors are not
            # zonally segmented - so we just need to generate one column
            # and copy it across all
            col = df[df[self.tp_col] == tp_from].copy()
            col = col.sort_values(by=self.zone_col)
            col = col[str(tp_to)].values
            matrix = np.broadcast_to(col, (len(col), len(col))).T

            # Cast if needed and attach to the dictionary
            if out_dtype is not None:
                matrix = matrix.astype(out_dtype)
            return_dict[tp_from][tp_to] = matrix

        return return_dict

    def get_from_home_factor_dictionary(
        self,
        segment_params: Dict[str, Any],
        out_dtype: type = None,
    ) -> Dict[int, np.ndarray]:
        """Generate a from-home factor dictionary

        Uses internal data to generate a dictionary describing the
        from-home factors to be used in a PA to OD conversion process.

        Parameters
        ----------
        segment_params:
            A segment_params dictionary where the keys are the names of
            segments, and the values are the values of the segments.
            This is the format returned when iterating over a
            nd.core.SegmentationLevel

        out_dtype:
            The datatype to cast the matrices to before writing to disk.
            If left as None, no casting is done.

        Returns
        -------
        from_home_factor_dict:
            A dictionary where the keys are `self.time_periods` and the values
            are `(len(self.zoning_system), len(self.zoning_system))` size
            arrays.
        """
        # Get the tour proportions
        tour_props = self.get_tour_proportions_dictionary(segment_params, out_dtype)

        # Simplify into from home factors
        return_dict = dict.fromkeys(self.time_periods)
        for fh_tp in self.time_periods:
            # Sum across the fh_tp to get the factors
            factor_matrix = np.zeros((len(self.zoning_system), len(self.zoning_system)))
            for th_tp in self.time_periods:
                factor_matrix += tour_props[fh_tp][th_tp]
            return_dict[fh_tp] = factor_matrix

        return return_dict

    def get_to_home_factor_dictionary(
        self,
        segment_params: Dict[str, Any],
        out_dtype: type = None,
    ) -> Dict[int, np.ndarray]:
        """Generate a to-home factor dictionary

        Uses internal data to generate a dictionary describing the
        to-home factors to be used in a PA to OD conversion process.

        Parameters
        ----------
        segment_params:
            A segment_params dictionary where the keys are the names of
            segments, and the values are the values of the segments.
            This is the format returned when iterating over a
            nd.core.SegmentationLevel

        out_dtype:
            The datatype to cast the matrices to before writing to disk.
            If left as None, no casting is done.

        Returns
        -------
        from_home_factor_dict:
            A dictionary where the keys are `self.time_periods` and the values
            are `(len(self.zoning_system), len(self.zoning_system))` size
            arrays.
        """
        # Get the tour proportions
        tour_props = self.get_tour_proportions_dictionary(segment_params, out_dtype)

        # Simplify into from home factors
        return_dict = dict.fromkeys(self.time_periods)
        for th_tp in self.time_periods:
            # Sum across the th_tp to get the factors
            factor_matrix = np.zeros((len(self.zoning_system), len(self.zoning_system)))
            for fh_tp in self.time_periods:
                factor_matrix += tour_props[fh_tp][th_tp]
            return_dict[th_tp] = factor_matrix

        return return_dict

    def write_tour_proportions(
        self,
        segment_params: Dict[str, Any],
        path: os.PathLike,
        out_dtype: type = None,
    ) -> None:
        """Generate the tour proportions and write to disk

        Uses internal data to generate a nested dictionary describing the tour
        proportions. The first key is the from-home time period. The second
        key is the to-home time period. The value is a
        `(len(self.zoning_system), len(self.zoning_system))` size array.
        This is then written to disk as a pickle.

        Parameters
        ----------
        segment_params:
            A segment_params dictionary where the keys are the names of
            segments, and the values are the values of the segments.
            This is the format returned when iterating over a
            nd.core.SegmentationLevel

        path:
            The path to write the generated tour proportions to. Written
            out file will be a pickle file.

        out_dtype:
            The datatype to cast the matrices to before writing to disk.
            If left as None, no casting is done.

        Returns
        -------
        None
        """
        tour_props = self.get_tour_proportions_dictionary(segment_params, out_dtype)
        file_ops.write_pickle(tour_props, path)

    def write_from_home_factors(
        self,
        segment_params: Dict[str, Any],
        path: os.PathLike,
        out_dtype: type = None,
    ) -> None:
        """Generate a from-home factor dictionary and write to disk

        Uses internal data to generate a dictionary describing the
        from-home factors to be used in a PA to OD conversion process.

        Parameters
        ----------
        segment_params:
            A segment_params dictionary where the keys are the names of
            segments, and the values are the values of the segments.
            This is the format returned when iterating over a
            nd.core.SegmentationLevel

        out_dtype:
            The datatype to cast the matrices to before writing to disk.
            If left as None, no casting is done.

        path:
            The path to write the generated tour proportions to. Written
            out file will be a pickle file.

        Returns
        -------
        None
        """
        fh_factors = self.get_from_home_factor_dictionary(segment_params, out_dtype)
        file_ops.write_pickle(fh_factors, path)

    def write_to_home_factors(
        self,
        segment_params: Dict[str, Any],
        path: os.PathLike,
        out_dtype: type = None,
    ) -> None:
        """Generate a to-home factor dictionary and write to disk

        Uses internal data to generate a dictionary describing the
        to-home factors to be used in a PA to OD conversion process.

        Parameters
        ----------
        segment_params:
            A segment_params dictionary where the keys are the names of
            segments, and the values are the values of the segments.
            This is the format returned when iterating over a
            nd.core.SegmentationLevel

        out_dtype:
            The datatype to cast the matrices to before writing to disk.
            If left as None, no casting is done.

        path:
            The path to write the generated tour proportions to. Written
            out file will be a pickle file.

        Returns
        -------
        None
        """
        fh_factors = self.get_to_home_factor_dictionary(segment_params, out_dtype)
        file_ops.write_pickle(fh_factors, path)


class PreMeTourProportionsGenerator(TourProportions):
    """Generator of pre-ME tour proportions for the DistributionModel"""

    def __init__(
        self,
        return_home_factors: ReturnHomeFactors,
        tp_splits: TimePeriodSplits,
    ):
        # Validate inputs
        if return_home_factors.time_periods != tp_splits.time_periods:
            raise ValueError(
                "return_home_factors and tp_splits must have the same " "time_periods."
            )

        # Save the required column names
        self.zoning_system = tp_splits.zoning_system
        self.zone_col = tp_splits.zoning_system.col_name
        self.purpose_col = tp_splits.purpose_col
        self.tp_col = tp_splits.tp_col
        self.time_th_cols = return_home_factors.time_periods
        tour_prop_df = self._generate_tour_prop_df(
            return_home_factors=return_home_factors,
            tp_splits=tp_splits,
        )

        super().__init__(
            tour_props_df=tour_prop_df,
            zone_col=self.zone_col,
            zoning_system=self.zoning_system,
            segmentation=tp_splits.output_seg,
            segment_cols=tp_splits.segment_cols,
            tp_col=self.tp_col,
            time_periods=tp_splits.time_periods,
        )

    def _generate_tour_prop_df(
        self,
        return_home_factors: ReturnHomeFactors,
        tp_splits: TimePeriodSplits,
    ) -> pd.DataFrame:
        """Generate a wide dataframe of tour proportion values"""
        # ## COMBINE INTO ONE DATAFRAME ## #
        rh_factors = return_home_factors.get_factors(
            purpose_col=self.purpose_col,
            time_fh_col=self.tp_col,
        )
        tour_prop_df = pd.merge(
            left=tp_splits.dataframe,
            right=rh_factors,
            how="left",
            on=[self.purpose_col, self.tp_col],
        )

        # # GENERATE THE TOUR PROPORTIONS ## #
        for col in self.time_th_cols:
            tour_prop_df[col] *= tour_prop_df[tp_splits.factor_col]
        tour_prop_df = tour_prop_df.drop(columns=tp_splits.factor_col)

        return tour_prop_df
