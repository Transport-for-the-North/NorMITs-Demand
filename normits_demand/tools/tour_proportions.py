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
import dataclasses

from typing import Tuple
from typing import Iterable
from dataclasses import InitVar

# Third Party
import pandas as pd

# Local Imports
from normits_demand import core as nd_core
from normits_demand.utils import general as du


@dataclasses.dataclass
class PhiFactors:
    """Storage of return home factor data"""

    time_fh_col: str
    time_th_col: str
    factor_col: str
    dataframe: pd.DataFrame

    def __post_init__(self):
        # Init
        needed_cols = [self.time_th_col, self.time_fh_col, self.factor_col]
        all_df_cols = self.dataframe.columns.to_list()

        # Make sure self.dataframe actually contains the needed columns
        missing_cols = set(needed_cols) - set(all_df_cols)
        if len(missing_cols) > 0:
            raise ValueError(
                "The given dataframe doesn't contain all the columns it "
                "should. The following given columns names are missing: "
                f"'{missing_cols}'"
            )

        # Figure out if self.dataframe contains any extra cols and save it
        self.additional_cols = du.list_safe_remove(
            lst=self.dataframe.columns.to_list(),
            remove=[self.time_th_col, self.time_fh_col, self.factor_col],
        )

    def get_factors(self, time_periods: Iterable[int]):
        """Get a square matrix of phi factors

        filters `self.dataframe` down to only include the time periods
        given in `time_periods` and returns a wide matrix where the columns
        are the `self.time_th_col` values

        Parameters
        ----------
        time_periods

        Returns
        -------

        """
        # Drop unneeded tps
        mask = self.dataframe[self.time_fh_col].isin(time_periods) & self.dataframe[
            self.time_th_col
        ].isin(time_periods)
        phi_df = self.dataframe[mask].copy()

        # Return as square matrix
        return phi_df.pivot(
            index=self.additional_cols + [self.time_fh_col],
            columns=self.time_th_col,
            values=self.factor_col,
        ).reset_index()


@dataclasses.dataclass
class TimePeriodSplits:
    """Calculation and storage of the zonal time period split data"""

    mode: nd_core.Mode
    zoning_system: nd_core.ZoningSystem

    # Values passed straight to __post_init__
    dvec: InitVar[nd_core.DVector]
    translation_weighting: InitVar[str] = "population"

    # Generated attributes
    df: pd.DataFrame = dataclasses.field(init=False)
    output_seg: nd_core.SegmentationLevel = dataclasses.field(init=False)

    def __post_init__(self, dvec: nd_core.DVector, translation_weighting: str):
        # Init
        aggregate_seg, output_seg = self._segmentation_lookup()
        self.output_seg = output_seg

        # Convert dvec into a dataframe - keeping only needed data
        dvec = dvec.aggregate(aggregate_seg)
        dvec = dvec.translate_zoning(self.zoning_system, weighting=translation_weighting)
        df = dvec.to_df()

        # Filter down by mode
        mask = df['m'].isin(self.mode.get_mode_values())
        df = df[mask].reset_index(drop=True)
        df = df.drop(columns=['m'])

        # Adjust factors back to 1
        group_cols = du.list_safe_remove(df.columns.to_list(), ['tp', 'val'])
        df['sum'] = df.groupby(group_cols)['val'].transform('sum')
        df['val'] /= df['sum']

        # If val is NaN, assume even split
        df['val'] = df['val'].fillna(0.25)
        df = df.drop(columns='sum')

        self.df = df

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


class PreMeTourProportionsGenerator:
    """Generator of pre-ME tour proportions for the DistributionModel"""

    def __init__(
        self,
        phi_factors: PhiFactors,
        tp_splits: TimePeriodSplits,
    ):
        pass
