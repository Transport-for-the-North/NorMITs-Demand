# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 12:13:07 2019

@author: Sneezy
"""

from functools import reduce
from typing import List
import sys

import numpy as np
import pandas as pd

from demand_utilities import error_management as err_check
from demand_utilities import utils as du


class ForecastConstrainer:
    def run(self,
            grown_dataframe: pd.DataFrame,
            constraint_method: str, # percentage, average
            constraint_area: str, # zone, designated, all
            constraint_on: str, # growth, all
            constraint_values: pd.DataFrame,
            base_year: str,
            years: List[str],
            designated_area: pd.DataFrame = None,
            zone_col: str = 'model_zone_id'
            ):
        """
        # TODO
        """
        # ## CONSTRAINT APPLICATION
        if constraint_method == "percentage":
            constrained_dataframe = self.constrain_by_percentage(
                grown_dataframe,
                constraint_area,
                constraint_on,
                constraint_values,
                years,
                designated_area,
                zone_col
            )
        elif constraint_method == "average":
            constrained_dataframe = self.constrain_by_average(
                grown_dataframe,
                constraint_area,
                constraint_on,
                constraint_values,
                years,
                designated_area,
                zone_col
            )
        else:
            raise ValueError("constraint_area '%s' does not exist." %
                             str(constraint_area))

        return constrained_dataframe

    def constrain_by_percentage(self,
                                grown_dataframe: pd.DataFrame,
                                constraint_area: str, # zone, designated, all
                                constraint_on: str, # growth, all
                                constraint_values: pd.DataFrame,
                                model_years: List[str],
                                designated_area: pd.DataFrame,
                                zone_col: str = 'model_zone_id'
                                ) -> pd.DataFrame:
        """
        #TODO
        """
        if constraint_area == "zone":
            # set each grouping area to one zone
            grouping_area = pd.DataFrame({
                "grouping_id": grown_dataframe[zone_col].values,
                zone_col: grown_dataframe[zone_col].values
            })

        elif constraint_area == "designated":
            # no need to change anything
            grouping_area = designated_area

        elif constraint_area == "all":
            # singular grouping area for all zones
            grouping_area = pd.DataFrame({
                "grouping_id": 1,
                zone_col: grown_dataframe[zone_col].values
            })

        else:
            raise ValueError("constraint_area '%s' does not exist." %
                             str(constraint_area))

        groupings = grouping_area["grouping_id"].unique()
        
        for group in groupings:
            # Create masks for the zones in this group
            group_mask = grouping_area["grouping_id"] == group
            zones = grouping_area[group_mask][zone_col].values
            grown_dataframe_mask = grown_dataframe[zone_col].isin(zones)
            constraint_dataframe_mask = constraint_values[zone_col].isin(zones)

            for year in model_years:
                # Sum all zones in this group for this year
                # Constraint, then grown vals
                constraint_total = constraint_values.loc[
                    constraint_dataframe_mask,
                    year
                ].sum()
                grown_total = grown_dataframe.loc[
                    grown_dataframe_mask,
                    year
                ].sum()
                percentage_shift = constraint_total / grown_total

                # Create mask for this year
                year_grown_dataframe_mask = (
                    grown_dataframe_mask
                    &
                    grown_dataframe[year] > 0
                )
                
                if pd.notna(percentage_shift):
                    grown_dataframe.loc[year_grown_dataframe_mask, year] = (
                        grown_dataframe.loc[year_grown_dataframe_mask, year]
                        *
                        percentage_shift
                    )
                
        constrained_dataframe = grown_dataframe
        return constrained_dataframe

    def constrain_by_average(self,
                             grown_dataframe: pd.DataFrame,
                             constraint_area: str, # zone, designated, all
                             constraint_on: str, # growth, all
                             constraint_values: pd.DataFrame,
                             model_years: List[str],
                             designated_area: pd.DataFrame,
                             zone_col: str = 'model_zone_id'
                             ) -> pd.DataFrame:
        """
        #TODO
        """
        if constraint_area == "zone":
            # set each grouping area to one zone
            grouping_area = pd.DataFrame({
                "grouping_id": grown_dataframe[zone_col].values,
                zone_col: grown_dataframe[zone_col].values
            })
        elif constraint_area == "designated":
            # no need to change anything
            grouping_area = designated_area

        elif constraint_area == "all":
            # singular grouping area for all zones
            grouping_area = pd.DataFrame({
                "grouping_id": 1,
                zone_col: grown_dataframe[zone_col].values
            })
        else:
            raise ValueError("constraint_area '%s' does not exist." %
                             str(constraint_area))
            
        groupings = grouping_area["grouping_id"].unique()
        
        for group in groupings:
            group_mask = grouping_area["grouping_id"] == group
            zones = grouping_area[group_mask][zone_col].values
            grown_dataframe_mask = grown_dataframe[zone_col].isin(zones)
            constraint_dataframe_mask = constraint_values[zone_col].isin(zones)

            for year in model_years:
                constraint_total = constraint_values.loc[
                    constraint_dataframe_mask,
                    year
                ].sum()
                grown_total = grown_dataframe.loc[
                    grown_dataframe_mask,
                    year
                ].sum()
                average_shift = (constraint_total - grown_total) / len(zones)
                
                if pd.notna(average_shift):
                    grown_dataframe.loc[grown_dataframe_mask, year] = (
                        grown_dataframe.loc[grown_dataframe_mask, year]
                        +
                        average_shift
                    )
                
        constrained_dataframe = grown_dataframe
        return constrained_dataframe

    def convert_constraint_off_base_year(self,
                                         constraint_dataframe: pd.DataFrame,
                                         base_year: str,
                                         all_years: List[str]
                                         ) -> pd.DataFrame:
        """
        #TODO
        """
        constraint_dataframe = constraint_dataframe.copy()

        # TODO: Get rid of this lambda
        constraint_dataframe.loc[:, all_years] = constraint_dataframe.apply(
            lambda x, columns_required=all_years, base_year=base_year:
                x[columns_required] - x[base_year],
            axis=1)
        
        return constraint_dataframe


def grow_constraint(base_values: pd.DataFrame,
                    growth_factors: pd.DataFrame,
                    base_year: str,
                    future_years: List[str]
                    ) -> pd.DataFrame:
    growth_factors = du.convert_growth_off_base_year(
        growth_factors,
        base_year,
        future_years
    )
    grown_vals = du.get_growth_values(
        base_values,
        growth_factors,
        base_year,
        future_years
    )
    # Add base year back in to get full grown values
    grown_vals = du.growth_recombination(
        grown_vals,
        base_year_col=base_year,
        future_year_cols=future_years,
        drop_base_year=False
    )

    return grown_vals
