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

sys.path.append("../../../../NorMITs Utilities/Python")
sys.path.append("C:/Users/Sneezy/Desktop/Code/S/NorMITs Utilities/Python")
import nu_error_management as err_check


class ExternalForecastSystemConstrainer:
    def run(self,
            grown_dataframe: pd.DataFrame,
            constraint_method: str, # percentage, average
            constraint_area: str, # zone, designated, all
            constraint_on: str, # growth, all
            constraint_values: pd.DataFrame,
            base_year: str,
            model_years: List[str],
            designated_area: pd.DataFrame = None
            ):
        """
        # TODO
        """        
        # ## CONSTRAINT APPLICATION
        if constraint_method == "percentage":
            # TODO
            constrained_dataframe = self.constrain_by_percentage(
                    grown_dataframe,
                    constraint_area,
                    constraint_on,
                    constraint_values,
                    model_years,
                    designated_area
                    )
            print("")
        elif constraint_method == "average":
            # TODO
            print("")
            constrained_dataframe = self.constrain_by_average(
                    grown_dataframe,
                    constraint_area,
                    constraint_on,
                    constraint_values,
                    model_years,
                    designated_area
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
                                designated_area: pd.DataFrame
                                ) -> pd.DataFrame:
        """
        #TODO
        """
        if constraint_area == "zone":
            # set each grouping area to one zone
            grouping_area = pd.DataFrame({
                "grouping_id": grown_dataframe["model_zone_id"].values,
                "model_zone_id": grown_dataframe["model_zone_id"].values
            })

        elif constraint_area == "designated":
            # no need to change anything
            grouping_area = designated_area

        elif constraint_area == "all":
            # singular grouping area for all zones
            grouping_area = pd.DataFrame({
                "grouping_id": 1,
                "model_zone_id": grown_dataframe["model_zone_id"].values
            })

        else:
            raise ValueError("constraint_area '%s' does not exist." %
                             str(constraint_area))

        groupings = grouping_area["grouping_id"].unique()
        
        for group in groupings:
            # Create masks for the zones in this group
            group_mask = grouping_area["grouping_id"] == group
            zones = grouping_area[group_mask]["model_zone_id"].values
            grown_dataframe_mask = grown_dataframe["model_zone_id"].isin(zones)
            constraint_dataframe_mask = constraint_values["model_zone_id"].isin(zones)

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
                             designated_area: pd.DataFrame
                             ) -> pd.DataFrame:
        """
        #TODO
        """
        if constraint_area == "zone":
            # set each grouping area to one zone
            grouping_area = pd.DataFrame({
                "grouping_id": grown_dataframe["model_zone_id"].values,
                "model_zone_id": grown_dataframe["model_zone_id"].values
            })
        elif constraint_area == "designated":
            # no need to change anything
            grouping_area = designated_area

        elif constraint_area == "all":
            # singular grouping area for all zones
            grouping_area = pd.DataFrame({
                "grouping_id": 1,
                "model_zone_id": grown_dataframe["model_zone_id"].values
            })
        else:
            raise ValueError("constraint_area '%s' does not exist." %
                             str(constraint_area))
            
        groupings = grouping_area["grouping_id"].unique()
        
        for group in groupings:
            group_mask = grouping_area["grouping_id"] == group
            zones = grouping_area[group_mask]["model_zone_id"].values
            grown_dataframe_mask = grown_dataframe["model_zone_id"].isin(zones)
            constraint_dataframe_mask = constraint_values["model_zone_id"].isin(zones)

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
