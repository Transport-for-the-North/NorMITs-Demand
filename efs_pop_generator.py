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

from efs_constrainer import ExternalForecastSystemConstrainer
sys.path.append("../../../../NorMITs Utilities/Python")
sys.path.append("C:/Users/Sneezy/Desktop/Code/S/NorMITs Utilities/Python")
import nu_error_management as err_check

class ExternalForecastSystemPopulationGenerator:
    # infill statics
    POPULATION_INFILL = 0.001
    
    # sub-classes
    efs_constrainer = None
    
    # procedural support
    webtag_certainty_bounds = dict
    
    def __init__(self,
                 webtag_certainty_bounds = {
                         "NC": ["NC"],
                         "MTL": ["NC", "MTL"],
                         "RF": ["NC", "MTL", "RF"],
                         "H": ["NC", "MTL", "RF", "H"]
                         }
                 ):
        """
        #TODO
        """
        self.efs_constrainer = ExternalForecastSystemConstrainer()
        self.webtag_certainty_bounds = webtag_certainty_bounds
    
    def run(self,
            population_growth: pd.DataFrame,
            population_values: pd.DataFrame,
            population_constraint: pd.DataFrame,
            population_split: pd.DataFrame,
            households_growth: pd.DataFrame,
            households_values: pd.DataFrame,
            households_constraint: pd.DataFrame,
            housing_split: pd.DataFrame,
            housing_occupancy: pd.DataFrame,
            development_log: pd.DataFrame = None,
            development_log_split: pd.DataFrame = None,
            integrating_development_log: bool = False,
            minimum_development_certainty: str = "MTL", # "NC", "MTL", "RF", "H"
            population_metric: str = "Households", # Households, Population
            constraint_required: List[bool] = [
                    True, # initial population metric constraint
                    True, # post-development constraint
                    True, # secondary post-development constraint used for matching HH pop
                    False # final trip based constraint
                    ],
            constraint_method: str = "Percentage", # Percentage, Average
            constraint_area: str = "Designated", # Zone, Designated, All
            constraint_on: str = "Growth", # Growth, All
            constraint_source: str = "Grown Base", # Default, Grown Base, Model Grown Base
            designated_area: pd.DataFrame = None,
            base_year_string: str = None,
            model_years: List[str] = List[None]
            ):
        """
        #TODO
        """
        print("Used population metric is: " + population_metric)
        
        if integrating_development_log:
            if development_log is not None:
                development_log = development_log.copy()
            else:
                print("No development_log dataframe passed to population "
                      + "generator but development_log is indicated to be "
                      + "required. Process will not function correctly.")
                exit(10)
        
        if population_metric == "population":
            # ## GROW POPULATION
            grown_population = self.population_grower(
                    population_growth,
                    population_values,
                    base_year_string,
                    model_years
                    ) 
        
            # ## initial population metric constraint
            if constraint_required[0] and (constraint_source != "model grown base"):
                print("Performing the first constraint on population...")
                grown_population = self.efs_constrainer.run(
                        grown_population,
                        constraint_method,
                        constraint_area,
                        constraint_on,
                        population_constraint,
                        model_years,
                        designated_area
                        )
            elif constraint_source == "model grown base":
                print("Generating model grown base constraint for use on "
                      + "development constraints...")
                population_constraint = grown_population.copy()
            
            # ## D-LOG INTEGRATION
            if integrating_development_log:
                print("Including development log...")
                development_households = self.development_log_house_generator(
                        development_log,
                        development_log_split,
                        minimum_development_certainty,
                        model_years
                        )
                # TODO: Generate population
            else:
                print("Not including development log...")
            
            # ## post-development log constraint
            if constraint_required[1]:
                print("Performing the post-development log on population...")
                grown_population = self.efs_constrainer.run(
                        grown_population,
                        constraint_method,
                        constraint_area,
                        constraint_on,
                        population_constraint,
                        base_year_string,
                        model_years,
                        designated_area
                        )
            return grown_population

        if population_metric == "households":
            # ## GROW HOUSEHOLDS
            grown_households = self.households_grower(
                    households_growth,
                    households_values,
                    base_year_string,
                    model_years
                    )
        
            # ## initial population metric constraint
            if constraint_required[0] and (constraint_source != "model grown base"):
                print("Performing the first constraint on households...")
                grown_households = self.efs_constrainer.run(
                        grown_households,
                        constraint_method,
                        constraint_area,
                        constraint_on,
                        households_constraint,
                        base_year_string,
                        model_years,
                        designated_area
                        )
            elif constraint_source == "model grown base":
                print("Generating model grown base constraint for use on "
                      + "development constraints...")
                households_constraint = grown_households.copy()

            # ## SPLIT HOUSEHOLDS
            split_households = self.split_housing(
                    grown_households,
                    housing_split,
                    base_year_string,
                    model_years
                    )
            
            # ## D-LOG INTEGRATION
            if integrating_development_log:
                print("Including development log...")
                # ## DEVELOPMENT SPLIT
                split_development_households = self.development_log_house_generator(
                        development_log,
                        development_log_split,
                        minimum_development_certainty,
                        model_years
                        )
                
                # ## COMBINE BASE + DEVELOPMENTS
                split_households = self.combine_households_and_developments(
                        split_households,
                        split_development_households
                        )
                
            else:
                print("Not including development log...")

            # ## post-development log constraint
            if constraint_required[1]:
                print("Performing the post-development log constraint on "
                      "households...")
                split_households = self.efs_constrainer.run(
                        split_households,
                        constraint_method,
                        constraint_area,
                        constraint_on,
                        households_constraint,
                        base_year_string,
                        model_years,
                        designated_area
                        )
            
            # ## POPULATION GENERATION
            population = self.generate_housing_occupancy(
                    split_households,
                    housing_occupancy,
                    base_year_string,
                    model_years
                    )
            
            print()
            # ## ENSURE WE HAVE NO MINUS POPULATION
            for year in model_years:
                population.loc[
                    population[year] < 0,
                    year
                ] = self.POPULATION_INFILL
            
            # ## secondary post-development constraint
            # (used for matching HH pop)
            if constraint_required[2] and (constraint_source != "model grown base"):
                print("Constraining to population on population in "
                      "households...")
                population = self.efs_constrainer.run(
                        population,
                        constraint_method,
                        constraint_area,
                        constraint_on,
                        population_constraint,
                        base_year_string,
                        model_years,
                        designated_area
                        )
            elif constraint_source == "model grown base":
                print("Population constraint in households metric selected.")
                print("No way to control population using model grown base "
                      + "constraint source.")
            
            # ## SPLIT POP (On traveller type)
            split_population = self.split_population(
                    population,
                    population_split,
                    base_year_string,
                    model_years
                    )
            
            # ## RECOMBINING POP
            final_population = self.growth_recombination(
                     split_population,
                     "base_year_population",
                     model_years
                     )
            
            final_population.sort_values(
                    by=[
                            "model_zone_id",
                            "property_type_id",
                            "traveller_type_id"
                            ],
                    inplace=True
                    )
            return final_population

        # Should only get here in bad situations
        # TODO: Exit a bit cleaner
        print("Incorrect population metric passed. Was given %s." %
              str(population_metric))
        exit(11)

    
    def population_grower(self,
                         population_growth: pd.DataFrame,
                         population_values: pd.DataFrame,
                         base_year: str,
                         year_string_list: List[str]
                         ) -> pd.DataFrame:
        # get population growth from base year
        print("Adjusting population growth to base year...")
        population_growth = self.convert_growth_off_base_year(
                population_growth,
                base_year,
                year_string_list
                )
        print("Adjusted population growth to base year!")
        
        
        print("Growing population from base year...")
        grown_population = self.get_grown_values(
                population_values,
                population_growth,
                "base_year_population",
                year_string_list
                )
        print("Grown population from base year!")
        
        return grown_population
            
    def households_grower(self,
                         households_growth: pd.DataFrame,
                         households_values: pd.DataFrame,
                         base_year: str,
                         year_string_list: List[str]
                         ) -> pd.DataFrame:
        
        households_growth = households_growth.copy()
        households_values = households_values.copy()
        print("Adjusting households growth to base year...")
        households_growth = self.convert_growth_off_base_year(
                households_growth,
                base_year,
                year_string_list
                )
        print("Adjusted households growth to base year!")
        
        print("Growing households from base year...")
        grown_households = self.get_grown_values(
                households_values,
                households_growth,
                "base_year_households",
                year_string_list
                )
        print("Grown households from base year!")
        
        return grown_households

    def convert_growth_off_base_year(self,
                                     growth_dataframe: pd.DataFrame,
                                     base_year: str,
                                     all_years: List[str]
                                     ) -> pd.DataFrame:
        """
        #TODO
        """
        growth_dataframe = growth_dataframe.copy()
        growth_dataframe.loc[
        :,
        all_years
        ] = growth_dataframe.apply(
            lambda x,
            columns_required=all_years,
            base_year=base_year:
                x[columns_required] / x[base_year],
                axis = 1
        )
        
        return growth_dataframe
    
    def get_grown_values(self,
                       base_year_dataframe: pd.DataFrame,
                       growth_dataframe: pd.DataFrame,
                       base_year_string: str,
                       all_years: List[str]
                       ) -> pd.DataFrame:
        """
        #TODO
        """
        base_year_dataframe = base_year_dataframe.copy()
        growth_dataframe = growth_dataframe.copy()
        
        # CREATE GROWN DATAFRAME
        grown_dataframe = base_year_dataframe.merge(
                growth_dataframe,
                on = "model_zone_id"
                )
        
        grown_dataframe.loc[:, all_years] = grown_dataframe.apply(
            lambda x,
            columns_required=all_years,
            base_year=base_year_string:
                (x[columns_required] - 1) * x[base_year],
                axis=1
        )
        
        return grown_dataframe
    
    def growth_recombination(self,
                             metric_dataframe: pd.DataFrame,
                             metric_column_name: str,
                             all_years: List[str]
                             ) -> pd.DataFrame:
        """
        #TODO
        """
        metric_dataframe = metric_dataframe.copy()
        
        ## combine together dataframe columns to give full future values
        ## f.e. base year will get 0 + base_year_population        
        for year in all_years:
            metric_dataframe[year] = (
                    metric_dataframe[year]
                    +
                    metric_dataframe[metric_column_name]
                    )
        
        ## drop the unnecessary metric column
        metric_dataframe = metric_dataframe.drop(
                labels = metric_column_name,
                axis = 1
                )
        
        return metric_dataframe
    
    def split_housing(self,
                      households_dataframe: pd.DataFrame,
                      housing_split_dataframe: pd.DataFrame,
                      base_year_string: str,
                      all_years: List[str]
                      ) -> pd.DataFrame:
        households_dataframe = households_dataframe.copy()
        housing_split_dataframe = housing_split_dataframe.copy()
        
        split_households_dataframe = pd.merge(
                households_dataframe,
                housing_split_dataframe,
                on = ["model_zone_id"],
                suffixes = {"", "_split"}
                )

        # Calculate the number of houses belonging to each split
        split_households_dataframe["base_year_households"] = (
                split_households_dataframe["base_year_households"]
                *
                split_households_dataframe[base_year_string + "_split"]
                )
        
        for year in all_years:
            # we iterate over each zone
            split_households_dataframe[year] = (
                    split_households_dataframe[year]
                    *
                    split_households_dataframe[year + "_split"]
                    )

        # extract just the needed columns
        required_columns = [
                "model_zone_id",
                "property_type_id",
                "base_year_households"
                ]
        required_columns.extend(all_years)
        split_households_dataframe = split_households_dataframe[required_columns]

        return split_households_dataframe
    
    def split_population(self,
                      population_dataframe: pd.DataFrame,
                      population_split_dataframe: pd.DataFrame,
                      base_year_string: str,
                      all_years: List[str]
                      ) -> pd.DataFrame:
        population_dataframe = population_dataframe.copy()
        population_split_dataframe = population_split_dataframe.copy()
        
        split_population_dataframe = pd.merge(
                population_dataframe,
                population_split_dataframe,
                on = ["model_zone_id", "property_type_id"],
                suffixes = {"", "_split"}
                )
                
        split_population_dataframe["base_year_population"] = (
                split_population_dataframe["base_year_population"]
                *
                split_population_dataframe[base_year_string + "_split"]
                )
        
        for year in all_years:
            # we iterate over each zone
            # create zone mask
            split_population_dataframe[year] = (
                    split_population_dataframe[year]
                    *
                    split_population_dataframe[year + "_split"]
                    )

        # Extract the required columns
        required_columns = [
                "model_zone_id",
                "property_type_id",
                "traveller_type_id",
                "base_year_population"
                ]
        required_columns.extend(all_years)
        split_population_dataframe = split_population_dataframe[required_columns]

        return split_population_dataframe
    
    def generate_housing_occupancy(self,
                                   households_dataframe: pd.DataFrame,
                                   housing_occupancy_dataframe: pd.DataFrame,
                                   base_year_string: str,
                                   all_years: List[str]
                                   ) -> pd.DataFrame:
        """
        #TODO
        """
        households_dataframe = households_dataframe.copy()
        housing_occupancy_dataframe = housing_occupancy_dataframe.copy()
        
        households_population = pd.merge(
                households_dataframe,
                housing_occupancy_dataframe,
                on = ["model_zone_id", "property_type_id"],
                suffixes = {"", "_occupancy"}
                )
        
        households_population["base_year_population"] = (
                households_population["base_year_households"]
                *
                households_population[base_year_string + "_occupancy"]
                )
        
        pop_columns = [
                "model_zone_id",
                "property_type_id",
                "base_year_population"
                ]
        pop_dictionary = {}
        
        for year in all_years:
            households_population[year + "_population"] = (
                    households_population[year]
                    *
                    households_population[year + "_occupancy"]
                    )
            pop_columns.append(year + "_population")
            pop_dictionary[year + "_population"] = year

        # Extract just the needed columns and rename to years
        households_population = households_population[pop_columns]
        households_population = households_population.rename(columns=pop_dictionary)
        
        return households_population
    
    def development_log_house_generator(self,
                                        development_log: pd.DataFrame,
                                        development_log_split: pd.DataFrame,
                                        minimum_development_certainty: str,
                                        all_years: List[str]
                                        ) -> pd.DataFrame:
        """
        #TODO
        """
        development_log = development_log.copy()
        development_log_split = development_log_split.copy()
        
        development_log = development_log[
                development_log["dev_type"] == "Housing"
                ]
        
        webtag_certainty_bounds = self.webtag_certainty_bounds[
                minimum_development_certainty
                ]
        
        development_log["used"] = False 
        
        for webtag_certainty in webtag_certainty_bounds:
            mask = (development_log["webtag_certainty"] == webtag_certainty)
            development_log.loc[
                    mask,
                    "used"
                    ] = True
        
        required_columns = [
                "model_zone_id"
                ]
        
        required_columns.extend(all_years)
        
        development_log = development_log[
                development_log["used"] == True
                ]
        
        development_log = development_log[required_columns]
        
        development_households = pd.merge(
                development_log,
                development_log_split,
                on = ["model_zone_id"],
                suffixes = {"", "_split"}
                )
        
        for year in all_years:
            # we iterate over each zone
            development_households[year] = (
                    development_households[year]
                    *
                    development_households[year + "_split"]
                    )
        
        required_columns = [
                "model_zone_id",
                "property_type_id"
                ]
        
        required_columns.extend(all_years)        
        
        development_households = development_households[required_columns]
        
        development_households = development_households.groupby(
                by = [
                        "model_zone_id",
                        "property_type_id"
                        ],
                as_index = False
                ).sum()
        
        return development_households
    
    def combine_households_and_developments(self,
                                            split_households: pd.DataFrame,
                                            split_developments: pd.DataFrame
                                            ) -> pd.DataFrame:
        """
        #TODO
        """
        split_households = split_households.copy()
        split_developments = split_developments.copy()
        
        combined_households = split_households.append(
                split_developments,
                ignore_index = True
                )
        
        combined_households = combined_households.groupby(
                by = [
                        "model_zone_id",
                        "property_type_id"
                        ],
                as_index = False
                ).sum()
        
        return combined_households