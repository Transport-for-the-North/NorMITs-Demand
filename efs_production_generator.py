# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 12:13:07 2019

@author: Sneezy
"""

from functools import reduce
from typing import List
import os
import sys
import warnings

import numpy as np
import pandas as pd

from tqdm import tqdm

import efs_constants as consts
from efs_constrainer import ForecastConstrainer
from demand_utilities import utils as du
# TODO: Move functions that can be static elsewhere.
#  Maybe utils?


class EFSProductionGenerator:
    
    def __init__(self,
                 tag_certainty_bounds=consts.TAG_CERTAINTY_BOUNDS,
                 population_infill: float = 0.001):
        """
        #TODO
        """
        self.efs_constrainer = ForecastConstrainer()
        self.tag_certainty_bounds = tag_certainty_bounds
        self.pop_infill = population_infill
    
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
            # lu_import_path: str,
            lu_year: int = 2018,
            d_log: pd.DataFrame = None,
            d_log_split: pd.DataFrame = None,
            minimum_development_certainty: str = "MTL",  # "NC", "MTL", "RF", "H"
            population_metric: str = "Population",  # Households, Population
            constraint_required: List[bool] = consts.DEFAULT_PRODUCTION_CONSTRAINTS,
            constraint_method: str = "Percentage",  # Percentage, Average
            constraint_area: str = "Designated",  # Zone, Designated, All
            constraint_on: str = "Growth",  # Growth, All
            constraint_source: str = "Grown Base",  # Default, Grown Base, Model Grown Base
            designated_area: pd.DataFrame = None,
            base_year: str = None,
            future_years: List[str] = List[None],
            out_path: str = None,
            area_types: pd.DataFrame = None,
            trip_rates: pd.DataFrame = None,
            merge_cols: List[str] = None,
            zone_col: str = 'msoa_zone_id',
            audits: bool = True
            ) -> pd.DataFrame:
        """
        #TODO
        """
        # Init
        all_years = [str(x) for x in [base_year] + future_years]
        create_productions = area_types is not None and trip_rates is not None
        integrate_d_log = d_log is not None and d_log_split is not None
        if integrate_d_log:
            d_log = d_log.copy()
            d_log_split = d_log_split.copy()

        # TODO: Make this more adaptive
        # Set merge and join cols
        if merge_cols is None:
            merge_cols = [
                'area_type',
                'traveller_type',
                'soc',
                'ns',
            ]

        production_group_cols = [zone_col, 'purpose_id'] + merge_cols
        production_cols = production_group_cols + [base_year] + future_years

        # Fix column naming
        if zone_col not in population_growth:
            population_growth = population_growth.copy().rename(
                columns={'model_zone_id': zone_col}
            )
        if zone_col not in designated_area:
            designated_area = designated_area.copy().rename(
                columns={'model_zone_id': zone_col}
            )
        if zone_col not in population_constraint:
            population_constraint = population_constraint.rename(
                columns={'model_zone_id': zone_col}
            )

        # TODO: Deal with case where land use year and base year don't match
        if str(lu_year) != str(base_year):
            raise ValueError("The base year and land use year are not the "
                             "same. Don't know how to deal with that at the"
                             "moment.")

        # TODO: FIX ME once dev over
        lu_import_path = r'Y:\NorMITs Land Use\iter3\land_use_output_msoa.csv'
        msoa_import_path = r'Y:\NorMITs Demand\inputs\default\zoning\msoa_zones.csv'
        trip_rate_path = r"Y:\NorMITs Demand\import\tfn_segment_production_params\hb_trip_rates.csv"
        time_splits_path = r"Y:\NorMITs Demand\import\tfn_segment_production_params\hb_time_split.csv"
        mean_time_splits_path = r"Y:\NorMITs Demand\import\tfn_segment_production_params\hb_ave_time_split.csv"
        population_metric = 'population'
        constraint_required[0] = False
        constraint_required[1] = False

        if population_metric == "households":
            raise ValueError("Production Model has changed. Households growth "
                             "is not currently supported.")

        # ## BASE YEAR POPULATION ## #
        print("Loading the base year population data...")
        base_year_pop = get_land_use_data(lu_import_path,
                                          msoa_import_path,
                                          zone_col=zone_col)
        base_year_pop = base_year_pop.rename(columns={'people': base_year})

        # Audit population numbers
        print("Base Year Population: %d" % base_year_pop[base_year].sum())

        # ## FUTURE YEAR POPULATION ## #
        print("Generating future year population data...")
        population = self.grow_population(
            base_year_pop,
            population_growth,
            base_year,
            future_years
        )

        # ## CONSTRAIN POPULATION ## #
        if constraint_required[0] and (constraint_source != "model grown base"):
            print("Performing the first constraint on population...")
            population = self.efs_constrainer.run(
                population,
                constraint_method,
                constraint_area,
                constraint_on,
                population_constraint,
                base_year,
                all_years,
                designated_area,
                zone_col
            )
        elif constraint_source == "model grown base":
            print("Generating model grown base constraint for use on "
                  "development constraints...")
            population_constraint = population.copy()

        # ## INTEGRATE D-LOG ## #
        if integrate_d_log:
            print("Integrating the development log...")
            raise NotImplementedError("D-Log population integration has not "
                                      "yet been implemented.")

        # ## POST D-LOG CONSTRAINT ## #
        if constraint_required[1]:
            print("Performing the post-development log constraint on population...")
            population = self.efs_constrainer.run(
                population,
                constraint_method,
                constraint_area,
                constraint_on,
                population_constraint,
                base_year,
                all_years,
                designated_area,
                zone_col
            )

        # Reindex and sum
        group_cols = [zone_col] + merge_cols
        index_cols = group_cols.copy() + all_years
        population = population.reindex(index_cols, axis='columns')
        population = population.groupby(group_cols).sum().reset_index()

        # Population Audit
        if audits:
            print('\n', '-'*15, 'Population Audit', '-'*15)
            for year in all_years:
                print('. Total population for year %s is: %.4f'
                      % (year, population[year].sum()))
            print('\n')

        # Write the produced population to file
        if out_path is None:
            print("WARNING! No output path given. "
                  "Not writing populations to file.")
        else:
            print("Writing population to file...")
            population.to_csv(os.path.join(out_path, "MSOA_population.csv"),
                              index=False)

        if not create_productions:
            return population

        # ## CREATE PRODUCTIONS ## #
        print("Population generated. Converting to productions...")
        productions = generate_productions(
            population=population,
            merge_cols=merge_cols,
            group_cols=production_group_cols,
            base_year=base_year,
            future_years=future_years,
            trip_rates_path=trip_rate_path,
            time_splits_path=time_splits_path,
            mean_time_splits_path=mean_time_splits_path
        )

        print(productions)
        sys.exit()

        return productions


        # USE THIS TO COMBINE WITH TRIP RATES LATER
        # Get trip rate cols for application
        # p_params.update({'tr_cols': ['traveller_type', 'area_type']})

        print("Used population metric is: " + population_metric)
        if d_log is not None:
            d_log = d_log.copy()

        # Choose the correct method for growing the productions
        if population_metric == "population":
            population = self.grow_by_population(
                population_growth,
                population_values,
                population_constraint,
                d_log,
                d_log_split,
                minimum_development_certainty,
                constraint_required,
                constraint_method,
                constraint_area,
                constraint_on,
                constraint_source,
                designated_area,
                base_year,
                model_years
            )
        elif population_metric == "households":
            population = self.grow_by_households(
                population_constraint,
                population_split,
                households_growth,
                households_values,
                households_constraint,
                housing_split,
                housing_occupancy,
                d_log,
                d_log_split,
                minimum_development_certainty,
                constraint_required,
                constraint_method,
                constraint_area,
                constraint_on,
                constraint_source,
                designated_area,
                base_year,
                model_years
            )
        else:
            raise ValueError("%s is not a valid population metric." %
                             str(population_metric))

        if out_path is None:
            print("WARNING! No output path given. "
                  "Not writing populations to file.")
        else:
            population.to_csv(os.path.join(out_path, "MSOA_population.csv"),
                              index=False)

        if area_types is None and trip_rates is None:
            return population

        print("Population generated. Converting to productions...")
        p_trips = self.production_generation(population,
                                             area_types,
                                             trip_rates,
                                             model_years)
        p_trips = self.convert_to_average_weekday(p_trips, model_years)

        print(p_trips)
        print(list(p_trips))
        sys.exit()

        return p_trips

    def grow_population(self,
                        population_values: pd.DataFrame,
                        population_growth: pd.DataFrame,
                        base_year: str = None,
                        future_years: List[str] = List[None],
                        growth_merge_col: str = 'msoa_zone_id'
                        ) -> pd.DataFrame:
        # TODO: Write grow_population() doc
        # Init
        all_years = [base_year] + future_years
        base_year_pop = population_values[base_year]

        print("Adjusting population growth to base year...")
        population_growth = du.convert_growth_off_base_year(
            population_growth,
            base_year,
            future_years
        )

        print("Growing population from base year...")
        grown_population = du.get_growth_values(
            population_values,
            population_growth,
            base_year,
            future_years,
            merge_col=growth_merge_col
        )

        # Make sure there is no minus growth
        for year in all_years:
            mask = (grown_population[year] < 0)
            grown_population.loc[mask, year] = self.pop_infill

        # Add base year back in to get full grown values
        grown_population = du.growth_recombination(
            grown_population,
            base_year_col=base_year,
            future_year_cols=future_years,
            drop_base_year=False
        )

        return grown_population

    def grow_by_population(self,
                           population_growth: pd.DataFrame,
                           population_values: pd.DataFrame,
                           population_constraint: pd.DataFrame,
                           d_log: pd.DataFrame = None,
                           d_log_split: pd.DataFrame = None,
                           minimum_development_certainty: str = "MTL",  # "NC", "MTL", "RF", "H"
                           constraint_required: List[bool] = consts.DEFAULT_PRODUCTION_CONSTRAINTS,
                           constraint_method: str = "Percentage",  # Percentage, Average
                           constraint_area: str = "Designated",  # Zone, Designated, All
                           constraint_on: str = "Growth",  # Growth, All
                           constraint_source: str = "Grown Base",  # Default, Grown Base, Model Grown Base
                           designated_area: pd.DataFrame = None,
                           base_year_string: str = None,
                           model_years: List[str] = List[None]
                           ) -> pd.DataFrame:
        """
        TODO: Write grow_by_population() doc
        """
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
                base_year_string,
                model_years,
                designated_area
            )
        elif constraint_source == "model grown base":
            print("Generating model grown base constraint for use on "
                  + "development constraints...")
            population_constraint = grown_population.copy()

        # ## D-LOG INTEGRATION
        if d_log is not None and d_log_split is not None:
            print("Including development log...")
            development_households = self.development_log_house_generator(
                d_log,
                d_log_split,
                minimum_development_certainty,
                model_years
            )
            raise NotImplementedError("D-Log pop generation not "
                                      "yet implemented.")
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

    def grow_by_households(self,
                           population_constraint: pd.DataFrame,
                           population_split: pd.DataFrame,
                           households_growth: pd.DataFrame,
                           households_values: pd.DataFrame,
                           households_constraint: pd.DataFrame,
                           housing_split: pd.DataFrame,
                           housing_occupancy: pd.DataFrame,
                           d_log: pd.DataFrame = None,
                           d_log_split: pd.DataFrame = None,
                           minimum_development_certainty: str = "MTL",  # "NC", "MTL", "RF", "H"
                           constraint_required: List[bool] = consts.DEFAULT_PRODUCTION_CONSTRAINTS,
                           constraint_method: str = "Percentage",  # Percentage, Average
                           constraint_area: str = "Designated",  # Zone, Designated, All
                           constraint_on: str = "Growth",  # Growth, All
                           constraint_source: str = "Grown Base",  # Default, Grown Base, Model Grown Base
                           designated_area: pd.DataFrame = None,
                           base_year_string: str = None,
                           model_years: List[str] = List[None]
                           ) -> pd.DataFrame:
        """
        TODO: Write grow_by_households() doc

        Parameters
        ----------
        population_constraint
        population_split
        households_growth
        households_values
        households_constraint
        housing_split
        housing_occupancy
        d_log
        d_log_split
        minimum_development_certainty
        constraint_required
        constraint_method
        constraint_area
        constraint_on
        constraint_source
        designated_area
        base_year_string
        model_years

        Returns
        -------

        """
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
        if d_log is not None and d_log_split is not None:
            print("Including development log...")
            # ## DEVELOPMENT SPLIT
            split_development_households = self.development_log_house_generator(
                d_log,
                d_log_split,
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

        # ## ENSURE WE HAVE NO MINUS POPULATION
        for year in model_years:
            mask = (population[year] < 0)
            population.loc[mask, year] = self.pop_infill

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
        final_population = self.growth_recombination(split_population,
                                                     "base_year_population",
                                                     model_years)

        final_population.sort_values(
            by=[
                "model_zone_id",
                "property_type_id",
                "traveller_type_id"
            ],
            inplace=True
        )
        return final_population

    def production_generation(self,
                              population: pd.DataFrame,
                              area_type: pd.DataFrame,
                              trip_rate: pd.DataFrame,
                              year_list: List[str]
                              ) -> pd.DataFrame:
        """
        #TODO
        """
        population = population.copy()
        area_type = area_type.copy()
        trip_rate = trip_rate.copy()

        # Multiple Population zones belong to each MSOA area  type
        population = pd.merge(
            population,
            area_type,
            on=["model_zone_id"]
        )

        # Calculate the trips of each traveller in an area, based on
        # their trip rates
        trips = pd.merge(
            population,
            trip_rate,
            on=["traveller_type_id", "area_type_id"],
            suffixes=("", "_trip_rate")
        )
        for year in year_list:
            trips.loc[:, year] = trips[year] * trips[year + "_trip_rate"]

        # Extract just the needed columns
        group_by_cols = [
            "model_zone_id",
            "purpose_id",
            "traveller_type_id",
            "soc",
            "ns",
            "area_type_id"
        ]
        needed_columns = group_by_cols.copy()
        needed_columns.extend(year_list)
        trips = trips[needed_columns]

        productions = trips.groupby(
            by=group_by_cols,
            as_index=False
        ).sum()

        return productions

    def convert_to_average_weekday(self,
                                   production_dataframe: pd.DataFrame,
                                   all_years: List[str]
                                   ) -> pd.DataFrame:
        """
        #TODO
        """
        output_dataframe = production_dataframe.copy()

        for year in all_years:
            output_dataframe.loc[:, year] = output_dataframe.loc[:, year] / 5

        return output_dataframe

    def population_grower(self,
                          population_growth: pd.DataFrame,
                          population_values: pd.DataFrame,
                          base_year: str,
                          year_string_list: List[str]
                          ) -> pd.DataFrame:
        # get population growth from base year
        print("Adjusting population growth to base year...")
        population_growth = du.convert_growth_off_base_year(
                population_growth,
                base_year,
                year_string_list
                )
        print("Adjusted population growth to base year!")
        
        
        print("Growing population from base year...")
        grown_population = du.get_growth_values(population_values,
                                                population_growth,
                                                base_year,
                                                year_string_list)
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
        households_growth = du.convert_growth_off_base_year(
                households_growth,
                base_year,
                year_string_list
                )
        print("Adjusted households growth to base year!")
        
        print("Growing households from base year...")
        grown_households = self.get_grown_values(households_values,
                                                 households_growth,
                                                 "base_year_households",
                                                 year_string_list)
        print("Grown households from base year!")
        
        return grown_households

    
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
        
        webtag_certainty_bounds = self.tag_certainty_bounds[
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


def get_land_use_data(land_use_path: str,
                      msoa_path: str,
                      land_use_cols: List[str] = None,
                      zone_col: str = 'msoa_zone_id'
                      ) -> pd.DataFrame:
    """
    Reads in land use outputs and aggregates up to land_use_cols

    Parameters
    ----------
    land_use_path:
        Path to the land use output file to import

    msoa_path:
        Path to the msoa file for converting from msoa string ids to
        integer ids

    land_use_cols:
        The columns to keep in the land use data. Must include msoa_zone_id
        and people. If None, defaults to:
         [
            zone_col,
            'area_type',
            'traveller_type',
            'soc',
            'ns',
            'people'
        ]

    zone_col:
        The name to give to the zone column in the final output

    Returns
    -------
    population:
        Population data segmented by land_use_cols
    """
    # Init
    if land_use_cols is None:
        land_use_cols = [
            'msoa_zone_id',
            'area_type',
            'traveller_type',
            'soc',
            'ns',
            'people'
        ]

    # Set up the columns to keep
    group_cols = land_use_cols.copy()
    group_cols.remove('people')

    # Read in Land use
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore')
        land_use = pd.read_csv(land_use_path)

    # Drop a lot of columns, group and sum the remaining

    land_use = land_use.reindex(land_use_cols, axis=1).groupby(
        group_cols
    ).sum().reset_index().sort_values(land_use_cols).reset_index()
    del group_cols

    # Read in MSOA conversion file
    msoa_zones = pd.read_csv(msoa_path).rename(
        columns={
            'model_zone_code': 'msoa_string'
        }
    )

    # Convert MSOA strings to id numbers
    land_use = pd.merge(land_use,
                        msoa_zones,
                        left_on='msoa_zone_id',
                        right_on='msoa_string')

    # Drop unneeded columns and rename
    land_use = land_use.drop(columns=['msoa_zone_id', 'msoa_string'])
    land_use = land_use.rename(columns={'model_zone_id': zone_col})

    return land_use


def merge_pop_trip_rates(population: pd.DataFrame,
                         merge_cols: List[str],
                         group_cols: List[str],
                         trip_rates_path: str,
                         time_splits_path: str,
                         mean_time_splits_path: str,
                         tp_needed: List[int] = consts.TP_NEEDED,
                         purpose_col: str = 'purpose_id'
                         ) -> pd.DataFrame:
    # Init
    index_cols = group_cols.copy()
    index_cols.append('trips')
    trip_rates = pd.read_csv(trip_rates_path)

    if purpose_col not in group_cols:
        raise ValueError("Cannot find purpose col '%s' in group_cols. "
                         "group_cols: %s"
                         % (purpose_col, str(group_cols)))

    # Get the weekly trip rate for the populations
    purpose_ph = dict()
    all_purposes = trip_rates['p'].drop_duplicates().reset_index(drop=True)
    desc = "Building trip rates by purpose"
    for p in tqdm(all_purposes, desc=desc):
        trip_rate_subset = trip_rates[trip_rates['p'] == p].copy()
        trip_rate_subset = trip_rate_subset.rename(columns={'p': purpose_col})
        ph = population.copy()

        if p in consts.SOC_P:
            # Update ns with none
            ph['ns'] = 'none'
            ph['soc'] = ph['soc'].astype(int)
            # Insurance policy
            trip_rate_subset['ns'] = 'none'
            trip_rate_subset['soc'] = trip_rate_subset['soc'].astype(int)

        elif p in consts.NS_P:
            # Update soc with none
            ph['soc'] = 'none'
            ph['ns'] = ph['ns'].astype(int)
            # Insurance policy
            trip_rate_subset['soc'] = 'none'
            trip_rate_subset['ns'] = trip_rate_subset['ns'].astype(int)

        # Merge and calculate productions
        ph = ph[ph['people'] > 0].copy()
        ph = pd.merge(ph, trip_rate_subset, on=merge_cols)

        print(ph)

        ph['trips'] = ph['trip_rate'] * ph['people']
        ph = ph.drop(['trip_rate'], axis=1)

        print(ph)
        print(group_cols)
        print(index_cols)

        # Group and sum
        ph = ph.reindex(index_cols, axis='columns')
        ph = ph.groupby(group_cols).sum().reset_index()

        # Update dictionary
        purpose_ph[p] = ph

        print(ph)
        print(list(ph))

        sys.exit()
    del trip_rates
    # Results in weekly trip rates by purpose and segmentation

    # ## SPLIT WEEKLY TRIP RATES BY TIME PERIOD ## #
    # Init
    time_splits = pd.read_csv(time_splits_path)
    mean_time_splits = pd.read_csv(mean_time_splits_path)
    merge_cols = ['area_type', 'traveller_type', 'p']

    tp_ph = {}
    desc = 'Splitting trip rates by time period'
    for tp in tqdm(tp_needed, desc=desc):
        needed_cols = merge_cols.copy() + [tp]
        tp_subset = time_splits.reindex(needed_cols, axis='columns').copy()
        tp_mean_subset = mean_time_splits.reindex(['p', tp], axis='columns').copy()

        for p, p_df in purpose_ph.items():
            # Get mean for infill
            tp_mean = tp_mean_subset[tp_mean_subset['p'] == p][tp]

            # Merge and infill
            tp_mat = p_df.copy()
            tp_mat = pd.merge(
                tp_mat,
                tp_subset,
                how='left',
                on=merge_cols
            )
            tp_mat[tp] = tp_mat[tp].fillna(tp_mean)

            # Apply tp split and divide by 5 to get average weekday by tp
            tp_mat['trips'] = (tp_mat['trips'] * tp_mat[tp]) / 5

            # Drop tp col
            tp_mat = tp_mat.drop(tp, axis=1)

            # Add to compilation dict
            tp_ph.update({('p' + str(key) + '_' + tp): tp_mat})

    return productions


def generate_productions(population: pd.DataFrame,
                         merge_cols: List[str],
                         group_cols: List[str],
                         base_year: str,
                         future_years: List[str],
                         trip_rates_path: str,
                         time_splits_path: str,
                         mean_time_splits_path: str,
                         ) -> pd.DataFrame:
    # Init
    all_years = [base_year] + future_years

    yr_ph = dict()
    for year in all_years:
        yr_pop = population.copy().reindex(group_cols + [year], axis='columns')
        yr_pop = yr_pop.rename(columns={year: 'people'})
        yr_prod = merge_pop_trip_rates(
            yr_pop,
            merge_cols=merge_cols,
            group_cols=group_cols,
            trip_rates_path=trip_rates_path,
            time_splits_path=time_splits_path,
            mean_time_splits_path=mean_time_splits_path
        )
        sys.exit()


    return productions


def _nhb_production_internal(hb_pa_import,
                             nhb_trip_rates,
                             year,
                             purpose,
                             mode,
                             segment,
                             car_availability):
    """
      The internals of nhb_production(). Useful for making the code more
      readable due to the number of nested loops needed
    """
    hb_dist = du.get_dist_name(
        'hb',
        'pa',
        str(year),
        str(purpose),
        str(mode),
        str(segment),
        str(car_availability),
        csv=True
    )

    # Seed the nhb productions with hb values
    hb_pa = pd.read_csv(
        os.path.join(hb_pa_import, hb_dist)
    )
    hb_pa = du.expand_distribution(
        hb_pa,
        year,
        purpose,
        mode,
        segment,
        car_availability,
        id_vars='p_zone',
        var_name='a_zone',
        value_name='trips'
    )

    # Aggregate to destinations
    nhb_prods = hb_pa.groupby([
        "a_zone",
        "purpose_id",
        "mode_id"
    ])["trips"].sum().reset_index()

    # join nhb trip rates
    nhb_prods = pd.merge(nhb_trip_rates,
                         nhb_prods,
                         on=["purpose_id", "mode_id"])

    # Calculate NHB productions
    nhb_prods["nhb_dt"] = nhb_prods["trips"] * nhb_prods["nhb_trip_rate"]

    # aggregate nhb_p 11_12
    nhb_prods.loc[nhb_prods["nhb_p"] == 11, "nhb_p"] = 12

    # Remove hb purpose and mode by aggregation
    nhb_prods = nhb_prods.groupby([
        "a_zone",
        "nhb_p",
        "nhb_m",
    ])["nhb_dt"].sum().reset_index()

    return nhb_prods


def nhb_production(hb_pa_import,
                   nhb_export,
                   required_purposes,
                   required_modes,
                   required_soc,
                   required_ns,
                   required_car_availabilities,
                   years_needed,
                   nhb_factor_import,
                   out_fname=consts.NHB_PRODUCTIONS_FNAME
                   ):
    """
    This function builds NHB productions by
    aggregates HB distribution from EFS output to destination

    TODO: Update to use the TMS method - see backlog

    Parameters
    ----------
    required lists:
        to loop over TfN segments

    Returns
    ----------
    nhb_production_dictionary:
        Dictionary containing NHB productions by year
    """
    # Init
    nhb_production_dictionary = dict()

    # Get nhb trip rates
    # Might do the other way - This emits CA segmentation
    nhb_trip_rates = pd.read_csv(
        os.path.join(nhb_factor_import, "IgammaNMHM.csv")
    ).rename(
        columns={"p": "purpose_id", "m": "mode_id"}
    )

    # For every: Year, purpose, mode, segment, ca
    for year in years_needed:
        loop_gen = list(du.segmentation_loop_generator(
            required_purposes,
            required_modes,
            required_soc,
            required_ns,
            required_car_availabilities
        ))
        yearly_nhb_productions = list()
        desc = 'Generating NHB Productions for yr%s' % year
        for purpose, mode, segment, car_availability in tqdm(loop_gen, desc=desc):
            nhb_productions = _nhb_production_internal(
                hb_pa_import,
                nhb_trip_rates,
                year,
                purpose,
                mode,
                segment,
                car_availability
            )
            yearly_nhb_productions.append(nhb_productions)

        # ## Output the yearly productions ## #
        # Aggregate all productions for this year
        yr_nhb_productions = pd.concat(yearly_nhb_productions)
        yr_nhb_productions = yr_nhb_productions.groupby(
            ["a_zone", "nhb_p", "nhb_m"]
        )["nhb_dt"].sum().reset_index()

        # Rename columns from NHB perspective
        yr_nhb_productions = yr_nhb_productions.rename(
            columns={
                'a_zone': 'p_zone',
                'nhb_p': 'p',
                'nhb_m': 'm',
                'nhb_dt': 'trips'
            }
        )

        # Print some audit vals
        # audit = yr_nhb_productions.groupby(
        #     ["p", "m"]
        # )["trips"].sum().reset_index()
        # print(audit)

        # Create year fname
        nhb_productions_fname = '_'.join(
            ["yr" + str(year), out_fname]
        )

        # Output
        yr_nhb_productions.to_csv(
            os.path.join(nhb_export, nhb_productions_fname),
            index=False
        )

        # save to dictionary by year
        nhb_production_dictionary[year] = yr_nhb_productions

    return nhb_production_dictionary
