# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 12:13:07 2019

@author: Sneezy
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd

from typing import List
from typing import Dict

from tqdm import tqdm

import efs_constants as consts
from efs_constrainer import ForecastConstrainer
from demand_utilities import utils as du
# TODO: Move functions that can be static elsewhere.
#  Maybe utils?

# TODO: Tidy up the no longer needed functions -
#  Production model was re-written to use TMS method


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
                'ca'
            ]

        production_group_cols = [zone_col] + merge_cols
        production_cols = production_group_cols + ['purpose_id'] + all_years

        land_use_cols = ['msoa_zone_id'] + merge_cols + ['people']

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
        trip_rates_path = r"Y:\NorMITs Demand\import\tfn_segment_production_params\hb_trip_rates.csv"
        time_splits_path = r"Y:\NorMITs Demand\import\tfn_segment_production_params\hb_time_split.csv"
        mean_time_splits_path = r"Y:\NorMITs Demand\import\tfn_segment_production_params\hb_ave_time_split.csv"
        mode_share_path = r"Y:\NorMITs Demand\import\tfn_segment_production_params\hb_mode_split.csv"

        ntem_control_dir = r'Y:/NorMITs Synthesiser/import/ntem_constraints'
        lad_lookup_dir = 'Y:/NorMITs Synthesiser/import'

        population_metric = 'population'
        constraint_required[0] = False
        constraint_required[1] = False

        # END OF STUFF TO FIX

        if population_metric == "households":
            raise ValueError("Production Model has changed. Households growth "
                             "is not currently supported.")


        # TODO: Convert all of the production mode to use MSOA codes instead
        #  of integers - more descriptive
        # ## BASE YEAR POPULATION ## #
        print("Loading the base year population data...")
        base_year_pop = get_land_use_data(lu_import_path,
                                          msoa_path=msoa_import_path,
                                          land_use_cols=land_use_cols,
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

        # Convert back to MSOA codes for id numbers
        population = du.convert_msoa_naming(
            population,
            msoa_col_name=zone_col,
            msoa_path=msoa_import_path,
            to='string'
        )

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
            group_cols=production_group_cols,
            base_year=base_year,
            future_years=future_years,
            trip_rates_path=trip_rates_path,
            time_splits_path=time_splits_path,
            mean_time_splits_path=mean_time_splits_path,
            mode_share_path=mode_share_path,
            audit_dir=out_path,
            ntem_control_dir=ntem_control_dir,
            lad_lookup_dir=lad_lookup_dir
        )

        print(productions)
        sys.exit()

        # Rename columns as needed

        return productions

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
                      msoa_path: str = None,
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
        integer ids. If left as None, then MSOA codes are returned instead

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

    # Convert to msoa zone numbers if needed
    if msoa_path is not None:
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
    else:
        # Make sure zone col has the correct name
        land_use.rename(columns={'msoa_zone_id': zone_col})

    return land_use


def merge_pop_trip_rates(population: pd.DataFrame,
                         group_cols: List[str],
                         trip_rates_path: str,
                         time_splits_path: str,
                         mean_time_splits_path: str,
                         mode_share_path: str,
                         audit_out: str,
                         ntem_control_path: str = None,
                         lad_lookup_dir: str = None,
                         lad_lookup_name: str = consts.DEFAULT_LAD_LOOKUP,
                         tp_needed: List[int] = consts.TP_NEEDED,
                         traveller_type_col: str = 'traveller_type',
                         ) -> pd.DataFrame:
    """
    TODO: Write merge_pop_trip_rates() doc

    Parameters
    ----------
    population
    group_cols
    trip_rates_path
    time_splits_path
    mean_time_splits_path
    mode_share_path
    audit_out
    ntem_control_path
    lad_lookup_dir
    lad_lookup_name
    tp_needed
    traveller_type_col

    Returns
    -------

    """
    # Init
    do_ntem_control = ntem_control_path is not None and lad_lookup_dir is not None

    group_cols = group_cols.copy()
    group_cols.insert(2, 'p')

    index_cols = group_cols.copy()
    index_cols.append('trips')

    # ## GET WEEKLY TRIP RATE FROM POPULATION ## #
    # Init
    trip_rates = pd.read_csv(trip_rates_path)

    # TODO: Make the production model more adaptable
    # Merge on all possible columns
    tr_cols = list(trip_rates)
    pop_cols = list(population)
    tr_merge_cols = [x for x in tr_cols if x in pop_cols]

    purpose_ph = dict()
    all_purposes = trip_rates['p'].drop_duplicates().reset_index(drop=True)
    desc = "Building trip rates by purpose"
    for p in tqdm(all_purposes, desc=desc):
        trip_rate_subset = trip_rates[trip_rates['p'] == p].copy()
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
        ph = pd.merge(ph, trip_rate_subset, on=tr_merge_cols)

        ph['trips'] = ph['trip_rate'] * ph['people']
        ph = ph.drop(['trip_rate'], axis=1)

        # Group and sum
        ph = ph.reindex(index_cols, axis='columns')
        ph = ph.groupby(group_cols).sum().reset_index()

        # Update dictionary
        purpose_ph[p] = ph
    del trip_rates
    # Results in weekly trip rates by purpose and segmentation

    # ## SPLIT WEEKLY TRIP RATES BY TIME PERIOD ## #
    # Also converts to average weekday trips!
    # Init
    time_splits = pd.read_csv(time_splits_path)
    mean_time_splits = pd.read_csv(mean_time_splits_path)
    tp_merge_cols = ['area_type', 'traveller_type', 'p']

    # Convert tp nums to strings
    tp_needed = ['tp' + str(x) for x in tp_needed]

    tp_ph = dict()
    desc = 'Splitting trip rates by time period'
    for tp in tqdm(tp_needed, desc=desc):
        needed_cols = tp_merge_cols.copy() + [tp]
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
                on=tp_merge_cols
            )
            tp_mat[tp] = tp_mat[tp].fillna(tp_mean)

            # Apply tp split and divide by 5 to get average weekday by tp
            tp_mat['trips'] = (tp_mat['trips'] * tp_mat[tp]) / 5

            # Group and sum
            tp_mat = tp_mat.reindex(index_cols, axis='columns')
            tp_mat = tp_mat.groupby(group_cols).sum().reset_index()

            # Add to compilation dict
            tp_ph[(p, tp)] = tp_mat
    del time_splits
    del mean_time_splits
    del purpose_ph
    # Results in average weekday trips by purpose, tp, and segmentation

    # Quick Audit
    approx_tp_totals = []
    for key, dat in tp_ph.items():
        total = dat['trips'].sum()
        approx_tp_totals.append(total)
    ave_wday = sum(approx_tp_totals)
    print('. Average weekday productions: %.2f' % ave_wday)

    # ## SPLIT AVERAGE WEEKDAY TRIP RATES BY MODE ## #
    # TODO: Apply at MSOA level rather than area type
    # Init
    mode_share = pd.read_csv(mode_share_path)
    m_merge_cols = ['area_type', 'ca', 'p']
    target_modes = ['m1', 'm2', 'm3', 'm5', 'm6']

    # Can get rid of traveller type now - too much detail
    # If we keep it we WILL have memory problems
    group_cols.remove(traveller_type_col)
    index_cols.remove(traveller_type_col)

    m_ph = dict()
    desc = 'Applying mode share splits'
    for m in tqdm(target_modes, desc=desc):
        needed_cols = m_merge_cols.copy() + [m]
        m_subset = mode_share.reindex(needed_cols, axis='columns').copy()

        for (p, tp), dat in tp_ph.items():
            m_mat = dat.copy()

            # Would merge all purposes, but left join should pick out target mode
            m_mat = pd.merge(
                m_mat,
                m_subset,
                how='left',
                on=m_merge_cols
            )

            # Apply m split
            m_mat['trips'] = (m_mat['trips'] * m_mat[m])

            # Reindex cols for efficiency
            m_mat = m_mat.reindex(index_cols, axis='columns')
            m_mat = m_mat.groupby(group_cols).sum().reset_index()

            m_mat = m_mat[m_mat['trips'] > 0]

            m_ph[(p, tp, m)] = m_mat
    del mode_share
    del tp_ph
    # Results in average weekday trips by purpose, tp, mode, and segmentation

    print("Writing topline audit...")
    approx_mode_totals = []
    for key, dat in m_ph.items():
        total = dat['trips'].sum()
        approx_mode_totals.append([key, total])

    # Build topline report
    topline = pd.DataFrame(approx_mode_totals, columns=['desc', 'total'])
    # Split key into components
    topline['p'], topline['tp'], topline['m'] = list(zip(*topline['desc']))
    topline = topline.reindex(['p', 'tp', 'm', 'total'], axis=1)
    topline = topline.groupby(['p', 'tp', 'm']).sum().reset_index()
    topline.to_csv(os.path.join(audit_out), index=False)

    # ## COMPILE ALL MATRICES INTO ONE ## #
    output_ph = list()
    desc = 'Compiling productions'
    for (p, tp, m), dat in tqdm(m_ph.items(), desc=desc):
        dat['p'] = p
        dat['tp'] = tp
        dat['m'] = m
        output_ph.append(dat)
    msoa_output = pd.concat(output_ph)

    # We now need to deal with tp and mode in one big matrix
    group_cols = group_cols + ['tp', 'm']
    index_cols = group_cols.copy()
    index_cols.append('trips')

    # Ensure matrix is in the correct format
    msoa_output = msoa_output.reindex(index_cols, axis='columns')
    msoa_output = msoa_output.groupby(group_cols).sum().reset_index()
    msoa_output['m'] = [int(m[1:]) for m in msoa_output['m']]
    msoa_output['tp'] = [int(tp[2:]) for tp in msoa_output['tp']]
    msoa_output['p'] = msoa_output['p'].astype(int)
    msoa_output['m'] = msoa_output['m'].astype(int)

    if do_ntem_control is not None:
        # Get ntem totals
        # TODO: Depends on the time period - but this is fixed for now
        ntem_totals = pd.read_csv(ntem_control_path)
        ntem_lad_lookup = pd.read_csv(os.path.join(lad_lookup_dir,
                                                   lad_lookup_name))

        print("Performing NTEM constraint...")
        msoa_output, ntem_p, ntem_a = du.control_to_ntem(
            msoa_output,
            ntem_totals,
            ntem_lad_lookup,
            group_cols=['p', 'm'],
            base_value_name='trips',
            ntem_value_name='Productions',
            purpose='hb'
        )

    return msoa_output


def combine_yearly_productions(year_dfs: Dict[str, pd.DataFrame],
                               unique_col: str,
                               purpose_col: str = 'p',
                               purposes: List[int] = None
                               ) -> pd.DataFrame:
    # Init
    keys = list(year_dfs.keys())
    merge_cols = list(year_dfs[keys[0]])
    merge_cols.remove(unique_col)

    if purposes is None:
        purposes = year_dfs[keys[0]]['p'].drop_duplicates().reset_index(drop=True)

    # ## SPLIT MATRICES AND JOIN BY PURPOSE ## #
    purpose_ph = list()
    desc = "Merging productions by purpose"
    for p in tqdm(purposes, desc=desc):

        # Get all the matrices that belong to this purpose
        yr_p_dfs = list()
        for year, df in year_dfs.items():
            temp_df = df[df[purpose_col] == p].copy()
            temp_df = temp_df.rename(columns={unique_col: year})
            yr_p_dfs.append(temp_df)

        # Iteratively merge all matrices into one
        merged_df = yr_p_dfs[0]
        for df in yr_p_dfs[1:]:
            merged_df = pd.merge(
                merged_df,
                df,
                on=merge_cols
            )
        purpose_ph.append(merged_df)
        del yr_p_dfs

    # ## CONCATENATE ALL MERGED MATRICES ## #
    return pd.concat(purpose_ph)


def generate_productions(population: pd.DataFrame,
                         group_cols: List[str],
                         base_year: str,
                         future_years: List[str],
                         trip_rates_path: str,
                         time_splits_path: str,
                         mean_time_splits_path: str,
                         mode_share_path: str,
                         audit_dir: str,
                         ntem_control_dir: str = None,
                         lad_lookup_dir: str = None
                         ) -> pd.DataFrame:
    # Init
    all_years = [base_year] + future_years
    audit_base_fname = 'yr%s_production_topline.csv'
    ntem_base_fname = 'ntem_pa_ave_wday_%s.csv'

    # Generate Productions for each year
    yr_ph = dict()
    for year in all_years:
        if ntem_control_dir is not None:
            ntem_control_path = os.path.join(ntem_control_dir,
                                             ntem_base_fname % year)
        else:
            ntem_control_path = None

        audit_out = os.path.join(audit_dir, audit_base_fname % year)

        yr_pop = population.copy().reindex(group_cols + [year], axis='columns')
        yr_pop = yr_pop.rename(columns={year: 'people'})
        yr_prod = merge_pop_trip_rates(
            yr_pop,
            group_cols=group_cols,
            trip_rates_path=trip_rates_path,
            time_splits_path=time_splits_path,
            mean_time_splits_path=mean_time_splits_path,
            mode_share_path=mode_share_path,
            audit_out=audit_out,
            ntem_control_path=ntem_control_path,
            lad_lookup_dir=lad_lookup_dir,
        )
        print(yr_prod)

        yr_ph[year] = yr_prod

        sys.exit()

    # Join all productions into one big matrix
    productions = combine_yearly_productions(
        yr_ph,
        unique_col='trips'
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
