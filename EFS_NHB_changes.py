# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 09:50:07 2019

@author: Sneezy
"""

from functools import reduce
from typing import List
import datetime
import multiprocessing
import os
import sys
import time

import numpy as np
import pandas as pd

import efs_constants as efs_consts

from efs_constrainer import ForecastConstrainer
from efs_pop_generator import ExternalForecastSystemPopulationGenerator
from efs_worker_generator import ExternalForecastSystemWorkerGenerator
from furness_process import FurnessProcess
from zone_translator import ZoneTranslator

"""
#TODO

# Find where below relative paths are pointing to
# Get rid of cross repo calls
"""

# Not sure where these are pointing to
# sys.path.append("../../../../NorMITs Utilities/Python")
# sys.path.append("../../../../NorMITs Utilities")

sys.path.append("C:/Users/Sneezy/Desktop/GitHub/Normits-Utils")

import nu_error_management as err_check
from nu_sector_reporter_v2 import SectorReporter


class ExternalForecastSystem:
    # constants
    VERSION_ID = "v2-1"
    
    # sub-classes
    efs_pop_generator = None
    efs_worker_generator = None
    efs_constrainer = None
    efs_furness = None
    
    # support utilities tools
    sector_reporter = None
    zone_translator = None
    
    # default value dataframes
    # alternative assumptions are joined onto these to allow for
    # tailoring to different zones without having to create a
    # new value for each one
    default_population_values = pd.DataFrame
    default_population_growth = pd.DataFrame
    default_population_constraint = pd.DataFrame
    default_future_population_ratio = pd.DataFrame
    
    default_households_values = pd.DataFrame
    default_households_growth = pd.DataFrame
    default_households_constraint = pd.DataFrame
    default_housing_type_split = pd.DataFrame
    default_housing_occupancy = pd.DataFrame
    
    default_worker_values = pd.DataFrame
    default_worker_growth = pd.DataFrame
    default_worker_constraint = pd.DataFrame
    default_worker_splits = pd.DataFrame
    
    default_production_trip_rates = pd.DataFrame
    default_hb_mode_time_split = pd.DataFrame
    default_split_handler = pd.DataFrame
    default_traveller_types = pd.DataFrame
    default_attraction_weights = pd.DataFrame
    
    default_value_zones  = pd.DataFrame
    default_area_types = pd.DataFrame
    default_area_grouping  = pd.DataFrame
    
    # other values handled on instantiation
    default_value_zoning = str
    
    # non-year columns
    column_dictionary = {}
    
    def __init__(self,
                 default_population_value_file: str = "base_year_population.csv",
                 default_population_growth_file: str = "future_population_no_splits_growth.csv",
                 default_population_constraint_file: str = "future_population_no_splits.csv",
                 default_future_population_ratio_file: str = "traveller_type/ntem/ntem_traveller_type_ratio.csv",
                 
                 default_households_value_file: str = "base_year_households.csv",
                 default_household_growth_file: str = "future_households_growth.csv",
                 default_households_constraint_file: str = "future_households.csv",
                 default_housing_type_split_file: str = "housing_property_ratio.csv",
                 default_housing_occupancy_file: str = "housing_occupancy.csv",
                 
                 default_worker_value_file: str = "base_year_workers.csv",
                 default_worker_growth_file: str = "future_workers_growth.csv",
                 default_worker_constraint_file: str = "future_workers_constraint_growth.csv",
                 default_worker_ratio_file: str = "future_worker_splits.csv",
                 
                 default_production_trip_rates_file: str = "traveller_type/ntem/ntem_trip_rates.csv",
                 default_hb_mode_split_file: str = "traveller_type/ntem/hb/hb_mode_split.csv",
                 default_hb_mode_time_split_file: str = "traveller_type/ntem/ntem_mode_time_split.csv",
                 default_split_handler_file: str = "traveller_type/ntem/mode_time_splits.csv",
                 default_traveller_types_file: str = "traveller_type/ntem/ntem_traveller_types.csv",
                 default_attraction_weights_file: str = "attractions/future_attraction_weights_i3.csv",
                 
                 default_value_zoning: str = "MSOA",
                 default_value_zones_file: str = "msoa_zones.csv",
                 default_area_types_file: str = "msoa_area_types.csv",
                 default_area_grouping_file: str = "lad_msoa_grouping.csv",
                 default_msoa_area_types_file: str = "msoa_area_types.csv",
                 default_zone_areatype_lookup_file: str = "norms_2015.csv",
                 default_file_location: str = "Y:/EFS/inputs/default/"
                 ):
        """
        #TODO
        """
        print("Initiating External Forecast System...")
        begin_time = time.time()
        current_time = begin_time
        last_time = begin_time
        # TODO: File input checks
        
        self.default_population_values = pd.read_csv(
                default_file_location + default_population_value_file
                )
        self.default_population_growth = pd.read_csv(
                default_file_location + default_population_growth_file
                )
        self.default_population_constraint = pd.read_csv(
                default_file_location + default_population_constraint_file
                )
        self.default_future_population_ratio = pd.read_csv(
                default_file_location + default_future_population_ratio_file                
                )
        
        self.default_households_values = pd.read_csv(
                default_file_location + default_households_value_file
                )
        self.default_households_growth = pd.read_csv(
                default_file_location + default_household_growth_file
                )
        self.default_households_constraint = pd.read_csv(
                default_file_location + default_households_constraint_file
                )
        self.default_housing_type_split = pd.read_csv(
                default_file_location + default_housing_type_split_file
                )
        self.default_housing_occupancy = pd.read_csv(
                default_file_location + default_housing_occupancy_file
                )
        
        self.default_worker_values = pd.read_csv(
                default_file_location + default_worker_value_file
                )
        self.default_worker_growth = pd.read_csv(
                default_file_location + default_worker_growth_file
                )
        self.default_worker_constraint = pd.read_csv(
                default_file_location + default_worker_constraint_file
                )
        self.default_worker_splits = pd.read_csv(
                default_file_location + default_worker_ratio_file
                )
        
        self.default_production_trip_rates = pd.read_csv(
                default_file_location + default_production_trip_rates_file
                )
        self.default_hb_mode_time_split = pd.read_csv(
            default_file_location + default_hb_mode_time_split_file
        )
        self.default_split_handler = pd.read_csv(
            default_file_location + default_split_handler_file
        )
        self.default_hb_mode_split = pd.read_csv(
                default_file_location + default_hb_mode_split_file
                )
        self.default_split_handler = pd.read_csv(
                default_file_location + default_split_handler_file
                )
        self.default_traveller_types = pd.read_csv(
                default_file_location + default_traveller_types_file
                )
        self.default_attraction_weights = pd.read_csv(
                default_file_location + default_attraction_weights_file
                )
        
        self.default_value_zoning = default_value_zoning
        self.default_value_zones = pd.read_csv(
                default_file_location + default_value_zones_file
                )
        self.default_area_types = pd.read_csv(
                default_file_location + default_area_types_file
                )
        self.default_area_grouping = pd.read_csv(
                default_file_location + default_area_grouping_file
                )  
        self.default_msoa_area_types = pd.read_csv(
                default_file_location + default_msoa_area_types_file
                )
        
        self.default_zone_areatype_lookup = pd.read_csv(
                default_file_location + default_zone_areatype_lookup_file
                )
        self.column_dictionary = {
                "base_year_population": [
                        "model_zone_id",
                        "base_year_population"
                        ],
                "base_year_households": [
                        "model_zone_id",
                        "base_year_households"
                        ],
                "base_year_workers": [
                        "model_zone_id",
                        "base_year_workers"
                        ],
                "population": [
                        "model_zone_id"
                        ],
                "population_ratio": [
                        "model_zone_id",
                        "property_type_id",
                        "traveller_type_id"
                        ],
                "households": [
                        "model_zone_id"
                        ],
                "employment": [
                        "model_zone_id"
                        ],
                "housing_occupancy": [
                        "model_zone_id",
                        "property_type_id"
                        ],
                "production_trips": [
                        "p",
                        "traveller_type",
                        "soc",
                        "ns",
                        "area_type"
                        ],
                "mode_split": [
                        "area_type_id",
                        "car_availability_id",
                        "purpose_id",
                        "mode_id",
                        ],
                # "mode_time_split": [
                #         "purpose_id",
                #         "traveller_type_id",
                #         "area_type_id",
                #         "mode_time_split"
                #         ],
                "employment_ratio": [
                        "model_zone_id",
                        "employment_class"
                        ],
                "attraction_weights": [
                        "employment_class",
                        "purpose_id"
                        ]
                }
        
        #TODO
        ## @@MSP / TY - NEED TO REMOVE FROM FINAL VERSION!! 
        # cut input down to 10 zones
        self.default_population_values = self.default_population_values.loc[
            self.default_population_values['model_zone_id'].isin([1,2,1055,1056,1057,1058,1059,1060,1061,1062])]

        self.default_population_growth = self.default_population_growth.loc[
            self.default_population_growth['model_zone_id'].isin([1,2,1055,1056,1057,1058,1059,1060,1061,1062])]
        
        self.default_population_constraint = self.default_population_constraint.loc[
            self.default_population_constraint['model_zone_id'].isin([1,2,1055,1056,1057,1058,1059,1060,1061,1062])]

        self.default_future_population_ratio = self.default_future_population_ratio.loc[
            self.default_future_population_ratio['model_zone_id'].isin([1,2,1055,1056,1057,1058,1059,1060,1061,1062])]        

        self.default_households_values = self.default_households_values.loc[
            self.default_households_values['model_zone_id'].isin([1,2,1055,1056,1057,1058,1059,1060,1061,1062])]

        self.default_households_growth = self.default_households_growth.loc[
            self.default_households_growth['model_zone_id'].isin([1,2,1055,1056,1057,1058,1059,1060,1061,1062])]

        self.default_households_constraint = self.default_households_constraint.loc[
            self.default_households_constraint['model_zone_id'].isin([1,2,1055,1056,1057,1058,1059,1060,1061,1062])]
        
        self.default_housing_type_split = self.default_housing_type_split.loc[
            self.default_housing_type_split['model_zone_id'].isin([1,2,1055,1056,1057,1058,1059,1060,1061,1062])]
        
        self.default_housing_occupancy = self.default_housing_occupancy.loc[
            self.default_housing_occupancy['model_zone_id'].isin([1,2,1055,1056,1057,1058,1059,1060,1061,1062])]

        self.default_worker_values = self.default_worker_values.loc[
            self.default_worker_values['model_zone_id'].isin([1,2,1055,1056,1057,1058,1059,1060,1061,1062])]

        self.default_worker_growth = self.default_worker_growth.loc[
            self.default_worker_growth['model_zone_id'].isin([1,2,1055,1056,1057,1058,1059,1060,1061,1062])]
        
        self.default_worker_constraint = self.default_worker_constraint.loc[
            self.default_worker_constraint['model_zone_id'].isin([1,2,1055,1056,1057,1058,1059,1060,1061,1062])]
        
        self.default_worker_splits = self.default_worker_splits.loc[
            self.default_worker_splits['model_zone_id'].isin([1,2,1055,1056,1057,1058,1059,1060,1061,1062])]

        self.default_value_zones = self.default_value_zones.loc[
            self.default_value_zones['model_zone_id'].isin([1,2,1055,1056,1057,1058,1059,1060,1061,1062])]

        self.default_area_types = self.default_area_types.loc[
            self.default_area_types['model_zone_id'].isin([1,2,1055,1056,1057,1058,1059,1060,1061,1062])]
        
        self.default_area_grouping = self.default_area_grouping.loc[
            self.default_area_grouping['model_zone_id'].isin([1,2,1055,1056,1057,1058,1059,1060,1061,1062])]
                    
        self.efs_constrainer = ForecastConstrainer()
        self.efs_pop_generator = ExternalForecastSystemPopulationGenerator()
        self.efs_worker_generator = ExternalForecastSystemWorkerGenerator()
        self.furness_process = FurnessProcess()
        self.sector_reporter = SectorReporter()
        self.zone_translator = ZoneTranslator()
    
        print("External Forecast System initiated!")
        last_time = current_time
        current_time = time.time()
        print("Initialisation took: "
              + str(round(current_time - last_time, 2))
              + " seconds."
              )
        
    def run(self,
            base_year: int = 2018,
            #future_years: List[int] = [2033, 2035, 2050],
            future_years: List[int] = [2033, 2035, 2050],
            desired_zoning: str = "MSOA",
            alternate_population_base_year_file: str = None,
            alternate_households_base_year_file: str = None,
            alternate_worker_base_year_file: str = None,
            alternate_population_growth_assumption_file: str = None,
            alternate_households_growth_assumption_file: str = None,
            alternate_worker_growth_assumption_file: str = None,
            alternate_population_split_file: str = None,
            distribution_method: str = "Furness",
            distribution_location: str = "Y:/EFS/inputs/distributions",
            distributions: dict = None,
                    
            # levels: purpose, car availability, mode, time
            purposes_needed: List[int] = [
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    ],
            soc_needed: List[int] = [
                    0,
                    1,
                    2,
                    3,
                    ],
            ns_needed: List[int] = [
                    1,
                    2,
                    3,
                    4,
                    5,                   
                    ],
            car_availabilities_needed: List[int] = [
                    1,
                    2,
                    ],
            modes_needed: List[int] = [
                    1,
                    2,              
                    3,
                    5,
                    6,
                    ],
            times_needed: List[int] = [
                    1,
                    2,
                    3,
                    4,
                    ],
            development_log_file: str = None,
            development_log_split_file: str = None,
            minimum_development_certainty: str = "MTL",
            integrating_development_log: bool = False,
            population_metric: str = "Households", # Households, Population
            constraint_required: List[bool] = [
                    True, # initial population metric constraint
                    True, # post-development constraint
                    True, # secondary post-development constraint used for matching HH pop
                    False, # initial worker metric constraint
                    False, # secondary worker metric constraint
                    False, # final trip based constraint
                    ],
            constraint_method: str = "Percentage", # Percentage, Average
            constraint_area: str = "Designated", # Zone, Designated, All
            constraint_on: str = "Growth", # Growth, All
            constraint_source: str = "Grown Base", # Default, Grown Base, Model Grown Base
            outputting_files: bool  = True,
            performing_sector_totals: bool = True,
            output_location: str = None
            ) -> None:
        """
        The main function for the External Forecast System.
        
        Performs the following pieces of functionality:
            - Generates trip production from population metrics
            - Generates trip attraction weight from worker metrics
            - Furnesses these to generate a distribution using Synthesiser distributions
        
        Parameters
        ----------
        base_year:
            This is the base year used for rebalancing growth and constraint
            metrics. Used throughout the program.
            Default input is: 2018
            Possible input is any integer between 2011 to 2051.
            
        future_years:
            These are the future years used for model outputs.
            Default input is: [2033, 2035, 2050]
            Possible input is a list containing any number of integers between
            2011 to 2051.
            
        desired_zoning:
            The desired output zoning for this data set.
            Default input is: "MSOA".
            Possible input is any string, preferably one that matches to a
            zoning system with a corresponding translation.
            
        alternate_population_base_year_file:
            A file location (including file suffix) containing an alternate
            population for the base year. This file does not need full
            alternate population metrics, just needs it for the appropriate
            zones.
            Default input is: None.
            Possible input is any string which refers to a file location.
            
        alternate_households_base_year_file:
            A file location (including file suffix) containing an alternate
            number of households for the base year. This file does not need full
            alternate households metrics, just needs it for the appropriate
            zones.
            Default input is: None.
            Possible input is any string which refers to a file location.
            
        alternate_worker_base_year_file:
            A file location (including file suffix) containing an alternate
            number of workers for the base year. This file does not need full
            alternate worker metrics, just needs it for the appropriate
            zones.
            Default input is: None.
            Possible input is any string which refers to a file location.
            
        alternate_population_growth_assumption_file:
            A file location (including file suffix) containing an alternate
            population growth for some future years. This file does not need full
            alternate population growth metrics, just needs it for the appropriate
            zones and years.
            Default input is: None.
            Possible input is any string which refers to a file location.
            
        alternate_households_growth_assumption_file:
            A file location (including file suffix) containing an alternate
            households growth for some future years. This file does not need full
            alternate households growth metrics, just needs it for the appropriate
            zones and years.
            Default input is: None.
            Possible input is any string which refers to a file location.
            
        alternate_worker_growth_assumption_file:
            A file location (including file suffix) containing an alternate
            workers growth for some future years. This file does not need full
            alternate worker growth metrics, just needs it for the appropriate
            zones and years.
            Default input is: None.
            Possible input is any string which refers to a file location.
            
        alternate_population_split_file:
            A file location (including file suffix) containing an alternate
            population split file. This *does* require it for every zone as it
            will be used to generate full new segmentation (i.e. NPR segments).
            Default input is: None.
            Possible input is any string which refers to a file location.
            
        distribution_method:
            The method to be used for distributing the trips.
            Default input is: "Furness".
            Possible inputs are: "Furness".
            
        distribution_location:
            The primary location for all the distributions.
            Default input is: "Y:/EFS/inputs/distributions".
            Possible input is any file location folder.
            
        distributions:dict
            A series of nested dictionary containing all the distributions
            and their appropriate purpose / car availiability / mode / time
            period splits.
            For example, to access purpose 1, car availability 1, mode 3, time
            period 1, distributions[1][1][3][1] is the correct input. Note that
            Synthesiser does not split time periods into separate files so
            currently time period is a series of copied files which have time
            periods split out by a dataframe call.
            Default input is a series of nested dictionaries.
            Possible input is any dictionary corresponding to the correct
            order.
            
        purposes_needed:
            What purposes are needed on distribution.
            Default input is: [1, 2, 3, 4, 5, 6, 7, 8]
            Possible input is a list containing integers corresponding to the
            purpose IDs.
            
        car_availabilities_needed:
            What car availabilities are needed on distribution.
            Default input is: [1, 2]
            Possible input is a list containing integers corresponding to the
            car availability IDs.
            
        modes_needed:
            What modes are needed on distribution.
            Default input is: [3, 6]
            Possible input is a list containing integers corresponding to the
            mode IDs.
            
        times_needed:
            What time periods are needed on distribution.
            Default input is: [1, 2, 3, 4]
            Possible input is a list containing integers corresponding to the
            time period IDs.
            
        development_log_file:
            A file location for the development log.
            Default input is: None
            Possible input is any file location folder.
            
        development_log_split_file:
            A file location for the housing stock split for the development log.
            Default input is: None
            Possible input is any file location folder.
            
        minimum_development_certainty:
            A string for the minimum development certainty required from the 
            development log.
            Default input is: "MTL" # More than likely
            Possible inputs are: "NC", "MTL", "RF", "H"
            
        integrating_development_log:
            Whether the development log is going to be used.
            Default input is: False
            Possible inputs are: True, False
            
        population_metric:
            What metric to use for population generation.
            Default input is: "Households"
            Possible inputs are: "Households", "Population"
            
        constraint_required:
            What constraints are required and where. The list position
            correlates to:
                - 0: Initial population metric constraint
                - 1: Post-development constraint
                - 2: Post-population constraint
                - 3: Initial worker metric constraint
                - 4: Secondary worker metric constraint
                - 5: Final trip-based constraint
            Default input is: [True, True, True, False, False, False]
            Possible inputs are any list of six booleans.
            
        constraint_method:
            What constraint method is to be used.
                - "Percentage": Reduce the non-constraint values by a percentage.
                - "Average": Reduce the non-constraint values by an average overflow.
            Default input is: "Percentage"
            Possible inputs are: "Percentage", "Average"
            
        constraint_area:
            What constraint area is to be used for balancing.
                - "Zone": Each zone is its own balancing area. Functionally matches perfectly to constraint.
                - "Designated": Each 'designated' grouping is a balancing area.
                - "All": All areas are combined as a balancing area.
            Default input is: "Designated"
            Possible inputs are: "Zone", "Designated", "All"
            
        constraint_on:
            Where the constraint is to be applied.
                - "Growth": Only constraint growth, not full amount.
                - "All": Constrain on all.
            Default input is: "Growth"
            Possible inputs are: "Growth", "All"
            
        constraint_source:
            Where to source the constraint from.
                - "Default": 'Default' constraint values, i.e. raw values from NTEM currently.
                - "Grown Base": New model base including default (NTEM) growth values to be used as constraint.
                - "Model Grown Base": Model base and model growth to be used as a constraint to restrict developments.
            Default input is: "Default"
            Possible inputs are: "Default", "Grown Base", "Model Grown Base"
            
        outputting_files:
            Whether files are being output.
            Default input is: True
            Possible inputs are: False, True
            
        performing_sector_totals:
            Whether sector totals are being output.
            Default input is: True
            Possible inputs are: False, True
            
        output_location:
            Where files are to be output.
            Default input is: None
            Possible input is any file location folder.
            
        Return
        ----------
        None:
            run() method does not provide any returns. This run method either
            outputs to file or saves within the class structure.
            
        Future Improvements
        ----------
            - Include more forms of distribution than just Furness.
            - Use purposes needed / car availabilities needed / modes needed / times needed to reduce the amount of calculations to be done.
        """
        ### TIME SET UP ###
        if distributions is None:
            distributions = efs_consts.EFS_RUN_DISTRIBUTIONS_DICT

        begin_time = time.time()
        current_time = begin_time
        last_time = begin_time
        
        ### INPUT CHECKS ###
        print("Starting input checks...")
        constraint_method = constraint_method.lower()
        constraint_area = constraint_area.lower()
        constraint_on = constraint_on.lower()
        constraint_source = constraint_source.lower()
        distribution_method = distribution_method.lower()
        population_metric = population_metric.lower()
        minimum_development_certainty = minimum_development_certainty.upper()
        
        year_string_list = [
                str(base_year)
                ]
        
        year_string_list.extend(
                [str(i) for i in future_years]
                )
        
        if (
                integrating_development_log
                and
                (
                        development_log_file is None
                        or
                        development_log_split_file is None
                        )
                ):
            print("If integrating development log then both a development "
                  + "log and development log split file need to be provided, "
                  + "ending process.")
            exit(1)
            
        
        base_year_population_columns = self.column_dictionary["base_year_population"]
        base_year_households_columns = self.column_dictionary["base_year_households"]
        
        base_year_workers_columns = self.column_dictionary["base_year_workers"]
        
        population_columns = self.column_dictionary["population"]
        population_columns.extend(year_string_list)
        
        households_columns = self.column_dictionary["households"]
        households_columns.extend(year_string_list)
        
        housing_occupancy_columns = self.column_dictionary["housing_occupancy"]
        housing_occupancy_columns.extend(year_string_list)
        
        employment_columns = self.column_dictionary["employment"]
        employment_columns.extend(year_string_list)
        
        population_ratio_columns = self.column_dictionary["population_ratio"]
        population_ratio_columns.extend(year_string_list)
        
        production_trip_columns = self.column_dictionary["production_trips"]
        production_trip_columns.extend(year_string_list)

        mode_split_columns = self.column_dictionary["mode_split"]
        mode_split_columns.extend(year_string_list)
        
        # mode_time_split_columns = self.column_dictionary["mode_time_split"]
        # mode_time_split_columns.extend(year_string_list)
        
        employment_ratio_columns = self.column_dictionary["employment_ratio"]
        employment_ratio_columns.extend(year_string_list)
        
        attraction_weight_columns = self.column_dictionary["attraction_weights"]
        attraction_weight_columns.extend(year_string_list)

        print("No known errors in the inputs!")
        last_time = current_time
        current_time = time.time()
        print("Input checks took: "
              + str(round(current_time - last_time, 2))
              + " seconds."
              )
        
        ### GET DATA ###
        if (
                # check if any of these are not none
                not all (
                        x is None for x in [
                                alternate_population_base_year_file,
                                alternate_households_base_year_file,
                                alternate_worker_base_year_file,
                                alternate_population_growth_assumption_file,
                                alternate_households_growth_assumption_file,
                                alternate_worker_growth_assumption_file
                                ]
                        )
                ):
            print("Need to integrate alternative assumptions.")
            print("Integrating alternate assumptions...")
            ## ALTERNATE ASSUMPTION INTEGRATION ##
            [
                    population_values,
                    households_values,
                    worker_values,
                    population_growth,
                    households_growth,
                    worker_growth
                    ] = self.integrate_alternate_assumptions(
                    alternate_population_base_year_file,
                    alternate_households_base_year_file,
                    alternate_worker_base_year_file,
                    alternate_population_growth_assumption_file,
                    alternate_households_growth_assumption_file,
                    alternate_worker_growth_assumption_file,
                    base_year_population_columns,
                    base_year_workers_columns,
                    population_columns,
                    employment_columns
                    )
            
            population_values = population_values[base_year_population_columns]
            households_values = households_values[base_year_households_columns]
            worker_values = worker_values[base_year_workers_columns]
            population_growth = population_growth[population_columns]
            households_growth = households_growth[households_columns]
            worker_growth = worker_growth[employment_columns]
            
            ## TODO: Remove self references and integrate into alternate assumptions
            ## TODO: Alternate population split read in
            population_split = self.default_future_population_ratio[population_ratio_columns].copy()
            housing_type_split = self.default_housing_type_split[housing_occupancy_columns].copy()
            housing_occupancy = self.default_housing_occupancy[housing_occupancy_columns].copy()
            trip_rates = self.default_production_trip_rates[production_trip_columns].copy().rename(
                columns = {
                        "traveller_type": "traveller_type_id",
                        "area_type": "area_type_id"
                        }
                )
            hb_mode_split = self.default_hb_mode_split[mode_split_columns].copy() 
            msoa_area_types = self.default_msoa_area_types.copy()            
            zone_areatype_lookup = self.default_zone_areatype_lookup.copy()
            # hb_mode_time_split = self.hb_mode_time_split[mode_time_split_columns].copy()
            worker_split = self.default_worker_splits[employment_ratio_columns].copy()
            # split_handler = self.split_handler.copy()
            
            car_association = self.default_traveller_types[[
                    "cars",
                    "traveller_type"
                    ]].copy().rename(
                    columns = {
                            "traveller_type": "traveller_type_id"
                            }
                    )
                    
            car_association["car_availability_id"] = 0
            no_car_mask = (car_association["cars"] == 0)
            
            car_association[no_car_mask]["car_availability_id"] = 1
            car_association[not (no_car_mask)]["car_availability_id"] = 2
            
            print("Integrated alternate assumptions!")
            last_time = current_time
            current_time = time.time()
            print("Integrating alternate assumptions took: "
                  + str(round(current_time - last_time, 2))
                  + " seconds."
                  )
        else:
            ## COPY OVER VALUES ##
            print("No need to integrate alternative assumptions.")
            print("Reading in default values...")
            population_values = self.default_population_values[
                    base_year_population_columns
                    ].copy()
            population_growth = self.default_population_growth[
                    population_columns
                    ].copy()
            population_split = self.default_future_population_ratio[
                    population_ratio_columns
                    ].copy()
            
            households_values = self.default_households_values[
                    base_year_households_columns
                    ].copy()
            households_growth = self.default_households_growth[
                    households_columns
                    ].copy()
            housing_type_split = self.default_housing_type_split[
                    housing_occupancy_columns
                    ].copy()
            housing_occupancy = self.default_housing_occupancy[
                    housing_occupancy_columns
                    ].copy()
            
            worker_values = self.default_worker_values[
                    base_year_workers_columns
                    ].copy()
            worker_growth = self.default_worker_growth[
                    employment_columns
                    ].copy()
            worker_split = self.default_worker_splits[
                    employment_ratio_columns
                    ].copy()
            
            trip_rates = self.default_production_trip_rates[
                    production_trip_columns
                    ].copy().rename(
                        columns = {
                            "traveller_type": "traveller_type_id",
                            "area_type": "area_type_id",
                            "p": "purpose_id"
                            }
                        )
            hb_mode_split = self.default_hb_mode_split[
                    mode_split_columns
                    ].copy()
            msoa_area_types = self.default_msoa_area_types.copy()            
            zone_areatype_lookup = self.default_zone_areatype_lookup.copy()
            
            # make norms_2015_AreaType_Lookup table
            
            zone_areatype_lookup = zone_areatype_lookup.merge(
                msoa_area_types, left_on= "msoa_zone_id", right_on= "model_zone_id"
                )
            zone_areatype_lookup = zone_areatype_lookup.groupby(['norms_2015_zone_id','area_type_id']).size().to_frame('count').reset_index()
            zone_areatype_lookup = zone_areatype_lookup.sort_values(
                by=[
                    'count', 
                    'area_type_id'
                    ], 
                ascending=[False, True]).drop_duplicates(
                    subset=[
                        'norms_2015_zone_id'
                        ]) 
            zone_areatype_lookup = zone_areatype_lookup[[
                'norms_2015_zone_id',
                'area_type_id'
                ]].sort_values('norms_2015_zone_id')    
            # zone_areatype_lookup.sort_values('norms_2015_zone_id').to_csv(lookup_location + "norms_2015_AreaType_Lookup.csv", index=False)
            
            # .rename(
            #             columns = {
            #                 "area_type": "area_type_id",
            #                 "ca": "car_availability_id",
            #                 "p": "purpose_id",
            #                 "m1": "1",
            #                 "m2": "2",
            #                 "m3": "3",
            #                 "m5": "5",
            #                 "m6": "6",
            #                 }
            #             )
                        
            # hb_mode_time_split = self.hb_mode_time_split[
            #         mode_time_split_columns
            #         ].copy()
            attraction_weights = self.default_attraction_weights[
                    attraction_weight_columns
                    ].copy()
            # split_handler = self.split_handler.copy()
            
            car_association = self.default_traveller_types[[
                    "cars",
                    "traveller_type"
                    ]].copy().rename(
                    columns = {
                            "traveller_type": "traveller_type_id"
                            }
                    )
                    
            car_association["car_availability_id"] = 0
            no_car_mask = (car_association["cars"] == "0")
            
            # set up ids (-no_car_mask is the inversion of no_car_mask)
            car_association.loc[
                    no_car_mask,
                    "car_availability_id"
                    ] = 1
            car_association.loc[
                    -no_car_mask,
                    "car_availability_id"
                    ] = 2
            
            car_association = car_association[
                    [
                            "traveller_type_id",
                            "car_availability_id"
                            ]
                    ]
                
            print("Read-in default values!")
            last_time = current_time
            current_time = time.time()
            print("Reading in default values took: "
                  + str(round(current_time - last_time, 2))
                  + " seconds."
                  )
        
        ### D-LOG READ-IN
        if (integrating_development_log):
            development_log = pd.read_csv(development_log_file)
            development_log_split = pd.read_csv(development_log_split_file)
        else:
            development_log = None
            development_log_split = None
        
        ### CONSTRAINT BUILDING
        if (constraint_source == "default"):
            print("Constraint 'default' selected, retrieving constraint "
                  + "data...")
            population_constraint = self.default_population_constraint[population_columns].copy()
            population_constraint = self.efs_constrainer.convert_constraint_off_base_year(
                population_constraint,
                str(base_year),
                year_string_list
                )
             
            households_constraint = self.default_households_constraint[households_columns].copy()
            households_constraint = self.efs_constrainer.convert_constraint_off_base_year(
                households_constraint,
                str(base_year),
                year_string_list
                )
             
            worker_constraint = self.default_worker_constraint[employment_columns].copy()
            worker_constraint = self.efs_constrainer.convert_constraint_off_base_year(
                worker_constraint,
                str(base_year),
                year_string_list
                )
             
            print("Constraint retrieved!")
            last_time = current_time
            current_time = time.time()
            print("Constraint retrieval took: "
                  + str(round(current_time - last_time, 2))
                  + " seconds."
                  )
            
        elif constraint_source == "grown base":
            print("Constraint 'grown base' source selected, growing given "
                  + "base by default growth factors...")
            population_constraint = self.default_population_growth[population_columns].copy()
            
            population_constraint = self.convert_growth_off_base_year(
                    population_constraint,
                    str(base_year),
                    year_string_list
                    )
            population_constraint = self.get_grown_values(
                    population_values,
                    population_constraint,
                    "base_year_population",
                    year_string_list
                    )
            
            households_constraint = self.default_households_growth[households_columns].copy()
            
            households_constraint = self.convert_growth_off_base_year(
                    households_constraint,
                    str(base_year),
                    year_string_list
                    )
            households_constraint = self.get_grown_values(
                    households_values,
                    households_constraint,
                    "base_year_households",
                    year_string_list
                    )
            
            worker_constraint = self.default_worker_growth[employment_columns].copy()
            
            worker_constraint = self.convert_growth_off_base_year(
                    worker_constraint,
                    str(base_year),
                    year_string_list
                    )
            worker_constraint = self.get_grown_values(
                    worker_values,
                    worker_constraint,
                    "base_year_workers",
                    year_string_list
                    )
            print("Constraint generated!")
            last_time = current_time
            current_time = time.time()
            print("Constraint generation took: "
                  + str(round(current_time - last_time, 2))
                  + " seconds."
                  )
            
        elif constraint_source == "model grown base":
            print("Constraint 'model grown base' source selected, this will "
                  + "be created later...")
            households_constraint = None
            population_constraint = None
            ##TODO: Remember to do this
        
        ### POPULATION GENERATION ###
        print("Generating population...")
        final_population = self.efs_pop_generator.run(
            minimum_development_certainty=minimum_development_certainty,
            population_metric=population_metric,
            constraint_required=constraint_required,
            constraint_method=constraint_method,
            constraint_area=constraint_area, constraint_on=constraint_on,
            constraint_source=constraint_source)
        print("Population generated!")
        last_time = current_time
        current_time = time.time()
        print("Population generation took: "
              + str(round(current_time - last_time, 2))
              + " seconds."
              )
        
        ### WORKER GENERATION ###
        print("Generating workers...")
        final_workers = self.efs_worker_generator.run(
            minimum_development_certainty=minimum_development_certainty,
            integrate_dlog=integrating_development_log,
            constraint_required=constraint_required,
            constraint_method=constraint_method,
            constraint_area=constraint_area, constraint_on=constraint_on,
            constraint_source=constraint_source)
        print("Workers generated!")
        last_time = current_time
        current_time = time.time()
        print("Workers generation took: "
              + str(round(current_time - last_time, 2))
              + " seconds."
              )
        
        ### PRODUCTION GENERATION ###
        print("Generating production...")
        production_trips = self.production_generation(
                final_population,
                self.default_area_types,
                trip_rates,
                car_association,
                year_string_list
                )
        production_trips = self.convert_to_average_weekday(
                production_trips,
                year_string_list
                )
        print("Productions generated!")
        last_time = current_time
        current_time = time.time()
        print("Production generation took: "
              + str(round(current_time - last_time, 2))
              + " seconds."
              )
        
#        print("Applying time-mode splits...")
#        split_production_trips = self.mode_time_split_application(
#                production_trips,
#                hb_mode_time_split,
#                year_string_list
#                )
#        print("Time-mode splits applied!")
#        last_time = current_time
#        current_time = time.time()
#        print("Time-mode splitting took: "
#              + str(round(current_time - last_time, 2))
#              + " seconds."
#              )
        
        print("Convert traveller type id to car availability id...")
        ca_production_trips = self.generate_car_availability(
                production_trips,
                car_association,
                year_string_list,
                [
                        "model_zone_id",
                        "purpose_id",
                        "car_availability_id",
                        "soc",
                        "ns"
#                        "mode_time_split"
                        ]
                )
        print("Converted to car availability!")
        last_time = current_time
        current_time = time.time()
        print("Car availability conversion took: "
              + str(round(current_time - last_time, 2))
              + " seconds."
              )
        
#        print("Reattaching mode-time split IDs to productions...")
#        mode_time_productions = self.reattach_mode_time_ids(
#                split_ca_production_trips,
#                split_handler,
#                year_string_list,
#                [
#                        "model_zone_id",
#                        "purpose_id",
#                        "car_availability_id",
#                        "mode_id",
#                        "time_period_id"
#                        ]
#                )
#        print("Reattached mode-time split IDs to productions!")
#        last_time = current_time
#        current_time = time.time()
#        print("Reattachment of mode-time split IDs took: "
#              + str(round(current_time - last_time, 2))
#              + " seconds."
#              )
        
        ### ATTRACTION GENERATION & MATCHING ###
        print("Generating attractions...")
        attraction_dataframe = self.attraction_generation(
                final_workers,
                attraction_weights,
                year_string_list
                )
        print("Attractions generated!")
        last_time = current_time
        current_time = time.time()
        print("Attraction generation took: "
              + str(round(current_time - last_time, 2))
              + " seconds."
              )
                
        print("Generating attraction weights...")
        attraction_weights = self.generate_attraction_weights(
                attraction_dataframe,
                year_string_list
                )
        print("Attraction weights generated!")
        last_time = current_time
        current_time = time.time()
        print("Attraction weight generation took: "
              + str(round(current_time - last_time, 2))
              + " seconds."
              )
        
#        print("Matching attractions...")
#        matched_attraction_dataframe = self.match_attractions_to_productions(
#                attraction_weights,
#                production_trips,
#                year_string_list
#                )
#        print("Attractions matched!")
#        last_time = current_time
#        current_time = time.time()
#        print("Attraction matching took: "
#              + str(round(current_time - last_time, 2))
#              + " seconds."
#              )
        
        ### ZONE TRANSLATION ###
        if (desired_zoning != self.default_value_zoning):
            # need to translate
            print("Need to translate zones.")
            print("Translating from: " + self.default_value_zoning)
            print("Translating to: " + desired_zoning)
            # read in translation dataframe
            path = "Y:/EFS/inputs/default/zone_translation"
            path = path + "/" + desired_zoning + ".csv"
            translation_dataframe = pd.read_csv(
                    path
                    )
            
            converted_productions = self.zone_translator.run(
                    ca_production_trips,
                    translation_dataframe,
                    self.default_value_zoning,
                    desired_zoning,
                    non_split_columns = [
                            "model_zone_id",
                            "purpose_id",
                            "car_availability_id",
                            "soc",
                            "ns"                            
#                            "mode_id",
#                            "time_period_id"
                            ]
                    )
            
            converted_pure_attractions =  self.zone_translator.run(
                    attraction_dataframe,
                    translation_dataframe,
                    self.default_value_zoning,
                    desired_zoning,
                    non_split_columns = [
                            "model_zone_id",
                            "purpose_id"
                            ]
                    )
            
            converted_attractions = self.zone_translator.run(
                    attraction_weights,
                    translation_dataframe,
                    self.default_value_zoning,
                    desired_zoning,
                    non_split_columns = [
                            "model_zone_id",
                            "purpose_id"
                            ]
                    )
            
            print("Zone translation completed!")
            last_time = current_time
            current_time = time.time()
            print("Zone translation took: "
                  + str(round(current_time - last_time, 2))
                  + " seconds."
                  )
        else:
            converted_productions = ca_production_trips.copy()
            converted_attractions = attraction_weights.copy()
        # check point 
        # converted_productions.to_csv("Y:/EFS/check/converted_productions.csv", index=False)
        # converted_attractions.to_csv("Y:/EFS/check/converted_attractions.csv", index=False)
        
        ### DISTRIBUTION ###
        if distribution_method == "furness":
            print("Generating distributions...")
            final_distribution_dictionary = self.distribute_dataframe(
                    production_dataframe=converted_productions,
                    attraction_weights_dataframe=converted_attractions,
                    mode_split_dataframe=hb_mode_split,
                    zone_areatype_lookup=zone_areatype_lookup,
                    required_purposes=purposes_needed,
                    required_soc=soc_needed,
                    required_ns=ns_needed,
                    required_car_availabilities=car_availabilities_needed,
                    required_modes=modes_needed,
                    required_times=times_needed,
                    year_string_list=future_years,
                    # year_string_list = year_string_list,
                    distribution_dataframe_dict=distributions,
                    distribution_file_location=distribution_location,
                    )
            print("Distributions generated!")
            last_time = current_time
            current_time = time.time()
            print("Distribution generation took: "
                  + str(round(current_time - last_time, 2))
                  + " seconds."
                  )
        
        ### SECTOR TOTALS ###
        sector_totals = self.sector_reporter.calculate_sector_totals(
                converted_productions,
                grouping_metric_columns = year_string_list,
                zone_system_name = "norms_2015",
                zone_system_file = "Y:/EFS/inputs/default/norms_2015.csv",
                sector_grouping_file = "Y:/EFS/inputs/default/zone_translation/tfn_level_one_sectors_norms_grouping.csv"
                )
        
        pm_sector_total_dictionary = {}
        
        for purpose in purposes_needed:
            # for mode in modes_needed:
            #     mask = ((converted_productions["purpose_id"] == purpose))
                        ##&
                        ##(converted_productions["mode_id"] == mode))
                        
              # pm_productions = converted_productions[mask].copy()
            pm_productions = converted_productions.copy()
                
            pm_sector_totals = self.sector_reporter.calculate_sector_totals(
                pm_productions,
                grouping_metric_columns = year_string_list,
                zone_system_name = "norms_2015",
                zone_system_file = "Y:/EFS/inputs/default/norms_2015.csv",
                sector_grouping_file = "Y:/EFS/inputs/default/zone_translation/tfn_level_one_sectors_norms_grouping.csv"
                )
                
            key_string = (
                # "mode"
                # +
                # str(mode)
                # +
                "purpose"
                +
                str(purpose)
                )
                
            pm_sector_total_dictionary[key_string] = pm_sector_totals
        
        ### OUTPUTS ###
        ## TODO: Properly integrate this
        if (outputting_files):
            if (output_location != None):
                print("Saving files to: " + output_location)
                date = datetime.datetime.now()
                if (constraint_required):
                    path = (
                            output_location
                            + self.VERSION_ID + " "
                            + "External Forecast System Output"
                            + " - "
                            + desired_zoning
                            + " - "
                            + date.strftime("%d-%m-%y")
                            + " - "
                            + "PM " + population_metric[0]
                            + " - "
                            + "CM" + constraint_method[0]
                            + " - "
                            + "CA " + constraint_area[0]
                            + " - "
                            + "CO " + constraint_on[0]
                            + " - "
                            + "CS " + constraint_source[0]
                            )
                else:
                    path = (
                            output_location
                            + self.VERSION_ID + " "
                            + "External Forecast System Output"
                            + " - "
                            + desired_zoning
                            + " - "
                            + date.strftime("%d-%m-%y")
                            + " - "
                            + "PM " + population_metric
                            )
                    
                if not os.path.exists(path):
                    os.mkdir(path)
                
                path = path + "/"
                for key, distribution in final_distribution_dictionary.items():
                    print("Saving distribution: " + key)
                    distribution.to_csv(
                            path
                            +
                            key
                            +
                            ".csv",
                            index = False
                            )
                    print("Saved distribution: " + key)
                
                final_population.to_csv(
                        path + "EFS_MSOA_population.csv",
                        index = False
                        )
                
                final_workers.to_csv(
                        path + "EFS_MSOA_workers.csv",
                        index = False
                        )
                
                ca_production_trips.to_csv(
                        path + "MSOA_production_trips.csv",
                        index = False
                        )
                
                attraction_dataframe.to_csv(
                        path + "MSOA_attractions.csv",
                        index = False
                        )
                
                converted_productions.to_csv(
                        path + desired_zoning + "_production_trips.csv",
                        index = False
                        )
                
                converted_pure_attractions.to_csv(
                        path + desired_zoning + "_attractions.csv",
                        index = False
                        )
                
                sector_totals.to_csv(
                        path + desired_zoning + "_sector_totals.csv",
                        index = False
                        )
                
                for key, sector_total in pm_sector_total_dictionary.items():
                    print("Saving sector total: " + key)
                    sector_total.to_csv(
                            path
                            +
                            "sector_total_"
                            +
                            key
                            +
                            ".csv",
                            index = False
                            )
                    print("Saved sector total: " + key)
                
                explanation_file = open(path + "input_parameters.txt", "w")
    
                inputs = [
                        "Base Year: " + str(base_year) + "\n",
                        "Future Years: " + str(future_years) + "\n",
                        "Zoning System: " + desired_zoning + "\n",
                        "Alternate Population Base Year File: " + str(alternate_population_base_year_file) + "\n",
                        "Alternate Households Base Year File: " + str(alternate_households_base_year_file) + "\n",
                        "Alternate Workers Base Year File: " + str(alternate_worker_base_year_file) + "\n",
                        "Alternate Population Growth File: " + str(alternate_population_growth_assumption_file) + "\n",
                        "Alternate Households Growth File: " + str(alternate_households_growth_assumption_file) + "\n",
                        "Alternate Workers Growth File: " + str(alternate_worker_growth_assumption_file) + "\n",
                        "Alternate Population Split File: " + str(alternate_population_split_file) + "\n",
                        "Distribution Method: " + distribution_method + "\n",
                        "Distribution Location: " + distribution_location + "\n",
                        "Purposes Used: " + str(purposes_needed) + "\n",
                        "Car Availabilities Used: " + str(car_availabilities_needed) + "\n",
                        "Modes Used: " + str(modes_needed) + "\n",
                        "Times Used: " + str(times_needed) + "\n",
                        "Development Log Integrated: " + str(False) + "\n",
                        "Minimum Development Certainty: " + str(minimum_development_certainty) + "\n",
                        "Population Metric: " + population_metric + "\n",
                        "Constraints Used On: " + str(constraint_required) + "\n",
                        "Constraint Method: " + constraint_method + "\n",
                        "Constraint Area: " + constraint_area + "\n",
                        "Constraint On: " + constraint_on + "\n",
                        "Constraint Source: " + constraint_source + "\n"
                        ]
    
                explanation_file.writelines(inputs)
                explanation_file.close()
                # TODO: Store output files into output location
                
            else:
                print("No output location given. Saving files to local storage "
                      + "for future usage.")
                self.sector_totals = sector_totals
                # TODO: Store output files into local storage (class storage)
        else:
            print("Not outputting files, saving files to local storage for "
                  + "future usage.")
            self.sector_totals = sector_totals
            # TODO: Store output files into local storage (class storage)
        
    
    def integrate_alternate_assumptions(self,
                                        alternate_population_base_year_file: str,
                                        alternate_households_base_year_file: str,
                                        alternate_worker_base_year_file: str,
                                        alternate_population_growth_assumption_file: str,
                                        alternate_households_growth_assumption_file: str,
                                        alternate_worker_growth_assumption_file: str,
                                        base_year_population_columns: List[str],
                                        base_year_households_columns: List[str],
                                        base_year_workers_columns: List[str],
                                        population_columns: List[str],
                                        households_columns: List[str],
                                        employment_columns: List[str]
                                        ) -> List[pd.DataFrame]:
        """
        #TODO
        """
        ### READ IN ALTERNATE ASSSUMPTIONS ###
        if (alternate_population_base_year_file != None):
            alternate_population_base_year = pd.read_csv(alternate_population_base_year_file)
        else:
            alternate_population_base_year = self.default_population_values.copy()
            
        if (alternate_households_base_year_file != None):
            alternate_households_base_year = pd.read_csv(alternate_households_base_year_file)
        else:
            alternate_households_base_year = self.default_households_values.copy()
        
        if (alternate_worker_base_year_file != None):
            alternate_worker_base_year = pd.read_csv(alternate_worker_base_year_file)
        else:
            alternate_worker_base_year = self.default_worker_values.copy()
            
        if (alternate_population_growth_assumption_file != None):
            alternate_population_growth = pd.read_csv(alternate_population_growth_assumption_file)
        else:
            alternate_population_growth = self.default_population_growth.copy()
            
        if (alternate_households_growth_assumption_file != None):
            alternate_households_growth = pd.read_csv(alternate_households_growth_assumption_file)
        else:
            alternate_households_growth = self.default_households_growth.copy()
            
        if (alternate_worker_growth_assumption_file != None):
            alternate_worker_growth = pd.read_csv(alternate_worker_growth_assumption_file)
        else:
            alternate_worker_growth = self.default_worker_growth.copy()
        
        ### ZONE TRANSLATION OF ALTERNATE ASSUMPTIONS ###
        # TODO: Maybe allow zone translation, maybe requiring sticking to base
        
        ### COMBINE BASE & ALTERNATE ASSUMPTIONS ###
        # alternate population base
        if (alternate_population_base_year_file != None):
            alternate_population_base_year_zones = alternate_population_base_year["model_zone_id"].values
            default_population_values = self.default_population_values[base_year_population_columns].copy()
            default_population_values.loc[
                    default_population_values["model_zone_id"].isin(alternate_population_base_year_zones),
                    "base_year_population"
                    ] = alternate_population_base_year["base_year_population"].values
                    
            alternate_population_base_year = default_population_values
            
        # alternate households base
        if (alternate_households_base_year_file != None):
            alternate_households_base_year_zones = alternate_households_base_year["model_zone_id"].values
            default_households_values = self.default_households_values[base_year_households_columns].copy()
            default_households_values.loc[
                    default_households_values["model_zone_id"].isin(alternate_households_base_year_zones),
                    "base_year_population"
                    ] = alternate_households_base_year["base_year_households"].values
                    
            alternate_households_base_year = default_households_values
            
        # alternate worker base
        if (alternate_worker_base_year_file != None):
            alternate_worker_base_year = pd.read_csv(alternate_worker_base_year_file)
            alternate_worker_base_year_zones = alternate_worker_base_year["model_zone_id"].values
            default_worker_values = self.default_worker_values[base_year_population_columns].copy()
            default_worker_values.loc[
                    default_worker_values["model_zone_id"].isin(alternate_worker_base_year_zones),
                    "base_year_population"
                    ] = alternate_worker_base_year["base_year_workers"].values
                    
            alternate_worker_base_year = default_worker_values
            
        # alternate population growth
        if (alternate_population_growth_assumption_file != None):
            alternate_population_growth = pd.read_csv(alternate_population_growth_assumption_file)
            alternate_population_growth_zones = alternate_population_growth["model_zone_id"].values
            columns = alternate_population_growth.columns[1:].values
            
            # replacing missing values
            alternate_population_growth = alternate_population_growth.replace(
                    '*',
                    alternate_population_growth.replace(
                            ['*'],
                            [None]
                            )
                    )
            
            for year in columns:
                alternate_population_growth[year] = alternate_population_growth[year].astype(float)
                alternate_population_growth[year + "_difference"] = None
                
            default_population_growth = self.default_population_growth.copy()
            
            for zone in alternate_population_growth_zones:
                for year in columns:
                    default_value = default_population_growth.loc[
                            default_population_growth["model_zone_id"] == zone,
                            year
                            ].values[0]
                    
                    new_value =  alternate_population_growth.loc[
                            alternate_population_growth["model_zone_id"] == zone,
                            year
                            ].values[0]
                    
                    difference = new_value - default_value
                    
                    alternate_population_growth.loc[
                            alternate_population_growth["model_zone_id"] == zone,
                            year + "_difference"
                            ] = difference
                            
                    if (pd.notna(difference)):
                        default_population_growth.loc[
                                default_population_growth["model_zone_id"] == zone,
                                year: default_population_growth.columns[-1]
                                ] = default_population_growth.loc[
                                    default_population_growth["model_zone_id"] == zone,
                                    year: default_population_growth.columns[-1]
                                    ] + difference
                            
                    
            alternate_population_growth = default_population_growth
            
        # alternate households growth
        if (alternate_households_growth_assumption_file != None):
            alternate_households_growth = pd.read_csv(alternate_households_growth_assumption_file)
            alternate_households_growth_zones = alternate_households_growth["model_zone_id"].values
            columns = alternate_households_growth.columns[1:].values
            
            # replacing missing values
            alternate_households_growth = alternate_households_growth.replace(
                    '*',
                    alternate_households_growth.replace(
                            ['*'],
                            [None]
                            )
                    )
            
            for year in columns:
                alternate_households_growth[year] = alternate_households_growth[year].astype(float)
                alternate_households_growth[year + "_difference"] = None
                
            default_households_growth = self.default_households_growth.copy()
            
            for zone in alternate_households_growth_zones:
                for year in columns:
                    default_value = default_households_growth.loc[
                            default_households_growth["model_zone_id"] == zone,
                            year
                            ].values[0]
                    
                    new_value =  alternate_households_growth.loc[
                            alternate_households_growth["model_zone_id"] == zone,
                            year
                            ].values[0]
                    
                    difference = new_value - default_value
                    
                    alternate_households_growth.loc[
                            alternate_households_growth["model_zone_id"] == zone,
                            year + "_difference"
                            ] = difference
                            
                    if (pd.notna(difference)):
                        default_households_growth.loc[
                                default_households_growth["model_zone_id"] == zone,
                                year: default_households_growth.columns[-1]
                                ] = default_households_growth.loc[
                                    default_households_growth["model_zone_id"] == zone,
                                    year: default_households_growth.columns[-1]
                                    ] + difference
                            
                    
            alternate_households_growth = default_households_growth
            
        # alternate worker growth
        if (alternate_worker_growth_assumption_file != None):
            alternate_worker_growth = pd.read_csv(alternate_worker_growth_assumption_file)
            alternate_worker_growth_zones = alternate_worker_growth["model_zone_id"].values
            columns = alternate_worker_growth.columns[1:].values
            
            # replacing missing values
            alternate_worker_growth = alternate_worker_growth.replace(
                    '*',
                    alternate_worker_growth.replace(
                            ['*'],
                            [None]
                            )
                    )
            
            for year in columns:
                alternate_worker_growth[year] = alternate_worker_growth[year].astype(float)
                alternate_worker_growth[year + "_difference"] = None
                
            default_worker_growth = self.default_worker_growth.copy()
            
            for zone in alternate_worker_growth_zones:
                for year in columns:
                    default_value = default_worker_growth.loc[
                            default_worker_growth["model_zone_id"] == zone,
                            year
                            ].values[0]
                    
                    new_value = alternate_worker_growth.loc[
                            alternate_worker_growth["model_zone_id"] == zone,
                            year
                            ].values[0]
                    
                    difference = new_value - default_value
                    
                    alternate_worker_growth.loc[
                            alternate_worker_growth["model_zone_id"] == zone,
                            year + "_difference"
                            ] = difference
                            
                    if (pd.notna(difference)):
                        default_worker_growth.loc[
                                default_worker_growth["model_zone_id"] == zone,
                                year: default_worker_growth.columns[-1]
                                ] = default_worker_growth.loc[
                                    default_worker_growth["model_zone_id"] == zone,
                                    year: default_worker_growth.columns[-1]
                                    ] + difference
                            
            alternate_population_growth = default_population_growth
        
        return [
                alternate_population_base_year,
                alternate_households_base_year,
                alternate_worker_base_year,
                alternate_population_growth,
                alternate_households_growth,
                alternate_worker_growth
                ]
    
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
            columns_required = all_years,
            base_year = base_year:
                x[columns_required] / x[base_year],
                axis = 1)
        
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
        
        grown_dataframe.loc[
        :,
        all_years
        ] = grown_dataframe.apply(
            lambda x,
            columns_required = all_years,
            base_year = base_year_string:
                (x[columns_required] - 1) * x[base_year],
                axis = 1)
        
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
        metric_dataframe.loc[
        :,
        all_years
        ] = metric_dataframe.apply(
            lambda x,
            columns_required = all_years,
            metric_column = metric_column_name:
                x[columns_required] + x[metric_column_name],
                axis = 1)
        
        ## drop the unnecessary metric column
        metric_dataframe = metric_dataframe.drop(
                labels = metric_column_name,
                axis = 1
                )
        
        return metric_dataframe
    
    def convert_to_average_weekday(self,
                             production_dataframe: pd.DataFrame,
                             all_years: List[str]
                             ) -> pd.DataFrame:
        """
        #TODO
        """
        output_dataframe = production_dataframe.copy()
        
        for year in all_years:
            output_dataframe.loc[
                    :,
                    year
                    ] = (
                        output_dataframe.loc[
                        :,
                        year
                        ]
                        /
                        5
                    )
        
        return output_dataframe
    
    def segment_dataframe(self,
                         combined_dataframe: pd.DataFrame,
                         split_dataframe: pd.DataFrame,
                         year_list: List[str]
                         ) -> pd.DataFrame:
        """
        #TODO
        """
        combined_dataframe = combined_dataframe.copy()
        split_dataframe = split_dataframe.copy()
        
        segmented_dataframe = pd.merge(
            split_dataframe,
            combined_dataframe,
            on = ["model_zone_id"],
            suffixes = ("_spl", "")
            )

        for year in year_list:
            segmented_dataframe.loc[
            :,
            year
            ] = segmented_dataframe.apply(
                lambda x,
                columns_required = year,
                year_spl = year + "_spl":
                    x[year] * x[year_spl],
                    axis = 1)
        
        split_names = [s + "_spl" for s in year_list]
            
        segmented_dataframe = segmented_dataframe.drop(
                labels = split_names,
                axis = 1
                )
            
        return segmented_dataframe
    
    def production_generation(self,
                         population_dataframe: pd.DataFrame,
                         area_type_dataframe: pd.DataFrame,
                         trip_rate_dataframe: pd.DataFrame,
                         car_association_dataframe: pd.DataFrame,
                         year_list: List[str]
                         ) -> pd.DataFrame:
        """
        #TODO
        """
        population_dataframe = population_dataframe.copy()
        area_type_dataframe = area_type_dataframe.copy()
        trip_rate_dataframe = trip_rate_dataframe.copy()
        car_association_dataframe = car_association_dataframe.copy()
        
        population_dataframe = pd.merge(
            population_dataframe,
            area_type_dataframe,
            on = ["model_zone_id"]
            )
        
        trip_dataframe = pd.merge(
                population_dataframe,
                trip_rate_dataframe,
                on = [
                        "traveller_type_id",
                        "area_type_id"
                        ],
                suffixes = (
                        "", "_trips"
                        )
                )

        for year in year_list:
            trip_dataframe.loc[
                    :,
                    year
                    ] = (
                    trip_dataframe[year]
                    *
                    trip_dataframe[year + "_trips"]
                    )
        
        needed_columns = [
                "model_zone_id",
                "purpose_id",
                "traveller_type_id",
                "soc",
                "ns",                
                "area_type_id"
                ]
        
        needed_columns.extend(year_list)
        
        trip_dataframe = trip_dataframe[needed_columns]
        
        trip_dataframe = trip_dataframe.groupby(
                by = [
                        "model_zone_id",
                        "purpose_id",
                        "traveller_type_id",
                        "soc",
                        "ns",
                        "area_type_id"
                        ],
                as_index = False
                ).sum()
            
        return trip_dataframe
    
    def mode_time_split_application(self,
                                    production_dataframe: pd.DataFrame,
                                    mode_time_split_dataframe: pd.DataFrame,
                                    year_list: List[str]
                                    ) -> pd.DataFrame:
        """
        #TODO
        """
        production_dataframe = production_dataframe.copy()
        mode_time_split_dataframe = mode_time_split_dataframe.copy()
        
        trip_dataframe = pd.merge(
                production_dataframe,
                mode_time_split_dataframe,
                on = [
                        "purpose_id",
                        "traveller_type_id",
                        "area_type_id"
                        ],
                suffixes = (
                        "", "_splits"
                        )
                )

        for year in year_list:
            trip_dataframe.loc[
                    :,
                    year
                    ] = (
                    trip_dataframe[year]
                    *
                    trip_dataframe[year + "_splits"]
                    )
        
        needed_columns = [
                "model_zone_id",
                "purpose_id",
                "traveller_type_id",
                "area_type_id",
                "mode_time_split"
                ]
        
        needed_columns.extend(year_list)
        
        trip_dataframe = trip_dataframe[needed_columns]
        
        trip_dataframe = trip_dataframe.groupby(
                by = [
                        "model_zone_id",
                        "purpose_id",
                        "traveller_type_id",
                        "area_type_id",
                        "mode_time_split"
                        ],
                as_index = False
                ).sum()
            
        return trip_dataframe
    
    def attraction_generation(self,
                              worker_dataframe: pd.DataFrame,
                              attraction_weight: pd.DataFrame,
                              year_list: List[str]
                              ) -> pd.DataFrame:
        """
        #TODO
        """
        worker_dataframe = worker_dataframe.copy()
        attraction_weight = attraction_weight.copy()
        
        attraction_dataframe = pd.merge(
                worker_dataframe,
                attraction_weight,
                on = [
                        "employment_class"
                        ],
                suffixes = (
                        "", "_weights"
                        )
                )

        for year in year_list:
            attraction_dataframe.loc[
                    :,
                    year
                    ] = (
                    attraction_dataframe[year]
                    *
                    attraction_dataframe[year + "_weights"]
                    )

        needed_columns = [
                "model_zone_id",
                "purpose_id"
                ]
        
        needed_columns.extend(year_list)
        
        attraction_dataframe = attraction_dataframe[needed_columns]
        
        attraction_dataframe = attraction_dataframe.groupby(
                by = [
                        "model_zone_id",
                        "purpose_id"
                        ],
                as_index = False
                ).sum()
            
        return attraction_dataframe
    
    def generate_attraction_weights(self,
                              attraction_dataframe: pd.DataFrame,
                              year_list: List[str]
                              ) -> pd.DataFrame:
        """
        #TODO
        """
        attraction_weights = attraction_dataframe.copy()
        purposes = attraction_weights["purpose_id"].unique()
        
        for purpose in purposes:
            for year in year_list:
                mask = (attraction_weights["purpose_id"] == purpose)
                attraction_weights.loc[
                        mask,
                        year
                        ] = (
                            attraction_weights.loc[
                                mask,
                                year
                                ]
                            /
                            attraction_weights.loc[
                                mask,
                                year
                                ].sum()
                        )
        
        return attraction_weights
    
    def match_attractions_to_productions(self,
                                         attraction_weights: pd.DataFrame,
                                         production_dataframe: pd.DataFrame,
                                         year_list: List[str]
                                         ) -> pd.DataFrame:
        """
        #TODO
        """
        attraction_weights = attraction_weights.copy()
        production_dataframe = production_dataframe.copy()
        
        purposes = attraction_weights["purpose_id"].unique()
        
        attraction_dataframe = pd.merge(
                attraction_weights,
                production_dataframe,
                on = [
                        "model_zone_id",
                        "purpose_id"
                        ],
                suffixes = (
                        "", "_productions"
                        )
                )
        
        for purpose in purposes:
            for year in year_list:
                mask = (attraction_dataframe["purpose_id"] == purpose)
                attraction_dataframe.loc[
                        mask,
                        year
                        ] = (
                            attraction_weights.loc[
                                mask,
                                year
                                ]
                            *
                            attraction_weights.loc[
                                mask,
                                year + "_productions"
                                ].sum()
                        )
                
        needed_columns = [
                "model_zone_id",
                "purpose_id"
                ]
        
        needed_columns.extend(year_list)
        
        attraction_dataframe = attraction_dataframe[needed_columns]
        
        attraction_dataframe = attraction_dataframe.groupby(
                by = [
                        "model_zone_id",
                        "purpose_id"
                        ],
                as_index = False
                ).sum()
            
        
        return attraction_dataframe
    
    def generate_car_availability(self,
                                  traveller_based_dataframe: pd.DataFrame,
                                  car_availability: pd.DataFrame,
                                  year_string_list: List[str],
                                  required_columns: List[str]
                                  ) -> pd.DataFrame:
        """
        #TODO
        """
        traveller_based_dataframe = traveller_based_dataframe.copy()
        car_availability = car_availability.copy()
        required_combined_columns = required_columns.copy()
        
        required_combined_columns.extend(year_string_list)
        
        car_availability_dataframe = pd.merge(
                traveller_based_dataframe,
                car_availability,
                on = [
                        "traveller_type_id"
                        ]
                )
        
        car_availability_dataframe = car_availability_dataframe[
                required_combined_columns
                ]
        
        car_availability_dataframe = car_availability_dataframe.groupby(
                by = required_columns,
                as_index = False
                ).sum()
            
        return car_availability_dataframe
    
    def reattach_mode_time_ids(self,
                                  split_dataframe: pd.DataFrame,
                                  time_split_types_dataframe: pd.DataFrame,
                                  year_string_list: List[str],
                                  required_columns: List[str]
                                  ) -> pd.DataFrame:
        """
        #TODO
        """
        split_dataframe = split_dataframe.copy()
        time_split_types_dataframe = time_split_types_dataframe.copy()
        required_combined_columns = required_columns.copy()
        
        required_combined_columns.extend(year_string_list)
        
        reattached_dataframe = pd.merge(
                split_dataframe,
                time_split_types_dataframe,
                on = [
                        "mode_time_split"
                        ]
                )
        
        reattached_dataframe = reattached_dataframe[
                required_combined_columns
                ]
            
        return reattached_dataframe
    
    def distribute_dataframe(self,
                             production_dataframe: pd.DataFrame,
                             attraction_weights_dataframe: pd.DataFrame,
                             mode_split_dataframe: pd.DataFrame,
                             zone_areatype_lookup: pd.DataFrame,
                             required_purposes: List[int],
                             required_soc: List[int],
                             required_ns: List[int],
                             required_car_availabilities: List[int],
                             required_modes: List[int],
                             required_times: List[int],
                             year_string_list: List[str],
                             distribution_dataframe_dict: dict,
                             distribution_file_location: str,
                             number_of_iterations: int = 1,
                             replace_zero_values: bool = True,
                             constrain_on_production: bool = True,
                             constrain_on_attraction: bool = True,
                             zero_replacement_value: float = 0.00001
                             ) -> pd.DataFrame:
        """
        #TODO
        """
        production_dataframe = production_dataframe.copy()
        attraction_weights_dataframe = attraction_weights_dataframe.copy() 
        mode_split_dataframe = mode_split_dataframe.copy()      
        zone_areatype_lookup = zone_areatype_lookup.copy()
        final_distribution_dictionary = {}
        required_segments = []
        distribution_dataframe_list = []
        
        mode_split_dataframe = mode_split_dataframe.merge(
            zone_areatype_lookup, 
            on=
                "area_type_id",              
            ).rename(
                columns={
                    "norms_2015_zone_id": "p_zone"}
                ).drop_duplicates()
        # make table wide to long
        # mode_split_dataframe = mode_split_dataframe.copy().melt(
        #     id_vars=['area_type_id', 'car_availability_id', 'purpose_id'], value_vars=['1', '2', '3', '5', '6'], 
        #     var_name='mode_id', value_name='factor'
        #     )
        # mode_split_dataframe.to_csv(r'Y:\EFS\inputs\default\traveller_type\hb_mode_split.csv', index=False)

        for year in year_string_list:
            for purpose in required_purposes:                
                if purpose in (1,2):
                    required_segments = required_soc
                else:
                    required_segments = required_ns   
                for segment in required_segments:                
                    car_availability_dataframe = pd.DataFrame
                    first_iteration = True
                    for car_availability in required_car_availabilities:
                        # for tp in required_times:
                        distribution_dataframe = pd.read_csv(                           
                        # distribution_dataframe_tp = pd.read_csv(
                            distribution_file_location
                            +
                            (distribution_dataframe_dict
                             [purpose]
                             [segment]
                             [car_availability]                             
            #                                    [mode]
            #                                    [time]
                                )
                                # + str(tp) 
                            # + ".csv"
                            )
                                    
                        print()  
                        
                            # make table long
                        # distribution_dataframe_tp = pd.melt(
                        distribution_dataframe = pd.melt(
                            distribution_dataframe, id_vars=['norms_zone_id'], var_name='a_zone', value_name='dt'
                            )                           
                        # distribution_dataframe_list.append(distribution_dataframe_tp)
                        # distribution_dataframe = pd.concat(distribution_dataframe_list)

                        distribution_dataframe = distribution_dataframe.rename(
                            columns = {
                               "norms_zone_id": "p_zone",
                               "dt": "seed_values"
                               }
                            )
                        # convert column object to int 
                        distribution_dataframe['a_zone'] = distribution_dataframe['a_zone'].astype(int)                       
                        distribution_dataframe = distribution_dataframe.groupby(
                            by = ["p_zone", "a_zone"],
                            as_index = False
                            ).sum()                          

                        # distribution_dataframe = distribution_dataframe[[
                        #         "p_zone",
                        #         "a_zone",
                        #         "dt"
                        #         ]]
                        # print()
                
                        #TODO
                        ## @@MSP / TY - NEED TO REMOVE FROM FINAL VERSION!!        
                        
                        distribution_dataframe = distribution_dataframe[distribution_dataframe['p_zone'].isin([259,267,268,270,275,1171,1173])]
                        distribution_dataframe = distribution_dataframe[distribution_dataframe['a_zone'].isin([259,267,268,270,275,1171,1173])]                        

                        if purpose in (1,2):
                            production_input = production_dataframe[
                                (production_dataframe["purpose_id"] == purpose)
                                &
                                (production_dataframe["car_availability_id"] == car_availability)
                                &
                                (production_dataframe["soc"] == str(segment))
            #                                   &
            #                                   (production_dataframe["mode_id"] == mode)
            #                                   &
            #                                   (production_dataframe["time_period_id"] == time)
                            ][
                                [
                                    "model_zone_id",
                                    str(year)
                                    ]
                            ].rename(
                                    columns = {
                                        str(year): "production_forecast"
                                        }
                                    )
                        else:
                            production_input = production_dataframe[
                                (production_dataframe["purpose_id"] == purpose)
                                &
                                (production_dataframe["car_availability_id"] == car_availability)
                                &
                                (production_dataframe["ns"] == str(segment)) 
                              ][
                                [
                                    "model_zone_id",
                                    str(year)
                                    ]
                            ].rename(
                                    columns = {
                                        str(year): "production_forecast"
                                        }
                                    )                                                
                                    
                        attraction_input = attraction_weights_dataframe[
                        attraction_weights_dataframe["purpose_id"] == purpose
                            ][
                                [
                                    "model_zone_id",
                                    str(year)                                    ]
                                ].rename(
                                columns = {
                                str(year): "attraction_forecast"
                                }
                                )
                                    
                        print()
                        final_distribution = self.furness_process.run(),


                                             final_distribution["purpose_id"] = purpose
                        final_distribution["car_availability_id"] = car_availability
                        final_distribution["soc_id"] = "none"
                        final_distribution["ns_id"] = "none"                       
            #                           final_distribution["mode_id"] = mode
            #                           final_distribution["time_period_id"] = time  
                        if purpose in (1,2):
                            final_distribution["soc_id"] = segment
                            final_distribution_all_mode_dict = (
                             "hb_pa"
                             +
                             "_yr"
                             +
                             str(year)
                             +
                             "_p"
                             +
                             str(purpose)                                    
                             +
                             "_soc"
                             +
                             str(segment)
                             +                                    
                             "_ca"
                             +
                             str(car_availability)
                              +
                              # "_24hr"
                              # +
                              ".csv"
                              )                           

                        else:
                            final_distribution["ns_id"] = segment   
                            final_distribution_all_mode_dict = (
                             "hb_pa"
                             +
                             "_yr"
                             +
                             str(year)
                             +
                             "_p"
                             +
                             str(purpose)                                    
                             +
                             "_ns"
                             +
                             str(segment)
                             +                                    
                             "_ca"
                             +
                             str(car_availability)
                              +
                              # "_24hr"
                              # +
                              ".csv"
                              )                           

                        # tfn mode split                            
                        final_distribution = final_distribution.merge(
                            mode_split_dataframe,
                            on = [
                                "p_zone",
                                "purpose_id",
                                "car_availability_id"                                
                                ]
                            )
                        # calculate dt by mode                    
                        final_distribution["dt"] = final_distribution["dt"] * final_distribution[str(year)]
                        final_distribution = final_distribution[[
                            "p_zone",
                            "a_zone",
                            "purpose_id",
                            "car_availability_id",
                            "mode_id",
                            "soc_id",
                            "ns_id",
                            "dt"
                            ]]
                        # .rename(columns={
                        #         "purpose_id": "p",
                        #         "car_availability_id": "ca",
                        #         "mode_id": "m",
                        #         "soc_id": "soc",
                        #         "ns_id": "ns",
                        #         "dt": "trips"
                        #         })
                                                                                                                                               
                        #output all modes demand for NHB
                        # final_distribution_all_mode = final_distribution.copy()  
                        final_distribution_all_mode_path = "C:/Users/Sneezy/Desktop/EFS"
                        final_distribution.to_csv(final_distribution_all_mode_path + final_distribution_all_mode_dict, index=False)
                        # loop over reqiured modes
                        for mode in required_modes:
                            final_distribution_mode = final_distribution[final_distribution["mode_id"] == mode]
                            if purpose in (1,2):
                                # final_distribution_mode["soc_id"] = segment
                                dict_string = (
                                        "hb_pa"
                                        +
                                        "_yr"
                                        +
                                        str(year)
                                        +
                                        "_p"
                                        +
                                        str(purpose)                                    
                                        +                                    
                                        "_m"
                                        +
                                        str(mode)                                    
                                        +
                                        "_soc"
                                        +
                                        str(segment)
                                        +                                    
                                        "_ca"
                                        +
                                        str(car_availability)
                #                                    +
                #                                    "_time"
                #                                    +
                #                                    str(time)
                                        # +
                                        # "_24hr"
                                        )                           
                            else:
                                # final_distribution_mode["ns_id"] = segment                                                                                                            
                                dict_string = (
                                        "hb_pa"
                                        +
                                        "_yr"
                                        +
                                        str(year)
                                        +
                                        "_p"
                                        +
                                        str(purpose)                                    
                                        +                                    
                                        "_m"
                                        +
                                        str(mode)                                    
                                        +
                                        "_ns"
                                        +
                                        str(segment)
                                        +                                   
                                        "_ca"
                                        +
                                        str(car_availability)
                #                                    +
                #                                    "_time"
                #                                    +
                #                                    str(time)
                                        # +
                                        # "_24hr"
                                        )                           
                                                            
                            final_distribution_dictionary[dict_string] = final_distribution_mode
                                
                            print("Distribution " + dict_string + " complete!")
                            if (first_iteration == True):
                                car_availability_dataframe = final_distribution_mode
                                first_iteration = False
                            else:
                                car_availability_dataframe = car_availability_dataframe.append(
                                    final_distribution_mode
                                    )
        
        return final_distribution_dictionary


def main():
    efs = ExternalForecastSystem()
    efs.run(
        constraint_source="Default",
        desired_zoning="norms_2015",
        output_location="C:/Users/Sneezy/Desktop/EFS"
    )


if __name__ == '__main__':
    main()
