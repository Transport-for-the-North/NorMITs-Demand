# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 09:50:07 2019

@author: Sneezy
"""
# Built-ins
import os
import sys
import time
import datetime
from typing import List

# External libs
import numpy as np
import pandas as pd

# self imports
import efs_constants as efs_consts
import furness_process as fp
from efs_constrainer import ForecastConstrainer
from efs_production_generator import EFSProductionGenerator
from efs_attraction_generator import EFSAttractionGenerator
from zone_translator import ZoneTranslator

from demand_utilities import utils as du
from demand_utilities import error_management as err_check
from demand_utilities.sector_reporter_v2 import SectorReporter


class ExternalForecastSystem:
    # ## Class Constants ## #
    __version__ = "v2-2"

    # defines all non-year columns
    column_dictionary = efs_consts.EFS_COLUMN_DICTIONARY

    def __init__(self,
                 population_value_file: str = "population/base_population_2018.csv",
                 population_growth_file: str = "population/future_population_growth.csv",
                 population_constraint_file: str = "population/future_population_values.csv",
                 future_population_ratio_file: str = "traveller_type/traveller_type_splits.csv",

                 households_value_file: str = "households/base_households_2018.csv",
                 household_growth_file: str = "households/future_households_growth.csv",
                 households_constraint_file: str = "households/future_households_values.csv",
                 housing_type_split_file: str = "households/housing_property_ratio.csv",
                 housing_occupancy_file: str = "households/housing_occupancy.csv",

                 worker_value_file: str = "employment/base_workers_2018.csv",
                 worker_growth_file: str = "employment/future_workers_growth.csv",
                 worker_constraint_file: str = "employment/future_workers_growth_values.csv",
                 worker_ratio_file: str = "employment/future_worker_splits.csv",

                 production_trip_rates_file: str = "traveller_type/hb_trip_rates.csv",
                 hb_mode_split_file: str = "traveller_type/hb_mode_split.csv",
                 hb_mode_time_split_file: str = "traveller_type/mode_time_split.csv",
                 split_handler_file: str = "traveller_type/mode_time_ids.csv",
                 traveller_types_file: str = "traveller_type/traveller_types.csv",
                 attraction_weights_file: str = "attractions/future_attraction_weights_i3.csv",

                 value_zoning: str = "MSOA",
                 value_zones_file: str = "zoning/msoa_zones.csv",
                 area_types_file: str = "zoning/msoa_area_types.csv",
                 area_grouping_file: str = "zoning/lad_msoa_grouping.csv",
                 msoa_area_types_file: str = "zoning/msoa_area_types.csv",
                 zone_areatype_lookup_file: str = "zoning/norms_2015.csv",
                 input_file_home: str = "Y:/NorMITs Demand/inputs/default/",

                 use_zone_id_subset: bool = False
                 ):
        """
        #TODO
        """
        self.use_zone_id_subset = use_zone_id_subset

        print("Initiating External Forecast System...")
        begin_time = time.time()
        current_time = begin_time

        # Read in population files
        file_path = os.path.join(input_file_home, population_value_file)
        self.population_values = du.safe_read_csv(file_path)

        file_path = os.path.join(input_file_home, population_growth_file)
        self.population_growth = du.safe_read_csv(file_path)

        file_path = os.path.join(input_file_home, population_constraint_file)
        self.population_constraint = du.safe_read_csv(file_path)

        file_path = os.path.join(input_file_home, future_population_ratio_file)
        self.future_population_ratio = du.safe_read_csv(file_path)

        # Households files
        file_path = os.path.join(input_file_home, households_value_file)
        self.households_values = du.safe_read_csv(file_path)

        file_path = os.path.join(input_file_home, household_growth_file)
        self.households_growth = du.safe_read_csv(file_path)

        file_path = os.path.join(input_file_home, households_constraint_file)
        self.households_constraint = du.safe_read_csv(file_path)

        file_path = os.path.join(input_file_home, housing_type_split_file)
        self.housing_type_split = du.safe_read_csv(file_path)

        file_path = os.path.join(input_file_home, housing_occupancy_file)
        self.housing_occupancy = du.safe_read_csv(file_path)

        # Worker files
        file_path = os.path.join(input_file_home, worker_value_file)
        self.worker_values = du.safe_read_csv(file_path)

        file_path = os.path.join(input_file_home, worker_growth_file)
        self.worker_growth = du.safe_read_csv(file_path)

        file_path = os.path.join(input_file_home, worker_constraint_file)
        self.worker_constraint = du.safe_read_csv(file_path)

        file_path = os.path.join(input_file_home, worker_ratio_file)
        self.worker_splits = du.safe_read_csv(file_path)

        # Production and attraction files
        file_path = os.path.join(input_file_home, production_trip_rates_file)
        self.production_trip_rates = du.safe_read_csv(file_path)

        file_path = os.path.join(input_file_home, hb_mode_split_file)
        self.hb_mode_split = du.safe_read_csv(file_path)

        file_path = os.path.join(input_file_home, hb_mode_time_split_file)
        self.hb_mode_time_split = du.safe_read_csv(file_path)

        file_path = os.path.join(input_file_home, split_handler_file)
        self.split_handler = du.safe_read_csv(file_path)

        file_path = os.path.join(input_file_home, traveller_types_file)
        self.traveller_types = du.safe_read_csv(file_path)

        file_path = os.path.join(input_file_home, attraction_weights_file)
        self.attraction_weights = du.safe_read_csv(file_path)

        # Zone and area files
        self.value_zoning = value_zoning

        file_path = os.path.join(input_file_home, value_zones_file)
        self.value_zones = du.safe_read_csv(file_path)

        file_path = os.path.join(input_file_home, area_types_file)
        self.area_types = du.safe_read_csv(file_path)

        file_path = os.path.join(input_file_home, area_grouping_file)
        self.area_grouping = du.safe_read_csv(file_path)

        file_path = os.path.join(input_file_home, msoa_area_types_file)
        self.msoa_area_types = du.safe_read_csv(file_path)

        file_path = os.path.join(input_file_home, zone_areatype_lookup_file)
        self.zone_areatype_lookup = du.safe_read_csv(file_path)

        if use_zone_id_subset:
            print("WARNING: Not using all of the input data. "
                  "This should only happen during testing or development!")
            self._subset_zone_ids()

        # sub-classes
        self.constrainer = ForecastConstrainer()
        self.production_generator = EFSProductionGenerator()
        self.attraction_generator = EFSAttractionGenerator()

        # support utilities tools
        self.sector_reporter = SectorReporter()
        self.zone_translator = ZoneTranslator()

        print("External Forecast System initiated!")
        last_time = current_time
        current_time = time.time()
        print("Initialisation took: %.2f seconds." %
              (current_time - last_time))

    def run(self,
            base_year: int = 2018,
            future_years: List[int] = efs_consts.FUTURE_YEARS,
            desired_zoning: str = "MSOA",
            alternate_population_base_year_file: str = None,
            alternate_households_base_year_file: str = None,
            alternate_worker_base_year_file: str = None,
            alternate_population_growth_assumption_file: str = None,
            alternate_households_growth_assumption_file: str = None,
            alternate_worker_growth_assumption_file: str = None,
            alternate_population_split_file: str = None,
            distribution_method: str = "Furness",
            seed_dist_location: str = efs_consts.DEFAULT_DIST_LOCATION,
            distributions: dict = efs_consts.EFS_RUN_DISTRIBUTIONS_DICT,
            purposes_needed: List[int] = efs_consts.PURPOSES_NEEDED,
            soc_needed: List[int] = efs_consts.SOC_NEEDED,
            ns_needed: List[int] = efs_consts.NS_NEEDED,
            car_availabilities_needed: List[int] = efs_consts.CA_NEEDED,
            modes_needed: List[int] = efs_consts.MODES_NEEDED,
            times_needed: List[int] = efs_consts.TIMES_NEEDED,
            development_log_file: str = None,
            development_log_split_file: str = None,
            minimum_development_certainty: str = "MTL",
            integrating_development_log: bool = False,
            population_metric: str = "Households",  # Households, Population
            constraint_required: List[bool] = efs_consts.CONSTRAINT_REQUIRED_DEFAULT,
            constraint_method: str = "Percentage",  # Percentage, Average
            constraint_area: str = "Designated",  # Zone, Designated, All
            constraint_on: str = "Growth",  # Growth, All
            constraint_source: str = "Grown Base",  # Default, Grown Base, Model Grown Base
            outputting_files: bool = True,
            performing_sector_totals: bool = True,
            output_location: str = None,
            echo_distribution: bool = True
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
            This is the base year used for re-balancing growth and constraint
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

        seed_dist_location:
            The primary location for all the distributions.
            Default input is: "Y:/EFS/inputs/distributions".
            Possible input is any file location folder.

        distributions:
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

        soc_needed:
            TODO: What is soc/ns in words?
            What soc are needed on distribution.
            Default input is: [0, 1, 2, 3]
            Possible input is a list containing integers corresponding to the
            soc IDs.

        ns_needed:
            What ns are needed on distribution.
            Default input is: [1, 2, 3, 4, 5]
            Possible input is a list containing integers corresponding to the
            ns IDs.

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
        # ## TIME SET UP ## #
        begin_time = time.time()
        current_time = begin_time

        # ## INPUT CHECKS ## #
        print("Starting input checks...")
        constraint_method = constraint_method.lower()
        constraint_area = constraint_area.lower()
        constraint_on = constraint_on.lower()
        constraint_source = constraint_source.lower()
        distribution_method = distribution_method.lower()
        population_metric = population_metric.lower()
        minimum_development_certainty = minimum_development_certainty.upper()

        year_list = [str(x) for x in [base_year] + future_years]

        # ## OUTPUT SETUP ## #
        final_out_path = self.generate_output_path(output_location,
                                                   desired_zoning,
                                                   population_metric,
                                                   constraint_method,
                                                   constraint_area,
                                                   constraint_on,
                                                   constraint_source)
        if not os.path.exists(final_out_path):
            os.mkdir(final_out_path)

        # Development Log Checks
        if (integrating_development_log and
            (development_log_file is None or
             development_log_split_file is None
             )):
            print("If integrating development log then both a development "
                  "log and development log split file need to be provided. "
                  "Ending process.")
            exit(1)

        # Distribute column names into more specific variables
        base_year_pop_cols = self.column_dictionary["base_year_population"]
        base_year_hh_cols = self.column_dictionary["base_year_households"]
        base_year_workers_cols = self.column_dictionary["base_year_workers"]

        pop_cols = self.column_dictionary["population"] + year_list
        pop_ratio_cols = self.column_dictionary["population_ratio"] + year_list

        hh_cols = self.column_dictionary["households"] + year_list
        hh_occupancy_cols = self.column_dictionary["housing_occupancy"] + year_list

        emp_cols = self.column_dictionary["employment"] + year_list
        emp_ratio_cols = self.column_dictionary["employment_ratio"] + year_list

        production_trip_cols = self.column_dictionary["production_trips"] + year_list
        mode_split_cols = self.column_dictionary["mode_split"] + year_list
        attraction_weight_cols = self.column_dictionary["attraction_weights"] + year_list

        print("No known errors in the inputs!")
        last_time = current_time
        current_time = time.time()
        print("Input checks took: %.2f seconds." %
              (current_time - last_time))

        # ## GET DATA ## #
        alternate_inputs = [
            alternate_population_base_year_file,
            alternate_households_base_year_file,
            alternate_worker_base_year_file,
            alternate_population_growth_assumption_file,
            alternate_households_growth_assumption_file,
            alternate_worker_growth_assumption_file
        ]

        # Integrate alternate inputs if given
        if all(x is not None for x in alternate_inputs):
            print("Need to integrate alternative assumptions.")
            print("Integrating alternate assumptions...")
            # # ALTERNATE ASSUMPTION INTEGRATION # #
            integrated_assumptions = self.integrate_alternate_assumptions(
                alternate_population_base_year_file,
                alternate_households_base_year_file,
                alternate_worker_base_year_file,
                alternate_population_growth_assumption_file,
                alternate_households_growth_assumption_file,
                alternate_worker_growth_assumption_file, base_year_pop_cols,
                base_year_hh_cols)

            population_values = integrated_assumptions[0][base_year_pop_cols]
            households_values = integrated_assumptions[1][base_year_hh_cols]
            worker_values = integrated_assumptions[2][base_year_workers_cols]
            population_growth = integrated_assumptions[3][pop_cols]
            households_growth = integrated_assumptions[4][hh_cols]
            worker_growth = integrated_assumptions[5][emp_cols]

            # TODO: Remove self references and integrate into alternate assumptions
            # TODO: Alternate population split read in
            population_split = self.future_population_ratio[pop_ratio_cols].copy()
            housing_type_split = self.housing_type_split[hh_occupancy_cols].copy()
            housing_occupancy = self.housing_occupancy[hh_occupancy_cols].copy()
            hb_mode_split = self.hb_mode_split[mode_split_cols].copy()
            msoa_area_types = self.msoa_area_types.copy()
            zone_areatype_lookup = self.zone_areatype_lookup.copy()
            worker_split = self.worker_splits[emp_ratio_cols].copy()

            trip_rates = self.production_trip_rates[
                production_trip_cols
            ].copy().rename(
                # Rename to cols names used in code
                columns={
                    "traveller_type": "traveller_type_id",
                    "area_type": "area_type_id",
                    "p": "purpose_id"
                }
            )

            car_association = self.traveller_types[[
                "cars", "traveller_type"
            ]].copy().rename(columns={"traveller_type": "traveller_type_id"})

            car_association["car_availability_id"] = 0
            no_car_mask = (car_association["cars"] == 0)

            car_association[no_car_mask]["car_availability_id"] = 1
            car_association[-no_car_mask]["car_availability_id"] = 2

            print("Integrated alternate assumptions!")
            last_time = current_time
            current_time = time.time()
            print("Integrating alternate assumptions took: %.2f seconds." %
                  (current_time - last_time))
        else:
            # # COPY OVER VALUES # #
            print("No need to integrate alternative assumptions.")
            print("Reading in default values...")
            population_values = self.population_values[base_year_pop_cols].copy()
            population_growth = self.population_growth[pop_cols].copy()
            population_split = self.future_population_ratio[pop_ratio_cols].copy()

            households_values = self.households_values[base_year_hh_cols].copy()
            households_growth = self.households_growth[hh_cols].copy()
            housing_type_split = self.housing_type_split[hh_occupancy_cols].copy()
            housing_occupancy = self.housing_occupancy[hh_occupancy_cols].copy()

            worker_values = self.worker_values[base_year_workers_cols].copy()
            worker_growth = self.worker_growth[emp_cols].copy()
            worker_split = self.worker_splits[emp_ratio_cols].copy()

            # Need to rename cols to names used in code
            trip_rates = self.production_trip_rates[production_trip_cols].copy()
            trip_rates = trip_rates.rename(
                columns={
                    "traveller_type": "traveller_type_id",
                    "area_type": "area_type_id",
                    "p": "purpose_id"
                }
            )

            hb_mode_split = self.hb_mode_split[mode_split_cols].copy()
            msoa_area_types = self.msoa_area_types.copy()
            zone_areatype_lookup = self.zone_areatype_lookup.copy()

            # TODO: make norms_2015_AreaType_Lookup table
            zone_areatype_lookup = zone_areatype_lookup.merge(
                msoa_area_types,
                left_on="msoa_zone_id",
                right_on="model_zone_id"
            )
            zone_areatype_lookup = zone_areatype_lookup.groupby(
                ['norms_2015_zone_id', 'area_type_id']
            ).size().to_frame('count').reset_index()

            zone_areatype_lookup = zone_areatype_lookup.sort_values(
                by=['count', 'area_type_id'],
                ascending=[False, True]
            ).drop_duplicates(subset=['norms_2015_zone_id'])

            zone_areatype_lookup = zone_areatype_lookup[[
                'norms_2015_zone_id', 'area_type_id'
            ]].sort_values('norms_2015_zone_id')

            # zone_areatype_lookup.sort_values('norms_2015_zone_id').to_csv(lookup_location + "norms_2015_AreaType_Lookup.csv", index=False)
            # .rename(
            #     columns = {
            #         "area_type": "area_type_id",
            #         "ca": "car_availability_id",
            #         "p": "purpose_id",
            #         "m1": "1",
            #         "m2": "2",
            #         "m3": "3",
            #         "m5": "5",
            #         "m6": "6",
            #         }
            #     )

            attraction_weights = self.attraction_weights[attraction_weight_cols].copy()

            car_association = self.traveller_types[[
                    "cars",
                    "traveller_type"
            ]].copy().rename(columns={"traveller_type": "traveller_type_id"})

            car_association["car_availability_id"] = 0
            no_car_mask = (car_association["cars"] == "0")

            # set up ids (-no_car_mask is the inversion of no_car_mask)
            car_association.loc[no_car_mask, "car_availability_id"] = 1
            car_association.loc[-no_car_mask, "car_availability_id"] = 2

            car_association = car_association[[
                "traveller_type_id",
                "car_availability_id"
            ]]

            print("Read-in default values!")
            last_time = current_time
            current_time = time.time()
            print("Reading in default values took: %.2f seconds." %
                  (current_time - last_time))

        # ## D-LOG READ-IN
        if integrating_development_log:
            development_log = pd.read_csv(development_log_file)
            development_log_split = pd.read_csv(development_log_split_file)
        else:
            development_log = None
            development_log_split = None

        # ## CONSTRAINT BUILDING
        if constraint_source == "default":
            print("Constraint 'default' selected, retrieving constraint "
                  + "data...")
            population_constraint = self.population_constraint[pop_cols].copy()
            population_constraint = self.constrainer.convert_constraint_off_base_year(
                population_constraint,
                str(base_year),
                year_list
            )

            households_constraint = self.households_constraint[hh_cols].copy()
            households_constraint = self.constrainer.convert_constraint_off_base_year(
                households_constraint,
                str(base_year),
                year_list
            )

            worker_constraint = self.worker_constraint[emp_cols].copy()
            worker_constraint = self.constrainer.convert_constraint_off_base_year(
                worker_constraint,
                str(base_year),
                year_list
            )

            print("Constraint retrieved!")
            last_time = current_time
            current_time = time.time()
            print("Constraint retrieval took: %.2f seconds." %
                  (current_time - last_time))

        elif constraint_source == "grown base":
            print("Constraint 'grown base' source selected, growing given "
                  "base by default growth factors...")
            population_constraint = self.population_growth[pop_cols].copy()

            population_constraint = self.convert_growth_off_base_year(
                population_constraint,
                str(base_year),
                year_list
            )
            population_constraint = self.get_grown_values(population_values,
                                                          population_constraint,
                                                          "base_year_population",
                                                          year_list)

            households_constraint = self.households_growth[hh_cols].copy()

            households_constraint = self.convert_growth_off_base_year(
                households_constraint,
                str(base_year),
                year_list
            )
            households_constraint = self.get_grown_values(households_values,
                                                          households_constraint,
                                                          "base_year_households",
                                                          year_list)

            worker_constraint = self.worker_growth[emp_cols].copy()

            worker_constraint = self.convert_growth_off_base_year(
                worker_constraint,
                str(base_year),
                year_list
            )
            worker_constraint = self.get_grown_values(worker_values,
                                                      worker_constraint,
                                                      "base_year_workers",
                                                      year_list)
            print("Constraint generated!")
            last_time = current_time
            current_time = time.time()
            print("Constraint generation took: %.2f seconds." %
                  (current_time - last_time))

        elif constraint_source == "model grown base":
            print("Constraint 'model grown base' source selected, this will "
                  + "be created later...")
            households_constraint = None
            population_constraint = None
            # TODO: Remember to do this

        # ## POPULATION GENERATION ## #
        print("Generating population...")
        production_trips = self.production_generator.run(
            population_growth=population_growth,
            population_values=population_values,
            population_constraint=population_constraint,
            population_split=population_split,
            households_growth=households_growth,
            households_values=households_values,
            households_constraint=households_constraint,
            housing_split=housing_type_split,
            housing_occupancy=housing_occupancy,
            d_log=development_log,
            d_log_split=development_log_split,
            minimum_development_certainty=minimum_development_certainty,
            population_metric=population_metric,
            constraint_required=constraint_required,
            constraint_method=constraint_method,
            constraint_area=constraint_area,
            constraint_on=constraint_on,
            constraint_source=constraint_source,
            designated_area=self.area_grouping.copy(),
            base_year_string=str(base_year),
            model_years=year_list,
            out_path=final_out_path,
            area_types=self.area_types,
            trip_rates=trip_rates
        )
        print("Productions generated!")
        last_time = current_time
        current_time = time.time()
        print("Production generation took: %.2f seconds" %
              (current_time - last_time))

        print("Converting traveller type id to car availability id...")
        required_columns = [
            "model_zone_id",
            "purpose_id",
            "car_availability_id",
            "soc",
            "ns"
        ]
        ca_production_trips = self.generate_car_availability(
            production_trips,
            car_association,
            year_list,
            required_columns
        )

        print("Converted to car availability!")
        last_time = current_time
        current_time = time.time()
        print("Car availability conversion took: %.2f seconds" %
              (current_time - last_time))

        # ## ATTRACTION GENERATION ###
        print("Generating workers...")
        attraction_dataframe = self.attraction_generator.run(
            worker_growth=worker_growth,
            worker_values=worker_values,
            worker_constraint=worker_constraint,
            worker_split=worker_split,
            development_log=development_log,
            development_log_split=development_log_split,
            minimum_development_certainty=minimum_development_certainty,
            integrating_development_log=integrating_development_log,
            constraint_required=constraint_required,
            constraint_method=constraint_method,
            constraint_area=constraint_area,
            constraint_on=constraint_on,
            constraint_source=constraint_source,
            designated_area=self.area_grouping.copy(),
            base_year_string=str(base_year),
            model_years=year_list,
            attraction_weights=attraction_weights,
            output_path=final_out_path
        )

        print("Attractions generated!")
        last_time = current_time
        current_time = time.time()
        print("Employment and Attraction generation took: %.2f seconds" %
              (current_time - last_time))

        # ## ATTRACTION WEIGHT GENERATION & MATCHING ## #
        print("Generating attraction weights...")
        attraction_weights = self.generate_attraction_weights(
            attraction_dataframe,
            year_list
        )

        print("Attraction weights generated!")
        last_time = current_time
        current_time = time.time()
        print("Attraction weight generation took: %.2f seconds" %
              (current_time - last_time))

        # TODO: Why has this been commented out?
        # print("Matching attractions...")
        # attraction_dataframe = self.match_attractions_to_productions(
        #    attraction_weights,
        #    production_trips,
        #    year_list
        # )
        # print("Attractions matched!")
        # last_time = current_time
        # current_time = time.time()
        # print("Attraction matching took: %.2f seconds" %
        #       (current_time - last_time))

        # ## ZONE TRANSLATION ## #
        if desired_zoning != self.value_zoning:
            print("Need to translate zones.")
            print("Translating from: " + self.value_zoning)
            print("Translating to: " + desired_zoning)
            # read in translation dataframe
            output_path = "Y:/EFS/inputs/default/zone_translation"
            output_path = os.path.join(output_path, desired_zoning + ".csv")
            translation_dataframe = pd.read_csv(output_path)

            converted_productions = self.zone_translator.run(
                ca_production_trips,
                translation_dataframe,
                self.value_zoning,
                desired_zoning,
                non_split_columns=[
                        "model_zone_id",
                        "purpose_id",
                        "car_availability_id",
                        "soc",
                        "ns"
                        ]
            )

            converted_pure_attractions = self.zone_translator.run(
                attraction_dataframe,
                translation_dataframe,
                self.value_zoning,
                desired_zoning,
                non_split_columns=["model_zone_id", "purpose_id"]
            )

            converted_attractions = self.zone_translator.run(
                attraction_weights,
                translation_dataframe,
                self.value_zoning,
                desired_zoning,
                non_split_columns=["model_zone_id", "purpose_id"]
            )

            print("Zone translation completed!")
            last_time = current_time
            current_time = time.time()
            print("Zone translation took: %.2f seconds" %
                  (current_time - last_time))
        else:
            converted_productions = ca_production_trips.copy()
            converted_attractions = attraction_weights.copy()
        # check point
        # converted_productions.to_csv("F:/EFS/EFS_Full/check/converted_productions.csv", index=False)
        # converted_attractions.to_csv("F:/EFS/EFS_Full/check/converted_attractions.csv", index=False)

        # ## DISTRIBUTION ## #
        if distribution_method == "furness":
            print("Generating distributions...")
            final_distribution_dictionary = self.distribute_dataframe(
                productions=converted_productions,
                attraction_weights=converted_attractions,
                mode_split_dataframe=hb_mode_split,
                zone_areatype_lookup=zone_areatype_lookup,
                required_purposes=purposes_needed, required_soc=soc_needed,
                required_ns=ns_needed,
                required_car_availabilities=car_availabilities_needed,
                required_modes=modes_needed, required_times=times_needed,
                year_string_list=year_list,
                distribution_dataframe_dict=distributions,
                distribution_file_location=seed_dist_location,
                echo=echo_distribution
            )
            print("Distributions generated!")
            last_time = current_time
            current_time = time.time()
            print("Distribution generation took: %.2f seconds" %
                  (current_time - last_time))
        else:
            raise ValueError("'%s' is not a valid distribution method!" %
                             (str(distribution_method)))

        # ## SECTOR TOTALS ## #
        sector_totals = self.sector_reporter.calculate_sector_totals(
                converted_productions,
                grouping_metric_columns = year_list,
                zone_system_name = "norms_2015",
                zone_system_file = "Y:/EFS/inputs/default/norms_2015.csv",
                sector_grouping_file = "Y:/EFS/inputs/default/zone_translation/tfn_level_one_sectors_norms_grouping.csv"
                )

        pm_sector_total_dictionary = {}

        for purpose in purposes_needed:
            pm_productions = converted_productions.copy()

            pm_sector_totals = self.sector_reporter.calculate_sector_totals(
                pm_productions,
                grouping_metric_columns=year_list,
                zone_system_name="norms_2015",
                zone_system_file="Y:/EFS/inputs/default/norms_2015.csv",
                sector_grouping_file="Y:/EFS/inputs/default/zone_translation/tfn_level_one_sectors_norms_grouping.csv"
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

        # TODO: Integrate output file setup into init!
        if outputting_files:
            if output_location is not None:
                print("Saving files to: " + output_location)
                # TODO: Integrate into furnessing!
                final_out_path = final_out_path + "/"
                for key, distribution in final_distribution_dictionary.items():
                    key = str(key)
                    out_path = os.path.join(final_out_path, key + '.csv')

                    # Output in wide format
                    distribution.pivot_table(
                        index='p_zone',
                        columns='a_zone',
                        values='trips'
                    ).to_csv(out_path)
                    print("Saved distribution: " + key)

                # Pop generation moved
                # Final workers out moved

                ca_production_trips.to_csv(
                    final_out_path + "MSOA_production_trips.csv",
                    index=False
                )

                attraction_dataframe.to_csv(
                    final_out_path + "MSOA_attractions.csv",
                    index=False
                )

                converted_productions.to_csv(
                    final_out_path + desired_zoning + "_production_trips.csv",
                    index=False
                )

                converted_pure_attractions.to_csv(
                    final_out_path + desired_zoning + "_attractions.csv",
                    index=False
                )

                sector_totals.to_csv(
                    final_out_path + desired_zoning + "_sector_totals.csv",
                    index=False
                )

                for key, sector_total in pm_sector_total_dictionary.items():
                    print("Saving sector total: " + key)
                    sector_total.to_csv(
                        final_out_path
                        +
                        "sector_total_"
                        +
                        key
                        +
                        ".csv",
                        index=False
                    )
                    print("Saved sector total: " + key)

                explanation_file = open(final_out_path + "input_parameters.txt", "w")

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
                    "Distribution Location: " + seed_dist_location + "\n",
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
                                        alt_pop_base_year_file: str,
                                        alt_households_base_year_file: str,
                                        alt_worker_base_year_file: str,
                                        alt_pop_growth_file: str,
                                        alt_households_growth_file: str,
                                        alt_worker_growth_file: str,
                                        base_year_pop_cols: List[str],
                                        base_year_households_cols: List[str]
                                        ) -> List[pd.DataFrame]:
        """
        # TODO
        """
        # ## READ IN ALTERNATE ASSUMPTIONS ## #
        if alt_pop_base_year_file is not None:
            alt_pop_base_year = pd.read_csv(alt_pop_base_year_file)
        else:
            alt_pop_base_year = self.population_values.copy()

        if alt_households_base_year_file is not None:
            alt_households_base_year = pd.read_csv(alt_households_base_year_file)
        else:
            alt_households_base_year = self.households_values.copy()

        if alt_worker_base_year_file is not None:
            alt_worker_base_year = pd.read_csv(alt_worker_base_year_file)
        else:
            alt_worker_base_year = self.worker_values.copy()

        if alt_pop_growth_file is not None:
            alt_pop_growth = pd.read_csv(alt_pop_growth_file)
        else:
            alt_pop_growth = self.population_growth.copy()

        if alt_households_growth_file is not None:
            alt_households_growth = pd.read_csv(alt_households_growth_file)
        else:
            alt_households_growth = self.households_growth.copy()

        if alt_worker_growth_file is not None:
            alt_worker_growth = pd.read_csv(alt_worker_growth_file)
        else:
            alt_worker_growth = self.worker_growth.copy()

        # ## ZONE TRANSLATION OF ALTERNATE ASSUMPTIONS ## #
        # TODO: Maybe allow zone translation, maybe requiring sticking to base

        # ## COMBINE BASE & ALTERNATE ASSUMPTIONS ## #
        # integrate alternate population base
        if alt_pop_base_year_file is not None:
            default_pop_vals = self.population_values[base_year_pop_cols].copy()

            # Create a mask of the overlaps
            mask = (default_pop_vals["model_zone_id"].isin(
                alt_pop_base_year["model_zone_id"].values
            ))

            # Copy alt data into default where they overlap
            default_pop_vals.loc[
                mask, "base_year_population"
            ] = alt_pop_base_year["base_year_population"].values

            alt_pop_base_year = default_pop_vals

        # alternate households base
        if alt_households_base_year_file is not None:
            default_households_values = self.households_values[base_year_households_cols].copy()

            # Create a mask of the overlaps
            mask = (default_households_values["model_zone_id"].isin(
                alt_households_base_year["model_zone_id"].values
            ))

            # Copy alt data into default where they overlap
            default_households_values.loc[
                mask,
                "base_year_population"
            ] = alt_households_base_year["base_year_households"].values

            alt_households_base_year = default_households_values

        # alternate worker base
        if alt_worker_base_year_file is not None:
            alt_worker_base_year = pd.read_csv(alt_worker_base_year_file)
            alternate_worker_base_year_zones = alt_worker_base_year["model_zone_id"].values
            default_worker_values = self.worker_values[base_year_pop_cols].copy()
            default_worker_values.loc[
                default_worker_values["model_zone_id"].isin(alternate_worker_base_year_zones),
                "base_year_population"
            ] = alt_worker_base_year["base_year_workers"].values

            alt_worker_base_year = default_worker_values

        # alternate population growth
        if alt_pop_growth_file is not None:
            alt_pop_growth_zones = alt_pop_growth["model_zone_id"].values
            columns = alt_pop_growth.columns[1:].values

            # replacing missing values
            alt_pop_growth = alt_pop_growth.replace('*', None)

            for year in columns:
                alt_pop_growth[year] = alt_pop_growth[year].astype(float)
                alt_pop_growth[year + "_difference"] = None

            default_pop_growth = self.population_growth.copy()

            for zone in alt_pop_growth_zones:
                for year in columns:
                    default_value = default_pop_growth.loc[
                        default_pop_growth["model_zone_id"] == zone,
                        year
                    ].values[0]

                    new_value = alt_pop_growth.loc[
                        alt_pop_growth["model_zone_id"] == zone,
                        year
                    ].values[0]

                    difference = new_value - default_value

                    alt_pop_growth.loc[
                        alt_pop_growth["model_zone_id"] == zone,
                        year + "_difference"
                    ] = difference

                    if pd.notna(difference):
                        default_pop_growth.loc[
                            default_pop_growth["model_zone_id"] == zone,
                            year: default_pop_growth.columns[-1]
                        ] = default_pop_growth.loc[
                            default_pop_growth["model_zone_id"] == zone,
                            year: default_pop_growth.columns[-1]
                        ] + difference

            alt_pop_growth = default_pop_growth

        # alternate households growth
        if alt_households_growth_file is not None:
            alt_households_growth = pd.read_csv(alt_households_growth_file)
            alternate_households_growth_zones = alt_households_growth["model_zone_id"].values
            columns = alt_households_growth.columns[1:].values

            # replacing missing values
            alt_households_growth = alt_households_growth.replace('*', None)

            for year in columns:
                alt_households_growth[year] = alt_households_growth[year].astype(float)
                alt_households_growth[year + "_difference"] = None

            default_households_growth = self.households_growth.copy()

            for zone in alternate_households_growth_zones:
                for year in columns:
                    default_value = default_households_growth.loc[
                        default_households_growth["model_zone_id"] == zone,
                        year
                    ].values[0]

                    new_value = alt_households_growth.loc[
                        alt_households_growth["model_zone_id"] == zone,
                        year
                    ].values[0]

                    difference = new_value - default_value

                    alt_households_growth.loc[
                        alt_households_growth["model_zone_id"] == zone,
                        year + "_difference"
                    ] = difference

                    if pd.notna(difference):
                        default_households_growth.loc[
                            default_households_growth["model_zone_id"] == zone,
                            year: default_households_growth.columns[-1]
                        ] = default_households_growth.loc[
                            default_households_growth["model_zone_id"] == zone,
                            year: default_households_growth.columns[-1]
                        ] + difference

            alt_households_growth = default_households_growth

        # alternate worker growth
        if alt_worker_growth_file is not None:
            alt_worker_growth = pd.read_csv(alt_worker_growth_file)
            alternate_worker_growth_zones = alt_worker_growth["model_zone_id"].values
            columns = alt_worker_growth.columns[1:].values

            # replacing missing values
            alt_worker_growth = alt_worker_growth.replace('*', None)

            for year in columns:
                alt_worker_growth[year] = alt_worker_growth[year].astype(float)
                alt_worker_growth[year + "_difference"] = None

            default_worker_growth = self.worker_growth.copy()

            for zone in alternate_worker_growth_zones:
                for year in columns:
                    default_value = default_worker_growth.loc[
                        default_worker_growth["model_zone_id"] == zone,
                        year
                    ].values[0]

                    new_value = alt_worker_growth.loc[
                        alt_worker_growth["model_zone_id"] == zone,
                        year
                    ].values[0]

                    difference = new_value - default_value

                    alt_worker_growth.loc[
                        alt_worker_growth["model_zone_id"] == zone,
                        year + "_difference"
                    ] = difference

                    if pd.notna(difference):
                        default_worker_growth.loc[
                            default_worker_growth["model_zone_id"] == zone,
                            year: default_worker_growth.columns[-1]
                        ] = default_worker_growth.loc[
                            default_worker_growth["model_zone_id"] == zone,
                            year: default_worker_growth.columns[-1]
                        ] + difference

            alt_worker_growth = default_worker_growth

        return [
            alt_pop_base_year,
            alt_households_base_year,
            alt_worker_base_year,
            alt_pop_growth,
            alt_households_growth,
            alt_worker_growth
        ]

    def convert_growth_off_base_year(self,
                                     growth_dataframe: pd.DataFrame,
                                     base_year: str,
                                     all_years: List[str]
                                     ) -> pd.DataFrame:
        """
        Converts the multiplicative growth value of each all_years to be
        based off of the base year.

        Parameters
        ----------
        growth_dataframe:
            The starting dataframe containing the growth values of all_years
            and base_year

        base_year:
            The new base year to base all the all_years growth off of.

        all_years:
            The years in growth_dataframe to convert to be based off of
            base_year growth

        Returns
        -------
        converted_growth_dataframe:
            The original growth dataframe with all growth values converted

        """
        growth_dataframe = growth_dataframe.copy()
        for year in all_years:
            growth_dataframe.loc[:, year] = (
                growth_dataframe.loc[:, year]
                /
                growth_dataframe.loc[:, base_year]
            )

        return growth_dataframe

    def get_grown_values(self,
                         base_year_dataframe: pd.DataFrame,
                         growth_dataframe: pd.DataFrame,
                         base_year: str,
                         all_years: List[str]
                         ) -> pd.DataFrame:
        """

        Parameters
        ----------
        base_year_dataframe
        growth_dataframe
        base_year
        all_years

        Returns
        -------

        """
        base_year_dataframe = base_year_dataframe.copy()
        growth_dataframe = growth_dataframe.copy()

        # CREATE GROWN DATAFRAME
        grown_dataframe = pd.merge(
            base_year_dataframe,
            growth_dataframe,
            on="model_zone_id"
        )
        for year in all_years:
            growth_dataframe.loc[:, year] = (
                growth_dataframe.loc[:, year] - 1
                *
                growth_dataframe.loc[:, base_year]
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
            metric_dataframe.loc[:, year] = (
                metric_dataframe.loc[:, year]
                +
                metric_dataframe.loc[:, metric_column_name]
            )

        ## drop the unnecessary metric column
        metric_dataframe = metric_dataframe.drop(
            labels=metric_column_name,
            axis=1
        )

        return metric_dataframe

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
            on=["model_zone_id"],
            suffixes=("_spl", "")
        )

        for year in year_list:
            segmented_dataframe.loc[:, year] = (
                segmented_dataframe.loc[:, year]
                /
                segmented_dataframe.loc[:, year + "_spl"]
            )

        split_names = [s + "_spl" for s in year_list]
        segmented_dataframe = segmented_dataframe.drop(
            labels=split_names,
            axis=1
        )

        return segmented_dataframe

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
            on=["purpose_id", "traveller_type_id", "area_type_id"],
            suffixes=("", "_splits")
        )

        # Multiply by proportions to get split values
        for year in year_list:
            trip_dataframe.loc[:, year] = (
                trip_dataframe[year]
                *
                trip_dataframe[year + "_splits"]
            )

        # Extract just the needed columns
        group_by_cols = [
            "model_zone_id",
            "purpose_id",
            "traveller_type_id",
            "area_type_id",
            "mode_time_split"
        ]
        needed_columns = group_by_cols.copy()
        needed_columns.extend(year_list)

        trip_dataframe = trip_dataframe[needed_columns]
        trip_dataframe = trip_dataframe.groupby(
            by=group_by_cols,
            as_index=False
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
            on=["employment_class"],
            suffixes=("", "_weights")
        )

        for year in year_list:
            attraction_dataframe.loc[:, year] = (
                attraction_dataframe[year]
                *
                attraction_dataframe[year + "_weights"]
            )

        group_by_cols = ["model_zone_id", "purpose_id"]
        needed_columns = group_by_cols.copy()
        needed_columns.extend(year_list)

        attraction_dataframe = attraction_dataframe[needed_columns]
        attraction_dataframe = attraction_dataframe.groupby(
            by=group_by_cols,
            as_index=False
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
                attraction_weights.loc[mask, year] = (
                    attraction_weights.loc[mask, year]
                    /
                    attraction_weights.loc[mask, year].sum()
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
            on=["model_zone_id", "purpose_id"],
            suffixes=("", "_productions")
        )

        for purpose in purposes:
            for year in year_list:
                mask = (attraction_dataframe["purpose_id"] == purpose)
                attraction_dataframe.loc[mask, year] = (
                    attraction_weights.loc[mask, year]
                    *
                    attraction_dataframe.loc[mask, year + "_productions"].sum()
                )

        group_by_cols = ["model_zone_id", "purpose_id"]
        needed_columns = group_by_cols.copy()
        needed_columns.extend(year_list)

        attraction_dataframe = attraction_dataframe[needed_columns]
        attraction_dataframe = attraction_dataframe.groupby(
            by=group_by_cols,
            as_index=False
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

        Where the traveller type has no cars indicated as available,
        set car availability to 1

        Where the traveller type has 1+ cars indicated as available,
        set car availability to 2
        """
        traveller_based_dataframe = traveller_based_dataframe.copy()
        car_availability = car_availability.copy()
        required_combined_columns = required_columns.copy()

        required_combined_columns.extend(year_string_list)

        # Get the car availability for each traveller type
        car_availability_dataframe = pd.merge(
            traveller_based_dataframe,
            car_availability,
            on=["traveller_type_id"]
        )

        # Extract the required columns
        car_availability_dataframe = car_availability_dataframe[
            required_combined_columns
        ]
        car_availability_dataframe = car_availability_dataframe.groupby(
            by=required_columns,
            as_index=False
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
            on=["mode_time_split"]
        )

        reattached_dataframe = reattached_dataframe[
            required_combined_columns
        ]

        return reattached_dataframe

    def generate_output_path(self,
                             output_location,
                             desired_zoning,
                             pop_metric,
                             constraint_method=None,
                             constraint_area=None,
                             constraint_on=None,
                             constraint_source=None
                             ) -> str:
        # Init
        date = datetime.datetime.now()

        # Generate the base name
        fname_parts = [
            self.__version__ + " EFS Output",
            desired_zoning,
            date.strftime("%d-%m-%y"),
            "PM " + pop_metric[0]
        ]

        # Optionally add constraint keys
        if constraint_method is not None:
            fname_parts += ["CM" + constraint_method[0]]

        if constraint_area is not None:
            fname_parts += ["CA" + constraint_area[0]]

        if constraint_on is not None:
            fname_parts += ["CO" + constraint_on[0]]

        if constraint_source is not None:
            fname_parts += ["CS" + constraint_source[0]]

        # Build the full path and return
        return os.path.join(output_location, ' - '.join(fname_parts))

    def distribute_dataframe(self,
                             productions: pd.DataFrame,
                             attraction_weights: pd.DataFrame,
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
                             trip_origin: str = 'hb',
                             number_of_iterations: int = 1,
                             replace_zero_values: bool = True,
                             constrain_on_production: bool = True,
                             constrain_on_attraction: bool = True,
                             zero_replacement_value: float = 0.00001,
                             echo: bool = False
                             ) -> pd.DataFrame:
        """
        #TODO
        """
        # TODO: Output files while it runs, instead of at the end!
        productions = productions.copy()
        attraction_weights = attraction_weights.copy()
        mode_split_dataframe = mode_split_dataframe.copy()
        zone_areatype_lookup = zone_areatype_lookup.copy()
        final_distribution_dictionary = {}
        required_segments = []
        distribution_dataframe_list = []

        mode_split_dataframe = mode_split_dataframe.merge(
            zone_areatype_lookup,
            on="area_type_id",
        ).rename(columns={"norms_2015_zone_id": "p_zone"}).drop_duplicates()

        # TODO: Is this still needed?
        # make table wide to long
        # mode_split_dataframe = mode_split_dataframe.copy().melt(
        #     id_vars=['area_type_id', 'car_availability_id', 'purpose_id'],
        #     value_vars=['1', '2', '3', '5', '6'],
        #     var_name='mode_id', value_name='factor'
        #     )
        # mode_split_dataframe.to_csv(r'F:\EFS\EFS_Full\inputs\default\traveller_type\hb_mode_split.csv', index=False)

        # TODO: Move inside of all nested loops into function (stops the
        #  indentation from making difficult to read code)
        # TODO: Move mode out to nested loops
        # TODO: Tidy this up
        # TODO: Generate synth_dists path based on segmentation
        #  and file location given
        for year in year_string_list:
            for purpose in required_purposes:

                # ns/soc depends on purpose
                if purpose in [1, 2]:
                    required_segments = required_soc
                else:
                    required_segments = required_ns

                for segment in required_segments:
                    car_availability_dataframe = pd.DataFrame
                    first_iteration = True
                    for car_availability in required_car_availabilities:
                        print()

                        # for tp in required_times:
                        dist_path = os.path.join(
                            distribution_file_location,
                            distribution_dataframe_dict[purpose][segment][car_availability]
                        )

                        # Convert from wide to long format
                        # (needed for furnessing)
                        synth_dists = pd.read_csv(dist_path)
                        synth_dists = pd.melt(
                            synth_dists,
                            id_vars=['norms_zone_id'],
                            var_name='a_zone',
                            value_name='seed_values'
                        ).rename(
                            columns={"norms_zone_id": "p_zone"})

                        # convert column object to int
                        synth_dists['a_zone'] = synth_dists['a_zone'].astype(int)
                        synth_dists = synth_dists.groupby(
                            by=["p_zone", "a_zone"],
                            as_index=False
                        ).sum()

                        if self.use_zone_id_subset:
                            zone_subset = [259, 267, 268, 270, 275, 1171, 1173]
                            synth_dists = du.get_data_subset(
                                synth_dists, 'p_zone', zone_subset)
                            synth_dists = du.get_data_subset(
                                synth_dists, 'a_zone', zone_subset)

                        # Generate productions input
                        if purpose in [1, 2]:
                            segment_mask = (
                                (productions["purpose_id"] == purpose)
                                & (productions["car_availability_id"] == car_availability)
                                & (productions["soc"] == str(segment))
                            )
                        else:
                            segment_mask = (
                                (productions["purpose_id"] == purpose)
                                & (productions["car_availability_id"] == car_availability)
                                & (productions["ns"] == str(segment))
                            )

                        production_input = productions[segment_mask][
                            ["model_zone_id", str(year)]
                        ].rename(columns={str(year): "production_forecast"})

                        # Generate attractions input
                        mask = attraction_weights["purpose_id"] == purpose
                        attraction_input = attraction_weights[mask][
                            ["model_zone_id", str(year)]
                        ].rename(columns={str(year): "attraction_forecast"})

                        # Furness the productions and attractions
                        target_percentage = 0.7 if self.use_zone_id_subset else 0.975
                        final_distribution = fp.furness(
                            productions=production_input,
                            attractions=attraction_input,
                            distributions=synth_dists,
                            number_of_iterations=number_of_iterations,
                            replace_zero_values=replace_zero_values,
                            constrain_on_production=constrain_on_production,
                            constrain_on_attraction=constrain_on_attraction,
                            zero_replacement_value=zero_replacement_value,
                            target_percentage=target_percentage,
                            echo=echo
                        )

                        final_distribution["purpose_id"] = purpose
                        final_distribution["car_availability_id"] = car_availability

                        # tfn mode split
                        final_distribution = final_distribution.merge(
                            mode_split_dataframe,
                            on=["p_zone", "purpose_id", "car_availability_id"]
                        )

                        # calculate dt by mode
                        final_distribution["dt"] = (
                                final_distribution["dt"]
                                *
                                final_distribution[str(year)])

                        final_distribution = final_distribution[[
                            "p_zone",
                            "a_zone",
                            "mode_id",
                            "dt"
                         ]]

                        # Rename to the common output names
                        final_distribution = final_distribution.rename(columns={
                            "mode_id": "m",
                            "dt": "trips"
                        })

                        # TODO: Make sure this works for NHB trips too

                        # loop over required modes
                        for mode in required_modes:
                            mask = (final_distribution["m"] == mode)
                            final_distribution_mode = final_distribution[mask]
                            final_distribution_mode = final_distribution_mode[[
                                'p_zone', 'a_zone', 'trips'
                            ]]

                            dict_string = du.get_dist_name(
                                str(trip_origin),
                                'pa',
                                str(year),
                                str(purpose),
                                str(mode),
                                str(segment),
                                str(car_availability)
                            )

                            final_distribution_dictionary[dict_string] = final_distribution_mode

                            print("Distribution " + dict_string + " complete!")
                            if first_iteration:
                                car_availability_dataframe = final_distribution_mode
                                first_iteration = False
                            else:
                                car_availability_dataframe = car_availability_dataframe.append(
                                    final_distribution_mode
                                    )

        return final_distribution_dictionary

    def _subset_zone_ids(self):
        """
        Shrink down all inputs. Useful for testing and dev.
        """
        self.population_values = du.get_data_subset(self.population_values)
        self.population_growth = du.get_data_subset(self.population_growth)
        self.population_constraint = du.get_data_subset(self.population_constraint)
        self.future_population_ratio = du.get_data_subset(self.future_population_ratio)

        self.households_values = du.get_data_subset(self.households_values)
        self.households_growth = du.get_data_subset(self.households_growth)
        self.households_constraint = du.get_data_subset(self.households_constraint)
        self.housing_type_split = du.get_data_subset(self.housing_type_split)
        self.housing_occupancy = du.get_data_subset(self.housing_occupancy)

        self.worker_values = du.get_data_subset(self.worker_values)
        self.worker_growth = du.get_data_subset(self.worker_growth)
        self.worker_constraint = du.get_data_subset(self.worker_constraint)
        self.worker_splits = du.get_data_subset(self.worker_splits)

        self.value_zones = du.get_data_subset(self.value_zones)
        self.area_types = du.get_data_subset(self.area_types)
        self.area_grouping = du.get_data_subset(self.area_grouping)


def main():
    efs = ExternalForecastSystem(use_zone_id_subset=False)
    efs.run(desired_zoning="norms_2015",
            constraint_source="Default",
            output_location="E:/NorMITs Demand/",
            echo_distribution=True)


if __name__ == '__main__':
    main()
