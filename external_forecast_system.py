# -*- coding: utf-8 -*-
"""
Created on: Mon Nov 25 09:50:07 2019
Updated on: Fri Sep 18 15:03:24 2020

Original author: Sneezy
Last Update Made by: Ben Taylor

File purpose:

"""
# Built-ins
import os
import time
import itertools

from typing import List
from typing import Tuple

# External libs
import numpy as np
import pandas as pd

# self imports
import pa_to_od as pa2od
import od_to_pa as od2pa
import matrix_processing as mat_p
import efs_constants as consts
import furness_process as fp
import efs_production_generator as pm
import efs_attraction_generator as am

from efs_constrainer import ForecastConstrainer
from zone_translator import ZoneTranslator

from demand_utilities import utils as du
from demand_utilities import vehicle_occupancy as vo
from demand_utilities.sector_reporter_v2 import SectorReporter

# TODO: Implement multiprocessing
# TODO: Determine the TfN model name based on the given mode
# TODO: Output a run log instead of printing everything to the terminal.
# TODO: On error, output a simple error report

# TODO: Fix dtype error from pandas on initialisation
#  More info here:
#  https://stackoverflow.com/questions/24251219/pandas-read-csv-low-memory-and-dtype-options


class ExternalForecastSystem:
    # ## Class Constants ## #
    __version__ = "v2_2"
    _out_dir = "NorMITs Demand"

    # defines all non-year columns
    column_dictionary = consts.EFS_COLUMN_DICTIONARY

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
                 traveller_types_file: str = "traveller_type/traveller_type_ids.csv",
                 attraction_weights_file: str = "attractions/future_attraction_weights_i3.csv",

                 value_zoning: str = "MSOA",
                 value_zones_file: str = "zoning/msoa_zones.csv",
                 area_types_file: str = "zoning/msoa_area_types.csv",
                 area_grouping_file: str = "zoning/lad_msoa_grouping.csv",
                 msoa_area_types_file: str = "zoning/msoa_area_types.csv",
                 zone_areatype_lookup_file: str = "zoning/norms_2015.csv",
                 input_file_home: str = "Y:/NorMITs Demand/inputs/default/",
                 import_location: str = "Y:/",
                 output_location: str = "E:/",

                 use_zone_id_subset: bool = False
                 ):
        """
        #TODO
        """
        self.use_zone_id_subset = use_zone_id_subset
        self.output_location = output_location
        self.import_location = import_location

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
        self.production_generator = pm.EFSProductionGenerator()
        self.attraction_generator = am.EFSAttractionGenerator()

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
            future_years: List[int] = consts.FUTURE_YEARS,
            desired_zoning: str = "MSOA",
            alt_pop_base_year_file: str = None,
            alt_households_base_year_file: str = None,
            alt_worker_base_year_file: str = None,
            alt_pop_growth_assumption_file: str = None,
            alt_households_growth_assumption_file: str = None,
            alt_worker_growth_assumption_file: str = None,
            alt_pop_split_file: str = None,  # THIS ISN'T USED ANYWHERE
            distribution_method: str = "Furness",
            seed_dist_location: str = consts.DEFAULT_DIST_LOCATION,
            distributions: dict = consts.EFS_RUN_DISTRIBUTIONS_DICT,
            purposes_needed: List[int] = consts.PURPOSES_NEEDED,
            modes_needed: List[int] = consts.MODES_NEEDED,
            soc_needed: List[int] = consts.SOC_NEEDED,
            ns_needed: List[int] = consts.NS_NEEDED,
            car_availabilities_needed: List[int] = consts.CA_NEEDED,
            dlog_file: str = None,
            dlog_split_file: str = None,
            minimum_development_certainty: str = "MTL",
            population_metric: str = "Households",  # Households, Population
            constraint_required: List[bool] = consts.CONSTRAINT_REQUIRED_DEFAULT,
            constraint_method: str = "Percentage",  # Percentage, Average
            constraint_area: str = "Designated",  # Zone, Designated, All
            constraint_on: str = "Growth",  # Growth, All
            constraint_source: str = "Grown Base",  # Default, Grown Base, Model Grown Base
            outputting_files: bool = True,
            iter_num: int = 0,
            performing_sector_totals: bool = True,
            output_location: str = 'E:/',
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

        alt_pop_base_year_file:
            A file location (including file suffix) containing an alternate
            population for the base year. This file does not need full
            alternate population metrics, just needs it for the appropriate
            zones.
            Default input is: None.
            Possible input is any string which refers to a file location.

        alt_households_base_year_file:
            A file location (including file suffix) containing an alternate
            number of households for the base year. This file does not need full
            alternate households metrics, just needs it for the appropriate
            zones.
            Default input is: None.
            Possible input is any string which refers to a file location.

        alt_worker_base_year_file:
            A file location (including file suffix) containing an alternate
            number of workers for the base year. This file does not need full
            alternate worker metrics, just needs it for the appropriate
            zones.
            Default input is: None.
            Possible input is any string which refers to a file location.

        alt_pop_growth_assumption_file:
            A file location (including file suffix) containing an alternate
            population growth for some future years. This file does not need full
            alternate population growth metrics, just needs it for the appropriate
            zones and years.
            Default input is: None.
            Possible input is any string which refers to a file location.

        alt_households_growth_assumption_file:
            A file location (including file suffix) containing an alternate
            households growth for some future years. This file does not need full
            alternate households growth metrics, just needs it for the appropriate
            zones and years.
            Default input is: None.
            Possible input is any string which refers to a file location.

        alt_worker_growth_assumption_file:
            A file location (including file suffix) containing an alternate
            workers growth for some future years. This file does not need full
            alternate worker growth metrics, just needs it for the appropriate
            zones and years.
            Default input is: None.
            Possible input is any string which refers to a file location.

        alt_pop_split_file:
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

        dlog_file:
            A file location for the development log.
            Default input is: None
            Possible input is any file location folder.

        dlog_split_file:
            A file location for the housing stock split for the development log.
            Default input is: None
            Possible input is any file location folder.

        minimum_development_certainty:
            A string for the minimum development certainty required from the
            development log.
            Default input is: "MTL" # More than likely
            Possible inputs are: "NC", "MTL", "RF", "H"

        integrate_dlog:
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
        # Set up timing
        begin_time = time.time()
        current_time = begin_time

        # Format inputs
        constraint_method = constraint_method.lower()
        constraint_area = constraint_area.lower()
        constraint_on = constraint_on.lower()
        constraint_source = constraint_source.lower()
        distribution_method = distribution_method.lower()
        population_metric = population_metric.lower()
        minimum_development_certainty = minimum_development_certainty.upper()
        integrate_dlog = dlog_split_file is not None and dlog_file is not None
        iter_name = 'iter' + str(iter_num)
        model_name = du.get_model_name(modes_needed[0])

        if iter_num == 0:
            Warning("iter_num is set to 0. This is should only be the case"
                    "during testing.")

        if len(modes_needed) > 1:
            raise ValueError("Was given more than one mode. EFS cannot run "
                             "using more than one mode at a time due to "
                             "different zoning systems for NoHAM and NoRMS "
                             "etc.")

        year_list = [str(x) for x in [base_year] + future_years]

        # ## PREPARE OUTPUTS ## #
        print("Initialising outputs...")
        imports, exports, _ = self.generate_output_paths(output_location,
                                                         model_name,
                                                         iter_name)

        write_input_info(
            os.path.join(exports['home'], "input_parameters.txt"),
            base_year,
            future_years,
            desired_zoning,
            alt_pop_base_year_file,
            alt_households_base_year_file,
            alt_worker_base_year_file,
            alt_pop_growth_assumption_file,
            alt_households_growth_assumption_file,
            alt_worker_growth_assumption_file,
            alt_pop_split_file,
            distribution_method,
            seed_dist_location,
            purposes_needed,
            modes_needed,
            soc_needed,
            ns_needed,
            car_availabilities_needed,
            integrate_dlog,
            minimum_development_certainty,
            population_metric,
            constraint_required,
            constraint_method,
            constraint_area,
            constraint_on,
            constraint_source,
        )

        # ## INPUT CHECKS ## #
        print("Starting input checks...")

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
            alt_pop_base_year_file,
            alt_households_base_year_file,
            alt_worker_base_year_file,
            alt_pop_growth_assumption_file,
            alt_households_growth_assumption_file,
            alt_worker_growth_assumption_file
        ]

        # Integrate alternate inputs if given
        if all(x is not None for x in alternate_inputs):
            print("Need to integrate alternative assumptions.")
            print("Integrating alternate assumptions...")
            # # ALTERNATE ASSUMPTION INTEGRATION # #
            integrated_assumptions = self.integrate_alternate_assumptions(
                alt_pop_base_year_file,
                alt_households_base_year_file,
                alt_worker_base_year_file,
                alt_pop_growth_assumption_file,
                alt_households_growth_assumption_file,
                alt_worker_growth_assumption_file, base_year_pop_cols,
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
        if integrate_dlog:
            development_log = pd.read_csv(dlog_file)
            development_log_split = pd.read_csv(dlog_split_file)
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
            households_constraint = None
            population_constraint = None
            raise NotImplementedError("Constraint 'model grown base' selected, "
                                      "this will be created later...")

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
            out_path=exports['productions'],
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
            integrating_development_log=integrate_dlog,
            constraint_required=constraint_required,
            constraint_method=constraint_method,
            constraint_area=constraint_area,
            constraint_on=constraint_on,
            constraint_source=constraint_source,
            designated_area=self.area_grouping.copy(),
            base_year_string=str(base_year),
            model_years=year_list,
            attraction_weights=attraction_weights,
            output_path=exports['attractions']
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

        # TODO: Save converted productions and attractions to file
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
                required_modes=modes_needed,
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
                grouping_metric_columns=year_list,
                zone_system_name="norms_2015",
                zone_system_file="Y:/EFS/inputs/default/norms_2015.csv",
                sector_grouping_file="Y:/EFS/inputs/default/zone_translation/tfn_level_one_sectors_norms_grouping.csv"
                )

        pm_sector_total_dictionary = {}

        for purpose in purposes_needed:
            # TODO: Update sector reporter.
            #  Sector totals don't currently allow this

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

        # ## OUTPUTS ## #
        # TODO: Properly integrate this
        # TODO: Tidy up outputs, separate into different files

        # TODO: Integrate output file setup into init!
        if outputting_files:
            if output_location is not None:
                print("Saving files to: " + output_location)
                # TODO: Integrate into furnessing!

                # Write distributions to file
                for key, distribution in final_distribution_dictionary.items():
                    key = str(key)
                    out_path = os.path.join(exports['pa_24'], key + '.csv')

                    # Output in wide format
                    distribution.pivot_table(
                        index='p_zone',
                        columns='a_zone',
                        values='trips'
                    ).to_csv(out_path)
                    print("Saved distribution: " + key)

                # Pop generation moved
                # Final workers out moved

                fname = "MSOA_production_trips.csv"
                ca_production_trips.to_csv(
                    os.path.join(exports['productions'], fname),
                    index=False
                )

                fname = "MSOA_attractions.csv"
                attraction_dataframe.to_csv(
                    os.path.join(exports['attractions'], fname),
                    index=False
                )

                fname = desired_zoning + "_production_trips.csv"
                converted_productions.to_csv(
                    os.path.join(exports['productions'], fname),
                    index=False
                )

                fname = desired_zoning + "_attractions.csv"
                converted_pure_attractions.to_csv(
                    os.path.join(exports['attractions'], fname),
                    index=False
                )

                fname = desired_zoning + "_sector_totals.csv"
                sector_totals.to_csv(
                    os.path.join(exports['sectors'], fname),
                    index=False
                )

                for key, sector_total in pm_sector_total_dictionary.items():
                    print("Saving sector total: " + key)
                    fname = "sector_total_%s.csv" % key
                    sector_total.to_csv(
                        os.path.join(exports['sectors'], fname),
                        index=False
                    )
                    print("Saved sector total: " + key)

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

    def pa_to_od(self,
                 years_needed: List[int] = consts.ALL_YEARS,
                 modes_needed: List[int] = consts.MODES_NEEDED,
                 purposes_needed: List[int] = consts.PURPOSES_NEEDED,
                 soc_needed: List[int] = consts.SOC_NEEDED,
                 ns_needed: List[int] = consts.NS_NEEDED,
                 ca_needed: List[int] = consts.CA_NEEDED,
                 output_location: str = None,
                 iter_num: int = 0,
                 overwrite_hb_tp_pa: bool = True,
                 overwrite_hb_tp_od: bool = True,
                 echo: bool = True
                 ) -> None:
        """
        Converts home based PA matrices into time periods split PA matrices,
        then OD matrices (to_home, from_home, and full OD)

        Parameters
        ----------
        years_needed:
            The years of PA matrices to convert to OD

        modes_needed:
            The modes of PA matrices to convert to OD

        purposes_needed:
            The purposes of PA matrices to convert to OD

        soc_needed:
            The skill levels of PA matrices to convert to OD

        ns_needed:
            The income levels of PA matrices to convert to OD

        ca_needed:
            The the car availability of PA matrices to convert to OD

        output_location:
            The directory to create the new output directory in - a dir named
            self._out_dir (NorMITs Demand) should exist here. Usually
            a drive name e.g. Y:/

        iter_num:
            The number of the iteration being run.

        # TODO: Update docs once correct functionality exists
        overwrite_hb_tp_pa:
            Whether to split home based PA matrices into time periods.

        overwrite_hb_tp_od:
            Whether to convert time period split PA matrices into OD matrices.

        echo:
            If True, suppresses some of the non-essential terminal outputs.

        Returns
        -------
        None
        """
        # Init
        if output_location is None:
            output_location = self.output_location
        iter_name = 'iter' + str(iter_num)
        model_name = du.get_model_name(modes_needed[0])

        if iter_num == 0:
            Warning("iter_num is set to 0. This is should only be the case"
                    "during testing.")

        if len(modes_needed) > 1:
            raise ValueError("Was given more than one mode. EFS cannot run "
                             "using more than one mode at a time due to "
                             "different zoning systems for NoHAM and NoRMS "
                             "etc.")

        # Generate paths
        imports, exports, _ = self.generate_output_paths(
            output_location=output_location,
            model_name=model_name,
            iter_name=iter_name
        )
        # TODO: Add time print outs
        # TODO: Change import paths to accept specific dir

        # TODO: Check if tp pa matrices exist first
        if overwrite_hb_tp_pa:
            print("Converting HB 24hr PA to time period split PA...")
            pa2od.efs_build_tp_pa(tp_import=imports['tp_splits'],
                                  pa_import=exports['pa_24'],
                                  pa_export=exports['pa'],
                                  years_needed=years_needed,
                                  required_purposes=purposes_needed,
                                  required_modes=modes_needed,
                                  required_soc=soc_needed,
                                  required_ns=ns_needed,
                                  required_ca=ca_needed)
            print('HB time period split PA matrices compiled!\n')

        # TODO: Check if od matrices exist first
        if overwrite_hb_tp_od:
            print('Converting time period split PA to OD...')
            pa2od.efs_build_od(
                pa_import=exports['pa'],
                od_export=exports['od'],
                required_purposes=purposes_needed,
                required_modes=modes_needed,
                required_soc=soc_needed,
                required_ns=ns_needed,
                required_car_availabilities=ca_needed,
                year_string_list=years_needed,
                phi_type='fhp_tp',
                aggregate_to_wday=True,
                echo=echo)
            print('HB OD matrices compiled!\n')
            # TODO: Create 24hr OD for HB

    def run_nhb(self,
                years_needed: List[int] = consts.ALL_YEARS,
                modes_needed: List[int] = consts.MODES_NEEDED,
                hb_purposes_needed: List[int] = consts.PURPOSES_NEEDED,
                hb_soc_needed: List[int] = consts.SOC_NEEDED,
                hb_ns_needed: List[int] = consts.NS_NEEDED,
                hb_ca_needed: List[int] = consts.CA_NEEDED,
                nhb_purposes_needed: List[int] = consts.NHB_PURPOSES_NEEDED,
                output_location: str = None,
                iter_num: int = 0,
                overwrite_nhb_productions: bool = True,
                overwrite_nhb_od: bool = True,
                overwrite_nhb_tp_od: bool = True,
                ):
        """
        Generates NHB distributions based from the time-period split
        HB distributions

        Performs the following actions:
            - Generates NHB productions using NHB factors and HB distributions
            - Furnesses NHB productions Synthesiser distributions as a seed

        Parameters
        ----------
        years_needed:
            The years used to produce NHB distributions for.

        modes_needed:
            The mode to generate a NHB distributions for.

        hb_purposes_needed:
            The home based purposes to use when generating NHB productions.

        hb_soc_needed:
            The home based soc_ids to use when generating NHB productions.

        hb_ns_needed:
            The home based ns_ids to use when generating NHB productions.

        hb_ca_needed:
            The car availability ids to use when generating NHB productions.

        nhb_purposes_needed:
            Which NHB purposes to generate NHb distributions for.

        output_location:
            The directory to create the new output directory in - a dir named
            self._out_dir (NorMITs Demand) should exist here. Usually
            a drive name e.g. Y:/

        iter_num:
            The number of the iteration being run.

        # TODO: Update docs once correct functionality exists
        overwrite_nhb_productions:
            Whether to generate nhb productions or not.

        overwrite_nhb_od
            Whether to generate nhb OD matrices or not.

        overwrite_nhb_tp_od
            Whether to generate nhb tp split OD matrices or not.

        Returns
        -------
        None
        """
        # Init
        if output_location is None:
            output_location = self.output_location
        iter_name = 'iter' + str(iter_num)
        model_name = du.get_model_name(modes_needed[0])

        if iter_num == 0:
            Warning("iter_num is set to 0. This is should only be the case"
                    "during testing.")

        if len(modes_needed) > 1:
            raise ValueError("Was given more than one mode. EFS cannot run "
                             "using more than one mode at a time due to "
                             "different zoning systems for NoHAM and NoRMS "
                             "etc.")

        # Generate paths
        imports, exports, _ = self.generate_output_paths(
            output_location=output_location,
            model_name=model_name,
            iter_name=iter_name
        )

        # TODO: Add time print outs
        # TODO: Change import paths to accept specific dir
        # TODO: Allow flexible segmentations

        # TODO: Check if nhb productions exist first
        if overwrite_nhb_productions:
            print("Generating NHB Productions...")
            pm.nhb_production(hb_pa_import=exports['pa_24'],
                              nhb_export=exports['productions'],
                              required_purposes=hb_purposes_needed,
                              required_modes=modes_needed,
                              required_soc=hb_soc_needed,
                              required_ns=hb_ns_needed,
                              required_car_availabilities=hb_ca_needed,
                              years_needed=years_needed,
                              nhb_factor_import=imports['home'])
            print('NHB productions generated!\n')

        # TODO: Check if NHB matrices exist first
        if overwrite_nhb_od:
            print("Furnessing NHB productions...")
            nhb_furness(p_import=exports['productions'],
                        seed_nhb_dist_dir=imports['seed_dists'],
                        od_export=exports['od_24'],
                        required_purposes=nhb_purposes_needed,
                        required_modes=modes_needed,
                        years_needed=years_needed,
                        replace_zero_vals=True,
                        zero_infill=0.01,
                        use_zone_id_subset=self.use_zone_id_subset)
            print('NHB productions "furnessed"\n')

        if overwrite_nhb_tp_od:
            print("Converting NHB 24hr OD to time period split OD...")
            pa2od.efs_build_tp_pa(tp_import=imports['tp_splits'],
                                  pa_import=exports['od_24'],
                                  pa_export=exports['od'],
                                  years_needed=years_needed,
                                  required_purposes=nhb_purposes_needed,
                                  required_modes=modes_needed,
                                  matrix_format='od')
            print('NHB time period split OD matrices compiled!\n')

        print("NHB run complete!")

    def pre_me_compile_od_matrices(self,
                                   year: int = consts.BASE_YEAR,
                                   hb_p_needed: List[int] = consts.PURPOSES_NEEDED,
                                   nhb_p_needed: List[int] = consts.NHB_PURPOSES_NEEDED,
                                   modes_needed: List[int] = consts.MODES_NEEDED,
                                   tp_needed: List[int] = consts.TIME_PERIODS,
                                   output_location: str = None,
                                   iter_num: int = 0,
                                   overwrite_aggregated_od: bool = True,
                                   overwrite_compiled_od: bool = True
                                   ) -> None:
        # TODO: write doc

        # Init
        if output_location is None:
            output_location = self.output_location
        iter_name = 'iter' + str(iter_num)
        model_name = du.get_model_name(modes_needed[0])

        if model_name == 'norms':
            ca_needed = consts.CA_NEEDED
        elif model_name == 'noham':
            ca_needed = [None]
        else:
            raise ValueError("Got an unexpected model name. Got %s, expected "
                             "either 'norms' or 'noham'." % str(model_name))

        if iter_num == 0:
            Warning("iter_num is set to 0. This is should only be the case"
                    "during testing.")

        if len(modes_needed) > 1:
            raise ValueError("Was given more than one mode. EFS cannot run "
                             "using more than one mode at a time due to "
                             "different zoning systems for NoHAM and NoRMS "
                             "etc.")

        # Generate paths
        imports, exports, params = self.generate_output_paths(
            output_location=output_location,
            model_name=model_name,
            iter_name=iter_name
        )

        if overwrite_aggregated_od:
            for matrix_format in ['od_from', 'od_to']:
                mat_p.aggregate_matrices(
                    import_dir=exports['od'],
                    export_dir=exports['aggregated_od'],
                    trip_origin='hb',
                    matrix_format=matrix_format,
                    years_needed=[year],
                    p_needed=hb_p_needed,
                    m_needed=modes_needed,
                    ca_needed=ca_needed,
                    tp_needed=tp_needed
                )
            mat_p.aggregate_matrices(
                import_dir=exports['od'],
                export_dir=exports['aggregated_od'],
                trip_origin='nhb',
                matrix_format='od',
                years_needed=[year],
                p_needed=nhb_p_needed,
                m_needed=modes_needed,
                ca_needed=ca_needed,
                tp_needed=tp_needed
            )

        if overwrite_compiled_od:
            mat_p.build_compile_params(
                import_dir=exports['aggregated_od'],
                export_dir=params['compile'],
                matrix_format='od',
                years_needed=[year],
                ca_needed=ca_needed)

            compile_params_fname = du.get_compile_params_name('od', str(year))
            compile_param_path = os.path.join(params['compile'],
                                              compile_params_fname)
            du.compile_od(
                od_folder=exports['aggregated_od'],
                write_folder=exports['compiled_od'],
                compile_param_path=compile_param_path,
                build_factor_pickle=True,
                factor_pickle_path=params['compile']
            )

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

    # TODO: Move these functions to utils as they are copied in multiple places
    #  convert_growth_off_base_year()
    #  get_grown_values()
    #  growth_recombination()
    #  check nearby functions for same issue
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
                (growth_dataframe.loc[:, year] - 1)
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
        #TODO GOt a better version in production_generator
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

    def generate_output_paths(self,
                              output_location: str,
                              model_name: str,
                              iter_name: str,
                              import_location: str = None
                              ) -> Tuple[dict, dict, dict]:
        """

        Parameters
        ----------
        output_location:
            The directory to create the new output directory in - a dir named
            self._out_dir (NorMITs Demand) should exist here. Usually
            a drive name e.g. Y:/

        model_name:
            TfN model name in use e.g. norms or noham

        iter_name:
            The name of the iteration being run. Usually of the format iterx,
            where x is a number, e.g. iter3

        import_location:
            The directory the import directory exists - a dir named
            self._out_dir (NorMITs Demand) should exist here. Usually
            a drive name e.g. Y:/
            If left as None, will default to class import location

        Returns
        -------
        imports:
            Dictionary of import paths with the following keys:
            imports, lookups, seed_dists, default

        exports:
            Dictionary of export paths with the following keys:
            productions, attractions, pa, od, pa_24, od_24, sectors

        params:
            Dictionary of parameter export paths with the following keys:
            compile, tours

        """
        # Init
        model_name = model_name.lower()
        if import_location is None:
            import_location = self.import_location

        # ## IMPORT PATHS ## #
        # Generate import and export paths
        model_home = os.path.join(import_location, self._out_dir)
        import_home = os.path.join(model_home, 'import')

        imports = {
            'home': import_home,
            'tp_splits': os.path.join(import_home, 'tp_splits'),
            'lookups': os.path.join(model_home, 'lookup'),
            'seed_dists': os.path.join(import_home, model_name, 'seed_distributions')
        }

        #  ## EXPORT PATHS ## #
        # Create home paths
        fname_parts = [
            output_location,
            self._out_dir,
            model_name,
            self.__version__ + "-EFS_Output",
            iter_name,
        ]
        export_home = os.path.join(*fname_parts)
        matrices_home = os.path.join(export_home, 'Matrices')
        post_me_home = os.path.join(matrices_home, 'Post-ME Matrices')

        # Create consistent filenames
        pa = 'PA Matrices'
        pa_24 = '24hr PA Matrices'
        od = 'OD Matrices'
        od_24 = '24hr OD Matrices'
        compiled = 'Compiled'
        aggregated = 'Aggregated'

        exports = {
            'home': export_home,
            'productions': os.path.join(export_home, 'Productions'),
            'attractions': os.path.join(export_home, 'Attractions'),
            'sectors': os.path.join(export_home, 'Sectors'),

            # Pre-ME
            'pa': os.path.join(matrices_home, pa),
            'pa_24': os.path.join(matrices_home, pa_24),
            'od': os.path.join(matrices_home, od),
            'od_24': os.path.join(matrices_home, od_24),

            'compiled_od': os.path.join(matrices_home, ' '.join([compiled, od])),

            'aggregated_pa_24': os.path.join(matrices_home, ' '.join([aggregated, pa_24])),
            'aggregated_od': os.path.join(matrices_home, ' '.join([aggregated, od])),
        }

        for _, path in exports.items():
            du.create_folder(path, chDir=False)

        # Post-ME
        post_me_exports = {
            'pa': os.path.join(post_me_home, pa),
            'pa_24': os.path.join(post_me_home, pa_24),
            'od': os.path.join(post_me_home, od),
            'od_24': os.path.join(post_me_home, od_24),
            'compiled_od': os.path.join(post_me_home, ' '.join([compiled, od])),
        }

        for _, path in post_me_exports.items():
            du.create_folder(path, chDir=False)

        # Combine into full export dict
        exports['post_me'] = post_me_exports

        # ## PARAMS OUT ## #
        param_home = os.path.join(export_home, 'Params')

        params = {
            'home': param_home,
            'compile': os.path.join(param_home, 'Compile Params'),
            'tours': os.path.join(param_home, 'Tour Proportions')
        }

        return imports, exports, params

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
                            max_iters=number_of_iterations,
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


def nhb_furness(p_import,
                seed_nhb_dist_dir,
                od_export,
                required_purposes,
                required_modes,
                years_needed,
                replace_zero_vals,
                zero_infill,
                nhb_productions_fname='internal_nhb_productions.csv',
                use_zone_id_subset=False):

    """
    Provides a one-iteration Furness constrained on production
    with options whether to replace zero values on the seed

    Essentially distributes the Productions based on the seed nhb dist
    TODO: Actually add in some furnessing
    TODO: Fully integrate into EFS

    Return:
    ----------
    None
    """
    # TODO: Add in file exists checks

    # For every year, purpose, mode
    loop_iter = itertools.product(years_needed,
                                  required_purposes,
                                  required_modes)
    for year, purpose, mode in loop_iter:
        # ## Read in Files ## #
        # Create year fname
        year_p_fname = '_'.join(
            ["yr" + str(year), nhb_productions_fname]
        )

        # Read in productions
        p_path = os.path.join(p_import, year_p_fname)
        productions = pd.read_csv(p_path)

        # select needed productions
        productions = productions.loc[productions["p"] == purpose]
        productions = productions.loc[productions["m"] == mode]

        # read in nhb_seeds
        seed_fname = du.get_dist_name(
            'nhb',
            'pa',
            purpose=str(purpose),
            mode=str(mode),
            csv=True
        )
        nhb_seeds = pd.read_csv(os.path.join(seed_nhb_dist_dir, seed_fname))

        # convert from wide to long format
        nhb_seeds = nhb_seeds.melt(
            id_vars=['p_zone'],
            var_name='a_zone',
            value_name='seed_vals'
        )

        # Need to make sure they are the correct types
        nhb_seeds['a_zone'] = nhb_seeds['a_zone'].astype(float).astype(int)
        productions['p_zone'] = productions['p_zone'].astype(int)

        if use_zone_id_subset:
            zone_subset = [259, 267, 268, 270, 275, 1171, 1173]
            nhb_seeds = du.get_data_subset(
                nhb_seeds, 'p_zone', zone_subset)
            nhb_seeds = du.get_data_subset(
                nhb_seeds, 'a_zone', zone_subset)

        # Check the productions and seed zones match
        p_zones = set(productions["p_zone"].tolist())
        seed_zones = set(nhb_seeds["p_zone"].tolist())

        # Skip check if we're using a subset
        if use_zone_id_subset:
            print("WARNING! Using a zone subset. Can't check seed "
                  "zones are valid!")
        else:
            if p_zones != seed_zones:
                raise ValueError("Production and seed attraction zones "
                                 "do not match.")

        # Infill zero values
        if replace_zero_vals:
            mask = (nhb_seeds["seed_vals"] == 0)
            nhb_seeds.loc[mask, "seed_vals"] = zero_infill

        # Calculate seed factors by zone
        # (The sum of zone seed factors should equal 1)
        unq_zone = nhb_seeds['p_zone'].drop_duplicates()
        for zone in unq_zone:
            zone_mask = (nhb_seeds['p_zone'] == zone)
            nhb_seeds.loc[zone_mask, 'seed_factor'] = (
                    nhb_seeds[zone_mask]['seed_vals'].values
                    /
                    nhb_seeds[zone_mask]['seed_vals'].sum()
            )
        nhb_seeds = nhb_seeds.reindex(
            ['p_zone', 'a_zone', 'seed_factor'],
            axis=1
        )

        # Use the seed factors to Init P-A trips
        init_pa = pd.merge(
            nhb_seeds,
            productions,
            on=["p_zone"])
        init_pa["trips"] = init_pa["seed_factor"] * init_pa["trips"]

        # TODO: Some actual furnessing should happen here!
        final_pa = init_pa

        # ## Output the furnessed PA matrix to file ## #
        # Generate path
        nhb_dist_fname = du.get_dist_name(
            'nhb',
            'od',
            str(year),
            str(purpose),
            str(mode),
            csv=True
        )
        out_path = os.path.join(od_export, nhb_dist_fname)

        # Convert from long to wide format and output
        # TODO: Generate output name based on model name
        du.long_to_wide_out(
            final_pa.rename(columns={'p_zone': 'norms_zone_id'}),
            v_heading='norms_zone_id',
            h_heading='a_zone',
            values='trips',
            out_path=out_path
        )
        print("NHB Distribution %s complete!" % nhb_dist_fname)


def write_input_info(output_path,
                     base_year: int,
                     future_years: List[int],
                     desired_zoning: str,
                     alt_pop_base_year_file: str,
                     alt_households_base_year_file: str,
                     alt_worker_base_year_file: str,
                     alt_pop_growth_assumption_file: str,
                     alt_households_growth_assumption_file: str,
                     alt_worker_growth_assumption_file: str,
                     alt_pop_split_file: str,
                     distribution_method: str,
                     seed_dist_location: str,
                     purposes_needed: List[int],
                     modes_needed: List[int],
                     soc_needed: List[int],
                     ns_needed: List[int],
                     car_availabilities_needed: List[int],
                     integrate_dlog: bool,
                     minimum_development_certainty: str,
                     population_metric: str,
                     constraint_required: List[bool],
                     constraint_method: str,
                     constraint_area: str,
                     constraint_on: str,
                     constraint_source: str,
                     ) -> None:

    out_lines = [
        'Run Date: ' + str(time.strftime('%D').replace('/', '_')),
        'Start Time: ' + str(time.strftime('%T').replace('/', '_')),
        "Base Year: " + str(base_year),
        "Future Years: " + str(future_years),
        "Zoning System: " + desired_zoning,
        "Alternate Population Base Year File: " + str(alt_pop_base_year_file),
        "Alternate Households Base Year File: " + str(alt_households_base_year_file),
        "Alternate Workers Base Year File: " + str(alt_worker_base_year_file),
        "Alternate Population Growth File: " + str(alt_pop_growth_assumption_file),
        "Alternate Households Growth File: " + str(alt_households_growth_assumption_file),
        "Alternate Workers Growth File: " + str(alt_worker_growth_assumption_file),
        "Alternate Population Split File: " + str(alt_pop_split_file),
        "Distribution Method: " + distribution_method,
        "Seed Distribution Location: " + seed_dist_location,
        "Purposes Used: " + str(purposes_needed),
        "Modes Used: " + str(modes_needed),
        "Soc Used: " + str(soc_needed),
        "Ns Used: " + str(ns_needed),
        "Car Availabilities Used: " + str(car_availabilities_needed),
        "Development Log Integrated: " + str(integrate_dlog),
        "Minimum Development Certainty: " + str(minimum_development_certainty),
        "Population Metric: " + population_metric,
        "Constraints Used On: " + str(constraint_required),
        "Constraint Method: " + constraint_method,
        "Constraint Area: " + constraint_area,
        "Constraint On: " + constraint_on,
        "Constraint Source: " + constraint_source
    ]
    with open(output_path, 'w') as out:
        out.write('\n'.join(out_lines))


def main():
    use_zone_id_subset = False
    echo = False

    iter_num = 0
    output_location = "E:/"

    efs = ExternalForecastSystem(use_zone_id_subset=use_zone_id_subset)
    # efs.run(desired_zoning="norms_2015",
    #         constraint_source="Default",
    #         output_location=output_location,
    #         iter_num=iter_num,
    #         echo_distribution=echo)

    # efs.pa_to_od(
    #     output_location=output_location,
    #     iter_num=iter_num,
    #     overwrite_hb_tp_pa=False,
    #     overwrite_hb_tp_od=False,
    #     echo=echo
    # )
    #
    # efs.run_nhb(
    #     output_location=output_location,
    #     iter_num=iter_num,
    #     overwrite_nhb_productions=False,
    #     overwrite_nhb_od=False,
    #     overwrite_nhb_tp_od=False
    # )

    efs.pre_me_compile_od_matrices(
        output_location=output_location,
        iter_num=iter_num,
        overwrite_aggregated_od=True,
        overwrite_compiled_od=True
    )

    # TODO:
    # efs.generate_post_me_tour_proportions()
    #
    # efs.future_year_pa_to_od()


def main2():
    model_name = 'noham'
    if model_name == 'norms':
        ca_needed = consts.CA_NEEDED
        from_pcu = False
    elif model_name == 'noham':
        ca_needed = None
        from_pcu = True
    else:
        raise ValueError("I don't know what model this is? %s"
                         % str(model_name))

    decompile_od_bool = True
    gen_tour_proportions_bool = True
    aggregate_pa_bool = False
    tour_prop_pa2od_bool = False

    # WARNING: PATHS HAVE CHANGED. THIS CODE WONT RUN NOW
    # Testing code for NoHam
    
    if decompile_od_bool:
        # od2pa.convert_to_efs_matrices(
        #     import_path=r'E:\NorMITs Demand\noham\v2_2-EFS_Output\iter0\Matrices\Post-ME Matrices\Compiled OD Matrices\from_noham',
        #     export_path=r'E:\NorMITs Demand\noham\v2_2-EFS_Output\iter0\Matrices\Post-ME Matrices\Compiled OD Matrices',
        #     matrix_format='od',
        #     user_class=True,
        #     to_wide=True,
        #     wide_col_name=model_name + '_zone_id'
        # )

        if from_pcu:
            vo.people_vehicle_conversion(
                input_folder=r'E:\NorMITs Demand\noham\v2_2-EFS_Output\iter0\Matrices\Post-ME Matrices\Compiled OD Matrices',
                import_folder=r'Y:\NorMITs Demand\import',
                export_folder=r'E:\NorMITs Demand\noham\v2_2-EFS_Output\iter0\Matrices\Post-ME Matrices\Compiled OD Matrices\no_pcu',
                mode=str(consts.MODES_NEEDED[0]),
                method='to_people',
                out_format='wide'
            )

        exit()

        od2pa.decompile_od(
            od_import=r'E:\NorMITs Demand\noham\v2_2-EFS_Output\iter0\Matrices\Post-ME Matrices\Compiled OD Matrices',
            od_export=r'E:\NorMITs Demand\noham\v2_2-EFS_Output\iter0\Matrices\Post-ME Matrices\OD Matrices',
            decompile_factors_path=r'E:\NorMITs Demand\noham\v2_2-EFS_Output\iter0\Params\Compile Params/od_compilation_factors.pickle',
            year=consts.BASE_YEAR
        )

    if gen_tour_proportions_bool:
        mat_p.generate_tour_proportions(
            od_import=r'E:\NorMITs Demand\noham\v2_2-EFS_Output\iter0\Matrices\Post-ME Matrices\OD Matrices',
            pa_export=r'E:\NorMITs Demand\noham\v2_2-EFS_Output\iter0\Matrices\Post-ME Matrices\PA Matrices',
            tour_proportions_export=r'E:\NorMITs Demand\noham\v2_2-EFS_Output\iter0\Params\Tour Proportions',
            year=consts.BASE_YEAR,
            ca_needed=ca_needed
        )

    if aggregate_pa_bool:
        mat_p.aggregate_matrices(
            import_dir=r'E:/NorMITs Demand\norms\v2_2-EFS_Output\iter0\24hr PA Matrices',
            export_dir=r'E:/NorMITs Demand\norms\v2_2-EFS_Output\iter0\Aggregated 24hr PA Matrices',
            trip_origin='hb',
            matrix_format='pa',
            years_needed=consts.FUTURE_YEARS,
            p_needed=consts.ALL_HB_P,
            ca_needed=ca_needed,
            m_needed=consts.MODES_NEEDED,
        )

    if tour_prop_pa2od_bool:
        pa2od.build_od_from_tour_proportions(
            pa_import=r'E:/NorMITs Demand\norms\v2_2-EFS_Output\iter0\Aggregated 24hr PA Matrices',
            od_export=r'E:/NorMITs Demand\norms\v2_2-EFS_Output\iter0\Post-ME OD Matrices',
            tour_proportions_dir=r'E:\NorMITs Demand\norms\v2_2-EFS_Output\iter0\tour_proportions',
            ca_needed=ca_needed
        )


if __name__ == '__main__':
    # main()
    main2()
