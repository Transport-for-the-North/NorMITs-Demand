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
import distribution as dm
import efs_production_generator as pm
import efs_attraction_generator as am
import efs_constrainer as constrainer

from efs_constrainer import ForecastConstrainer
from zone_translator import ZoneTranslator

from demand_utilities import utils as du
from demand_utilities.sector_reporter_v2 import SectorReporter

# TODO: Implement multiprocessing
# TODO: Determine the TfN model name based on the given mode
# TODO: Output a run log instead of printing everything to the terminal.
# TODO: On error, output a simple error report

# TODO: Fix dtype error from pandas on initialisation
#  More info here:
#  https://stackoverflow.com/questions/24251219/pandas-read-csv-low-memory-and-dtype-options

# TODO: CLean up unnecessary processing and files left over from production
#  model rewrite.
# This includes: all household files, car association , trip_rates,
# mode splits, mode time splits.
# THe new production model reads these files in as needed


class ExternalForecastSystem:
    # ## Class Constants ## #
    __version__ = "v2_3"
    _out_dir = "NorMITs Demand"

    # defines all non-year columns
    column_dictionary = consts.EFS_COLUMN_DICTIONARY

    def __init__(self,
                 model_name: str,

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
                 import_location: str = "Y:/",
                 output_location: str = "E:/",

                 use_zone_id_subset: bool = False,
                 ):
        """
        #TODO
        """
        self.model_name = model_name
        self.use_zone_id_subset = use_zone_id_subset
        self.output_location = output_location
        self.import_location = import_location

        input_dir = os.path.join(import_location,
                                 self._out_dir,
                                 'inputs',
                                 'default')

        self.msoa_zones_path = os.path.join(input_dir, value_zones_file)

        print("Initiating External Forecast System...")
        
        begin_time = time.time()
        current_time = begin_time

        # Read in population files
        file_path = os.path.join(input_dir, population_value_file)
        self.population_values = du.safe_read_csv(file_path)

        file_path = os.path.join(input_dir, population_growth_file)
        self.population_growth = du.safe_read_csv(file_path)

        file_path = os.path.join(input_dir, population_constraint_file)
        self.population_constraint = du.safe_read_csv(file_path)

        # file_path = os.path.join(input_dir, future_population_ratio_file)
        # self.future_population_ratio = du.safe_read_csv(file_path)

        # Households files
        # file_path = os.path.join(input_dir, households_value_file)
        # self.households_values = du.safe_read_csv(file_path)
        #
        # file_path = os.path.join(input_dir, household_growth_file)
        # self.households_growth = du.safe_read_csv(file_path)
        #
        # file_path = os.path.join(input_dir, households_constraint_file)
        # self.households_constraint = du.safe_read_csv(file_path)
        #
        # file_path = os.path.join(input_dir, housing_type_split_file)
        # self.housing_type_split = du.safe_read_csv(file_path)
        #
        # file_path = os.path.join(input_dir, housing_occupancy_file)
        # self.housing_occupancy = du.safe_read_csv(file_path)

        # Worker files
        file_path = os.path.join(input_dir, worker_value_file)
        self.worker_values = du.safe_read_csv(file_path)

        file_path = os.path.join(input_dir, worker_growth_file)
        self.worker_growth = du.safe_read_csv(file_path)

        file_path = os.path.join(input_dir, worker_constraint_file)
        self.worker_constraint = du.safe_read_csv(file_path)

        # file_path = os.path.join(input_dir, worker_ratio_file)
        # self.worker_splits = du.safe_read_csv(file_path)

        # Production and attraction files
        # file_path = os.path.join(input_dir, production_trip_rates_file)
        # self.production_trip_rates = du.safe_read_csv(file_path)

        # file_path = os.path.join(input_dir, hb_mode_split_file)
        # self.hb_mode_split = du.safe_read_csv(file_path)

        # file_path = os.path.join(input_dir, hb_mode_time_split_file)
        # self.hb_mode_time_split = du.safe_read_csv(file_path)

        # file_path = os.path.join(input_dir, split_handler_file)
        # self.split_handler = du.safe_read_csv(file_path)

        # file_path = os.path.join(input_dir, traveller_types_file)
        # self.traveller_types = du.safe_read_csv(file_path)

        # file_path = os.path.join(input_dir, attraction_weights_file)
        # self.attraction_weights = du.safe_read_csv(file_path)

        # Zone and area files
        self.value_zoning = value_zoning

        file_path = os.path.join(input_dir, value_zones_file)
        self.value_zones = du.safe_read_csv(file_path)

        file_path = os.path.join(input_dir, area_types_file)
        self.area_types = du.safe_read_csv(file_path)

        file_path = os.path.join(input_dir, area_grouping_file)
        self.area_grouping = du.safe_read_csv(file_path)

        file_path = os.path.join(input_dir, msoa_area_types_file)
        self.msoa_area_types = du.safe_read_csv(file_path)

        file_path = os.path.join(input_dir, zone_areatype_lookup_file)
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
            purposes_needed: List[int] = consts.PURPOSES_NEEDED,
            modes_needed: List[int] = consts.MODES_NEEDED,
            soc_needed: List[int] = consts.SOC_NEEDED,
            ns_needed: List[int] = consts.NS_NEEDED,
            car_availabilities_needed: List[int] = consts.CA_NEEDED,
            dlog_file: str = None,
            dlog_split_file: str = None,
            minimum_development_certainty: str = "MTL",
            population_metric: str = "Population",  # Households, Population
            constraint_required: List[bool] = consts.CONSTRAINT_REQUIRED_DEFAULT,
            constraint_method: str = "Percentage",  # Percentage, Average
            constraint_area: str = "Designated",  # Zone, Designated, All
            constraint_on: str = "Growth",  # Growth, All
            constraint_source: str = "Grown Base",  # Default, Grown Base, Model Grown Base
            outputting_files: bool = True,
            recreate_productions: bool = True,
            recreate_attractions: bool = True,
            iter_num: int = 0,
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
        # Init
        if output_location is None:
            output_location = self.output_location

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

        year_list = [str(x) for x in [base_year] + future_years]

        # Validate inputs
        _input_checks(iter_num=iter_num, m_needed=modes_needed)
        mode_needed = modes_needed[0]

        # ## PREPARE OUTPUTS ## #
        # TODO: Generate output paths when EFS is initialised
        print("Initialising outputs...")
        imports, exports, _ = self.generate_output_paths(output_location,
                                                         self.model_name,
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
            imports['seed_dists'],
            purposes_needed,
            mode_needed,
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
        # pop_ratio_cols = self.column_dictionary["population_ratio"] + year_list

        # hh_cols = self.column_dictionary["households"] + year_list
        # hh_occupancy_cols = self.column_dictionary["housing_occupancy"] + year_list

        emp_cols = self.column_dictionary["employment"] + year_list
        # emp_ratio_cols = self.column_dictionary["employment_ratio"] + year_list

        # production_trip_cols = self.column_dictionary["production_trips"] + year_list
        # mode_split_cols = self.column_dictionary["mode_split"] + year_list
        # attraction_weight_cols = self.column_dictionary["attraction_weights"] + year_list

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
                alt_worker_growth_assumption_file,
                base_year_pop_cols,
                base_year_hh_cols
            )

            population_values = integrated_assumptions[0][base_year_pop_cols]
            # households_values = integrated_assumptions[1][base_year_hh_cols]
            worker_values = integrated_assumptions[2][base_year_workers_cols]
            population_growth = integrated_assumptions[3][pop_cols]
            # households_growth = integrated_assumptions[4][hh_cols]
            worker_growth = integrated_assumptions[5][emp_cols]

            # TODO: Remove unneeded files
            # population_split = self.future_population_ratio[pop_ratio_cols].copy()
            # housing_type_split = self.housing_type_split[hh_occupancy_cols].copy()
            # housing_occupancy = self.housing_occupancy[hh_occupancy_cols].copy()
            # hb_mode_split = self.hb_mode_split[mode_split_cols].copy()
            # msoa_area_types = self.msoa_area_types.copy()
            zone_areatype_lookup = self.zone_areatype_lookup.copy()
            # worker_split = self.worker_splits[emp_ratio_cols].copy()

            # trip_rates = self.production_trip_rates[
            #     production_trip_cols
            # ].copy().rename(
            #     # Rename to cols names used in code
            #     columns={
            #         "traveller_type": "traveller_type_id",
            #         "area_type": "area_type_id",
            #         "p": "purpose_id"
            #     }
            # )
            #
            # car_association = self.traveller_types[[
            #     "cars", "traveller_type"
            # ]].copy().rename(columns={"traveller_type": "traveller_type_id"})
            #
            # car_association["car_availability_id"] = 0
            # no_car_mask = (car_association["cars"] == 0)
            #
            # car_association[no_car_mask]["car_availability_id"] = 1
            # car_association[-no_car_mask]["car_availability_id"] = 2

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
            # population_split = self.future_population_ratio[pop_ratio_cols].copy()

            # households_values = self.households_values[base_year_hh_cols].copy()
            # households_growth = self.households_growth[hh_cols].copy()
            # housing_type_split = self.housing_type_split[hh_occupancy_cols].copy()
            # housing_occupancy = self.housing_occupancy[hh_occupancy_cols].copy()

            worker_values = self.worker_values[base_year_workers_cols].copy()
            worker_growth = self.worker_growth[emp_cols].copy()
            # worker_split = self.worker_splits[emp_ratio_cols].copy()

            # # Need to rename cols to names used in code
            # trip_rates = self.production_trip_rates[production_trip_cols].copy()
            # trip_rates = trip_rates.rename(
            #     columns={
            #         "traveller_type": "traveller_type_id",
            #         "area_type": "area_type_id",
            #         "p": "purpose_id"
            #     }
            # )

            # hb_mode_split = self.hb_mode_split[mode_split_cols].copy()
            msoa_area_types = self.msoa_area_types.copy()
            zone_areatype_lookup = self.zone_areatype_lookup.copy()

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

            # car_association = self.traveller_types[[
            #         "cars",
            #         "traveller_type"
            # ]].copy().rename(columns={"traveller_type": "traveller_type_id"})
            #
            # car_association["car_availability_id"] = 0
            # no_car_mask = (car_association["cars"] == "0")
            #
            # # set up ids (-no_car_mask is the inversion of no_car_mask)
            # car_association.loc[no_car_mask, "car_availability_id"] = 1
            # car_association.loc[-no_car_mask, "car_availability_id"] = 2

            # car_association = car_association[[
            #     "traveller_type_id",
            #     "car_availability_id"
            # ]]

            # attraction_weights = self.attraction_weights[attraction_weight_cols].copy()

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

            # households_constraint = self.households_constraint[hh_cols].copy()

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
            population_constraint = self.population_constraint[pop_cols].copy()
            population_constraint = constrainer.grow_constraint(
                population_constraint,
                population_growth,
                str(base_year),
                [str(x) for x in future_years]
            )

            # households_constraint = self.households_growth[hh_cols].copy()
            # households_constraint = constrainer.grow_constraint(
            #     households_values,
            #     households_constraint,
            #     str(base_year),
            #     [str(x) for x in future_years]
            # )

            # Update this with attraction model updates
            worker_constraint = self.worker_growth[emp_cols].copy()

            worker_constraint = du.convert_growth_off_base_year(
                worker_constraint,
                str(base_year),
                year_list
            )
            worker_constraint = du.get_grown_values(worker_values,
                                                    worker_constraint,
                                                    "base_year_workers",
                                                    year_list)
            print("Constraint generated!")
            last_time = current_time
            current_time = time.time()
            print("Constraint generation took: %.2f seconds." %
                  (current_time - last_time))

        elif constraint_source == "model grown base":
            raise NotImplementedError("Constraint 'model grown base' selected, "
                                      "this will be created later...")
        else:
            raise ValueError("'%s' is not a recognised constraint source."
                             % constraint_source)

        # ## PRODUCTION GENERATION ## #
        print("Generating productions...")
        production_trips = self.production_generator.run(
            base_year=str(base_year),
            future_years=[str(x) for x in future_years],
            population_growth=population_growth,
            population_constraint=population_constraint,
            import_home=imports['home'],
            msoa_conversion_path=self.msoa_zones_path,
            control_productions=True,
            d_log=development_log,
            d_log_split=development_log_split,
            constraint_required=constraint_required,
            constraint_method=constraint_method,
            constraint_area=constraint_area,
            constraint_on=constraint_on,
            constraint_source=constraint_source,
            designated_area=self.area_grouping.copy(),
            out_path=exports['productions'],
            recreate_productions=recreate_productions,

            population_metric=population_metric,
        )
        print("Productions generated!")
        last_time = current_time
        current_time = time.time()
        print("Production generation took: %.2f seconds" %
              (current_time - last_time))

        # ## ATTRACTION GENERATION ###
        print("Generating attractions...")
        attraction_dataframe, nhb_att = self.attraction_generator.run(
            base_year=str(base_year),
            future_years=[str(x) for x in future_years],
            employment_growth=worker_growth,
            employment_constraint=worker_constraint,
            import_home=imports['home'],
            msoa_conversion_path=self.msoa_zones_path,
            attraction_weights_path=imports['a_weights'],
            control_attractions=True,
            d_log=development_log,
            d_log_split=development_log_split,
            constraint_required=constraint_required,
            constraint_method=constraint_method,
            constraint_area=constraint_area,
            constraint_on=constraint_on,
            constraint_source=constraint_source,
            designated_area=self.area_grouping.copy(),
            out_path=exports['attractions'],
            recreate_attractions=recreate_attractions
        )

        print("Attractions generated!")
        last_time = current_time
        current_time = time.time()
        print("Employment and Attraction generation took: %.2f seconds" %
              (current_time - last_time))

        # # ## ATTRACTION MATCHING ## #
        print("Matching attractions...")
        attraction_dataframe = match_attractions_to_productions(
            attraction_dataframe, production_trips, year_list)
        print("Attractions matched!")
        last_time = current_time
        current_time = time.time()
        print("Attraction matching took: %.2f seconds" %
              (current_time - last_time))

        # # ## ATTRACTION WEIGHT GENERATION ## #
        print("Generating attraction weights...")
        attraction_weights = du.convert_to_weights(
            attraction_dataframe,
            year_list
        )

        print("Attraction weights generated!")
        last_time = current_time
        current_time = time.time()
        print("Attraction weight generation took: %.2f seconds" %
              (current_time - last_time))

        # To avoid errors lets make sure all columns have the same datatype
        production_trips.columns = production_trips.columns.astype(str)
        attraction_dataframe.columns = attraction_dataframe.columns.astype(str)
        attraction_weights.columns = attraction_weights.columns.astype(str)

        # ## ZONE TRANSLATION ## #
        if desired_zoning != self.value_zoning:
            print("Need to translate zones.")
            print("Translating from: " + self.value_zoning)
            print("Translating to: " + desired_zoning)

            # read in translation dataframe
            output_path = os.path.join(imports['zoning'], desired_zoning + ".csv")
            translation_dataframe = pd.read_csv(output_path)

            # Figure out which columns are the segmentation
            non_split_columns = list(production_trips.columns)
            for year in year_list:
                non_split_columns.remove(year)

            converted_productions = self.zone_translator.run(
                production_trips,
                translation_dataframe,
                self.value_zoning,
                desired_zoning,
                non_split_columns=non_split_columns
            )

            non_split_columns = list(attraction_dataframe.columns)
            non_split_columns = [x for x in non_split_columns if x not in year_list]
            converted_pure_attractions = self.zone_translator.run(
                attraction_dataframe,
                translation_dataframe,
                self.value_zoning,
                desired_zoning,
                non_split_columns=non_split_columns
            )

            non_split_columns = list(nhb_att.columns)
            non_split_columns = [x for x in non_split_columns if x not in year_list]
            converted_nhb_att = self.zone_translator.run(
                nhb_att,
                translation_dataframe,
                self.value_zoning,
                desired_zoning,
                non_split_columns=non_split_columns
            )

            non_split_columns = list(attraction_weights.columns)
            non_split_columns = [x for x in non_split_columns if x not in year_list]
            converted_attractions = self.zone_translator.run(
                attraction_weights,
                translation_dataframe,
                self.value_zoning,
                desired_zoning,
                non_split_columns=non_split_columns
            )

            print("Zone translation completed!")
            last_time = current_time
            current_time = time.time()
            print("Zone translation took: %.2f seconds" %
                  (current_time - last_time))
        else:
            converted_productions = production_trips.copy()
            converted_attractions = attraction_weights.copy()
            converted_pure_attractions = attraction_dataframe.copy()
            converted_nhb_att = nhb_att.copy()

        # Write Translated p/a to file
        fname = desired_zoning + "_productions.csv"
        converted_productions.to_csv(
            os.path.join(exports['productions'], fname),
            index=False
        )

        fname = desired_zoning + "_attractions.csv"
        converted_pure_attractions.to_csv(
            os.path.join(exports['attractions'], fname),
            index=False
        )

        fname = desired_zoning + "_nhb_attractions.csv"
        converted_nhb_att.to_csv(
            os.path.join(exports['attractions'], fname),
            index=False
        )

        # ## DISTRIBUTION ## #
        if distribution_method == "furness":
            print("Generating distributions...")
            dm.distribute_pa(
                productions=converted_productions,
                attraction_weights=converted_attractions,
                years_needed=year_list,
                p_needed=purposes_needed,
                m_needed=modes_needed,
                soc_needed=soc_needed,
                ns_needed=ns_needed,
                ca_needed=car_availabilities_needed,
                seed_dist_dir=imports['seed_dists'],
                dist_out=exports['pa_24'],
                audit_out=exports['audits'],
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
        zone_system_file = os.path.join(imports['zoning'],
                                        desired_zoning + '.csv')
        sector_grouping_file = os.path.join(imports['zoning'],
                                            "tfn_level_one_sectors_norms_grouping.csv")

        sector_totals = self.sector_reporter.calculate_sector_totals(
                converted_productions,
                grouping_metric_columns=year_list,
                zone_system_name=desired_zoning,
                zone_system_file=zone_system_file,
                sector_grouping_file=sector_grouping_file
                )

        pm_sector_total_dictionary = {}

        for purpose in purposes_needed:
            # TODO: Update sector reporter.
            #  Sector totals don't currently allow per purpose reporting

            pm_productions = converted_productions.copy()

            pm_sector_totals = self.sector_reporter.calculate_sector_totals(
                pm_productions,
                grouping_metric_columns=year_list,
                zone_system_name=desired_zoning,
                zone_system_file=zone_system_file,
                sector_grouping_file=sector_grouping_file
            )

            key_string = str(purpose)
            pm_sector_total_dictionary[key_string] = pm_sector_totals

        # ## OUTPUTS ## #
        # TODO: Properly integrate this

        if outputting_files:
            if output_location is not None:
                print("Saving files to: " + output_location)

                # Distributions moved to furness
                # Pop generation moved
                # Production Generation moved
                # Final workers out moved
                # Attractions output moved
                # Translated production and attractions moved

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
        _input_checks(iter_num=iter_num, m_needed=modes_needed)

        # Generate paths
        imports, exports, _ = self.generate_output_paths(
            output_location=output_location,
            model_name=self.model_name,
            iter_name=iter_name
        )
        # TODO: Add time print outs

        # TODO: Check if tp pa matrices exist first
        if overwrite_hb_tp_pa:
            print("Converting HB 24hr PA to time period split PA...")
            pa2od.efs_build_tp_pa(
                tp_import=imports['tp_splits'],
                pa_import=exports['pa_24'],
                pa_export=exports['pa'],
                years_needed=years_needed,
                p_needed=purposes_needed,
                m_needed=modes_needed,
                soc_needed=soc_needed,
                ns_needed=ns_needed,
                ca_needed=ca_needed
            )
            print('HB time period split PA matrices compiled!\n')

        # TODO: Check if od matrices exist first
        if overwrite_hb_tp_od:
            print('Converting time period split PA to OD...')
            pa2od.efs_build_od(
                pa_import=exports['pa'],
                od_export=exports['od'],
                p_needed=purposes_needed,
                m_needed=modes_needed,
                soc_needed=soc_needed,
                ns_needed=ns_needed,
                ca_needed=ca_needed,
                years_needed=years_needed,
                phi_type='fhp_tp',
                aggregate_to_wday=True,
                echo=echo
            )
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
        _input_checks(iter_num=iter_num, m_needed=modes_needed)

        # Generate paths
        imports, exports, _ = self.generate_output_paths(
            output_location=output_location,
            model_name=self.model_name,
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
            dm.nhb_furness(
                p_import=exports['productions'],
                a_import=exports['attractions'],
                seed_dist_dir=imports['seed_dists'],
                pa_export=exports['pa_24'],
                model_name=self.model_name,
                p_needed=nhb_purposes_needed,
                m_needed=modes_needed,
                years_needed=[str(x) for x in years_needed],
                seed_infill=0.01
            )
            print('NHB productions "furnessed"\n')

        if overwrite_nhb_tp_od:
            print("Converting NHB 24hr OD to time period split OD...")
            pa2od.efs_build_tp_pa(
                tp_import=imports['tp_splits'],
                pa_import=exports['pa_24'],
                pa_export=exports['pa'],
                years_needed=years_needed,
                p_needed=nhb_purposes_needed,
                m_needed=modes_needed,
                matrix_format='pa'
            )
            print('NHB time period split PA matrices compiled!\n')

        print("NHB run complete!")

    def pre_me_compile_od_matrices(self,
                                   year: int = consts.BASE_YEAR,
                                   hb_p_needed: List[int] = consts.PURPOSES_NEEDED,
                                   nhb_p_needed: List[int] = consts.NHB_PURPOSES_NEEDED,
                                   m_needed: List[int] = consts.MODES_NEEDED,
                                   tp_needed: List[int] = consts.TIME_PERIODS,
                                   output_location: str = None,
                                   iter_num: int = 0,
                                   overwrite_aggregated_od: bool = True,
                                   overwrite_compiled_od: bool = True
                                   ) -> None:
        """
        Compiles pre-ME OD matrices produced by EFS into User Class format
        i.e. business, commute, other

        Performs the following actions:
            - Aggregates OD matrices up to p/m/tp segmentation. Will also
              include ca/nca if run for norms.
            - Compiles the aggregated OD matrices into User Class format,
              saving the split factors for decompile later.

        Parameters
        ----------
        year:
            The year to produce compiled OD matrices for.

        hb_p_needed:
            The home based purposes to use when compiling and aggregating
            OD matrices.

        nhb_p_needed:
            The non home based purposes to use when compiling and aggregating
            OD matrices.

        m_needed:
            The mode to use when compiling and aggregating OD matrices. This
            will be used to determine if car availability needs to be included
            or not

        tp_needed:
            The time periods to use when compiling and aggregating OD matrices.

        output_location:
            The directory to create the new output directory in - a dir named
            self._out_dir (NorMITs Demand) should exist here. Usually
            a drive name e.g. Y:/

        iter_num:
            The number of the iteration being run.

        # TODO: Update docs once correct functionality exists
        overwrite_aggregated_od:
            Whether to generate aggregated od matrices or not.

        overwrite_compiled_od
            Whether to generate compiled OD matrices or not.


        Returns
        -------
        None
        """
        # Init
        if output_location is None:
            output_location = self.output_location
        iter_name = 'iter' + str(iter_num)
        _input_checks(iter_num=iter_num, m_needed=m_needed)

        if self.model_name == 'norms' or self.model_name == 'norms_2015':
            ca_needed = consts.CA_NEEDED
        elif self.model_name == 'noham':
            ca_needed = [None]
        else:
            raise ValueError("Got an unexpected model name. Got %s, expected "
                             "either 'norms', 'norms_2015' or 'noham'."
                             % str(self.model_name))

        # Generate paths
        imports, exports, params = self.generate_output_paths(
            output_location=output_location,
            model_name=self.model_name,
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
                    m_needed=m_needed,
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
                m_needed=m_needed,
                ca_needed=ca_needed,
                tp_needed=tp_needed
            )

        if overwrite_compiled_od:
            mat_p.build_compile_params(
                import_dir=exports['aggregated_od'],
                export_dir=params['compile'],
                matrix_format='od',
                years_needed=[year],
                ca_needed=ca_needed,
                tp_needed=tp_needed
            )

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

    def generate_post_me_tour_proportions(self,
                                          year: int = consts.BASE_YEAR,
                                          m_needed: List[int] = consts.MODES_NEEDED,
                                          output_location: str = None,
                                          iter_num: int = 0,
                                          overwrite_decompiled_od=True,
                                          overwrite_tour_proportions=True
                                          ) -> None:
        """
        Uses post-ME OD matrices from the TfN model (NoRMS/NoHAM) to generate
        tour proportions for each OD pair, for each purpose (and ca as
        needed). Also converts OD matrices to PA.

        Performs the following actions:
            - Converts post-ME files into and EFS format as needed. (file name
              changes, converting long to wide as needed.)
            - Decompiles the converted post-ME matrices into purposes (and ca
              when needed) using the split factors produced during pre-me
              OD compilation
            - Generates tour proportions for each OD pair, for each purpose
              (and ca as needed), saving for future year post-ME compilation
              later.
            - Converts OD matrices to PA.

        Parameters
        ----------
        year:
             The year to decompile OD matrices for. (Usually the base year)

        m_needed:
            The mode to use when decompiling OD matrices. This will be used
            to determine if car availability needs to be included or not.

        output_location:
            The directory to create the new output directory in - a dir named
            self._out_dir (NorMITs Demand) should exist here. Usually
            a drive name e.g. Y:/

        iter_num:
            The number of the iteration being run.

        # TODO: Update docs once correct functionality exists
        overwrite_decompiled_od:
            Whether to decompile the post-me od matrices or not

        overwrite_tour_proportions:
            Whether to generate tour proportions or not.

        Returns
        -------
        None
        """
        # Init
        if output_location is None:
            output_location = self.output_location
        iter_name = 'iter' + str(iter_num)
        _input_checks(iter_num=iter_num, m_needed=m_needed)

        if self.model_name == 'norms' or self.model_name == 'norms_2015':
            ca_needed = consts.CA_NEEDED
            from_pcu = False
        elif self.model_name == 'noham':
            ca_needed = [None]
            from_pcu = True
        else:
            raise ValueError("Got an unexpected model name. Got %s, expected "
                             "either 'norms', 'norms_2015' or 'noham'."
                             % str(self.model_name))

        # Generate paths
        imports, exports, params = self.generate_output_paths(
            output_location=output_location,
            model_name=self.model_name,
            iter_name=iter_name
        )

        # TODO: Fix OD2PA to use norms_2015/norms for zone names

        if overwrite_decompiled_od:
            print("Decompiling OD Matrices into purposes...")
            need_convert = od2pa.need_to_convert_to_efs_matrices(
                model_import=exports['post_me']['model_output'],
                od_import=exports['post_me']['compiled_od']
            )
            if need_convert:
                od2pa.convert_to_efs_matrices(
                    import_path=exports['post_me']['model_output'],
                    export_path=exports['post_me']['compiled_od'],
                    matrix_format='od',
                    year=year,
                    user_class=True,
                    to_wide=True,
                    wide_col_name=du.get_model_name(m_needed[0]) + '_zone_id',
                    from_pcu=from_pcu,
                    vehicle_occupancy_import=imports['home']
                )

            # TODO: Stop the filename being hardcoded after integration with TMS
            decompile_factors_path = os.path.join(
                params['compile'],
                'od_compilation_factors.pickle'
            )
            od2pa.decompile_od(
                od_import=exports['post_me']['compiled_od'],
                od_export=exports['post_me']['od'],
                decompile_factors_path=decompile_factors_path,
                year=year
            )

        if overwrite_tour_proportions:
            print("Converting OD matrices to PA and generating tour "
                  "proportions...")
            mat_p.generate_tour_proportions(
                od_import=exports['post_me']['od'],
                zone_translate_dir=imports['zone_translation'],
                pa_export=exports['post_me']['pa'],
                tour_proportions_export=params['tours'],
                year=year,
                ca_needed=ca_needed
            )

    def compile_future_year_od_matrices(self,
                                        years_needed: List[int] = consts.FUTURE_YEARS,
                                        hb_p_needed: List[int] = consts.ALL_HB_P,
                                        m_needed: List[int] = consts.MODES_NEEDED,
                                        output_location: str = None,
                                        iter_num: int = 0,
                                        overwrite_aggregated_pa: bool = True,
                                        overwrite_future_year_od: bool = True
                                        ) -> None:
        """
        Generates future year post-ME OD matrices using the generated tour
        proportions from decompiling post-ME base year matrices, and the
        EFS generated future year PA matrices.

        Performs the following actions:
            - Aggregates EFS future year PA matrices up to the required
              segmentation level to match the generated tour proportions.
              (purpose and car_availability as needed).
            - Uses the base year post-ME tour proportions to convert 24hr PA
              matrices into time-period split OD matrices - outputting to
              file.

        Parameters
        ----------
        years_needed:
            The future years that need converting from PA to OD.

        hb_p_needed:
            The home based purposes to use while converting PA to OD

        m_needed:
            The mode to use during the conversion. This will be used to
            determine if car availability needs to be included or not.

        output_location:
            The directory to create the new output directory in - a dir named
            self._out_dir (NorMITs Demand) should exist here. Usually
            a drive name e.g. Y:/

        iter_num:
            The number of the iteration being run.

        # TODO: Update docs once correct functionality exists
        overwrite_aggregated_pa:
            Whether to generate the aggregated pa matrices or not

        overwrite_future_year_od:
            Whether to convert pa to od or not.

        Returns
        -------
        None
        """
        # Init
        if output_location is None:
            output_location = self.output_location
        iter_name = 'iter' + str(iter_num)
        _input_checks(iter_num=iter_num, m_needed=m_needed)

        if self.model_name == 'norms' or self.model_name == 'norms_2015':
            ca_needed = consts.CA_NEEDED
        elif self.model_name == 'noham':
            ca_needed = [None]
        else:
            raise ValueError("Got an unexpected model name. Got %s, expected "
                             "either 'norms', 'norms_2015' or 'noham'."
                             % str(self.model_name))

        # Generate paths
        imports, exports, params = self.generate_output_paths(
            output_location=output_location,
            model_name=self.model_name,
            iter_name=iter_name
        )

        if overwrite_aggregated_pa:
            mat_p.aggregate_matrices(
                import_dir=exports['pa_24'],
                export_dir=exports['aggregated_pa_24'],
                trip_origin='hb',
                matrix_format='pa',
                years_needed=years_needed,
                p_needed=hb_p_needed,
                ca_needed=ca_needed,
                m_needed=m_needed
            )

        if overwrite_future_year_od:
            pa2od.build_od_from_tour_proportions(
                pa_import=exports['aggregated_pa_24'],
                od_export=exports['post_me']['od'],
                tour_proportions_dir=params['tours'],
                zone_translate_dir=imports['zone_translation'],
                ca_needed=ca_needed
            )

        # TODO: Compile to OD/PA when we know the correct format

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
        # Attraction weights are a bit special, we get these directly from
        # TMS to ensure they are the same - update this on integration
        temp_model_name = 'norms' if model_name == 'norms_2015' else model_name
        tms_path_parts = [
            import_location,
            "NorMITs Synthesiser",
            temp_model_name,
            "Model Zone Lookups",
            "attraction_weights.csv"
        ]
        a_weights_path = os.path.join(*tms_path_parts)

        # Generate import and export paths
        model_home = os.path.join(import_location, self._out_dir)
        import_home = os.path.join(model_home, 'import')
        input_home = os.path.join(model_home, 'inputs', 'default')

        imports = {
            'home': import_home,
            'default_inputs': input_home,
            'tp_splits': os.path.join(import_home, 'tp_splits'),
            'zone_translation': os.path.join(import_home, 'zone_translation'),
            'lookups': os.path.join(model_home, 'lookup'),
            'seed_dists': os.path.join(import_home, model_name, 'seed_distributions'),
            'zoning': os.path.join(input_home, 'zoning'),
            'a_weights': a_weights_path
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
            'audits': os.path.join(export_home, 'Audits'),

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
        compiled_od_path = os.path.join(post_me_home, ' '.join([compiled, od]))
        post_me_exports = {
            'pa': os.path.join(post_me_home, pa),
            'pa_24': os.path.join(post_me_home, pa_24),
            'od': os.path.join(post_me_home, od),
            'od_24': os.path.join(post_me_home, od_24),
            'compiled_od': compiled_od_path,
            'model_output': os.path.join(compiled_od_path, ''.join(['from_', model_name]))
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
        for _, path in params.items():
            du.create_folder(path, chDir=False)

        return imports, exports, params

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


def match_attractions_to_productions(attractions: pd.DataFrame,
                                     productions: pd.DataFrame,
                                     year_list: List[str],
                                     infill: float = 0.001,
                                     echo: bool = False
                                     ) -> pd.DataFrame:
    """
    TODO: Write match_attractions_to_productions doc
    """
    attractions = attractions.copy()
    productions = productions.copy()

    # Make sure all column names are strings
    productions.columns = productions.columns.astype(str)
    attractions.columns = attractions.columns.astype(str)

    purposes = attractions["purpose_id"].unique()

    if echo:
        print("Balancing Attractions...")
        print("Before:")
        for year in year_list:
            print("Year: %s\tProductions: %.2f\tAttractions: %.2f"
                  % (year, productions[year].sum(), attractions[year].sum()))

    attractions = pd.merge(
        attractions,
        productions,
        on=["model_zone_id", "purpose_id"],
        how='outer',
        suffixes=("", "_productions")
    )

    # Infill where P/A don't match
    attractions_cols = year_list.copy()
    productions_cols = [x + '_productions' for x in year_list]
    for col in attractions_cols + productions_cols:
        attractions[col] = attractions[col].fillna(infill)

    # Balance the attractions to the productions
    for purpose in purposes:
        for year in year_list:
            mask = (attractions["purpose_id"] == purpose)
            attractions.loc[mask, year] = (
                    attractions.loc[mask, year].values
                    /
                    (
                        attractions.loc[mask, year].sum()
                        /
                        attractions.loc[mask, year + '_productions'].sum()
                    )
            )

    group_by_cols = ["model_zone_id", "purpose_id"]
    needed_columns = group_by_cols.copy()
    needed_columns.extend(year_list)

    attractions = attractions[needed_columns]
    attractions = attractions.groupby(
        by=group_by_cols,
        as_index=False
    ).sum()

    if echo:
        print("After:")
        for year in year_list:
            print("Year: %s\tProductions: %.2f\tAttractions: %.2f"
                  % (year, productions[year].sum(), attractions[year].sum()))

    return attractions


def _input_checks(iter_num=None,
                  m_needed=None
                  ) -> None:
    """
    Checks that any arguments given are OK. Will raise an error
    for any given input that is not correct.
    """
    if iter_num is not None and iter_num == 0:
        Warning("iter_num is set to 0. This is should only be the case"
                "during testing.")

    if m_needed is not None and len(m_needed) > 1:
        raise ValueError("Was given more than one mode. EFS cannot run "
                         "using more than one mode at a time due to "
                         "different zoning systems for NoHAM and NoRMS "
                         "etc.")


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

    # Running control
    run_base_efs = False
    recreate_productions = True
    recreate_attractions = True

    constrain_population = False

    run_nhb_efs = True
    run_compile_od = False
    run_decompile_od = False
    run_future_year_compile_od = False

    # Controls I/O
    iter_num = 1
    import_location = "Y:/"
    output_location = "E:/"
    model_name = 'norms_2015'   # Make sure the correct mode is being used!!!

    # Set up constraints
    if constrain_population:
        constraints = consts.CONSTRAINT_REQUIRED_DEFAULT
    else:
        constraints = [False] * 6

    # ## RUN START ## #
    efs = ExternalForecastSystem(
        model_name=model_name,
        use_zone_id_subset=use_zone_id_subset,
        import_location=import_location,
        output_location=output_location
    )

    if run_base_efs:
        # Generates HB PA matrices
        efs.run(
            desired_zoning="norms_2015",
            constraint_source="Default",
            output_location=output_location,
            recreate_productions=recreate_productions,
            recreate_attractions=recreate_attractions,
            iter_num=iter_num,
            echo_distribution=echo,
            constraint_required=constraints
        )

    if run_nhb_efs:
        # Convert to HB to OD
        # efs.pa_to_od(
        #     output_location=output_location,
        #     iter_num=iter_num,
        #     overwrite_hb_tp_pa=True,
        #     overwrite_hb_tp_od=True,
        #     echo=echo
        # )

        # Generate NHB PA/OD matrices
        efs.run_nhb(
            output_location=output_location,
            iter_num=iter_num,
            overwrite_nhb_productions=False,
            overwrite_nhb_od=False,
            overwrite_nhb_tp_od=True
        )

    # TODO: Update Integrated OD2PA codebase
    if run_compile_od:
        # Compiles base year OD matrices
        efs.pre_me_compile_od_matrices(
            output_location=output_location,
            iter_num=iter_num,
            overwrite_aggregated_od=True,
            overwrite_compiled_od=True
        )

    if run_decompile_od:
        # Decompiles post-me base year OD matrices - generates tour
        # proportions in the process
        efs.generate_post_me_tour_proportions(
            output_location=output_location,
            iter_num=iter_num,
            overwrite_decompiled_od=False,
            overwrite_tour_proportions=True,
        )

    if run_future_year_compile_od:
        # Uses the generated tour proportions to compile Post-ME OD matrices
        # for future years
        efs.compile_future_year_od_matrices(
            output_location=output_location,
            iter_num=iter_num,
            overwrite_aggregated_pa=True,
            overwrite_future_year_od=True
        )


if __name__ == '__main__':
    main()
