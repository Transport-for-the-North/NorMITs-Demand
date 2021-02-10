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

from typing import List
from typing import Dict
from typing import Tuple

# External libs
import pandas as pd

# self imports
import normits_demand as nd
from normits_demand import version
from normits_demand import efs_constants as consts
from normits_demand.models import efs_production_model as pm

from normits_demand.matrices import pa_to_od as pa2od
from normits_demand.matrices import od_to_pa as od2pa
from normits_demand.matrices import matrix_processing as mat_p

from normits_demand.distribution import furness

from normits_demand.reports import pop_emp_comparator

from normits_demand.utils import general as du
from normits_demand.utils import vehicle_occupancy as vo
from normits_demand.utils import exceptional_growth as eg
from normits_demand.utils import sector_reporter_v2 as sr_v2


# TODO: Output a run log instead of printing everything to the terminal.
# TODO: On error, output a simple error report

# BACKLOG: Implement a logger to log EFS run progress
#  labels: QoL Updates


class ExternalForecastSystem:
    # ## Class Constants ## #
    __version__ = '%s.%s' % (version.MAJOR, version.MINOR)
    _out_dir = "NorMITs Demand"

    # defines all non-year columns
    column_dictionary = consts.EFS_COLUMN_DICTIONARY

    def __init__(self,
                 model_name: str,
                 iter_num: int,
                 scenario_name: str,

                 integrate_dlog: bool = False,

                 dlog_pop_path: str = None,
                 dlog_emp_path: str = None,

                 msoa_lookup_path: str = "zoning/msoa_zones.csv",
                 lad_msoa_lookup_path: str = "zoning/lad_msoa_grouping.csv",

                 import_home: str = "Y:/",
                 export_home: str = "E:/",
                 verbose: str = True
                 ):
        # TODO: Write EFS constructor docs
        # Initialise the timer
        begin_time = time.time()
        current_time = begin_time
        print("Initiating External Forecast System...")

        # Initialise
        du.validate_model_name_and_mode(model_name, consts.MODES_NEEDED)
        self.model_name = du.validate_model_name(model_name)
        self.iter_name = 'iter' + str(iter_num)
        self.scenario_name = du.validate_scenario_name(scenario_name)
        self.integrate_dlog = integrate_dlog
        self.import_location = import_home
        self.output_location = export_home
        self.verbose = verbose

        # TODO: Write function to determine if CA is needed for model_names
        self.is_ca_needed = True
        if self.model_name == 'noham':
            self.is_ca_needed = False

        self.input_zone_system = "MSOA"
        self.output_zone_system = self.model_name

        # TODO: Build zone lookup paths inside generate_output_paths()
        # TODO: Rename generate_output_paths() to _generate_paths()
        self.msoa_lookup_path = msoa_lookup_path
        self.lad_msoa_lookup_path = lad_msoa_lookup_path

        # Don't NTEM Control Future years in scenarios
        self.ntem_control_future_years = not (scenario_name in consts.TFN_SCENARIOS)

        # Setup up import/export paths
        path_dicts = self.generate_output_paths(consts.BASE_YEAR_STR)
        self.imports, self.exports, self.params = path_dicts
        self._setup_scenario_paths()
        self._build_pop_emp_paths()
        self._read_in_default_inputs()
        self._set_up_dlog(dlog_pop_path, dlog_emp_path)
        self.msoa_zones_path = os.path.join(self.imports["zoning"], "msoa_zones.csv")

        # sub-classes
        self.production_generator = nd.EFSProductionGenerator(model_name=model_name)
        self.attraction_generator = nd.EFSAttractionGenerator(model_name=model_name)

        # support utilities tools
        self.sector_reporter = sr_v2.SectorReporter()
        self.zone_translator = nd.ZoneTranslator()

        print("External Forecast System initiated!")
        last_time = current_time
        current_time = time.time()
        print("Initialisation took: %.2f seconds." % (current_time - last_time))

    def _read_in_default_inputs(self) -> None:
        # Change this dep
        input_dir = self.imports['default_inputs']

        # Read in soc and ns as strings if in inputs
        dtypes = {'soc': str, 'ns': str}

        # Read in population files
        file_path = os.path.join(input_dir, self.pop_growth_path)
        self.pop_growth = du.safe_read_csv(file_path, dtype=dtypes)

        file_path = os.path.join(input_dir, self.pop_constraint_path)
        self.pop_constraint = du.safe_read_csv(file_path, dtype=dtypes)

        # Employment files
        file_path = os.path.join(input_dir, self.emp_growth_path)
        self.emp_growth = du.safe_read_csv(file_path, dtype=dtypes)

        file_path = os.path.join(input_dir, self.emp_constraint_path)
        self.emp_constraint = du.safe_read_csv(file_path, dtype=dtypes)

        # Zone and area files
        input_dir = self.imports['default_inputs']
        file_path = os.path.join(input_dir, self.msoa_lookup_path)
        self.msoa_lookup = du.safe_read_csv(file_path)

        file_path = os.path.join(input_dir, self.lad_msoa_lookup_path)
        self.lad_msoa_lookup = du.safe_read_csv(file_path)

    def _setup_scenario_paths(self) -> None:
        """
        Sets up the pop/emp constraint and growth paths

        The paths are built depending on the scenario.

        Returns
        -------
        None
        """
        # Path building is slightly different for default NTEM
        if self.scenario_name == consts.SC00_NTEM:
            # Setup directory paths
            home = self.imports['default_inputs']
            pop_home = os.path.join(home, 'population')
            emp_home = os.path.join(home, 'employment')

            # Build paths
            pop_growth_path = os.path.join(pop_home, 'future_population_growth.csv')
            pop_constraint_path = os.path.join(pop_home, 'future_population_values.csv')
            emp_growth_path = os.path.join(emp_home, 'future_workers_growth.csv')
            emp_constraint_path = os.path.join(emp_home, 'future_workers_growth_values.csv')

        elif self.scenario_name in consts.TFN_SCENARIOS:
            # Setup directory paths
            scenario_home = os.path.join(self.imports['scenarios'],
                                         self.scenario_name)
            pop_home = os.path.join(scenario_home, 'population')
            emp_home = os.path.join(scenario_home, 'employment')

            # Build paths
            pop_growth_path = os.path.join(pop_home, 'future_growth_factors.csv')
            pop_constraint_path = os.path.join(pop_home, 'future_growth_values.csv')
            emp_growth_path = os.path.join(emp_home, 'future_growth_factors.csv')
            emp_constraint_path = os.path.join(emp_home, 'future_growth_values.csv')

        else:
            # Shouldn't be able to get here
            raise ValueError(
                "The given scenario seems to be real, but I don't know how "
                "to build the path for it. Given scenario name: %s"
                % str(self.scenario_name)
            )

        # Finally assign to class attributes
        self.pop_constraint_path = pop_constraint_path
        self.pop_growth_path = pop_growth_path
        self.emp_constraint_path = emp_constraint_path
        self.emp_growth_path = emp_growth_path

    def _build_pop_emp_paths(self):
        # Init
        zone_lookups = consts.TFN_MSOA_SECTOR_LOOKUPS

        # Build the pop paths
        pop_paths = {
            "import_home": self.imports["home"],
            "growth_csv": self.pop_growth_path,
            "constraint_csv": self.pop_constraint_path,
            "sector_grouping_file": os.path.join(self.imports['zoning'],
                                                 zone_lookups["population"])
        }

        # Build the emp paths
        emp_paths = {
            "import_home": self.imports["home"],
            "growth_csv": self.emp_growth_path,
            "constraint_csv": self.emp_constraint_path,
            "sector_grouping_file": os.path.join(self.imports['zoning'],
                                                 zone_lookups["employment"])
        }

        # Assign to dictionary
        self.pop_emp_inputs = {
            "population": pop_paths,
            "employment": emp_paths,
        }

    def _set_up_dlog(self, dlog_pop_path: str, dlog_emp_path: str) -> None:
        # If we're not integrating d-log, set to None
        if not self.integrate_dlog:
            self.dlog_paths = {
                'pop': None,
                'emp': None
            }
            return

        # Set to default paths if nothing given
        if dlog_pop_path is None:
            dlog_pop_path = os.path.join(self.imports["default_inputs"],
                                         'population',
                                         'dlog_residential.csv')

        if dlog_emp_path is None:
            dlog_emp_path = os.path.join(self.imports["default_inputs"],
                                         'employment',
                                         'dlog_nonresidential.csv')

        self.dlog_paths = {
            'pop': dlog_pop_path,
            'emp': dlog_emp_path,
        }

    def run(self,
            base_year: int = 2018,
            future_years: List[int] = consts.FUTURE_YEARS,
            alt_pop_base_year_file: str = None,
            alt_households_base_year_file: str = None,
            alt_worker_base_year_file: str = None,
            alt_pop_growth_assumption_file: str = None,
            alt_households_growth_assumption_file: str = None,
            alt_worker_growth_assumption_file: str = None,
            alt_pop_split_file: str = None,  # THIS ISN'T USED ANYWHERE
            distribution_method: str = "Furness",
            purposes_needed: List[int] = consts.PURPOSES_NEEDED,
            nhb_purposes_needed: List[int] = consts.NHB_PURPOSES_NEEDED,
            modes_needed: List[int] = consts.MODES_NEEDED,
            soc_needed: List[int] = consts.SOC_NEEDED,
            ns_needed: List[int] = consts.NS_NEEDED,
            car_availabilities_needed: List[int] = consts.CA_NEEDED,
            minimum_development_certainty: str = "MTL",
            constraint_required: Dict[str, bool] = consts.CONSTRAINT_REQUIRED_DEFAULT,
            constraint_source: str = "Grown Base",  # Default, Grown Base, Model Grown Base
            recreate_productions: bool = True,
            recreate_attractions: bool = True,
            recreate_nhb_productions: bool = True,
            apply_growth_criteria: bool = True,
            outputting_files: bool = True,
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
            - Use purposes needed / car availabilities needed / modes needed /
            times needed to reduce the amount of calculations to be done.
        """
        # Init
        if output_location is None:
            output_location = self.output_location

        if self.model_name == 'noham':
            car_availabilities_needed = None

        # Set up timing
        begin_time = time.time()
        current_time = begin_time

        # Format inputs
        constraint_source = constraint_source.lower()
        distribution_method = distribution_method.lower()
        minimum_development_certainty = minimum_development_certainty.upper()

        year_list = [str(x) for x in [base_year] + future_years]

        # Validate inputs
        _input_checks(m_needed=modes_needed,
                      constraint_required=constraint_required)

        # ## PREPARE OUTPUTS ## #
        print("Initialising outputs...")
        write_input_info(
            os.path.join(self.exports['home'], "input_parameters.txt"),
            base_year,
            future_years,
            self.output_zone_system,
            alt_pop_base_year_file,
            alt_households_base_year_file,
            alt_worker_base_year_file,
            alt_pop_growth_assumption_file,
            alt_households_growth_assumption_file,
            alt_worker_growth_assumption_file,
            alt_pop_split_file,
            distribution_method,
            self.imports['seed_dists'],
            purposes_needed,
            modes_needed,
            soc_needed,
            ns_needed,
            car_availabilities_needed,
            self.integrate_dlog,
            minimum_development_certainty,
            constraint_required,
            constraint_source,
        )

        # ## GET DATA ## #
        # TODO: Tidy this up
        pop_growth = self.pop_growth.copy()
        emp_growth = self.emp_growth.copy()

        # ## CONSTRAINT BUILDING
        pop_constraint = self.pop_constraint.copy()
        emp_constraint = self.emp_constraint.copy()

        print("Constraints retrieved!")
        last_time = current_time
        current_time = time.time()
        print("Constraint retrieval took: %.2f seconds." %
              (current_time - last_time))

        # ## PRODUCTION GENERATION ## #
        print("Generating productions...")
        production_trips = self.production_generator.run(
            base_year=str(base_year),
            future_years=[str(x) for x in future_years],
            population_growth=pop_growth,
            population_constraint=pop_constraint,
            import_home=self.imports['home'],
            export_home=self.exports['home'],
            msoa_lookup_path=self.msoa_zones_path,
            control_productions=True,
            control_fy_productions=self.ntem_control_future_years,
            dlog=self.dlog_paths['pop'],
            pre_dlog_constraint=constraint_required['pop_pre_dlog'],
            post_dlog_constraint=constraint_required['pop_post_dlog'],
            designated_area=self.lad_msoa_lookup.copy(),
            out_path=self.exports['productions'],
            recreate_productions=recreate_productions,
        )
        last_time = current_time
        current_time = time.time()
        elapsed_time = current_time - last_time
        print("Production generation took: %.2f seconds" % elapsed_time)

        # ## ATTRACTION GENERATION ###
        print("Generating attractions...")
        attraction_dataframe, nhb_att = self.attraction_generator.run(
            out_path=self.exports['attractions'],
            base_year=str(base_year),
            future_years=[str(x) for x in future_years],
            employment_growth=emp_growth,
            employment_constraint=emp_constraint,
            import_home=self.imports['home'],
            export_home=self.exports['home'],
            msoa_lookup_path=self.msoa_zones_path,
            attraction_weights_path=self.imports['a_weights'],
            control_attractions=True,
            control_fy_attractions=self.ntem_control_future_years,
            dlog=self.dlog_paths['emp'],
            pre_dlog_constraint=constraint_required['emp_pre_dlog'],
            post_dlog_constraint=constraint_required['emp_post_dlog'],
            designated_area=self.lad_msoa_lookup.copy(),
            recreate_attractions=recreate_attractions
        )

        last_time = current_time
        current_time = time.time()
        print("Attraction generation took: %.2f seconds" %
              (current_time - last_time))

        # ## Audit the pop/emp inputs/outputs ## #
        print("Auditing population/employment numbers...")
        # Build paths
        pop_path = os.path.join(self.exports['productions'],
                                self.production_generator.pop_fname)
        emp_path = os.path.join(self.exports['attractions'],
                                self.attraction_generator.emp_fname)

        # TODO: Add toggle to turn pop/emp comparator on/off

        # Build the comparators
        pop_comp = pop_emp_comparator.PopEmpComparator(
            **self.pop_emp_inputs['population'],
            output_csv=pop_path,
            data_type='population',
            base_year=str(base_year),
            verbose=self.verbose
        )
        emp_comp = pop_emp_comparator.PopEmpComparator(
            **self.pop_emp_inputs['employment'],
            output_csv=emp_path,
            data_type='employment',
            base_year=str(base_year),
            verbose=self.verbose
        )

        # Write comparisons to disk
        pop_comp.write_comparisons(self.exports['reports'], 'csv', True)
        emp_comp.write_comparisons(self.exports['reports'], 'csv', True)

        last_time = current_time
        current_time = time.time()
        elapsed_time = current_time - last_time
        print("Population/Employment auditing took: %.2f seconds" % elapsed_time)

        # ## Generate NHB Productions ## #
        print("Generating Non-Home Based Productions...")
        nhb_pm = nd.NhbProductionModel(
            import_home=self.imports['home'],
            export_home=self.exports['home'],
            model_name=self.model_name,
            msoa_conversion_path=self.msoa_zones_path,
            base_year=str(base_year),
            future_years=[str(x) for x in future_years],
            control_productions=True,
            control_fy_productions=self.ntem_control_future_years
        )
        nhb_productions = nhb_pm.run(
            recreate_productions=recreate_nhb_productions
        )

        last_time = current_time
        current_time = time.time()
        elapsed_time = current_time - last_time
        print("NHB Production generation took: %.2f seconds" % elapsed_time)

        # # ## ATTRACTION WEIGHT GENERATION ## #
        print("Generating attraction weights...")
        attraction_weights = du.convert_to_weights(
            attraction_dataframe,
            year_list
        )

        nhb_a_weights = du.convert_to_weights(
            nhb_att,
            year_list
        )

        print("Attraction weights generated!")
        last_time = current_time
        current_time = time.time()
        print("Attraction weight generation took: %.2f seconds" %
              (current_time - last_time))

        # To avoid errors lets make sure all columns have the same datatype
        production_trips.columns = production_trips.columns.astype(str)
        nhb_productions.columns = nhb_productions.columns.astype(str)

        attraction_dataframe.columns = attraction_dataframe.columns.astype(str)
        attraction_weights.columns = attraction_weights.columns.astype(str)
        nhb_a_weights.columns = nhb_a_weights.columns.astype(str)

        # ## ZONE TRANSLATION ## #
        model_zone_col = '%s_zone_id' % self.model_name
        if self.output_zone_system != self.input_zone_system:
            print("Need to translate zones.")
            print("Translating from: " + self.input_zone_system)
            print("Translating to: " + self.output_zone_system)

            pop_translation, emp_translation = self.get_translation_dfs()

            # Figure out which columns are the segmentation
            non_split_columns = list(production_trips.columns)
            non_split_columns = du.list_safe_remove(non_split_columns, year_list)
            converted_productions = self.zone_translator.run(
                production_trips,
                pop_translation,
                self.input_zone_system,
                self.output_zone_system,
                non_split_cols=non_split_columns
            )

            non_split_columns = list(nhb_productions.columns)
            non_split_columns = du.list_safe_remove(non_split_columns, year_list)
            converted_nhb_productions = self.zone_translator.run(nhb_productions, pop_translation,
                                                                 self.input_zone_system,
                                                                 self.output_zone_system,
                                                                 non_split_cols=non_split_columns)

            non_split_columns = list(attraction_dataframe.columns)
            non_split_columns = du.list_safe_remove(non_split_columns, year_list)
            converted_pure_attractions = self.zone_translator.run(attraction_dataframe,
                                                                  emp_translation,
                                                                  self.input_zone_system,
                                                                  self.output_zone_system,
                                                                  non_split_cols=non_split_columns)

            non_split_columns = list(nhb_att.columns)
            non_split_columns = du.list_safe_remove(non_split_columns, year_list)
            converted_nhb_att = self.zone_translator.run(nhb_att, emp_translation,
                                                         self.input_zone_system,
                                                         self.output_zone_system,
                                                         non_split_cols=non_split_columns)

            non_split_columns = list(attraction_weights.columns)
            non_split_columns = du.list_safe_remove(non_split_columns, year_list)
            converted_attractions = self.zone_translator.run(attraction_weights, emp_translation,
                                                             self.input_zone_system,
                                                             self.output_zone_system,
                                                             non_split_cols=non_split_columns)

            non_split_columns = list(nhb_a_weights.columns)
            non_split_columns = du.list_safe_remove(non_split_columns, year_list)
            converted_nhb_attractions = self.zone_translator.run(nhb_a_weights, emp_translation,
                                                                 self.input_zone_system,
                                                                 self.output_zone_system,
                                                                 non_split_cols=non_split_columns)

            print("Zone translation completed!")
            last_time = current_time
            current_time = time.time()
            print("Zone translation took: %.2f seconds" %
                  (current_time - last_time))
        else:
            converted_productions = production_trips.copy()
            converted_nhb_productions = nhb_productions.copy()

            converted_attractions = attraction_weights.copy()
            converted_pure_attractions = attraction_dataframe.copy()

            converted_nhb_att = nhb_att.copy()
            converted_nhb_attractions = nhb_a_weights.copy()

        # Write Translated p/a to file
        fname = consts.PRODS_FNAME % (self.output_zone_system, 'hb')
        converted_productions.to_csv(
            os.path.join(self.exports['productions'], fname),
            index=False
        )

        fname = consts.PRODS_FNAME % (self.output_zone_system, 'nhb')
        converted_nhb_productions.to_csv(
            os.path.join(self.exports['productions'], fname),
            index=False
        )

        fname = consts.ATTRS_FNAME % (self.output_zone_system, 'hb')
        converted_pure_attractions.to_csv(
            os.path.join(self.exports['attractions'], fname),
            index=False
        )

        fname = consts.ATTRS_FNAME % (self.output_zone_system, 'nhb')
        converted_nhb_att.to_csv(
            os.path.join(self.exports['attractions'], fname),
            index=False
        )

        if apply_growth_criteria:
            # Apply the growth criteria using the post-ME P/A vectors
            # (normal and exceptional zones)
            pa_dfs = self._handle_growth_criteria(
                synth_productions=converted_productions,
                synth_attractions=converted_pure_attractions,
                base_year=str(base_year),
                future_years=[str(x) for x in future_years],
                integrate_dlog=self.integrate_dlog
            )
            converted_productions, converted_pure_attractions = pa_dfs

        # Convert the new attractions to weights
        converted_attractions = du.convert_to_weights(
            converted_pure_attractions,
            year_list
        )

        # Write grown productions and attractions to file
        # Save as exceptional - e.g. "exc_productions"
        fname = consts.PRODS_FNAME % (self.output_zone_system, 'hb')
        fname = fname.replace("_productions", "_exc_productions")
        converted_productions.to_csv(
            os.path.join(self.exports['productions'], fname),
            index=False
        )

        fname = consts.ATTRS_FNAME % (self.output_zone_system, 'hb')
        fname = fname.replace("_attractions", "_exc_attractions")
        converted_pure_attractions.to_csv(
            os.path.join(self.exports['attractions'], fname),
            index=False
        )

        # TODO: Move conversion to attraction weights down here

        # ## DISTRIBUTION ## #
        if distribution_method == "furness":
            print("Generating HB distributions...")
            furness.distribute_pa(
                productions=converted_productions,
                attraction_weights=converted_attractions,
                trip_origin='hb',
                years_needed=year_list,
                p_needed=purposes_needed,
                m_needed=modes_needed,
                soc_needed=soc_needed,
                ns_needed=ns_needed,
                ca_needed=car_availabilities_needed,
                zone_col=model_zone_col,
                seed_dist_dir=self.imports['seed_dists'],
                dist_out=self.exports['pa_24'],
                audit_out=self.exports['dist_audits'],
                echo=echo_distribution
            )

            print("Generating NHB distributions...")
            furness.distribute_pa(
                productions=converted_nhb_productions,
                attraction_weights=converted_nhb_attractions,
                trip_origin='nhb',
                years_needed=year_list,
                p_needed=nhb_purposes_needed,
                m_needed=modes_needed,
                soc_needed=soc_needed,
                ns_needed=ns_needed,
                ca_needed=car_availabilities_needed,
                zone_col=model_zone_col,
                seed_dist_dir=self.imports['seed_dists'],
                dist_out=self.exports['pa_24'],
                audit_out=self.exports['dist_audits'],
                echo=echo_distribution
            )

            last_time = current_time
            current_time = time.time()
            print("Distribution generation took: %.2f seconds" %
                  (current_time - last_time))
        else:
            raise ValueError("'%s' is not a valid distribution method!" %
                             (str(distribution_method)))

        # ## SECTOR TOTALS ## #
        zone_system_file = os.path.join(self.imports['zoning'],
                                        self.output_zone_system + '.csv')
        sector_grouping_file = os.path.join(self.imports['zoning'],
                                            "tfn_level_one_sectors_norms_grouping.csv")

        sector_totals = self.sector_reporter.calculate_sector_totals(
                converted_productions,
                grouping_metric_columns=year_list,
                sector_grouping_file=sector_grouping_file,
                zone_col=model_zone_col
                )

        pm_sector_total_dictionary = {}

        for purpose in purposes_needed:
            # TODO: Update sector reporter.
            #  Sector totals don't currently allow per purpose reporting

            pm_productions = converted_productions.copy()

            pm_sector_totals = self.sector_reporter.calculate_sector_totals(
                pm_productions,
                grouping_metric_columns=year_list,
                sector_grouping_file=sector_grouping_file,
                zone_col=model_zone_col
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

                fname = self.output_zone_system + "_sector_totals.csv"
                sector_totals.to_csv(
                    os.path.join(self.exports['sectors'], fname),
                    index=False
                )

                for key, sector_total in pm_sector_total_dictionary.items():
                    print("Saving sector total: " + key)
                    fname = "sector_total_%s.csv" % key
                    sector_total.to_csv(
                        os.path.join(self.exports['sectors'], fname),
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

    def _handle_growth_criteria(self,
                                synth_productions: pd.DataFrame,
                                synth_attractions: pd.DataFrame,
                                base_year: str,
                                future_years: List[str],
                                integrate_dlog: bool
                                ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Init
        all_years = [base_year] + future_years

        # Load the exceptional zone definitions from production/attraction
        # generation
        print("Loading Exceptional Growth Datafiles")
        # Only handle exceptional zones if the d-log is being integrated
        if integrate_dlog:
            print("Fetching Exceptional Zones")
            p_ez, a_ez = eg.load_exceptional_zones(
                productions_export=self.exports["productions"],
                attractions_export=self.exports["attractions"]
            )
        else:
            print("Ignoring Exceptional Zones")
            p_ez, a_ez = (None, None)
        # Reload aggregated population and employment data to calculate
        # sector level trip rates
        fname = consts.POP_FNAME % self.input_zone_system
        grown_pop_path = os.path.join(self.exports["productions"], fname)

        fname = consts.EMP_FNAME % self.input_zone_system
        grown_emp_path = os.path.join(self.exports["attractions"], fname)

        # Us the matrices in seed distribution as the base observed
        observed_pa_path = self.imports["seed_dists"]

        # ## APPLY GROWTH CRITERIA ## #

        # TODO: Need norms_to_tfn sector lookups.
        #  Should these be pop/emp weighted too?
        sector_system = "tfn_sectors"
        model_zone_to_sector_path = os.path.join(
            self.imports["zone_translation"],
            "{}_to_{}.csv".format(self.output_zone_system, sector_system)
        )
        from_zone_column = "{}_zone_id".format(self.output_zone_system)
        to_sector_column = "{}_zone_id".format(sector_system)

        # Load sector mapping for calculating the exceptional zone trip rates
        sector_lookup = pd.read_csv(model_zone_to_sector_path).rename(
            columns={
                from_zone_column: "model_zone_id",
                to_sector_column: "grouping_id"
            })
        sector_lookup = sector_lookup.set_index("model_zone_id")["grouping_id"]

        # Zone translation arguments for population/employment and
        # exceptional zone translation - reduces number of arguments required
        pop_translation, emp_translation = self.get_translation_dfs()

        # TODO: How to deal with NHB productions/attractions??
        #  Run this after the P/A models, then base the NHB off this?
        # Apply growth criteria to "normal" and "exceptional" zones
        productions, attractions = eg.growth_criteria(
            synth_productions=synth_productions,
            synth_attractions=synth_attractions,
            observed_pa_path=observed_pa_path,
            prod_exceptional_zones=p_ez,
            attr_exceptional_zones=a_ez,
            population_path=grown_pop_path,
            employment_path=grown_emp_path,
            model_name=self.model_name,
            base_year=base_year,
            future_years=future_years,
            zone_translator=self.zone_translator,
            zt_from_zone=self.input_zone_system,
            zt_pop_df=pop_translation,
            zt_emp_df=emp_translation,
            trip_rate_sectors=sector_lookup,
            soc_weights_path=self.imports['soc_weights'],
            prod_audits=self.exports["productions"],
            attr_audits=self.exports["attractions"]
        )

        return productions, attractions

    def get_translation_dfs(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Returns the dataframes for translating from input_zone_system to
        output_zone_system.

        A different translation dataframe is returned for translating
        population data and employment data.

        Returns
        -------
        pop_translation:
            A population weighted translation dataframe to convert from
            input_zone_system to output_zone_system

        emp_translation:
            A employment weighted translation dataframe to convert from
            input_zone_system to output_zone_system
        """
        # Init
        fname_args = (self.input_zone_system, self.output_zone_system)

        # Read in pop translation
        fname = consts.POP_TRANSLATION_FNAME % fname_args
        path = os.path.join(self.imports['zoning'], fname)
        pop_translation = pd.read_csv(path)

        # Read in emp translation
        fname = consts.EMP_TRANSLATION_FNAME % fname_args
        path = os.path.join(self.imports['zoning'], fname)
        emp_translation = pd.read_csv(path)

        return pop_translation, emp_translation

    def _get_time_splits_from_p_vector(self,
                                       trip_origin: str,
                                       years_needed: List[int] = consts.ALL_YEARS,
                                       ignore_cache: bool = False,
                                       ) -> pd.DataFrame:
        # TODO: cache!
        # TODO: check trip_origin is valid
        # If the file already exists, just return that
        file_type = '%s_tp_splits' % trip_origin
        fname = consts.PRODS_FNAME % (self.output_zone_system, file_type)
        output_path = os.path.join(self.exports['productions'], fname)
        if not ignore_cache and os.path.exists(output_path):
            return pd.read_csv(output_path)

        # Init
        yr_cols = [str(x) for x in years_needed]
        base_zone_col = "%s_zone_id"
        input_zone_col = base_zone_col % self.input_zone_system.lower()
        output_zone_col = base_zone_col % self.output_zone_system

        # Figure out the segmentation to keep
        seg_level = du.validate_seg_level('tfn')
        seg_cols = du.get_seg_level_cols(seg_level,
                                         keep_ca=self.is_ca_needed)
        non_split_cols = [input_zone_col] + seg_cols

        # Read in raw production to get time period splits from
        file_type = 'raw_%s' % trip_origin
        fname = consts.PRODS_FNAME % (self.input_zone_system, file_type)
        raw_prods_path = os.path.join(self.exports['productions'], fname)
        productions = pd.read_csv(raw_prods_path)

        # Filter down to just the segmentation we need
        group_cols = non_split_cols.copy()
        index_cols = non_split_cols + yr_cols

        productions = productions.reindex(columns=index_cols)
        productions = productions.groupby(group_cols).sum().reset_index()

        # Translate to the correct zoning system
        pop_translation, _ = self.get_translation_dfs()

        group_cols = list(productions.columns)
        group_cols = du.list_safe_remove(group_cols, yr_cols)
        productions = self.zone_translator.run(
            productions,
            pop_translation,
            self.input_zone_system,
            self.output_zone_system,
            non_split_cols=group_cols
        )

        # Extract the time splits
        non_split_cols = [output_zone_col] + seg_cols
        tp_splits = pm.get_production_time_split(productions,
                                                 non_split_cols=non_split_cols,
                                                 data_cols=yr_cols)

        # Write the time period splits to disk
        tp_splits.to_csv(output_path, index=False)

        return tp_splits

    def pa_to_od(self,
                 years_needed: List[int] = consts.ALL_YEARS,
                 m_needed: List[int] = consts.MODES_NEEDED,
                 p_needed: List[int] = consts.ALL_P,
                 soc_needed: List[int] = consts.SOC_NEEDED,
                 ns_needed: List[int] = consts.NS_NEEDED,
                 ca_needed: List[int] = consts.CA_NEEDED,
                 use_bespoke_pa: bool= True,
                 overwrite_hb_tp_pa: bool = True,
                 overwrite_hb_tp_od: bool = True,
                 echo: bool = True
                 ) -> None:
        """
        Converts home based PA matrices into time periods split PA matrices,
        then OD matrices (to_home, from_home, and full OD).

        NHB tp split PA matrices are simply copied and renamed as they are
        already in OD format

        Parameters
        ----------
        years_needed:
            The years of PA matrices to convert to OD

        m_needed:
            The modes of PA matrices to convert to OD

        p_needed:
            The purposes of PA matrices to convert to OD

        soc_needed:
            The skill levels of PA matrices to convert to OD

        ns_needed:
            The income levels of PA matrices to convert to OD

        ca_needed:
            The the car availability of PA matrices to convert to OD

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
        _input_checks(m_needed=m_needed)
        base_zone_col = "%s_zone_id"
        output_zone_col = base_zone_col % self.output_zone_system

        if not self.is_ca_needed:
            ca_needed = None

        # Split into hb and nhb purposes
        hb_p_needed = list()
        nhb_p_needed = list()
        for p in p_needed:
            if p in consts.ALL_HB_P:
                hb_p_needed.append(p)
            elif p in consts.ALL_NHB_P:
                nhb_p_needed.append(p)
            else:
                raise ValueError(
                    "%s is not a valid HB or NHB purpose" % str(p)
                )

        # TODO: Add time print outs
        hb_nhb_iterator = zip(['hb', 'nhb'], [hb_p_needed, nhb_p_needed])
        for trip_origin, to_p_needed in hb_nhb_iterator:
            print("Running conversions for %s trips..." % trip_origin)

            # TODO: Check if tp pa matrices exist first
            if overwrite_hb_tp_pa:
                tp_splits = self._get_time_splits_from_p_vector(trip_origin, ignore_cache=True)

                print("Converting %s 24hr PA to time period split PA..." % trip_origin)
                pa_import = 'pa_24_bespoke' if use_bespoke_pa else 'pa_24'
                pa2od.efs_build_tp_pa(
                    pa_import=self.exports[pa_import],
                    pa_export=self.exports['pa'],
                    tp_splits=tp_splits,
                    model_zone_col=output_zone_col,
                    years_needed=years_needed,
                    p_needed=to_p_needed,
                    m_needed=m_needed,
                    soc_needed=soc_needed,
                    ns_needed=ns_needed,
                    ca_needed=ca_needed
                )
                print('HB time period split PA matrices compiled!\n')

        # TODO: Check if od matrices exist first
        if overwrite_hb_tp_od:
            print('Converting time period split PA to OD...')
            pa2od.efs_build_od(
                pa_import=self.exports['pa'],
                od_export=self.exports['od'],
                model_name=self.model_name,
                p_needed=hb_p_needed,
                m_needed=m_needed,
                soc_needed=soc_needed,
                ns_needed=ns_needed,
                ca_needed=ca_needed,
                years_needed=years_needed,
                phi_type='fhp_tp',
                aggregate_to_wday=True,
                echo=echo
            )

            # Copy over NHB matrices as they are already in NHB format
            mat_p.copy_nhb_matrices(
                import_dir=self.exports['pa'],
                export_dir=self.exports['od'],
                replace_pa_with_od=True,
            )

            print('HB OD matrices compiled!\n')
            # TODO: Create 24hr OD for HB

    def pre_me_compile_od_matrices(self,
                                   year: int = consts.BASE_YEAR,
                                   hb_p_needed: List[int] = consts.PURPOSES_NEEDED,
                                   nhb_p_needed: List[int] = consts.NHB_PURPOSES_NEEDED,
                                   m_needed: List[int] = consts.MODES_NEEDED,
                                   tp_needed: List[int] = consts.TIME_PERIODS,
                                   overwrite_aggregated_od: bool = True,
                                   overwrite_compiled_od: bool = True,
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
        _input_checks(m_needed=m_needed)

        if self.is_ca_needed:
            ca_needed = consts.CA_NEEDED
        else:
            ca_needed = [None]
            
        if overwrite_aggregated_od:
            for matrix_format in ['od_from', 'od_to']:
                mat_p.aggregate_matrices(
                    import_dir=self.exports['od'],
                    export_dir=self.exports['aggregated_od'],
                    trip_origin='hb',
                    matrix_format=matrix_format,
                    years_needed=[year],
                    p_needed=hb_p_needed,
                    m_needed=m_needed,
                    ca_needed=ca_needed,
                    tp_needed=tp_needed
                )
            mat_p.aggregate_matrices(
                import_dir=self.exports['od'],
                export_dir=self.exports['aggregated_od'],
                trip_origin='nhb',
                matrix_format='od',
                years_needed=[year],
                p_needed=nhb_p_needed,
                m_needed=m_needed,
                ca_needed=ca_needed,
                tp_needed=tp_needed
            )

        if overwrite_compiled_od:
            # Build the compile params for this model
            if self.model_name == 'noham':
                compile_params_path = mat_p.build_compile_params(
                    import_dir=self.exports['aggregated_od'],
                    export_dir=self.params['compile'],
                    matrix_format='od',
                    years_needed=[year],
                    m_needed=m_needed,
                    ca_needed=ca_needed,
                    tp_needed=tp_needed,
                )
            else:
                raise ValueError(
                    "Not sure how to compile matrices for model %s"
                    % self.model_name
                )

            mat_p.compile_matrices(
                mat_import=self.exports['aggregated_od'],
                mat_export=self.exports['compiled_od'],
                compile_params_path=compile_params_path,
                build_factor_pickle=True,
                factor_pickle_path=self.params['compile']
            )

            # Need to convert into hourly average PCU for noham
            if self.model_name == 'noham':
                vo.people_vehicle_conversion(
                    mat_import=self.exports['compiled_od'],
                    mat_export=self.exports['compiled_od_pcu'],
                    import_folder=self.imports['home'],
                    mode=m_needed[0],
                    method='to_vehicles',
                    out_format='wide',
                    hourly_average=True
                )


    def generate_post_me_tour_proportions(self,
                                          model_name: str,
                                          year: int = consts.BASE_YEAR,
                                          m_needed: List[int] = consts.MODES_NEEDED,
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
        model_name:
            The name of the model this is being run for.

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
        _input_checks(m_needed=m_needed)

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

        # TODO: Fix OD2PA to use norms_2015/norms for zone names

        if overwrite_decompiled_od:
            print("Decompiling OD Matrices into purposes...")
            need_convert = od2pa.need_to_convert_to_efs_matrices(
                model_import=self.exports['post_me']['model_output'],
                od_import=self.exports['post_me']['compiled_od']
            )
            if need_convert:
                od2pa.convert_to_efs_matrices(
                    import_path=self.exports['post_me']['model_output'],
                    export_path=self.exports['post_me']['compiled_od'],
                    matrix_format='od',
                    year=year,
                    user_class=True,
                    to_wide=True,
                    wide_col_name='%s_zone_id' % model_name,
                    from_pcu=from_pcu,
                    vehicle_occupancy_import=self.imports['home']
                )

            # TODO: Stop the filename being hardcoded after integration with TMS
            decompile_factors_path = os.path.join(
                self.params['compile'],
                'od_compilation_factors.pickle'
            )
            od2pa.decompile_od(
                od_import=self.exports['post_me']['compiled_od'],
                od_export=self.exports['post_me']['od'],
                decompile_factors_path=decompile_factors_path,
                year=year
            )

        if overwrite_tour_proportions:
            print("Converting OD matrices to PA and generating tour "
                  "proportions...")
            mat_p.generate_tour_proportions(
                od_import=self.exports['post_me']['od'],
                zone_translate_dir=self.imports['zone_translation'],
                pa_export=self.exports['post_me']['pa'],
                tour_proportions_export=self.params['tours'],
                year=year,
                ca_needed=ca_needed
            )

    def compile_future_year_od_matrices(self,
                                        years_needed: List[int] = consts.FUTURE_YEARS,
                                        hb_p_needed: List[int] = consts.ALL_HB_P,
                                        m_needed: List[int] = consts.MODES_NEEDED,
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
        _input_checks(m_needed=m_needed)

        if self.model_name == 'norms' or self.model_name == 'norms_2015':
            ca_needed = consts.CA_NEEDED
        elif self.model_name == 'noham':
            ca_needed = [None]
        else:
            raise ValueError("Got an unexpected model name. Got %s, expected "
                             "either 'norms', 'norms_2015' or 'noham'."
                             % str(self.model_name))

        if overwrite_aggregated_pa:
            mat_p.aggregate_matrices(
                import_dir=self.exports['pa_24'],
                export_dir=self.exports['aggregated_pa_24'],
                trip_origin='hb',
                matrix_format='pa',
                years_needed=years_needed,
                p_needed=hb_p_needed,
                ca_needed=ca_needed,
                m_needed=m_needed
            )

        if overwrite_future_year_od:
            pa2od.build_od_from_tour_proportions(
                pa_import=self.exports['aggregated_pa_24'],
                od_export=self.exports['post_me']['od'],
                tour_proportions_dir=self.params['tours'],
                zone_translate_dir=self.imports['zone_translation'],
                ca_needed=ca_needed
            )

        # TODO: Compile to OD/PA when we know the correct format

    def generate_output_paths(self, base_year: str) -> Tuple[Dict[str, str],
                                                             Dict[str, str],
                                                             Dict[str, str]]:
        """
        Returns imports, exports and params dictionaries

        Calls du.build_io_paths() with class attributes.

        Parameters
        ----------
        base_year:
            The base year of the model being run. This is used to build the
            base year file paths

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
        return du.build_io_paths(self.import_location,
                                 self.output_location,
                                 base_year,
                                 self.model_name,
                                 self.iter_name,
                                 self.scenario_name,
                                 self.__version__,
                                 self._out_dir)


def _input_checks(iter_num: int = None,
                  m_needed: List[int] = None,
                  constraint_required: Dict[str, bool] = None
                  ) -> None:
    """
    Checks that any arguments given are OK. Will raise an error
    for any given input that is not correct.
    """
    if iter_num is not None and iter_num == 0:
        Warning("iter_num is set to 0. This is should only be the case"
                "during testing.")

    if m_needed is not None and len(m_needed) > 1:
        raise du.ExternalForecastSystemError(
            "Was given more than one mode. EFS cannot run using more than "
            "one mode at a time due to different zoning systems for NoHAM "
            "and NoRMS etc.")

    # Make sure all the expected keys exist
    if constraint_required is not None:
        expected_keys = consts.CONSTRAINT_REQUIRED_DEFAULT.keys()
        for key in expected_keys:
            if key not in constraint_required:
                raise du.ExternalForecastSystemError(
                    "Missing '%s' key in constraint_required. Expected to "
                    "find all of the following keys:\n%s"
                    % (str(key), str(expected_keys))
                )


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
                     constraint_required: List[bool],
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
        "Constraints Used On: " + str(constraint_required),
        "Constraint Source: " + constraint_source
    ]
    with open(output_path, 'w') as out:
        out.write('\n'.join(out_lines))
