# -*- coding: utf-8 -*-
"""
Created on: Mon Nov 25 09:50:07 2019
Updated on: Fri Sep 18 15:03:24 2020

Original author: Sneezy
Last Update Made by: Ben Taylor

File purpose:
Home of the NorMITs External Forecast System
"""
# Built-ins
import os
import time
import operator

from typing import List
from typing import Dict
from typing import Tuple
from typing import Union

# External libs
import pandas as pd

# self imports
import normits_demand as nd
from normits_demand import constants as consts
from normits_demand import efs_constants as efs_consts
from normits_demand.models import efs_production_model as pm

from normits_demand.concurrency import multiprocessing

from normits_demand.matrices import decompilation
from normits_demand.matrices import pa_to_od as pa2od
from normits_demand.matrices import matrix_processing as mat_p

from normits_demand.distribution import furness
from normits_demand.distribution import external_growth as ext_growth

from normits_demand.reports import pop_emp_comparator

from normits_demand.utils import file_ops
from normits_demand.utils import general as du
from normits_demand.utils import vehicle_occupancy as vo
from normits_demand.utils import exceptional_growth as eg
from normits_demand.utils import sector_reporter_v2 as sr_v2


# TODO: Output a run log instead of printing everything to the terminal.
# TODO: On error, output a simple error report

# BACKLOG: Implement a logger to log EFS run progress
#  labels: QoL Updates

# TODO(MB) Move to forecasting sub-package

class ExternalForecastSystem:
    # ## Class Constants ## #
    __version__ = nd.__version__
    out_dir = "NorMITs Demand"

    # defines all non-year columns
    column_dictionary = efs_consts.EFS_COLUMN_DICTIONARY

    def __init__(self,
                 model_name: str,
                 iter_num: Union[int, str],
                 scenario_name: str,

                 integrate_dlog: bool = False,
                 run_pop_emp_comparison: bool = True,
                 apply_wfh_adjustments: bool = True,

                 dlog_pop_path: str = None,
                 dlog_emp_path: str = None,

                 import_home: str = "I:/",
                 export_home: str = "E:/",

                 land_use_drive: str = "Y:/",
                 by_land_use_iteration: str = 'iter3b',
                 fy_land_use_iteration: str = 'iter3c',
                 modes_needed: List[int] = None,
                 verbose: bool = True,
                 ):
        # TODO: Write EFS constructor docs
        # Initialise the timer
        begin_time = time.time()
        current_time = begin_time
        print("Initiating External Forecast System...")

        if modes_needed is None:
            modes_needed = efs_consts.MODEL_MODES[model_name]

        # Initialise
        du.validate_model_name_and_mode(model_name, modes_needed)
        self.model_name = du.validate_model_name(model_name)
        self.iter_name = du.create_iter_name(iter_num)
        self.scenario_name = du.validate_scenario_name(scenario_name)
        self.integrate_dlog = integrate_dlog
        self.run_pop_emp_comparison = run_pop_emp_comparison
        self.apply_wfh_adjustments = apply_wfh_adjustments
        self.import_location = import_home
        self.output_location = export_home
        self.verbose = verbose

        self.by_land_use_iteration = by_land_use_iteration
        self.fy_land_use_iteration = fy_land_use_iteration
        self.land_use_drive = land_use_drive

        # TODO: Write function to determine if CA is needed for model_names
        # TODO: Write function to determine if from/to PCU is needed for model_names
        self.is_ca_needed = True
        self.uses_pcu = False
        if self.model_name == 'noham':
            self.is_ca_needed = False
            self.uses_pcu = True
        self.ca_needed = efs_consts.CA_NEEDED if self.is_ca_needed else None

        self.input_zone_system = "MSOA"
        self.output_zone_system = self.model_name

        # We never control future years, not even in NTEM!
        self.ntem_control_future_years = False

        # Setup up import/export paths
        path_dicts = self._generate_paths(efs_consts.BASE_YEAR_STR)
        self.imports, self.exports, self.params = path_dicts
        self._setup_scenario_paths()
        self._build_pop_emp_paths()
        self._read_in_default_inputs()
        self._set_up_dlog(dlog_pop_path, dlog_emp_path)

        # Load in the internal and external zones
        self._load_ie_zonal_info()

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
        zt_dict = self.imports['zone_translation']
        self.msoa_lookup = du.safe_read_csv(zt_dict['msoa_str_int'])

        file_path = os.path.join(zt_dict['one_to_one'], 'lad_to_msoa.csv')
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

    def _load_ie_zonal_info(self):
        """
        Populates self.model_internal_zones and self.model_external_zones
        """
        # Init
        zone_col = '%s_zone_id' % self.model_name

        int_fname = consts.INTERNAL_AREA % self.model_name
        int_path = os.path.join(self.imports['model_schema'], int_fname)
        self.model_internal_zones = pd.read_csv(int_path)[zone_col].tolist()

        ext_fname = consts.EXTERNAL_AREA % self.model_name
        ext_path = os.path.join(self.imports['model_schema'], ext_fname)
        self.model_external_zones = pd.read_csv(ext_path)[zone_col].tolist()

    def _build_pop_emp_paths(self):
        # Init
        zone_lookups = efs_consts.TFN_MSOA_SECTOR_LOOKUPS

        # Build the pop paths
        pop_paths = {
            "by_pop_path": self.imports["pop_by"],
            "by_emp_path": self.imports["emp_by"],
            "growth_csv": self.pop_growth_path,
            "constraint_csv": self.pop_constraint_path,
            "sector_grouping_file": os.path.join(self.imports['zone_translation']['weighted'],
                                                 zone_lookups["population"])
        }

        # Build the emp paths
        emp_paths = {
            "by_pop_path": self.imports["pop_by"],
            "by_emp_path": self.imports["emp_by"],
            "growth_csv": self.emp_growth_path,
            "constraint_csv": self.emp_constraint_path,
            "sector_grouping_file": os.path.join(self.imports['zone_translation']['weighted'],
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
            future_years: List[int] = efs_consts.FUTURE_YEARS,
            hb_purposes_needed: List[int] = efs_consts.HB_PURPOSES_NEEDED,
            nhb_purposes_needed: List[int] = efs_consts.NHB_PURPOSES_NEEDED,
            modes_needed: List[int] = efs_consts.MODES_NEEDED,
            soc_needed: List[int] = efs_consts.SOC_NEEDED,
            ns_needed: List[int] = efs_consts.NS_NEEDED,
            car_availabilities_needed: List[int] = efs_consts.CA_NEEDED,
            constraint_required: Dict[str, bool] = efs_consts.CONSTRAINT_REQUIRED_DEFAULT,
            recreate_productions: bool = True,
            recreate_attractions: bool = True,
            recreate_nhb_productions: bool = True,
            combine_internal_external: bool = False,
            outputting_files: bool = True,
            output_location: str = None,
            echo_distribution: bool = True,
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

        hb_purposes_needed:
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

        outputting_files:
            Whether files are being output.
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
        # TODO (BT): Update EFS.run() docs
        # Init
        if output_location is None:
            output_location = self.output_location

        if self.model_name == 'noham':
            car_availabilities_needed = None

        # Set up timing
        begin_time = time.time()
        current_time = begin_time

        year_list = [str(x) for x in [base_year] + future_years]

        # Validate inputs
        _input_checks(m_needed=modes_needed,
                      constraint_required=constraint_required)

        # ## PREPARE OUTPUTS ## #
        print("Initialising outputs...")
        write_input_info(
            os.path.join(self.exports['home'], "input_parameters.txt"),
            self.__version__,
            self.by_land_use_iteration,
            self.fy_land_use_iteration,
            base_year,
            future_years,
            self.output_zone_system,
            self.imports['decomp_post_me'],
            hb_purposes_needed + nhb_purposes_needed,
            modes_needed,
            soc_needed,
            ns_needed,
            car_availabilities_needed,
            self.integrate_dlog,
            constraint_required,
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
        p_vector = self.production_generator.run(
            base_year=str(base_year),
            future_years=[str(x) for x in future_years],
            by_pop_import_path=self.imports['pop_by'],
            fy_pop_import_dir=self.imports['land_use_fy_dir'],
            pop_constraint=pop_constraint,
            import_home=self.imports['home'],
            export_home=self.exports['home'],
            msoa_lookup_path=self.imports['zone_translation']['msoa_str_int'],
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
        a_vector, nhb_a_vector = self.attraction_generator.run(
            out_path=self.exports['attractions'],
            base_year=str(base_year),
            future_years=[str(x) for x in future_years],
            by_emp_import_path=self.imports['emp_by'],
            fy_emp_import_dir=self.imports['land_use_fy_dir'],
            emp_constraint=emp_constraint,
            import_home=self.imports['home'],
            export_home=self.exports['home'],
            msoa_lookup_path=self.imports['zone_translation']['msoa_str_int'],
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

        if self.run_pop_emp_comparison:
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
            msoa_conversion_path=self.imports['zone_translation']['msoa_str_int'],
            base_year=str(base_year),
            future_years=[str(x) for x in future_years],
            control_productions=True,
            control_fy_productions=self.ntem_control_future_years
        )
        nhb_p_vector = nhb_pm.run(
            recreate_productions=recreate_nhb_productions
        )

        last_time = current_time
        current_time = time.time()
        elapsed_time = current_time - last_time
        print("NHB Production generation took: %.2f seconds" % elapsed_time)

        # To avoid errors lets make sure all columns have the same datatype
        p_vector.columns = p_vector.columns.astype(str)
        nhb_p_vector.columns = nhb_p_vector.columns.astype(str)

        a_vector.columns = a_vector.columns.astype(str)
        nhb_a_vector.columns = nhb_a_vector.columns.astype(str)

        # ## ZONE TRANSLATION ## #
        model_zone_col = '%s_zone_id' % self.model_name
        if self.output_zone_system != self.input_zone_system:
            print("Need to translate zones.")
            print("Translating from: " + self.input_zone_system)
            print("Translating to: " + self.output_zone_system)

            pop_translation, emp_translation = self.get_translation_dfs()

            # Figure out which columns are the segmentation
            non_split_columns = list(p_vector.columns)
            non_split_columns = du.list_safe_remove(non_split_columns, year_list)
            model_p_vector = self.zone_translator.run(
                p_vector,
                pop_translation,
                self.input_zone_system,
                self.output_zone_system,
                non_split_cols=non_split_columns
            )

            non_split_columns = list(nhb_p_vector.columns)
            non_split_columns = du.list_safe_remove(non_split_columns, year_list)
            model_nhb_p_vector = self.zone_translator.run(
                nhb_p_vector,
                pop_translation,
                self.input_zone_system,
                self.output_zone_system,
                non_split_cols=non_split_columns
            )

            non_split_columns = list(a_vector.columns)
            non_split_columns = du.list_safe_remove(non_split_columns, year_list)
            model_a_vector = self.zone_translator.run(
                a_vector,
                emp_translation,
                self.input_zone_system,
                self.output_zone_system,
                non_split_cols=non_split_columns
            )

            non_split_columns = list(nhb_a_vector.columns)
            non_split_columns = du.list_safe_remove(non_split_columns, year_list)
            model_nhb_a_vector = self.zone_translator.run(
                nhb_a_vector,
                emp_translation,
                self.input_zone_system,
                self.output_zone_system,
                non_split_cols=non_split_columns
            )

            print("Zone translation completed!")
            last_time = current_time
            current_time = time.time()
            print("Zone translation took: %.2f seconds" %
                  (current_time - last_time))
        else:
            model_p_vector = p_vector.copy()
            model_nhb_p_vector = nhb_p_vector.copy()

            model_a_vector = a_vector.copy()
            model_nhb_a_vector = nhb_a_vector.copy()

        # ## WRITE TRANSLATED VECTORS TO DISK ## #
        fname = efs_consts.PRODS_FNAME % (self.output_zone_system, 'hb')
        out_path = os.path.join(self.exports['productions'], fname)
        model_p_vector.to_csv(out_path, index=False)

        fname = efs_consts.PRODS_FNAME % (self.output_zone_system, 'nhb')
        out_path = os.path.join(self.exports['productions'], fname)
        model_nhb_p_vector.to_csv(out_path, index=False)

        fname = efs_consts.ATTRS_FNAME % (self.output_zone_system, 'hb')
        out_path = os.path.join(self.exports['attractions'], fname)
        model_a_vector.to_csv(out_path, index=False)

        fname = efs_consts.ATTRS_FNAME % (self.output_zone_system, 'nhb')
        out_path = os.path.join(self.exports['attractions'], fname)
        model_nhb_a_vector.to_csv(out_path, index=False)

        # Save a copy of the vectors to deal with int/ext trips later
        pre_eg_model_p_vector = model_p_vector.copy()
        pre_eg_model_nhb_p_vector = model_nhb_p_vector.copy()

        # Apply the growth criteria using the post-ME P/A vectors
        # (normal and exceptional zones)

        # APPLY TO INTERNAL ONLY
        dtype = {model_zone_col: int}
        internal_zones = pd.read_csv(self.imports['internal_zones'], dtype=dtype).squeeze().tolist()

        print("Applying growth criteria...")
        vectors = self._handle_growth_criteria(
            synth_productions=model_p_vector,
            synth_nhb_productions=model_nhb_p_vector,
            synth_attractions=model_a_vector,
            synth_nhb_attractions=model_nhb_a_vector,
            base_year=str(base_year),
            future_years=[str(x) for x in future_years],
            integrate_dlog=self.integrate_dlog,
            internal_zones=internal_zones,
            external_zones=None,
        )
        model_p_vector, model_nhb_p_vector, model_a_vector, model_nhb_a_vector = vectors

        # Write grown productions and attractions to file
        fname = efs_consts.PRODS_FNAME % (self.output_zone_system, 'hb_exc')
        out_path = os.path.join(self.exports['productions'], fname)
        model_p_vector.to_csv(out_path, index=False)

        fname = efs_consts.PRODS_FNAME % (self.output_zone_system, 'nhb_exc')
        out_path = os.path.join(self.exports['productions'], fname)
        model_nhb_p_vector.to_csv(out_path, index=False)

        fname = efs_consts.ATTRS_FNAME % (self.output_zone_system, 'hb_exc')
        out_path = os.path.join(self.exports['attractions'], fname)
        model_a_vector.to_csv(out_path, index=False)

        fname = efs_consts.ATTRS_FNAME % (self.output_zone_system, 'nhb_exc')
        out_path = os.path.join(self.exports['attractions'], fname)
        model_nhb_a_vector.to_csv(out_path, index=False)

        # ## DISTRIBUTE THE INTERNAL AND EXTERNAL DEMAND ## #
        # Create the temporary output folders
        dist_out = self.exports['pa_24']
        int_dir = os.path.join(dist_out, 'internal')
        ext_dir = os.path.join(dist_out, 'external')

        for path in [int_dir, ext_dir]:
            du.create_folder(path, verbose=False)

        # Distribute the internal trips, write to disk
        self._distribute_internal_demand(
            p_vector=model_p_vector,
            nhb_p_vector=model_nhb_p_vector,
            a_vector=model_a_vector,
            nhb_a_vector=model_nhb_a_vector,
            years_needed=year_list,
            hb_p_needed=hb_purposes_needed,
            nhb_p_needed=nhb_purposes_needed,
            m_needed=modes_needed,
            soc_needed=soc_needed,
            ns_needed=ns_needed,
            ca_needed=car_availabilities_needed,
            zone_col=model_zone_col,
            internal_zones_path=self.imports['internal_zones'],
            seed_dist_dir=self.imports['decomp_post_me'],
            seed_infill=0,
            normalise_seeds=False,
            dist_out=int_dir,
            report_out=self.exports['dist_reports'],
            csv_out=False,
            compress_out=True,
            verbose=echo_distribution
        )

        # Distribute the external trips, write to disk
        # DO NOT INCLUDE EG in external
        self._distribute_external_demand(
            p_vector=pre_eg_model_p_vector,
            nhb_p_vector=pre_eg_model_nhb_p_vector,
            zone_col=model_zone_col,
            years_needed=year_list,
            hb_p_needed=hb_purposes_needed,
            nhb_p_needed=nhb_purposes_needed,
            m_needed=modes_needed,
            soc_needed=soc_needed,
            ns_needed=ns_needed,
            ca_needed=car_availabilities_needed,
            external_zones_path=self.imports['external_zones'],
            post_me_dir=self.imports['decomp_post_me'],
            dist_out=ext_dir,
            report_out=self.exports['dist_reports'],
            csv_out=False,
            compress_out=True,
            verbose=True,
        )

        last_time = current_time
        current_time = time.time()
        print("Distribution generation took: %.2f seconds" %
              (current_time - last_time))

        # ## WFH ADJUSTMENTS ## #
        if self.apply_wfh_adjustments:
            print("Applying WFH adjustments...")

            # Create the temporary output folders
            dist_out_wfh = self.exports['pa_24_wfh']
            int_dir_wfh = os.path.join(dist_out_wfh, 'internal')
            ext_dir_wfh = os.path.join(dist_out_wfh, 'external')

            for path in [int_dir_wfh, ext_dir_wfh]:
                du.create_folder(path, verbose=False)

            # Apply WFH adjustment to the internal and external mats
            for in_dir, out_dir in zip([int_dir, ext_dir], [int_dir_wfh, ext_dir_wfh]):
                self._apply_wfh_adjustments(
                    import_dir=in_dir,
                    export_dir=out_dir,
                    years=future_years,
                )

            int_dir = int_dir_wfh
            ext_dir = ext_dir_wfh
            dist_out = dist_out_wfh

            last_time = current_time
            current_time = time.time()
            print("WFH adjustments applied! Took: %.2f seconds" %
                  (current_time - last_time))

        # If we're not combining, we need to exit here!
        if not combine_internal_external:
            return

        # Combine the internal and external trips
        print("Recombining internal and external matrices...")
        mat_p.recombine_internal_external(
            internal_import=int_dir,
            external_import=ext_dir,
            full_export=dist_out,
            force_csv_out=True,
            years=year_list,
        )

        last_time = current_time
        current_time = time.time()
        print("Recombining internal and external matrices took: %.2f seconds" %
              (current_time - last_time))

        # ## SECTOR TOTALS ## #
        sector_grouping_file = os.path.join(self.imports['zone_translation']['home'],
                                            "tfn_level_one_sectors_norms_grouping.csv")

        sector_totals = self.sector_reporter.calculate_sector_totals(
                model_p_vector,
                grouping_metric_columns=year_list,
                sector_grouping_file=sector_grouping_file,
                zone_col=model_zone_col
                )

        pm_sector_total_dictionary = {}

        for purpose in hb_purposes_needed:
            # TODO: Update sector reporter.
            #  Sector totals don't currently allow per purpose reporting

            pm_productions = model_p_vector.copy()

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

    def _apply_wfh_adjustments_worker(self,
                                      import_path: nd.PathLike,
                                      export_path: nd.PathLike,
                                      adjustment: nd.PathLike,
                                      ) -> None:
        # Read in, adjust, write out
        mat = file_ops.read_df(import_path, index_col=0)
        mat = mat * adjustment
        file_ops.write_df(mat, export_path)

    def _apply_wfh_adjustments(self,
                               import_dir: nd.PathLike,
                               export_dir: nd.PathLike,
                               years: List[int],
                               ) -> None:
        """
        Applies a WFH adjustment to import_dir mats and writes out to export_dir
        """
        # TODO(BT): This is a quick write - revisit when we have time
        # Init
        all_mats = file_ops.list_files(import_dir, efs_consts.VALID_MATRIX_FORMATS)
        wfh_adj = pd.read_csv(self.imports['wfh_adj'])
        unq_soc = wfh_adj['soc'].unique()

        # Filter the adjustment
        mask = (wfh_adj['scenario'] == self.scenario_name)
        wfh_adj = wfh_adj[mask]

        # Only apply to commute matrices
        p_str = "_p%s_" % 1
        commute_mats = [x for x in all_mats if p_str in x]

        # Figure out the adjustments to apply to each matrix
        adjusted_mats = list()
        kwarg_list = list()
        for year in years:
            # Get the mats and adjustments
            yr_str = "_yr%s_" % year
            yr_mats = [x for x in commute_mats if yr_str in x]

            mask = (wfh_adj['year'] == year)
            yr_adj = wfh_adj[mask]

            for soc in unq_soc:
                # Filter the mats
                soc_str = "_soc%s" % soc
                soc_mats = [x for x in yr_mats if soc_str in x]

                # Keep track of all matrices being adjusted
                adjusted_mats += soc_mats

                # Figure out the adjustment to apply
                mask = (yr_adj['soc'] == soc)
                soc_adj = yr_adj[mask]

                if len(soc_adj) > 1:
                    raise ValueError(
                        "There seems to be more than one WFH adjustment for:\n"
                        "scenario: %s, year: %s, soc: %s"
                        % (self.scenario_name, year, soc)
                    )

                if len(soc_adj) < 1:
                    raise ValueError(
                        "There seems to be no WFH adjustment for:\n"
                        "scenario: %s, year: %s, soc: %s"
                        % (self.scenario_name, year, soc)
                    )

                # Update the multiprocessing kwargs
                for fname in soc_mats:
                    kwarg_list.append({
                        'import_path': os.path.join(import_dir, fname),
                        'export_path': os.path.join(export_dir, fname),
                        'adjustment': soc_adj['commute_correction'].squeeze(),
                    })

        # Adjust all the matrices at once
        multiprocessing.multiprocess(
            fn=self._apply_wfh_adjustments_worker,
            kwargs=kwarg_list,
            process_count=consts.PROCESS_COUNT,
        )

        # Copy over all the non-adjusted mats
        copy_mats = [x for x in all_mats if x not in adjusted_mats]
        file_ops.copy_files(
            src_dir=import_dir,
            dst_dir=export_dir,
            filenames=copy_mats,
            process_count=consts.PROCESS_COUNT,
        )

    def _distribute_internal_demand(self,
                                    p_vector: pd.DataFrame,
                                    nhb_p_vector: pd.DataFrame,
                                    a_vector: pd.DataFrame,
                                    nhb_a_vector: pd.DataFrame,
                                    internal_zones_path: nd.PathLike,
                                    zone_col: str,
                                    years_needed: List[str],
                                    hb_p_needed: List[int],
                                    nhb_p_needed: List[int],
                                    verbose: bool = False,
                                    **kwargs,
                                    ) -> None:
        """
        Distributes the internal demand only Using a furness process.

        Essentially a wrapper around furness.distribute_pa() to make sure
        only the internal proportion of each vector is furnessed

        Given p and a vectors should contain internal demand only!

        """
        # Init
        hb_vals = [p_vector, a_vector, 'hb', hb_p_needed]
        nhb_vals = [nhb_p_vector, nhb_a_vector, 'nhb', nhb_p_needed]
        seed_year, _ = du.split_base_future_years_str(years_needed)

        # Read in the internal zones
        dtype = {zone_col: p_vector[zone_col].dtype}
        internal_zones = pd.read_csv(internal_zones_path, dtype=dtype).squeeze().tolist()

        # Do for the HB and then NHB trips
        for p, a, to, p_needed in [hb_vals, nhb_vals]:

            # Get the weights
            a_weights = du.convert_to_weights(a, years_needed)

            # Distribute the trips and write to disk
            print("Generating %s internal distributions..." % to.upper())
            furness.distribute_pa(
                productions=p,
                attraction_weights=a_weights,
                trip_origin=to,
                seed_year=seed_year,
                years_needed=years_needed,
                zone_col=zone_col,
                unique_zones=internal_zones,
                unique_zones_join_fn=operator.and_,
                p_needed=p_needed,
                fname_suffix='_int',
                **kwargs,
            )

    def _distribute_external_demand(self,
                                    p_vector: pd.DataFrame,
                                    nhb_p_vector: pd.DataFrame,
                                    external_zones_path: nd.PathLike,
                                    post_me_dir: nd.PathLike,
                                    dist_out: nd.PathLike,
                                    report_out: nd.PathLike,
                                    zone_col: str,
                                    years_needed: List[str],
                                    hb_p_needed: List[int],
                                    nhb_p_needed: List[int],
                                    m_needed: List[int],
                                    soc_needed: List[int] = None,
                                    ns_needed: List[int] = None,
                                    ca_needed: List[int] = None,
                                    csv_out: bool = True,
                                    compress_out: bool = True,
                                    verbose: bool = False,
                                    ) -> None:
        """
        Distributes the external demand only by growing post-me matrices.
        """
        # Init
        hb_vals = [p_vector, 'hb', hb_p_needed]
        nhb_vals = [nhb_p_vector, 'nhb', nhb_p_needed]
        base_year, future_years = du.split_base_future_years_str(years_needed)

        # Load in the external zones
        dtype = {zone_col: p_vector[zone_col].dtype}
        external_zones = pd.read_csv(external_zones_path, dtype=dtype).squeeze().tolist()

        # ## COPY OVER SEED EXTERNAL FOR BASE YEAR ## #
        print("Getting base year external distributions...")
        mat_p.split_internal_external(
            mat_import=post_me_dir,
            matrix_format='pa',
            year=base_year,
            external_zones=external_zones,
            external_export=dist_out,
        )

        # ## GROW THE FUTURE YEARS ## #
        # Do for the HB and then NHB trips
        for vector, to, p_needed in [hb_vals, nhb_vals]:

            # Calculate the growth factors
            growth_factors = vector.copy()
            for year in future_years:
                growth_factors[year] /= growth_factors[base_year]
            growth_factors.drop(columns=[base_year], inplace=True)

            print("Generating %s external distributions..." % to.upper())
            ext_growth.grow_external_pa(
                import_dir=dist_out,
                export_dir=dist_out,
                growth_factors=growth_factors,
                zone_col=zone_col,
                base_year=base_year,
                future_years=future_years,
                trip_origin=to,
                p_needed=p_needed,
                m_needed=m_needed,
                soc_needed=soc_needed,
                ns_needed=ns_needed,
                ca_needed=ca_needed,
                report_out=report_out,
                fname_suffix='_ext',
                csv_out=csv_out,
                compress_out=compress_out,
                verbose=verbose,
            )

    def _handle_growth_criteria(self,
                                synth_productions: pd.DataFrame,
                                synth_nhb_productions: pd.DataFrame,
                                synth_attractions: pd.DataFrame,
                                synth_nhb_attractions: pd.DataFrame,
                                base_year: str,
                                future_years: List[str],
                                integrate_dlog: bool,
                                internal_zones: List[int] = None,
                                external_zones: List[int] = None,
                                ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
        fname = efs_consts.POP_FNAME % self.input_zone_system
        grown_pop_path = os.path.join(self.exports["productions"], fname)

        fname = efs_consts.EMP_FNAME % self.input_zone_system
        grown_emp_path = os.path.join(self.exports["attractions"], fname)

        # ## APPLY GROWTH CRITERIA ## #
        # TODO: Need norms_to_tfn sector lookups.
        #  Should these be pop/emp weighted too?
        sector_system = "tfn_sectors"
        model_zone_to_sector_path = os.path.join(
            self.imports["zone_translation"]['one_to_one'],
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
        hb_p, nhb_p, hb_a, nhb_a = eg.growth_criteria(
            synth_productions=synth_productions,
            synth_nhb_productions=synth_nhb_productions,
            synth_attractions=synth_attractions,
            synth_nhb_attractions=synth_nhb_attractions,
            observed_pa_path=self.imports["decomp_post_me"],
            observed_cache=self.exports['post_me']['cache'],
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
            attr_audits=self.exports["attractions"],
            internal_zones=internal_zones,
            external_zones=external_zones,
        )

        return hb_p, nhb_p, hb_a, nhb_a

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
        fname = efs_consts.POP_TRANSLATION_FNAME % fname_args
        path = os.path.join(self.imports['zone_translation']['weighted'], fname)
        pop_translation = pd.read_csv(path)

        # Read in emp translation
        fname = efs_consts.EMP_TRANSLATION_FNAME % fname_args
        path = os.path.join(self.imports['zone_translation']['weighted'], fname)
        emp_translation = pd.read_csv(path)

        return pop_translation, emp_translation

    def _get_time_splits_from_p_vector(self,
                                       trip_origin: str,
                                       years_needed: List[int] = efs_consts.ALL_YEARS,
                                       ignore_cache: bool = False,
                                       ) -> pd.DataFrame:
        # TODO: cache!
        # TODO: check trip_origin is valid
        # If the file already exists, just return that
        file_type = '%s_tp_splits' % trip_origin
        fname = efs_consts.PRODS_FNAME % (self.output_zone_system, file_type)
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
        fname = efs_consts.PRODS_FNAME % (self.input_zone_system, file_type)
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
                 years_needed: List[int] = efs_consts.ALL_YEARS,
                 m_needed: List[int] = efs_consts.MODES_NEEDED,
                 p_needed: List[int] = consts.ALL_P,
                 round_dp: int = consts.DEFAULT_ROUNDING,
                 use_bespoke_pa: bool= False,
                 use_elasticity_pa: bool= True,
                 verbose: bool = True
                 ) -> None:
        """
        Converts home based PA matrices into time periods split OD matrices.

        Generates to_home, from_home, and full OD for HB trips. Conversion is
        based on the tour proportions derived from decompiling the post-ME
        matrices.

        NHB matrices are split using a set of time period splitting factors
        derived from decompiling the post-ME matrices.

        Parameters
        ----------
        years_needed:
            The years of PA matrices to convert to OD

        m_needed:
            The modes of PA matrices to convert to OD

        p_needed:
            The purposes of PA matrices to convert to OD

        round_dp:
            The number of decimal places to round the output values to.
            Uses efs_consts.DEFAULT_ROUNDING by default.

        verbose:
            If True, suppresses some of the non-essential terminal outputs.

        Returns
        -------
        None
        """
        # Init
        _input_checks(m_needed=m_needed)
        base_zone_col = "%s_zone_id"
        pa_import = 'pa_24_bespoke' if use_bespoke_pa else 'pa_24_wfh'
        pa_import = 'pa_24_elast' if use_elasticity_pa else pa_import
        hb_p_needed, nhb_p_needed = du.split_hb_nhb_purposes(p_needed)

        # Set up the iterator
        iterator = zip(
            [hb_p_needed, nhb_p_needed],
            ['hb', 'nhb'],
        )

        # Aggregate to TMS level?
        for sub_p_needed, to in iterator:
            mat_p.aggregate_matrices(
                import_dir=self.exports[pa_import],
                export_dir=self.exports['aggregated_pa'],
                trip_origin=to,
                matrix_format='pa',
                years_needed=years_needed,
                p_needed=sub_p_needed,
                m_needed=m_needed,
                ca_needed=self.ca_needed,
                round_dp=round_dp,
            )

        # Set up the segmentation params
        seg_level = 'tms'
        seg_params = {
            'p_needed': hb_p_needed,
            'm_needed': m_needed,
            'ca_needed': self.ca_needed,
        }

        # Convert HB to OD via tour proportions
        pa2od.build_od_from_fh_th_factors_old(
            pa_import=self.exports['aggregated_pa'],
            od_export=self.exports['od'],
            fh_th_factors_dir=self.imports['post_me_fh_th_factors'],
            years_needed=years_needed,
            seg_level=seg_level,
            seg_params=seg_params
        )

        # Convert NHB to tp split via factors
        nhb_seg_params = seg_params.copy()
        nhb_seg_params['p_needed'] = nhb_p_needed

        mat_p.nhb_tp_split_via_factors(
            import_dir=self.exports['aggregated_pa'],
            export_dir=self.exports['od'],
            import_matrix_format='pa',
            export_matrix_format='od',
            tour_proportions_dir=self.imports['post_me_tours'],
            model_name=self.model_name,
            future_years_needed=years_needed,
            **nhb_seg_params,
        )

    def old_pa_to_od(self,
                     years_needed: List[int] = efs_consts.ALL_YEARS,
                     m_needed: List[int] = efs_consts.MODES_NEEDED,
                     p_needed: List[int] = consts.ALL_P,
                     soc_needed: List[int] = efs_consts.SOC_NEEDED,
                     ns_needed: List[int] = efs_consts.NS_NEEDED,
                     ca_needed: List[int] = efs_consts.CA_NEEDED,
                     round_dp: int = consts.DEFAULT_ROUNDING,
                     use_bespoke_pa: bool= True,
                     overwrite_hb_tp_pa: bool = True,
                     overwrite_hb_tp_od: bool = True,
                     verbose: bool = True
                     ) -> None:
        # BACKLOG: Need Tour proportions from NoRMS in order to properly
        #  integrate bespoke zones. Currently using TMS tp_splits and
        #  phi_factors to convert PA to OD
        #  labels: NoRMS, EFS
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

        round_dp:
            The number of decimal places to round the output values to.
            Uses efs_consts.DEFAULT_ROUNDING by default.

        # TODO: Update docs once correct functionality exists
        overwrite_hb_tp_pa:
            Whether to split home based PA matrices into time periods.

        overwrite_hb_tp_od:
            Whether to convert time period split PA matrices into OD matrices.

        verbose:
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

            if to_p_needed == list():
                print("Not splitting %s trips into time period as no "
                      "purposes for this mode were given.")
                continue

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
                    model_name=self.model_name,
                    years_needed=years_needed,
                    p_needed=to_p_needed,
                    m_needed=m_needed,
                    soc_needed=soc_needed,
                    ns_needed=ns_needed,
                    ca_needed=ca_needed,
                    round_dp=round_dp,
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
                round_dp=round_dp,
                verbose=verbose,
            )

            # Copy over NHB matrices as they are already in NHB format
            mat_p.copy_nhb_matrices(
                import_dir=self.exports['pa'],
                export_dir=self.exports['od'],
                replace_pa_with_od=True,
            )

            print('HB OD matrices compiled!\n')
            # TODO: Create 24hr OD for HB

    def compile_matrices(self,
                         years: List[int],
                         m_needed: List[int] = efs_consts.MODES_NEEDED,
                         tp_needed: List[int] = efs_consts.TIME_PERIODS,
                         round_dp: int = consts.DEFAULT_ROUNDING,
                         use_bespoke_pa: bool = False,
                         use_elasticity_pa: bool = False,
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

        m_needed:
            The mode to use when compiling and aggregating OD matrices. This
            will be used to determine if car availability needs to be included
            or not

        tp_needed:
            The time periods to use when compiling and aggregating OD matrices.

        round_dp:
            The number of decimal places to round the output values to.
            Uses efs_consts.DEFAULT_ROUNDING by default.

        Returns
        -------
        None
        """
        # Init
        _input_checks(m_needed=m_needed)
        pa_import = 'pa_24_bespoke' if use_bespoke_pa else 'pa_24'
        pa_import = 'pa_24_elast' if use_elasticity_pa else pa_import

        if self.is_ca_needed:
            ca_needed = efs_consts.CA_NEEDED
        else:
            ca_needed = [None]

        if self.model_name == 'noham':
            compile_params_paths = mat_p.build_compile_params(
                import_dir=self.exports['od'],
                export_dir=self.params['compile'],
                matrix_format='od',
                years_needed=years,
                m_needed=m_needed,
                ca_needed=ca_needed,
                tp_needed=tp_needed,
            )

            for path in compile_params_paths:
                mat_p.compile_matrices(
                    mat_import=self.exports['od'],
                    mat_export=self.exports['compiled_od'],
                    compile_params_path=path,
                    round_dp=round_dp,
                )
            
            car_occupancies = pd.read_csv(os.path.join(
                self.imports['home'],
                'vehicle_occupancies',
                'car_vehicle_occupancies.csv',
            ))

            # Need to convert into hourly average PCU for noham
            vo.people_vehicle_conversion(
                mat_import=self.exports['compiled_od'],
                mat_export=self.exports['compiled_od_pcu'],
                car_occupancies=car_occupancies,
                mode=m_needed[0],
                method='to_vehicles',
                out_format='wide',
                hourly_average=True,
                round_dp=round_dp,
            )

        elif self.model_name == 'norms':
            # Load in the splitting factors
            fname = consts.POSTME_FROM_TO_FACTORS_FNAME
            path = os.path.join(self.imports['params'], fname)
            from_to_split_factors = pd.read_pickle(path)

            for year in years:
                # Compile
                mat_p.compile_norms_to_vdm(
                    mat_pa_import=self.exports[pa_import],
                    # TODO(BT): Actually pass in OD here
                    mat_od_import=self.exports[pa_import],
                    mat_export=self.exports['compiled_pa'],
                    params_export=self.params['compile'],
                    year=year,
                    m_needed=m_needed,
                    internal_zones=self.model_internal_zones,
                    external_zones=self.model_external_zones,
                    pa_matrix_format='pa',
                    od_to_matrix_format='pa',
                    od_from_matrix_format='pa',
                    nhb_od_matrix_format='pa',
                    from_to_split_factors=from_to_split_factors,
                )
        else:
            raise ValueError(
                "Not sure how to compile matrices for model %s"
                % self.model_name
            )

    def decompile_post_me(self,
                          year: int = efs_consts.BASE_YEAR,
                          m_needed: List[int] = efs_consts.MODES_NEEDED,
                          make_new_observed: bool = False,
                          overwrite_decompiled_matrices: bool = True,
                          overwrite_tour_proportions: bool = True,
                          ) -> None:
        """
        Decompiles post-me matrices ready to be used in an EFS future years run.

        Reads in the post-me matrices from the TfN model defined in
        self.model_name and decompiles them into TfN segmented 24hr PA
        matrices. These matrices are needed by EFS to generate future year
        travel matrices.

        In the case of NoHAM post-me matrices, they need converting from
        OD to PA matrices, and therefore produce a set of tour proportions
        alongside the decompiled matrices.

        This process CANNOT run unless TMS has already completed a base year
        run and compiled a set of matrices for the TfN model, therefore
        producing a set of decompilation factors that EFS will need.

        This function acts as a front end for calling model specific
        decompilation functions. See model decompilation functions for more
        information.

        Parameters
        ----------
        year:
             The year to decompile OD matrices for. (Usually the base year)

        m_needed:
            The mode to use when decompiling OD matrices. This will be used
            to determine if car availability needs to be included or not.

        make_new_observed:
            Whether to copy the decompiled matarices back into the EFS
            imports ready for a new run of EFS, using these values as the
            observed data.

        # TODO: Update docs once correct functionality exists
        overwrite_decompiled_matrices:
            Whether to decompile the post-me matrices or not

        overwrite_tour_proportions:
            Whether to generate tour proportions or not.


        Returns
        -------
        None
        """
        # Init
        _input_checks(m_needed=m_needed)

        if self.model_name == 'noham':
            # Build the segmentation parameters for OD2PA
            # TODO(BT): Convert to use class arguments once implemented
            seg_params = {
                'p_needed': consts.ALL_P,
                'm_needed': m_needed,
                'ca_needed': self.ca_needed,
            }
            decompilation.decompile_noham(
                year=year,
                seg_level='tms',
                seg_params=seg_params,
                post_me_import=self.imports['post_me_matrices'],
                post_me_renamed_export=self.exports['post_me']['compiled_od'],
                od_export=self.exports['post_me']['od'],
                pa_export=self.exports['post_me']['pa'],
                pa_24_export=self.exports['post_me']['pa_24'],
                zone_translate_dir=self.imports['zone_translation']['one_to_one'],
                tour_proportions_export=self.params['tours'],
                decompile_factors_path=self.imports['post_me_factors'],
                vehicle_occupancy_import=self.imports['home'],
                overwrite_decompiled_od=overwrite_decompiled_matrices,
                overwrite_tour_proportions=overwrite_tour_proportions,
            )

        elif self.model_name == 'norms':
            if not overwrite_decompiled_matrices:
                print("WARNING: Not decompiling Norms matrices!!!")
                return

            fname = consts.POSTME_FROM_TO_FACTORS_FNAME
            from_to_factors_out = os.path.join(self.params['home'], fname)

            decompilation.decompile_norms(
                year=year,
                post_me_import=self.imports['post_me_matrices'],
                post_me_renamed_export=self.exports['post_me']['vdm_pa_24'],
                post_me_decompiled_export=self.exports['post_me']['pa_24'],
                decompile_factors_dir=self.imports['params'],
                from_to_factors_out=from_to_factors_out
            )

            # Copy all of our outputs into the observed import location
            if make_new_observed:
                file_ops.copy_all_files(
                    import_dir=self.exports['post_me']['pa_24'],
                    export_dir=self.imports['decomp_post_me'],
                    force_csv_out=True,
                )

        else:
            raise nd.NormitsDemandError(
                "Cannot decompile post-me matrices for %s. No function "
                "exists for this model to decompile matrices."
                % self.model_name
            )

    def _generate_paths(self, base_year: str) -> Tuple[Dict[str, str],
                                                       Dict[str, str],
                                                       Dict[str, str]]:
        """
        Returns imports, efs_exports and params dictionaries

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

        efs_exports:
            Dictionary of export paths with the following keys:
            productions, attractions, pa, od, pa_24, od_24, sectors

        params:
            Dictionary of parameter export paths with the following keys:
            compile, tours
        """
        return du.build_efs_io_paths(
            import_location=self.import_location,
            export_location=self.output_location,
            model_name=self.model_name,
            base_year=base_year,
            iter_name=self.iter_name,
            scenario_name=self.scenario_name,
            demand_dir_name=self.out_dir,
            by_land_use_iteration=self.by_land_use_iteration,
            fy_land_use_iteration=self.fy_land_use_iteration,
            land_use_drive=self.land_use_drive,
            verbose=self.verbose,
        )


def _input_checks(iter_num: int = None,
                  m_needed: List[int] = None,
                  constraint_required: Dict[str, bool] = None,
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
        expected_keys = efs_consts.CONSTRAINT_REQUIRED_DEFAULT.keys()
        for key in expected_keys:
            if key not in constraint_required:
                raise du.ExternalForecastSystemError(
                    "Missing '%s' key in constraint_required. Expected to "
                    "find all of the following keys:\n%s"
                    % (str(key), str(expected_keys))
                )


def write_input_info(output_path: str,
                     efs_version: str,
                     by_land_use_iter: str,
                     fy_land_use_iter: str,
                     base_year: int,
                     future_years: List[int],
                     output_zone_system: str,
                     post_me_location: str,
                     p_needed: List[int],
                     modes_needed: List[int],
                     soc_needed: List[int],
                     ns_needed: List[int],
                     car_availabilities_needed: List[int],
                     integrate_dlog: bool,
                     constraint_required: List[bool],
                     ) -> None:

    out_lines = [
        'EFS version: ' + str(efs_version),
        'BY Land Use Iter: ' + str(by_land_use_iter),
        'FY Land Use Iter: ' + str(fy_land_use_iter),
        'Run Date: ' + str(time.strftime('%D').replace('/', '_')),
        'Start Time: ' + str(time.strftime('%T').replace('/', '_')),
        "Base Year: " + str(base_year),
        "Future Years: " + str(future_years),
        "Output Zoning System: " + output_zone_system,
        "Post-ME Matrices Location: " + post_me_location,
        "Purposes Used: " + str(p_needed),
        "Modes Used: " + str(modes_needed),
        "Soc Used: " + str(soc_needed),
        "Ns Used: " + str(ns_needed),
        "Car Availabilities Used: " + str(car_availabilities_needed),
        "Development Log Integrated: " + str(integrate_dlog),
        "Constraints Used On: " + str(constraint_required),
    ]
    with open(output_path, 'w') as out:
        out.write('\n'.join(out_lines))
