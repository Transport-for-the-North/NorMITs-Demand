# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 16:43:09 2020

Wrapper script to run TMS start to finish.
# Steps as follows
# *Optional

# Import params sheet
# Check import folders
# Run required lookups
# Run production model
# Run attraction model
# Run cjtw translation
# Run distribution model
# Run external model
# Compile tp pa
# *Run pa to od
# Run nhb production model
# Compile tp nhb pa
# *Compile aggregate matrices
# *Translate trips to vehicles

"""

import os
import sys

sys.path.append(r'C:\Users\genie\Documents\GitHub\Travel-Market-Synthesiser')

import pandas as pd
import numpy as np

from normits_demand.models import production_model as pm
from normits_demand.models import attraction_model as am
from normits_demand.models import distribution_model as dm

from normits_demand.reports import reports_audits as ra
from normits_demand.utils import cjtw_processing as cjtw
from normits_demand.models import external_model as em
from normits_demand.matrices import tms_pa_to_od as pa2od
from normits_demand.utils import utils as nup
from normits_demand.utils import vehicle_occupancy as vo
from normits_demand.distribution import segment_disaggregator as sd

# TODO: define run as class
# class TMS_Run()

class TravelMarketSynthesiser:
    """
    """

    def __init__(self,
                 config_path = 'Y:/NorMITs Synthesiser/config/',
                 params_file = 'norms_params_sheet_i6.xlsx'):
        """
        """
        # Set config path
        self.config_path = config_path
        # Import lookup requirements
        self.input_reqs = pd.read_csv(os.path.join(config_path,
                                                   'input_reqs.csv'),
                                      squeeze = True)

        # Import and parse run parameters
        param_path = os.path.join(config_path, params_file)
        self.param_path = param_path

        if '.csv' in param_path:
            params = pd.read_csv(param_path,
                                 skiprows=1,
                                 index_col='param_index')
        else:
            params = pd.read_excel(param_path,
                                   skiprows=1,
                                   index_col='param_index',
                                   engine='openpyxl')

        param_dict = {}
        for index, row in params.iterrows():
            param_name = row['param_name'].lower().replace(' ','_')
            param_dat = row['param']
    
            if ',' in str(param_dat):
                param_dat = list(param_dat.replace(' ','').split(','))
    
            # Handle booleans
            if param_dat == 'TRUE':
                param_dat = True
            elif param_dat == 'FALSE':
                param_dat = False
    
            # Handle Nones
            if param_dat == 'None':
                param_dat = None
            elif param_dat == 'nan':
                param_dat = None
    
            param_dict.update({param_name:param_dat})
            
        self.params = param_dict

        mlz_folder = 'Model Zone Lookups'
        import_folder = 'import'

        self.run_folder = os.path.join(
            self.params['base_directory'],
            self.params['model_name'])
        # Index existing lookups
        self.lookup_folder = os.path.join(
            self.params['base_directory'],
            self.params['model_name'],
            mlz_folder)
        self.import_folder = os.path.join(
            self.params['base_directory'],
            import_folder)

        # Build run order from input params
        run_dict = {}
        run_dict.update({'hb_p_run':''})
        run_dict.update({'hb_a_run':''})
        run_dict.update({'nhb_a_run':''})
        run_dict.update({'nhb_p_run':''})

        self.run_dict = run_dict

    def audit_params(self):
        """
        Function to take the config dict derived from param import and check
        that it's all going to work.
    
        Takes:
        param_dict:
            Dictionary of params from get_params
    
        Returns:
        status:
            True or False. Model ready to run.
        """
        # BACKLOG: Needs to be filled out
        if self.params:
            status = True
        else:
            status = False
    
        return(status)

    def lookup_audit(self,
                     run_missing = True):
        """
        Go through config list of lookups and config data required to run model
    
        Takes:
        lookup_folder:
            Model lookup folder, should take the 'Model Zone Lookups' path as str
        lookup_list_path:
            Path to lookup list of required lookups/data for run. From config.
        """
        lookup_list = self.input_reqs

        conts = os.listdir(self.lookup_folder)

        pass_status = True

        pass_dict = {}
        for l in lookup_list:
            if len([x for x in conts if l in x]) > 0:
                exists = True
            else:
                if run_missing == True:
                    if self.run_lookup(l):
                        exists = True
                    else:
                        exists = False
                        print(l + ' run failed')

                # Change pass status - need them all!
                pass_status = False
            pass_dict.update({l:exists})
    
        return(pass_status, pass_dict)

    def run_lookup(self, lookup):
        """
        ['cjtw',
         'msoa_pop_weighted',
         'msoa_emp_weighted',
         'internal_area',
         'external_area',
         'init_params_hb',
         'init_params_nhb',
         'pa_matrix_params',
         'od_matrix_params',
         'ie_factors',
         'tlb']
        All of the information to do this comprehensively is in the object.
        """
        status = False

        return status

    def project_check(self):
        """
        
        """
        status = False
        
        run_dict = {}
        
        return status, run_dict
        
    def run(self):
    
        """
        Wrapper function to run TMS start to finish based on specified params.
        """

        # Check status of lookup folder        
        self.lookups = self.lookup_audit()

        # TODO: Project status
        self.project_status = self.project_check()
        
        """
        ni6 = TravelMarketSynthesiser()

        hb_p = pm.ProductionModel(
            model_name = ni6.params['model_name'].lower(),
            build_folder = ni6.run_folder,
            iteration = ni6.params['iteration'],
            trip_origin = 'hb',
            input_zones = 'msoa',
            output_zones = ni6.params['model_name'].lower(),
            import_folder = ni6.import_folder,
            model_folder = ni6.lookup_folder,
            output_segments = ni6.params['hb_trip_end_segmentation'],
            lu_path = ni6.params['land_use_path'],
            trip_rates = ni6.params['hb_trip_rates'],
            time_splits = ni6.params['hb_time_splits'],
            mode_splits = ni6.params['hb_mode_splits'],
            ntem_control = ni6.params['production_ntem_control'],
            ntem_path = ni6.params['ntem_control_path'],
            k_factor_control = ni6.params['production_k_factor_control'],
            k_factor_path = ni6.params['production_k_factor_path'],
            export_msoa = False,
            export_uncorrected = True,
            export_target = True)
        
        a = am.AttractionModel(
            model_name = ni6.params['model_name'].lower(),
            build_folder = ni6.run_folder,
            iteration = ni6.params['iteration'],
            input_zones = 'msoa',
            output_zones = ni6.params['model_zoning'].lower(),
            import_folder = ni6.import_folder,
            model_folder = ni6.lookup_folder,
            output_segments = ni6.params['hb_trip_end_segmentation'],
            attractions_name = ni6.params['attractions_name'],
            attraction_weights = ni6.params['attraction_weights'],
            attraction_mode_split = ni6.params['attraction_mode_split']
            ntem_control = ni6.params['attraction_ntem_control'],
            ntem_path = ni6.params['ntem_control_path'],
            k_factor_control = ni6.params['production_k_factor_control'],
            k_factor_path = ni6.params['production_k_factor_path'],
            export_msoa = ni6.params['export_msoa'],
            export_lad = ni6.params['export_lad'],
            export_uncorrected = ni6.params['export_uncorrected'],
            export_target = ni6.params['export_model_zoning'])
        
        nhb_p = pm.ProductionModel(
            model_name = ni6.params['model_name'].lower(),
            build_folder = ni6.run_folder,
            iteration = ni6.params['iteration'],
            trip_origin = 'nhb',
            input_zones = 'msoa',
            output_zones = ni6.params['model_zoning'].lower(),
            import_folder = ni6.import_folder,
            model_folder = ni6.lookup_folder,
            output_segments = ni6.params['nhb_trip_end_segmentation'],
            trip_rates = ni6.params['nhb_trip_rates'],
            time_splits = ni6.params['nhb_time_splits'],
            mode_splits = ni6.params['nhb_mode_splits'],
            ntem_control = ni6.params['production_ntem_control'],
            ntem_path = ni6.params['ntem_control_path'],
            k_factor_control = ni6.params['production_k_factor_control'],
            k_factor_path = ni6.params['production_k_factor_path'],
            export_msoa = ni6.params['export_msoa'],
            export_lad = ni6.params['export_lad'],
            export_uncorrected = ni6.params['export_uncorrected'],
            export_target = ni6.params['export_model_zoning'])

        """

        # Initialise production model run
        hb_p = pm.ProductionModel(
            model_name = self.params['model_name'].lower(),
            build_folder = self.run_folder,
            iteration = self.params['iteration'],
            trip_origin = 'hb',
            input_zones = 'msoa',
            output_zones = self.params['model_zoning'].lower(),
            import_folder = self.import_folder,
            model_folder = self.lookup_folder,
            output_segments = self.params['hb_trip_end_segmentation'],
            lu_path = self.params['land_use_path'],
            trip_rates = self.params['hb_trip_rates'],
            time_splits = self.params['hb_time_splits'],
            mode_splits = self.params['hb_mode_splits'],
            ntem_control = self.params['production_ntem_control'],
            ntem_path = self.params['ntem_control_path'],
            k_factor_control = self.params['production_k_factor_control'],
            k_factor_path = self.params['production_k_factor_path'],
            export_msoa = self.params['export_msoa'],
            export_lad = self.params['export_lad'],
            export_uncorrected = self.params['export_uncorrected'],
            export_target = self.params['export_model_zoning']) # For current k factors only

        hb_p_out = hb_p.run_hb()
        # Update run dict
        self.run_dict.update({'hb_p_run':hb_p_out[0]})

        # Run attraction model - does hb & nhb
        a = am.AttractionModel(
            model_name = self.params['model_name'].lower(),
            build_folder = self.run_folder,
            iteration = self.params['iteration'],
            input_zones = 'msoa',
            output_zones = self.params['model_zoning'].lower(),
            import_folder = self.input_folder,
            model_folder = self.lookup_folder,
            output_segments = self.params['hb_trip_end_segmentation'],
            attractions_name = self.params['attractions_name'],
            attraction_weights = self.params['attraction_weights'],
            attraction_mode_split = self.params['attraction_mode_split'],
            ntem_control = self.params['attraction_ntem_control'],
            ntem_path = self.params['ntem_control_path'],
            k_factor_control = self.params['production_k_factor_control'],
            k_factor_path = self.params['production_k_factor_path'],
            export_msoa = self.params['export_msoa'],
            export_lad = self.params['export_lad'],
            export_uncorrected = self.params['export_uncorrected'],
            export_target = self.params['export_model_zoning'])

        a_out = a.run()
        # Update run dict
        self.run_dict.update({'hb_a_run':a_out[0][0]})
        self.run_dict.update({'nhb_a_run':a_out[0][1]})

        # Run nhb production model
        nhb_p = pm.ProductionModel(
            model_name = self.params['model_name'].lower(),
            build_folder = self.run_folder,
            iteration = self.params['iteration'],
            trip_origin = 'nhb',
            input_zones = 'msoa',
            output_zones = self.params['model_zoning'].lower(),
            import_folder = self.import_folder,
            model_folder = self.lookup_folder,
            output_segments = self.params['nhb_trip_end_segmentation'],
            trip_rates = self.params['nhb_trip_rates'],
            time_splits = self.params['nhb_time_splits'],
            mode_splits = self.params['nhb_mode_splits'],
            production_vector = self.run_dict['hb_p_run'],
            attraction_vector = self.run_dict['hb_a_run'],
            ntem_control = self.params['production_ntem_control'],
            ntem_path = self.params['ntem_control_path'],
            k_factor_control = self.params['production_k_factor_control'],
            k_factor_path = self.params['production_k_factor_path'],
            export_msoa = self.params['export_msoa'],
            export_lad = self.params['export_lad'],
            export_uncorrected = self.params['export_uncorrected'],
            export_target = self.params['export_model_zoning'])

        nhb_p.run_nhb() 

        # TODO: LA PA report for external output
        # Run HB external model
        ext_hb = em.ExternalModel(file_drive = params['base_directory'],
                                  model_name = params['model_name'],
                                  iteration = params['iteration'],
                                  tlb_area = 'gb',
                                  segmentation = 'tfn',
                                  distribution_segments = params['hb_distribution_segments'],
                                  trip_origin = 'hb',
                                  cost_type = '24hr',
                                  export_ext_modes = [],
                                  export_non_dist_modes = ['6'])
        ext_hb.run()

        # Run NHB external model
        ext_nhb = em.ExternalModel(file_drive = params['base_directory'],
                                   model_name = params['model_name'],
                                   iteration = params['iteration'],
                                   tlb_area = 'gb',
                                   segmentation = 'tfn',
                                   distribution_segments = params['nhb_distribution_segments'],
                                   trip_origin = 'nhb',
                                   cost_type = 'tp',
                                   export_ext_modes = [],
                                   export_non_dist_modes = ['6'])
        ext_nhb.run()
    
        # Run distribution model for hb
        # TODO: flexible trip length band building, or at least pathing
    
        # Will also run external model
        # TODO: Move path building out of DM function, make part of class
        # TODO: Specify distribution method, or try both.

        int_hb = dm.DistributionModel(
            model_name = params['model_name'],
            iteration = params['iteration'],
            tlb_area = 'north',
            segmentation = 'tfn',
            distribution_segments = params['hb_distribution_segments'],
            dist_function = 'ln',
            trip_origin = 'hb',
            cost_type = '24hr',
            furness_loops=1999,
            fitting_loops=100,
            iz_cost_infill = .5,
            export_modes = params['output_modes'],
            echo=True,
            mp_threads = -1)

        int_hb.run()

        int_nhb = dm.DistributionModel(,
                                       model_name = params['model_name'],
                                       iteration = params['iteration'],
                                       tlb_area = 'north',
                                       segmentation = 'tfn',
                                       distribution_segments = params['nhb_distribution_segments'],
                                       dist_function = 'ln',
                                       trip_origin = 'nhb',
                                       cost_type = 'tp',
                                       furness_loops=1999,
                                       fitting_loops=100,
                                       iz_cost_infill = .5,
                                       export_modes = params['output_modes'],
                                       echo=True,
                                       mp_threads = -1)
        int_nhb.run()
 
        # BUILD INTERNAL 24HR PA BY MODE
        # get all zone movements for OD conversion
    
        # Definitions from dm calls
    
        # TODO: Specify import export folders

        # Compile tp pa
        pa2od.build_tp_pa(file_drive = params['base_directory'],
                          model_name = params['model_name'],
                          iteration = params['iteration'],
                          distribution_segments = params['hb_output_segments'],
                          internal_input = 'synthetic',
                          external_input = 'synthetic',
                          write_modes = [3],
                          arrivals = False,
                          write = True)
    
        # *Compile tp pa @ 24hr
        pa2od.build_tp_pa(file_drive = params['base_directory'],
                          model_name = params['model_name'],
                          iteration = params['iteration'],
                          distribution_segments =params['hb_output_segments'],
                          internal_input = 'synthetic',
                          external_input = 'synthetic',
                          write_modes = [3],
                          arrivals = False,
                          export_24hr = True,
                          write = True)
    
        # Build any non dist tps as required
        pa2od.build_tp_pa(file_drive = params['base_directory'],
                          model_name = params['model_name'],
                          iteration = params['iteration'],
                          distribution_segments =params['hb_output_segments'],
                          internal_input = 'non_dist',
                          external_input = 'non_dist',
                          write_modes = [6],
                          arrivals = False,
                          export_24hr = False,
                          write = True)

        # TODO: Run new segment split function
        import_folder = 'Y:/NorMITs Synthesiser/Noham/iter8c/Distribution Outputs/PA Matrices'
        target_tld_folder = 'Y:/NorMITs Synthesiser/import/trip_length_bands/north/enhanced_segments'
        # TODO: Using single tld for whole country - run North and GB and compare
        base_hb_productions = 'Y:/NorMITs Synthesiser/Noham/iter8c/Production Outputs/hb_productions_noham.csv'
        base_nhb_productions = 'Y:/NorMITs Synthesiser/Noham/iter8c/Production Outputs/nhb_productions_noham.csv'
        base_hb_attractions = 'Y:/NorMITs Synthesiser/Noham/iter8c/Attraction Outputs/noham_hb_attractions.csv'
        base_nhb_attractions = 'Y:/NorMITs Synthesiser/Noham/iter8c/Attraction Outputs/noham_nhb_attractions.csv'
        export_folder = 'Y:/NorMITs Synthesiser/Noham/iter8c/Distribution Outputs/Segmented Distributions'
        #lookup_folder = 'Y:/NorMITs Synthesiser/Noham/Model Zone Lookups'
        # replaced above lookup_folder with NoRMS model zone lookups path
        lookup_folder = 'Y:/NorMITs Synthesiser/Norms/Model Zone Lookups'
    
        base_productions_path = base_hb_productions
        base_attractions_path = base_hb_attractions
    
        sd.disaggregate_segments(import_folder,
                                 target_tld_folder,
                                 base_productions_path,
                                 base_attractions_path,
                                 export_folder,
                                 lookup_folder,
                                 aggregate_surplus_segments = True,
                                 rounding = 5,
                                 trip_origin = 'hb',
                                 iz_infill = 0.5,
                                 furness_loops = 1999,
                                 mp_threads = -1)
    
        base_productions_path = base_nhb_productions
        base_attractions_path = base_nhb_attractions
    
        sd.disaggregate_segments(import_folder,
                                 target_tld_folder,
                                 base_productions_path,
                                 base_attractions_path,
                                 export_folder,
                                 lookup_folder,
                                 aggregate_surplus_segments = True,
                                 rounding = 5,
                                 trip_origin = 'nhb',
                                 tp='tp',
                                 iz_infill = 0.5,
                                 furness_loops = 1999,
                                 mp_threads = -1)
    
        if not params['rail_fusion']:
        # Build OD
            od_audit = pa2od.build_od(file_drive = params['base_directory'],
                                      model_name = params['model_name'],
                                      iteration = params['iteration'],
                                      distribution_segments = params['hb_output_segments'],
                                      internal_input = 'synthetic',
                                      external_input = 'synthetic',
                                      phi_type = 'fhp_tp',
                                      export_modes = [3],
                                      write = True)
    
            # Export other modes from non-compiled
            pa2od.build_od(file_drive = params['base_directory'],
                                      model_name = params['model_name'],
                                      iteration = params['iteration'],
                                      distribution_segments = params['hb_output_segments'],
                                      internal_input = 'non_dist',
                                      external_input = 'non_dist',
                                      phi_type = 'fhp_tp',
                                      export_modes = [6], # Define as parameter
                                      write = True)
    
            print(od_audit)
    
            pa2od.compile_nhb_pa(file_drive = params['base_directory'],
                                 model_name = params['model_name'],
                                 iteration = params['iteration'],
                                 distribution_segments = ['p','m','tp'],
                                 internal_input = 'synthetic',
                                 external_input = 'synthetic',
                                 export_modes = [3],
                                 write = True)
    
            # TODO: This needs some attention to work with non-dist segs
            pa2od.compile_nhb_pa(file_drive = params['base_directory'],
                                 model_name = params['model_name'],
                                 iteration = params['iteration'],
                                 distribution_segments = ['p','m','tp'],
                                 internal_input = 'non_dist',
                                 external_input = 'non_dist',
                                 export_modes = [1,2,5],
                                 write = True)
    
            # Copy across and rename pa
            # TODO: Should do these at the start and have as references
            pa_folder = od_folder = os.path.join(params['base_directory'],
                                                 'NorMITs Synthesiser',
                                                 params['model_name'],
                                                 params['iteration'],
                                                 'Distribution Outputs',
                                                 'PA Matrices')
    
            od_folder = os.path.join(params['base_directory'],
                                                 'NorMITs Synthesiser',
                                                 params['model_name'],
                                                 params['iteration'],
                                                 'Distribution Outputs',
                                                 'OD Matrices')
    
            nhb_pa = [x for x in [x for x in os.listdir(
            pa_folder) if '.csv' in x] if 'nhb' in x]
    
            for nhb_mat in nhb_pa:
                od_mat = nhb_mat.replace('pa', 'od')
                od_out = pd.read_csv(os.path.join(pa_folder, nhb_mat))
                print(od_mat)
                od_out.to_csv(os.path.join(od_folder, od_mat), index=False)
    
            # Run reports
            ra.distribution_report(params['base_directory'],
                                   params['model_name'],
                                   params['iteration'],
                                   params['hb_output_segments'],
                                   distributions = 'Distribution Outputs/OD Matrices',
                                   matrix_format = 'wide',
                                   report_tp = '24hr',
                                   internal_reports = True,
                                   write = True)
    
            # PA distribution reports (production audit)
            # TODO: Get to work with new outputs
            ra.distribution_report(params['base_directory'],
                                   params['model_name'],
                                   params['iteration'],
                                   params['hb_output_segments'],
                                   distributions = 'Distribution Outputs/PA Matrices',
                                   matrix_format = 'wide',
                                   report_tp = '24hr',
                                   internal_reports = True,
                                   write = True)
    
            # Import compilation params
            pa_compilation_params, od_compilation_params = nup.get_compilation_params(lookup_folder)
    
            if params['compile_pa']:
                compiled_pa_matrices = nup.compile_pa(o_paths['pa'],
                                                      (i_paths['lookups'] +
                                                       '/' +
                                                       pa_compilation_params)) # Split time = true
    
                # Export compiled PA
                for mat in compiled_pa_matrices:
                    for key,value in mat.items():
                        print(key)
                        c_pa_out = (o_paths['c_pa_export'] +
                                    '/'+
                                    key +
                                    '.csv')
                        value.to_csv(c_pa_out, index=False)
    
                # Reports on compiled PA and OD
                ra.distribution_report(params['base_directory'],
                                       params['model_name'],
                                       params['iteration'],
                                       ['p', 'm', 'ca'],
                                       distributions = 'Distribution Outputs/Compiled PA Matrices',
                                       matrix_format = 'long',
                                       report_tp = 'tp',
                                       internal_reports = False,
                                       write = True)
    
            # TODO: Export factors
            # TODO: Export from function
            if params['compile_od']:
                compiled_od_matrices = nup.compile_od(
                        od_folder = os.path.join(params['base_directory'],
                                                 'NorMITs Synthesiser',
                                                 params['model_name'],
                                                 params['iteration'],
                                                 'Distribution Outputs',
                                                 'OD Matrices'),
                        write_folder = os.path.join(params['base_directory'],
                                                 'NorMITs Synthesiser',
                                                 params['model_name'],
                                                 params['iteration'],
                                                 'Distribution Outputs',
                                                 'Compiled OD Matrices'),
                        compile_param_path = os.path.join(lookup_folder,
                                                          params['model_name'].lower() +
                                                          '_od_matrix_params.csv'),
                        build_factor_pickle = True)
    
                # Build compiled od for non dist segs
                # TODO: if statement
                nup.compile_od(
                        od_folder = os.path.join(params['base_directory'],
                                                 'NorMITs Synthesiser',
                                                 params['model_name'],
                                                 params['iteration'],
                                                 'Distribution Outputs',
                                                 'OD Matrices Non Dist'),
                                                 write_folder = 'C:/Users/genie/Documents/Nelum frh', # Write fast
                                                         compile_param_path = os.path.join(
                                                                 lookup_folder,
                                                                 params['model_name'].lower() +
                                                                 '_non_dist_od_frh_matrix_params.csv'),
                                                                 build_factor_pickle = False)
    
                # Run audit on compiled OD
                ra.distribution_report(params['base_directory'],
                                       params['model_name'],
                                       params['iteration'],
                                       params['hb_output_segments'],
                                       distributions = 'Distribution Outputs/Compiled OD Matrices',
                                       matrix_format = 'wide',
                                       report_tp = 'tp',
                                       internal_reports = False,
                                       write = True)
    
            # *Translate trips to vehicles
    
            # Export should be:
            os.path.join(params['base_directory'],
                                                 'NorMITs Synthesiser',
                                                 params['model_name'],
                                                 params['iteration'],
                                                 'PCU Outputs',
                                                 'PCU OD Matrices')
    
    
            # Convert to vehicles
            if params['vehicle_demand']:
                vo.people_vehicle_conversion(input_folder = os.path.join(params['base_directory'],
                                                 'NorMITs Synthesiser',
                                                 params['model_name'],
                                                 params['iteration'],
                                                 'Distribution Outputs',
                                                 'Compiled OD Matrices'),
                                             export_folder = 'D:/',
                                             mode = '3',
                                             method = 'to_vehicles',
                                             hourly_average = True,
                                             out_format = 'long',
                                             rounding_factor = None,
                                             header = False,
                                             write = True)
                # TODO: add 3 sectore report call
    
        elif params['rail_fusion']:
    
            # Run fusion engine integration
            # TODO: Call rail fusion
    
            # TODO:
            # Filter init_params to rail only
            rail_hb_init_params = init_params.copy()
            rail_hb_init_params = rail_hb_init_params[rail_hb_init_params['m']==6]
    
            if pa2od.build_tp_pa(params['hb_output_segments'],
                                 rail_hb_init_params,
                                 model_name,
                                 productions,
                                 internal_import = o_paths['fusion_summaries'],
                                 external_import = o_paths['external'],
                                 pa_export = 'D:/NorMITs Synthesiser/',
                                 write_modes = [6],
                                 arrivals = False,
                                 write = True):
                print('Transposed PA to tp PA')
    
            # Do nhb TP
            # 24hr OD to TP OD for NHB
            if pa2od.build_tp_pa(['p', 'm'],
                                 nhb_rail_init_params,
                                 model_name,
                                 nhb_productions,
                                 internal_import = o_paths['fusion_summaries'],
                                 external_import = o_paths['external'],
                                 pa_export = 'D:/NorMITs Synthesiser/',
                                 write_modes = [6],
                                 arrivals = False,
                                 write = True):
                print('Transposed NHB PA to tp PA and OD')
            else:
                raise ValueError('TP PA failed')
    
            # Build fusion OD
            fusion_od_audit = pa2od.build_od(params['hb_output_segments'],
                                             model_name,
                                             rail_hb_init_params,
                                             input_folder = o_paths['fusion_pa_export'],
                                             output_folder = 'D:/NorMITs Synthesiser/OD',
                                             phi_type = 'fhp_tp',
                                             export_modes = [6],
                                             write = True)
            print(fusion_od_audit)
    
        # Do NHB manually, copy from PA, rename OD
    
        # Get compilation params
    
        # Compile fusion PA
        if pa_compilation_params is not None:
            compiled_pa_matrices = nup.compile_pa(o_paths['fusion_pa_export'],
                            (i_paths['lookups'] +
                             '/' +
                             pa_compilation_params)) # Split time = true
    
            # Export compiled PA
            for mat in compiled_pa_matrices:
                for key,value in mat.items():
                    print(key)
                    c_pa_out = (o_paths['c_fusion_pa_export'] +
                                '/'+
                                key +
                                '.csv')
                    value.to_csv(c_pa_out, index=False)
    
        # Compile fusion OD
        if od_compilation_params is not None:
    
            compiled_od_matrices = nup.compile_od(o_paths['fusion_od_export'],
                            (i_paths['lookups'] +
                             '/' +
                             od_compilation_params))
    
            for mat in compiled_od_matrices:
                for key,value in mat.items():
                    print(key)
                    c_od_out = (o_paths['c_fusion_od_export'] +
                                '/'+
                                key +
                                '.csv')
                    value = value.sort_values(['o_zone','d_zone']).reset_index(drop=True)
                    value.to_csv(c_od_out, index=False)
    
        ra.distribution_report(params['base_directory'],
                               model_name = params['model_name'],
                               iteration = params['iteration'],
                               model_segments = ['p', 'm', 'ca'],
                               distributions = 'Fusion Outputs/Fusion OD Matrices',
                               matrix_format = 'wide',
                               report_tp = 'tp',
                               internal_reports = True,
                               write = True)
    
        ra.distribution_report(params['base_directory'],
                               model_name = params['model_name'],
                               iteration = params['iteration'],
                               model_segments = ['p', 'm', 'ca'],
                               distributions = 'Fusion Outputs/Compiled Fusion PA Matrices',
                               matrix_format = 'long',
                               report_tp = 'tp',
                               internal_reports = True,
                               write = True)
    
        ra.distribution_report(params['base_directory'],
                               model_name = params['model_name'],
                               iteration = params['iteration'],
                               model_segments = ['p', 'm', 'ca'],
                               distributions = 'Fusion Outputs/Compiled Fusion OD Matrices',
                               matrix_format = 'long',
                               report_tp = 'tp',
                               internal_reports = True,
                               write = True)
    
    
        # TODO: Run od 2 pa from post-me mats

        # Run segment split function
        import_folder = 'Y:/NorMITs Demand/noham/v2_2-EFS_Output/iter1/Matrices/Post-ME Matrices/24hr PA Matrices'
        target_tld_folder = 'Y:/NorMITs Synthesiser/import/trip_length_bands/north/enhanced_segments'
        # TODO: Using single tld for whole country - run North and GB and compare
        base_hb_productions = 'Y:/NorMITs Synthesiser/Noham/iter8c/Production Outputs/hb_productions_noham.csv'
        base_nhb_productions = 'Y:/NorMITs Synthesiser/Noham/iter8c/Production Outputs/nhb_productions_noham.csv'
        base_hb_attractions = 'Y:/NorMITs Synthesiser/Noham/iter8c/Attraction Outputs/noham_hb_attractions.csv'
        base_nhb_attractions = 'Y:/NorMITs Synthesiser/Noham/iter8c/Attraction Outputs/noham_nhb_attractions.csv'
        export_folder = 'Y:/NorMITs Demand/noham/v2_2-EFS_Output/iter1/Matrices/Segmented Post-ME Matrices'
        #lookup_folder = 'Y:/NorMITs Synthesiser/Noham/Model Zone Lookups'
        # replaced above lookup_folder with NoRMS model zone lookups path
        lookup_folder = 'Y:/NorMITs Synthesiser/Norms/Model Zone Lookups'
    
        base_productions_path = base_hb_productions
        base_attractions_path = base_hb_attractions
    
        sd.disaggregate_segments(import_folder,
                                 target_tld_folder,
                                 base_productions_path,
                                 base_attractions_path,
                                 export_folder,
                                 lookup_folder,
                                 aggregate_surplus_segments = True,
                                 rounding = 5,
                                 trip_origin = 'hb',
                                 iz_infill = 0.5,
                                 furness_loops = 1999,
                                 mp_threads = -1)
    
        base_productions_path = base_nhb_productions
        base_attractions_path = base_nhb_attractions
    
        sd.disaggregate_segments(import_folder,
                                 target_tld_folder,
                                 base_productions_path,
                                 base_attractions_path,
                                 export_folder,
                                 lookup_folder,
                                 aggregate_surplus_segments = True,
                                 rounding = 5,
                                 trip_origin = 'nhb',
                                 tp='tp',
                                 iz_infill = 0.5,
                                 furness_loops = 1999,
                                 mp_threads = -1)
    
        return 0