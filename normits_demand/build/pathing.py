# -*- coding: utf-8 -*-
"""
Normits Demand class wrapper
"""

import os
import pandas as pd

from normits_demand.utils import general as du


# TODO: define run as class
# class TMS_Run()

class Pathing:
    """
    """
    input_request = "input_reqs.csv"

    def __init__(self,
                 config_path='I:/NorMITs Synthesiser/config/',
                 params_file=None):
        """
        """
        # Set config path
        config_path_fname = os.path.join(config_path, self.input_request)
        # Import lookup requirements
        self.input_reqs = du.safe_read_csv(config_path_fname, squeeze=True)
        """
        Commented as of now but can be removed if not required
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
    
            #self.params = self.params_to_dict(params)
        """
        self.params = params_file
        # param_dict = params_to_dict(params)

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
        run_dict.update({'hb_p_run': ''})
        run_dict.update({'hb_a_run': ''})
        run_dict.update({'nhb_a_run': ''})
        run_dict.update({'nhb_p_run': ''})

        self.run_dict = run_dict

    @staticmethod
    def params_to_dict(params: pd.DataFrame) -> dict:
        """
        Parse input params dataframe
        return a dict, formatted for model use.

        Parameters:
             params: Data
        """

        param_dict = {}
        for index, row in params.iterrows():
            param_name = str(row['param_name']).lower().replace(' ', '_')
            param_dat = row['param']

            # Unpack commas
            if ',' in str(param_dat):
                param_dat = param_dat.replace(' ', '').split(',')
                lpd = []
                for p in param_dat:
                    if p.isnumeric():
                        lpd.append(int(p))
                    else:
                        lpd.append(p)
                param_dat = lpd

            # Handle booleans
            if str(param_dat).lower() == 'true':
                param_dat = True
            elif str(param_dat).lower() == 'false':
                param_dat = False

            # List type as list, avoid single ints, drop 'None'
            if row['param_dtype'] == 'list' and type(param_dat) != list:
                lpd = param_dat
                param_dat = []
                if type(lpd) is not int:
                    if lpd != 'None':
                        for l_it in lpd:
                            param_dat.append(l_it)
                else:
                    param_dat.append(lpd)

            param_dict.update({param_name: param_dat})

        return param_dict

    @classmethod
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

        return status

    def lookup_audit(self,
                     run_missing=True):
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
            pass_dict.update({l: exists})

        return (pass_status, pass_dict)

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
