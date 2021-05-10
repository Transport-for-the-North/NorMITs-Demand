
import os
import warnings
from typing import List

import numpy as np
import pandas as pd

import normits_demand.build.tem_pathing as tempath

import normits_demand.trip_end_constants as tec

from normits_demand.utils import utils as nup
from normits_demand.constraints import ntem_control as ntem
from normits_demand.utils.general import safe_dataframe_to_csv

# from normits_demand.models import production_model_new as pm

# ni6 = pm.ProductionModel(config_path, params_file)
# hb = ni6.run(trip_origin='hb')
# nhb = ni6.run(trip_origin='nhb')


class ProductionModel(tempath.TEMPathing):

    def __init__(
            self,
            config_path='I:/NorMITs Synthesiser/config/',
            param_file='norms_params_sheet_i6.xlsx'
            ):
        super().__init__(config_path,
                         param_file)

        # Get run paths
        self.hb_run_paths = self._get_run_paths(
            trip_origin='hb')
        self.nhb_run_paths = self._get_run_paths(
            trip_origin='nhb')

    def run(self,
            trip_origin='hb'):
        """

        """
        # Validate trip origin

        # Get run params
        run_data = self._get_run_data(
            trip_origin)

        return run_data

    @staticmethod
    def _build_production_vector(run_data,
                                 out_segmentation):
        p_vector = 0
        return p_vector

    def _get_run_paths(self,
                       trip_origin,
                       production_params_folder = None,
                       verbose=True):
        """
        """
        # Define where to look for non input vector production parameters
        if production_params_folder is not None:
            p_param_path = os.path.join(
                self.import_folder,
                production_params_folder)
        else:
            p_param_path = os.path.join(
                self.import_folder,
                'production_params')

        # Names that are different between runs
        if trip_origin == 'hb':
            input_vector = self.params['land_use_path']
            ave_time_split = os.path.join(
                p_param_path,
                self.params['hb_ave_time_split'])
        elif trip_origin == 'nhb':
            input_vector = self.export['a_in_hb']
            ave_time_split = 'Not used in NHB'

        # Names that are the same apart from trip origin
        trip_rates = os.path.join(
            p_param_path,
            self.params[trip_origin + '_trip_rates'])
        time_split = os.path.join(
            p_param_path,
            self.params[trip_origin + '_time_split'])
        mode_split = os.path.join(
            p_param_path,
            self.params[trip_origin + '_mode_split'])

        run_paths = {
            'input_vector': input_vector,
            'trip_rates': trip_rates,
            'time_split': time_split,
            'mode_split': mode_split,
            'ave_time_split': ave_time_split
        }

        if verbose:
            print(trip_origin + ' run:')
            for name, var in run_paths.items():
                print(name + ' from ' + var)

        return run_paths

    def _get_run_data(self,
                      trip_origin,
                      verbose=True):

        # Import hb input vectors
        if trip_origin == 'hb':
            input_vector = pd.read_csv(
                self.hb_run_paths['input_vector'])
            trip_rates = pd.read_csv(
                self.hb_run_paths['trip_rates'])
            time_split = pd.read_csv(
                self.hb_run_paths['time_split'])
            mode_split = pd.read_csv(
                self.hb_run_paths['mode_split'])
            ave_time_split = pd.read_csv(
                self.hb_run_paths['ave_time_split']
            )

        # Import nhb input vectors
        elif trip_origin == 'nhb':
            if os.path.exists(
                    self.hb_run_paths['input_vector']):
                input_vector = pd.read_csv(
                    self.hb_run_paths['input_vector'])
            else:
                # TODO: Value Error
                print('Input vector not available')
            trip_rates = pd.read_csv(
                self.nhb_run_paths['trip_rates'])
            time_split = pd.read_csv(
                self.nhb_run_paths['time_split'])
            mode_split = pd.read_csv(
                self.nhb_run_paths['mode_split'])
            # Just take the apology message from the path
            ave_time_split = self.nhb_run_paths['ave_time_split']
        else:
            raise ValueError('Non-valid trip origin')

        run_data = {
            'input_vector': input_vector,
            'trip_rates': trip_rates,
            'time_split': time_split,
            'mode_split': mode_split,
            'ave_time_split': ave_time_split
        }

        return run_data











