# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 11:26:27 2019
Script to apply car occupancies by model purpose and time period.
Also works the opposite way to get person trips from vehicle trips

@author: genie
"""

import os

import pandas as pd

from tqdm import tqdm

# File paths
_default_home_dir = 'Y:/NorMITs Synthesiser/Noham'
_import_file_drive = "Y:/"
_import_folder = 'Y:/NorMITs Synthesiser/import/'
_default_model_write_folder = '/Model PA'

_default_model_folder = 'Y:/NorMITs Model Zone Lookups/Noham'

_default_rounding = 3

_default_iter = 'iter3'

# if applying cars to people
_default_folder = ('Y:/NorMITs Synthesiser/Noham/' +
                   _default_iter +
                   '/Distribution Outputs/OD Matrices')


def people_vehicle_conversion(mat_import: str,
                              import_folder: str,
                              mat_export: str,
                              mode: int,
                              method: str = 'to_vehicles',
                              hourly_average: bool = True,
                              out_format: str = 'long',
                              header: bool = True,
                              write: bool = True
                              ) -> None:
    # TODO: Write people_vehicle_conversion() docs

    # TODO: Add refactor up to totals after conversion
    if method not in ['to_vehicles', 'to_people']:
        raise ValueError('method should be to_vehicles or to_people')

    file_sys = os.listdir(mat_import)

    # Should be isin and take list
    m_str = 'm' + str(mode)
    internal_file = [x for x in file_sys if m_str in x]

    c_o = pd.read_csv(import_folder + '/vehicle_occupancies/car_vehicle_occupancies.csv')

    # read in
    tps = ['tp1', 'tp2', 'tp3', 'tp4']

    period_hours = {
        1: 3,
        2: 6,
        3: 3,
        4: 12
    }

    # Define purpose lookup list
    purpose_lookup = [['commute', 'commuting'],
                      ['business', 'work'],
                      ['other', 'other']]

    # If people to vehicles, export to vehicle export

    desc = 'converting matrices by purpose'
    for pl in tqdm(purpose_lookup, desc=desc):
        # print(pl)

        # Do commute business and other seperately
        mats = [x for x in internal_file if pl[0] in x]

        for mpt in tps:
            # print(mpt)

            sub_co = c_o[c_o['trip_purpose'] == pl[1]]
            tp_int = int(mpt[-1])
            sub_co = sub_co[sub_co['time_period']==tp_int]
            factor = sub_co['car_occupancy'].squeeze()

            tp_mat = [x for x in mats if mpt in x]
            # print(tp_mat)
            # print(factor)

            # Get period factor
            p_factor = period_hours[tp_int]
            # print('Dividing by ' + str(p_factor))

            for f_loop in tp_mat:

                # print(input_folder + '/' + f_loop)
                ph_mat = pd.read_csv(mat_import + '/' + f_loop)

                cols = list(ph_mat)[1:-1]

                # For converting from people to vehicles hourly average refers
                # to the output matrix
                if method == 'to_vehicles':
                    for col in cols:
                        ph_mat[col] = ph_mat[col] / factor
                        if hourly_average:
                            ph_mat[col] = ph_mat[col] / p_factor

                # For converting from vehicles to people hourly average refers
                # to the input OD matrix
                elif method == 'to_people':
                    for col in cols:
                        if hourly_average:
                            ph_mat[col] = ph_mat[col] * p_factor
                        ph_mat[col] = ph_mat[col] * factor

                export_path = (mat_export + '/' + f_loop)
                # print(export_path)

                if out_format == 'long':
                    ph_mat = pd.melt(ph_mat,
                                     id_vars='o_zone',
                                     var_name='d_zone',
                                     value_name='dt',
                                     col_level=0)

                if write:
                    ph_mat.to_csv(export_path, index=False, header=header)

    return(None)