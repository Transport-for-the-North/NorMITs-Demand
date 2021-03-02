# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 11:26:27 2019
Script to apply car occupancies by model purpose and time period.
Also works the opposite way to get person trips from vehicle trips

@author: genie
"""
# Built ins
import os

# Third Party
import pandas as pd
from tqdm import tqdm

# Local imports
from normits_demand import efs_constants as consts


def people_vehicle_conversion(mat_import: str,
                              mat_export: str,
                              import_folder: str,
                              mode: int,
                              method: str = 'to_vehicles',
                              round_dp: int = consts.DEFAULT_ROUNDING,
                              out_format: str = 'long',
                              hourly_average: bool = True,
                              header: bool = True,
                              write: bool = True
                              ) -> None:
    # TODO: Write people_vehicle_conversion() docs

    # TODO: Add refactor up to totals after conversion
    if method not in ['to_vehicles', 'to_people']:
        raise ValueError('method should be to_vehicles or to_people')

    file_sys = os.listdir(mat_import)

    # Should be is in and take list
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

        # Do commute business and other separately
        mats = [x for x in internal_file if pl[0] in x]

        for mpt in tps:
            sub_co = c_o[c_o['trip_purpose'] == pl[1]]
            tp_int = int(mpt[-1])
            sub_co = sub_co[sub_co['time_period'] == tp_int]
            factor = sub_co['car_occupancy'].squeeze()

            tp_mat = [x for x in mats if mpt in x]

            # Get period factor
            p_factor = period_hours[tp_int]

            for mat_fname in tp_mat:
                in_path = os.path.join(mat_import, mat_fname)
                ph_mat = pd.read_csv(in_path, index_col=0)

                # For converting from people to vehicles hourly average refers
                # to the output matrix
                if method == 'to_vehicles':
                    ph_mat /= factor
                    if hourly_average:
                        ph_mat /= p_factor

                # For converting from vehicles to people hourly average refers
                # to the input OD matrix
                elif method == 'to_people':
                    ph_mat *= factor
                    if hourly_average:
                        ph_mat *= p_factor

                if out_format == 'long':
                    ph_mat = pd.melt(
                        ph_mat,
                        id_vars='o_zone',
                        var_name='d_zone',
                        value_name='dt',
                        col_level=0
                    )

                if write:
                    ph_mat = ph_mat.round(decimals=round_dp)
                    export_path = os.path.join(mat_export, mat_fname)
                    ph_mat.to_csv(export_path, header=header)
