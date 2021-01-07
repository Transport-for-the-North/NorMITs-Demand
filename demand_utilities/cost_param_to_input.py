# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 11:14:38 2020

@author: ChristopherStorey
"""
import os

import pandas as pd

_cost_params = 'Y:/NorMITs Demand/import/scenarios/Future Scenarios Costs Inputs.xlsx'

_output_folder = r'Y:\NorMITs Demand\import\scenarios\Cost Inputs'


def build_cost_inputs(input_xl = _cost_params,
                      output_folder = _output_folder):
    """
    Builds a python parsable input file from A human readable input sheet.
        
    """

    xl_lookup = pd.ExcelFile(_cost_params)

    sheet_names = xl_lookup.sheet_names

    xl_dict = {}

    for sheet in sheet_names:
        dat = xl_lookup.parse(sheet,
                              header=None)
        dat = dat.fillna(0)

        syear = 0
        sscen = 0
    
        for col in list(dat)[3:]:
            print(dat[col])
            if dat[col][0] != 0:
                syear = dat[col][0]
            else:
                dat[col][0] = syear
            if dat[col][1] != 0:
                sscen = dat[col][1]
            else:
                dat[col][1] = sscen
    
        dat = dat.drop(2, axis=0)

        cap_cols = list(dat[2][3:])

        all_cols = ['year', 'scenario', 'mode']
        for col in cap_cols:    
            if type(col) == str:
                col = _replace_non_flat_string(col)
            all_cols.append(col)

        dat = dat.loc[:,3:]

        dat = dat.transpose()
        dat.columns = all_cols

        sheet_name = _replace_non_flat_string(sheet)

        xl_dict.update({sheet_name:dat})

    for name,dat in xl_dict.items():
        dat.to_csv(os.path.join(output_folder,
                                name+'.csv'), index=False)

def _replace_non_flat_string(string):
    """
    Tidy up col strings
    """
    new_string = string.lower(
                    ).replace(' ','_').replace(',','').replace(
                        '(','').replace(')','').replace('-','')
    return new_string