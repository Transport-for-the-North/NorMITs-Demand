# -*- coding: utf-8 -*-
"""
Created on Wed May  6 11:37:35 2020

@author: genie
"""

import os
import pandas as pd

path = 'Y:/NorMITs Synthesiser/Norms/iter4/Distribution Outputs/Compiled PA Matrices'

outputs = os.listdir(path)
nhb_outputs = [x for x in outputs if 'nhb' in x]

nca_factor = .18
ca_factor = .82

for nhbo in nhb_outputs:
    print(nhbo)
    ph = pd.read_csv(path + '/' + nhbo)

    ph_nca = ph.copy()
    ph_nca['dt'] = ph_nca['dt'] * nca_factor

    nca_path = nhbo
    nca_path = nca_path.replace('_tp', '_nca_tp')

    ph_ca = ph.copy()
    ph_ca['dt'] = ph_ca['dt'] * ca_factor

    ca_path = nhbo
    ca_path = ca_path.replace('_tp', '_ca_tp')

    ph_nca.to_csv((path + '/' + nca_path), index=False)
    ph_ca.to_csv((path + '/' + ca_path), index=False)