# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 12:32:59 2020

Temporary script to recompile fusion outputs to target model matrices

@author: genie
"""

import matrix_processing as mp


norms_fusion_folder = 'Y:/NorMITs Synthesiser/Norms/iter3/Fusion Outputs/24hr Fusion PA Distributions'
norms_matrix_params = 'Y:/NorMITs Synthesiser/Norms/Model Zone Lookups/norms_pa_matrix_params.csv'

fusion = mp.compile_pa(pa_folder = norms_fusion_folder,
               compile_param_path = norms_matrix_params)