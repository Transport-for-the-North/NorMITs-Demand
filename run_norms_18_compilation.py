import os

import pandas as pd

import normits_demand.utils.utils as nup
import normits_demand.matrices.matrix_processing as mp
import normits_demand.constants as consts

# Sourced CA factors from utils in TMS

# compile norms to vdm segmentation
n18_path = r'I:\NorMITs Synthesiser\Norms\iter4\Distribution Outputs\PA Matrices'
opa = r'C:\Users\genie\Documents\liability'
opa2 = r'C:\Users\genie\Documents\liability2'
params = os.path.join(opa2, 'params')

mats = nup.parse_mat_output(list_dir=os.listdir(n18_path),
                            sep='_',
                            mat_type = 'pa',
                            file_format = '.csv',
                            file_name = 'file')

# Compile to distribution segments
asd = mp.compile_matrices(mat_import = n18_path,
                          mat_export=opa,
                          compile_params_path = os.path.join(opa, 'norms_compilation_params.csv'),
                          factor_pickle_path = None)

# Flat factor nhb - not amazing
nhb_file = mats[mats['trip_origin'] == 'nhb']
nhb_file = nhb_file['file']

for nhb in nhb_file:
    name = os.path.join(n18_path, nhb)
    print(name)
    mat = pd.read_csv(name, index_col=0)  # make it read one col off left

    mat_ca1 = mat*.18
    mat_ca2 = mat*.82

    # Keep indices
    mat_ca1.to_csv(os.path.join(opa, nhb.replace('tp', 'ca1_tp')))
    mat_ca2.to_csv(os.path.join(opa, nhb.replace('tp', 'ca2_tp')))

new_mats = nup.parse_mat_output(list_dir=os.listdir(opa),
                            sep='_',
                            mat_type = 'pa',
                            file_format = '.csv',
                            file_name = 'file')

new_mats.to_csv(os.path.join(opa, 'asdasdd.csv'))

new_compile_params = os.path.join(opa, 'new_compile_params.csv')


compile = mp.compile_norms_to_vdm(
    mat_pa_import = opa,
    # TODO(BT): Actually pass in OD here
    mat_od_import = opa,
    mat_export = opa2,
    params_export = params,
    year = '2018',
    pa_matrix_format='pa',
    od_to_matrix_format='pa',
    od_from_matrix_format='pa',
    internal_zones=list(range(1,1157)),
    external_zones=list(range(1157,1301))
    )

files = os.listdir(opa)

for f in files:
    if '.csv' in f and 'params' not in f:
        filepath = os.path.join(opa, f)
        print(filepath)
        newfilepath = filepath.replace('pa_', 'pa_yr2018_')
        shutil.move(filepath, newfilepath)
