from normits_demand.efs_constants import *

# VALID VALUES FOR ARGS
VECTOR_TYPES = [
    'productions',
    'attractions',
    'origins',
    'destinations',
]


# STANDARD FILE NAMES
# i/e areas - add model_zone
INTERNAL_AREA = "%s_internal_area.csv"
EXTERNAL_AREA = "%s_external_area.csv"

BASE_YEAR_POP_FNAME = 'land_use_output_safe_msoa.csv'
BASE_YEAR_EMP_FNAME = 'land_use_2018_emp.csv'

COMPRESSION_SUFFIX = '.pbz2'


# TODO: Parse norms input names in a function to generate this!!!
NORMS_VDM_SEG_TO_NORMS_POSTME_NAMING = {
    'HB_W_CA_int': ['HBWCA_Int'],
    'HB_W_NCA_int': ['HBWNCA_Int'],
    'HB_EB_CA_int': ['HBEBCA_Int'],
    'HB_EB_NCA_int': ['HBEBNCA_Int'],
    'NHB_EB_CA_int': ['NHBEBCA_Int'],
    'NHB_EB_NCA_int': ['NHBEBNCA_Int'],
    'HB_O_CA_int': ['HBOCA_Int'],
    'HB_O_NCA_int': ['HBONCA_Int'],
    'NHB_O_CA_int': ['NHBOCA_Int'],
    'NHB_O_NCA_int': ['NHBONCA_Int'],

    'W_CA_ext': ['HBWCA_Ext_FM', 'HBWCA_Ext_TO'],
    'W_NCA_ext': ['HBWNCA_Ext'],
    'EB_CA_ext': ['EBCA_Ext_FM', 'EBCA_Ext_TO'],
    'EB_NCA_ext': ['EBNCA_Ext'],
    'O_CA_ext': ['OCA_Ext_FM', 'OCA_Ext_TO'],
    'O_NCA_ext': ['ONCA_Ext_TO'],
}


# SEGMENTATION AGGREGATION DICTIONARIES
NORMS_VDM_SEG_INTERNAL = {
    'HB_W_CA_int': {'to': ['hb'], 'ca': [2], 'uc': 'commute'},
    'HB_W_NCA_int': {'to': ['hb'], 'ca': [1], 'uc': 'commute'},

    'HB_EB_CA_int': {'to': ['hb'], 'ca': [2], 'uc': 'business'},
    'HB_EB_NCA_int': {'to': ['hb'], 'ca': [1], 'uc': 'business'},
    'NHB_EB_CA_int': {'to': ['nhb'], 'ca': [2], 'uc': 'business'},
    'NHB_EB_NCA_int': {'to': ['nhb'], 'ca': [1], 'uc': 'business'},

    'HB_O_CA_int': {'to': ['hb'], 'ca': [2], 'uc': 'other'},
    'HB_O_NCA_int': {'to': ['hb'], 'ca': [1], 'uc': 'other'},
    'NHB_O_CA_int': {'to': ['nhb'], 'ca': [2], 'uc': 'other'},
    'NHB_O_NCA_int': {'to': ['nhb'], 'ca': [1], 'uc': 'other'},
}

# Note that NoRMS needs CA splitting into to and from for the VDM
# See NORMS_VDM_SEG_TO_NORMS_POSTME_NAMING
NORMS_VDM_SEG_EXTERNAL = {
    'W_CA_ext': {'to': ['hb', 'nhb'], 'ca': [2], 'uc': 'commute'},
    'W_NCA_ext': {'to': ['hb', 'nhb'], 'ca': [1], 'uc': 'commute'},

    'EB_CA_ext': {'to': ['hb', 'nhb'], 'ca': [2], 'uc': 'business'},
    'EB_NCA_ext': {'to': ['hb', 'nhb'], 'ca': [1], 'uc': 'business'},

    'O_CA_ext': {'to': ['hb', 'nhb'], 'ca': [2], 'uc': 'other'},
    'O_NCA_ext': {'to': ['hb', 'nhb'], 'ca': [1], 'uc': 'other'},
}

NORMS_VDM_MATRIX_NAMES = list(NORMS_VDM_SEG_TO_NORMS_POSTME_NAMING.keys())

# USEFUL GEO CONSTANTS
# LAs in North area
GEO_AREAS = ['gb', 'north']

NORTH_LA = north_la = [
    'E06000001', 'E06000002', 'E06000003', 'E06000004', 'E06000005',
    'E06000006', 'E06000007', 'E06000008', 'E06000009', 'E06000010',
    'E06000011', 'E06000012', 'E06000013', 'E06000014', 'E06000021',
    'E06000047', 'E06000049', 'E06000050', 'E06000057', 'E07000026',
    'E07000027', 'E07000028', 'E07000029', 'E07000030', 'E07000031',
    'E07000033', 'E07000034', 'E07000035', 'E07000037', 'E07000038',
    'E07000117', 'E07000118', 'E07000119', 'E07000120', 'E07000121',
    'E07000122', 'E07000123', 'E07000124', 'E07000125', 'E07000126',
    'E07000127', 'E07000128', 'E07000137', 'E07000142', 'E07000163',
    'E07000164', 'E07000165', 'E07000166', 'E07000167', 'E07000168',
    'E07000169', 'E07000170', 'E07000171', 'E07000174', 'E07000175',
    'E07000198', 'E08000001', 'E08000002', 'E08000003', 'E08000004',
    'E08000005', 'E08000006', 'E08000007', 'E08000008', 'E08000009',
    'E08000010', 'E08000011', 'E08000012', 'E08000013', 'E08000014',
    'E08000015', 'E08000016', 'E08000017', 'E08000018', 'E08000019',
    'E08000021', 'E08000022', 'E08000023', 'E08000024', 'E08000032',
    'E08000033', 'E08000034', 'E08000035', 'E08000036', 'E08000037',
    'W06000001', 'W06000002', 'W06000003', 'W06000004', 'W06000005',
    'W06000006']

# TfN area type to aggregate area type
AGG_AT = {
    'tfn_area_type': [1, 2, 3, 4, 5, 6, 7, 8],
    'nelum_area_type': [1, 1, 1, 2, 2, 2, 3, 3],  # ?? Needs audit
    'agg_tfn_area_type': [1, 1, 2, 2, 3, 3, 4, 4]}
