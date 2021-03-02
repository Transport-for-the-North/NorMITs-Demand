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

COMPRESSION_SUFFIX = '.pbz2'


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

# Note that NoRMS needs
NORMS_VDM_SEG_EXTERNAL = {
    'W_CA_ext': {'to': ['hb', 'nhb'], 'ca': [2], 'uc': 'commute'},
    'W_NCA_ext': {'to': ['hb', 'nhb'], 'ca': [1], 'uc': 'commute'},

    'EB_CA_ext': {'to': ['hb', 'nhb'], 'ca': [2], 'uc': 'business'},
    'EB_NCA_ext': {'to': ['hb', 'nhb'], 'ca': [1], 'uc': 'business'},

    'O_CA_ext': {'to': ['hb', 'nhb'], 'ca': [2], 'uc': 'other'},
    'O_NCA_ext': {'to': ['hb', 'nhb'], 'ca': [1], 'uc': 'other'},
}
