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


# SEGMENTATION AGGREGATION DICTIONARIES
NORMS_VDM_SEG_INTERNAL = {
    'HB_W_CA': {'to': ['hb'], 'ca': [2], 'uc': 'commute'},
    'HB_W_NCA': {'to': ['hb'], 'ca': [1], 'uc': 'commute'},

    'HB_EB_CA': {'to': ['hb'], 'ca': [2], 'uc': 'business'},
    'HB_EB_NCA': {'to': ['hb'], 'ca': [1], 'uc': 'business'},
    'NHB_EB_CA': {'to': ['nhb'], 'ca': [2], 'uc': 'business'},
    'NHB_EB_NCA': {'to': ['nhb'], 'ca': [1], 'uc': 'business'},

    'HB_O_CA': {'to': ['hb'], 'ca': [2], 'uc': 'other'},
    'HB_O_NCA': {'to': ['hb'], 'ca': [1], 'uc': 'other'},
    'NHB_O_CA': {'to': ['nhb'], 'ca': [2], 'uc': 'other'},
    'NHB_O_NCA': {'to': ['nhb'], 'ca': [1], 'uc': 'other'},
}

# Note that NoRMS needs
NORMS_VDM_SEG_EXTERNAL = {
    'W_CA': {'to': ['hb', 'nhb'], 'ca': [2], 'uc': 'commute'},
    'W_NCA': {'to': ['hb', 'nhb'], 'ca': [1], 'uc': 'commute'},

    'EB_CA': {'to': ['hb', 'nhb'], 'ca': [2], 'uc': 'business'},
    'EB_NCA': {'to': ['hb', 'nhb'], 'ca': [1], 'uc': 'business'},

    'O_CA': {'to': ['hb', 'nhb'], 'ca': [2], 'uc': 'other'},
    'O_NCA': {'to': ['hb', 'nhb'], 'ca': [1], 'uc': 'other'},
}
