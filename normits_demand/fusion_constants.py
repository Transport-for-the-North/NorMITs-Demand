"""
Created on: Mon 23 August 2021
Updated on:

Original author: Peter Morris
Last update made by:
Other updates made by: Paul Woodfinden

File purpose:
A file of constants to be used within MDD fusion.
Keeps all constants in one place, for all files in the project to refer to,
and keeps the code more readable.
"""
# TODO: add constants from efs
import os  # whats this here for

# ### Constant Values ### #
# Taken from efs_constants.py
# General
SOC_P = [1, 2, 12]
NS_P = [3, 4, 5, 6, 7, 8, 13, 14, 15, 16, 18]
ALL_HB_P = [1, 2, 3, 4, 5, 6, 7, 8]
ALL_NHB_P = [12, 13, 14, 15, 16, 18]
ALL_P = ALL_HB_P + ALL_NHB_P

ALL_MODES = [1, 2, 3, 5, 6]

TIME_PERIODS = [1, 2, 3, 4]
TIME_PERIOD_STRS = ['tp' + str(x) for x in TIME_PERIODS]

# Trip origins to purpose
_trip_origin_purposes = [
    ('hb', ALL_HB_P),
    ('nhb', ALL_NHB_P),
]
TRIP_ORIGINS = [x[0] for x in _trip_origin_purposes]
TRIP_ORIGIN_TO_PURPOSE = {to: p for to, p in _trip_origin_purposes}

PROCESS_COUNT = -2
DEFAULT_ROUNDING = 8

# ## VDM/ME2 constants ## #
VDM_TRIP_ORIGINS = ['hb', 'nhb']
USER_CLASSES = ['commute', 'business', 'other']

NORMS_SUB_USER_CLASS_SEG = {
    'ca_from':  {'to': ['hb', 'nhb'],   'ca': [2], 'od_ft': ['od_from']},
    'ca_to':    {'to': ['hb'],          'ca': [2], 'od_ft': ['od_to']},
    'nca':      {'to': ['hb', 'nhb'],   'ca': [1], 'od_ft': ['od_to', 'od_from']},
}
# Convert between Purpose int and strings
_p_str_int = [
    ('Commute',             1),
    ('Business',            2),
    ('Education',           3),
    ('Shopping',            4),
    ('Personal_business',   5),
    ('Recreation_social',   6),
    ('Visiting_friends',    7),
    ('Holiday_day_trip',    8),
]
P_STR2INT = {s: i for s, i in _p_str_int}
P_INT2STR = {i: s for s, i in _p_str_int}

# Valid levels of segmentation
SEG_LEVEL_COLS = {
    'tms': ['p', 'tp'],             # Distribution segmentation - only p/m
    'me2': ['uc', 'tp'],            # Compiled for Matrix Estimation. User classes + time periods
    'vdm': ['uc', 'tp'],            # Similar to ME but split by trip origin 3 HB, and 2 NHB
    'tfn': ['p', 'm', 'soc', 'ns', 'ca', 'tp'],  # Segment as much as possible - likely TfN segmentation
}

SEG_LEVELS = list(SEG_LEVEL_COLS.keys())

# Valid zoning systems
ZONING_SYSTEMS = [
    'msoa',
    'lad',
    'tfn_sector',
    'noham',
    'norms',
    'norms_2015'
]

ZONE_SYSTEM_ZONE_COUNTS = {
    'norms': 1300,
    'noham': 2770,
}

# Valid model names
_model_name_modes = [
    ('noham',       [3]),
    ('norms',       [6]),
    ('norms_2015',  [6]),
]
MODEL_NAMES = [x[0] for x in _model_name_modes]
MODEL_MODES = {name: modes for name, modes in _model_name_modes}

# Order of segmentation in outputs
SEGMENTATION_ORDER = [
    'yr',
    'p',
    'm',
    'soc',
    'ns',
    'ca',
    'tp',
]

# Valid Scenario Names
SC00_NTEM = 'NTEM'
SC01_JAM = 'SC01_JAM'
SC02_PP = 'SC02_PP'
SC03_DD = 'SC03_DD'
SC04_UZC = 'SC04_UZC'

TFN_SCENARIOS = [
    SC01_JAM,
    SC02_PP,
    SC03_DD,
    SC04_UZC
]
SCENARIOS = [SC00_NTEM] + TFN_SCENARIOS

# DIRECTORY NAMES
AUDITS_DIRNAME = 'Audits'
PRODUCTIONS_DIRNAME = 'Productions'
ATTRACTIONS_DIRNAME = 'Attractions'
NHB_PARAMS_DIRNAME = 'nhb_factors'

PA_MATRIX_FORMATS = ['pa']
OD_MATRIX_FORMATS = ['od', 'od_to', 'od_from']
VALID_MATRIX_FORMATS = PA_MATRIX_FORMATS + OD_MATRIX_FORMATS


TAG_CERTAINTY_BOUNDS = {
    "NC": ["NC"],
    "MTL": ["NC", "MTL"],
    "RF": ["NC", "MTL", "RF"],
    "H": ["NC", "MTL", "RF", "H"]
}

# ## File Names and Paths ## #
# Zone_system, trip_origin, year
PRODS_FNAME_YEAR = '%s_%s_%d_productions.csv'
ATTRS_FNAME_YEAR = '%s_%s_%d_attractions.csv'

# Zone_system, trip_origin
PRODS_FNAME = '%s_%s_productions.csv'
ATTRS_FNAME = '%s_%s_attractions.csv'

ORIGS_FNAME = '%s_%s_origins.csv'
DESTS_FNAME = '%s_%s_destinations.csv'

# Year
LU_POP_FNAME = 'land_use_%s_pop.csv'
LU_EMP_FNAME = 'land_use_%s_emp.csv'

# Additive growth audit
PRODS_MG_FNAME = '%s_%s_productions_multiplicative_growth.csv'
ATTRS_MG_FNAME = '%s_%s_attractions_multiplicative_growth.csv'

# zone_system
POP_FNAME = '%s_population.csv'
EMP_FNAME = '%s_employment.csv'

EG_FNAME = "exceptional_zones.csv"

# ## Zone Translations ## #
# from_zone_system, to_zone_system
POP_TRANSLATION_FNAME = '%s_%s_pop_weighted_lookup.csv'
EMP_TRANSLATION_FNAME = '%s_%s_emp_weighted_lookup.csv'

# ## NTEM Controls ## #
# matrix_format, year
NTEM_CONTROL_FNAME = 'ntem_%s_ave_wday_%s.csv'
DEFAULT_LAD_LOOKUP = 'lad_to_msoa.csv'