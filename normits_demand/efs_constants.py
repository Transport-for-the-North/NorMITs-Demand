# -*- coding: utf-8 -*-
"""
Created on: Mon August 28 11:36:24 2020
Updated on:

Original author: Ben Taylor
Last update made by:
Other updates made by:

File purpose:
A file of constants to be used.
Keeps all constants in one place, for all files in the project to refer to,
and keeps the code more readable.
"""
# TODO: Re-organise constants
import os
from normits_demand import constants as consts

# ### Constant Values ### #

# General
SOC_P = [1, 2, 12]
NS_P = [3, 4, 5, 6, 7, 8, 13, 14, 15, 16, 18]

ALL_MODES = [1, 2, 3, 5, 6]

TIME_PERIODS = [1, 2, 3, 4]
TIME_PERIOD_STRS = ['tp' + str(x) for x in TIME_PERIODS]

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

# ## Zone Translations

# from_zone_system, to_zone_system
POP_TRANSLATION_FNAME = '%s_%s_pop_weighted_lookup.csv'
EMP_TRANSLATION_FNAME = '%s_%s_emp_weighted_lookup.csv'

# ## NTEM Controls
# matrix_format, year
NTEM_CONTROL_FNAME = 'ntem_%s_ave_wday_%s.csv'
DEFAULT_LAD_LOOKUP = 'lad_to_msoa.csv'

# TODO: can likely remove lots of EFS_COLUMN_DICTIONARY
EFS_COLUMN_DICTIONARY = {
    "base_year_population": [
        "model_zone_id",
        "base_year_population"
    ],
    "base_year_households": [
        "model_zone_id",
        "base_year_households"
    ],
    "base_year_workers": [
        "model_zone_id",
        "base_year_workers"
    ],
    "population": [
        "msoa_zone_id"
    ],
    "population_ratio": [
        "model_zone_id",
        "property_type_id",
        "traveller_type_id"
    ],
    "employment": [
        "msoa_zone_id"
    ],
    "housing_occupancy": [
        "model_zone_id",
        "property_type_id"
    ],
    "production_trips": [
        "p",
        "traveller_type",
        "soc",
        "ns",
        "area_type"
    ],
    "mode_split": [
        "area_type_id",
        "car_availability_id",
        "purpose_id",
        "mode_id",
    ],
    "mode_time_split": [
        "purpose_id",
        "traveller_type_id",
        "area_type_id",
        "mode_time_split"
    ],
    "employment_ratio": [
        "model_zone_id",
        "employment_class"
    ],
    "attraction_weights": [
        "employment_class",
        "purpose_id"
    ]
}

# ### Default Function Arguments ### #

# ## EFS_RUN ## #
# We don't constrain by default. Land Use provides us with accurate enough
# data. Unless we include D-log!
CONSTRAINT_REQUIRED_DEFAULT = {
    'pop_pre_dlog': False,
    'pop_post_dlog': None,      # Automatically turns on/off if dlog in use
    'emp_pre_dlog': False,
    'emp_post_dlog': None,      # Automatically turns on/off if dlog in use
}

TFN_MSOA_SECTOR_LOOKUPS = {
    "population": "tfn_sector_msoa_pop_weighted_lookup.csv",
    "employment": "tfn_sector_msoa_emp_weighted_lookup.csv"
}

# BACKLOG: Move EFS running constants into a config file instead of constants
#  labels: EFS, demand merge

# RUNNING CONSTANTS
MODEL_NAME = 'noham'

# YEARS
BASE_YEAR = 2021
FUTURE_YEARS = [2030, 2040]

# HB efs_consts
HB_PURPOSES_NEEDED = consts.ALL_HB_P
MODES_NEEDED = MODEL_MODES[MODEL_NAME]
SOC_NEEDED = [0, 1, 2, 3]
NS_NEEDED = [1, 2, 3, 4, 5]
CA_NEEDED = [1, 2]
TP_NEEDED = [1, 2, 3, 4]

# NHB efs_consts
NHB_PURPOSES_NEEDED = consts.ALL_NHB_P

ALL_PURPOSES_NEEDED = HB_PURPOSES_NEEDED + NHB_PURPOSES_NEEDED

# Built from running args
ALL_YEARS = [BASE_YEAR] + FUTURE_YEARS
BASE_YEAR_STR = str(BASE_YEAR)
FUTURE_YEARS_STR = [str(x) for x in FUTURE_YEARS]
ALL_YEARS_STR = [str(x) for x in ALL_YEARS]

# Bespoke zones
# BACKLOG: Move bespoke zone input into a config file
#  labels: EFS, QoL updates

BESPOKE_ZONES_INPUT_FILE = os.path.join(
    "I:/",
    "NorMITs Demand",
    "import",
    "bespoke zones",
    "MANSAM",
    "Bespoke Zone - MANSAM Inputs v1b.xlsx",
)



