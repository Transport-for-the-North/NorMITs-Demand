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
import os

# ### Constant Values ### #

# General
SOC_P = [1, 2, 12]
NS_P = [3, 4, 5, 6, 7, 8]
ALL_HB_P = [1, 2, 3, 4, 5, 6, 7, 8]
ALL_NHB_P = [12, 13, 14, 15, 16, 18]
ALL_P = ALL_HB_P + ALL_NHB_P

ALL_MODES = [1, 2, 3, 5, 6]

TIME_PERIODS = [1, 2, 3, 4]
TIME_PERIOD_STRS = ['tp' + str(x) for x in TIME_PERIODS]

BASE_YEAR = 2018
FUTURE_YEARS = [2033, 2035, 2050]
ALL_YEARS = [BASE_YEAR] + FUTURE_YEARS
ALL_YEARS_STR = [str(x) for x in ALL_YEARS]

# ## VDM/ME2 constants ## #
VDM_TRIP_ORIGINS = ['hb', 'nhb']
USER_CLASSES = ['commute', 'business', 'other']

# How do user classes relate to purposes
USER_CLASS_PURPOSES = {
    'commute': [1],
    'business': [2, 12],
    'other': [3, 4, 5, 6, 7, 8, 13, 14, 15, 16, 18]
}

HB_USER_CLASS_PURPOSES = {
    'commute': [1],
    'business': [2],
    'other': [3, 4, 5, 6, 7, 8]
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
SEG_LEVELS = [
    'tms',      # Distribution segmentation - only p/m
    'me2',      # Compiled for Matrix Estimation. User classes + time periods
    'vdm',      # Similar to ME but split by trip origin 3 HB, and 2 NHB
    'tfn',      # Segment as much as possible - likely TfN segmentation
]

# Valid zoning systems
ZONING_SYSTEMS = [
    'msoa',
    'lad',
    'tfn_sector',
    'noham',
    'norms',
    'norms_2015'
]

# Valid model names
MODEL_NAMES = [
    'noham',
    'norms',
    'norms_2015'
]

# DIRECTORY NAMES
PRODUCTIONS_DIRNAME = 'Productions'
ATTRACTIONS_DIRNAME = 'Attractions'
NHB_PARAMS_DIRNAME = 'nhb_factors'

# HB consts
PURPOSES_NEEDED = [1, 2, 3, 4, 5, 6, 7, 8]
MODES_NEEDED = [6]
SOC_NEEDED = [0, 1, 2, 3]
NS_NEEDED = [1, 2, 3, 4, 5]
CA_NEEDED = [1, 2]
TP_NEEDED = [1, 2, 3, 4]

# NHB consts
NHB_PURPOSES_NEEDED = [12, 13, 14, 15, 16, 18]
NHB_FUTURE_YEARS = [2033, 2035, 2050]

VALID_MATRIX_FORMATS = ['pa', 'od']

TAG_CERTAINTY_BOUNDS = {
    "NC": ["NC"],
    "MTL": ["NC", "MTL"],
    "RF": ["NC", "MTL", "RF"],
    "H": ["NC", "MTL", "RF", "H"]
}

# ## File Names and Paths ## #
# Zone_system, trip_origin
HB_PRODS_FNAME = '%s_%s_productions.csv'
HB_ATTRS_FNAME = '%s_%s_attractions.csv'


NHB_PRODUCTIONS_FNAME = 'nhb_productions.csv'

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
        "model_zone_id"
    ],
    "population_ratio": [
        "model_zone_id",
        "property_type_id",
        "traveller_type_id"
    ],
    "households": [
        "model_zone_id"
    ],
    "employment": [
        "model_zone_id"
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

# ### Default Values ### #

# What is this area?
DEFAULT_ZONE_SUBSET = [1, 2, 1055, 1056, 1057, 1058, 1059, 1060, 1061, 1062]

# ### Default Function Arguments ### #

# ## EFS_RUN ## #

# By default EFS should no longer constrain to NTEM planning data
# for populations and employment!
# WE now trust NorMITs Land Use to provide us with accurate population
# and employment data
CONSTRAINT_REQUIRED_DEFAULT = [
    False,   # initial population metric constraint
    False,   # post-development constraint
    False,   # secondary post-development constraint used for matching HH pop
    False,  # initial worker metric constraint
    False,  # secondary worker metric constraint
    False,  # final trip based constraint
]

# ## Attraction Generation ## #
DEFAULT_ATTRACTION_CONSTRAINTS = [
    False,   # initial population metric constraint
    False,   # post-development constraint
    False,   # secondary post-development constraint used for matching HH pop
    False,  # initial worker metric constraint
    False,  # secondary worker metric constraint
    False,  # final trip based constraint
]

# ## Production Generations ## #
DEFAULT_PRODUCTION_CONSTRAINTS = [
    False,   # initial population metric constraint
    False,   # post-development constraint
    False,   # secondary post-development constraint used for matching HH pop
    False   # final trip based constraint
]



