# -*- coding: utf-8 -*-
"""
    Module containing all the required constants for the
    `elasticity_calcs` modules.
"""

# Expected elasticity file names, found in elasticities folder
SEGMENTS_FILE = "elasticity_segments.csv"
ELASTICITIES_FILE = "elasticity_values.csv"
CONSTRAINTS_FOLDER = "constraint_matrices"

# Cost file name
COST_NAMES = "{mode}_costs_p{purpose}.csv"
# Expected columns in the cost files
COST_LOOKUP = {
    "rail": {
        "origin": "from_model_zone_id",
        "destination": "to_model_zone_id",
        "walk": "AE_cost",
        "wait": "Wait_Actual_cost",
        "ride": "IVT_cost",
        "fare": "fare_cost",
        "num_int": "Interchange_cost",
    },
    "car": {
        "origin": "from_model_zone_id",
        "destination": "to_model_zone_id",
        "time": "time",
        "dist": "distance",
        "toll": "toll",
    },
}


# Lookup for the elasticity types and what modes/costs they affect
GC_ELASTICITY_TYPES = {
    "Car_JourneyTime": ("car", "time"),
    "Car_FuelCost": ("car", "voc"),
    "Rail_Fare": ("rail", "fare"),
    "Rail_IVTT": ("rail", "ride"),
    "Bus_Fare": ("bus", "fare"),
    "Bus_IVTT": ("bus", "ride"),
    "Car_RUC": ("car", "gc"),
}

PURPOSES = ['commute', 'business', 'other']

# ID and zone system for each mode
MODE_ID = {"car": 3, "rail": 6}
MODE_ZONE_SYSTEM = {
    "car": "noham",
    "rail": "norms",
}
# List of modes which do not have demand data
OTHER_MODES = ["bus", "active", "no_travel"]

# Default factors for rail GC calculation
RAIL_GC_FACTORS = {"walk": 1.5, "wait": 2, "interchange_penalty": 5}

# Expected name of zone system lookup files (found in translation folder)
ZONE_LOOKUP_NAME = "{from_zone}_to_{to_zone}.csv"
# The zone system that the elasticity calculations are done in
COMMON_ZONE_SYSTEM = "norms"

# Tolerance for raising errors when comparing matrix totals
MATRIX_TOTAL_TOLERANCE = 1e-10
