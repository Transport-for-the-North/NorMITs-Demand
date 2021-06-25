"""
Temporary file for testing DVec - should be moved over to proper tests when there is time!
"""

# BACKLOG: Move run script for DVector into pytest
#  labels: core, testing

# Third party imports
import pandas as pd

import tqdm

# local imports
import normits_demand as nd

from normits_demand.models import notem_production_model as notem

# GLOBAL VARIABLES
# I Drive Path locations
POPULATION_PATH = {2018: r"I:\NorMITs Land Use\base_land_use\iter3b\outputs\land_use_output_tfn_msoa1.csv"}
TRIP_RATES_PATH = r"I:\Data\NTS\outputs\hb\hb_trip_rates\hb_trip_rates_normalised.csv"
MODE_TIME_SPLITS_PATH = r"I:\Data\NTS\outputs\hb\hb_time_mode_split_tfn_long.csv"
constraint_PATH = {2018: r"I:\NorMITs Land Use\base_land_use\iter3b\outputs\land_use_output_tfn_msoa1.csv"}
export_path = r"C:\Data\Nirmal_Atkins"

hb_prod = notem.HBProductionModel(POPULATION_PATH,TRIP_RATES_PATH,MODE_TIME_SPLITS_PATH,constraint_PATH,export_path)
hb = hb_prod.run()
