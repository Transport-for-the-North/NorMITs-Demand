# -*- coding: utf-8 -*-
"""
    Module containing the test function for running the PopEmpComparator without
    running the whole EFS process.
"""

##### IMPORTS #####
# Standard imports
import os

# Local imports
from pop_emp_comparator import PopEmpComparator
from demand_utilities import utils as du
from external_forecast_system import ExternalForecastSystem


##### CONSTANTS #####
# Constants for running the test function
MODEL_NAME = "norms_2015"
ITER_NUM = 1
IMPORT_LOC = "Y:/"
EXPORT_LOC = "Y:/"
BASE_YEAR = "2018"


##### FUNCTIONS #####
def test():
    """Tests the PopEmpComparator class on data from previous run of EFS. """
    imports, exports, _ = du.build_io_paths(
        IMPORT_LOC,
        EXPORT_LOC,
        MODEL_NAME,
        f"iter{ITER_NUM}",
        ExternalForecastSystem.__version__,
        ExternalForecastSystem._out_dir,
    )
    # Population csv files, locations from ExternalForecastSysten.__init__ parameters
    population_value_file = "population/base_population_2018.csv"
    population_growth_file = "population/future_population_growth.csv"
    population_constraint_file = "population/future_population_values.csv"
    # Employment csv files
    worker_value_file = "employment/base_workers_2018.csv"
    worker_growth_file = "employment/future_workers_growth.csv"
    worker_constraint_file = "employment/future_workers_growth_values.csv"

    # Compare the population inputs and outputs
    pop_comp = PopEmpComparator(
        os.path.join(imports["default_inputs"], population_value_file),
        os.path.join(imports["default_inputs"], population_growth_file),
        os.path.join(imports["default_inputs"], population_constraint_file),
        os.path.join(exports["productions"], "MSOA_population.csv"),
        "population",
        BASE_YEAR,
        msoa_lookup_file=os.path.join(imports["zoning"], "msoa_zones.csv"),
        sector_grouping_file=os.path.join(
            imports["zoning"], "tfn_sector_msoa_pop_weighted_lookup.csv"
        ),
    )
    pop_comp.write_comparisons(exports["reports"], output_as="csv", year_col=True)
    pop_comp.write_comparisons(exports["reports"], output_as="excel", year_col=True)
    # Compare the employment inputs and outputs
    emp_comp = PopEmpComparator(
        os.path.join(imports["default_inputs"], worker_value_file),
        os.path.join(imports["default_inputs"], worker_growth_file),
        os.path.join(imports["default_inputs"], worker_constraint_file),
        os.path.join(exports["attractions"], "MSOA_employment.csv"),
        "employment",
        BASE_YEAR,
        msoa_lookup_file=os.path.join(imports["zoning"], "msoa_zones.csv"),
        sector_grouping_file=os.path.join(
            imports["zoning"], "tfn_sector_msoa_emp_weighted_lookup.csv"
        ),
    )
    emp_comp.write_comparisons(exports["reports"], output_as="csv", year_col=True)
    emp_comp.write_comparisons(exports["reports"], output_as="excel", year_col=True)
    return


##### MAIN #####
if __name__ == "__main__":
    test()
