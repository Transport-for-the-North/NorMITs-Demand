import pandas as pd

from demand_utilities import d_log_processor as dlog

print("Reading Inputs")
population = pd.read_csv(r"C:\NorMITs Demand\norms\v2_3-EFS_Output\iter0\Productions\population_pre_dlog.csv")
employment = pd.read_csv(r"C:\NorMITs Demand\norms\v2_3-EFS_Output\iter0\Attractions\employment_pre_dlog.csv")
base_year = "2018"
future_years = ["2033", "2035", "2050"]
d_log = r"C:\Users\Monopoly\Documents\EFS\data\dlog\dlog_residential.csv"
population_constraint = pd.read_csv(r"Y:\NorMITs Demand\inputs\default\population\future_population_values.csv")
constraint_required = [False, True]

population_constraint.rename({"model_zone_id": "msoa_zone_id"}, axis=1, inplace=True)

print("Applying DLOG")
population = dlog.apply_d_log_population(
                population=population,
                base_year=base_year,
                future_years=future_years,
                dlog_path=d_log,
                constraints=population_constraint,
                constraints_zone_equivalence=r"Y:\NorMITs Demand\inputs\default\zoning\lad_msoa_grouping.csv",
                household_factors=2.4,
                development_zone_lookup=r"C:\Users\Monopoly\Documents\EFS\data\dlog\development_msoa_lookup.csv",
                msoa_zones=r"C:\Users\Monopoly\Documents\EFS\data\zoning\msoa_zones.csv",
                perform_constraint=constraint_required[1]
            )

# employment = dlog.apply_d_log_population(
#                 population=employment,
#                 base_year=base_year,
#                 future_years=future_years,
#                 dlog_path=d_log,
#                 constraints=population_constraint,
#                 constraints_zone_equivalence=r"Y:\NorMITs Demand\inputs\default\zoning\lad_msoa_grouping.csv",
#                 household_factors=2.4,
#                 development_zone_lookup=r"C:\Users\Monopoly\Documents\EFS\data\dlog\development_msoa_lookup.csv",
#                 msoa_zones=r"C:\Users\Monopoly\Documents\EFS\data\zoning\msoa_zones.csv",
#                 perform_constraint=constraint_required[1]
#             )