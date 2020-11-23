import os

import pandas as pd

from demand_utilities import d_log_processor as dlog

print("Reading Inputs")
population = pd.read_csv(r"C:\NorMITs Demand\norms\v2_3-EFS_Output\iter0\Productions\population_pre_dlog.csv")
employment = pd.read_csv(r"C:\NorMITs Demand\norms\v2_3-EFS_Output\iter0\Attractions\employment_pre_dlog.csv")
base_year = "2018"
future_years = ["2033", "2035", "2050"]
d_log = r"C:\Users\Monopoly\Documents\EFS\data\dlog\201123_DevelopmentLogResidential.csv"
d_log_emp = r"C:\Users\Monopoly\Documents\EFS\data\dlog\201123_DevelopmentLogNonResidential.csv"
population_constraint = pd.read_csv(r"Y:\NorMITs Demand\inputs\default\population\future_population_values.csv")
constraint_required = [False, True]
audit_location = r"C:\Users\Monopoly\Documents\EFS\data\dlog"

population_constraint.rename({"model_zone_id": "msoa_zone_id"}, axis=1, inplace=True)

print("Applying DLOG")

pop_audit_loc = os.path.join(audit_location, "pop")
emp_audit_loc = os.path.join(audit_location, "emp")

if not os.path.isdir(pop_audit_loc):
    os.makedirs(pop_audit_loc)
if not os.path.isdir(emp_audit_loc):
    os.makedirs(emp_audit_loc)

population = dlog.apply_d_log(
                pre_dlog_df=population,
                base_year=base_year,
                future_years=future_years,
                dlog_path=d_log,
                constraints=population_constraint,
                constraints_zone_equivalence=r"Y:\NorMITs Demand\inputs\default\zoning\lad_msoa_grouping.csv",
                segment_cols=["soc", "ns", "ca"],
                dlog_conversion_factor=1.0,
                development_zone_lookup=r"C:\Users\Monopoly\Documents\EFS\data\dlog\development_msoa_lookup.csv",
                msoa_zones=r"C:\Users\Monopoly\Documents\EFS\data\zoning\msoa_zones.csv",
                dlog_data_column_key="population",
                perform_constraint=constraint_required[1],
                audit_location=pop_audit_loc
            )

employment = dlog.apply_d_log(
                pre_dlog_df=employment,
                base_year=base_year,
                future_years=future_years,
                dlog_path=d_log_emp,
                constraints=population_constraint,
                constraints_zone_equivalence=r"Y:\NorMITs Demand\inputs\default\zoning\lad_msoa_grouping.csv",
                segment_cols=["employment_cat"],
                dlog_conversion_factor=1.0,
                development_zone_lookup=r"C:\Users\Monopoly\Documents\EFS\data\dlog\development_msoa_lookup.csv",
                msoa_zones=r"C:\Users\Monopoly\Documents\EFS\data\zoning\msoa_zones.csv",
                dlog_data_column_key="employees",
                perform_constraint=constraint_required[1],
                audit_location=emp_audit_loc
            )