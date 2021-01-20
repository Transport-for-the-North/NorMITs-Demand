import os

from demand_utilities import exceptional_growth as eg
from external_forecast_system import ExternalForecastSystem

efs = ExternalForecastSystem(
    model_name="norms",
    iter_num=4,
    import_home="Y:/",
    export_home="C:/"
)

generation_file = (
    "20201217 TfN EFS - Task D5 Bespoke Zone - MANSAM Example Inputs v1b.xlsx"
)

test_generations_path = (
    os.path.join(r"Y:\NorMITs Demand\inputs\MANSAM", generation_file)
)
eg.test_bespoke_zones(test_generations_path, 
                      efs.exports, 
                      efs.model_name,
                      recreate_donor=True,
                      audit_path=efs.exports["print_audits"])
