import os

from demand_utilities import exceptional_growth as eg
from external_forecast_system import ExternalForecastSystem
import efs_constants as consts

efs = ExternalForecastSystem(
    model_name=consts.MODEL_NAME,
    iter_num=0,
    import_home="Y:/",
    export_home="C:/",
    integrate_dlog=True,
    scenario_name=consts.SC00_NTEM
)

generation_file = (
    "20201217 TfN EFS - Task D5 Bespoke Zone - MANSAM Example Inputs v1b.xlsx"
)

test_generations_path = (
    os.path.join(r"Y:\NorMITs Demand\inputs\MANSAM", generation_file)
)
eg.adjust_bespoke_zones(test_generations_path,
                        efs.exports,
                        efs.model_name,
                        base_year=consts.BASE_YEAR_STR,
                        recreate_donor=True,
                        audit_path=efs.exports["audits"])
