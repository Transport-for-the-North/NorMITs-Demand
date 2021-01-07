from demand_utilities import exceptional_growth as eg
from external_forecast_system import ExternalForecastSystem

efs = ExternalForecastSystem(
    model_name="norms",
    iter_num=4,
    import_home="Y:/",
    export_home="C:/"
)

test_generations_path = (r"C:\Users\Monopoly\Documents\EFS\data"
                         r"\20201214 TfN EFS - Task D5 Bespoke Zone - "
                         "MANSAM Example Inputs v1b.xlsx")
eg.test_bespoke_zones(test_generations_path, 
                      efs.exports, 
                      efs.model_name,
                      recreate_donor=True,
                      audit_path=efs.exports["print_audits"])

# df = eg.test_growth_criteria()