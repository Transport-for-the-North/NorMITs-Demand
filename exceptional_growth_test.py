from demand_utilities import exceptional_growth as eg
from external_forecast_system import ExternalForecastSystem

efs = ExternalForecastSystem(
    model_name="norms_2015",
    iter_num=1,
    import_home="Y:/",
    export_home="C:/"
)

test_generations_path = (r"C:\Users\Monopoly\Documents\EFS\data"
                         r"\20201214 TfN EFS - Task D5 Bespoke Zone - "
                         "MANSAM Example Inputs v1b.xlsx")
eg.test_bespoke_zones(test_generations_path, efs.exports, efs.model_name, "",
                      recreate_donor=False)

# df = eg.test_growth_criteria()