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
from normits_demand.models import notem_attraction_model as notem_attr

# ## GLOBAL VARIABLES ## #
# PRODUCTIONS
POPULATION_PATH = {
    2018: r"I:\NorMITs Land Use\base_land_use\iter3d\outputs\land_use_output_msoa.csv",
    2033: r"I:\NorMITs Land Use\future_land_use\iter3d\outputs\scenarios\SC01_JAM\land_use_2033_pop.pbz2",
    2040: r"I:\NorMITs Land Use\future_land_use\iter3d\outputs\scenarios\SC01_JAM\land_use_2040_pop.pbz2",
    2050: r"I:\NorMITs Land Use\future_land_use\iter3d\outputs\scenarios\SC01_JAM\land_use_2050_pop.pbz2",
}
TRIP_RATES_PATH = r"I:\NorMITs Demand\import\NoTEM\HB_Productions\hb_trip_rates_v2.0.csv"
MODE_TIME_SPLITS_PATH = r"I:\NorMITs Demand\import\NoTEM\HB_Productions\hb_mode_time_split_v2.0.csv"

# p_export_path = r"C:\Data\Nirmal_Atkins\ss"
p_export_path = "E:/NoTEM/Productions"

# ATTRACTIONS
attraction_path = {
    2018: r"I:\NorMITs Land Use\base_land_use\iter3d\outputs\land_use_2018_emp.csv",
    2033: r"I:\NorMITs Land Use\future_land_use\iter3d\outputs\scenarios\SC01_JAM\land_use_2033_emp.pbz2",
    2040: r"I:\NorMITs Land Use\future_land_use\iter3d\outputs\scenarios\SC01_JAM\land_use_2040_emp.pbz2",
    2050: r"I:\NorMITs Land Use\future_land_use\iter3d\outputs\scenarios\SC01_JAM\land_use_2050_emp.pbz2",
}
attr_trip_rates_path = r"I:\NorMITs Demand\import\NoTEM\Attractions\sample_attraction_trip_rate.csv"
attr_mode_splits_path = r"I:\NorMITs Demand\import\NoTEM\Attractions\attraction_mode_split_new_infill.csv"

# pure_demand_production = r"C:\Data\Nirmal_Atkins\hb_msoa_pure_demand_2018_dvec.pkl"
pure_demand_production = {2018: r"E:\NoTEM\Productions\hb_msoa_notem_segmented_2018_dvec.pkl"}

# attr_export_path = r"C:\Data\Nirmal_Atkins\Attractions"
attr_export_path = "E:/NoTEM/Attractions"

# NHB PRODUCTIONS
# nhbp_export_path = r"C:\Data\Nirmal_Atkins\NHB_Productions"
# hb_attractions = {2018: r"C:\Data\Nirmal_Atkins\Attractions\hb_msoa_notem_segmented_2018_dvec.pkl"}

nhbp_export_path = r"E:/NoTEM/NHB_Productions"
hb_attractions = {2018: r"E:/NoTEM/Attractions/hb_msoa_notem_segmented_2018_dvec.pkl"}

nhb_prod_trip_rates = r"I:\NorMITs Demand\import\NoTEM\NHB_Productions\nhb_ave_wday_enh_trip_rates_v1.5.csv"
nhb_prod_time_splits = r"I:\NorMITs Demand\import\NoTEM\NHB_Productions\tfn_nhb_ave_week_time_split_18_v1.5.csv"

# NHB ATTRACTIONS
# nhb_prod = {2018: r"C:\Data\Nirmal_Atkins\NHB_Productions\nhb_msoa_notem_segmented_2018_dvec.pkl"}
# nhba_export_path = r"C:\Data\Nirmal_Atkins\NHB_Attractions"

nhb_prod = {2018: r"E:/NoTEM/NHB_Productions\nhb_msoa_notem_segmented_2018_dvec.pkl"}
nhba_export_path = r"E:\NoTEM\NHB_Attractions"



def nhb_attr_main():
    nhb_attr = notem_attr.NHBAttractionModel(
        hb_attractions,
        nhb_prod,
        nhba_export_path
    )

    nhb_attr.run(
        export_nhb_pure_attractions=True,
        export_notem_segmentation=True,
        export_reports=True,
        verbose=True,
    )


def nhb_prod_main():
    nhb_prod = notem.NHBProductionModel(
        hb_attractions_paths=hb_attractions,
        land_use_paths=POPULATION_PATH,
        nhb_trip_rates_path=nhb_prod_trip_rates,
        nhb_time_splits_path=nhb_prod_time_splits,
        export_path=nhbp_export_path,
    )

    nhb_prod.run(
        export_nhb_pure_demand=True,
        export_fully_segmented=False,
        export_notem_segmentation=True,
        export_reports=True,
        verbose=True,
    )


def main():
    hb_prod = notem.HBProductionModel(
        POPULATION_PATH,
        TRIP_RATES_PATH,
        MODE_TIME_SPLITS_PATH,
        p_export_path
    )

    hb_prod.run(
        export_pure_demand=True,
        export_fully_segmented=False,
        export_notem_segmentation=True,
        export_reports=True,
        verbose=True,
    )


def main_attr():
    hb_attr = notem_attr.HBAttractionModel(
        land_use_paths=attraction_path,
        control_production_paths=pure_demand_production,
        attraction_trip_rates_path=attr_trip_rates_path,
        mode_splits_path=attr_mode_splits_path,
        export_path=attr_export_path,
    )

    hb_attr.run(
        export_pure_attractions=True,
        export_fully_segmented=False,
        export_reports=True,
        verbose=True,
    )


if __name__ == '__main__':
    main()
    # main_attr()
    # nhb_prod_main()
    # nhb_attr_main()
