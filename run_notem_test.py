"""
Temporary file for testing notem
"""
from normits_demand.models import notem


# GLOBAL VARIABLES
year = [2018]
scenario = "NTEM"
lu_drive = "I:/"
by_iteration = "iter3d"
fy_iteration = "iter3b"
notem_import_home = r"I:\NorMITs Demand\import\NoTEM"
notem_export_home = r"C:\Data\Nirmal_Atkins"


def main():
    n = notem.NoTEM(
        years=year,
        scenario=scenario,
        land_use_import_home=lu_drive,
        by_land_use_iter=by_iteration,
        fy_land_use_iter=fy_iteration,
        notem_import_home=notem_import_home,
        notem_export_home=notem_export_home
    )
    n.run(
        generate_all=False,
        generate_hb=False,
        generate_hb_production=True,
        generate_hb_attraction=True,
        generate_nhb=False,
        generate_nhb_production=False,
        generate_nhb_attraction=False
    )


if __name__ == '__main__':
    main()
