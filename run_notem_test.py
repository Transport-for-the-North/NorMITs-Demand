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


def main():
    n = notem.NoTEM(
        years=year,
        scenario=scenario,
        land_use_import_drive=lu_drive,
        by_land_use_iter=by_iteration,
        fy_land_use_iter=fy_iteration,
    )
    n.run(
        generate_all_trip_ends=False,
        generate_hb_trip_ends=True,
        generate_nhb_trip_ends=False,
    )


if __name__ == '__main__':
    main()
