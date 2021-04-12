# -*- coding: utf-8 -*-
"""
Created on: Wed 10 15:28:32 2020
Updated on:

Original author: Ben Taylor
Last update made by:
Other updates made by:

File purpose:
Running automatic audits of EFS outputs to NTEM data and returning reports
"""

# local imports
import normits_demand as nd
from normits_demand import efs_constants as consts


def main():

    # ## SETUP ## #
    # Controls I/O
    scenario = consts.SC00_NTEM
    iter_num = '3g'
    import_home = "I:/"
    export_home = "F:/"
    model_name = consts.MODEL_NAME

    auditor = nd.EfsReporter(
        iter_num=iter_num,
        model_name=model_name,
        scenario_name=scenario,
        years_needed=consts.ALL_YEARS_STR,
        # years_needed=['2050'],
        import_home=import_home,
        export_home=export_home,
    )

    auditor.run(
        run_raw_vector_report=False,
        compare_trip_lengths=False,
    )


if __name__ == "__main__":
    main()
