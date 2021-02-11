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
    iter_num = 0
    import_home = "Y:/"
    export_home = "E:/"
    model_name = consts.MODEL_NAME

    auditor = nd.EfsAudits(
        iter_num=iter_num,
        model_name=model_name,
        scenario_name=scenario,
        import_home=import_home,
        export_home=export_home,
    )

    # ## RUN ## #
    # Compare HB and NHB P/A vectors to NTEM

    # Compare post-exceptional_growth P/A vectors to NTEM

    # Compare furnessed PA matrices to NTEM

    # Compare furnessed PA matrices to P/A vectors

    # Compare furnessed OD matrices to NTEM

    raise NotImplementedError


if __name__ == "__main__":
    main()
