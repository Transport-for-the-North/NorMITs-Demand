# -*- coding: utf-8 -*-
"""
Created on: Fri December 10 2021
Updated on:

Original author: Nirmal Kumar
Last update made by:
Other updates made by:

File purpose:
Master run file to run NorMITs Supply
"""
# Built-Ins
import sys

# Third Party

# Local Imports
sys.path.append("..")
from normits_demand.models.supply import NorMITsSupply

# import the local config
try:
    import supply_config
except ModuleNotFoundError as e:
    msg = (
        "Unable to import the config file. Has it been defined? "
        "Expecting a file named 'supply_config.py' in the current directory. "
        "If none exists, make a copy of 'supply_config.py-example', rename, "
        "and define all arguments in there."
    )
    raise ModuleNotFoundError(msg) from e

scenario_id = 2
users_id = 1
value_id = 2
skim_type_id = 1
time_period = "TS1"
export_home = r"I:\NorMITs Supply\Test"


def main():
    query_fname_parts = [
        supply_config.SUPPLY_SCHEMA_NAME,
        supply_config.SUPPLY_MASTER_TABLE,
    ]

    p = NorMITsSupply(
        user=supply_config.USERNAME,
        password=supply_config.PASSWORD,
        server=supply_config.DB_HOST_IP,
        database=supply_config.DB_NAME,
        port=supply_config.DB_HOST_PORT,
        query_fname='.'.join(query_fname_parts),
        scenario_id=scenario_id,
        users_id=users_id,
        value_id=value_id,
        skim_type_id=skim_type_id,
        time_period=time_period,
        export_home=export_home,
    )
    p.run(od_to_pa_conversion=False)


if __name__ == '__main__':
    main()
