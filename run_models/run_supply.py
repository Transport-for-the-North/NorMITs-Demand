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

from normits_demand.models.supply import NorMITsSupply

user = "prjt_normits_supply_ed@tfn-gis-server"
password = "prjt_normits_supply_ed"
host = "10.1.2.6"
port = "5432"
database = "gis_db"
query_fname = "prjt_normits_supply.skim_master"
scenario_id = 2
users_id = 1
value_id = 2
skim_type_id = 1
time_period = "TS1"
export_home = r"I:\NorMITs Supply\Test"


def main():
    p = NorMITsSupply(
        user=user,
        password=password,
        server=host,
        database=database,
        port=port,
        query_fname=query_fname,
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
