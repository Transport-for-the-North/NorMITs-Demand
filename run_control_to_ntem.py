

import pandas as pd

from normits_demand.constraints import ntem_control

from normits_demand.utils import file_ops
from normits_demand.utils import compress

def main():
    # Load df
    path = r"E:\NorMITs Demand\noham\v0.3-EFS_Output\NTEM\iter3d\Productions\msoa_raw_nhb_productions.csv"

    dtypes = {'p': int, 'm': int, 'tp': int}

    print("Reading!")
    df = pd.read_csv(path, dtype=dtypes)

    # Load in other files
    ntem_totals = pd.read_csv(r"I:\NorMITs Demand\import\ntem_constraints\ntem_pa_ave_wday_2018.csv")
    ntem_lad_lookup = pd.read_csv(r"I:\NorMITs Demand\import\zone_translation\no_overlap\lad_to_msoa.csv")
    ntem_control_cols = ['p', 'm', 'tp']
    year = '2018'
    trip_origin = 'nhb'

    # Control to NTEM
    print("Controlling")
    controlled = ntem_control.new_control_to_ntem(
        control_df=df,
        ntem_totals=ntem_totals,
        zone_to_lad=ntem_lad_lookup,
        constraint_cols=ntem_control_cols,
        constraint_dtypes=dtypes,
        base_value_name=year,
        ntem_value_name='productions',
        trip_origin=trip_origin
    )

    print("DONE! Controlled!")


if __name__ == '__main__':
    main()
