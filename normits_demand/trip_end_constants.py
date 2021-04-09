"""
Import constants for trip end models
"""

# BACKLOG: Check if these exist on import

NTEM_PA = 'I:/NorMITs Demand/import/ntem_constraints/ntem_pa_ave_wday_2018.csv'

TRIP_RATES = 'I:/NorMITs Synthesiser/import/trip_rates/tfn_hb_trip_rates_18_0620.csv'

TIME_SPLIT = 'Y:/NorMITs Synthesiser/import/trip_rates/tfn_hb_time_split_18_0620.csv'

MODE_SPLIT = 'I:/NorMITs Synthesiser/import/trip_rates/tfn_hb_mode_split_18_0620.csv'

AVE_TIME_SPLIT = 'I:/NorMITs Synthesiser/import/trip_rates/hb_ave_time_split.csv'

MSOA_LAD = 'I:/NorMITs Synthesiser/import/lad_to_msoa.csv'

HB_PURPOSE = [1, 2, 3, 4, 5, 6, 7, 8]

NHB_PURPOSE = [12, 13, 14, 15, 16, 18]

# From Sex_B01ID use in CTripEnd
# 3 = Children, 1 = Male, 2 = Female
TT_GENDER = {'traveller_type': list(range(1, 89)),
             'g': [3]*8 + [1]*40 + [2]*40}
