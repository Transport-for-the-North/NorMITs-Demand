"""
File purpose: Reads landuse and trip rate vectors and creates corresonding demand vectors
Also creates production values by taking the product of landuse and trip rate
"""
import numpy as np
import pandas as pd
from normits_demand.utils import general as du
from datetime import datetime

now = datetime.now()

current_time = now.strftime("%H:%M:%S")
print("Starting to read files =", current_time)

population = pd.read_csv(r"C:\Data\NorMITS\land_use_output_tfn_msoa1.csv")
trip_rate = pd.read_csv(r"C:\Data\NorMITS\hb_trip_rates_normalised.csv")
#to be moved to efs_constants
target_cols = {
    'land_use': ['msoa_zone_id', 'area_type', 'tfn_traveller_type', 'people'],
    'trip_rate': ['tfn_traveller_type', 'area_type', 'p', 'trip_rate']
}
now = datetime.now()

current_time = now.strftime("%H:%M:%S")
print("Finished reading files =", current_time)

population = population.reindex(target_cols['land_use'], axis='columns')
population['segment'] = population['tfn_traveller_type'].astype(str) + "_" + population['area_type'].astype(str)
population['zone_at'] = population['msoa_zone_id'].astype(str) + "_" + population['area_type'].astype(str)
population_zone_at = population['zone_at'].drop_duplicates()
population_seg= population['segment'].drop_duplicates()

pop_array = population.reindex(columns=['msoa_zone_id'])
pop_array['people'] = 0.0
pop_array = pop_array.drop_duplicates()
x = pop_array.copy()

now = datetime.now()

current_time = now.strftime("%H:%M:%S")
print("Starting population dvector =", current_time)
##Starting population dvector
dvec_pop = dict()
for i in population_seg:
    print(i)
    d = population.loc[population['segment'] == i]
    f = d['msoa_zone_id'].values
    d = d['people'].values
    for j in range(len(d)):
        pop_array.at[pop_array[pop_array['msoa_zone_id'] == f[j]].index[0], 'people'] = d[j]
        pop_arr = pop_array.copy()

    dvec_pop[i] = pop_arr['people'].values
    pop_array['people'] = 0.0

trip_rate['segment'] = trip_rate['p'].astype(str) + "_" + trip_rate['tfn_traveller_type'].astype(str) + "_" + trip_rate['area_type'].astype(str)
p_tfntt = (trip_rate['p'].astype(str) + "_" + trip_rate['tfn_traveller_type'].astype(str)).drop_duplicates()
trip_rate_new = trip_rate['segment'].drop_duplicates()

now = datetime.now()

current_time = now.strftime("%H:%M:%S")
print("Starting trip rate dvector =", current_time)

dvec_trip = dict()
for j in trip_rate_new:
    print(j)
    e = trip_rate.loc[trip_rate['segment'] == j]
    e = e['trip_rate'].values
    dvec_trip[j] = e

now = datetime.now()

current_time = now.strftime("%H:%M:%S")
print("Starting inital production dvector =", current_time)

exp_dvec_pop = dict()

for k in trip_rate_new:
    print(k)
    k1 = str(k).split("_", 1)[1]
    for l in dvec_pop:
        if k1 == l:
            exp_dvec_pop[k] = np.multiply(dvec_pop[l], dvec_trip[k])

now = datetime.now()

current_time = now.strftime("%H:%M:%S")
print("Starting final production dvector =", current_time)

dvec_trips = dict()
c = 1
for m in p_tfntt:
    print(m)
    for o in exp_dvec_pop:
        o1 = o.rsplit("_", 1)[0]
        if o1 == m and c == 1:
            dvec_trips[m] = exp_dvec_pop[o]
            c += 1
        elif o1 == m and c > 1:
            dvec_trips[m] = np.add(dvec_trips[m], exp_dvec_pop[o])
    c = 1

now = datetime.now()

current_time = now.strftime("%H:%M:%S")
print("Finished production dvector =", current_time)
