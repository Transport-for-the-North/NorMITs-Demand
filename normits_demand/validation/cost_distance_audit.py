# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 10:29:57 2020

@author: genie
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

_24_cost_path = 'Y:/NorMITs Synthesiser/Norms/Model Zone Lookups/costs/norms_24hr_costs.csv'
_am_cost_path = 'Y:/NorMITs Synthesiser/Norms/Model Zone Lookups/norms_am_costs.csv'
_distance_path = 'Y:/NorMITs Synthesiser/Norms/Model Zone Lookups/norms_am_distances.csv'

# Cost audit

# Get 24 i2 costs
norms_24 = pd.read_csv(_24_cost_path)

# Get norms am costs
norms_am = pd.read_csv(_am_cost_path)
# Full p/a zone to index distance
full_pa = norms_am.reindex(['p_zone', 'a_zone'], axis=1).reset_index(drop=True)

# Get distance
norms_distance = pd.read_csv(_distance_path)
norms_distance = full_pa.merge(norms_distance,
                               how='left',
                               on=['p_zone', 'a_zone'])

# Correlations
print(list(norms_24))
print(list(norms_am))
print(list(norms_distance))

# 24 commute ca
_24_commute_ca = norms_24['hb_commute_ca'].values
_24_commute_ca = np.where(_24_commute_ca==-100000, 0, _24_commute_ca)

# Am commute ca
am_commute_ca = norms_am['commute_ca_from'].values

# 24 business ca
_24_business_ca = norms_24['hb_business_ca'].values
_24_business_ca = np.where(_24_business_ca==-100000, 0, _24_business_ca)

# Am business ca
am_business_ca = norms_am['business_ca_from'].values

# Distance commute ca
distance_commute_ca = norms_distance['commute_ca_from'].values
distance_commute_ca = np.where(np.isnan(distance_commute_ca), 0, distance_commute_ca)

# Coeffs
am_x_24_commute_ca = np.corrcoef(_24_commute_ca, am_commute_ca)
am_x_24_commute_ca_sq = am_x_24_commute_ca**am_x_24_commute_ca

am_x_24_business_ca =  np.corrcoef(_24_business_ca, am_business_ca)
am_x_24_business_ca_sq = am_x_24_business_ca**am_x_24_business_ca

am_x_distance_commute_ca = np.corrcoef(am_commute_ca, distance_commute_ca)
am_x_distance_commute_ca_sq = am_x_distance_commute_ca**am_x_distance_commute_ca

distance_x_24_commute_ca = np.corrcoef(distance_commute_ca, _24_commute_ca )
distance_x_24_commute_ca_sq = distance_x_24_commute_ca**distance_x_24_commute_ca

# Plots
print('am x 24 commute: r2')
print(am_x_24_commute_ca_sq[1][0])
plt.scatter(_24_commute_ca, am_commute_ca)
plt.show()

print('distance x am commute: r2')
print(am_x_distance_commute_ca_sq[1][0])
plt.scatter(distance_commute_ca, am_commute_ca)
plt.show()

print('distance x 24 commute: r2')
print(distance_x_24_commute_ca_sq[1][0])
plt.scatter(distance_commute_ca, _24_commute_ca)
plt.show()


## rename cols for use in i3
for col in list(norms_am):
    print(col)
    if col != 'p_zone':
        if col != 'a_zone':
            norms_am = norms_am.rename(columns={col:('hb_' + col)})

am_write_path = ('Y:/NorMITs Synthesiser/Norms/'+
                 'Model Zone Lookups/costs/am/norms_am_costs.csv')

norms_am.to_csv(am_write_path, index=False)
