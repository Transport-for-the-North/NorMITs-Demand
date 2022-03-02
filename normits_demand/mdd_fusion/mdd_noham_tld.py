# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 15:32:36 2021

@author: WoodfiP

TODO:
    2. Remove short distance trips in 2km bands and produce trip and tripend plots with regression
    All done without E-E
    Can re-use a lot of MDD check
    
    All run for cars by purpose and time period
"""

import pandas as pd
import numpy as np
import pickle as pk
import matplotlib.pyplot as plt

# Locations
fusion_loc = 'Y:/Mobile Data/Processing/3-2_Fusion_ME/TS1/AM_MELoop6/'
noham_loc = 'Y:/Mobile Data/Processing/1-3_NoHAM_ME/TS1/AM_MELoop6/'
export_loc = 'Y:/Mobile Data/Processing/Fusion_Check/'

lad_loc = 'Y:/Mobile Data/Processing/MDD_Check/lookups/lads20_to_noham_correspondence.csv'
noham_internal_loc = 'Y:/Mobile Data/Processing/MDD_Check/lookups/noham_internal.csv'
# sector_loc = 'Y:/Mobile Data/Processing/MDD_Check/lookups/sector_to_noham_correspondence.csv'
distance_loc = 'Y:/NoHAM/17.TUBA_Runs/-TPT/Skims/RefDist_Skims/NoHAM_Base_2018_TS2_v106_Dist_Other.csv'

# Import
zone_int = pd.read_csv(noham_internal_loc,
                       names=['zone', 'internal', 'internal_scot'],
                       skiprows=1)

distance = pd.read_csv(distance_loc,
                       names=['o_zone', 'd_zone', 'dist_km'])

lad_cor = pd.read_csv(lad_loc,
                      names=['lad', 'zone', 'lads_f', 'zone_f', 'internal'],
                      skiprows=1)
lad_cor = lad_cor.drop('lads_f', axis=1)

# TLD distance band
tldDist = [0,1,2,3,4,5,6,7,8,9,10,12.5,15,17.5,20,25,30,35,40,50,75,100,150,200,250,300,400,600,999]
tldLabels = list(range(1, 29))
tldBand = {'band': [], 'lower': [], 'upper': []}

for row in zip(range(1, len(tldDist)), tldDist[:-1], tldDist[1:]):
    tldBand['band'].append(row[0])
    tldBand['lower'].append(row[1])
    tldBand['upper'].append(row[2])

# Distance - add intrazonals (half minimum distance)
intra = distance.groupby(['o_zone'])['dist_km'].agg('min').reset_index()
intra['d_zone'] = intra['o_zone']
intra['dist_km'] = intra['dist_km'] / 2
intra = intra[['o_zone', 'd_zone', 'dist_km']]
distance = pd.concat([distance, intra], ignore_index=True)

# zone list
unq_zones = list(range(1, 2771))

# Dictionary layers
dctmode = {3: ['Car']}
dctday = {1: ['Weekday']}
dctpurp = {1: ['business'], 2: ['commute'], 3: ['other']}
dcttp = {1: ['AM']}


# Zone-zone regression plot
def zz_reg(mat,
           pp,
           md,
           tp,
           scen):
    # add linear regression for zone-zone
    d = np.polyfit(mat['nmtrip'], mat['mddtrip'], 1)
    f = np.poly1d(d)
    mat['linearfit'] = f(mat['nmtrip'])
    # R^2 value
    res = mat['nmtrip'].sub(mat['mddtrip']).pow(2).sum()
    tot = mat['nmtrip'].sub(mat['nmtrip'].mean()).pow(2).sum()
    r2 = 1 - res / tot

    # plot and export
    ax = mat.plot.scatter(y='mddtrip', x='nmtrip',
                          title='Regression: y = ' + str(d[0].round(2)) + 'x + ' + str(d[0].round(1)) +
                                ', R^2 = ' + str(r2.round(2)))
    mat.plot(x='nmtrip', y='linearfit', color='Red', ax=ax,
             xlabel='NoHAM trips',
             ylabel='MDD trips')
    ax.figure.savefig(f'Y:/Mobile Data/Processing/MDD_Check/Short_distance_ex/zone_od_p{pp}_m{md}_tp{tp}_{scen}.png')
    plt.close('all')


# TripEnds analysis
# Zone TripEnds
def te_ans(mat,
           pp,
           md,
           tp,
           scen):
    o_trips = mat.groupby(['o_zone']).agg({'mddtrip': sum, 'nmtrip': sum}).reset_index()
    o_trips = o_trips.rename(columns={'mddtrip': 'o_mdd', 'nmtrip': 'o_nm'})
    d_trips = mat.groupby(['d_zone']).agg({'mddtrip': sum, 'nmtrip': sum}).reset_index()
    d_trips = d_trips.rename(columns={'mddtrip': 'd_mdd', 'nmtrip': 'd_nm'})
    # Join
    zone_te = pd.merge(o_trips,
                       d_trips,
                       left_on=['o_zone'],
                       right_on=['d_zone'])
    zone_te = zone_te[['o_zone',
                       'o_mdd',
                       'd_mdd',
                       'o_nm',
                       'd_nm']]
    zone_te = zone_te.rename(columns={'o_zone': 'zone'})

    # Tripend plots
    # Origins
    # add linear regression for zone-zone
    d = np.polyfit(zone_te['o_nm'], zone_te['o_mdd'], 1)
    f = np.poly1d(d)
    zone_te['linearfit'] = f(zone_te['o_nm'])
    # R^2 value
    res = zone_te['o_nm'].sub(zone_te['o_mdd']).pow(2).sum()
    tot = zone_te['o_nm'].sub(zone_te['o_mdd'].mean()).pow(2).sum()
    r2 = 1 - res / tot
    # plot and export
    ax = zone_te.plot.scatter(y='o_mdd', x='o_nm',
                          title='Regression: y = ' + str(d[0].round(2)) + 'x + ' + str(d[0].round(1)) +
                                ', R^2 = ' + str(r2.round(2)))
    zone_te.plot(x='o_nm', y='linearfit', color='Red', ax=ax,
             xlabel='NoHAM trips',
             ylabel='MDD trips')
    ax.figure.savefig(
        f'Y:/Mobile Data/Processing/MDD_Check/Short_distance_ex/zone_TE_origin_p{pp}_m{md}_tp{tp}_{scen}.png')
    # Dests
    # add linear regression for zone-zone
    d = np.polyfit(zone_te['d_nm'], zone_te['d_mdd'], 1)
    f = np.poly1d(d)
    zone_te['linearfit'] = f(zone_te['d_nm'])
    # R^2 value
    res = zone_te['d_nm'].sub(zone_te['d_mdd']).pow(2).sum()
    tot = zone_te['d_nm'].sub(zone_te['d_mdd'].mean()).pow(2).sum()
    r2 = 1 - res / tot
    # plot and export
    ax = zone_te.plot.scatter(y='d_mdd', x='d_nm',
                              title='Regression: y = ' + str(d[0].round(2)) + 'x + ' + str(d[0].round(1)) +
                                    ', R^2 = ' + str(r2.round(2)))
    zone_te.plot(x='d_nm', y='linearfit', color='Red', ax=ax,
                 xlabel='NoHAM trips',
                 ylabel='MDD trips')
    ax.figure.savefig(
        f'Y:/Mobile Data/Processing/MDD_Check/Short_distance_ex/zone_TE_dest_p{pp}_m{md}_tp{tp}_{scen}.png')
    # Close plots
    plt.close('all')


# LAD sectors, origins then dests for many-many relationship
def lad_ans(mat,
           pp,
           md,
           tp,
           scen):
    mat_lad = pd.merge(mat,
                       lad_cor,
                       left_on=['o_zone'],
                       right_on=['zone'])
    mat_lad['mdd_lad'] = mat_lad['mddtrip'] * mat_lad['zone_f']
    mat_lad['nm_lad'] = mat_lad['nmtrip'] * mat_lad['zone_f']
    mat_lad = mat_lad.groupby(['lad', 'd_zone', 'internal']).agg({'mdd_lad': sum, 'nm_lad': sum}).reset_index()
    mat_lad = mat_lad.rename(columns={'lad': 'o_lad', 'internal': 'o_int'})

    mat_lad = pd.merge(mat_lad,
                       lad_cor,
                       left_on=['d_zone'],
                       right_on=['zone'])
    mat_lad['mdd_lad'] = mat_lad['mdd_lad'] * mat_lad['zone_f']
    mat_lad['nm_lad'] = mat_lad['nm_lad'] * mat_lad['zone_f']
    mat_lad = mat_lad.groupby(['o_lad', 'lad', 'o_int', 'internal']).agg({'mdd_lad': sum, 'nm_lad': sum}).reset_index()
    mat_lad = mat_lad.rename(columns={'lad': 'd_lad', 'internal': 'd_int'})

    # LAD tripends
    o_trips = mat_lad.groupby(['o_lad', 'o_int']).agg({'mdd_lad': sum, 'nm_lad': sum}).reset_index()
    o_trips = o_trips.rename(columns={'mdd_lad': 'o_mdd', 'nm_lad': 'o_nm'})
    d_trips = mat_lad.groupby(['d_lad', 'd_int']).agg({'mdd_lad': sum, 'nm_lad': sum}).reset_index()
    d_trips = d_trips.rename(columns={'mdd_lad': 'd_mdd', 'nm_lad': 'd_nm'})
    # Join
    lad_te = pd.merge(o_trips,
                      d_trips,
                      left_on=['o_lad'],
                      right_on=['d_lad'])
    lad_te = lad_te[['o_lad',
                     'o_int',
                     'o_mdd',
                     'd_mdd',
                     'o_nm',
                     'd_nm']]
    lad_te = lad_te.rename(columns={'o_lad': 'lad', 'o_int': 'internal'})
    # Export as csv
    lad_te.to_csv(f'Y:/Mobile Data/Processing/MDD_Check/Short_distance_ex/lad_te_od_p{pp}_m{md}_tp{tp}_{scen}.csv',
                  index=False)

    # add LAD-LAD linear regression
    d = np.polyfit(mat_lad['nm_lad'], mat_lad['mdd_lad'], 1)
    f = np.poly1d(d)
    mat_lad['linearfit'] = f(mat_lad['nm_lad'])
    # R^2 value
    res = mat_lad['nm_lad'].sub(mat_lad['mdd_lad']).pow(2).sum()
    tot = mat_lad['nm_lad'].sub(mat_lad['nm_lad'].mean()).pow(2).sum()
    r2 = 1 - res / tot
    # plot and export
    ax = mat_lad.plot.scatter(y='mdd_lad', x='nm_lad',
                              title='Regression: y = ' + str(d[0].round(2)) + 'x + ' + str(d[0].round(1)) +
                                    ', R^2 = ' + str(r2.round(2)))
    mat_lad.plot(x='nm_lad', y='linearfit', color='Red', ax=ax,
                 xlabel='NoHAM LAD trips',
                 ylabel='MDD LAD trips')
    ax.figure.savefig(f'Y:/Mobile Data/Processing/MDD_Check/Short_distance_ex/lad_od_p{pp}_m{md}_tp{tp}_{scen}.png')
    # Close all plots
    plt.close('all')

dband_list = []

# Loop over purpose and time periods
for md in dctmode:
    for wd in dctday:
        for pp in dctpurp:
            for tp in dcttp:
                # Matrix names
                fusion_nm = f'Fusion_v3-6_tp{tp}_I6{pp}.csv'
                noham_nm = f'NoHAM_v3_tp{tp}_I6{pp}.csv'

                # Import Matrices
                fusion = np.genfromtxt(fusion_loc + fusion_nm, delimiter=',')
                noham = np.genfromtxt(noham_loc + noham_nm, delimiter=',')

                # Convert to pandas
                wide_fs = pd.DataFrame(fusion,
                                       index=unq_zones,
                                       columns=unq_zones).reset_index()
                long_fs = pd.melt(wide_fs,
                                  id_vars=['index'],
                                  var_name='d_zone',
                                  value_name='fstrip',
                                  col_level=0)
                long_fs = long_fs.rename(columns={'index': 'o_zone'})

                wide_nm = pd.DataFrame(noham,
                                       index=unq_zones,
                                       columns=unq_zones).reset_index()
                long_nm = pd.melt(wide_nm,
                                  id_vars=['index'],
                                  var_name='d_zone',
                                  value_name='nmtrip',
                                  col_level=0)
                long_nm = long_nm.rename(columns={'index': 'o_zone'})

                # Join MDD and NoHAM
                mat = pd.merge(long_fs,
                               long_nm,
                               on=['o_zone', 'd_zone'])

                ######
                # Trip length distribution
                # join distance
                mat = pd.merge(mat,
                               distance,
                               on=['o_zone', 'd_zone'])
                # join internal marker
                mat = pd.merge(mat,
                               zone_int[['zone', 'internal_scot']],
                               left_on=['o_zone'],
                               right_on=['zone'])
                mat = mat.rename(columns={'internal_scot': 'o_int'})
                mat = pd.merge(mat,
                               zone_int[['zone', 'internal_scot']],
                               left_on=['d_zone'],
                               right_on=['zone'])
                mat = mat.rename(columns={'internal_scot': 'd_int'})
                mat.drop(['zone_x', 'zone_y'], axis=1, inplace=True)
                # Remove external-external
                mat = mat[(mat["o_int"] == 1) | (mat["d_int"] == 1)]
                # cut for distance bands
                mat['dband'] = pd.cut(mat['dist_km'], bins=tldDist, right=False, labels=tldLabels)

                # Produce trip plots
                scen = 'all'
                # zone-zone linear regression
                #zz_reg(mat, pp, md, tp, scen)
                # Zone tripends
                #te_ans(mat, pp, md, tp, scen)
                # LAD
                #lad_ans(mat, pp, md, tp, scen)

                # TLD
                distbands1 = mat.groupby(['dband', 'o_int', 'd_int']).agg({'fstrip': sum, 'nmtrip': sum}).reset_index()
                distbands = mat.groupby(['dband']).agg({'fstrip': sum, 'nmtrip': sum}).reset_index()
                distbands['fusion_pct'] = distbands['fstrip'] / distbands['fstrip'].sum()
                distbands['noham_pct'] = distbands['nmtrip'] / distbands['nmtrip'].sum()
                # plot and export
                ax = distbands.plot(x='dband', y='noham_pct', color='Blue',
                                          title='Trip Length Distribution Post-ME')
                distbands.plot(x='dband', y='fusion_pct', color='Red', linestyle='dashed', ax=ax,
                             xlabel='Distance Band',
                             ylabel='Percentage of Trips')
                ax.figure.savefig(export_loc + f'tld_fusion_noham_p{pp}_m{md}_tp{tp}_{scen}.png')

                # Master list
                distbands1['mode'] = md
                distbands1['weekday'] = wd
                distbands1['purp'] = pp
                distbands1['tp'] = tp
                dband_list.append(distbands1)

                # Close all plots
                plt.close('all')

# Concat and export TLD
master_tld = pd.concat(dband_list)
master_tld.to_csv(export_loc + 'tld_internals.csv', index=False)
