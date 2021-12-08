# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 15:32:36 2021

@author: WoodfiP

TODO:
    For both mdd car and NoHAM OD car matrices
    Zone trip ends
    LAD-LAD sector matrix
    LAD trip ends
    Screenline sector-sector matrix
    Screenline sector trip ends
    TLDs
    
    All run for cars by purpose and time period
"""

import pandas as pd
import numpy as np
import pickle as pk
import matplotlib.pyplot as plt

#Locations
mdd_car_loc = 'Y:/Mobile Data/Processing/dct_MDDCar.pkl'
noham_loc = 'Y:/Mobile Data/Processing'
lad_loc = 'Y:/Mobile Data/Processing/MDD_Check/lookups/lads20_to_noham_correspondence.csv'
noham_internal_loc = 'Y:/Mobile Data/Processing/MDD_Check/lookups/noham_internal.csv'
sector_loc = 'Y:/Mobile Data/Processing/MDD_Check/lookups/sector_to_noham_correspondence.csv'
distance_loc = 'Y:/NoHAM/17.TUBA_Runs/-TPT/Skims/RefDist_Skims/NoHAM_Base_2018_TS2_v106_Dist_Other.csv'

#Read in MDD
with open(r'Y:/Mobile Data/Processing/dct_MDDCar.pkl', 'rb') as log:
        dct_mdd_car = pk.load(log)
#[md][wd][pp][hr]
#[3][1][1-5][1-3]
        
print(dct_mdd_car.keys())
print(dct_mdd_car[3].keys())
print(dct_mdd_car[3][1].keys())
print(dct_mdd_car[3][1][1].keys())
print(dct_mdd_car[3][1][1][1])


#Read in NoHAM
with open(r'Y:/Mobile Data/Processing/dctNoHAM_mddpurp.pkl', 'rb') as log:
        dct_noham_car = pk.load(log)
#[md][wd][pp][hr]
#[3][1][1-5][1-4]
        
print(dct_noham_car.keys())
print(dct_noham_car[3].keys())
print(dct_noham_car[3][1].keys())
print(dct_noham_car[3][1][1].keys())
print(dct_noham_car[3][1][1][1])
np.shape(dct_noham_car[3][1][2][3])
print((dct_noham_car[3][1][2][3]))

#Read in lookups
lad_cor = pd.read_csv(lad_loc, 
                      names=['lad', 'zone', 'lads_f', 'zone_f', 'internal'],
                      skiprows=1)
lad_cor = lad_cor.drop('lads_f', axis=1)

zone_int = pd.read_csv(noham_internal_loc, 
                      names=['zone', 'internal'],
                      skiprows=1)

sector_cor = pd.read_csv(sector_loc, 
                      names=['sector', 'zone', 'factor', 'internal'],
                      skiprows=1)

distance = pd.read_csv(distance_loc,
                       names = ['o_zone', 'd_zone', 'dist_km'])

#TLD distance band
tldDist = [0,1,2,3,4,5,6,7,8,9,10,12.5,15,17.5,20,25,30,35,40,50,75,100,150,200,250,300,400,600,999]
tldLabels = list(range(1,29))
tldBand = {'band':[],'lower':[],'upper':[]}

for row in zip(range(1,len(tldDist)),tldDist[:-1],tldDist[1:]):
    tldBand['band'].append(row[0])
    tldBand['lower'].append(row[1])
    tldBand['upper'].append(row[2])

#zone list
unq_zones = list(range(1, 2771))

#Dictionary layers
dctmode = {3: ['Car']}
dctday = {1: ['Weekday']}
dctpurp = {1: ['hbw_fr'], 2: ['hbw_to'], 3: ['hbo_fr'], 4: ['hbo_to'], 5: ['nhb']}
dcttp = {1: ['AM'], 2: ['IP'], 3: ['PM']}

zone_te_list = []
lad_te_list = []
lad_list = []
sec_te_list = []
sec_list = []
dband_list = []


for md in dctmode:
        for wd in dctday:
            for pp in dctpurp:
                for tp in dcttp:
                    #Convert to pandas
                    wide_mdd = pd.DataFrame(dct_mdd_car[md][wd][pp][tp], 
                                           index = unq_zones, 
                                           columns = unq_zones).reset_index()
                    long_mdd = pd.melt(wide_mdd, 
                                      id_vars = ['index'], 
                                      var_name = 'd_zone', 
                                      value_name = 'mddtrip',
                                      col_level = 0)
                    long_mdd = long_mdd.rename(columns = {'index':'o_zone'})
                    
                    wide_nm = pd.DataFrame(dct_noham_car[md][wd][pp][tp], 
                                           index = unq_zones, 
                                           columns = unq_zones).reset_index()
                    long_nm = pd.melt(wide_nm, 
                                      id_vars = ['index'], 
                                      var_name = 'd_zone', 
                                      value_name = 'nmtrip',
                                      col_level = 0)
                    long_nm = long_nm.rename(columns = {'index':'o_zone'})
                    
                    #Join MDD and NoHAM
                    mat = pd.merge(long_mdd,
                                   long_nm,
                                   on = ['o_zone', 'd_zone'])
                    
                    ######
                    #Zone tripends
                    o_trips = mat.groupby(['o_zone']).agg({'mddtrip':sum, 'nmtrip':sum}).reset_index()
                    o_trips = o_trips.rename(columns = {'mddtrip':'o_mdd', 'nmtrip':'o_nm'})
                    d_trips = mat.groupby(['d_zone']).agg({'mddtrip':sum, 'nmtrip':sum}).reset_index()
                    d_trips = d_trips.rename(columns = {'mddtrip':'d_mdd', 'nmtrip':'d_nm'})
                    #Join
                    zone_te = pd.merge(o_trips,
                                       d_trips,
                                       left_on = ['o_zone'],
                                       right_on = ['d_zone'])
                    zone_te = zone_te[['o_zone',
                                       'o_mdd',
                                       'd_mdd',
                                       'o_nm',
                                       'd_nm']]
                    zone_te = zone_te.rename(columns = {'o_zone':'zone'})
                    #Join internal/external marker
                    zone_te = pd.merge(zone_te,
                                       zone_int,
                                       on=['zone'])
                    #Export as csv
                    zone_te.to_csv(f'Y:/Mobile Data/Processing/MDD_Check/zone_te_od_p{pp}_m{md}_tp{tp}.csv', 
                                   index = False)
                    #Master list
                    zone_te['mode'] = md
                    zone_te['weekday'] = wd
                    zone_te['purp'] = pp
                    zone_te['tp'] = tp
                    zone_te_list.append(zone_te)
                    
                    #add linear regression for zone-zone
                    d = np.polyfit(mat['nmtrip'], mat['mddtrip'], 1)
                    f = np.poly1d(d)
                    mat['linearfit'] = f(mat['nmtrip'])
                    #plot and export
                    ax = mat.plot.scatter(y='mddtrip', x='nmtrip', 
                                          title='Regression: y = ' + str(d[0].round(2)) + 'x + ' + str(d[0].round(1)))
                    mat.plot(x='nmtrip', y='linearfit', color='Red', ax=ax,
                                  xlabel='NoHAM trips',
                                  ylabel='MDD trips')
                    ax.figure.savefig(f'Y:/Mobile Data/Processing/MDD_Check/zone_od_p{pp}_m{md}_tp{tp}.png')
                    
                    #Repeat zone matrix plots, excluding External-External movements
                    mat1 = pd.merge(mat,
                                    zone_int,
                                    left_on = ['o_zone'],
                                    right_on = ['zone'])
                    mat1 = mat1.rename(columns = {'internal':'o_int'})
                    mat1 = pd.merge(mat1,
                                    zone_int,
                                    left_on = ['d_zone'],
                                    right_on = ['zone'])
                    mat1 = mat1.rename(columns = {'internal':'d_int'})
                    mat1.drop(['zone_x', 'zone_y'], axis=1, inplace=True)
                    
                    mat1 = mat1[(mat1['o_int'] == 1) | (mat1['d_int'] == 1)]
                    mat1.drop('linearfit', axis=1, inplace=True)
                    #add linear regression for zone-zone
                    d = np.polyfit(mat1['nmtrip'], mat1['mddtrip'], 1)
                    f = np.poly1d(d)
                    mat1['linearfit'] = f(mat1['nmtrip'])
                    #plot and export
                    ax = mat1.plot.scatter(y='mddtrip', x='nmtrip', 
                                          title='Regression: y = ' + str(d[0].round(2)) + 'x + ' + str(d[0].round(1)))
                    mat1.plot(x='nmtrip', y='linearfit', color='Red', ax=ax,
                                  xlabel='NoHAM trips',
                                  ylabel='MDD trips')
                    ax.figure.savefig(f'Y:/Mobile Data/Processing/MDD_Check/zone_od_xEE_p{pp}_m{md}_tp{tp}.png')
                    
                    
                    ######
                    #LAD sectors, origins then dests for many-many relationship
                    mat_lad = pd.merge(mat,
                                       lad_cor,
                                       left_on = ['o_zone'],
                                       right_on = ['zone'])
                    mat_lad['mdd_lad'] = mat_lad['mddtrip'] * mat_lad['zone_f']
                    mat_lad['nm_lad'] = mat_lad['nmtrip'] * mat_lad['zone_f']
                    mat_lad = mat_lad.groupby(['lad', 'd_zone', 'internal']).agg({'mdd_lad':sum, 'nm_lad':sum}).reset_index()
                    mat_lad = mat_lad.rename(columns = {'lad':'o_lad', 'internal':'o_int'})
                    
                    mat_lad = pd.merge(mat_lad,
                                       lad_cor,
                                       left_on = ['d_zone'],
                                       right_on = ['zone'])
                    mat_lad['mdd_lad'] = mat_lad['mdd_lad'] * mat_lad['zone_f']
                    mat_lad['nm_lad'] = mat_lad['nm_lad'] * mat_lad['zone_f']
                    mat_lad = mat_lad.groupby(['o_lad', 'lad', 'o_int', 'internal']).agg({'mdd_lad':sum, 'nm_lad':sum}).reset_index()
                    mat_lad = mat_lad.rename(columns = {'lad':'d_lad', 'internal':'d_int'})
                    
                    #LAD tripends
                    o_trips = mat_lad.groupby(['o_lad', 'o_int']).agg({'mdd_lad':sum, 'nm_lad':sum}).reset_index()
                    o_trips = o_trips.rename(columns = {'mdd_lad':'o_mdd', 'nm_lad':'o_nm'})
                    d_trips = mat_lad.groupby(['d_lad', 'd_int']).agg({'mdd_lad':sum, 'nm_lad':sum}).reset_index()
                    d_trips = d_trips.rename(columns = {'mdd_lad':'d_mdd', 'nm_lad':'d_nm'})
                    #Join
                    lad_te = pd.merge(o_trips,
                                       d_trips,
                                       left_on = ['o_lad'],
                                       right_on = ['d_lad'])
                    lad_te = lad_te[['o_lad',
                                     'o_int',
                                     'o_mdd',
                                     'd_mdd',
                                     'o_nm',
                                     'd_nm']]
                    lad_te = lad_te.rename(columns = {'o_lad':'lad', 'o_int':'internal'})
                    #Export as csv
                    lad_te.to_csv(f'Y:/Mobile Data/Processing/MDD_Check/lad_te_od_p{pp}_m{md}_tp{tp}.csv', 
                                   index = False)
                    #Master list
                    lad_te['mode'] = md
                    lad_te['weekday'] = wd
                    lad_te['purp'] = pp
                    lad_te['tp'] = tp
                    lad_te_list.append(lad_te)
                    
                    #add linear regression for LAD-LAD
                    d = np.polyfit(mat_lad['nm_lad'], mat_lad['mdd_lad'], 1)
                    f = np.poly1d(d)
                    mat_lad['linearfit'] = f(mat_lad['nm_lad'])
                    #plot and export
                    ax = mat_lad.plot.scatter(y='mdd_lad', x='nm_lad', 
                                 title='Regression: y = ' + str(d[0].round(2)) + 'x + ' + str(d[0].round(1)))
                    mat_lad.plot(x='nm_lad', y='linearfit', color='Red', ax=ax,
                                  xlabel='NoHAM LAD trips',
                                  ylabel='MDD LAD trips')
                    ax.figure.savefig(f'Y:/Mobile Data/Processing/MDD_Check/lad_od_p{pp}_m{md}_tp{tp}.png')
                    #Close all plots
                    plt.close('all')
                    
                    #Repeat LAD, excluding External-External movements
                    mat_lad2 = mat_lad[(mat_lad['o_int'] == 1) | (mat_lad['d_int'] == 1)]
                    mat_lad2.drop('linearfit', axis=1, inplace=True)
                    #add linear regression for LAD-LAD
                    d = np.polyfit(mat_lad2['nm_lad'], mat_lad2['mdd_lad'], 1)
                    f = np.poly1d(d)
                    mat_lad2['linearfit'] = f(mat_lad2['nm_lad'])
                    #plot and export
                    ax = mat_lad2.plot.scatter(y='mdd_lad', x='nm_lad', 
                                  title='Regression: y = ' + str(d[0].round(2)) + 'x + ' + str(d[0].round(1)))
                    mat_lad2.plot(x='nm_lad', y='linearfit', color='Red', ax=ax,
                                  xlabel='NoHAM LAD trips',
                                  ylabel='MDD LAD trips')
                    ax.figure.savefig(f'Y:/Mobile Data/Processing/MDD_Check/lad_od_xEE_p{pp}_m{md}_tp{tp}.png')
                    plt.close('all')
                    
                    #Add LAD to master list
                    mat_lad['mode'] = md
                    mat_lad['weekday'] = wd
                    mat_lad['purp'] = pp
                    mat_lad['tp'] = tp
                    lad_list.append(mat_lad)
                    
                    ######
                    #Screenline sectors
                    #Join sectors
                    mat_sec = pd.merge(mat,
                                       sector_cor,
                                       left_on = ['o_zone'],
                                       right_on = ['zone'])
                    mat_sec = mat_sec.rename(columns = {'sector':'o_sec', 'factor':'of', 'internal':'o_int'})
                    mat_sec.drop(['zone', 'linearfit'], axis=1, inplace=True)
                    mat_sec = pd.merge(mat_sec,
                                       sector_cor,
                                       left_on = ['d_zone'],
                                       right_on = ['zone'])
                    mat_sec = mat_sec.rename(columns = {'sector':'d_sec', 'factor':'df', 'internal':'d_int'})
                    mat_sec.drop(['zone'], axis=1, inplace=True)
                    #Apply factors to trips and group
                    mat_sec['mdd_sec'] = mat_sec['mddtrip'] * mat_sec['of'] * mat_sec['df']
                    mat_sec['nm_sec'] = mat_sec['nmtrip'] * mat_sec['of'] * mat_sec['df']
                    mat_sec = mat_sec.groupby(['o_sec', 'd_sec', 'o_int', 'd_int']).agg({'mdd_sec':sum, 'nm_sec':sum}).reset_index()
                    
                    #Sector tripends
                    o_trips = mat_sec.groupby(['o_sec', 'o_int']).agg({'mdd_sec':sum, 'nm_sec':sum}).reset_index()
                    o_trips = o_trips.rename(columns = {'mdd_sec':'o_mdd', 'nm_sec':'o_nm'})
                    d_trips = mat_sec.groupby(['d_sec', 'd_int']).agg({'mdd_sec':sum, 'nm_sec':sum}).reset_index()
                    d_trips = d_trips.rename(columns = {'mdd_sec':'d_mdd', 'nm_sec':'d_nm'})
                    #Join
                    sec_te = pd.merge(o_trips,
                                       d_trips,
                                       left_on = ['o_sec'],
                                       right_on = ['d_sec'])
                    sec_te = sec_te[['o_sec',
                                     'o_int',
                                     'o_mdd',
                                     'd_mdd',
                                     'o_nm',
                                     'd_nm']]
                    sec_te = sec_te.rename(columns = {'o_sec':'sector', 'o_int':'internal'})
                    #Export as csv
                    sec_te.to_csv(f'Y:/Mobile Data/Processing/MDD_Check/sector_te_od_p{pp}_m{md}_tp{tp}.csv', 
                                   index = False)
                    #Master list
                    sec_te['mode'] = md
                    sec_te['weekday'] = wd
                    sec_te['purp'] = pp
                    sec_te['tp'] = tp
                    sec_te_list.append(sec_te)
                    
                    #add linear regression for sector-sector
                    d = np.polyfit(mat_sec['nm_sec'], mat_sec['mdd_sec'], 1)
                    f = np.poly1d(d)
                    mat_sec['linearfit'] = f(mat_sec['nm_sec'])
                    #plot and export
                    ax = mat_sec.plot.scatter(y='mdd_sec', x='nm_sec', 
                                 title='Regression: y = ' + str(d[0].round(2)) + 'x + ' + str(d[0].round(1)))
                    mat_sec.plot(x='nm_sec', y='linearfit', color='Red', ax=ax,
                                  xlabel='NoHAM Sector trips',
                                  ylabel='MDD Sector trips')
                    ax.figure.savefig(f'Y:/Mobile Data/Processing/MDD_Check/sector_od_p{pp}_m{md}_tp{tp}.png')
                    #Close all plots
                    plt.close('all')
                    
                    #Repeat Sector, excluding External-External movements
                    mat_sec2 = mat_sec[(mat_sec['o_int'] == 1) | (mat_sec['d_int'] == 1)]
                    mat_sec2.drop('linearfit', axis=1, inplace=True)
                    #add linear regression for LAD-LAD
                    d = np.polyfit(mat_sec2['nm_sec'], mat_sec2['mdd_sec'], 1)
                    f = np.poly1d(d)
                    mat_sec2['linearfit'] = f(mat_sec2['nm_sec'])
                    #plot and export
                    ax = mat_sec2.plot.scatter(y='mdd_sec', x='nm_sec', 
                                  title='Regression: y = ' + str(d[0].round(2)) + 'x + ' + str(d[0].round(1)))
                    mat_sec2.plot(x='nm_sec', y='linearfit', color='Red', ax=ax,
                                  xlabel='NoHAM Sector trips',
                                  ylabel='MDD Sector trips')
                    ax.figure.savefig(f'Y:/Mobile Data/Processing/MDD_Check/sector_od_xEE_p{pp}_m{md}_tp{tp}.png')
                    plt.close('all')
                    
                    #Add LAD to master list
                    mat_sec['mode'] = md
                    mat_sec['weekday'] = wd
                    mat_sec['purp'] = pp
                    mat_sec['tp'] = tp
                    sec_list.append(mat_sec)
                    
                    ######
                    #Trip length distribution
                    #join distance
                    mat = pd.merge(mat,
                                   distance,
                                   on=['o_zone', 'd_zone'])
                    #join internal marker
                    mat = pd.merge(mat,
                                    zone_int,
                                    left_on = ['o_zone'],
                                    right_on = ['zone'])
                    mat = mat.rename(columns = {'internal':'o_int'})
                    mat = pd.merge(mat,
                                    zone_int,
                                    left_on = ['d_zone'],
                                    right_on = ['zone'])
                    mat = mat.rename(columns = {'internal':'d_int'})
                    mat.drop(['zone_x', 'zone_y'], axis=1, inplace=True)
                    #cut for distance bands and group
                    mat['dband'] = pd.cut(mat['dist_km'], bins=tldDist, right=False, labels=tldLabels)
                    distbands = mat.groupby(['dband', 'o_int', 'd_int']).agg({'mddtrip':sum, 'nmtrip':sum}).reset_index()
                    
                    #Master list
                    distbands['mode'] = md
                    distbands['weekday'] = wd
                    distbands['purp'] = pp
                    distbands['tp'] = tp
                    dband_list.append(distbands)
                    
                    
                    
#Concat and export Zone trip ends
master_te = pd.concat(zone_te_list)
master_te.to_csv('Y:/Mobile Data/Processing/MDD_Check/trip_ends_zone.csv', 
                                   index = False)
#Concat and export LAD trip ends
master_lad_te = pd.concat(lad_te_list)
master_lad_te.to_csv('Y:/Mobile Data/Processing/MDD_Check/trip_ends_lad.csv', 
                                   index = False)               
#Concat and export LAD-LAD matrix
master_lad = pd.concat(lad_list)
master_lad.to_csv('Y:/Mobile Data/Processing/MDD_Check/lad-lad_matrices.csv', 
                                   index = False)
#Concat and export Sector trip ends
master_sec_te = pd.concat(sec_te_list)
master_sec_te.to_csv('Y:/Mobile Data/Processing/MDD_Check/trip_ends_sectors.csv', 
                                   index = False)
#Concat and export sector-Sector matrix
master_sec = pd.concat(sec_list)
master_sec.to_csv('Y:/Mobile Data/Processing/MDD_Check/sector-sector_matrices.csv', 
                                   index = False)
#Concat and export TLD
master_tld = pd.concat(dband_list)
master_tld.to_csv('Y:/Mobile Data/Processing/MDD_Check/tld.csv', 
                                   index = False)

# save as excel file
with pd.ExcelWriter('Y:/Mobile Data/Processing/MDD_Check/MDD-NoHAM_Summary.xlsx') as writer:
    #master_lad.to_excel(writer, 
    #                    sheet_name = 'LAD-LAD', 
    #                    index=None)
    master_te.to_excel(writer, 
                       sheet_name = 'Zone_TE', 
                       index=None)
    master_lad_te.to_excel(writer, 
                           sheet_name = 'LAD_TE', 
                           index=None)
    master_sec_te.to_excel(writer, 
                           sheet_name = 'Sector_TE', 
                           index=None)
    master_sec.to_excel(writer, 
                           sheet_name = 'Sec-Sec', 
                           index=None)
    master_tld.to_excel(writer, 
                           sheet_name = 'TLD', 
                           index=None)





