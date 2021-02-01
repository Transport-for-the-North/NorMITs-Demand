# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 11:33:58 2019

Upgrade census journey to work to 2018
Resolve non standard MSOA zones
Output at model level

@author: cruella
"""

import os

# _LU_PATH = 'C:/Users/' + os.getlogin() + '/S/NorMITs Land Use/Python/GB Property Database'

import sys
import pandas as pd
import geopandas as gpd


# sys.path.append(_LU_PATH)

# import gb_property_database as nlu

_default_model_folder = 'Y:/NorMITs Synthesiser/Norms/'
_default_cjtw_path = 'Y:/Data Strategy/Data/ct_stage/WU03UK_msoa_v3/'

def count_list_shp(shp, idCol=None):
    # Go and fetch a shape and return a count and a list of unq values
    # TODO: Make this properly shape agnostic
    shp = gpd.read_file(shp)
    if idCol is None:
        idCol=list(shp)[0]
    shp = shp.loc[:,idCol]
    return(len(shp),shp)

def hybrid_zone_counts(hybridMsoaCol, unqHybrid):
    # This just doesn't work at all now
    print('Comparing hybrid MSOA zones to census journey to work')    
    unqZonesInCol = hybridMsoaCol.drop_duplicates()
    hybridMsoaInResidence = unqHybrid[unqHybrid.isin(unqZonesInCol)]
    if len(hybridMsoaInResidence) == len(unqHybrid):
        print('All MSOAs accounted for')
        return(True)
    else:
        hybridMsoaNotInResidence = unqHybrid[~unqHybrid.isin(unqZonesInCol)]
        print(len(hybridMsoaNotInResidence), 'residential MSOAs not accounted for')
        return(False)

def split_audit(gbCjtw, unqHybrid):
    # Test split total is same as number of distinct zones
    # If so zone split has worked
    # TODO: Can't just brute force the reindex like this - need to figure out what's there first
    auditSet = gbCjtw.reindex(['1_Areaofusualresidence','overlap_msoa_hybrid_pop_split_factor'],axis=1).drop_duplicates()
    splitSum = auditSet['overlap_msoa_hybrid_pop_split_factor'].sum()
    print('Total split:', splitSum)
    if splitSum == len(unqHybrid):
        print('Total split factors match splits after joins')
        return(True)
    else:
        return(False)

def factor_col(df,
               method=None,
               totalCol='3_Allcategories_Methodoftraveltowork',
               workingCol=None):
    # function to factor columns up or down by a factor in the same pandas row
    # method needs to be 'Up' or 'Down'
    if workingCol is not None:
        if method == 'Up':
            print('factoring up', workingCol)
            df[workingCol] = df[workingCol]*df[totalCol]
        elif method == 'Down':
            print('factoring down', workingCol)
            df[workingCol] = df[workingCol]/df[totalCol]
        else:
            raise ValueError('No factoring method supplied: set method to \'Up\' or \'Down\'')
    else:
        print('No working column supplied')
    return(df)

def clean_cjtw(cjtw):
    # TODO: Function to clean top end trips.
    # Iterate over by mode
    # Get zone to zone distance
    # Get sigma of distribution
    # Clean out any trips 3sigma over mean.
    return(cjtw)

# What will be function parameter defs
def cjtw_to_zone_translation(model_folder = _default_model_folder,
                             model_name = os.path.basename(
                                     os.path.normpath(_default_model_folder)),
                             cjtw_import = _default_cjtw_path,
                             cjtw_dat_file = 'wu03uk_msoa_v3.csv',
                             cjtw_header_file = 'WU03UK_msoa.txt',
                             TestSubset = False):

    """
    What this does
    
    
    """      

    lookups = os.path.join(model_folder, 'Model Zone Lookups')

    # Import MSOA zone blocks
    # msoaRef = nlu.msoaRef
    # msoaShp = nlu.CountListShp(shp=msoaRef, idCol='msoa11cd')
    # unqMsoa = msoaShp[1]

    # sdzRef = 'Y:/Data Strategy/GIS Shapefiles/Scottish_Intermediate_Zones_2001/SG_IntermediateZone_Bdry_2001.shp'
    # sdzShp = nlu.CountListShp(shp=sdzRef, idCol='IZ_CODE')
    # unqSdz = sdzShp[1]

    msoaHybridRef = 'Y:/Data Strategy/GIS Shapefiles/UK MSOA 2011 IZ 2001 Hybrid/UK MSOA 2011 IZ 2001 Hybrid.shp'
    msoaHybridShp = count_list_shp(shp=msoaHybridRef, idCol='msoa11cd')
    unqHybrid = msoaHybridShp[1]
    msoaHybrid = gpd.read_file(msoaHybridRef).reindex(['objectid','msoa11cd'],axis=1)

    cjtw_header = []
    with open ((cjtw_import + cjtw_header_file), "r") as myfile:
        for columns in (raw.strip() for raw in myfile):
            cjtw_header.append(columns)

    cjtw_header = pd.Series(cjtw_header)
    cjtw_header = cjtw_header[7:21].reset_index(drop=True)
    cjtw_header = cjtw_header.str.replace(',','').str.replace(' ','').str.replace(':','_')

    print('Importing 2011 census journey to work')
    cjtw = pd.read_csv((cjtw_import + cjtw_dat_file), names=cjtw_header)

    if TestSubset:
        cjtw = cjtw[cjtw['1_Areaofusualresidence'] == 'E02000025']

    # Get total trip counts in hybrid area for comparison
    inclusiveZones = cjtw[cjtw['1_Areaofusualresidence'].isin(unqHybrid)]
    inclusiveZones = inclusiveZones[inclusiveZones['2_Areaofworkplace'].isin(unqHybrid)]
    totalTrips1 = inclusiveZones['3_Allcategories_Methodoftraveltowork'].sum()
    print(totalTrips1)
    del(inclusiveZones)

    # Look in model folder for 'hybrid pop' translation
    # Should be pathed to the model folder
    file_sys = os.listdir(lookups)
    msoa_hybrid_pop_lookup_path = [x for x in file_sys if ('pop_weighted') in x]
    msoa_hybrid_pop_lookup_path = [x for x in msoa_hybrid_pop_lookup_path
                                   if '_hybrid' in x][0]

    hybrid_msoa_trans = pd.read_csv(lookups +
                                    '/' +
                                    msoa_hybrid_pop_lookup_path)

    hmt_cols = ['msoa_hybrid_zone_id',
               (model_name.lower() + '_zone_id'),
               'overlap_msoa_hybrid_split_factor',
               ('overlap_' + model_name.lower() + '_split_factor')]

    # Append msoa11cd
    hybrid_msoa_trans = hybrid_msoa_trans.reindex(hmt_cols,axis=1)
    hybrid_msoa_trans = hybrid_msoa_trans.merge(msoaHybrid,
                                                how='inner',
                                                left_on='msoa_hybrid_zone_id',
                                                right_on='objectid').drop(
                                                        'objectid',axis=1)

    gb_cjtw = cjtw[cjtw['1_Areaofusualresidence'].isin(unqHybrid)]
    gb_cjtw = gb_cjtw[gb_cjtw['2_Areaofworkplace'].isin(unqHybrid)]
    totalTrips2 = gb_cjtw['3_Allcategories_Methodoftraveltowork'].sum()

    # Join translation - msoa11cd is retained somehow - need to keep it.
    gb_cjtw = gb_cjtw.merge(hybrid_msoa_trans,
                            how='inner',
                            left_on='1_Areaofusualresidence',
                            right_on='msoa11cd').drop_duplicates(
                                    ).drop('msoa_hybrid_zone_id',axis=1)

    gb_cjtw = gb_cjtw.rename(columns={(
            model_name.lower() + '_zone_id'):(
                    '1_' + model_name.lower() +'Areaofresidence')})
    totalTrips3 = gb_cjtw['3_Allcategories_Methodoftraveltowork'].sum()
    del(cjtw)

    # TODO: define totals by column for audits
    # Reduce segments to factors
    factor_cols = ['4_Workmainlyatorfromhome',
                   '5_Undergroundmetrolightrailtram',
                   '6_Train',
                   '7_Busminibusorcoach',
                   '8_Taxi',
                   '9_Motorcyclescooterormoped',
                   '10_Drivingacarorvan',
                   '11_Passengerinacarorvan',
                   '12_Bicycle',
                   '13_Onfoot',
                   '14_Othermethodoftraveltowork']

    # Factor down columns for split adjustment
    for col in factor_cols:
        gb_cjtw = factor_col(gb_cjtw, method='Down', workingCol=col)

    # Apply split adjustment
    # TODO: May be a bit more complicated than this - need to check
    gb_cjtw['3_Allcategories_Methodoftraveltowork'] = (
            gb_cjtw['3_Allcategories_Methodoftraveltowork']*
            gb_cjtw[hmt_cols[2]]) # Overlap_msoa_hybrid_split_factor

    totalTrips4 = gb_cjtw['3_Allcategories_Methodoftraveltowork'].sum()

    # Factor up columns to resolve splits
    for col in factor_cols:
        gb_cjtw = factor_col(gb_cjtw, method='Up', workingCol=col)

    zone_audit = hybrid_zone_counts(gb_cjtw['1_Areaofusualresidence'],
                                   unqHybrid)
    audit_status = split_audit(gb_cjtw,
                              unqHybrid)
    print(audit_status)

    # Build reindex columns - cool method :D
    zone_cjtw_cols = [('1_' + model_name.lower() +'Areaofresidence'),
                      '1_Areaofusualresidence',
                      '2_Areaofworkplace',
                      '3_Allcategories_Methodoftraveltowork']
    for col in factor_cols:
        zone_cjtw_cols.append(col)

    zone_cjtw = gb_cjtw.reindex(zone_cjtw_cols,
                                axis=1)
    del(gb_cjtw)

    totalTrips5 = zone_cjtw['3_Allcategories_Methodoftraveltowork'].sum()

    # Do the same with the commute attraction weightings for the attraction end
    msoa_hybrid_emp_lookup_path = [x for x in file_sys if (
            'emp_weighted') in x and 'hybrid' in x][0]
    hybrid_msoa_emp_trans = pd.read_csv(lookups +
                                        '/' +
                                        msoa_hybrid_emp_lookup_path)

    hybrid_msoa_emp_trans = hybrid_msoa_emp_trans.reindex(hmt_cols,
                                                          axis=1)
    hybrid_msoa_emp_trans = hybrid_msoa_emp_trans.merge(
            msoaHybrid,
            how='inner',
            left_on='msoa_hybrid_zone_id',
            right_on='objectid').drop('objectid',axis=1)

    zone_cjtw = zone_cjtw.merge(hybrid_msoa_emp_trans,
                                how='inner',
                                left_on='2_Areaofworkplace',
                                right_on='msoa11cd')

    # Audit it
    zone_audit = hybrid_zone_counts(zone_cjtw['2_Areaofworkplace'],
                                    unqHybrid)
    zone_audit = hybrid_zone_counts(zone_cjtw['2_Areaofworkplace'],
                                    unqHybrid)

    # Not dropping duplicates here, apparently they're required.
    zone_cjtw = zone_cjtw.rename(columns={(
            model_name.lower() + '_zone_id'):(
                    '2_' + model_name.lower() +'Areaofworkplace')})

    zone_audit = hybrid_zone_counts(zone_cjtw['2_Areaofworkplace'], unqHybrid)
    print(zone_audit)

    totalTrips6 = zone_cjtw['3_Allcategories_Methodoftraveltowork'].sum()

    # Factor down columns for split adjustment
    for col in factor_cols:
        zone_cjtw = factor_col(zone_cjtw, method='Down', workingCol=col)

    # Apply split adjustment
    zone_cjtw['3_Allcategories_Methodoftraveltowork'] = (
            zone_cjtw['3_Allcategories_Methodoftraveltowork']*
            zone_cjtw['overlap_msoa_hybrid_split_factor'])

    totalTrips7 = zone_cjtw['3_Allcategories_Methodoftraveltowork'].sum()

    # Factor up columns to resolve splits
    for col in factor_cols:
        zone_cjtw = factor_col(zone_cjtw, method='Up', workingCol=col)

    zone_cjtw_cols = [('1_' + model_name.lower() +'Areaofresidence'),
                      ('2_' + model_name.lower() + 'Areaofworkplace'),
                      '3_Allcategories_Methodoftraveltowork']
    for col in factor_cols:
        zone_cjtw_cols.append(col)

    zone_cjtw = zone_cjtw.reindex(zone_cjtw_cols,axis=1)
    totalTrips8 = zone_cjtw['3_Allcategories_Methodoftraveltowork'].sum()

    # TODO: Upgrade CJTW to 2018!

    zone_cjtw = zone_cjtw.groupby(
            [('1_' +
              model_name.lower() +
              'Areaofresidence'),
        ('2_' +
         model_name.lower() +
         'Areaofworkplace')]).sum().reset_index()

    # TODO: write clean_cjtw
    zone_cjtw = clean_cjtw(zone_cjtw)

    totalTrips9 = zone_cjtw['3_Allcategories_Methodoftraveltowork'].sum()

    zone_cjtw.to_csv(lookups + '/cjtw_' + model_name.lower() + '.csv',
                     index=False)

    # Segment Rail
    segCols = list(zone_cjtw)
    railSegment = [segCols[0],segCols[1],'6_Train']
    zone_cjtw_rail = zone_cjtw.reindex(railSegment,axis=1)

    zone_cjtw_rail.to_csv(lookups + '/cjtw_' + model_name.lower() + '_rail_only.csv',index=False)

    zone_cjtw_rail_excl = zone_cjtw.copy()
    zone_cjtw_rail_excl[
            '3_Allcategories_Methodoftraveltowork'] = (zone_cjtw_rail_excl[
                    '3_Allcategories_Methodoftraveltowork']-zone_cjtw_rail_excl['6_Train'])

    zone_cjtw_rail_excl = zone_cjtw_rail_excl.drop('6_Train',axis=1)

    zone_cjtw_rail_excl.to_csv(lookups + '/cjtw_' + model_name.lower() + '_rail_excl.csv',index=False)

    audit_numbers = pd.DataFrame([totalTrips1, totalTrips2,
                                  totalTrips3, totalTrips4,
                                  totalTrips5, totalTrips6,
                                  totalTrips7, totalTrips8,
                                  totalTrips9])

    return(zone_cjtw, audit_numbers)