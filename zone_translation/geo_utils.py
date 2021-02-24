# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 

Nest 2 shapefiles, ready for population attribution for more detailed splits.

@author: ChristopherStorey
"""

# have the main functions return a name vector as string to export straight to csv.

import geopandas as gpd
import pandas as pd
import gc

import zone_translation.zt_constants as ztc

export = "Y:/Zone Translation/Export/"

# Import paths to local population and employment figures
# populations built on 2017 mid year population estimates
# attractions built from 2015 HSL lsoa employment data

localLSOAPopulationsPath = 'Y:/NorMITs Synthesiser/Zone Translation/Import/LSOA Populations/lsoa__populations_2017.csv'
# note that extra underscore there - had a read only problem
localMSOAPopulationsPath = 'Y:/NorMITs Synthesiser/Zone Translation/Import/MSOA Populations/msoa_populations_2011.csv'

localLSOAAllPath = 'Y:/NorMITs Synthesiser/Zone Translation/Import/LSOA Employment/lsoa_employment_2018.csv'

localLSOACommutePath = 'Y:/NorMITs Synthesiser/Zone Translation/Import/LSOA Employment/commuteAttractionsLSOA.csv'
localLSOABusinessPath = 'Y:/NorMITs Synthesiser/Zone Translation/Import/businessAttractionsLSOA.csv'
localLSOAOtherPath = 'Y:/NorMITs Synthesiser/Zone Translation/Import/otherAttractionsLSOA.csv'

lsoaPop = pd.read_csv(localLSOAPopulationsPath)
msoaPop = pd.read_csv(localMSOAPopulationsPath)

# Imports two shapefiles, (identifies - see to do) and names unique ID columns, calculates larger zoning system relatively, returns list of shapefiles

def dissolve_on_var(shp,
                    dissolve_var):
    shp = gpd.read_file(shp)
    shp = shp[[dissolve_var, 'geometry']]
    new_shp = shp.dissolve(by=dissolve_var)    
    return new_shp

def shape_import(zone_path1,
                 zone_path2,
                 zone_name1='zones1',
                 zone_name2='zones2',
                 zone1_index=0,
                 zone2_index=0):

    # Shape Import functions
    
    # TODO: 
    # Check that the path ends in .shp
    # Check that the two shape paths lead to shapefiles
    # Two single read functions

    zone1 = gpd.read_file(zone_path1)
    zone2 = gpd.read_file(zone_path2)
    
    zone_list = [zone1, zone2]

    # Change projection system to OSGB1936
    for zone_item in zone_list:
        if(zone_item.crs != ztc.OSGB_CRS):
            zone_item.crs = ztc.OSGB_CRS

    # Drop non-type geometry on area
    zone1['area'] = zone1.area
    zone1 = zone1.dropna(subset=['area'])
    zone2['area'] = zone2.area
    zone2 = zone2.dropna(subset=['area'])

    if(sum(zone1['area'])/len(zone1) > sum(zone2['area'])/len(zone2)):
        # Assign to maj/min
        major_zone = zone1.copy()
        major_zone_name = zone_name1
        minor_zone = zone2.copy()
        minor_zone_name = zone_name2
        # Rename based on index
        major_zone = major_zone.rename(
                columns={
                    major_zone.columns[zone1_index]: major_zone_name + '_zone_id'})
        minor_zone = minor_zone.rename(
                columns={
                    minor_zone.columns[zone2_index]: minor_zone_name + '_zone_id'})
    elif(sum(zone1.area)/len(zone1) < sum(zone2.area)/len(zone2)):
        # Assign to maj/min
        major_zone = zone2.copy()
        major_zone_name = zone_name2
        minor_zone = zone1.copy()
        minor_zone_name = zone_name1
        # Rename based on index
        major_zone = major_zone.rename(
                columns={
                    major_zone.columns[zone2_index]: major_zone_name + '_zone_id'})
        minor_zone = minor_zone.rename(
                columns={
                    minor_zone.columns[zone1_index]: minor_zone_name + '_zone_id'})
    del zone1, zone2
    del major_zone['area'], minor_zone['area']

    return major_zone, minor_zone

# Calls ShapeImport as above - calculates nesting of 1 shape vector inside the other
       
def zone_nest(
        zone_path1,
        zone_path2,
        zone_name1='zones1',
        zone_name2='zones2',
        upper_tolerance=.85,
        lower_tolerance=.10,
        zone1_index=0,
        zone2_index=0):

    # Import both spatial files using function parameters    
    zone_list = shape_import(
        zone_path1,
        zone_path2,
        zone_name1=zone_name1,
        zone_name2=zone_name2,
        zone1_index=zone1_index,
        zone2_index=zone2_index)

    major_zone = zone_list[0]
    major_zone_id = major_zone.columns.values[zone1_index]
    print(major_zone_id)
    major_zone_name = major_zone_id.replace('_zone_id', '')
    print(major_zone_name)
    minor_zone = zone_list[1]
    minor_zone_id = minor_zone.columns.values[zone2_index]
    print(minor_zone_id)
    minor_zone_name = minor_zone_id.replace('_zone_id', '')
    print(minor_zone_name)

    del zone_list

    overlay_list = []
    # First loop iterates over major zones and creates a set of minor zones that at least partially nest into each one.
    for majorZoneIndex, majorZoneRow in major_zone.iterrows():

        gc.collect()
        # Write a loop to explode any multipolygons - delete the small bit
        tempMajorZone = gpd.GeoDataFrame(major_zone.iloc[[majorZoneIndex]])
        tempMajorZone.crs = ztc.OSGB_CRS

        print('Major Zone', tempMajorZone[major_zone_id].values)

        tempMajorZoneArea = sum(tempMajorZone.area)
        tempJoin = gpd.sjoin(tempMajorZone, minor_zone).reset_index()
        tempJoin = tempJoin.drop(['index_right'], axis=1)

        #print(len(tempJoin), 'matches')

        # Second loop iterates over each matches minor zones and quantifies the overlap using the tolerance parameters
        # This could potentially be functionalised as it's currently quite slow.
        for minorZoneIndex, minorZoneRow in tempJoin.iterrows():

            print('match', minorZoneIndex+1, 'of', len(tempJoin), end="\r", flush=True) # These parameters are new to try and get the console to overwrite (for auditing)
                                                                                        # Good odds it's going to kill the whole thing.
        
            tempMinorZoneNo = tempJoin.iloc[[minorZoneIndex]][minor_zone_id].values
            #print('Minor Zone', tempMinorZoneNo)            
            tempMinorZone = minor_zone[minor_zone[minor_zone_id].isin(tempMinorZoneNo)].reset_index(drop = True)
            tempMinorZoneArea = sum(tempMinorZone.area)
        
            tempOverlay = gpd.overlay(tempMajorZone, tempMinorZone, how='intersection').reset_index()
            if(len(tempOverlay) == 0):
                tempOverlayArea = 0
                #print('No overlap')
            else:
                tempOverlayArea = sum(tempOverlay.area)
                #print('Min Zone 1 overlay area', tempOverlayArea)        
                minToMaj = tempOverlayArea/tempMinorZoneArea
                #print('Minor area overlap', minToMaj*100, '%')
                majToMin = tempOverlayArea/tempMajorZoneArea
                #print('Major area covered', majToMin*100, '%')
                overlayDesc = ""                
                # Case handling to describe overlap types        
                if(minToMaj > upper_tolerance and majToMin > upper_tolerance):
                    overlayDesc = "Close match"
                elif(minToMaj > upper_tolerance and majToMin < upper_tolerance):
                    overlayDesc = "Min-Maj nest"
                elif(minToMaj < upper_tolerance and majToMin > upper_tolerance):
                    overlayDesc = "Maj-Min nest"
                elif(minToMaj < upper_tolerance and minToMaj > lower_tolerance or majToMin < upper_tolerance and majToMin > lower_tolerance):
                    overlayDesc = "Partial Nest"
                elif(minToMaj < upper_tolerance and majToMin < upper_tolerance):
                    overlayDesc = "Marginal overlap"
                #print(overlayDesc)

            # Create descriptive series for matched data frame
                tempOverlayList = tempOverlay[[major_zone_id, minor_zone_id]].drop_duplicates()
                tempOverlayList['overlap_type'] = overlayDesc
                tempOverlayList[major_zone_name + '_to_' + minor_zone_name] = majToMin
                tempOverlayList[minor_zone_name + '_to_' + major_zone_name] = minToMaj
        
                if(overlayDesc != "Marginal overlap"):
                    overlay_list.append(tempOverlayList)
                
                del(tempMinorZoneNo, tempMinorZone, tempMinorZoneArea, tempOverlay, tempOverlayArea, minToMaj, majToMin, overlayDesc, tempOverlayList)
                gc.collect()

        print('end of zone', tempMajorZone[major_zone_id].values)
        del(tempMajorZone, tempMajorZoneArea, tempJoin)

    #format list for editing
    
    zoneMatchTable = pd.concat(overlay_list)
       
    unqMajor = major_zone[major_zone_id].drop_duplicates().tolist()
    unqMajorMatch = zoneMatchTable[major_zone_id].drop_duplicates().tolist()
    
    # Derive a list of major zones in the import set that aren't in the finished overlap file
    majorDrops = set(unqMajor) - set(unqMajorMatch)
    majorDrops = list(majorDrops)
    majorDrops = pd.DataFrame(majorDrops, columns = ['major_drops'])
    
    print('Major zones unmatched:', majorDrops)
    # Write major zone audit to export file
    majorDrops.to_csv(major_zone_name + '_to_' + minor_zone_name + '_dropped_zone_audit.csv')
    
    unqMinor = minor_zone[minor_zone_id].drop_duplicates().tolist()
    unqMinorMatch = zoneMatchTable[minor_zone_id].drop_duplicates().tolist()
    
    # Derive a list of minor zones in the import set that aren't in the finished overlap file
    minorDrops = set(unqMinor) - set(unqMinorMatch)
    minorDrops = list(minorDrops)
    minorDrops = pd.DataFrame(minorDrops, columns = ['minor_drops'])
    
    print('Minor zones unmatched:', minorDrops)
    # Write minor zone audit to export file
    minorDrops.to_csv(minor_zone_name + '_to_' + major_zone_name + '_dropped_zone_audit.csv')
        
    return zoneMatchTable
    
    # End of function

# LSOA Method for applying splits to partially overlapping data files - needs a link to the LSOA/DataZone level data

def pop_apply(areaTranslationPath, populationsPath, populationPaddingFactor = 0, populationMatch = False):
    
    # TODO: Add audit to make sure number of zones coming in are same as those going out

    zonalPopulations = pd.read_csv(populationsPath)
    zonalPopulations['var'] = zonalPopulations['var'].astype(float)
    unqAM = len(zonalPopulations)

    areaTranslation = pd.read_csv(areaTranslationPath, index_col = False)
    atcols = list(areaTranslation)
    # 0 = major zone ID, 1 = minor zone ID, 2 = overlap_type, 3 = major to minor nest factor, 4 = minor to major nest factor 

    #Count unique LSOAs
    areaTranslationunqAM = len(areaTranslation.loc[:, 'lsoa_zone_id'].drop_duplicates())

    # Unq LSOA Matrix audit
    if areaTranslationunqAM == unqAM:
        print('All LSOA zones accounted for in zone system')
        z1Nominal = True
    else:
        print((unqAM - areaTranslationunqAM), ' zones unaccounted for in zone system')
        z1Nominal = False  
    del(areaTranslationunqAM)

    # Inner join lsoas on the lsoa ID
    # TO DO
    # If 'z1Nominal = False' do an outer join and figure out how to assign the leftovers
    if(z1Nominal == True):
        print('inner joining')
        areaTranslationPop = pd.merge(areaTranslation, zonalPopulations, how = 'inner', left_on = 'lsoa_zone_id', right_on = 'objectid') # need to fix these column assigments
    else:
        print('outer joining')
        areaTranslationPop = pd.merge(areaTranslation, zonalPopulations, how = 'outer', left_on = 'lsoa_zone_id', right_on = 'objectid') # need to fix these column assigments
    
    # Derive overcount due to duplicate joins
    overcount = sum(areaTranslationPop['var']) - sum(zonalPopulations['var'])

    # Multiply zones with 'Partial Nest' by a rounded version of the 'minorZone_to_majorZone factor column'
    # TODO Potentially replace this with a threshold call? Possibly not
    at1pn = areaTranslationPop[atcols[2]] == 'Partial Nest'
    at1pn = areaTranslationPop[at1pn]
    # Change PN=True 'pop' to be overlap factor * pop
    # TODO If population match = false, brute force the padding factor until the difference is 0
    at1pn['var'] = (at1pn.iloc[:, 4] * at1pn['var']) + (at1pn.iloc[:, 4] * at1pn['var']) * populationPaddingFactor

    # Filter Partial Nest == False
    at2pn = areaTranslationPop[atcols[2]] != 'Partial Nest'
    at2pn = areaTranslationPop[at2pn]

    # Remerge matrices
    areaTranslationPop = pd.concat([at1pn, at2pn]).sort_index()

    # Check new figure
    newcount = sum(areaTranslationPop['var']) - sum(zonalPopulations['var'])

    print('Difference was', overcount, ' now ', newcount)
    
    areaTranslationPop = areaTranslationPop.drop(['objectid'], axis=1)

    # Subpot the successes
    atsuccess = areaTranslationPop.iloc[:,0] == areaTranslationPop.iloc[:,0]
    atsuccess = areaTranslationPop[atsuccess]
    
    # Count failed lsoa nests
    atfail = areaTranslationPop.iloc[:,0] != areaTranslationPop.iloc[:,0]
    atfail = areaTranslationPop[atfail]
    atfailcount = len(atfail)

    if atfailcount > 0:
        atfail.iloc[:,0] = 999999
    
    areaTranslationPop = pd.concat([atsuccess, atfail]).sort_index()        
    areaTranslationPop = areaTranslationPop.drop(['overlap_type'], axis=1) # Removed population from here
        
    return(areaTranslationPop)

def zone_split(areaTranslationPath1, areaTranslationPath2 = None, splitMethod = 'lsoa_pop', classifyOverlaps=True): # Needs something to capture the population balancing method

    if splitMethod == 'lsoa_pop':
        populationsPath = localLSOAPopulationsPath
        popCol = 'lsoa'
    elif splitMethod == 'msoa_pop':
        populationsPath = localMSOAPopulationsPath
        popCol = 'msoa'
    elif splitMethod == 'lsoa_emp':
        populationsPath = localLSOAAllPath
        popCol = 'lsoa'
    elif splitMethod == 'lsoa_emp_commute':
        populationsPath = localLSOACommutePath
        popCol = 'lsoa'
    elif splitMethod == 'lsoa_emp_business':
        # Can just use the same one for both as there's MSOAs in there too.
        populationsPath = localLSOABusinessPath
        popCol = 'lsoa'
    elif splitMethod == 'lsoa_emp_other':
        # Can just use the same one for both as there's MSOAs in there too.
        populationsPath = localLSOAOtherPath
        popCol = 'lsoa'
    else:
        populationsPath = splitMethod

    if areaTranslationPath2 == None:
        areaTranslationPop = pop_apply(areaTranslationPath1, populationsPath)
        atname = areaTranslationPop.columns[0].replace('_zone_id', '')
        atnames = [atname, popCol]
        ats = [areaTranslationPop, areaTranslationPop]
    else:
        atlist = [areaTranslationPath1, areaTranslationPath2]
        ats = [0, 0]
        atnames = [0, 0]
        i = 0
        for at in atlist:
            areaTranslationPop = pop_apply(at, populationsPath)
            # Get zone system name - should be standardised as this is output from one of my functions anyway
            atname = areaTranslationPop.columns[0].replace('_zone_id', '')
            atnames[i] = atname
            ats[i] = areaTranslationPop
            i = i+1
            # End of loop
        
        # Nest bit - two zone conditional        
        areaTranslationPop = pd.merge(ats[0], ats[1], how = 'outer', left_on = ['lsoa11cd', 'lsoa_zone_id'], right_on = ['lsoa11cd', 'lsoa_zone_id'])

        popmatch = areaTranslationPop['var_x'] == areaTranslationPop['var_y']
        popmatch = areaTranslationPop[popmatch]
        popmatch['var'] = popmatch[['var_x', 'var_y']].min(axis=1)
        popmatch = popmatch.drop(['var_x', 'var_y'], axis=1)
        popmismatch = areaTranslationPop['var_x'] != areaTranslationPop['var_y']
        popmismatch = areaTranslationPop[popmismatch]

        # pick the lowest figure
        # CS - 25/4 Added an 'if' to catch exceptions here, as there is sometimes no mismatch with government zones
        if len(popmismatch) > 0:
            popmismatch['var'] = min(1,2)
            popmismatch['var'] = popmismatch[['var_x', 'var_y']].min(axis=1)
            popmismatch = popmismatch.drop(['var_x', 'var_y'], axis=1)
            areaTranslationPop = pd.concat([popmatch, popmismatch]).sort_index()
        else:
            # if no mismatch we still need to drop one of the populations
            areaTranslationPop = areaTranslationPop.drop('var_y',axis=1).rename(columns={'var_x':'var'})
            print('no mismatch')

    # Loop to get sums from newly adjusted totals
    atsums = [0]*len(atnames)
    j = 0
    for at in atnames:
        zoneCol = at + '_zone_id'
        areaTranslationSum = areaTranslationPop.groupby(areaTranslationPop.loc[:,zoneCol])['var'].agg('sum').reset_index()
        areaTranslationSum = areaTranslationSum.rename(columns = {areaTranslationSum.columns[1] : atnames[j] + '_var'})
        del(zoneCol)
        atsums[j] = areaTranslationSum
        j = j+1
        ### BUILD THE TOTALS HERE - same as the loop but better now
    #now - group by nelumID, rtmID, & sum population
    popMergeStep = areaTranslationPop.groupby([areaTranslationPop.loc[:,atnames[0]+'_zone_id'], areaTranslationPop.loc[:,atnames[1]+'_zone_id']])['var'].sum().reset_index()
    popMergeStep = popMergeStep.rename(columns = {popMergeStep.columns[2] : 'overlap_var'})

    zoneCorrespondence = pd.merge(popMergeStep, atsums[0], how = 'inner', left_on = atnames[0]+'_zone_id', right_on = atnames[0] + '_zone_id')  
    zoneCorrespondence = pd.merge(zoneCorrespondence, atsums[1], how = 'inner', left_on = atnames[1]+'_zone_id' , right_on = atnames[1]+'_zone_id')

    # Seed any missing values with a 1
    # Don't want to do this, should know before now why it's running the 1 in the first place
    missingOverlap = zoneCorrespondence[zoneCorrespondence['overlap_var']==0]
    missingZ1 = zoneCorrespondence[zoneCorrespondence[atnames[0]+'_var']==0]
    missingZ2 = zoneCorrespondence[zoneCorrespondence[atnames[1]+'_var']==0]
    missing = pd.concat([missingOverlap,missingZ1,missingZ2]).sort_index().drop_duplicates()
    #intact = 

    zoneCorrespondence['overlap_'+atnames[0]+'_split_factor'] = zoneCorrespondence['overlap_var']/zoneCorrespondence[atnames[0]+'_var']
    zoneCorrespondence['overlap_'+atnames[1]+'_split_factor'] = zoneCorrespondence['overlap_var']/zoneCorrespondence[atnames[1]+'_var']

    # zoneCorrespondence should be in configuration: 0 'zone1ID', 1 'zone2ID', 2 'overlap_population', 3 'zone1_population', 4 'zone2_population', 5 'overlap_zone1_factor', 6 'overlap_zone2_factor'
    # all calls to iloc are based on this so if it's not right these won't work
    # .98 here as hard coded tolerance for exact match - this will probably do for now

    if classifyOverlaps:
        tolerance = .98

        zn1 = (zoneCorrespondence.iloc[:, 5] >= tolerance) & (zoneCorrespondence.iloc[:,6] <= tolerance)
        zn1 = zoneCorrespondence[zn1]
        if len(zn1)>0:
            zn1.loc[:,'overlap_type'] = atnames[0] + ' ' + atnames[1] + ' nest'
    
        zn2 = (zoneCorrespondence.iloc[:, 5] <= tolerance) & (zoneCorrespondence.iloc[:,6] >= tolerance)
        zn2 = zoneCorrespondence[zn2]

        if len(zn2)>0:
            zn2.loc[:,'overlap_type'] = atnames[1] + ' ' + atnames[0] + ' nest'

        zn3 = (zoneCorrespondence.iloc[:, 5] >= tolerance) & (zoneCorrespondence.iloc[:,6] >= tolerance)
        zn3 = zoneCorrespondence[zn3]
        if len(zn3)>0:
            zn3.loc[:,'overlap_type'] = 'close match'

        zn4 = (zoneCorrespondence.iloc[:, 5] <= tolerance) & (zoneCorrespondence.iloc[:,6] <= tolerance)
        zn4 = zoneCorrespondence[zn4]
        if len(zn4)>0:
            zn4.loc[:,'overlap_type'] = 'partial nest'

        zoneNest = pd.concat([zn1,zn2,zn3, zn4]).sort_index()
        del(zn1, zn2, zn3, zn4)
        return zoneNest

    else:
        return zoneCorrespondence

    # TODO: Balance demand to zones here
