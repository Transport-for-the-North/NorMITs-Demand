# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 14:22:56 2020

@author: cruella
"""

import os
import pandas as pd
import numpy as np
import geopandas as gpd

from normits_demand.matrices import matrix_processing as mp
from normits_demand import version

# TODO: Delete duplicate functions in multiprocessing and dm

# TODO: Should find a way to automate this. Put zone in model folder?
_default_shp_path = 'Y:/Data Strategy/GIS Shapefiles/Norms zones/TfN_Zones_Combined.shp'



def get_average_trip_length(model_lookup_path,
                            distribution,
                            internal_distance = None,
                            join_distance=False,
                            join_type='inner'):
    """
    This function gets average trip length from a distributed PA matrix.

    Parameters
    ----------
    distribution:
        A 24hr pa matrix with production zones, attraction zones, distributed
        trips and distance.

    Returns
    ----------
    [0] atl:
        Average trip length as 64bit float
    """

    # Need to make this explicitly distance or its a bit confusing

    # Join distance
    if join_distance:
        distribution = distribution.merge(internal_distance,
                                          how=join_type,
                                          on=['p_zone', 'a_zone'])

    distribution['total_distance'] = (distribution['dt'] *
                distribution['distance'])
    atl = distribution['total_distance'].sum() / distribution['dt'].sum()

    return(atl)


def trip_length_audit(file_drive='Y:/',
                      model_name='Norms_2015',
                      iteration='iter2',
                      model_segments = ['purpose', 'mode', 'car_availability'],
                      distributions = 'OD Matrices',
                      matrix_format = 'wide',
                      time_period = None,
                      trip_length_distribution_report = True,
                      sector_report = True,
                      calibration_report = False,
                      internal_only = False,
                      write=True):

    """
    DEPRECATED - Served well, time for the knacker's yard
    Use get_trip_length_by_band.
    
    Function to go through the distributions folder and give average trip
    lengths.
    
    Parameters
    ----------
    file_drive:
        Base drive for matrices. Will look for a 'NorMITs Synthesiser' folder
        on this drive.

    model_name:
        Name of the model to run trip length audits for. Used to define path.
        Case sensitive.

    iteration:
        Iteration of the model to run trip length audits for.

    distributions = '24hr PA Distributions':
        Distribution folder to get distributions from.

    matrix_format = 'long':
        Format of the matrices to be audited. Takes 'long' or 'wide'.
        If wide, matrices will be translated back to long format for audits.
    
    time_period = '24hr' or 'tp'

    trip_length_distribution_report = False:
        Boolean to toggle trip length distribution report.
        If true will return DataFrame containing trip length distributions.

    calibration_report = False:
        Boolean to toggle calibration report. Imports initial betas and
        compares the calibrated trip lengths against them.

    Returns
    ----------
    trip_length_report:
        Data frame of trip length audits.

    tld_report:
        Trip length distribution report if requested, else None.

    iz_report:
        Intra zonal report if requested, else None.

    """
    # TODO: Auto config for wide/long if None type passed
    # TODO: Add call to dist curve build script when it does that report

    w_d = (file_drive +
           'NorMITs Synthesiser/' +
           model_name +
           '/' +
           iteration +
           '/Distribution Outputs/Logs & Reports')

    dist_dir = (file_drive +
                '/NorMITs Synthesiser/' +
                model_name +
                '/' +
                iteration +
                '/Distribution Outputs/' +
                distributions)
    dists = os.listdir(dist_dir)

    # Filter out non csv (subfolders for instance)
    dists = [x for x in dists if '.csv' in x]

    # Fix for being sent to the PA folder.
    # There are different distribution types in there.
    if time_period:
        # Not internal_24hr
        dists = [x for x in dists if 'internal_24hr' not in x]
        if time_period == '24hr':
            dists = [x for x in dists if 'hb_pa' in x]
        elif time_period == 'tp':
            dists = [x for x in dists if 'hb_tp_pa' in x]

    model_folder = (file_drive +
                    '/NorMITs Synthesiser/' +
                    model_name +
                    '/Model Zone Lookups')

    p_h = [] # Placeholder for topline report
    tld_ph = [] # Placeholder for trip length distribution
    sector_ph = [] # Placeholder for segment reports

    for dist in dists:
        print(dist)
        file = dist
        import_file = pd.read_csv(dist_dir + '/' + dist)

        # If wide translate to long
        if matrix_format == 'wide':
            import_file = pd.melt(import_file,
                                  id_vars=['o_zone'],
                                  var_name='d_zone',
                                  value_name='dt',
                                  col_level=0)
            # TODO: Catch key error on o_zone
            import_file = import_file[import_file['dt']>0]
            # Also rename columns back to p/a
            import_file = import_file.rename(columns={'o_zone':'p_zone',
                                                      'd_zone':'a_zone'})
            
            import_file['p_zone'] = import_file['p_zone'].astype('int16')
            import_file['a_zone'] = import_file['a_zone'].astype('int16')

        if internal_only == True:
            # Try and get internal area lookup
            # Filter down input file based on filtered internal
            try:
                ia_files = os.listdir(model_folder)
                ia_file = [x for x in ia_files if model_name.lower()+'_internal_area' in x]
                internal_area = pd.read_csv(model_folder + '/' + ia_file[0])
            except:
                print('Couldn\'t fine internal area file, check pathing')
            else:
                # Filter to internal zones only
                unq_internal = internal_area.drop_duplicates(
                        ).unstack().reset_index(drop=True)
                import_file = import_file[
                        import_file['p_zone'].isin(unq_internal)]
                import_file = import_file[
                        import_file['a_zone'].isin(unq_internal)]

        # Do as list eventually
        # Try and get the purposes from the file
        if 'nhb' in file:
            origin = 'nhb'
        elif 'hb_' in file:
            origin = 'hb'
        else:
            print('Undefined origin')
            origin = 'undefined'

        # TODO: Figure out how to use these to define purpose
        commute_purpose = ['purpose1', 'purpose_1']
        business_purpose = ['purpose2', 'purpose_2',
                            'purpose12', 'purpose_12']

        try:
            purpose = import_file['purpose'].drop_duplicates().squeeze()
            mode = import_file['mode'].drop_duplicates().squeeze()
        # If that doesn't work, get them from the dist name
        # Note, these are different from the integer ones, need another loop to catch those.
        # TODO: This is actually pretty useful - functionalise to call in other bits        
        except:
            if 'commute' in file:
                purpose = 'commute'
            elif 'business' in file:
                purpose = 'business'
            elif 'other' in file:
                purpose = 'other'
            
            else:
                # Print warning
                print('Undefined purpose')
                purpose = 'undefined'
            if 'mode1' in file or 'mode_1' in file:
                mode = 1
            elif 'mode2' in file or 'mode_2' in file:
                mode = 2
            elif 'mode3' in file or 'mode_3' in file:
                mode = 3
            elif 'mode5' in file or 'mode_5' in file:
                mode = file
            elif 'mode6' in file or 'mode_6' in file:
                mode = 6
            else:
                print('Undefined mode')
                mode = 'undefined'

        # Car availability bodge, treat differently depending on which output folder     
        # TODO: Auto seg config based on passed params or config file
        if 'Compiled' in distributions:
            if 'car_availability1' in file or 'car_availability_1' in file or 'nca' in file:
                car_availability = 1
            if 'car_availability2' in file or 'car_availability_2' in file or '_ca' in file:
                car_availability = 2
            else:
                car_availability = None
        else:
            if 'car_availability1' in file or 'car_availability_1' in file:
                car_availability = 1
            elif 'car_availability2' in file or 'car_availability_2' in file:
                car_availability = 2
            else:
                car_availability = None

        # Safe var config for rows
        if car_availability == 1:
            c_a = False
        elif car_availability == 2:
            c_a = True
        else:
            c_a = None

        # TODO: The below is meaningless - fix
        if purpose == 1:
            j_p = 'commute'
        elif purpose in [2,12]:
            j_p = 'business'
        elif purpose in [3,4,5,6,7,8,13,14,15,16,18]:
            j_p = 'other'
        else:
            j_p = purpose

        # Get production numbers
        productions = import_file['dt'].sum()

        # Get intra-zonal metrics
        iz_trips = import_file[import_file['p_zone'] == import_file['a_zone']]
        iz_trips = iz_trips['dt'].sum()
        iz_perc = iz_trips/productions

        distance = dm.get_distance_and_costs(model_folder,
                                             request_type='distance',
                                             journey_purpose=j_p,
                                             direction=None,
                                             car_available=c_a,
                                             seed_intrazonal=True)

        atl = get_average_trip_length(model_folder,
                                         import_file,
                                         internal_distance = distance,
                                         join_distance=True)

        row = {'file':file,
               'origin':origin,
               'mode':mode,
               'purpose':purpose,
               'car_availability':car_availability,
               'atl':atl,
               'productions':productions,
               'intra_zonal_trips':iz_trips,
               'intra_zonal_perc':iz_perc}

        if trip_length_distribution_report:
            
            tld_dist = import_file.merge(distance,
                                         how='left',
                                         on=['p_zone', 'a_zone'])

            # Output trips by target trip length distribution
            dist_cols = ['dt','distance']
            dist_bins = tld_dist.reindex(dist_cols,axis=1)
            del(tld_dist)

            dist_bins['distance'] = np.ceil(dist_bins['distance'])
            dist_bins = dist_bins.groupby('distance').sum().reset_index()

            summary_name = file.replace('.csv','')

            dist_bins = dist_bins.rename(columns={'dt':summary_name})
            tld_ph.append(dist_bins)
            del(dist_bins)
            # End of distribution build report

        # Do sector report work
        if sector_report:
            # Get zone to sector lookup
            zone_to_sector = get_zone_to_sector_lookup(model_folder,
                                                       model_name,
                                                       best_match = True)

            # Join to O-Zone
            zts_o = zone_to_sector.rename(
                    columns={model_name.lower()+'_zone_id':'p_zone'})
            zts_file = import_file.merge(zts_o,
                                         how='left',
                                         on='p_zone')
            zts_file = zts_file.rename(
                    columns={
                            'tfn_sectors_zone_id':'p_sector'})
            
            # Join to D-Zone
            zts_d = zone_to_sector.rename(
                    columns={model_name.lower()+'_zone_id':'a_zone'})
            zts_file = zts_file.merge(zts_d,
                                      how='left',
                                      on='a_zone')
            zts_file = zts_file.rename(
                    columns={
                            'tfn_sectors_zone_id':'a_sector'})
            
            sector_segs = ['p_sector', 'a_sector']
            for seg in model_segments:
                sector_segs.append(seg)
            
            # Group and compile
            zts_file = zts_file.groupby(
                    sector_segs).sum().reset_index()
            zts_file = zts_file.drop(['p_zone','a_zone'],axis=1)

            # Put run description onto report
            # TODO: Add som sensible grouping segments here
            zts_file['run'] = file

            # Append sector report
            sector_ph.append(zts_file)
            del(zts_file)
            # End of sector report loop

        print(row)
        p_h.append(row)

    # Append initial betas
    # Get initial betas
    initial_betas = dm.get_initial_betas(path = model_folder,
                                         distribution_type = 'hb',
                                         model_name = model_name.lower())
    del(initial_betas['source'])

    # Get initial NHB betas
    initial_nhb_betas = dm.get_initial_betas(path = model_folder,
                                             distribution_type = 'nhb',
                                             model_name = model_name.lower())
    del(initial_nhb_betas['source'])

    all_initial_betas = pd.concat([initial_betas, initial_nhb_betas],
                                  sort=True)

    output = pd.DataFrame(p_h)

    export_name = distributions.lower().replace(' ','_')

    if calibration_report:
        output = all_initial_betas.merge(output,
                                         how = 'left',
                                         on = model_segments)

        output['cal'] = 1-abs(1-output['atl']/output['average_trip_length'])

        if write:
            output.to_csv((w_d +
                           '/' +
                           export_name +
                           '_actual_trip_lengths.csv'),
            index=False)

    if trip_length_distribution_report:
        tld = pd.concat(tld_ph, sort=False)
        tld = tld.fillna(0)
        tld = tld.groupby('distance').sum().reset_index()
        
        if write:
            print('Writing trip distribution report')
            tld.to_csv((w_d +
                        '/' +
                        export_name +
                        '_trip_distribution_report.csv'),
                index=False)
    else:
        tld = None

    if sector_report:
        sector_report = pd.concat(sector_ph, sort=False)
        # TODO: Get sector summary
        sector_summary = sector_report.copy()
        sector_summary = sector_report.reindex(
                ['p_sector',
                 'a_sector',
                 'dt'],axis=1).groupby(['p_sector',
                     'a_sector']).sum().reset_index()

        if write:
            print('Writing sector report')
            sector_report.to_csv((w_d +
                                  '/' +
                                  export_name +
                                  '_all_distribution_sector_report.csv'),
            index = False)
        
            sector_summary.to_csv((w_d +
                               export_name +
                               '_sector_report_summary.csv'),
            index=False)
    else:
        sector_report = None
        sector_summary = None

    return(output,
           tld)

def get_row_or_column_by_band(band_atl,
                              distance,
                              internal_pa,
                              axis = 0,
                              echo = False):
    """
    
    """
    # reset index, needed or not
    band_atl = band_atl.reset_index(drop=True)

    # Get min max for each
    if 'tlb_desc' in list(band_atl):
        # R built
        ph = band_atl['tlb_desc'].str.split('-', n=1, expand=True)
        band_atl['min'] = ph[0].str.replace('(', '')
        band_atl['max'] = ph[1].str.replace('[', '')
        band_atl['min'] = band_atl['min'].str.replace('(', '').values
        band_atl['max'] = band_atl['max'].str.replace(']', '').values
        del(ph)
    elif 'lower' in list(band_atl):
        # Python built
        # Convert bands to km
        band_atl['min'] = band_atl['lower']*1.61
        band_atl['max'] = band_atl['upper']*1.61

    dist_mat = []

    # Loop over rows in band_atl
    for index, row in band_atl.iterrows():

        # Get total distance
        band_mat = np.where((distance >= float(row['min'])) & (distance < float(row['max'])), distance, 0)

        # Get subset matrix for distance
        distance_bool = np.where(band_mat==0, band_mat, 1)
        band_trips = internal_pa * distance_bool

        # Get output parameters
        axis_trips = band_trips.sum(axis)

        dist_mat.append({'tlb_index': index,
                         'totals': axis_trips})

    return(dist_mat)
