# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 14:22:56 2020

@author: cruella
"""

import os
import pandas as pd
import numpy as np
#import geopandas as gpd

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

    
def run_production_reports(file_drive='Y:/',
                           model_name='Norms_2015',
                           iteration='iter2',
                           production_type = 'hb',
                           model_segments = ['mode', 'purpose', 'car_availability'],
                           internal_only = False,
                           write = True):

    """
    Function to export production summary report for hb and nhb.
    """

    # Define productions path & validate passed prod type var
    if production_type.lower() == 'hb':
        p_desc = 'hb_'
    elif production_type.lower() == 'nhb':
        p_desc = 'nhb_'
    else:
        ValueError('Production type must be \'hb\' or \'nhb\'')
    productions_path = (file_drive +
                        'NorMITs Synthesiser/' +
                        model_name + '/' +
                        iteration + '/' +
                        'Production Outputs/' +
                        p_desc +
                        'productions_' +
                        model_name.lower() +
                        '.csv')

    # Import productions
    productions = pd.read_csv(productions_path)

    # Define lookup folder
    lookup_folder = (file_drive +
                     '/NorMITs Synthesiser/' +
                     model_name +
                     '/Model Zone Lookups')

    export_folder = (file_drive +
                     '/NorMITs Synthesiser/' +
                     model_name +
                     '/' +
                     iteration +
                     '/Production Outputs/Production Run Logs')

    # TODO if no productions, get productions.
    # Make a direcory under it.
    # Always assuming zoneid will be first col.
    # If it isn't you've got bigger problems.
    col_list = list(productions)

    # TODO: Individual write outs
    unq_seg_cols = [col_list[0]]
    for ts in model_segments:
        unq_seg_cols.append(ts)

    unq_segments = productions.reindex(
            unq_seg_cols,
            axis=1).drop_duplicates(
                    ).reset_index(drop=True)

    # Placeholder list
    unq_seg_cols_trips = unq_seg_cols.copy()
    unq_seg_cols_trips.append('trips')

    # All segment report
    all_segment_report = productions.reindex(
            unq_seg_cols_trips,
            axis=1).groupby(
                    unq_seg_cols).sum(
                            ).reset_index()

    # Append internal external
    try:
        lookup_dir = os.listdir(lookup_folder)
        ia_file = [x for x in lookup_dir if model_name.lower()+'_internal_area' in x]
        internal_area = pd.read_csv(lookup_folder + '/' + ia_file[0])
        ie_factors = [x for x in lookup_dir if model_name.lower()+'_ie_factors' in x]
        ie_factors = pd.read_csv(lookup_folder + '/' + ie_factors[0])
        ie_factors = ie_factors[ie_factors['m_type']=='i_to_i']
        ie_factors = ie_factors.drop('m_type', axis=1)
        
        if list(ie_factors)[0] != list(all_segment_report)[0]:
            ie_factors = ie_factors.rename(
                    columns={list(
                            ie_factors)[0]:list(
                                    all_segment_report)[0]})
    except:
        print('Couldn\'t find internal area file, check pathing')

    # If the above fails this will crash anyway
    internal_area['internal'] = 1

    if list(internal_area)[0] != list(all_segment_report)[0]:
        internal_area = internal_area.rename(
                columns={list(
                        internal_area)[0]:list(
                                all_segment_report)[0]})

    all_segment_report = all_segment_report.merge(
            internal_area,
            how = 'left',
            on = col_list[0])

    export_string = (export_folder +
                     '/all_segment_report_' +
                     production_type +
                     '.csv')

    if internal_only:
        all_segment_report = all_segment_report[
                all_segment_report['internal']==1]

        all_segment_report = all_segment_report.merge(ie_factors,
                                                      how='left',
                                                      on=list(all_segment_report)[0])
        all_segment_report['trips'] = (all_segment_report['trips']*
                          all_segment_report['ie_factor'])
        all_segment_report = all_segment_report.drop('ie_factor', axis=1)

        export_string = export_string.replace('report_',
                                              'report_internal_')

    if write:
        all_segment_report.to_csv(
                export_string,
                index=False)

    return(all_segment_report)

def lad_from_to_report(file_drive='Y:/',
                       model_name='Norms_2015',
                       model_shp_path = _default_shp_path,
                       iteration='iter2',
                       model_segments = ['car_availability'],
                       distributions = 'OD Matrices',
                       matrix_format = 'wide',
                       report_tp = '24hr',
                       mode_subset = ['6'],
                       internal_only = True,
                       bind_segments = True,
                       write = True):

    """
    Reports for turning OD into lad level reports. For dev report mapping.
    """
    # TODO: Segment handling doesn't quite work - needs revisiting
    # Not sure this is accurate now - it was just about behaving by the end.

    # Define OD/PA params
    if 'od ' in distributions or 'OD ' in distributions:
        from_name = 'o'
        to_name = 'd'
    elif 'pa ' in distributions or 'PA ' in distributions:
        from_name = 'p'
        to_name = 'a'

    # Define model folder
    model_folder = (file_drive + 'NorMITs Synthesiser/'+ model_name + '/Model Zone Lookups/')

    # Define write folder
    w_d = (file_drive +
           'NorMITs Synthesiser/' +
           model_name +
           '/' +
           iteration +
           '/Distribution Outputs/Logs & Reports')

    # Always need mode
    report_segments = model_segments.copy()
    model_segments.append('mode')

    # Get segments (full reimport)
    # TODO: This is pretty inefficient - should take a mode subset - IT DOES USE IT
    from_to_segment_report = mp.distribution_report(file_drive,
                                                    model_name,
                                                    iteration,
                                                    model_segments,
                                                    distributions,
                                                    matrix_format,
                                                    report_tp,
                                                    mode_subset = mode_subset,
                                                    internal_reports = True,
                                                    write = False)

    from_to_trips = from_to_segment_report[3].copy()
    # Mode subset - list like isnt working
    from_to_trips = from_to_trips[from_to_trips['mode'].isin(mode_subset)]

    # Get distinct segments
    distinct_segs = from_to_trips.reindex(report_segments,axis=1).drop_duplicates().reset_index(drop=True)

    del(from_to_segment_report)

    from_index = [(from_name + '_zone')]
    from_groups = from_index.copy()
    to_index = [(to_name + '_zone')]
    to_groups = to_index.copy()
    for seg in report_segments:
        from_index.append(seg)
        from_groups.append(seg)
        to_index.append(seg)
        to_groups.append(seg)
    from_index.append('dt')
    to_index.append('dt')

    from_trips = from_to_trips.reindex(
            from_index,
            axis=1).groupby(
                    from_groups).sum(
                            ).reset_index() 
    to_trips = from_to_trips.reindex(
            to_index,
            axis=1).groupby(
                    to_groups).sum(
                            ).reset_index()

    print(list(from_trips))
    print(list(to_trips))
    # good_from = from_trips.copy()
    # good_to = to_trips.copy()

    # get lad lookups
    lad_lookup = mp.get_zone_to_lad_lookup(model_folder,
                                           model_name.lower())
    
    # Add area col on zones
    zone_shp = gpd.read_file(model_shp_path)
    zone_shp = zone_shp.rename(columns={'ZoneID':(model_name.lower() + '_zone_id')})

    # Here here here - divide zone shp to get it in thousands - do not sqrt
    zone_shp[(model_name.lower() + '_zone_area')] = zone_shp.area
    # Smaller area value pls
    zone_shp[(model_name.lower() + '_zone_area')] = zone_shp[(model_name.lower() + '_zone_area')]/1000
    zone_shp = zone_shp.reindex([(model_name.lower() + '_zone_id'),
                                 (model_name.lower() + '_zone_area')], axis=1)
    lad_lookup = lad_lookup.merge(zone_shp,
                                  how='left',
                                  on=(model_name.lower() + '_zone_id'))
    del(zone_shp)

    # Join to O-Zone
    # TODO: Go from here - add division by area
    lad_from = lad_lookup.rename(columns={'lad_zone_id':(from_name + '_lad'),
                                  (model_name.lower()+'_zone_id'):(from_name + '_zone')})
    from_trips = from_trips.merge(lad_from,
                                  on = (from_name + '_zone'))

    # TODO: Flexible segments
    from_groups = [(from_name + '_lad'), 'car_availability']
    from_index = from_groups.copy()
    from_index.append(model_name.lower() + '_zone_area')
    from_index.append('dt')

    lad_from = from_trips.reindex(
            from_index,
            axis=1).groupby(
                    from_groups).sum().reset_index()
    # Get trips/area
    lad_from['dt_over_area'] = (lad_from['dt']/
            lad_from[model_name.lower() + '_zone_area'])
    del(lad_from[(model_name.lower() + '_zone_area')])

    # Join to D-Zone
    # TODO: Go from here - add division by area
    lad_to = lad_lookup.rename(columns={'lad_zone_id':(to_name + '_lad'),
                                        (model_name.lower()+'_zone_id'):(to_name + '_zone')})
    to_trips = to_trips.merge(lad_to,
                              on = (to_name + '_zone'))

    # TODO: Flexible segments
    to_groups = [(to_name + '_lad'), 'car_availability']
    to_index = to_groups.copy()
    to_index.append(model_name.lower() + '_zone_area')
    to_index.append('dt')

    lad_to = to_trips.reindex(to_index,axis=1).groupby(to_groups).sum().reset_index()

    # Get trips/area
    lad_to['dt_over_area'] = (lad_from['dt']/
            lad_to[model_name.lower() + '_zone_area'])
    del(lad_to[(model_name.lower() + '_zone_area')])

    segment_ph_from = []
    segment_ph_to = []

    # TODO: Need to rewrite this with multiple cols
    for index, seg in distinct_segs.iterrows():
        from_ph = lad_from.copy()
        to_ph = lad_to.copy()
        # Seg loop
        seg_name = 'lad_report_'
        for col in list(distinct_segs):
            seg_desc = (col + '_' + seg[col]) 
            from_ph = from_ph[from_ph[col] == seg[col]]
            to_ph = to_ph[to_ph[col] == seg[col]]
        segment_ph_from.append({(seg_name+seg_desc+'_from'):from_ph})
        segment_ph_to.append({(seg_name+seg_desc+'_to'):to_ph})

    # Export loop
    if write:
        for report in segment_ph_from:
            for export_name, data in report.items():
                data.to_csv(w_d + '/' + export_name + '.csv', index=False)
        for report in segment_ph_to:
            for export_name, data in report.items():
                data.to_csv(w_d + '/' + export_name + '.csv', index=False)

    return segment_ph_from, segment_ph_to

def get_trip_length(distance, demand):
    """
    Take trip length as matrix
    Take pa as matrix
    Trim distance if needed
    Return average trip length
    Return trip length distribution vector
    
    distance = distance matrix as numpy ndarray
    internal_pa = demand as 
    """

    # TODO: Just copy that bit below
    global_trips = demand.sum(axis=1).sum()
    global_distance = demand * distance

    global_atl = global_distance.sum(axis=1).sum() / global_trips

    return global_atl


def get_trip_length_by_band(band_atl,
                            distance,
                            internal_pa):
    """
    Take ttl by band, return atl by band.
    """
    # TODO: Drop averages of nothing in trip length band
    # reset index, needed or not
    band_atl = band_atl.reset_index(drop=True)

    # Get global trips
    global_trips = internal_pa.sum(axis=1).sum()

    # Get global atl
    global_atl = get_trip_length(distance,
                                 internal_pa)

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
        # Convert bands to km
        band_atl['min'] = band_atl['lower']*1.61
        band_atl['max'] = band_atl['upper']*1.61

    dist_mat = []
    bs_mat = []

    # Loop over rows in band_atl
    for index, row in band_atl.iterrows():

        # Get total distance
        band_mat = np.where((distance >= float(row['min'])) & (distance < float(row['max'])), distance, 0)
        total_distance = (internal_pa * band_mat).sum()

        # Get subset matrix for distance
        distance_bool = np.where(band_mat==0, band_mat, 1)
        band_trips = internal_pa * distance_bool

        # Get output parameters
        if isinstance(band_trips, pd.DataFrame):
            band_trips = band_trips.values
        total_trips = np.sum(band_trips)
        band_share = total_trips/global_trips

        # Get average trip length
        if total_trips > 0:
            atl = total_distance / total_trips
        else:
            atl = 0

        dist_mat.append({'tlb_index': index,
                         'atl': atl,
                         'ttl': row['ave_km']})
        bs_mat.append({'tlb_index': index,
                       'bs': band_share,
                       'tbs': row['band_share']})

    # TODO: Handle on output side to avoid error
    dist_mat = pd.DataFrame(dist_mat)
    dist_mat = dist_mat.reindex(['tlb_index', 'ttl', 'atl'], axis=1)

    bs_mat = pd.DataFrame(bs_mat)
    bs_mat = bs_mat.reindex(['tlb_index', 'tbs', 'bs'], axis=1)

    return dist_mat, bs_mat, global_atl


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
