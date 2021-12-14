# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 18:02:20 2019

@author: cruella
"""
import os
import sys
import pandas as pd
import numpy as np

# Get local modules
from normits_demand.utils import utils as nup # Folder management, reindexing, optimisation

_iter1_norms_folder = ('Y:/NorMITs Synthesiser/Norms/iter1/' +
                       'Distribution Outputs/Model PA')
_default_home_dir = 'Y:/NorMITs Synthesiser'

_default_model_folder = 'Y:/NorMITs Model Zone Lookups/Norms'
_default_import_folder = _iter1_norms_folder
_default_export_folder = 'Y:/NorMITs Model Zone Lookups/Norms/PA Matrices'

# Import and aggregate pa matrices based on import characteristics
# Long to wide format

# Import loop handling from dist model code

def hb_import_loop(import_folder,
                   mode_list = None,
                   purpose_list = None,
                   car_availability_list = None):
    """
    This function imports every csv from an import folder and appends them to a
    single DataFrame. Filters out distributions based on mode list and purpose
    list passed to function. The code is madness, frankly but it works.

    Parameters
    ----------
    import_folder:
        Target folder. Will work for any folder containing csvs.

    Returns
    ----------
    matrix_list:
        Compiled csvs.
    """
    target_dir = os.listdir(import_folder)

    # Turn mode list into something we can check outputs for
    if mode_list is not None:
        lookup_mode_list = []
        for mode in mode_list:
            lookup_mode_list.append('mode_' + str(mode))

    # Turn purpose list into something we can check outputs for
    if purpose_list is not None:
        lookup_purpose_list = []
        for purpose in purpose_list:
            lookup_purpose_list.append('purpose_' + str(purpose))

    # Turn purpose list into something we can check outputs for
    if car_availability_list is not None:
        lookup_car_availability_list = []
        for car_availability in car_availability_list:
            lookup_car_availability_list.append('car_availability_' +
                                                str(car_availability))

    # Filter out omitted modes
    if mode_list is not None:
        m_for_import = []
        for mode in lookup_mode_list:
            m_for_import.append([x for x in target_dir if mode in x])
        m_for_import = [inner for outer in m_for_import for inner in outer]
    else:
        m_for_import = target_dir

    # Filter out omitted purposes
    if purpose_list is not None:
        p_for_import = []
        for purpose in lookup_purpose_list:
            p_for_import.append([x for x in target_dir if purpose in x])
        p_for_import = [inner for outer in p_for_import for inner in outer]
    else:
        p_for_import = target_dir

    # Filter out omitted car_availability
    if car_availability_list is not None:
        ca_for_import = []
        for car_available in lookup_car_availability_list:
            ca_for_import.append([x for x in target_dir if car_available in x])
        ca_for_import = [inner for outer in ca_for_import for inner in outer]
    else:
        ca_for_import = target_dir

    # Go back to target dir and filter out all omitted modes and purposes
    import_list = []
    for import_path in target_dir:
        if (import_path in m_for_import
            and import_path in p_for_import
            and import_path in ca_for_import):
            import_list.append(import_path)

    matrix_list = []
    for matrix in import_list:
        # Import matrices and append to list
        print('Re-importing ' + matrix)
        matrix_list.append(pd.read_csv(import_folder +
                                       '/' +
                                       matrix))
     
    if len(import_list) > 1:
        matrix_list = pd.concat(matrix_list)
    else:
        matrix_list = matrix_list[0]

    # Filter to target matrices
    return(matrix_list)

def zones_to_pa_pairs(zone_heading, zone_list_1, zone_list_2=None):
    """
    """
    zone_p = zone_list_1.copy()
    zone_p['dummy'] = 1
    zone_p = zone_p.rename(columns={zone_heading:'p_zone'})
    
    if zone_list_2 is not None:
        zone_a = zone_list_2.copy()
        zone_a['dummy'] = 1
        zone_a = zone_a.rename(columns={zone_heading:'a_zone'})
    else:
        zone_a = zone_p.copy()
        zone_a = zone_a.rename(columns={'p_zone':'a_zone'})

    zone_to_zone = zone_p.merge(zone_a,
                                how = 'outer',
                                on = 'dummy').reset_index(
                                        drop=True)
    del(zone_to_zone['dummy'])

    return(zone_to_zone)

def define_zone_movements(ia_name,
                          internal_area,
                          external_area,
                          movement_type = 'i_to_i',
                          labels = False):
    """
    movement_type takes 'i_to_i, i_to_e, e_to_i, e_to_e, external, all'
    """
    # This has been audited. It craetes the right amount of zone to zones.
    # Build internal to internal lookup
    i_to_i = zones_to_pa_pairs(ia_name, internal_area)
    i_to_e = zones_to_pa_pairs(ia_name, internal_area, external_area)
    e_to_i = zones_to_pa_pairs(ia_name, external_area, internal_area)
    e_to_e = zones_to_pa_pairs(ia_name, external_area)

    if labels:
        i_to_i.loc[:,'m_type'] = 'i_to_i'
        i_to_e.loc[:,'m_type'] = 'i_to_e'
        e_to_i.loc[:,'m_type'] = 'e_to_i'
        e_to_e.loc[:,'m_type'] = 'e_to_e'

    if movement_type == 'i_to_i':
        return(i_to_i)
    elif movement_type == 'i_to_e':
        return(i_to_e)
    elif movement_type == 'e_to_i':
        return(e_to_i)
    elif movement_type == 'e_to_e':
        return(e_to_e)
    elif movement_type == 'external':
        return(pd.concat([i_to_e, e_to_i, e_to_e], sort=True))
    elif movement_type == 'all':
        return(pd.concat([i_to_i, i_to_e, e_to_i, e_to_e], sort=True))
    else:
        # TODO: Catch & handle
        return('Error!')

def define_internal_external_areas(model_folder = _default_model_folder):
    """
    This function imports an internal area definition from a model folder.

    Parameters
    ----------
    model_folder:
        Takes a model folder to look for an internal area definition.

    Returns
    ----------
    [0] internal_area:
        The internal area of a given model.

    [1] external_area:
        The external area of a given model.
    """
    file_sys = os.listdir(model_folder)
    internal_file = [x for x in file_sys if 'internal_area' in x][0]
    external_file = [x for x in file_sys if 'external_area' in x][0]
    
    internal_area = pd.read_csv(model_folder + '/' + internal_file)
    external_area = pd.read_csv(model_folder + '/' + external_file)

    return(internal_area, external_area)

def matrix_long_to_wide(long_matrix,
                        all_zone_movements,
                        merge_cols = ['p_zone', 'a_zone']):
    
    """
    Takes p/o, a/d & trips
    """

    long_matrix = all_zone_movements.merge(long_matrix,
                                           how = 'left',
                                           on = merge_cols)

    # TODO: Check len matches
    long_matrix['dt'] = long_matrix['dt'].fillna(0)

    # Go long to wide
    wide_matrix = long_matrix.pivot(index = merge_cols[0],
                                    columns = merge_cols[1],
                                    values = 'dt')

    return(wide_matrix)
    
def process_norms_hb_output(model_folder = _default_model_folder,
                            import_folder = _default_import_folder,
                            export_folder = _default_export_folder):
    """
    Should just be able to amend the spreadsheet and have it do it itself
    """

    os.chdir(_default_home_dir)
    
    matrix_build_params = pd.read_csv(model_folder + '/norms_matrix_params.csv')
    
    ia_areas = define_internal_external_areas(model_folder=model_folder)
    internal_area = ia_areas[0]
    ia_name = list(internal_area)[0]
    external_area = ia_areas[1]   
    del(ia_areas)
    
    # Get all internal zone movements
    all_zone_movements = define_zone_movements(ia_name,
                                               internal_area,
                                               external_area,
                                               movement_type = 'i_to_i',
                                               labels = False)

    # Reimport and aggregate for model parameters
    prod_list = []
    for index,row in matrix_build_params.iterrows():
        # Get export name
        export_name = row['export_name']
        print(export_name)

        # Import purposes from row and convert to integer
        purpose = list(map(int, row['ns_purpose'].split(',')))
        # Get car availability from row
        car_availability = [row['ns_ca']]

        # Hard code 6 for rail only
        compilation = hb_import_loop(import_folder,
                                     mode_list = [6],
                                     purpose_list = purpose,
                                     car_availability_list = car_availability)

        index_cols = list(compilation)
        index_cols.remove('mode')
        index_cols.remove('purpose')
        index_cols.remove('car_availability')
        compilation = compilation.reindex(index_cols, axis=1)
        index_cols.remove('dt')
        compilation = compilation.groupby(index_cols).sum().reset_index()
        
        # Add total productions to placeholder matrix
        prod_list.append(compilation['dt'].sum())
        print('total_productions ' + str(sum(prod_list)))

        # Pivot to wide.
        compilation = matrix_long_to_wide(compilation,
                                          all_zone_movements,
                                          merge_cols = ['p_zone', 'a_zone'])

        # Export
        export_path = (export_folder +
                       '/' + export_name + '_mode6' + '_internal.csv')
        compilation.to_csv(export_path)

    return(prod_list)

def get_zone_to_sector_lookup(model_folder,
                              model_name,
                              best_match = True):
    
    """
    """
    
    file_sys = os.listdir(model_folder)
    zone_to_sector = [x for x in file_sys if model_name.lower() in x]
    zone_to_sector = [x for x in zone_to_sector if 'tfn_sector' in x]
                
    if len(zone_to_sector)==0:
        print('No sector lookup in model folder, run one')
        return(None)

    zone_to_sector = zone_to_sector[0]

    zone_to_sector = pd.read_csv(model_folder + '/' + zone_to_sector)
                
    if best_match:
        model_col = model_name.lower()+'_zone_id'
        overlap_col = 'overlap_'+model_name.lower()+'_split_factor'

        reindex_cols = ['tfn_sectors_zone_id',
                        model_col,
                        overlap_col]

        zone_to_sector = zone_to_sector.reindex(reindex_cols, axis=1)
        max_zts = zone_to_sector.groupby([model_col],
                                         sort=False)[
                                         overlap_col].max().reset_index()
        max_zts['flag'] = 1
        zone_to_sector = zone_to_sector.merge(max_zts,
                                              how = 'left',
                                              on = [(model_name.lower() + '_zone_id'),
                                                    'overlap_norms_2015_split_factor'])
        zone_to_sector = zone_to_sector[zone_to_sector['flag'] == 1]

        # Quick echo of worst overlap
        lco = zone_to_sector[overlap_col].min()                    
        print('Least convincing overlap ' + str(lco))
        if lco < .2:
            print('Please check lookup provided, this is very poor')
        zone_to_sector = zone_to_sector.drop([overlap_col, 'flag'], axis=1)
                
    return(zone_to_sector)


def distribution_report(file_drive='Y:/',
                        model_name='Norms_2015',
                        iteration='iter2',
                        model_segments = ['purpose', 'mode', 'car_availability'],
                        distributions = 'Distribution Outputs/PA Matrices',
                        matrix_format = 'long', # ['long', 'wide']
                        report_tp = '24hr', # ['24hr', 'tp', '12hr']
                        mode_subset = None,
                        internal_reports = True,
                        write = True):

    """
    Designed to work with pure outputs - ie. not aggregations.
    
    """
    # TODO: Get to pull purpose from compilations properly

    # Parameter handling
    if matrix_format not in ['long', 'wide']:
        ValueError('Matrix format should be \'wide\' or \'long\'')
    if report_tp not in ['24hr', 'tp', '12hr']:
        ValueError('Report time period should be \'24hr\' or \'tp\'')
    
    # Handle difference in multiprocessing name
    if 'mode' in model_segments:
        mode_col = 'mode'
    elif 'm' in model_segments:
        mode_col = 'm'
    
    if 'purpose' in model_segments:
        purpose_col = 'purpose'
    elif 'p' in model_segments:
        purpose_col = 'p'

    # TODO: One for the mode subset

    w_d = (file_drive +
           'NorMITs Synthesiser/' +
           model_name +
           '/' +
           iteration +
           '/Distribution Outputs/Logs & Reports')

    export_name = distributions.lower().replace(' ', '_')
    export_name = export_name.replace('/', '_')

    # get distributions
    dist_dir = (file_drive +
                '/NorMITs Synthesiser/' +
                model_name +
                '/' +
                iteration +
                '/' +
                distributions)

    lookup_folder = (file_drive +
                     '/NorMITs Synthesiser/' +
                     model_name +
                     '/Model Zone Lookups')

    # List dists
    dists = os.listdir(dist_dir)
    
    # If these are compiled, purpose needs to be compiled differently
    if 'Compiled' in dist_dir:
        compiled_purpose = True
    else:
        compiled_purpose = False

    # Filter out non csv (subfolders for instance)
    dists = [x for x in dists if '.csv' in x]
    # Mode filter
    if mode_subset is not None:
        dist_ph = []
        for mode in mode_subset:
            temp = [x for x in dists if mode in x]
            dist_ph.append(temp)
        dists = [x for y in dist_ph for x in y]

    # Establish matrix type
    # Lots of assumptions going into this around NorMITs output formats
    # Need to keep consistent & won't work with other distributions
    if 'OD' in distributions or 'od' in distributions:
        matrix_type = 'od'
        origin_col = 'o_zone'
        destination_col = 'd_zone'
    elif 'PA' in distributions or 'pa' in distributions:
        matrix_type = 'pa'
        origin_col = 'p_zone'
        destination_col = 'a_zone'
        # Filter contents of folder based on target report, if PA
        if report_tp == '24hr' or report_tp == '12hr':
            dists = [x for x in dists if 'tp' not in x]
            dists = [x for x in dists if 'internal' not in x]
        else:
            dists = [x for x in dists if 'tp' in x]
            dists = [x for x in dists if 'internal' not in x]

    # If 12 hour drop off peak
    if report_tp == '12hr':
        dists = [x for x in dists if 'tp4' not in x]
        # Make it explicit in the export name too
        export_name = (export_name + '_12hr')

    else:
        print('This report function may not be designed for these matrices')
        matrix_type = 'unknown'

    all_dists = []
    for dist in dists:
        print(dist) 
        # Set params
        # Establish hb/nhb
        if dist.startswith('hb'):
            origin = 'hb'
        elif dist.startswith('nhb'):
            origin = 'nhb'

        # Get tp
        if report_tp == 'tp':
            if 'tp1' in dist:
                tp = 1
            elif 'tp2' in dist:
                tp = 2
            elif 'tp3' in dist:
                tp = 3
            elif 'tp4' in dist:
                tp = 4

        # All other params
        report_params = {}
        for param in model_segments:
            print(param)
            
            purpose_types = ['purpose', 'p']
            # Underscore fix for p being weird
            if param in purpose_types:
                temp_param = '_p'
            else:
                temp_param = param
            # Underscore fix for mode as m
            if param == 'm':
                temp_param = '_m'
            else:
                temp_param = param
    
            # Get the character after the segment name
            p_index = dist.find(temp_param)
            # Capture char and one after in case longer
            value = dist[p_index+len(temp_param):p_index+len(temp_param)+2]
            # Get rid of any extra fluff
            value = value.replace('_','')
            value = value.replace('.','')
            
            if param in purpose_types and compiled_purpose:
                if 'commute' in dist:
                    value = 'commute'
                if 'business' in dist:
                    value = 'business'
                if 'other' in dist:
                    value = 'other'
            
            ca_types = ['ca', 'car_availability']
            
            if param in ca_types:
                if 'nca' in dist:
                    value = 1
                elif '_ca' in dist:
                    value = 2
                else:
                    value = 'none'
            print(origin,param,value)
            report_params.update({param:value})

        # Import file
        ph = pd.read_csv(dist_dir + '/' + dist)

        # Pivot to long if required
        if matrix_format == 'wide':
            if matrix_type == 'od':
                ph = pd.melt(ph,
                             id_vars=['o_zone'],
                             var_name='d_zone',
                             value_name='dt',
                             col_level=0)
                ph = ph[ph['dt']>0]

                ph['o_zone'] = ph['o_zone'].astype('int16')
                ph['d_zone'] = ph['d_zone'].astype('int16')

            elif matrix_type == 'pa':
                ph = pd.melt(ph,
                             id_vars=['p_zone'],
                             var_name='a_zone',
                             value_name='dt',
                             col_level=0)
                ph = ph[ph['dt']>0]

                ph['p_zone'] = ph['p_zone'].astype('int16')
                ph['a_zone'] = ph['a_zone'].astype('int16')
                # Wide to long handled
        
        # Add segments back on to long data
        for name, value in report_params.items():
            ph[name] = value

        # Add other info on
        if report_tp == 'tp':
            ph['time'] = tp
        ph['origin'] = origin

        all_dists.append(ph)

    distribution_report = pd.concat(all_dists, sort=True)

    # TODO: Would be very easy to add sector reports in here.
    
    summary_groups = ['origin', mode_col, purpose_col]
    summary_cols = summary_groups.copy()
    summary_cols.append('dt')
    summary_report = distribution_report.reindex(
            summary_cols,axis=1).groupby(
                    summary_groups).sum().reset_index(drop=True)

    origin_groups = summary_groups.copy()
    origin_groups.insert(0, origin_col)
    origin_cols = origin_groups.copy()
    origin_cols.append('dt')
    origin_report = distribution_report.reindex(
            origin_cols, axis=1).groupby(
                    origin_groups).sum().reset_index()

    segment_groups = ['origin']
    for seg in model_segments:
        segment_groups.append(seg)
    segment_cols = segment_groups.copy()
    segment_cols.append('dt')
    segment_report = distribution_report.reindex(
            segment_cols, axis=1).groupby(
                    segment_groups).sum().reset_index()

    if report_tp == 'tp':
        print('steps to aggregate both reports on tp')
        tp_summary_groups = summary_groups.copy()
        tp_summary_groups.append('time')
        tp_summary_cols = tp_summary_groups.copy()
        tp_summary_cols.append('dt')
        summary_report = distribution_report.reindex(
                tp_summary_cols,axis=1).groupby(
                        tp_summary_groups).sum().reset_index()

    if internal_reports == True:
        # Try and get internal area lookup
        # Build segment report based on filtered internal
        try:
            ia_files = os.listdir(lookup_folder)
            ia_file = [x for x in ia_files if model_name.lower()+'_internal_area' in x]
            internal_area = pd.read_csv(lookup_folder + '/' + ia_file[0])
        except:
            print('Couldn\'t fine internal area file, check pathing')
        else:
            unq_internal = internal_area.drop_duplicates().unstack().reset_index(drop=True)
            internal_segment_report = distribution_report.copy()

            internal_segment_report = internal_segment_report[
                    internal_segment_report[origin_col].isin(unq_internal)]
            internal_segment_report = internal_segment_report[
                    internal_segment_report[destination_col].isin(unq_internal)]

            zone_internal_segment_report = internal_segment_report.copy()

            internal_segment_report = internal_segment_report.reindex(
                    segment_cols, axis=1).groupby(
                            segment_groups).sum().reset_index()

            if write:
                print('Writing internal summary report')
                internal_segment_report.to_csv((w_d +
                                                '/' +
                                                export_name +
                                                '_internal_segment_report.csv'),
                index = False)
    else:
        zone_internal_segment_report = None

    if write:
        print('Writing summary reports')

        summary_report.to_csv((w_d +
                               '/' +
                               export_name +
                               '_summary_report.csv'), index=False)

        origin_report.to_csv((w_d +
                               '/' +
                               export_name +
                               '_origin_report.csv'), index=False)

        segment_report.to_csv((w_d +
                               '/' +
                               export_name +
                               '_segment_report.csv'), index = False)

    return(summary_report,
           origin_report,
           segment_report,
           zone_internal_segment_report)
    
