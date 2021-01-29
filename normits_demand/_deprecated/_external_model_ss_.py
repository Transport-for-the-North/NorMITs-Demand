# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 13:45:03 2020

@author: cruella
"""

import pandas as pd
import reports_audits as ra
import distribution_model as dm
import os # File ops
import sys # File ops & sys config

# TODO: Object layer
# TODO: Update calls from distribution_model

_UTILS_PATH = ('C:/Users/' +
               os.getlogin() +
               '/Documents/GitHub/Normits-Utils')
sys.path.append(_UTILS_PATH)
import normits_utils as nup # Folder management, reindexing, optimisation

# Manually define some elements
# external_commute_pa = pd.read_csv(r"Y:\NorMITs Synthesiser\Norms_2015\iter3\Distribution Outputs\External Distributions\external_purpose_1.csv")
# external_productions = pd.read_csv(r"Y:\NorMITs Synthesiser\Norms_2015\iter3\Production Outputs\export_nhb_productions_norms_2015.csv")
# print(external_productions)
# internal_area = pd.read_csv(r"Y:\NorMITs Synthesiser\Norms_2015\Model Zone Lookups\norms_2015_internal_area.csv")
# external_area = pd.read_csv(r"Y:\NorMITs Synthesiser\Norms_2015\Model Zone Lookups\norms_2015_external_area.csv")
# model_zone_lookups = r"Y:\NorMITs Synthesiser\Norms_2015\Model Zone Lookups"

def adjust_external_trips(external_movements,
                          trip_length_bands,
                          distance_bands,
                          model_zone_lookups):
    # Uses external movements, trip length bands
    """
    Adjust distributions to meet bin length by mode.
    """

    # Get unq mode
    unq_mode = external_movements['mode'].drop_duplicates(
            ).squeeze().sort_values().reset_index(drop=True)

    # TODO: Get segment splits
    # TODO: Drop unneeded segments

    # Get unq bands from distance banding
    unq_bands = distance_bands['distance_band'].drop_duplicates(
            ).squeeze().reset_index(drop=True)

    # Join distance
    external_movements = external_movements.merge(distance_bands,
                                                  how='inner',
                                                  on=['p_zone','a_zone'])

    # Set placeholder to capture mode totals
    mode_bin = []
    # Set loop off running by mode
    for mode in unq_mode:
        print(mode)
        # Subset mode
        mode_subset = external_movements[
                external_movements['mode']==mode].sort_values(
                by=['distance']).reset_index(drop=True)

        benchmark = mode_subset['dt'].sum()
        print('Initial productions ' + str(benchmark))

        mode_ttl = trip_length_bands[trip_length_bands['mode']==mode] # tll_external
        """
        print(mode_ttl)
        answer = input('Do you want to continue?:')
        if answer.lower().startswith("y"):
            print("ok, carry on then")
        elif answer.lower().startswith("n"):
            print("ok, sayonnara")
            exit()
        """
        # Join targets
        mode_subset = mode_subset.merge(mode_ttl,
                                        how='left',
                                        on=['purpose',
                                            'mode',
                                            'distance_band'])
        mode_subset = mode_subset.rename(
                columns={'average_trip_length':'ttl'})

        # Deal with null segments
        good_segments = mode_subset[~mode_subset['ttl'].isnull()]
        gs_trips = good_segments['dt'].sum()
        bad_segments = mode_subset[mode_subset['ttl'].isnull()]
        bs_trips = bad_segments['dt'].sum()
        increase = (gs_trips + bs_trips)/gs_trips
        good_segments.loc[:,'dt'] = good_segments.loc[:,'dt'] * increase
        mode_subset = good_segments.copy()
        del(good_segments, bad_segments)

        # Get unq_bands
        unq_bands = mode_subset[
                'distance_band'].drop_duplicates(
                )

        # Set placeholder band for capturing atls
        band_bin = []
        # Get average trip length by band
        for band in unq_bands:
            print(band)
            subset = mode_subset[
                    mode_subset['distance_band']==band].copy()
            # print(subset.columns.tolist())
            # print(subset.head(n=5))
            # subset_pre_total = subset.groupby(['mode','purpose','distance_band'])['dt'].sum().reset_index()
            # subset_pre_total.to_csv(r'C:\Users\scartissue\Desktop\Pre_Totals.csv', mode='a', header=False)
            # external_pro_check2 = external_productions.groupby(['mode','purpose'])['trips'].sum().reset_index()
            # answer = input('export?:')
            # Manual adjustment to stop trying to hit really low trip lengths
            ttl = subset['ttl'].drop_duplicates().squeeze()
            if ttl < 5:
                subset['ttl'] = 5
                ttl = subset['ttl'].drop_duplicates().squeeze()

            atl = ra.get_average_trip_length(model_zone_lookups,
                                             distribution = subset,
                                             join_distance=False,
                                             join_type='inner')
            if ttl < atl:
                # Target trip length longer than actual
                # Increase volume of longer trips
                while ttl < atl:
                    # Print total p for audit
                    print(subset['dt'].sum())
                    # Subset longer than ttl
                    ttl_more = subset[subset['distance'] > ttl].copy()
                    # Get trips longer than ttl
                    dropped_trips = ttl_more['dt'].sum()/100
                    # Drop 1% of trips
                    ttl_more['dt'] = ttl_more['dt'] * .99
                    # Subset shorter than ttl
                    ttl_less = subset[subset['distance'] <= ttl].copy()
                    # Get trips shorter than atl
                    less_trips = ttl_less['dt'].sum()
                    # Get factor to increase less trips by
                    less_increase = (less_trips + dropped_trips)/less_trips
                    # Increase less trips
                    ttl_less['dt'] = ttl_less['dt']*less_increase
                    # Re-join
                    subset = pd.concat([ttl_more, ttl_less])
                    atl = ra.get_average_trip_length(model_zone_lookups,
                                                     distribution = subset,
                                                     join_distance=False)
                    print(atl, 'aiming for', ttl)
                    print(subset['dt'].sum())
                    # TODO: Check dropped trips have gone back in & the total is intact
                    # End of loop
                del(ttl_more, ttl_less)

            elif ttl > atl:
                # Target trip length shorter than actual
                # Increase volume of shorter trips
                while ttl > atl:
                    # Print total p for audit
                    print(subset['dt'].sum())
                    # Subset shorter than ttl
                    ttl_less = subset[subset['distance'] <= ttl].copy()
                    # Get trips shorter than ttl
                    dropped_trips = ttl_less['dt'].sum()/100
                    # Drop 1% of trips
                    ttl_less['dt'] = ttl_less['dt'] * .99
                    # Subset longer than ttl
                    ttl_more = subset[subset['distance'] > ttl].copy()
                    # Get out if no longer trips, unadjustable with current method
                    if len(ttl_more) == 0:
                        print('No longer trips in zone to zone, unadjustable')
                        break
                    # Get trips longer than ttl
                    more_trips = ttl_more['dt'].sum()
                    # Get factor to increase more trips by
                    more_increase = (more_trips + dropped_trips)/more_trips
                    # Increase more trips
                    ttl_more['dt'] = ttl_more['dt']*more_increase
                    # TODO: Check for passengers, see above
                    subset = pd.concat([ttl_more,ttl_less])
                    atl = ra.get_average_trip_length(model_zone_lookups,
                                                     distribution = subset,
                                                     join_distance=False)
                    print(atl, 'aiming for', ttl)
                    print(subset['dt'].sum())
                    # TODO: Check dropped trips have gone back in & the total is intact
                    # End of loop

                del(ttl_more, ttl_less)

            else:
                # These two are already the same
                print('already matched')
                # End of if

            # subset_post_total = subset.groupby(['mode','purpose','distance_band'])['dt'].sum().reset_index()
            # subset_post_total.to_csv(r'C:\Users\scartissue\Desktop\Post_Totals.csv', mode='a', header=False)
            band_bin.append(subset)

            # End of for
        band_dat = pd.concat(band_bin)
        # End of for
        # Delete the band bin or it will fill up more
        del(band_bin)
        mode_bin.append(band_dat)
        # End of for
    # Concat the band bin
    mode_dat = pd.concat(mode_bin)
    # Drop meta cols
    drop_cols = ['distance', 'distance_band', 'ttl', 'total_distance']
    mode_dat = mode_dat.drop(drop_cols,axis=1)

    return(mode_dat)

def factor_distributed_trips(external_movements, target_productions):
    """
    Factors the commute pa matrix to a set of target productions

    Takes single zone productions only - not a big aggregated mess
    """
    
    tdt_cols = ['p_zone', 'purpose', 'mode', 'dt']
    # Get productions in external movements
    total_dt = external_movements.reindex(tdt_cols,
                                          axis=1).groupby(
                                                  ['p_zone', 'purpose', 'mode'])['dt'].sum(
                                                          ).reset_index()
    # Check dataframe contents
    # print('total_dt1 headings : ',total_dt.columns.tolist())
    # print('total_dt1 content: ',total_dt.head(n=5))
    # wait = input("PRESS ENTER TO CONTINUE.")
    total_dt = total_dt.rename(columns={'dt':'total_dt'})
    # Check dataframe contents
    # print('total_dt2 headings : ',total_dt.columns.tolist())
    # print('total_dt2 content: ',total_dt.head(n=5))
    # wait = input("PRESS ENTER TO CONTINUE.")
    # Join total productions on to external movements to factor
    external_movements = external_movements.merge(total_dt,
                                                how='left',
                                                on=['p_zone', 'purpose', 'mode'])
    # Check dataframe contents
    # print('external_movements1 headings : ',external_movements.columns.tolist())
    # print('external_movements1 content: ',external_movements.head(n=5))
    # wait = input("PRESS ENTER TO CONTINUE.")
    external_movements['ds'] = (external_movements['dt'].values/
                     external_movements['total_dt'])
    # Check dataframe contents
    # print('external_movements2 headings : ',external_movements.columns.tolist())
    # print('external_movements2 content: ',external_movements.head(n=5))
    # wait = input("PRESS ENTER TO CONTINUE.")
    # Adjust distribution to meet target productions
    external_movements = external_movements.merge(target_productions,
                                                  how='left',
                                                  on=['p_zone', 'purpose', 'mode'])
    # Check dataframe contents
    # print('external_movements3 headings : ',external_movements.columns.tolist())
    # print('external_movements3 content: ',external_movements.head(n=5))
    # external_movements_mid_total = external_movements.groupby(['purpose','mode'])['dt','total_productions'].sum().reset_index()
    # external_movements_mid_total.to_csv(r'C:\Users\scartissue\Desktop\6-Mid_Externalmov_Totals.csv', mode='a', header=False)
    #wait = input("PRESS ENTER TO CONTINUE.")
    external_movements['da'] = (external_movements['ds'] *
                      external_movements['total_productions'])
    # Check dataframe contents
    # print('external_movements4 headings : ',external_movements.columns.tolist())
    # print('external_movements4 content: ',external_movements.head(n=5))
    # external_movements_post_total = external_movements.groupby(['purpose','mode'])['dt','da'].sum().reset_index()
   #  external_movements_post_total.to_csv(r'C:\Users\scartissue\Desktop\7-Post_Externalmov_Totals.csv', mode='a', header=False)
    #wait = input("PRESS ENTER TO CONTINUE.")
    del(external_movements['total_dt'],
        external_movements['ds'],
        external_movements['total_productions'])

    target_p = target_productions['total_productions'].sum()
    distributed_p = external_movements['dt'].sum()
    achieved_p = external_movements['da'].sum()

    # Print for now - make formal audit in future
    print(target_p, achieved_p, distributed_p)
    
    del(external_movements['dt'])
    external_movements = external_movements.rename(columns={'da':'dt'})
    # Check dataframe contents
    # print('external_movements5 headings : ',external_movements.columns.tolist())
    # print('external_movements5 content: ',external_movements.head(n=5))
    # wait = input("PRESS ENTER TO CONTINUE.")
    return(external_movements)

def define_distance_bands(distances,
                          distance_banding):

    """
    """

    unq_dist = distance_banding['max_distance'].squeeze()
    unq_band = distance_banding['distance_band'].squeeze()

    distance_df = []
    for d in unq_dist.index:
        print(unq_dist[d])
        print(unq_band[d])
        subset = distances[distances['distance'] < unq_dist[d]].copy()
        subset.loc[:,'distance_band'] = unq_band[d]
        distances = distances.merge(subset,
                                    how='outer',
                                    on=['p_zone', 'a_zone', 'distance'])
        distances = distances[distances['distance_band'].isnull()]
        del(distances['distance_band'])
        distance_df.append(subset)

    distances = pd.concat(distance_df, sort=True)

    return(distances)

def create_externals(external_commute_pa,
                     external_productions,
                     internal_area,
                     external_area,
                     model_zone_lookups,
                     ia_name = None,
                     trip_origin = 'hb',
                     trip_length_adjustment = True,
                     test_outputs = False):

    """
    Function to approximate external trips using Census Journey to Work
    and a trip length adjustment.

    Parameters
    ----------
    external_commute_pa:
        A 24hr commute matrix distributed by census journey to work.

    productions:
        Mainland GB productions.

    ia_name:
        Name of the heading on the column of the internal area.

    internal_area:
        Vector of zones in the internal area.

    external_area:
        Vector of zones in the external area.

    model_zone_lookups:
        Path to model zone lookup folder.

    ia_name = None:
        Internal area name.

    trip_origin = 'hb':
        Origin, takes 'hb' or 'nhb'.

    trip_length_adjustment = True:
        Adjust trip lengths or not. Defaults to True.
    
    test_outputs = False:
        troubleshooting option to export production totals at key process milestones

    Returns
    ----------
    external_trips:
        Dataframe of i-to-e, e-to-i & e-to-e trips.
    """

    # If test_output True then give input info and export Productions by Mode and Purpose
    if test_outputs:
        external_pro_check1 = []
        external_pro_check1 = external_productions.groupby([ia_name,'mode','purpose'])['trips'].sum().reset_index()
        external_pro_check2 = []
        external_pro_check2 = external_productions.groupby(['purpose','mode'])['trips'].sum().reset_index()
        answer = input('Export Input_Productions? (y/n): ')
        if answer.lower().startswith("y"):
            # TODO: add in relative export path
            external_pro_check1.to_csv(r'C:\Users\scartissue\Desktop\1-Entry_Productions_Zns.csv')
            external_pro_check2.to_csv(r'C:\Users\scartissue\Desktop\2-Entry_Productions_modepurp.csv')
        elif answer.lower().startswith("n"):
            print("ok, as you were") 

    # Get correct index for reindexing at the end
    original_cols = list(external_commute_pa)

    # PLACEHOLDER IMPORTS
    # TODO: Productionise
    if trip_origin == 'hb':
        trip_length_bands = pd.read_csv(
                'Y:/NorMITs Synthesiser/import/trip_length_bands.csv')

        # Define external purposes
        external_purposes = [1,2,3,4,5,6,7,8]

    # TODO: Adjustment if nhb on trip length bands
    if trip_origin == 'nhb':
        trip_length_bands = pd.read_csv(
                'Y:/NorMITs Synthesiser/import/trip_length_bands_nhb.csv')

        external_purposes = [12,13,14,15,16,18]

    # TODO: Move optimisation to before not after.

    # Benchmarks
    print('Number of P/A combinations ' +
          str(len(external_commute_pa.reindex(
                  ['p_zone','a_zone'],axis=1).drop_duplicates())))
    print('Total external productions ' +
          str(external_commute_pa['dt'].sum()))

    # List non-1 trip purposes to iterate over.
    # I think 1 still needs the trip length adjustment...

    external_trips = []
    # Moved external commute pa out due to trip length correction
    # external_trips.append(external_commute_pa)
    # TODO: build log for audits

    # Get distance bands
    distance_params = pd.read_csv(
            'Y:/NorMITs Synthesiser/import/distance_band_index.csv')

    distances = dm.get_distance_and_costs(model_zone_lookups,
                                       request_type='distance',
                                       journey_purpose='commute',
                                       direction=None,
                                       car_available=True,
                                       seed_intrazonal=True)
    # Band distances
    distance_bands = define_distance_bands(distances, distance_params)

    for purpose in external_purposes:
        print('external trips for', purpose)
        # Subset target trip lengths
        ttl_external = trip_length_bands[
                trip_length_bands['purpose']==purpose].reset_index(drop=True)
        ttl_external = ttl_external.rename(columns={
                'average_trip_length':'ttl'})
        
        # Filter on purpose to get target production totals
        ttp_cols = [ia_name, 'mode', 'purpose', 'trips']
        target_productions = external_productions[
                external_productions['purpose']==purpose]
        
        target_productions = target_productions.reindex(ttp_cols, axis=1)
        
        target_productions = target_productions.groupby(
                [ia_name, 'mode', 'purpose'])['trips'].sum(
                        ).reset_index()
        
        target_productions = target_productions.rename(columns={
                ia_name:'p_zone',
                'trips':'total_productions'})
        
        # Seed new purpose external matrix
        external_movements = external_commute_pa.copy()

        # Move commute pa across to external movements and re-assign purpose
        external_movements.loc[:,'purpose'] = purpose
        # TODO: add error handling iferror loop
        if test_outputs:
            external_movements_pre_total = []
            external_movements_pre_total = external_movements.groupby(['purpose','mode'])['dt'].sum().reset_index()
            external_movements_pre_total.to_csv(r'C:\Users\scartissue\Desktop\3-Externalmov_Totals.csv', mode='a', header=False)
            target_production_pre_total = []
            target_production_pre_total = target_productions.groupby(['purpose','mode'])['total_productions'].sum().reset_index()
            target_production_pre_total.to_csv(r'C:\Users\scartissue\Desktop\4-Predist_TargetPro_Totals.csv', mode='a', header=False)
            
        # wait = input("PRESS ENTER TO CONTINUE.")
        # production_pre_total = []
        # production_pre_total = external_productions.groupby(['purpose','mode'])['trips'].sum().reset_index()
        # production_pre_total.to_csv(r'C:\Users\scartissue\Desktop\5-Predist_externalPro_Totals.csv', mode='a', header=False)
        # Factor trips to meet target productions
        external_movements = factor_distributed_trips(external_movements,
                                                      target_productions)
        # TODO: add error handling iferror loop
        if test_outputs:
            production_postdist_total = []
            production_postdist_total = external_movements.groupby(['purpose','mode'])['dt'].sum().reset_index()
            production_postdist_total.to_csv(r'C:\Users\scartissue\Desktop\5-Post_dist_Totals.csv', mode='a', header=False)
            
        # print(external_movements['dt'].sum())

        # TODO: Add band allocation process from adjust external trips here
        # Adjust trip lengths to meet segmented targets
        external_movements = adjust_external_trips(external_movements,
                                                   ttl_external,
                                                   distance_bands,
                                                   model_zone_lookups)

        external_movements = external_movements.reset_index(drop=True)
        if test_outputs:
            production_posttl_total = []
            production_posttl_total = external_movements.groupby(['purpose','mode'])['dt'].sum().reset_index()
            production_posttl_total.to_csv(r'C:\Users\scartissue\Desktop\6-Post_tl_Totals.csv', mode='a', header=False)
        external_movements = nup.optimise_data_types(external_movements)
        print('productions to placeholder', external_movements['dt'].sum())
        # wait = input("PRESS ENTER TO CONTINUE.")
        # Append to output frame
        external_trips.append(external_movements)
        del(external_movements)

        # End of loop

    external_trips = pd.concat(external_trips, sort=True)

    # Re-index to col order
    external_trips = external_trips.reindex(original_cols, axis=1)
    print(external_trips.head(n=5))
    if test_outputs:
        answer = input('Do you want to export that?:')
        if answer.lower().startswith("y"):
            external_trips.to_csv(r'C:\Users\scartissue\Desktop\8-External_Export.csv')
            print("ok, carry on then")
        elif answer.lower().startswith("n"):
            print("ok, as you were")
    return(external_trips)