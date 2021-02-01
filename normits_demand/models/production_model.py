"""
Created on: 13/05/2019

File purpose: Production Model for the Travel Market Synthesiser
Updated with master function

Version number: 0.9
Written using: Python 3.6.6

Module versions used for writing:

pandas v0.23.4

"""

# TODO: __init__
# TODO: Work out how to limit segments - currently ~70k per zone
# TODO: Add sort calls before joins for efficiency
# TODO: Warning handling
# TODO: Input audits (unit tests)
# TODO: Production object

import os, sys, time, warnings # File operations
from typing import List

import numpy as np # Vector operations
import pandas as pd # Bread and butter

from normits_demand.utils import utils as nup # Folder build utils

# Globals
_default_ntem = ('Y:/NorMITs Synthesiser/import/' +
                 'ntem_constraints/ntem_pa_ave_wday_2018.csv')
_default_trip_rates = ('Y:/NorMITs Synthesiser/import/' +
                       'trip_rates/tfn_hb_trip_rates_18_0620.csv')
_default_time_splits = ('Y:/NorMITs Synthesiser/import/' +
                        'trip_rates/tfn_hb_time_split_18_0620.csv')
_default_mode_splits = ('Y:/NorMITs Synthesiser/import/' +
                        'trip_rates/tfn_hb_mode_split_18_0620.csv')
_default_ave_time_splits = ('Y:/NorMITs Synthesiser/import/' +
                            'trip_rates/hb_ave_time_split.csv')
_default_msoa_lad = ('Y:/NorMITs Synthesiser/import/lad_to_msoa.csv')

class ProductionModel:
    """
    """

    def __init__(
            self,
            model_name: str = 'test',
            build_folder: str = 'Y:/NorMITs Synthesiser',
            iteration: str = 'iterx',
            trip_origin: str = 'hb',
            input_zones: str = 'msoa',
            output_zones: str = 'test',
            import_folder: str = 'Y:/NorMITs Synthesiser/import',
            model_folder: str = 'Y:/',
            output_segments: List[str] = ['p','m'],
            lu_path: str = 'Y:/Path to Land use',
            trip_rates = _default_trip_rates,
            time_splits = _default_time_splits,
            ave_time_splits = _default_ave_time_splits,
            mode_splits = _default_mode_splits,
            production_vector: str = '',
            attraction_vector: str = '',
            ntem_control: bool = True,
            ntem_path = _default_ntem,
            k_factor_control: bool = False,
            k_factor_path = None,
            export_msoa: bool = False,
            export_lad: bool = False,
            export_uncorrected: bool = True,
            export_target: bool = True
            ):
        """
        """

        # Model setup variables
        self.model_name = model_name
        self.build_folder = build_folder
        self.iteration = iteration
        self.trip_origin = trip_origin
        self.input_zones = input_zones
        self.output_zones = output_zones
        self.import_folder = import_folder
        self.model_folder = model_folder
        
        # Model option variables
        self.output_segments = output_segments
        if trip_rates == 'default':
            trip_rates = _default_trip_rates
        self.trip_rates = trip_rates
        self.time_splits = time_splits
        self.ave_time_splits = ave_time_splits
        self.mode_splits = mode_splits
        self.ntem_control = ntem_control      
        if ntem_path == 'default':
            ntem_path = _default_ntem
        self.ntem_path = ntem_path
        self.lu_path = lu_path
        # TODO: self.lad_path = lad_path Need this???
        self.k_factor_control = k_factor_control
        self.k_factor_path = k_factor_path
        self.export_msoa = export_msoa
        self.export_lad = export_lad
        self.export_uncorrected = export_uncorrected
        self.export_target = export_target

        # Project functions - Functions to define project folder & working directory
        # TODO: If these are useful they should go into utilities repo
        # Get functions - Functions to import data from various sources

    def get_trip_rates(self):

        # TODO: This is just a test
        # This is a demonstration
        # Another test
        """
        Import a csv of NTEM trip rates.
        This will not always be just pointed at a csv - may use modelling
        folders in future. Hence seperate function.
    
        Parameters
        ----------
        trip_rate_path:
            Path to csv of trip rates.
    
        Returns
        ----------
        ntem_trip_rates:
            DataFrame containing NTEM trip rates.
        """
        trip_rates = pd.read_csv(os.path.join(self.import_folder,
                                              self.trip_rates))
    
        return(trip_rates)
        
    def get_land_use_output(self):
        """
        Read in, and take a sample from, output from Land Use, given an import path
        Parameters
        ----------
        land_use_output_path:
            Path to land use output.
    
        handle_gaps = True:
            Filter out any gaps in the land use that may cause issues in the
            productions model.
    
        do_format = True:
            Optimise Land Use data on import.
    
        Returns
        ----------
        land_use_output:
            A DataFrame containing output from the NorMITs Land use model in a
            given zoning system. Should be MSOA or LSOA.
        """
        with warnings.catch_warnings():
                warnings.simplefilter(action='ignore',
                                      category=FutureWarning)
                land_use_output = pd.read_csv(self.lu_path)
                
        return(land_use_output)

    def aggregate_to_zones(self,
                           productions,
                           p_params,
                           spatial_aggregation_input = None,
                           spatial_aggregation_output = None,
                           model_folder = None,
                           pop_weighted = False):
        # TODO: Method to apply split zones to non-nesting zone systems.
        """
        Aggregates a DataFrame to a target zoning system.
        If given no modelling folder will aggregate by zones provided.
        If given a modelling folder, it will look for a zone translation with
        the correct format - apply the zonal splits & aggreagate to the target
        model zones.
    
        Parameters
        ----------
        productions:
            a DataFrame with a single zone and category variables.
            Designed for a production vector.
    
        p_params:
    
    
        spatial_aggregation_input:
            Name of the zoning system the base productions are built in.
            Should be defined by the run. Should be MSOA or LSOA.
    
        spatial_aggregation_output:
            Name of the target zoning system. Will likely be an Analytical
            Framework model eg. Norms, Nelum
    
        Returns
        ----------
        (if no model folder provided)
        zone_productions:
            DataFrame of productions aggregated by zones with no category.
    
        (if model folder provided)
        target_productions:
            DataFrame of segmented productions converted to a given model
            zoning system.
        """
        # TODO: Use the model lookup folder - run new zone translations
        # TODO: Standardise zone translations in folder
        # TODO: This function is a bit of a mess now - needs to be consolidated
    
        if model_folder is None:
            zone_col = list(productions)[0]
            print('No target zoning system provided')
            print('Aggregating on zones provided:', zone_col)
            zone_col = list(productions)[0]
            zone_productions = productions.reindex([zone_col,'trips'],
                                                   axis=1).groupby(
                                                           zone_col).sum(
                                                                   ).reset_index()
            return(zone_productions)
        else:
            # Find and import correct lookup from model folder
            # These need to be in the correct format!
            # TODO: Correct import format audit.
            if pop_weighted == False:
                print('Aggregating to', spatial_aggregation_output, 'zones')
    
                # Define lookup type
                maj_to_min = (spatial_aggregation_output.lower() +
                              '_to_' +
                              spatial_aggregation_input.lower())
                min_to_maj = (spatial_aggregation_input.lower() +
                              '_to_' +
                              spatial_aggregation_output.lower())
    
                file_sys = os.listdir(model_folder)
                mzc_path = [x for x in file_sys if (maj_to_min + '.csv') in x][0]
                model_zone_conversion = pd.read_csv(model_folder +
                                                    '/' +
                                                    mzc_path)
                # Count unique model zones in minor zoning system
                major_zone = list(model_zone_conversion)[0]
                minor_zone = list(model_zone_conversion)[1]
    
                print(major_zone)
                print(minor_zone)
    
                unq_minor_zones = model_zone_conversion[
                        minor_zone
                        ].drop_duplicates()
                umz_len = len(unq_minor_zones)
    
                # If there are zone splits - just go with the zone that overlaps most
                if umz_len < len(model_zone_conversion[minor_zone]):
                    model_zone_conversion = model_zone_conversion.groupby(
                            [
                                    minor_zone]).max(
                                    axis=1,
                                    level=[min_to_maj]).reset_index()
                    # if this has worked - this number should never be more than .5
                    # TODO: Formal warning message
                    min_upscale = model_zone_conversion[min_to_maj].min()
                    print('minimum zone overlap = ',
                          min_upscale,
                          ' : this should never be below 0.5')
                    # TODO: check this now matches target number of zones
                    # Reindex
                    mzc_cols = [minor_zone,
                                major_zone]
                    model_zone_conversion = model_zone_conversion.reindex(mzc_cols,
                                                                          axis=1)
                    # Need another function to pick the biggest zone in a split
                    # To make sure it's not in the internal area
                    target_productions = productions.merge(
                            model_zone_conversion,
                            how='left',
                            on=minor_zone)
                    del(target_productions[minor_zone])
                    p1 = sum(target_productions['trips'])
                    print('Sum of productions ' + str(p1))
                    target_productions = self.optimise_data_types(
                        target_productions)
                    # Use disaggregation function to aggregate
                    index_cols = [major_zone, 'mode', 'time', 'purpose',
                                  'car_availability', 'employment_type',
                                  'age']
                    # Sort in place
                    target_productions.sort_values(by=index_cols,
                                                   inplace=True)
    
            if pop_weighted == True:
    
                # Define lookup type
                maj_to_min = (spatial_aggregation_output.lower() +
                              '_' + spatial_aggregation_input.lower() +
                              '_pop')
                min_to_maj = (spatial_aggregation_input.lower() +
                              '_' + spatial_aggregation_output.lower() +
                              '_pop')
                # Some elif loop
                file_sys = os.listdir(model_folder)
                mzc_path = [x for x in file_sys if (maj_to_min) in x][0]
    
                model_zone_conversion = pd.read_csv(model_folder +
                                                    '/' +
                                                    mzc_path).drop(
                                                            'overlap_type',axis=1)
    
                # Count unique model zones in minor zoning system
                major_zone = list(model_zone_conversion)[0]
                minor_zone = list(model_zone_conversion)[1]
    
                print(major_zone)
                print(minor_zone)
    
                # TODO: Turn this into an audit to make sure every minor zone is matched.
                # If not, bump minor overlap to 1.
                unq_major_zones = model_zone_conversion[
                        major_zone].drop_duplicates()
                unq_minor_zones = model_zone_conversion[
                        minor_zone
                        ].drop_duplicates()
                umz_len = len(unq_minor_zones)
                print(unq_major_zones)
    
                total_trips = productions['trips'].sum()
                print('Starting with ' + str(total_trips))
    
                target_productions = productions.merge(model_zone_conversion,
                                                       how='left',
                                                       on = (spatial_aggregation_input +
                                                             '_zone_id'))
    
                # Relativise minor split column
                overlap_col = ('overlap_' +
                               spatial_aggregation_input.lower() +
                               '_split_factor')
    
                target_productions['trips'] = (target_productions['trips']  *
                                  target_productions[overlap_col])
    
                target_productions = target_productions.reindex(
                        p_params['target_output_cols'],
                        axis = 1)
    
                group_cols = p_params['target_output_cols'].copy()
                group_cols.remove('trips')
    
                target_productions = target_productions.groupby(
                        group_cols).sum().reset_index()
    
                final_trips = target_productions['trips'].sum()
    
                # Control
                control = total_trips/final_trips
                target_productions['trips'] = target_productions['trips'] * control
    
                final_trips = target_productions['trips'].sum()
    
                print('Ending with ' + str(final_trips))
    
    
        return(target_productions)
    
    # Merge functions - functions to bring datasets together
    
    def merge_trip_rates(productions,
                         trip_rates,
                         join_cols = ['area_type', 'traveller_type', 'soc', 'ns']):
        """
        Merge NTEM trip rates into productions on traveller type and area type.
        Create trips by multiplying people from land use by NTEM productions.
        Drops trip rates and people for efficiency.
    
        Parameters
        ----------
        productions:
            a DataFrame with a single zone and category variables.
            Designed for a production vector.
    
        trip_rates:
            NTEM trip rates by traveller type. No mode segmentation.
            Current assumption is that these are full week trip rates.
            This means mode time split needed to give 24hr trip rates.
    
        Returns
        ----------
        productions:
            Productions with trips by traveller type appended.
        """
    
        productions = productions[productions['people']>0].copy()
        productions = productions.merge(trip_rates,
                                        on=join_cols)
        productions['trips'] = productions['trip_rate']*productions['people']
        productions = productions.drop(['trip_rate'], axis=1)
        return(productions)

    def apply_k_factors(self,
                        productions,
                        k_factor_path):
    
        """
        Function to apply k-factors from previous run to MSOA productions.
    
        Parameters
        ----------
        productions:
            a productions dataframe with purposes and trips. Other segmentation
            will be ignored. Purposes should be NTEM standard HB (1-8)
    
        k_factor_path:
            a path to the k-factor lookup. Should be in model folders and specified
            by iteration.
    
        Returns
        ----------
        productions:
            Productions with trips updated at sector level.
    
        adjusted_trip_rates:
            Trip rates after k-factor adjustment.
    
        district_toplines:
            Summary of change applied at GB district level.
    
        purpose_toplines:
            Summary of change applied at aggregate purpose level.
        """
        # Import k factors
        k_factors = pd.read_csv(k_factor_path)
        k_factors = k_factors.reindex(
                ['lad_zone_id', 'purpose', 'k_factor_10'], axis=1)
    
        # Rename purpose to p mask to facilitate join
        k_factors = k_factors.rename(columns={'purpose':'p_mask'})
    
        lad_to_msoa = pd.read_csv(self.import_folder + '/lad_to_msoa.csv')
        lad_to_msoa = lad_to_msoa.reindex(['lad_zone_id', 'msoa_zone_id'],
                                          axis=1)
        # Join
        productions = productions.merge(lad_to_msoa,
                                        how='left',
                                        on='msoa_zone_id')
    
        # Put a production type mesh/mask on the productions to join on purpose
        commute_p = [1]
        business_p = [2]
        other_p = [3,4,5,6,7,8]
        productions['p_mask'] = productions['purpose']
        productions.loc[productions['purpose'].isin(commute_p), 'p_mask'] = 'commute'
        productions.loc[productions['purpose'].isin(business_p), 'p_mask'] = 'business'
        productions.loc[productions['purpose'].isin(other_p), 'p_mask'] = 'other'
    
        productions = productions.merge(k_factors,
                                        how='left',
                                        on = ['lad_zone_id', 'p_mask'])
    
        # Fill gaps with 1s
        productions['k_factor_10'] = productions['k_factor_10'].fillna(0)
        # Get factored trips
        productions['k_trips'] = productions['trips']*productions['k_factor_10']
    
        # Backfill new trip rates
        trip_rate_cols = ['purpose',
                          'traveller_type',
                          'area_type',
                          'people',
                          'trips']
        adjusted_trip_rates = productions.reindex(trip_rate_cols,
                                                  axis=1).groupby(['purpose',
                                                        'traveller_type',
                                                        'area_type']).sum(
                                                        ).reset_index()
        adjusted_trip_rates['trip_rate'] = (adjusted_trip_rates['trips']/
                           adjusted_trip_rates['people'])
        adjusted_trip_rates = adjusted_trip_rates.drop(
                ['people', 'trips'], axis=1)
    
        # Get district toplines
        district_toplines = productions.reindex(['lad_zone_id',
                                                 'trips',
                                                 'k_trips'],
        axis=1).groupby('lad_zone_id').sum().reset_index()
    
        # Apply percentage
        district_toplines['perc'] = (district_toplines['k_trips'] -
                         district_toplines['trips']) / district_toplines['trips']
    
        # Get purpose toplines
        purpose_toplines = productions.reindex(['purpose',
                                                'trips',
                                                'k_trips'],
        axis=1).groupby('purpose').sum().reset_index()
    
        # Apply percentage
        purpose_toplines['perc'] = (purpose_toplines['k_trips'] -
                        purpose_toplines['trips']) / purpose_toplines['trips']
    
        # Clean prods
        productions['trips'] = productions['k_trips'].copy()
        k_drops = ['lad_zone_id', 'p_mask', 'k_factor_10', 'k_trips']
        productions = productions.drop(k_drops, axis=1)
    
        productions['trips'] = productions['trips'].fillna(0)
    
        return(productions,
               adjusted_trip_rates,
               district_toplines,
               purpose_toplines)
    
    # Audit functions - Functions to aggregate and check
    
    def land_use_report(self,
                        lu,
                        var_cols,
                        output_path):
    
        """
        Aggregate and sum land use input, so I can have a look at it
        """
    
        group_cols = var_cols.copy()
        group_cols.remove('people')
    
        # TODO: Take relative cols from var cols param
        #    lu_report = lu.reindex(var_cols,
        #        axis=1).groupby(group_cols).sum().reset_index()
  
        # population before trip rates per unique row
        lu_report = lu.groupby(group_cols)['people'].sum().reset_index()
    
    
    
        lu_report.to_csv((output_path +
                         '/land_use_audit.csv'),
                         index=False)
    
        return(lu_report)

    # Run functions - run the whole production model
    def run_hb(self
               ):

        """
        """

        # TODO: Rewrite the frame audit to return dictionaries
        # Assign filter set name to a placeholder variable
        start_time = nup.set_time()
        print(start_time)
    
    
        # TODO: don't change wd working
        output_dir = os.path.join(self.build_folder,
                                  self.iteration)
        output_f = 'Production Outputs'
        lookup_f = os.path.join(output_f, 'Production Lookups')
        run_log_f = os.path.join(output_f, 'Production Run Logs')
        nup.create_folder(os.path.join(output_dir, output_f), chDir=False)
        nup.create_folder(os.path.join(output_dir, lookup_f), chDir=False)
        nup.create_folder(os.path.join(output_dir, run_log_f), chDir=False)

        # Get segmentation types, use to set import params
        # TODO: Functionalise
        p_params = {}
    
        # Get drop params for import
        # LU columns: ['soc_cat', 'ns_sec']
        mandatory_lu = ['msoa_zone_id', 'area_type',
                        'traveller_type',
                        'ca', 'people']
        optional_lu = ['soc', 'ns', 'g']

        for seg in self.output_segments:
            if seg in optional_lu:
                mandatory_lu.append(seg)

        # Set mean time period import for infill
        p_params.update({'mean_time_periods':
                self.ave_time_splits})

        # Set mode share cols for applications
        p_params.update({'ms_cols':['area_type', 'ca', 'p']})

        # Set output cols for re-aggregation
        # does this need to be ordered?
        test = [ 'msoa_zone_id', 'area_type', 'tp', 'trips']
        p_params.update({'output_cols': test + self.output_segments})
    
    
        # TODO: Take segmentation from output_segments
        # does this need to be ordered?
        test2 = [(self.output_zones + '_zone_id'), 'area_type', 'tp', 'trips']
        p_params.update({'target_output_cols': test2 + self.output_segments})
        p_params['target_output_cols']

        # Other params as dict
        land_use_output = self.get_land_use_output()
        # if not in p_params['output_cols'] add to dict
        extra_landuse_columns = []
        for column in land_use_output:
            if column not in p_params['output_cols']:
                extra_landuse_columns.append(column)
    
        p_params.update({'extra_landuse_cols': extra_landuse_columns})
        del(extra_landuse_columns)

        # Apply ca
        # This should be in Land Use
        # drop land use 'ca' column first - error in this column
        # Fix car ownership
        ca = pd.DataFrame({'cars':['0',0,'1',1,'1+','2','2+'],
                           'ca':[0,0,1,1,1,1,1]})
        land_use_output = land_use_output.merge(ca,
                                                how='left',
                                                on='cars')

        # Add gender code and stick to it - Turn gender into an integer
        # Document code order for reference - Females(1), Male(2), Children(3)
        # This should be in land use
        g_l = pd.DataFrame({'gender':['Females','Male','Children'],
                            'g':[1,2,3]})
        land_use_output = land_use_output.merge(g_l,
                                                how='left',
                                                on='gender')

        #subset check:
        landuse_subset = land_use_output.head(5)

        # cars and gender cols no longer needed -
        # these are re-formatted above
        land_use_output = land_use_output.drop(['gender','cars'], axis=1)
        list(land_use_output)

        # Do a report
        lu_rep = self.land_use_report(land_use_output,
                                      var_cols = p_params['lu_cols'],
                                      output_path = (output_dir + output_f))

        # Get trip rates
        trip_rates = self.get_trip_rates()

        # Audit population numbers
        print(land_use_output['people'].sum())
    
        # Merge trip rates
        # len should be production length * purposes :.
        # TODO: By purpose in to ph dict
        target_purpose = trip_rates['p'].drop_duplicates(
                ).reset_index(drop=True)
    
        aggregate_ns_sec = [1,2]
        aggregate_soc = [3,4,5,6,7,8]
    
        # Consolidate segments with no reasonable differentiation
        # TODO: Could take parameters in future, if you want to generalise
        # Build col list for reindex
        group_cols = ['msoa_zone_id',
                      'area_type',
                      'traveller_type',
                      'soc',
                      'ns',
                      'g',
                      'ca',
                      'p']
        index_cols = group_cols.copy()
        index_cols.append('trips')
    
        # Build segments for merge
    
    
        # Weekly trip rate application loop
        # output per purpose
        purpose_ph = {}
        for p in target_purpose:
            trip_rate_subset = trip_rates[trip_rates['p']==p].copy()
    
            print('building purpose ' + str(p) + ' trip rates')
            ph = land_use_output.copy()
    
            if p in aggregate_ns_sec:
                # Update ns with none
                ph['ns'] = 'none'
                ph['soc'] = ph['soc'].astype(int)
                # Insurance policy
                trip_rate_subset['ns'] = 'none'
                trip_rate_subset['soc'] = trip_rate_subset['soc'].astype(int)
    
            elif p in aggregate_soc:
                # Update soc with none
                ph['soc'] = 'none'
                ph['ns'] = ph['ns'].astype(int)
                # Insurance policy
                trip_rate_subset['soc'] = 'none'
                trip_rate_subset['ns'] = trip_rate_subset['ns'].astype(int)
    
            ph = self.merge_trip_rates(
                ph, trip_rate_subset, p_params['tr_cols'])
    
            # Group and sum
            ph = ph.reindex(index_cols,
                            axis=1).groupby(
                                    group_cols).sum(
                                            ).reset_index()
    
            # Update dictionary
            purpose_ph.update({p:ph})

        approx_totals = []
        for key, dat in purpose_ph.items():
            total = dat['trips'].sum()
            print(key)
            print(total)
            approx_totals.append(total/7)
    
        time_splits = pd.read_csv(p_params['time_periods'])
        mean_time_splits = pd.read_csv(p_params['mean_time_periods'])
    
        #  Time Period: 1= AM(peak), 2=IP(interpeak), 3=PM(peak), 4=OP(off-peak)
        target_tp = ['tp1', 'tp2', 'tp3', 'tp4']
        tp_ph = {}
        for tp in target_tp:
            print('Building time period ' + str(tp))
            tp_subset = time_splits.reindex(['area_type','traveller_type',
                                             'p',tp], axis=1).copy()
            tp_mean_subset = mean_time_splits.reindex(['p',tp], axis=1).copy()
    
            for key, dat in purpose_ph.items():
                print('For purpose ' + str(key))
                # Get mean for infill
                tp_mean = tp_mean_subset[tp_mean_subset['p']==key][tp]
    
                tp_mat = dat.copy()
                tp_mat = tp_mat.merge(tp_subset, how='left', on=['area_type',
                                                                 'traveller_type',
                                                                 'p'])
                tp_mat[tp] = tp_mat[tp].fillna(tp_mean)
    
                # Apply tp split and divide by 5 to get average weekday by tp
                tp_mat['trips'] = (tp_mat['trips'] * tp_mat[tp])/5
    
                # Drop tp col
                tp_mat = tp_mat.drop(tp, axis=1)
    
                # Add to compilation dict
                tp_ph.update({('p'+str(key)+'_'+tp):tp_mat})
    
        approx_tp_totals = []
        for key, dat in tp_ph.items():
            total = dat['trips'].sum()
            print(key)
            print(total)
            approx_tp_totals.append(total)
    
        ave_wday = sum(approx_tp_totals)
        print('Average weekday productions: ' + str(round(ave_wday,0)))
    
        # Get mode splits
        # TODO: Apply at MSOA level rather than area type
        mode_share = pd.read_csv(p_params['mode_share'])
    
        target_mode = ['m1', 'm2', 'm3', 'm5', 'm6']
    
        # Loop to get mode share trips
        m_ph = {}
        for m in target_mode:
            print('Building modes ' + str(m))
            m_subset = mode_share.reindex(['area_type','ca','p', m], axis=1).copy()
            print(m_subset)
    
            for key, dat in tp_ph.items():
                # Get p from key
                # Keep 2 chars and replace_, so I can copy for NHB
                tp_p = (key[key.index('p')+1:key.index('p')+3])
                tp_p = tp_p.replace('_','')
                print('For purpose ' + str(tp_p))
    
                m_mat = dat.copy()
                # Would merge all purposes, but left join should pick out target mode
                m_mat = m_mat.merge(m_subset, how='left', on=['area_type',
                                                              'ca',
                                                              'p'])
    
                # Apply m split
                m_mat['trips'] = (m_mat['trips'] * m_mat[m])
    
                # Reindex cols for efficiency
                # Not a fan of the global definitions now - makes this tricky
                group_cols = p_params['output_cols'].copy()
                group_cols.remove('trips')
                group_cols.remove('p')
                group_cols.remove('tp')
                group_cols.remove('m')
    
                m_mat = m_mat.reindex(
                        p_params['output_cols'],
                        axis=1).groupby(
                                group_cols).sum().reset_index()
    
                m_mat = m_mat[m_mat['trips']>0]
    
                m_ph.update({(str(key)+'_'+m):m_mat})
    
        # Drop unwanted columns
        # TODO: Topline audit here
        approx_mode_totals = []
    
        for key, dat in m_ph.items():
            total = dat['trips'].sum()
            approx_mode_totals.append([key, total])
    
        # Build topline report
        # TODO: Functionalise
        topline = pd.DataFrame(approx_mode_totals, columns=['desc', 'total'])
        descs = topline['desc'].str.split('_', expand=True)
        topline['p'] = descs[0]
        topline['tp'] = descs[1]
        topline['m'] = descs[2]
        topline = topline.reindex(
                ['p', 'tp', 'm', 'total'],
                axis=1).groupby(
                        ['p', 'tp', 'm']).sum().reset_index()
        topline.to_csv((output_dir +
                        run_log_f +
                        '/production_topline.csv'), index=False)
    
        # K Factor adjustment here, mode specific
        # TODO: Rewrite to take k factors at some point - doesn't work anymore
    
        output_ph = []
        for key, dat in m_ph.items():
            print('Compiling productions for ' + key)
    
            output_list = key.split('_')
    
            purpose = output_list[0].replace('p','')
            time_period = output_list[1].replace('tp','')
            mode = output_list[2].replace('m','')
    
            dat['p'] = purpose
            dat['tp'] = time_period
            dat['m'] = mode
    
            output_ph.append(dat)
    
        msoa_output = pd.concat(output_ph)
    
        # add gender in here, sum the number of trips per unique row
        sum_cols = ['msoa_zone_id', 'area_type', 'soc',
                    'ns', 'ca', 'g', 'p', 'tp', 'm', 'trips']
        group_cols = ['msoa_zone_id', 'area_type', 'soc',
                      'ns', 'ca', 'g', 'p', 'tp', 'm']
    
        msoa_output = msoa_output.reindex(
                sum_cols, axis=1).groupby(group_cols).sum().reset_index()
    
        msoa_output['p'] = msoa_output['p'].astype(int)
        msoa_output['m'] = msoa_output['m'].astype(int)
    
        # TODO: Topline audit
        # NTEM control
        msoa_lad_lookup = pd.read_csv(self._default_msoa_lad)

        if self.export_uncorrected:
            msoa_output.to_csv((output_dir +
                                output_f +
                                '/hb_productions_' +
                                'uncorrected.csv'),
            index=False)
    
        if self.ntem_control:
            # Get ntem totals
            # TODO: Depends on the time period - but this is fixed for now
            ntem_totals = pd.read_csv(self.ntem_path())

            msoa_output, ntem_p, ntem_a, lad_output = nup.control_to_ntem(
                    msoa_output,
                    ntem_totals,
                    msoa_lad_lookup,
                    group_cols = ['p','m'],
                    base_value_name = 'trips',
                    ntem_value_name = 'Productions',
                    purpose = 'hb')

            if self.export_lad:
                lad_output.to_csv((output_dir +
                                   output_f +
                                   '/hb_productions_lad_ntem.csv'),
                index=False)
    
        if self.k_factor_control is not None:
            # TODO: Loop over all modes in the list. k factor paths as list only
            # TODO: La level reports for ntem & k adjust < .2 & >5
            print('Before: ' + str(msoa_output['trips'].sum()))
    
            k_factors = pd.read_csv(self.k_factor_path)
            k_factors = k_factors.reindex(['lad_zone_id','p','m','tp','prod_k'],
                                          axis=1)
    
            # TODO: adjustment to tweak time period
            hb_purpose = [1,2,3,4,5,6,7,8]
            hb_k_factors = k_factors[k_factors['p'].isin(hb_purpose)]
            hb_k_factors = hb_k_factors.drop('tp', axis=1)
    
            lad_lookup = msoa_lad_lookup.reindex(['lad_zone_id', 'msoa_zone_id'],
                                            axis=1)
    
            msoa_output = msoa_output.merge(lad_lookup,
                                            how = 'left',
                                            on = 'msoa_zone_id')
            # Seed zero infill
            msoa_output['trips'] = msoa_output['trips'].replace(0,0.001)
    
            # Build LA adjustment
            adj_fac = msoa_output.reindex(['lad_zone_id',
                                           'p',
                                           'm',
                                           'trips'], axis=1).groupby(['lad_zone_id',
                                                  'p',
                                                  'm']).sum().reset_index()
    
            adj_fac = adj_fac.merge(hb_k_factors,
                                    how = 'left',
                                    on = ['lad_zone_id',
                                          'p',
                                          'm'])
            adj_fac['adj_fac'] = adj_fac['prod_k']/adj_fac['trips']
            adj_fac = adj_fac.reindex(['lad_zone_id',
                                       'p',
                                       'm',
                                       'adj_fac'], axis=1)
            adj_fac['adj_fac'] = adj_fac['adj_fac'].replace(np.nan, 1)
    
            # TODO: Report adj factors here
    
            msoa_output = msoa_output.merge(adj_fac,
                                            how = 'left',
                                            on = ['lad_zone_id',
                                                  'p',
                                                  'm'])
    
            msoa_output['trips'] = msoa_output['trips'] * msoa_output['adj_fac']
    
            msoa_output = msoa_output.drop(['lad_zone_id','adj_fac'], axis=1)
    
            print('After: ' + str(msoa_output['trips'].sum()))
    
            # TODO: Make export reports mode specific
            # TODO: Get trip rates back somehow.
    
        # Export outputs with full segmentation
        # TODO: There is no productions unless k factors run
        if self.export_msoa:
            # TODO: Make work with aggregations
            msoa_output.to_csv((output_dir +
                                output_f +
                            '/hb_productions_' +
                            self.input_zones.lower() +
                            '.csv'), index=False)
    
        # Aggregate to target model zones
        target_output = self.aggregate_to_zones(self,
                                                msoa_output,
                                                p_params,
                                                pop_weighted = True)

        # Compile and out
        # add gender here
        target_output = target_output.reindex([(self.output_zones + '_zone_id'),
                                               'area_type',
                                               'p', 'm', 'tp', 'soc',
                                               'ns', 'ca', 'g', 'trips'],
        axis=1).sort_values([(self.output_zones + '_zone_id'),
                           'area_type',
                           'p', 'm', 'tp', 'soc',
                           'ns', 'g', 'ca']).reset_index(drop=True)
    
        # TODO: output_segments - should use the main params
        out_path = os.path.join(
            output_dir,
            output_f,
            'hb_productions_' +
            self.output_zones +
            '.csv')
        if self.export_target:
            target_output.to_csv(
                out_path,
                index=False)

        end_time = nup.set_time()
        print(end_time)
    
        # Call mp.production_report w/ hb param
        """
        pr = ra.run_production_reports(file_drive='Y:/',
                                       model_name=output_zones.capitalize(),
                                       iteration=iteration,
                                       production_type = 'hb',
                                       model_segments = ['m', 'p'],
                                       internal_only = False,
                                       write = True)
        """
    
        return out_path, target_output, lu_rep

    def run_nhb(self,
                input_segments = ['area_type', 'p', 'soc', 'ns', 'ca'],
                output_segments = ['p', 'm', 'soc', 'ns', 'ca'],
                filter_set = None,
                ntem_control = True,
                k_factor_paths = None,
                export_uncorrected = False,
                export_msoa = False,
                export_lad = True,
                export_target = True,
                trip_rate_type = 'tms'):

        """
        Builds NHB production vector from Homebased attraction vectors, while
        balancing those attractions to a detailed hb production vector to
        get some idea of discrete quantity and additional segmentation.
    
        NB. This could be used to build a full TfN segmentation in NHB, but with
        the area types baked in to segmentation, sample sizes will start to suffer
        badly.
    
        Parameters
        ----------
        home_dir:
            home dir
    
        iteration:
            Iteration
    
        model_folder:
            Path to model folder
    
        input_segments:
            Should be the segmentation to be retained from HB productions
            See notes on balancing above.
    
        output_segments:
            asdasd
    
        filter_set:
            asdasd
    
        ntem_control:
            Control to ntem or not
    
        k_factor_paths:
            Control to a set of k factors or not
    
        export:
            Export or not
    
        Returns
        ----------
        output:
            TP Origin of non-homebased productions.
        """
            
        lad_path = self.lad_path,
        ntem_path = self.ntem_path
    
        # Assign filter set name to a placeholder variable
        start_time = nup.set_time()
        print(start_time)
    
        # Get nhb trip rates
        trip_rates = self.get_trip_rates()

        # Get nhb mode splits
        nhb_mode_split = pd.read_csv(os.path.join(i_paths['imports'],
                                                  'production_params',
                                                  'nhb_ave_wday_mode_split.csv'))
        # Get nhb time splits
        nhb_time_split = pd.read_csv(os.path.join(i_paths['imports'],
                                                  'production_params',
                                                  'nhb_ave_wday_time_split.csv'))

        # Import HB PA
        pa = nup.import_pa(self.production_vector,
                           self.attraction_vector)

        productions = pa[0]
        print(productions['trips'].sum())
        attractions = pa[1]
        del(pa)
    
        # Get unique production segments
        unq_seg = productions.reindex(
                input_segments,
                axis=1).drop_duplicates(
                        ).sort_values(
                                input_segments).reset_index(
                                        drop=True)
    
        # Balance attractions to productions by target segments
        prod_ph = []
        a_vec = []
    
        for index, row in unq_seg.iterrows():
            calib_params = {}
            for label, dat in row.iteritems():
                calib_params.update({label:dat})
            print(calib_params)

            # Filter productions to target distribution type
            sub_p = nup.filter_pa_vector(productions,
                                         self.output_zones,
                                         calib_params,
                                         round_val = 3,
                                         value_var = 'trips',
                                         echo=False)
    
            # Get the productions from the tuple
            sub_p = sub_p[0]
            sub_p = sub_p.rename(columns={'trips':'productions'})
    
            # Work out which attractions to use from purpose
            sub_a = nup.filter_pa_vector(attractions,
                                         self.output_zones,
                                         calib_params,
                                         round_val = 3,
                                         value_var = 'attractions',
                                         echo=False)
    
            # Get the Attractions from the tuple
            sub_a = sub_a[0]
    
            a_t = nup.get_attraction_type(calib_params)
            print(a_t)
    
            # Balance a to p
            # This is why the productions are here!
            print(sub_p['productions'].sum())
            sub_a = nup.balance_a_to_p(self.output_zones,
                                       sub_p,
                                       sub_a,
                                       round_val = 3,
                                       echo=False)
            a_vec.append(sub_a['attractions'].sum())
    
            sub_tr = trip_rates.copy()
            for name, dat in calib_params.items():
                if name in list(sub_tr):
                    if dat != 'none':
                        sub_tr = sub_tr[sub_tr[name]==dat]

            print(sub_tr)
    
            for i, r in sub_tr.iterrows():
                new_a = sub_a.copy()
                new_a['productions'] = new_a['attractions'] * r['trip_rate']
                new_a = new_a.drop(['attractions'], axis=1)
    
                out_dict = {}
                for name, dat in calib_params.items():
                    if name in list(sub_tr):
                        out_dict.update({name:dat})
                out_dict.update({'nhb_p':int(r['nhb_p'])})
                out_dict.update({'productions':new_a})
                out_dict.update({'total':new_a['productions'].sum()})
                prod_ph.append(out_dict)
    
        vector = []
        for item in prod_ph:
            vector.append(item['total'])
    
        print(sum(vector))
    
        print(sum(a_vec))
    
        # Recompile into dataframe
        nhb_ph = []
        for item in prod_ph:
            new_row = item['productions']
            for name, dat in item.items():
                if name != 'productions':
                 if name != 'total':
                    new_row[name] = dat
            nhb_ph.append(new_row)
        nhb = pd.concat(nhb_ph)
    
        # Build the cols to retain and reindex from the input segments
        retain_cols = [self.output_zones]
        for i_s in input_segments:
            # Train both types of purpose
            if i_s == 'p':
                retain_cols.append('p')
                retain_cols.append('nhb_p')
            else:
                retain_cols.append(i_s)
        retain_index = retain_cols.copy()
        retain_index.append('productions')
    
        nhb = nhb.reindex(retain_index,
                axis=1).groupby(retain_cols).sum().reset_index()

        mode_seg = nhb.reindex(
                ['area_type', 'p', 'ca', 'nhb_p'],
                axis=1).drop_duplicates().reset_index(drop=True)
    
        print('Building mode shares')
        mode_bin = []
        for index, row in mode_seg.iterrows():
            p_sub = nhb.copy()
            for name, dat in row.iteritems():
                p_sub = p_sub[p_sub[name]==dat]
            p_sub = p_sub.merge(nhb_mode_split,
                                how='left',
                                on=['area_type', 'ca', 'p', 'nhb_p'])
            p_sub['productions'] = p_sub['mode_share'] * p_sub['productions']
            p_sub = p_sub.drop('mode_share', axis=1)
            mode_bin.append(p_sub)
    
        nhb = pd.concat(mode_bin)
    
        # Build the cols to retain and reindex from the output segments
        # TODO: Needs area type
        retain_cols = [self.output_zones]
        for i_s in output_segments:
            # Train both types of purpose
            if i_s == 'p':
                retain_cols.append('p')
                retain_cols.append('nhb_p')
            elif i_s == 'tp':
                continue
            else:
                retain_cols.append(i_s)
        if 'area_type' not in retain_cols:
            retain_cols.append('area_type')
        retain_index = retain_cols.copy()
        retain_index.append('productions')
    
        nhb = nhb.reindex(retain_index,
            axis=1).groupby(
                    retain_cols).sum().reset_index()

    
        # Time seg
        time_seg = nhb.reindex(
                ['area_type', 'ca', 'nhb_p', 'm'],
                axis=1).drop_duplicates().reset_index(drop=True)
    
        print('Building time shares')
        time_bin = []
        for index, row in time_seg.iterrows():
            print('Time seg ' + str(index+1) + '/' + str(len(time_seg)))
            # TODO: Pre sort to make join smoother
            p_sub = nhb.copy()
            for name, dat in row.iteritems():
                p_sub = p_sub[p_sub[name]==dat]
            p_sub = p_sub.merge(nhb_time_split,
                                how='left',
                                on=['area_type', 'ca', 'nhb_p', 'm'])
            p_sub['productions'] = p_sub['time_share'] * p_sub['productions']
            p_sub = p_sub.drop('time_share', axis=1)
            time_bin.append(p_sub)
    
        nhb = pd.concat(time_bin)
    
        # Build the cols to retain and reindex from the output segments
        retain_cols = [ia_name]
        for i_s in output_segments:
            # Train both types of purpose
            if i_s == 'p':
                retain_cols.append('nhb_p')
            else:
                retain_cols.append(i_s)
        retain_index = retain_cols.copy()
        retain_index.append('productions')
    
        nhb = nhb.reindex(retain_index,
            axis=1).groupby(
                    retain_cols).sum().reset_index()
        nhb = nhb.sort_values(retain_index).reset_index(drop=True)
    
        nhb = nhb.rename(columns={'nhb_p':'p'})
        nhb = nhb.rename(columns={'productions':'trips'})
    
        ntem_lad_lookup = pd.read_csv(_default_lad_path)
    
        # Pre - NTEM export
        if export_uncorrected:
            raw_nhb_path = os.path.join(o_paths['production_export'],
                                    'nhb_productions_' +
                                    'uncorrected.csv')
            nhb.to_csv(raw_nhb_path, index=False)
    
        ## NTEM Controls
        if self.ntem_control:
            # TODO: Global ntem location
            msoa_list = list(nhb)[1:]
            msoa_toc = ['msoa_zone_id']
            for col in msoa_list:
                msoa_toc.append(col)
            p_params = {'target_output_cols':msoa_toc}
    
            # TODO: This is on the fly a little bit & the method drops a little demand.
            # Convert back to msoa
            nhb_msoa = aggregate_to_zones(nhb,
                                          p_params,
                                          model_name.lower(),
                                          'msoa',
                                          model_folder,
                                          pop_weighted = True)
    
            ntem_totals = pd.read_csv(ntem_path)
    
            nhb_msoa, ntem_a, ntem_f, nhb_lad = nup.control_to_ntem(
                    nhb_msoa,
                    ntem_totals,
                    ntem_lad_lookup,
                    group_cols = ['p', 'm', 'tp'],
                    base_value_name = 'trips',
                    ntem_value_name = 'Productions',
                    purpose = 'nhb')
            print(ntem_a)
    
            if export_lad:
                nhb_lad.to_csv(os.path.join(o_paths['production_export'],
                                            'nhb_productions_' +
                                            'lad_ntem.csv'))
    
            if export_msoa:
                # TODO: Actually export msoa..
                print('Export msoa')
    
            # Convert back to zones
            p_params = {'target_output_cols':list(nhb)}
            nhb = self.aggregate_to_zones(nhb_msoa,
                                     p_params,
                                     'msoa',
                                     model_name.lower(),
                                     model_folder,
                                     pop_weighted = True)
    
        # Factors
        if k_factor_paths is not None:
            k_factors = os.listdir(k_factor_paths)[0]
            k_factors = pd.read_csv(k_factor_paths + '/' + k_factors)
            k_factors = k_factors.reindex(['lad_zone_id','p','m','tp','prod_k'],
                                          axis=1)
    
            nhb_purpose = [12,13,14,15,16,18]
    
            nhb_k_factors = k_factors[k_factors['p'].isin(nhb_purpose)]
    
            # Get hb_adjustment factors
            # TODO: Should be relative
            lad_lookup = pd.read_csv(lad_path)
            lad_lookup = lad_lookup.reindex(['lad_zone_id', ia_name], axis=1)
    
            nhb = nhb.merge(lad_lookup,
                            how = 'left',
                            on = ia_name)
            nhb['trips'] = nhb['trips'].replace(0,0.001)
    
            # Build LA adjustment
            adj_fac = nhb.reindex(['lad_zone_id',
                                   'p',
                                   'm',
                                   'tp',
                                   'trips'], axis=1).groupby(
            ['lad_zone_id',
             'p',
             'm',
             'tp']).sum().reset_index()
            adj_fac = adj_fac.merge(nhb_k_factors,
                                    how = 'left',
                                    on = ['lad_zone_id',
                                          'p',
                                          'm',
                                          'tp'])
            adj_fac['adj_fac'] = adj_fac['prod_k']/adj_fac['trips']
            adj_fac = adj_fac.reindex(['lad_zone_id',
                                       'p',
                                       'm',
                                       'tp',
                                       'adj_fac'], axis=1)
            adj_fac['adj_fac'] = adj_fac['adj_fac'].replace(np.nan, 1)
    
            nhb = nhb.merge(adj_fac,
                            how = 'left',
                            on = ['lad_zone_id',
                                  'p',
                                  'm',
                                  'tp'])
            nhb['trips'] = nhb['trips'] * nhb['adj_fac']
    
            nhb = nhb.drop(['lad_zone_id','adj_fac'], axis=1)
    
        # Write out
        nhb_path = os.path.join(
            o_paths['production_export'],
            'nhb_productions_' +
            self.model_name.lower() +
            '.csv')

        if export_target:
            nhb.to_csv(nhb_path, index=False)
    
        return(nhb_path, nhb)