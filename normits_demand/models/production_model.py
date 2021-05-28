"""
Created on: 13/05/2019

File purpose: Production Model for the Travel Market Synthesiser
Updated with master function

Version number: 0.9
Written using: Python 3.6.6

Module versions used for writing:

pandas v0.23.4

"""

import os
import warnings
from typing import List

import numpy as np
import pandas as pd

import normits_demand.build.pathing as demand

import normits_demand.trip_end_constants as tec

from normits_demand.utils import utils as nup
from normits_demand.utils import compress as com
from normits_demand.utils import general as gnu
from normits_demand.constraints import ntem_control as ntem
from normits_demand.utils.general import safe_dataframe_to_csv


class ProductionModel(demand.Pathing):

    """
    """

    def __init__(
            self,
            config_path,
            param_file
            ):
        super().__init__(config_path,
                         param_file)
        # Check for existing exports
        self.export = self.ping_outpath()

    def get_trip_rates(self,
                       model_type: str= 'hb',
                       verbose=True):

        """
        Import a csv of NTEM trip rates.
        This will not always be just pointed at a csv - may use modelling
        folders in future. Hence separate function.

        Parameters
        ----------
        model_type:
        'hb' or 'nhb'

        verbose:
        Echo or no.

        Returns
        ----------
        ntem_trip_rates:
            DataFrame containing NTEM trip rates.
        """
        if model_type == 'hb':
            tr_var = 'hb_trip_rates'
        elif model_type == 'nhb':
            tr_var = 'nhb_trip_rates'

        tr_path = os.path.join(
            self.import_folder,
            'production_params',
            self.params[tr_var])

        if verbose:
            print('Importing trip rates from:')
            print(tr_path)

        trip_rates = pd.read_csv(tr_path)

        return trip_rates

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
            land_use_output = pd.read_csv(self.params['resi_land_use_path'],
                                          low_memory=False)

        return land_use_output

    def aggregate_to_zones(self,
                           productions,
                           zone_to_target = True,
                           pop_weighted: bool = False):
        """
        Aggregates a DataFrame to a target zoning system.
        If given no modelling folder will aggregate by zones provided.
        If given a modelling folder, it will look for a zone translation with
        the correct format - apply the zonal splits & aggregate to the target
        model zones.

        Parameters
        ----------
        productions:
            a DataFrame with a single zone and category variables.
            Designed for a production vector.

        p_params:

        pop_weighted: bool:
        Weight by population or not. If not will just use a spatial lookup.

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
        # BACKLOG: Need this function so often it should be broader.
        # Some of the inputs are out of date, could be simpler too
        model_folder = self.lookup_folder

        print(list(productions))

        if model_folder is None:
            zone_col = list(productions)[0]
            print('No target zoning system provided')
            print('Aggregating on zones provided:', zone_col)
            zone_col = list(productions)[0]
            zone_productions = productions.reindex([zone_col, 'trips'],
                                                   axis=1).groupby(
                                                           zone_col).sum(
                                                                   ).reset_index()
            return zone_productions
        else:

            spatial_aggregation_input = self.params['land_use_zoning'].lower()
            spatial_aggregation_output = self.params['model_zoning'].lower()

            # Find and import correct lookup from model folder
            # These need to be in the correct format!
            if not pop_weighted:
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
                print(mzc_path)
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
                    min_upscale = model_zone_conversion[min_to_maj].min()
                    print('minimum zone overlap = ',
                          min_upscale,
                          ' : this should never be below 0.5')
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

            elif pop_weighted:

                # Define lookup type
                maj_to_min = (spatial_aggregation_output.lower() +
                              '_' + spatial_aggregation_input.lower() +
                              '_pop')
                min_to_maj = (spatial_aggregation_input.lower() +
                              '_' + spatial_aggregation_output.lower() +
                              '_pop')
                # Some elif loop
                file_sys = os.listdir(model_folder)
                mzc_path = [x for x in file_sys if maj_to_min in x][0]
                print(mzc_path)
                model_zone_conversion = pd.read_csv(model_folder +
                                                    '/' +
                                                    mzc_path).drop(
                                                            'overlap_type',
                    axis=1)

                # Count unique model zones in minor zoning system
                major_zone = list(model_zone_conversion)[0]
                minor_zone = list(model_zone_conversion)[1]

                # If not, bump minor overlap to 1.
                unq_major_zones = model_zone_conversion[
                        major_zone].drop_duplicates()
                unq_minor_zones = model_zone_conversion[
                        minor_zone
                        ].drop_duplicates()
                umz_len = len(unq_minor_zones)
                print(unq_major_zones)
                print(list(productions))

                total_trips = productions['trips'].sum()
                print('Starting with ' + str(total_trips))

                if (spatial_aggregation_input.lower() + '_zone_id') not in list(productions):
                    target_productions = productions.merge(model_zone_conversion,
                                                           how='left',
                                                           on=(spatial_aggregation_output.lower() +
                                                           '_zone_id'))
                else:
                    target_productions = productions.copy()

                # Relativise minor split column
                overlap_col = ('overlap_' +
                               spatial_aggregation_output.lower() +
                               '_split_factor')
                print(list(target_productions))

                target_productions['trips'] = (target_productions['trips'] *
                                  target_productions[overlap_col])

                final_trips = target_productions['trips'].sum()

                # Control
                control = total_trips/final_trips
                target_productions['trips'] = target_productions['trips'] * control

                final_trips = target_productions['trips'].sum()

                print('Ending with ' + str(final_trips))

        return target_productions

    # Merge functions - functions to bring datasets together

    def merge_trip_rates(self,
                         productions,
                         trip_rates,
                         join_cols: List[str]):
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
        productions = productions[productions['people'] > 0].copy()
        productions = productions.merge(trip_rates,
                                        on=join_cols)
        productions['trips'] = productions['trip_rate']*productions['people']
        productions = productions.drop(['trip_rate'], axis=1)
        return productions

    def apply_k_factors(self):

        """

        """

        return 0

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

        # population before trip rates per unique row
        lu_report = lu.groupby(group_cols)['people'].sum().reset_index()

        safe_dataframe_to_csv(lu_report,
                              os.path.join(
            output_path,
            'land_use_audit.csv'),
            index=False)

        return lu_report

    def ping_outpath(self):

        """
        """

        # BACKLOG: This solves a lot of problems, integrate into main
        # BACKLOG: Should take paths from path building class

        output_dir = os.path.join(self.run_folder,
                                  self.params['iteration'])
        output_f = 'Production Outputs'

        in_hb = os.path.join(
            output_dir,
            output_f,
            'hb_productions_' +
            self.params['land_use_zoning'].lower() +
            '.pbz2')

        in_nhb = os.path.join(
            output_dir,
            output_f,
            'nhb_productions_' +
            self.params['land_use_zoning'] +
            '.pbz2')

        out_hb = os.path.join(
            output_dir,
            output_f,
            'hb_productions_' +
            self.params['model_zoning'].lower() +
            '.pbz2')

        out_nhb = os.path.join(
            output_dir,
            output_f,
            'nhb_productions_' +
            self.params['model_zoning'].lower() +
            '.pbz2')

        if not os.path.exists(in_hb):
            in_hb = ''
        if not os.path.exists(in_nhb):
            in_nhb = ''
        if not os.path.exists(out_hb):
            out_hb = ''
        if not os.path.exists(out_nhb):
            out_nhb = ''

        export_dict = {'in_hb': in_hb,
                       'in_nhb': in_nhb,
                       'out_hb': out_hb,
                       'out_nhb': out_nhb}

        self.export = export_dict

        return export_dict

    # Run functions - run the whole production model
    def run_hb(self,
               verbose: bool = False):

        """
        """

        # Assign filter set name to a placeholder variable
        start_time = nup.set_time()
        print(start_time)

        # Build pathing
        output_dir = os.path.join(self.run_folder,
                                  self.params['iteration'])
        output_f = 'Production Outputs'
        lookup_f = os.path.join(output_f, 'Production Lookups')
        run_log_f = os.path.join(output_f, 'Production Run Logs')
        nup.create_folder(os.path.join(output_dir, output_f), chDir=False)
        nup.create_folder(os.path.join(output_dir, lookup_f), chDir=False)
        nup.create_folder(os.path.join(output_dir, run_log_f), chDir=False)

        # Get segmentation types, use to set import params
        p_params = {}

        # Get drop params for import
        # LU columns: ['soc_cat', 'ns_sec']
        mandatory_lu = ['msoa_zone_id', 'area_type',
                        'traveller_type',
                        'ca', 'people']
        optional_lu = ['soc', 'ns', 'g']

        for seg in self.params['hb_trip_end_segmentation']:
            if seg in optional_lu:
                mandatory_lu.append(seg)

        # Set mode share cols for applications
        p_params.update({'ms_cols': ['area_type', 'ca', 'p']})

        # Set output cols for re-aggregation
        # Work around mandatory segments being passed in as outputs
        out_cols = ['msoa_zone_id']
        for col in ['area_type', 'tp']:
            if col not in self.params['hb_trip_end_segmentation']:
                out_cols.append(col)
        p_params.update({'output_cols': out_cols +
                         self.params['hb_trip_end_segmentation'] + ['trips']})

        # does this need to be ordered?
        tout_cols = [self.params['model_zoning'].lower() + '_zone_id']
        for col in ['area_type', 'tp']:
            if col not in self.params['hb_trip_end_segmentation']:
                out_cols.append(col)
        p_params.update({'target_output_cols': tout_cols +
                         self.params['hb_trip_end_segmentation'] + ['trips']})

        # Other params as dict
        land_use_output = self.get_land_use_output()
        # if not in p_params['output_cols'] add to dict
        x_lu_cols = []
        for col in land_use_output:
            if col not in p_params['output_cols']:
                x_lu_cols.append(col)

        p_params.update({'x_lu_cols': x_lu_cols})
        del x_lu_cols

        if verbose:
            print('MS Cols')
            print(p_params['ms_cols'])

        # Add gender code and stick to it - Turn gender into an integer
        # Document code order for reference - Females(1), Male(2), Children(3)
        # This should be in land use
        if 'g' not in list(land_use_output):
            g_l = pd.DataFrame(tec.TT_GENDER)
            land_use_output = land_use_output.merge(g_l,
                                                    how='left',
                                                    on='traveller_type')

        # Do a report
        print(mandatory_lu)
        report_cols = mandatory_lu.copy()
        report_cols.remove('msoa_zone_id')

        lu_rep = self.land_use_report(
            land_use_output,
            var_cols=report_cols,
            output_path=os.path.join(output_dir,
                                     output_f))

        # Get trip rates
        trip_rates = self.get_trip_rates(
            model_type='hb',
            verbose=verbose)

        # Get join columns for trip rates
        tr_cols = ['traveller_type']
        for col in list(trip_rates):
            if col in p_params['output_cols'] and col not in tr_cols:
                # Purpose is in there already
                # Retain traveller type at all costs
                if col != 'p':
                    tr_cols.append(col)

        # Build col list for reindex
        tr_index_cols = p_params['output_cols'].copy()
        tr_index_cols.append('traveller_type') # Need for tp
        tr_index_cols.remove('tp')  # Not in yet
        tr_index_cols.remove('m')  # Not in yet
        tr_group_cols = tr_index_cols.copy()
        tr_group_cols.remove('trips')
        if verbose:
            print('Trip rate index cols:')
            print(tr_index_cols)
            print('Trip rate group cols')
            print(tr_group_cols)

        # Merge trip rates
        # len should be production length * purposes :.
        target_purpose = trip_rates['p'].drop_duplicates(
                ).reset_index(drop=True)

        aggregate_ns_sec = [1, 2]
        aggregate_soc = [3, 4, 5, 6, 7, 8]

        # Weekly trip rate application loop
        # output per purpose
        purpose_ph = {}
        for p in target_purpose:
            trip_rate_subset = trip_rates[trip_rates['p'] == p].copy()

            print('building purpose ' + str(p) + ' trip rates')
            lu_sub = land_use_output.copy()

            if p in aggregate_ns_sec:
                # Update ns with none
                lu_sub['ns'] = 'none'
                lu_sub['soc'] = lu_sub['soc'].astype(int)
                # Insurance policy
                trip_rate_subset['ns'] = 'none'
                trip_rate_subset['soc'] = trip_rate_subset['soc'].astype(int)

            elif p in aggregate_soc:
                # Update soc with none
                lu_sub['soc'] = 'none'
                lu_sub['ns'] = lu_sub['ns'].astype(int)
                # Insurance policy
                trip_rate_subset['soc'] = 'none'
                trip_rate_subset['ns'] = trip_rate_subset['ns'].astype(int)

            lu_sub = self.merge_trip_rates(lu_sub,
                                           trip_rate_subset,
                                           tr_cols)
            # Group and sum
            lu_sub = lu_sub.reindex(tr_index_cols,
                                    axis=1).groupby(
                                        tr_group_cols).sum(
                                            ).reset_index()

            # Update dictionary
            purpose_ph.update({p: lu_sub})

        time_splits = pd.read_csv(
            os.path.join(self.import_folder,
                         'production_params',
                         self.params['hb_time_split'],
                         ))
        mean_time_splits = pd.read_csv(
            os.path.join(self.import_folder,
                         'production_params',
                         self.params['hb_ave_time_split']
                         ))

        #  Time Period: 1= AM(peak), 2=IP(inter peak), 3=PM(peak), 4=OP(off-peak)
        target_tp = ['tp1', 'tp2', 'tp3', 'tp4']
        tp_ph = {}
        for tp in target_tp:
            print('Building time period ' + str(tp))
            tp_subset = time_splits.reindex(['area_type', 'traveller_type',
                                             'p', tp], axis=1).copy()
            tp_mean_subset = mean_time_splits.reindex(['p', tp], axis=1).copy()

            for key, dat in purpose_ph.items():
                print('For purpose ' + str(key))
                # Get mean for infill
                tp_mean = tp_mean_subset[tp_mean_subset['p'] == key][tp]

                tp_mat = dat.copy()
                tp_mat = tp_mat.merge(
                    tp_subset,
                    how='left', on=['area_type',
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
            approx_tp_totals.append(total)

        ave_wday = sum(approx_tp_totals)
        print('Average weekday productions: ' + str(round(ave_wday,0)))

        # Get mode splits
        mode_share = pd.read_csv(
            os.path.join(self.import_folder,
                         'production_params',
                         self.params['hb_mode_split']
                         ))

        # Build join cols
        m_cols = []
        for col in list(mode_share):
            if col in p_params['output_cols']:
                # Mode is in there already
                if col != 'm':
                    m_cols.append(col)

        # Build col list for reindex
        m_index_cols = p_params['output_cols'].copy()
        m_index_cols.remove('m')
        # tp is also built in
        m_index_cols.remove('tp')
        m_group_cols = m_index_cols.copy()
        m_group_cols.remove('trips')

        target_mode = ['m1', 'm2', 'm3', 'm5', 'm6']

        # Loop to get mode share trips
        m_ph = {}
        for m in target_mode:
            print('Building modes ' + str(m))

            # TODO: Function

            m_group = m_cols.copy()
            m_group.append(m)

            m_subset = mode_share.reindex(m_group, axis=1).copy()

            for key, dat in tp_ph.items():
                # Get p from key
                # Keep 2 chars and replace_, so I can copy for NHB
                tp_p = (key[key.index('p')+1:key.index('p')+3])
                tp_p = tp_p.replace('_', '')
                print('For purpose ' + str(tp_p))

                m_mat = dat.copy()
                # Would merge all purposes, but left join should pick out target mode
                m_mat = m_mat.merge(
                    m_subset,
                    how='left',
                    on=m_cols)

                m_mat['trips'] = (m_mat['trips'] * m_mat[m])

                # Reindex cols for efficiency
                m_mat = m_mat.reindex(
                    m_index_cols,
                    axis=1).groupby(
                        m_group_cols).sum().reset_index()

                m_mat = m_mat[m_mat['trips']> 0]

                if verbose:
                    print(m_mat['trips'].sum())
                    print(list(m_mat))
                    print(m_index_cols)
                    print(m_group_cols)
                    print(m_mat['trips'].sum())

                m_ph.update({(str(key)+'_'+m):m_mat})

        output_ph = []
        for key, dat in m_ph.items():
            print('Compiling productions for ' + key)

            output_list = key.split('_')

            purpose = output_list[0].replace('p', '')
            time_period = output_list[1].replace('tp', '')
            mode = output_list[2].replace('m', '')

            dat['p'] = purpose
            dat['tp'] = time_period
            dat['m'] = mode

            output_ph.append(dat)

        msoa_output = pd.concat(output_ph)

        # Output reindex - the last one!
        index_cols = p_params['output_cols'].copy()
        group_cols = index_cols.copy()
        group_cols.remove('trips')

        msoa_output = msoa_output.reindex(
                index_cols,
                axis=1).groupby(group_cols).sum().reset_index()

        # NTEM control
        msoa_lad_lookup = pd.read_csv(tec.MSOA_LAD)

        if self.params['export_uncorrected']:
            safe_dataframe_to_csv(msoa_output,
                                  (output_dir +
                                   output_f +
                                   '/hb_productions_' +
                                   'uncorrected.csv'),
            index=False)

        if self.params['production_ntem_control']:
            # Get ntem totals
            ntem_totals = pd.read_csv(tec.NTEM_PA)

            msoa_output, ntem_p, ntem_a, lad_output = ntem.control_to_ntem(
                    msoa_output,
                    ntem_totals,
                    msoa_lad_lookup,
                    constraint_cols = ['p', 'm', 'tp'],
                    base_value_name='trips',
                    ntem_value_name='productions',
                    trip_origin='hb')

            if self.params['export_lad']:
                safe_dataframe_to_csv(
                    lad_output,
                    (output_dir +
                    output_f +
                    '/hb_productions_lad_ntem.csv'),
                    index=False)

        if self.params['production_k_factor_control']:
            # BACKLOG: Implement k factor adjustment

            print('Before: ' + str(msoa_output['trips'].sum()))

            k_factors = pd.read_csv(self.params['production_k_factor_path'])
            k_factors = k_factors.reindex(['lad_zone_id', 'p', 'm', 'tp', 'prod_k'],
                                          axis=1)

            # Adjustment to tweak time period
            hb_purpose = tec.HB_PURPOSE
            hb_k_factors = k_factors[k_factors['p'].isin(hb_purpose)]
            hb_k_factors = hb_k_factors.drop('tp', axis=1)

            lad_lookup = msoa_lad_lookup.reindex(['lad_zone_id', 'msoa_zone_id'],
                                            axis=1)

            msoa_output = msoa_output.merge(lad_lookup,
                                            how = 'left',
                                            on='msoa_zone_id')
            # Seed zero infill
            msoa_output['trips'] = msoa_output['trips'].replace(0, 0.001)

            # Build LA adjustment
            adj_fac = msoa_output.reindex(['lad_zone_id',
                                           'p',
                                           'm',
                                           'trips'], axis=1).groupby(['lad_zone_id',
                                                  'p',
                                                  'm']).sum().reset_index()

            adj_fac = adj_fac.merge(hb_k_factors,
                                    how='left',
                                    on=['lad_zone_id',
                                          'p',
                                          'm'])
            adj_fac['adj_fac'] = adj_fac['prod_k']/adj_fac['trips']
            adj_fac = adj_fac.reindex(['lad_zone_id',
                                       'p',
                                       'm',
                                       'adj_fac'], axis=1)
            adj_fac['adj_fac'] = adj_fac['adj_fac'].replace(np.nan, 1)

            # BACKLOG: Report adj factors here
            msoa_output = msoa_output.merge(adj_fac,
                                            how='left',
                                            on=['lad_zone_id',
                                                  'p',
                                                  'm'])

            msoa_output['trips'] = msoa_output['trips'] * msoa_output['adj_fac']

            msoa_output = msoa_output.drop(['lad_zone_id','adj_fac'], axis=1)

            print('After: ' + str(msoa_output['trips'].sum()))

            # BACKLOG: Make export reports mode specific
            # BACKLOG: Get trip rates back somehow.

        # Export outputs with full segmentation
        if self.params['export_msoa']:
            com.write_out(
                msoa_output,
                os.path.join(
                    output_dir,
                    output_f,
                    'hb_productions_' +
                    self.params['land_use_zoning'].lower())
                )

        # Aggregate to target model zones
        # Don't do this. It fails to often. NHB should be based off MSOA.
        """
        target_output = self.aggregate_to_zones(
            msoa_output,
            p_params,
            pop_weighted=True)

        print(target_output)
        print(list(target_output))
        print(target_output['trips'].sum())

        # Compile and out
        t_group_cols = p_params['target_output_cols'].copy()
        t_group_cols.remove('trips')

        print(p_params['target_output_cols'])
        print(t_group_cols)

        target_output = target_output.reindex(
            p_params['target_output_cols'],
            axis=1).groupby(t_group_cols).sum().reset_index()
        target_output = target_output.sort_values(
            t_group_cols).reset_index(drop=True)

        out_path = os.path.join(
            output_dir,
            output_f,
            'hb_productions_' +
            self.params['model_zoning'].lower() +
            '.csv')
        if self.params['export_model_zoning']:
            safe_dataframe_to_csv(
                target_output,
                out_path,
                index=False)
        """

        end_time = nup.set_time()
        print(end_time)

        # Call multiprocessing.production_report w/ hb param
        """
        pr = ra.run_production_reports(file_drive='Y:/',
                                       model_name=output_zones.capitalize(),
                                       iteration=iteration,
                                       production_type = 'hb',
                                       model_segments = ['m', 'p'],
                                       internal_only = False,
                                       write = True)
        """

        return msoa_output, lu_rep

    def run_nhb(self,
                attraction_vector,
                verbose=True):

        """
        Builds NHB production vector from Homebased attraction vectors, while
        balancing those attractions to a detailed hb production vector to
        get some idea of discrete quantity and additional segmentation.

        NB. This could be used to build a full TfN segmentation in NHB, but with
        the area types baked in to segmentation, sample sizes will start to suffer
        badly.

        Parameters
        ----------
        self:
        Production model object

        production_vector:
        Path to hb prods

        attraction_vector:
        Path to hb attrs

        verbose:
        Text out or not

        Returns
        ----------
        output:
            TP Origin of non-homebased productions.
        """
        # Assign filter set name to a placeholder variable
        start_time = nup.set_time()
        print(start_time)

        # Build pathing
        output_dir = os.path.join(self.run_folder,
                                  self.params['iteration'])
        output_f = 'Production Outputs'
        lookup_f = os.path.join(output_f, 'Production Lookups')
        run_log_f = os.path.join(output_f, 'Production Run Logs')
        nup.create_folder(os.path.join(output_dir, output_f), chDir=False)
        nup.create_folder(os.path.join(output_dir, lookup_f), chDir=False)
        nup.create_folder(os.path.join(output_dir, run_log_f), chDir=False)

        # Import LAD lookup from globals
        lad_path = tec.MSOA_LAD
        ia_name = self.params['model_zoning'].lower() + '_zone_id'

        # Assign filter set name to a placeholder variable
        start_time = nup.set_time()
        print(start_time)

        # Get nhb trip rates
        trip_rates = self.get_trip_rates(
            model_type='nhb',
            verbose=verbose)
        trip_rates['p'] = trip_rates['p'].astype(str)

        # Get nhb mode splits
        nhb_mode_split = pd.read_csv(
            os.path.join(self.import_folder,
                         'production_params',
                         self.params['nhb_mode_split']))
        nhb_mode_split['p'] = nhb_mode_split['p'].astype(str)

        # Get nhb time splits
        nhb_time_split = pd.read_csv(
            os.path.join(self.import_folder,
                         'production_params',
                         self.params['nhb_time_split']))

        # Import HB attraction vector - should be balanced
        attractions = com.read_in(attraction_vector)
        # This should be optimised but maybe it's not
        i_cols = list(attractions)
        g_cols = i_cols.copy()
        g_cols.remove('2018')
        attractions = attractions.reindex(
            i_cols, axis=1).groupby(g_cols).sum().reset_index()
        attractions = attractions.sort_values(g_cols)

        # Get unique production segments
        # Unique segments are what we can get against what we have
        input_segments = ['area_type', 'p', 'm', 'ca']
        # TODO: MODE IS STILL IN THE ATTRACTIONS

        if verbose:
            print('Returned:')
            print(input_segments)
            print('Target:')
            print(self.params[
                'nhb_trip_end_segmentation'])

        # Build rate, mode, time frame
        single_rate = trip_rates.merge(
            nhb_mode_split,
            how='left',
            on=['area_type', 'ca', 'p', 'nhb_p']
        ).merge(nhb_time_split,
                how='left',
                on=['area_type', 'ca', 'nhb_p', 'm'])
        single_rate['trip_rate'] *= single_rate['mode_share'] * single_rate['time_share']
        single_rate = single_rate.drop(['mode_share', 'time_share'], axis=1)
        correct_order = [
            'area_type', 'ca', 'p', 'nhb_p', 'm', 'tp', 'trip_rate']
        single_rate = single_rate.reindex(
            correct_order, axis=1).reset_index(drop=True)

        # Get cols for iteration
        tr_cols = list(single_rate)
        ignore_cols = ['nhb_p', 'm', 'tp', 'trip_rate']
        tr_filters = [x for x in tr_cols if x not in ignore_cols]

        # Apply trip rates
        tr_ph = []
        for i, r in single_rate.iterrows():
            # Build subset
            print('%d of %d' % (i+1, len(single_rate)))

            sub_a = attractions.copy()
            for tr in tr_filters:
                print(tr)
                sub_a = sub_a[sub_a[tr] == r[tr]]



                print(len(sub_a))
            print(list(sub_a))

            sub_a['productions'] = sub_a['2018'] * r['trip_rate']
            sub_a['p'] = r['nhb_p']
            sub_a['m'] = r['m']
            sub_a['tp'] = r['tp']
            sub_a = sub_a.drop(['2018'], axis=1)

            tr_ph.append(sub_a)
            del sub_a

        nhb = pd.concat(tr_ph)

        if verbose:
            print('Pre mode reindex')
            print(list(nhb))

        # Order
        index_cols = list(nhb)
        group_cols = index_cols.copy()
        group_cols.remove('productions')
        nhb = nhb.reindex(
            index_cols,
            axis=1).groupby(
            group_cols).sum().reset_index()
        nhb = nhb.sort_values(group_cols).reset_index(drop=True)

        # Build iterator for mode segmentation
        mode_seg = nhb.reindex(
                ['area_type', 'p', 'ca', 'nhb_p'],
                axis=1).drop_duplicates().reset_index(drop=True)

        # Apply mode segmentation
        print('Building mode shares')
        mode_bin = []
        for index, row in mode_seg.iterrows():
            p_sub = nhb.copy()
            for name, dat in row.iteritems():
                p_sub = p_sub[p_sub[name] == dat]
            if verbose:
                print('List of p sub')
                print(list(p_sub))
                print(p_sub)
                print('List of mode split')
                print(list(nhb_mode_split))
            p_sub = p_sub.merge(nhb_mode_split,
                                how='left',
                                on=['area_type', 'ca', 'p', 'nhb_p'])
            p_sub['productions'] = p_sub['mode_share'] * p_sub['productions']
            p_sub = p_sub.drop('mode_share', axis=1)

            mode_bin.append(p_sub)

        nhb = pd.concat(mode_bin)

        if verbose:
            print('Pre time reindex')
            print(list(nhb))

        # Order
        index_cols = list(nhb)
        group_cols = index_cols.copy()
        group_cols.remove('productions')
        nhb = nhb.reindex(
            index_cols,
            axis=1).groupby(
            group_cols).sum().reset_index()

        # Time seg
        # BACKLOG: Need to check these are in all the way through
        time_seg = nhb.reindex(
                ['area_type', 'ca', 'nhb_p', 'm'],
                axis=1).drop_duplicates().reset_index(drop=True)

        print('Building time shares')
        time_bin = []
        for index, row in time_seg.iterrows():
            print('Time seg ' + str(index+1) + '/' + str(len(time_seg)))
            p_sub = nhb.copy()
            for name, dat in row.iteritems():
                p_sub = p_sub[p_sub[name] == dat]
            p_sub = p_sub.merge(nhb_time_split,
                                how='left',
                                on=['area_type', 'ca', 'nhb_p', 'm'])
            p_sub['productions'] = p_sub['time_share'] * p_sub['productions']
            p_sub = p_sub.drop('time_share', axis=1)
            time_bin.append(p_sub)

        nhb = pd.concat(time_bin)

        # Build the cols to retain and reindex from the output segments
        if verbose:
            print('Pre correct reindex')
            print(list(nhb))
        # Order
        index_cols = list(nhb)
        index_cols.drop('p', axis=1)
        group_cols = index_cols.copy()
        group_cols.remove('productions')
        nhb = nhb.reindex(
            index_cols,
            axis=1).groupby(
            group_cols).sum().reset_index()

        nhb = nhb.rename(columns={'nhb_p': 'p'})
        nhb = nhb.rename(columns={'productions': 'trips'})

        ntem_lad_lookup = pd.read_csv(lad_path)

        # Pre - NTEM export
        if self.params['export_uncorrected']:
            raw_nhb_path = os.path.join(
                output_dir,
                output_f,
                'nhb_productions_' +
                'uncorrected.csv')
            safe_dataframe_to_csv(nhb, raw_nhb_path, index=False)

        # NTEM Controls
        if self.params['production_ntem_control']:

            ntem_totals = pd.read_csv(tec.NTEM_PA)

            if verbose:
                print('NTEM totals cols')
                print(list(ntem_totals))

            nhb, ntem_a, ntem_f, nhb_lad = ntem.control_to_ntem(
                    nhb,
                    ntem_totals,
                    ntem_lad_lookup,
                    constraint_cols=['p', 'm', 'tp'],
                    base_value_name='trips',
                    ntem_value_name='productions',
                    trip_origin='nhb',
                    replace_drops=False)

            if self.params['export_lad']:
                safe_dataframe_to_csv(
                    nhb_lad,
                    os.path.join(
                        output_dir,
                        output_f,
                        'nhb_productions_' +
                        'lad_ntem.csv'))

            if self.params['export_msoa']:
                com.write_out(nhb,
                              os.path.join(
                                  output_dir,
                                  output_f,
                                  'nhb_productions_' +
                                  self.params['land_use_zoning'].lower()
                              ))
                print('Export msoa')

        # Factors
        if self.params['production_k_factor_control']:
            # BACKLOG: Reliable k factor control
            print('k factor control - to be re-written')


        return nhb
