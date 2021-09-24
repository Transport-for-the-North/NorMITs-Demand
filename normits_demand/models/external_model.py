# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 13:45:03 2020

@author: cruella
"""
import os

import pandas as pd
import numpy as np

import normits_demand.build.tms_pathing as tms

from normits_demand.reports import reports_audits as ra
from normits_demand.utils import utils as nup
from normits_demand.utils import general as du
from normits_demand.utils import timing
from normits_demand.utils import math_utils
from normits_demand.matrices import utils as mat_utils
from normits_demand.distribution import furness

# TODO: Object layer

# Folder management, reindexing, optimisation

_default_pld_path = ('Y:/NorMITs Synthesiser/import/' +
                     'pld_od_matrices/external_pa_pld.csv')


class ExternalModel(tms.TMSPathing):
    pass

    def run(self,
            trip_origin: str = 'hb',
            cost_type: str = '24hr',
            seed_infill=.1):

        """
        """
        # Define internal name
        zoning_name = self.params['model_zoning'].lower()
        zone_col = (zoning_name + '_zone_id')
        val_col = 'val'

        # INIT
        # Get internal and external area
        int_zones = nup.get_internal_area(self.lookup_folder)
        ext_zones = nup.get_external_area(self.lookup_folder)
        all_zones = int_zones + ext_zones

        internal_index = np.array(int_zones) - 1
        external_index = np.array(ext_zones) - 1

        # Pick different imports if it's HB or NHB
        if trip_origin == 'hb':
            in_list = ['hb_p', 'hb_a']
        elif trip_origin == 'nhb':
            in_list = ['nhb_p', 'nhb_a']
        else:
            raise ValueError(
                "Don't know what the trip origin is!"
            )

        # ## IMPORT PA ## #
        print("Reading in P/A from NoTEM...")
        productions, attractions = nup.import_pa(
            self.tms_in[in_list[0]],
            self.tms_in[in_list[1]],
            zoning_name,
            trip_origin,
        )

        # ## GET DISTRIBUTION PARAMETERS ## #
        init_params = nup.get_init_params(
            self.lookup_folder,
            distribution_type=trip_origin,
            model_name=self.params['model_zoning'],
            mode_subset=None,
            purpose_subset=None,
        )

        # Drop any sic or soc segments from init params - not needed for externals
        # Also alert and error if there are any differences
        for ts in self.params['external_segmentation']:
            if ts not in list(init_params):
                raise ValueError('Init params and segmentation misaligned')

        init_params = init_params.reindex(self.params['external_segmentation'], axis=1)
        init_params = init_params.drop_duplicates().reset_index(drop=True)

        print('init_params:\n %s\n' % init_params)

        # Define mode subset
        unq_mode = self.params['external_export_modes']

        # Append non dist modes to list for init params
        if self.params['non_dist_export_modes'] is not None:
            [unq_mode.append(x) for x in self.params['non_dist_export_modes'] if x not in unq_mode]
        init_params = init_params[init_params['m'].isin(unq_mode)].reset_index(drop=True)

        # ## GET CJTW ## #
        print('Importing cjtw...')
        cjtw = nup.get_cjtw(self.lookup_folder,
                            self.params['model_zoning'].lower(),
                            subset=None,
                            reduce_to_pa_factors=False)
        # Aggregate mode
        p_col = list(cjtw)[0]
        a_col = list(cjtw)[1]
        cjtw = cjtw.reindex([p_col, a_col, 'trips'], axis=1)
        cjtw = cjtw.groupby([p_col, a_col]).sum().reset_index()

        # Handle cols - transpose
        print('cjtw:\n %s\n' % cjtw)
        unq_zones = nup.get_zone_range(productions[zone_col])
        cjtw = nup.df_to_np(cjtw,
                            v_heading=p_col,
                            h_heading=a_col,
                            values='trips',
                            unq_internal_zones=unq_zones)

        # Small infill
        cjtw = np.where(cjtw == 0, seed_infill, cjtw)

        # ## DO THE EXTERNAL DISTRIBUTION ## #
        ei = init_params.index

        # Path tlb folder
        tlb_folder = os.path.join(
            self.import_folder,
            'trip_length_bands',
            self.params['external_tlb_area'],
            self.params['external_tlb_name'],
        )
        print("tlb_folder: %s" % tlb_folder)

        # Loop through each of the segments
        internal_p_vector_eff_df = list()
        internal_a_vector_eff_df = list()
        for ed in ei:
            # Get segment_params from table
            segment_params = {}
            for ds in self.params['external_segmentation']:
                segment_params.update({ds: init_params[ds][ed]})

            # Get target trip length distribution
            trip_length_dist = nup.get_trip_length_bands(
                tlb_folder,
                segment_params,
                segmentation=self.params['external_tlb_name'],
                trip_origin=trip_origin,
            )

            # Convert from miles to KM
            trip_length_dist = trip_length_dist.reset_index(drop=True)
            trip_length_dist['min'] = trip_length_dist['lower'] * 1.61
            trip_length_dist['max'] = trip_length_dist['upper'] * 1.61

            # ## GET P/A VECTORS FOR THIS SEGMENT ## #
            # Filter productions to target distribution type
            sub_p, _ = nup.filter_pa_vector(
                productions,
                zone_col,
                segment_params,
                value_var='val',
                verbose=False,
            )
            sub_p = sub_p.rename(columns={'val': 'productions'})

            # Work out which attractions to use from purpose
            sub_a, _ = nup.filter_pa_vector(
                attractions,
                zone_col,
                segment_params,
                value_var='val',
                verbose=False,
            )
            sub_a = sub_a.rename(columns={'val': 'attractions'})

            print('attraction type: ', nup.get_attraction_type(segment_params))

            # Balance a to p
            print("total productions in segment: %s" % sub_p['productions'].sum())
            sub_a = nup.balance_a_to_p(
                zone_col,
                sub_p,
                sub_a,
                round_val=3,
                verbose=True,
            )

            # ## GET THE COSTS FOR THIS SEGMENT ## #
            print('Importing costs...')
            print("cost type: %s" % cost_type)
            print("Getting costs from: %s" % self.lookup_folder,)

            int_costs, cost_name = nup.get_costs(
                self.lookup_folder,
                segment_params,
                tp=cost_type,
                iz_infill=0.5,
                replace_nhb_with_hb=(trip_origin == 'nhb'),
            )
            print('Retrieved costs: %s' % cost_name)

            # Translate costs to array
            costs = nup.df_to_np(
                int_costs,
                v_heading='p_zone',
                h_heading='a_zone',
                values='cost',
                unq_internal_zones=unq_zones,
            )

            # ## RUN THE EXTERNAL MODEL ## #
            # Logging set up
            log_fname = du.calib_params_to_dist_name(
                trip_origin=trip_origin,
                matrix_format='external_log',
                calib_params=segment_params,
                csv=True,
            )
            log_path = os.path.join(self.tms_out['reports'], log_fname)

            # Replace the log if it already exists
            if os.path.isfile(log_path):
                os.remove(log_path)

            # Run
            gb_pa, bs_con = self._external_model(
                p=sub_p,
                a=sub_a,
                base_matrix=cjtw,
                costs=costs,
                convergence_target=0.9,
                target_tld=trip_length_dist,
                log_path=log_path,
                furness_tol=0.1,
                furness_max_iters=5000,
            )

            # ## WRITE A REPORT OF HOW THE EXTERNAL MODEL DID ## #
            # Build a report from external model
            _, tlb_con, _ = ra.get_trip_length_by_band(trip_length_dist, costs, gb_pa)
            report = tlb_con
            report['bs_con'] = bs_con

            # Build the output path
            fname = trip_origin + '_external_report'
            audit_path = os.path.join(self.tms_out['reports'], fname)
            audit_path = nup.build_path(audit_path, segment_params)
            report.to_csv(audit_path, index=False)

            # Build an IE report
            # Do int-ext report
            ie_report = nup.n_matrix_split(
                gb_pa,
                indices=[internal_index, external_index],
                index_names=['i', 'e'],
                summarise=True,
            )
            ie_report = pd.DataFrame(ie_report)

            # Build the output path
            fname = trip_origin + '_ie_report'
            ie_path = os.path.join(self.tms_out['reports'], fname)
            ie_path = nup.build_path(ie_path, segment_params)
            ie_report.to_csv(ie_path, index=False)

            # ## WRITE FULL DEMAND TO DISK ## #
            # Append zone names
            gb_pa = pd.DataFrame(gb_pa, index=unq_zones, columns=unq_zones)
            gb_pa = gb_pa.rename(columns={'index': zone_col})

            # Generate the path
            fname = trip_origin + '_external'
            ext_path = os.path.join(self.tms_out['external'], fname)
            ext_path = nup.build_path(ext_path, segment_params)
            gb_pa.to_csv(ext_path)

            # ## BUILD THE INTERNAL ONLY VECTORS ## #
            # Get the internal only demand
            internal_mask = mat_utils.get_internal_mask(
                df=pd.DataFrame(data=gb_pa, index=all_zones, columns=all_zones),
                zones=int_zones,
            )
            internal_pa = np.where(internal_mask, gb_pa, 0)

            # Create an index for the dataframes
            zone_index = pd.Index(all_zones, name=zone_col)

            # Production vector
            prods_eff_df = segment_params.copy()
            prods_eff_df['df'] = pd.DataFrame(
                data=np.sum(internal_pa, axis=1),
                index=zone_index,
                columns=[val_col],
            ).reset_index()
            internal_p_vector_eff_df.append(prods_eff_df)

            # Attraction Vector
            attrs_eff_df = segment_params.copy()
            attrs_eff_df['df'] = pd.DataFrame(
                data=np.sum(internal_pa, axis=0),
                index=zone_index,
                columns=[val_col],
            ).reset_index()
            internal_a_vector_eff_df.append(attrs_eff_df)

        # ## EXTERNAL DISTRIBUTION DONE WRITE OUTPUTS ## #
        # Build base names
        base_fname = "%s_%s_internal_%s.csv"
        col_names = [zone_col] + list(init_params) + [val_col]

        # Write out productions
        internal_productions = du.compile_efficient_df(internal_p_vector_eff_df, col_names)
        fname = base_fname % (zoning_name, trip_origin, 'productions')
        out_path = os.path.join(self.tms_out['p'], fname)
        internal_productions.to_csv(out_path, index=False)

        # Write out attractions
        internal_attractions = du.compile_efficient_df(internal_a_vector_eff_df, col_names)
        fname = base_fname % (zoning_name, trip_origin, 'attractions')
        out_path = os.path.join(self.tms_out['a'], fname)
        internal_attractions.to_csv(out_path, index=False)

    @staticmethod
    def _external_model(
        p,
        a,
        base_matrix,
        costs,
        log_path,
        target_tld: pd.DataFrame,
        convergence_target: float = 0.9,
        max_iters: int = 100,
        furness_tol: float = 0.1,
        furness_max_iters: int = 5000,
    ):
        """
        p:
            Production vector

        a:
            Attraction vector

        base_matrix:
            Seed matrix for transformation. Will usually be cjtw for shape.
            Could be 1s.

        costs:
            Cost/distance matrix for transformation

        calib_params:
            Distribution calib params. For write out desc only.

        model_lookup_path:
            Where are the lookups, for finding cjtw.

        model_name:
            Model name. Needs to be lower case I think.

        trip_origin:
            Is this HB or NHB? Takes 'hb' or 'nhb', shockingly.

        target_tld:
            pandas dataframe of the target trip length distributions. Should
            have min and max columns for each band, and these should be in
            KM.
        """
        # Transform original p/a into vectors
        target_p = p['productions'].values
        target_a = a['attractions'].values

        # Infill zeroes
        target_p = np.where(target_p == 0, 0.0001, target_p)
        target_a = np.where(target_a == 0, 0.0001, target_a)

        # Seed base
        gb_pa = base_matrix.copy()

        # Seed con crit
        bs_con = 0
        print('Initial band share convergence: ' + str(bs_con))

        # Calibrate
        for iter_num in range(max_iters):
            iter_start_time = timing.current_milli_time()

            # ## BAND SHARE ADJUSTMENT ## #
            gb_pa = correct_band_share(
                pa_mat=gb_pa,
                distance=costs,
                tld_band=target_tld,
            )

            # Furness across the other 2 dimensions
            gb_pa, furn_iters, furn_r2 = furness.doubly_constrained_furness(
                seed_vals=gb_pa,
                row_targets=target_p,
                col_targets=target_a,
                tol=furness_tol,
                max_iters=furness_max_iters,
            )

            # Get convergence
            _, tlb_con, _ = ra.get_trip_length_by_band(target_tld, costs, gb_pa)
            mse = math_utils.vector_mean_squared_error(
                vector1=tlb_con['tbs'].values,
                vector2=tlb_con['bs'].values,
            )

            prior_bs_con = bs_con
            bs_con = math_utils.curve_convergence(tlb_con['tbs'], tlb_con['bs'])

            iter_end_time = timing.current_milli_time()
            time_taken = iter_end_time - iter_start_time

            pa_diff = nup.get_pa_diff(
                gb_pa.sum(axis=1),
                target_p,
                gb_pa.sum(axis=0),
                target_a,
            )

            # ## LOG THIS ITERATION ## #
            # Log this iteration
            log_dict = {
                'Loop Num': str(iter_num),
                'run_time': time_taken,
                'furness_loops': furn_iters,
                'furness_r2': furn_r2,
                'pa_diff': np.round(pa_diff, 6),
                'bs_con': np.round(bs_con, 6),
                'bs_mse': np.round(mse, 8),
            }

            # Append this iteration to log file
            nup.safe_dataframe_to_csv(
                pd.DataFrame(log_dict, index=[0]),
                log_path,
                mode='a',
                header=(not os.path.exists(log_path)),
                index=False,
            )

            # ## EXIT EARLY IF CONDITIONS MET ## #
            # If we're stuck near the target, exit early
            diff = np.abs(bs_con - prior_bs_con)
            if diff < .0001 and bs_con > convergence_target - 0.1:
                break

            if bs_con > convergence_target:
                break

        return gb_pa, bs_con

    @staticmethod
    def adjust_trip_length_by_band(band_atl,
                                   adj_fac,
                                   distance,
                                   base_matrix):
        """
        Go over atl, adjust trips as required
        """

        # TODO: Drop averages of nothing in trip length band
        # reset index, needed or not
        band_atl = band_atl.reset_index(drop=True)

        # Get global atl
        global_atl = ra.get_trip_length(distance,
                                        base_matrix)

        # Get min max for each
        ph = band_atl['tlb_desc'].str.split('-', n=1, expand=True)
        band_atl['min'] = ph[0].str.replace('(', '')
        band_atl['max'] = ph[1].str.replace('[', '')
        band_atl['min'] = band_atl['min'].str.replace('(', '').values
        band_atl['max'] = band_atl['max'].str.replace(']', '').values
        del ph

        mat_ph = []
        out_mat = np.empty(shape=[len(base_matrix), len(base_matrix)])

        # Loop over rows in band_atl
        for index, row in band_atl.iterrows():
            # Get band mask
            band_mat = np.where(
                (distance >= float(row['min'])) & (
                        distance < float(row['max'])), True, False)

            adj_mat = band_mat * base_matrix * adj_fac['adj_fac'][index]

            mat_ph.append(adj_mat)

        for o_mat in mat_ph:
            out_mat = out_mat + o_mat

        return out_mat

    def pld_infill(self,
                   external_pa,
                   pld_path=_default_pld_path,
                   infill_mode=6):
        """
        external_pa:
            Full set of external pa before infill
    
        pld_path:
            Path to pld_data. Auto fills from script. Just keep an eye on it.
    
        infill_mode:
            mode to apply the infill to.
    
        """
        pld_pa = pd.read_csv(pld_path)

        # Translate the ca/nca to 2,1
        ca_mat = pd.DataFrame({'ca': ['ca', 'nca'],
                               'ca_code': [2, 1]})
        pld_pa = pld_pa.merge(ca_mat,
                              how='left',
                              on='ca')
        del (pld_pa['ca'])
        pld_pa = pld_pa.rename(columns={'ca_code': 'ca',
                                        'purpose': 'ap'})
        pld_pa = pld_pa.reindex(['p_zone',
                                 'a_zone',
                                 'ap',
                                 'ca',
                                 'dt'], axis=1)

        # Get original total
        pld_total = pld_pa['dt'].sum()

        # Get external pa
        w_external_pa = external_pa.copy()

        # Isolate mode to be infilled
        infill_subset = w_external_pa[
            w_external_pa['m'] == infill_mode].copy()
        other_mode = w_external_pa[
            w_external_pa['m'] != infill_mode].copy()

        # Get original pa total    
        opa_total = infill_subset['dt'].sum()

        # Get shares by zone for infill
        infill_subset = infill_subset.reindex(['p_zone',
                                               'p',
                                               'ca',
                                               'dt'],
                                              axis=1).groupby(
            ['p_zone',
             'p',
             'ca']).sum().reset_index()

        # Build aggregate purpose
        agg_purp = pd.DataFrame({'p': [1, 2, 3, 4, 5, 6, 7, 8],
                                 'ap': ['commute',
                                        'business',
                                        'other',
                                        'other',
                                        'other',
                                        'other',
                                        'other',
                                        'other']})

        infill_subset = infill_subset.merge(agg_purp,
                                            how='left',
                                            on='p')

        # Get zone totals by aggregate purpose and ca
        zone_totals = infill_subset.reindex(
            ['p_zone', 'ap', 'ca', 'dt'],
            axis=1).groupby(
            ['p_zone', 'ap', 'ca']).sum().reset_index()
        zone_totals = zone_totals.rename(columns={'dt': 'total'})

        infill_subset = infill_subset.merge(zone_totals,
                                            how='left',
                                            on=['p_zone', 'ap', 'ca'])
        infill_subset['factor'] = infill_subset['dt'] / infill_subset['total']
        infill_subset = infill_subset.drop(['dt', 'total'], axis=1)

        # Factor test
        factor_test = infill_subset.reindex(
            ['p_zone', 'ap', 'ca', 'factor'],
            axis=1).groupby(['p_zone', 'ap', 'ca']).sum().reset_index()

        print('Min factor: ' + str(min(factor_test['factor'])))
        print('Max factor: ' + str(max(factor_test['factor'])))

        # Gonna be:
        # Infill left = [p_zone, a_zone, ap, ca, dt]
        # Infill right = [p_zone, ap, ca, p, factor]]
        # On = [p_zone, ap, ca]
        # dt * factor
        infill_output = pld_pa.merge(infill_subset,
                                     how='left',
                                     on=['p_zone', 'ap', 'ca'])
        infill_output['dt'] = infill_output['dt'] * infill_output['factor']
        infill_output['m'] = infill_mode
        list(infill_output)

        # Get original cols
        org_cols = list(other_mode)
        og_cols = org_cols.copy()
        og_cols.remove('dt')

        # Reindex for reentry
        infill_output = infill_output.reindex(org_cols,
                                              axis=1).groupby(
            og_cols).sum(
        ).reset_index(
        )

        # Correct
        corr_fac = pld_total / infill_output['dt'].sum()
        infill_output['dt'] = (infill_output['dt'] * corr_fac).round(3)
        new_total = infill_output['dt'].sum()

        # Benchmark
        print('Orignal pld total: ' + str(pld_total))
        print('Original external total: ' + str(opa_total))
        print('New total: ' + str(new_total))

        # Reappend
        w_external_pa = pd.concat([infill_output, other_mode], sort=True)

        return w_external_pa


def correct_band_share(pa_mat,
                       distance,
                       tld_band,
                       seed_infill=.001,
                       ):
    """
    Adjust band shares of rows or columnns

    pa_mat pa:
        Square matrix
    band_totals:
        list of dictionaries of trip lenth bands
    seed_infill = .0001:
        Seed in fill to balance
    axis = 1:
        Axis to adjust band share, takes 0 or 1
    """
    # Init
    total_trips = np.sum(pa_mat)
    out_mat = np.zeros_like(pa_mat)

    # Internal trips: i-i, i-e
    # External trips: e-i, e-e

    # Adjust bands one at a time
    for index, row in tld_band.iterrows():
        # Get proportion of all trips that should be in this band
        target_band_share = row['band_share']
        target_band_trips = total_trips * target_band_share

        # Get proportion of all trips that are in this band
        distance_mask = (distance >= float(row['min'])) & (distance < float(row['max']))
        distance_bool = np.where(distance_mask, 1, 0)
        band_trips = pa_mat * distance_bool
        achieved_band_trips = np.sum(band_trips)

        # infill
        achieved_band_trips = np.where(achieved_band_trips==0, seed_infill, achieved_band_trips)

        # Adjust the matrix by difference
        adjustment = target_band_trips / achieved_band_trips
        adj_mat = band_trips * adjustment

        # Add into the return matrix
        out_mat += adj_mat

    return out_mat
