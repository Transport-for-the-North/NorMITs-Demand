# -*- coding: utf-8 -*-
"""
Created on: 27/02/2020
Updated on:

Original author: Chris Storey
Last update made by: Ben Taylor
Other updates made by:

File purpose:

"""
# Built-Ins
import os

from typing import List
from typing import Union

# Third Party
import pandas as pd
import numpy as np

# Local Imports
import normits_demand as nd

from normits_demand.distribution import furness
from normits_demand.matrices import utils as mat_utils
from normits_demand.reports import reports_audits as ra

from normits_demand.utils import utils as nup
from normits_demand.utils import general as du
from normits_demand.utils import timing
from normits_demand.utils import math_utils
from normits_demand.utils import costs as costs_utils
from normits_demand.utils import trip_length_distributions as tld_utils


# import normits_demand.build.tms_pathing as tms
# class ExternalModel(tms.TMSPathing):
class ExternalModel():
    _base_zone_col = "%s_zone_id"

    def __init__(self,
                 zoning_system: nd.core.ZoningSystem,
                 zone_col: str = None,
                 ):
        # Validate inputs
        if not isinstance(zoning_system, nd.core.zoning.ZoningSystem):
            raise ValueError(
                "Expected and instance of a normits_demand ZoningSystem. "
                "Got a %s instance instead."
                % type(zoning_system)
            )

        # Assign attributes
        self.zoning_system = zoning_system
        self.zone_col = zone_col

        if self.zone_col is None:
            self.zone_col = zoning_system.col_name

    def run(self,
            trip_origin: str,
            cost_type: str,
            productions: pd.DataFrame,
            attractions: pd.DataFrame,
            seed_matrix: np.ndarray,
            internal_tld_dir: nd.PathLike,
            external_tld_dir: nd.PathLike,
            costs_dir: nd.PathLike,
            reports_dir: nd.PathLike,
            production_out: nd.PathLike,
            attraction_out: nd.PathLike,
            external_dist_out: nd.PathLike,
            running_segmentation: nd.core.SegmentationLevel,
            ):
        """

        Parameters
        ----------
        trip_origin: 'hb'
        cost_type: '24hr'
        internal_tld_path
        external_tld_path

        Returns
        -------

        """
        # TODO(BT): Make sure the P/A vectors are the right zoning system
        # Define internal name
        val_col = 'val'

        # Loop through each of the segments
        internal_p_vector_eff_df = list()
        internal_a_vector_eff_df = list()
        for segment_params in running_segmentation:

            # Get target trip length distribution
            internal_tld = tld_utils.get_trip_length_distributions(
                import_dir=internal_tld_dir,
                segment_params=segment_params,
                trip_origin=trip_origin,
            )

            external_tld = tld_utils.get_trip_length_distributions(
                import_dir=external_tld_dir,
                segment_params=segment_params,
                trip_origin=trip_origin,
            )

            # Convert from miles to KM
            internal_tld = internal_tld.reset_index(drop=True)
            internal_tld['min'] = internal_tld['lower'] * 1.61
            internal_tld['max'] = internal_tld['upper'] * 1.61

            # Convert from miles to KM
            external_tld = external_tld.reset_index(drop=True)
            external_tld['min'] = external_tld['lower'] * 1.61
            external_tld['max'] = external_tld['upper'] * 1.61

            # ## GET P/A VECTORS FOR THIS SEGMENT ## #
            # Filter productions to target distribution type
            sub_p, _ = nup.filter_pa_vector(
                productions,
                self.zone_col,
                segment_params,
                value_var='val',
                verbose=False,
            )
            sub_p = sub_p.rename(columns={'val': 'productions'})

            # Work out which attractions to use from purpose
            sub_a, _ = nup.filter_pa_vector(
                attractions,
                self.zone_col,
                segment_params,
                value_var='val',
                verbose=False,
            )
            sub_a = sub_a.rename(columns={'val': 'attractions'})

            # Balance A to P
            adj_factor = sub_p['productions'].sum() / sub_a['attractions'].sum()
            sub_a['attractions'] *= adj_factor

            # ## GET THE COSTS FOR THIS SEGMENT ## #
            print('Importing costs...')
            print("cost type: %s" % cost_type)
            print("Getting costs from: %s" % costs_dir)

            int_costs, cost_name = costs_utils.get_costs(
                costs_dir,
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
                unq_internal_zones=self.zoning_system.unique_zones,
            )

            # ## RUN THE EXTERNAL MODEL ## #
            # Logging set up
            log_fname = du.calib_params_to_dist_name(
                trip_origin=trip_origin,
                matrix_format='external_log',
                calib_params=segment_params,
                csv=True,
            )
            log_path = os.path.join(reports_dir, log_fname)

            # Replace the log if it already exists
            if os.path.isfile(log_path):
                os.remove(log_path)

            # Run
            gb_pa, int_bs_con, ext_bs_con, overall_bs_con = self._external_model(
                p=sub_p,
                a=sub_a,
                base_matrix=seed_matrix,
                costs=costs,
                convergence_target=0.9,
                int_target_tld=internal_tld,
                ext_target_tld=external_tld,
                internal_index=internal_index,
                external_index=external_index,
                log_path=log_path,
                furness_tol=0.1,
                furness_max_iters=5000,
            )

            # ## WRITE A REPORT OF HOW THE EXTERNAL MODEL DID ## #
            # Build a report from external model
            _, tlb_con, _ = ra.get_trip_length_by_band(internal_tld, costs, gb_pa)
            report = tlb_con
            report['int_bs_con'] = int_bs_con
            report['ext_bs_con'] = ext_bs_con
            report['overall_bs_con'] = overall_bs_con

            # Build the output path
            fname = trip_origin + '_external_report'
            audit_path = os.path.join(reports_dir, fname)
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
            # Dictionary of ii, ie, ei, and ee trips
            ie_report = pd.DataFrame(ie_report)

            # Build the output path
            fname = trip_origin + '_ie_report'
            ie_path = os.path.join(reports_dir, fname)
            ie_path = nup.build_path(ie_path, segment_params)
            ie_report.to_csv(ie_path, index=False)

            # ## WRITE FULL DEMAND TO DISK ## #
            # Append zone names
            all_zones = self.zoning_system.unique_zones
            gb_pa = pd.DataFrame(gb_pa, index=all_zones, columns=all_zones)
            gb_pa = gb_pa.rename(columns={'index': self.zone_col})

            # Generate the path
            fname = trip_origin + '_external'
            ext_path = os.path.join(external_dist_out, fname)
            ext_path = nup.build_path(ext_path, segment_params)
            gb_pa.to_csv(ext_path)

            # ## BUILD THE INTERNAL ONLY VECTORS ## #
            # Get the internal only demand
            internal_mask = mat_utils.get_internal_mask(
                df=pd.DataFrame(data=gb_pa, index=all_zones, columns=all_zones),
                zones=self.zoning_system.internal_zones,
            )
            internal_pa = np.where(internal_mask, gb_pa, 0)

            # Create an index for the dataframes
            zone_index = pd.Index(all_zones, name=self.zone_col)

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
        segment_names = running_segmentation.naming_order
        col_names = [self.zone_col] + segment_names + [val_col]

        # Write out productions
        internal_productions = du.compile_efficient_df(internal_p_vector_eff_df, col_names)
        fname = base_fname % (self.zoning_system.name, trip_origin, 'productions')
        out_path = os.path.join(production_out, fname)
        internal_productions.to_csv(out_path, index=False)

        # Write out attractions
        internal_attractions = du.compile_efficient_df(internal_a_vector_eff_df, col_names)
        fname = base_fname % (self.zoning_system.name, trip_origin, 'attractions')
        out_path = os.path.join(attraction_out, fname)
        internal_attractions.to_csv(out_path, index=False)

    @staticmethod
    def _external_model(
        p,
        a,
        base_matrix,
        costs,
        log_path,
        int_target_tld: pd.DataFrame,
        ext_target_tld: pd.DataFrame,
        internal_index: Union[np.ndarray, List[int]],
        external_index: Union[np.ndarray, List[int]],
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

        # Calibrate
        int_bs_con = 0
        ext_bs_con = 0
        overall_bs_con = 0
        for iter_num in range(max_iters):
            iter_start_time = timing.current_milli_time()

            # ## BAND SHARE ADJUSTMENT ## #
            gb_pa = correct_band_share(
                pa_mat=gb_pa,
                distance=costs,
                int_tld_band=int_target_tld,
                ext_tld_band=ext_target_tld,
                int_index=internal_index,
                ext_index=external_index,
            )

            # Furness across the other 2 dimensions
            gb_pa, furn_iters, furn_r2 = furness.doubly_constrained_furness(
                seed_vals=gb_pa,
                row_targets=target_p,
                col_targets=target_a,
                tol=furness_tol,
                max_iters=furness_max_iters,
            )

            # Split out the internal and external trips
            internal_pa = gb_pa[:max(internal_index), :]
            internal_dist = costs[:max(internal_index), :]

            external_pa = gb_pa[max(internal_index):, :]
            external_dist = costs[max(internal_index):, :]

            # Internal vals
            _, tlb_con, _ = ra.get_trip_length_by_band(int_target_tld, internal_dist, internal_pa)
            int_bs_con = math_utils.curve_convergence(tlb_con['tbs'], tlb_con['bs'])
            int_mse = math_utils.vector_mean_squared_error(
                vector1=tlb_con['tbs'].values,
                vector2=tlb_con['bs'].values,
            )

            # External vals
            _, tlb_con, _ = ra.get_trip_length_by_band(ext_target_tld, external_dist, external_pa)
            ext_bs_con = math_utils.curve_convergence(tlb_con['tbs'], tlb_con['bs'])
            ext_mse = math_utils.vector_mean_squared_error(
                vector1=tlb_con['tbs'].values,
                vector2=tlb_con['bs'].values,
            )

            # Overall
            _, tlb_con, _ = ra.get_trip_length_by_band(int_target_tld, costs, gb_pa)
            overall_bs_con = math_utils.curve_convergence(tlb_con['tbs'], tlb_con['bs'])

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
                'internal_bs_con': np.round(int_bs_con, 6),
                'external_bs_con': np.round(ext_bs_con, 6),
                'overall_bs_con': np.round(overall_bs_con, 6),
                'internal_bs_mse': np.round(int_mse, 8),
                'external_bs_mse': np.round(ext_mse, 8),
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
            if int_bs_con > convergence_target and ext_bs_con > convergence_target:
                break

        return gb_pa, int_bs_con, ext_bs_con, overall_bs_con

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


def correct_band_share(pa_mat,
                       distance,
                       int_tld_band,
                       ext_tld_band,
                       int_index,
                       ext_index,
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
    # Internal trips: i-i, i-e
    # External trips: e-i, e-e

    # Split out the internal and external trips
    internal_pa = pa_mat[:max(int_index), :]
    internal_dist = distance[:max(int_index), :]
    int_out = np.zeros_like(internal_pa)

    external_pa = pa_mat[max(int_index):, :]
    external_dist = distance[max(int_index):, :]
    ext_out = np.zeros_like(external_pa)

    for trips, dist, out_mat, tld_band in [(internal_pa, internal_dist, int_out, int_tld_band),
                                           (external_pa, external_dist, ext_out, ext_tld_band)]:
        # Init
        total_trips = np.sum(trips)

        # Adjust bands one at a time
        for index, row in tld_band.iterrows():
            # Get proportion of all trips that should be in this band
            target_band_share = row['band_share']
            target_band_trips = total_trips * target_band_share

            # Get proportion of all trips that are in this band
            distance_mask = (dist >= float(row['min'])) & (dist < float(row['max']))
            distance_bool = np.where(distance_mask, 1, 0)
            band_trips = trips * distance_bool
            achieved_band_trips = np.sum(band_trips)

            # infill
            achieved_band_trips = np.where(achieved_band_trips==0, seed_infill, achieved_band_trips)

            # Adjust the matrix by difference
            adjustment = target_band_trips / achieved_band_trips
            adj_mat = band_trips * adjustment

            # Add into the return matrix
            out_mat += adj_mat

    # Stick internal and external back together
    out_mat = np.vstack([int_out, ext_out])

    return out_mat
