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

from normits_demand.pathing import ExternalModelExportPaths

from normits_demand.utils import utils as nup
from normits_demand.utils import general as du
from normits_demand.utils import timing
from normits_demand.utils import math_utils
from normits_demand.utils import costs as costs_utils
from normits_demand.utils import pandas_utils as pd_utils
from normits_demand.utils import trip_length_distributions as tld_utils


# import normits_demand.build.tms_pathing as tms
# class ExternalModel(tms.TMSPathing):
class ExternalModel(ExternalModelExportPaths):
    _log_fname = "External_Model_log.log"

    _base_zone_col = "%s_zone_id"
    _pa_val_col = 'trips'

    def __init__(self,
                 zoning_system: nd.core.ZoningSystem,
                 export_home: nd.PathLike,
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

        # Make sure the reports paths exists
        report_home = os.path.join(export_home, "Logs & Reports")

        # Build the output paths
        super().__init__(
            export_home=export_home,
            report_home=report_home,
        )

        # Create a logger
        logger_name = "%s.%s" % (__name__, self.__class__.__name__)
        log_file_path = os.path.join(self.export_home, self._log_fname)
        self._logger = nd.get_logger(
            logger_name=logger_name,
            log_file_path=log_file_path,
            instantiate_msg="Initialised new TMS Logger",
        )

    def run(self,
            trip_origin: str,
            productions: pd.DataFrame,
            attractions: pd.DataFrame,
            seed_matrix: np.ndarray,
            internal_tld_dir: nd.PathLike,
            external_tld_dir: nd.PathLike,
            costs_path: nd.PathLike,
            running_segmentation: nd.core.SegmentationLevel,
            pa_val_col: str = 'val',
            ) -> None:
        # TODO(BT): Make sure the P/A vectors are the right zoning system
        # TODO(BT): Make sure pa_val_col is in P/A vectors
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
            # Figure out which columns we need
            segments = list(segment_params.keys())
            needed_cols = segments + [self._pa_val_col]
            rename_cols = {pa_val_col: self._pa_val_col}

            # Filter productions
            productions = pd_utils.filter_df(productions, segment_params)
            productions = productions.rename(columns=rename_cols)
            productions = productions.set_index(self.zone_col)
            productions = productions.reindex(
                index=self.zoning_system.unique_zones,
                columns=needed_cols,
                fill_value=0,
            ).reset_index()

            # Filter attractions
            attractions = pd_utils.filter_df(attractions, segment_params)
            attractions = attractions.rename(columns=rename_cols)
            attractions = attractions.set_index(self.zone_col)
            attractions = attractions.reindex(
                index=self.zoning_system.unique_zones,
                columns=needed_cols,
                fill_value=0,
            ).reset_index()

            # Balance A to P
            adj_factor = productions[self._pa_val_col].sum() / attractions[self._pa_val_col].sum()
            attractions[self._pa_val_col] *= adj_factor

            # ## GET THE COSTS FOR THIS SEGMENT ## #
            self._logger.info('Importing costs')
            self._logger.debug("Getting costs from: %s" % costs_path)

            int_costs, cost_name = costs_utils.get_costs(
                costs_path,
                segment_params,
                iz_infill=0.5,
                replace_nhb_with_hb=(trip_origin == 'nhb'),
            )

            # Translate costs to array
            costs = pd_utils.long_to_wide_infill(
                df=int_costs,
                index_col='p_zone',
                columns_col='a_zone',
                values_col='cost',
                index_vals=self.zoning_system.unique_zones,
                column_vals=self.zoning_system.unique_zones,
                infill=0,
            )

            # ## RUN THE EXTERNAL MODEL ## #
            # Logging set up
            log_fname = du.segment_params_to_dist_name(
                trip_origin=trip_origin,
                matrix_format='external_log',
                calib_params=segment_params,
                csv=True,
            )
            log_path = os.path.join(self.report_paths.model_log_dir, log_fname)

            # Replace the log if it already exists
            if os.path.isfile(log_path):
                os.remove(log_path)

            # Run
            gb_pa, int_report, ext_report = self._external_model(
                productions=productions,
                attractions=attractions,
                base_matrix=seed_matrix,
                costs=costs,
                convergence_target=0.9,
                int_target_tld=internal_tld,
                ext_target_tld=external_tld,
                log_path=log_path,
                furness_tol=0.1,
                furness_max_iters=5000,
            )

            # TODO: CHNAGE HOW OUTPUT PATHS ARE GENERATED
            # ## WRITE A REPORT OF HOW THE EXTERNAL MODEL DID ## #
            # Build the output path
            fname = trip_origin + '_internal_report'
            path = os.path.join(self.report_paths.tld_report_dir, fname)
            path = nup.build_path(path, segment_params)  # TODO: CHANGE
            int_report.to_csv(path, index=False)

            # Build the output path
            fname = trip_origin + '_external_report'
            path = os.path.join(self.report_paths.tld_report_dir, fname)
            path = nup.build_path(path, segment_params) # TODO: CHANGE
            ext_report.to_csv(path, index=False)

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
            ie_path = os.path.join(self.report_paths.ie_report_dir, fname)
            ie_path = nup.build_path(ie_path, segment_params)
            ie_report.to_csv(ie_path, index=False)

            # ## WRITE FULL DEMAND TO DISK ## #
            # Append zone names
            all_zones = self.zoning_system.unique_zones
            gb_pa = pd.DataFrame(gb_pa, index=all_zones, columns=all_zones)
            gb_pa = gb_pa.rename(columns={'index': self.zone_col})

            # Generate the path
            fname = trip_origin + '_external'
            ext_path = os.path.join(self.export_paths.external_distribution_dir, fname)
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

    def _external_model(self,
                        productions,
                        attractions,
                        base_matrix,
                        costs,
                        log_path,
                        int_target_tld: pd.DataFrame,
                        ext_target_tld: pd.DataFrame,
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
        # Make sure productions and attractions contain all zones
        if len(productions) != self.zoning_system.n_zones:
            raise ValueError(
                "The passed in productions do not have the right number of "
                "zones. Expected %s zones, got %s zones."
                % (self.zoning_system.n_zones, len(productions))
            )

        if len(attractions) != self.zoning_system.n_zones:
            raise ValueError(
                "The passed in attractions do not have the right number of "
                "zones. Expected %s zones, got %s zones."
                % (self.zoning_system.n_zones, len(attractions))
            )

        # Infill zeroes
        productions = productions.mask(productions == 0, 0.0001)
        attractions = attractions.mask(attractions == 0, 0.0001)

        # Seed base
        gb_pa = base_matrix.copy()

        # Initialise report values
        int_bs_con = -1
        ext_bs_con = -1
        int_tlb_con = pd.DataFrame({'bs_con': int_bs_con}, index=[0])
        ext_tlb_con = pd.DataFrame({'bs_con': ext_bs_con}, index=[0])

        # Perform a 3D furness
        for iter_num in range(max_iters):
            iter_start_time = timing.current_milli_time()

            # ## BAND SHARE ADJUSTMENT ## #
            gb_pa = correct_band_share(
                pa_mat=gb_pa,
                distance=costs,
                int_tld_band=int_target_tld,
                ext_tld_band=ext_target_tld,
                internal_zones=self.zoning_system.internal_zones,
                external_zones=self.zoning_system.external_zones,
            )

            # Furness across the other 2 dimensions
            gb_pa, furn_iters, furn_r2 = furness.furness_pandas_wrapper(
                seed_values=gb_pa,
                row_targets=productions,
                col_targets=attractions,
                tol=furness_tol,
                max_iters=furness_max_iters,
                normalise_seeds=False,
                idx_col=self.zone_col,
                unique_col=self._pa_val_col,
            )

            # Split out the internal and external trips
            internal_mask = gb_pa.index.isin(self.zoning_system.internal_zones)
            internal_pa = gb_pa[internal_mask].values
            internal_dist = costs[internal_mask].values

            external_mask = gb_pa.index.isin(self.zoning_system.external_zones)
            external_pa = gb_pa[external_mask].values
            external_dist = costs[external_mask].values

            # Internal vals
            _, int_tlb_con, _ = ra.get_trip_length_by_band(int_target_tld, internal_dist, internal_pa)
            int_bs_con = math_utils.curve_convergence(int_tlb_con['tbs'], int_tlb_con['bs'])
            int_mse = math_utils.vector_mean_squared_error(
                vector1=int_tlb_con['tbs'].values,
                vector2=int_tlb_con['bs'].values,
            )

            # External vals
            _, ext_tlb_con, _ = ra.get_trip_length_by_band(ext_target_tld, external_dist, external_pa)
            ext_bs_con = math_utils.curve_convergence(ext_tlb_con['tbs'], ext_tlb_con['bs'])
            ext_mse = math_utils.vector_mean_squared_error(
                vector1=ext_tlb_con['tbs'].values,
                vector2=ext_tlb_con['bs'].values,
            )

            iter_end_time = timing.current_milli_time()
            time_taken = iter_end_time - iter_start_time

            pa_diff = nup.get_pa_diff(
                gb_pa.values.sum(axis=1),
                productions[self._pa_val_col].values,
                gb_pa.values.sum(axis=0),
                attractions[self._pa_val_col].values,
            )

            # ## LOG THIS ITERATION ## #
            # Log this iteration
            log_dict = {
                'Loop Num': str(iter_num),
                'run_time(ms)': time_taken,
                'furness_loops': furn_iters,
                'furness_r2': furn_r2,
                'pa_diff': np.round(pa_diff, 6),
                'internal_bs_con': np.round(int_bs_con, 6),
                'external_bs_con': np.round(ext_bs_con, 6),
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

        # Put together a report on the final performance
        int_tlb_con['bs_con'] = int_bs_con
        ext_tlb_con['bs_con'] = ext_bs_con

        return gb_pa, int_tlb_con, ext_tlb_con

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
                       internal_zones,
                       external_zones,
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

    out_ph = list()
    for zones, tld_band in [(internal_zones, int_tld_band), (external_zones, ext_tld_band)]:
        # Filter down to wanted area
        mask = pa_mat.index.isin(zones)
        trips = pa_mat[mask]
        dist = distance[mask]
        out = np.zeros_like(trips)

        total_trips = trips.values.sum()

        # Adjust bands one at a time
        for index, row in tld_band.iterrows():
            # Get proportion of all trips that should be in this band
            target_band_share = row['band_share']
            target_band_trips = total_trips * target_band_share

            # Get proportion of all trips that are in this band
            distance_mask = (dist >= float(row['min'])) & (dist < float(row['max']))
            distance_bool = np.where(distance_mask, 1, 0)
            band_trips = trips * distance_bool
            ach_band_trips = band_trips.values.sum()

            # # infill
            # ach_band_trips = ach_band_trips.mask(ach_band_trips == 0, seed_infill)

            # Adjust the matrix by difference
            adjustment = target_band_trips / ach_band_trips
            adj_mat = band_trips * adjustment

            # Add into the return matrix
            out += adj_mat

        out_ph.append(out)

    # Stick internal and external back together
    return pd.concat(out_ph)
