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

from typing import Optional

# Third Party
import pandas as pd
import numpy as np

# Local Imports
import normits_demand as nd
from normits_demand import constants as consts

from normits_demand.distribution import furness

from normits_demand.pathing.travel_market_synthesiser import ExternalModelExportPaths
from normits_demand.validation import checks

from normits_demand.concurrency import multiprocessing

from normits_demand.utils import general as du
from normits_demand.utils import timing
from normits_demand.utils import math_utils
from normits_demand.utils import file_ops
from normits_demand.utils import costs as costs_utils
from normits_demand.utils import pandas_utils as pd_utils
from normits_demand.utils import trip_length_distributions as tld_utils


class ExternalModel(ExternalModelExportPaths):
    _log_fname = "External_Model_log.log"

    _base_zone_col = "%s_zone_id"
    _pa_val_col = 'trips'

    _external_only_suffix = 'ext'

    def __init__(self,
                 year: int,
                 running_mode: nd.Mode,
                 zoning_system: nd.core.ZoningSystem,
                 export_home: nd.PathLike,
                 zone_col: str = None,
                 process_count: Optional[int] = consts.PROCESS_COUNT,
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
        self.process_count = process_count

        if self.zone_col is None:
            self.zone_col = zoning_system.col_name

        # Make sure the reports paths exists
        report_home = os.path.join(export_home, "Logs & Reports")
        file_ops.create_folder(report_home)

        # Build the output paths
        super().__init__(
            year=year,
            running_mode=running_mode,
            export_home=export_home,
        )

        # Create a logger
        logger_name = "%s.%s" % (nd.get_package_logger_name(), self.__class__.__name__)
        log_file_path = os.path.join(self.export_home, self._log_fname)
        self._logger = nd.get_logger(
            logger_name=logger_name,
            log_file_path=log_file_path,
            instantiate_msg="Initialised new External Model Logger",
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
            intrazonal_cost_infill: Optional[float] = 0.5,
            pa_val_col: Optional[str] = 'val',
            convergence_target: float = 0.9,
            furness_tol: float = 0.1,
            furness_max_iters: int = 5000,
            ) -> None:
        # TODO(BT): Make sure the P/A vectors are the right zoning system
        # TODO(BT): Make sure pa_val_col is in P/A vectors
        # Validate the trip origin
        trip_origin = checks.validate_trip_origin(trip_origin)

        # ## MULTIPROCESS ACROSS SEGMENTS ## #
        unchanging_kwargs = {
            'trip_origin': trip_origin,
            'internal_tld_dir': internal_tld_dir,
            'external_tld_dir': external_tld_dir,
            'running_segmentation': running_segmentation,
            'pa_val_col': pa_val_col,
            'costs_path': costs_path,
            'intrazonal_cost_infill': intrazonal_cost_infill,
            'seed_matrix': seed_matrix,
            'convergence_target': convergence_target,
            'furness_tol': furness_tol,
            'furness_max_iters': furness_max_iters,
        }

        pbar_kwargs = {
            'desc': 'External model',
            'unit': 'segment',
        }

        # Build a list of kwargs
        kwarg_list = list()
        for segment_params in running_segmentation:
            # ## GET P/A VECTORS FOR THIS SEGMENT ## #
            # Figure out which columns we need
            segments = list(segment_params.keys())
            needed_cols = segments + [self._pa_val_col]
            rename_cols = {pa_val_col: self._pa_val_col}

            # Filter productions
            seg_productions = pd_utils.filter_df(
                df=productions,
                df_filter=segment_params,
                throw_error=True,
            )
            seg_productions = seg_productions.rename(columns=rename_cols)
            seg_productions = seg_productions.set_index(self.zone_col)
            seg_productions = seg_productions.reindex(
                index=self.zoning_system.unique_zones,
                columns=needed_cols,
                fill_value=0,
            ).reset_index()

            # Filter attractions
            seg_attractions = pd_utils.filter_df(
                df=attractions,
                df_filter=segment_params,
                throw_error=True,
            )
            seg_attractions = seg_attractions.rename(columns=rename_cols)
            seg_attractions = seg_attractions.set_index(self.zone_col)
            seg_attractions = seg_attractions.reindex(
                index=self.zoning_system.unique_zones,
                columns=needed_cols,
                fill_value=0,
            ).reset_index()

            # Check we actually got something
            production_sum = seg_productions[self._pa_val_col].values.sum()
            attraction_sum = seg_attractions[self._pa_val_col].values.sum()
            if production_sum <= 0 or attraction_sum <= 0:
                raise nd.NormitsDemandError(
                    "Missing productions and/or attractions after filtering to "
                    "this segment.\n"
                    "\tSegment: %s\n"
                    "\tProductions sum: %s\n"
                    "\tAttractions sum: %s"
                    % (segment_params, production_sum, attraction_sum)
                )

            # Balance A to P
            adj_factor = production_sum / attraction_sum
            seg_attractions[self._pa_val_col] *= adj_factor

            # Build the kwargs
            kwargs = unchanging_kwargs.copy()
            kwargs.update({
                'segment_params': segment_params,
                'seg_productions': seg_productions,
                'seg_attractions': seg_attractions,
            })
            kwarg_list.append(kwargs)

        # Multiprocess
        return_vals = multiprocessing.multiprocess(
            fn=self._run_internal,
            kwargs=kwarg_list,
            process_count=self.process_count,
            pbar_kwargs=pbar_kwargs,
        )

        # Unpack the return values
        internal_p_vector_eff_df, internal_a_vector_eff_df = zip(*return_vals)

        # ## EXTERNAL DISTRIBUTION DONE WRITE OUTPUTS ## #
        # Choose which dirs to use
        if trip_origin == 'hb':
            productions_path = self.export_paths.hb_internal_productions
            attractions_path = self.export_paths.hb_internal_attractions
        elif trip_origin == 'nhb':
            productions_path = self.export_paths.nhb_internal_productions
            attractions_path = self.export_paths.nhb_internal_attractions
        else:
            raise ValueError(
                "'%s' is not a valid trip origin! How did this error happen? "
                "Trip origin should have already been validated in this "
                "function.s"
            )

        # Build col names
        segment_names = running_segmentation.naming_order
        col_names = [self.zone_col] + segment_names + [pa_val_col]

        # Write out productions
        internal_productions = du.compile_efficient_df(internal_p_vector_eff_df, col_names)
        internal_productions.to_csv(productions_path, index=False)

        # Write out attractions
        internal_attractions = du.compile_efficient_df(internal_a_vector_eff_df, col_names)
        internal_attractions.to_csv(attractions_path, index=False)

    def _run_internal(self,
                      trip_origin,
                      internal_tld_dir,
                      external_tld_dir,
                      running_segmentation,
                      pa_val_col,
                      costs_path,
                      intrazonal_cost_infill,
                      seed_matrix,
                      segment_params,
                      seg_productions,
                      seg_attractions,
                      convergence_target,
                      furness_tol,
                      furness_max_iters,
                      ):
        """Internal looping function of self.run

        Returns
        -------
        productions_efficient_df:
            An "efficient dataframe" dictionary of the internal productions
            generated in this segment.

        attractions_efficient_df:
            An "efficient dataframe" dictionary of the internal productions
            generated in this segment.
        """
        name = running_segmentation.generate_file_name(segment_params)
        self._logger.debug("Running for %s" % name)

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

        # ## GET THE COSTS FOR THIS SEGMENT ## #
        self._logger.debug("Getting costs from: %s" % costs_path)

        int_costs, cost_name = costs_utils.get_costs(
            costs_path,
            segment_params,
            iz_infill=intrazonal_cost_infill,
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
            productions=seg_productions,
            attractions=seg_attractions,
            base_matrix=seed_matrix,
            costs=costs,
            int_target_tld=internal_tld,
            ext_target_tld=external_tld,
            log_path=log_path,
            convergence_target=convergence_target,
            furness_tol=furness_tol,
            furness_max_iters=furness_max_iters,
        )

        # ## WRITE REPORTS ON HOW THE EXTERNAL MODEL DID ## #
        # Internal TLD Report
        fname = running_segmentation.generate_file_name(
            trip_origin=trip_origin,
            year=str(self.year),
            file_desc='internal_tld_report',
            segment_params=segment_params,
            csv=True,
        )
        path = os.path.join(self.report_paths.tld_report_dir, fname)
        int_report.to_csv(path, index=False)

        # External TLD Report
        fname = running_segmentation.generate_file_name(
            trip_origin=trip_origin,
            year=str(self.year),
            file_desc='external_tld_report',
            segment_params=segment_params,
            csv=True,
        )
        path = os.path.join(self.report_paths.tld_report_dir, fname)
        ext_report.to_csv(path, index=False)

        # Build an IE report
        ie_report = pd_utils.internal_external_report(
            df=gb_pa,
            internal_zones=self.zoning_system.internal_zones,
            external_zones=self.zoning_system.external_zones,
        )

        # Build the output path
        fname = running_segmentation.generate_file_name(
            trip_origin=trip_origin,
            year=str(self.year),
            file_desc='ie_report',
            segment_params=segment_params,
            csv=True,
        )
        path = os.path.join(self.report_paths.ie_report_dir, fname)
        ie_report.to_csv(path)

        # ## WRITE EXTERNAL DEMAND TO DISK ## #
        # Write out full demand
        # Generate path and write out
        fname = running_segmentation.generate_file_name(
            trip_origin=trip_origin,
            year=str(self.year),
            file_desc='synthetic_pa',
            segment_params=segment_params,
            compressed=True,
        )
        path = os.path.join(self.export_paths.full_distribution_dir, fname)
        nd.write_df(gb_pa, path)

        # Get just the external
        external_demand = pd_utils.get_external_values(
            df=gb_pa,
            zones=self.zoning_system.external_zones,
        )

        # Generate path and write out
        fname = running_segmentation.generate_file_name(
            trip_origin=trip_origin,
            year=str(self.year),
            file_desc='synthetic_pa',
            segment_params=segment_params,
            suffix=self._external_only_suffix,
            compressed=True,
        )
        path = os.path.join(self.export_paths.external_distribution_dir, fname)
        nd.write_df(external_demand, path)

        # ## BUILD THE INTERNAL ONLY VECTORS ## #
        # Get just the internal
        internal_demand = pd_utils.get_internal_values(
            df=gb_pa,
            zones=self.zoning_system.internal_zones,
        )

        # Production vector
        prods_eff_df = segment_params.copy()
        prods_eff_df['df'] = pd.DataFrame(
            data=internal_demand.values.sum(axis=1),
            index=internal_demand.index,
            columns=[pa_val_col],
        ).reset_index()

        # Attraction Vector
        attrs_eff_df = segment_params.copy()
        attrs_eff_df['df'] = pd.DataFrame(
            data=internal_demand.values.sum(axis=0),
            index=internal_demand.index,
            columns=[pa_val_col],
        ).reset_index()

        return prods_eff_df, attrs_eff_df

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
            _, int_tlb_con, _ = tld_utils.get_trip_length_by_band(int_target_tld, internal_dist, internal_pa)
            int_bs_con = math_utils.curve_convergence(int_tlb_con['tbs'], int_tlb_con['bs'])
            int_mse = math_utils.vector_mean_squared_error(
                vector1=int_tlb_con['tbs'].values,
                vector2=int_tlb_con['bs'].values,
            )

            # External vals
            _, ext_tlb_con, _ = tld_utils.get_trip_length_by_band(ext_target_tld, external_dist, external_pa)
            ext_bs_con = math_utils.curve_convergence(ext_tlb_con['tbs'], ext_tlb_con['bs'])
            ext_mse = math_utils.vector_mean_squared_error(
                vector1=ext_tlb_con['tbs'].values,
                vector2=ext_tlb_con['bs'].values,
            )

            iter_end_time = timing.current_milli_time()
            time_taken = iter_end_time - iter_start_time

            pa_diff = math_utils.get_pa_diff(
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
            file_ops.safe_dataframe_to_csv(
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
                       ):
    """
    Adjust band shares of rows or columns

    pa_mat pa:
        Square matrix
    band_totals:
        list of dictionaries of trip length bands
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

            # We can't adjust if there are no trips in this band
            if ach_band_trips <= 0:
                adj_mat = band_trips

            # Adjust the matrix towards target
            else:
                adjustment = target_band_trips / ach_band_trips
                adj_mat = band_trips * adjustment

            # Add into the return matrix
            out += adj_mat

        out_ph.append(out)

    # Stick internal and external back together
    return pd.concat(out_ph)
