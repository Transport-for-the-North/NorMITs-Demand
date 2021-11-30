# -*- coding: utf-8 -*-
"""
Created on: 06/10/2021
Updated on:

Original author: Ben Taylor
Last update made by: Ben Taylor
Other updates made by: Chris Storey

File purpose:

"""
# Built-Ins
import os

from typing import Any
from typing import Dict
from typing import Optional

# Third Party
import numpy as np
import pandas as pd

# Local Imports
import normits_demand as nd
from normits_demand import constants

from normits_demand import cost

from normits_demand.cost import utils as cost_utils2

from normits_demand.utils import file_ops
from normits_demand.utils import costs as cost_utils
from normits_demand.utils import pandas_utils as pd_utils
from normits_demand.utils import trip_length_distributions as tld_utils

from normits_demand.distribution import gravity_model

from normits_demand.validation import checks
from normits_demand.concurrency import multiprocessing

from normits_demand.pathing.travel_market_synthesiser import GravityModelExportPaths


class GravityModel(GravityModelExportPaths):
    _log_fname = "Gravity_Model_log.log"

    _base_zone_col = "%s_zone_id"
    _pa_val_col = 'trips'

    _internal_only_suffix = 'int'

    def __init__(self,
                 year: int,
                 running_mode: nd.Mode,
                 zoning_system: nd.core.ZoningSystem,
                 export_home: nd.PathLike,
                 zone_col: str = None,
                 process_count: Optional[int] = constants.PROCESS_COUNT,
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
            instantiate_msg="Initialised new Gravity Model Logger",
        )

    def run(self,
            trip_origin: str,
            running_segmentation: nd.core.segments.SegmentationLevel,
            productions: pd.DataFrame,
            attractions: pd.DataFrame,
            init_params: pd.DataFrame,
            target_tld_dir: pd.DataFrame,
            cost_dir: nd.PathLike,
            cost_function: cost.CostFunction,
            intrazonal_cost_infill: Optional[float] = 0.5,
            pa_val_col: Optional[str] = 'val',
            convergence_target: float = 0.95,
            fitting_loops: int = 100,
            furness_max_iters: int = 5000,
            furness_tol: float = 1.0,
            init_param_cols: str = None,
            ):
        # Validate the trip origin
        trip_origin = checks.validate_trip_origin(trip_origin)

        # If no cols given, get from the cost function
        if init_param_cols is None:
            init_param_cols = cost_function.parameter_names

        # Replace the overall log if it exists
        if trip_origin == 'hb':
            overall_log_path = self.report_paths.hb_overall_log
        elif trip_origin == 'nhb':
            overall_log_path = self.report_paths.nhb_overall_log
        else:
            raise ValueError("Don't know what trip_origin %s is" % trip_origin)

        if os.path.isfile(overall_log_path):
            os.remove(overall_log_path)

        # ## MULTIPROCESS ACROSS SEGMENTS ## #
        unchanging_kwargs = {
            'trip_origin': trip_origin,
            'running_segmentation': running_segmentation,
            'target_tld_dir': target_tld_dir,
            'cost_dir': cost_dir,
            'cost_function': cost_function,
            'overall_log_path': overall_log_path,
            'intrazonal_cost_infill': intrazonal_cost_infill,
            'furness_max_iters': furness_max_iters,
            'fitting_loops': fitting_loops,
            'convergence_target': convergence_target,
            'furness_tol': furness_tol,
        }

        pbar_kwargs = {
            'desc': 'Gravity model',
            'unit': 'segment',
        }

        # Build a list of kwargs
        kwarg_list = list()
        for segment_params in running_segmentation:
            # ## GET P/A VECTORS FOR THIS SEGMENT ## #
            # Figure out which columns we need
            segments = list(segment_params.keys())
            rename_cols = {pa_val_col: self._pa_val_col}
            needed_cols = segments + [self._pa_val_col]

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

            # ## GET INIT PARAMS FOP THIS SEGMENT ## #
            seg_init_params = pd_utils.filter_df(init_params, segment_params)

            if len(seg_init_params) > 1:
                seg_name = running_segmentation.generate_file_name(segment_params)
                raise ValueError(
                    "%s rows found in init_params for segment %s. "
                    "Expecting only 1 row."
                    % (len(seg_init_params), seg_name)
                )

            # Make sure the columns we need do exist
            seg_init_params = pd_utils.reindex_cols(
                df=seg_init_params,
                columns=init_param_cols,
                dataframe_name='init_params',
            )

            init_cost_params = {x: seg_init_params[x].squeeze() for x in init_param_cols}

            # Build the kwargs
            kwargs = unchanging_kwargs.copy()
            kwargs.update({
                'segment_params': segment_params,
                'seg_productions': seg_productions,
                'seg_attractions': seg_attractions,
                'init_cost_params': init_cost_params
            })
            kwarg_list.append(kwargs)

        # Multiprocess
        multiprocessing.multiprocess(
            fn=self._run_internal,
            kwargs=kwarg_list,
            pbar_kwargs=pbar_kwargs,
            # process_count=0,
            process_count=self.process_count,
        )

    def _run_internal(self,
                      segment_params: Dict[str, Any],
                      trip_origin: str,
                      running_segmentation: nd.core.segments.SegmentationLevel,
                      seg_productions: pd.DataFrame,
                      seg_attractions: pd.DataFrame,
                      init_cost_params: Dict[str, float],
                      target_tld_dir: pd.DataFrame,
                      cost_dir: nd.PathLike,
                      cost_function: cost.CostFunction,
                      overall_log_path: nd.PathLike,
                      intrazonal_cost_infill: Optional[float] = 0.5,
                      convergence_target: float = 0.95,
                      fitting_loops: int = 100,
                      furness_max_iters: int = 5000,
                      furness_tol: float = 1.0,
                      ):
        seg_name = running_segmentation.generate_file_name(segment_params)
        self._logger.info("Running for %s" % seg_name)

        # ## READ IN TLD FOR THIS SEGMENT ## #
        target_tld = tld_utils.get_trip_length_distributions(
            import_dir=target_tld_dir,
            segment_params=segment_params,
            trip_origin=trip_origin,
        )

        # Convert to expected format
        rename = {'lower': 'min', 'upper': 'max'}
        target_tld = target_tld.rename(columns=rename)
        target_tld['min'] *= constants.MILES_TO_KM
        target_tld['max'] *= constants.MILES_TO_KM

        # ## GET THE COSTS FOR THIS SEGMENT ## #
        self._logger.debug("Getting costs from: %s" % cost_dir)

        # Generate the fname
        fname = running_segmentation.generate_file_name(
            trip_origin=trip_origin,
            file_desc="%s_cost" % self.zoning_system.name,
            segment_params=segment_params,
            csv=True,
        )
        path = os.path.join(cost_dir, fname)

        # Read in the costs and infill
        cost = nd.read_df(path, find_similar=True, index_col=0).values
        if intrazonal_cost_infill is not None:
            cost = cost_utils.iz_infill_costs(
                cost,
                iz_infill=intrazonal_cost_infill,
            )

        # ## SET UP LOG AND RUN ## #
        # Logging set up
        log_fname = running_segmentation.generate_file_name(
            trip_origin=trip_origin,
            file_desc='gravity_log',
            segment_params=segment_params,
            csv=True,
        )
        log_path = os.path.join(self.report_paths.model_log_dir, log_fname)

        # Need to convert into numpy vectors to work with old code
        seg_productions = seg_productions[self._pa_val_col].values
        seg_attractions = seg_attractions[self._pa_val_col].values

        # Replace the log if it already exists
        if os.path.isfile(log_path):
            os.remove(log_path)

        # ## CALIBRATE THE GRAVITY MODEL ## #
        calib = gravity_model.GravityModelCalibrator(
            row_targets=seg_productions,
            col_targets=seg_attractions,
            cost_function=cost_function,
            costs=cost,
            target_cost_distribution=target_tld,
            target_convergence=convergence_target,
            furness_max_iters=furness_max_iters,
            furness_tol=furness_tol,
            running_log_path=log_path,
        )

        optimal_cost_params = calib.calibrate(
            init_params=init_cost_params,
            max_iters=fitting_loops,
            ftol=1e-5,
            verbose=2,
        )

        # ## WRITE OUT GRAVITY MODEL OUTPUTS ## #
        # TODO(BT): Make this a standard function for external model too
        # Create tld report
        rename = {
            'min': 'min (km)',
            'max': 'max (km)',
            'ave_km': 'target_ave_length (km)',
            'band_share': 'target_band_share',
        }
        tld_report = pd_utils.reindex_cols(target_tld, rename.keys())
        tld_report = tld_report.rename(columns=rename)

        # Add in achieved values
        tld_report['ach_band_share'] = calib.achieved_band_share
        tld_report['convergence'] = calib.achieved_convergence
        tld_report['ach_band_trips'] = tld_report['ach_band_share'].copy()
        tld_report['ach_band_trips'] *= calib.achieved_distribution.sum()

        tld_report['ach_ave_length (km)'] = tld_utils.calculate_average_trip_lengths(
            min_bounds=tld_report['min (km)'].values,
            max_bounds=tld_report['max (km)'].values,
            trip_lengths=cost,
            trips=calib.achieved_distribution,
        )

        tld_report['cell count'] = cost_utils2.cells_in_bounds(
            min_bounds=tld_report['min (km)'].values,
            max_bounds=tld_report['max (km)'].values,
            cost=cost,
        )
        tld_report['cell proportions'] = tld_report['cell count'].copy()
        tld_report['cell proportions'] /= tld_report['cell proportions'].values.sum()

        # Order columns for output
        col_order = [
            'min (km)',
            'max (km)',
            'target_ave_length (km)',
            'ach_ave_length (km)',
            'target_band_share',
            'ach_band_share',
            'ach_band_trips',
            'cell count',
            'cell proportions',
            'convergence',
        ]
        tld_report = pd_utils.reindex_cols(tld_report, col_order)

        # Write out tld report
        fname = running_segmentation.generate_file_name(
            trip_origin=trip_origin,
            year=str(self.year),
            file_desc='tld_report',
            segment_params=segment_params,
            csv=True,
        )
        path = os.path.join(self.report_paths.tld_report_dir, fname)
        tld_report.to_csv(path, index=False)

        # ## WRITE DISTRIBUTED DEMAND ## #
        # Put the demand into a df
        demand_df = pd.DataFrame(
            index=self.zoning_system.unique_zones,
            columns=self.zoning_system.unique_zones,
            data=calib.achieved_distribution.astype(np.float32),
        )

        # Generate path and write out
        fname = running_segmentation.generate_file_name(
            trip_origin=trip_origin,
            year=str(self.year),
            file_desc='synthetic_pa',
            segment_params=segment_params,
            suffix=self._internal_only_suffix,
            compressed=True,
        )
        path = os.path.join(self.export_paths.distribution_dir, fname)
        nd.write_df(demand_df, path)

        # ## ADD TO THE OVERALL LOG ## #
        # Rename keys for log
        init_cost_params = {"init_%s" % k: v for k, v in init_cost_params.items()}
        optimal_cost_params = {"final_%s" % k: v for k, v in optimal_cost_params.items()}

        # Generate the log
        log_dict = segment_params.copy()
        log_dict.update(init_cost_params)
        log_dict.update({'init_bs_con': calib.initial_convergence})
        log_dict.update(optimal_cost_params)
        log_dict.update({'final_bs_con': calib.achieved_convergence})

        # Append this iteration to log file
        file_ops.safe_dataframe_to_csv(
            pd.DataFrame(log_dict, index=[0]),
            overall_log_path,
            mode='a',
            header=(not os.path.exists(overall_log_path)),
            index=False,
        )
