# -*- coding: utf-8 -*-
"""
Created on: 08/12/2021
Updated on:

Original author: Ben Taylor
Last update made by:
Other updates made by:

File purpose:

"""
# Built-Ins
import os
import abc
import enum

from typing import Any
from typing import List
from typing import Dict
from typing import Tuple
from typing import Optional

# Third Party
import numpy as np
import pandas as pd

# Local Imports
import normits_demand as nd
from normits_demand import constants

from normits_demand import cost

from normits_demand.cost import utils as cost_utils

from normits_demand.utils import file_ops
from normits_demand.utils import pandas_utils as pd_utils
from normits_demand.utils import trip_length_distributions as tld_utils

from normits_demand.distribution import gravity_model

from normits_demand.validation import checks
from normits_demand.concurrency import multiprocessing

from normits_demand.pathing.travel_market_synthesiser import GravityModelExportPaths


class AbstractDistributor(abc.ABC):
    # Default class constants that can be overwritten
    _default_name = 'Distributor'

    # Internal variables for consistent naming
    _pa_val_col = 'trips'

    def __init__(self,
                 year: int,
                 running_mode: nd.Mode,
                 zoning_system: nd.core.ZoningSystem,
                 export_home: nd.PathLike,
                 process_count: Optional[int] = constants.PROCESS_COUNT,
                 zone_col: str = None,
                 name: str = None,
                 ):
        # Validate inputs
        if not isinstance(zoning_system, nd.core.zoning.ZoningSystem):
            raise ValueError(
                "Expected and instance of a normits_demand ZoningSystem. "
                "Got a %s instance instead."
                % type(zoning_system)
            )

        # Set default values where not set
        name = self._default_name if name is None else name
        zone_col = zoning_system.col_name if zone_col is None else zone_col

        # Assign attributes
        self.name = name
        self.year = year
        self.running_mode = running_mode
        self.zoning_system = zoning_system
        self.zone_col = zone_col
        self.export_home = export_home
        self.process_count = process_count

    def _filter_productions_attractions(self,
                                        segment_params: Dict[str, Any],
                                        productions: pd.DataFrame,
                                        attractions: pd.DataFrame,
                                        pa_val_col: str,
                                        ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Extracts and returns the productions and attractions for segment_params"""
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

        return seg_productions, seg_attractions

    @abc.abstractmethod
    def distribute_segment(self,
                           productions: pd.DataFrame,
                           attractions: pd.DataFrame,
                           running_segmentation: nd.SegmentationLevel,
                           overall_log_path: nd.PathLike,
                           **kwargs,
                           ):
        pass

    def distribute(self,
                   productions: pd.DataFrame,
                   attractions: pd.DataFrame,
                   running_segmentation: nd.SegmentationLevel,
                   overall_log_path: nd.PathLike,
                   pa_val_col: Optional[str] = 'val',
                   by_segment_kwargs: List[Dict[str, Any]] = None,
                   **kwargs,
                   ):
        # Set defaults
        by_segment_kwargs = dict() if by_segment_kwargs is None else by_segment_kwargs

        # Make a new log file if one already exists
        if os.path.isfile(overall_log_path):
            os.remove(overall_log_path)

        # ## MULTIPROCESS ACROSS SEGMENTS ## #
        unchanging_kwargs = kwargs.copy()
        unchanging_kwargs.update({
            'running_segmentation': running_segmentation,
            'overall_log_path': overall_log_path,
        })

        pbar_kwargs = {
            'desc': self.name,
            'unit': 'segment',
        }

        # Build a list of kwargs - one for each segment
        kwarg_list = list()
        for segment_params in running_segmentation:
            # Get productions, attractions
            seg_productions, seg_attractions = self._filter_productions_attractions(
                segment_params=segment_params,
                productions=productions,
                attractions=attractions,
                pa_val_col=pa_val_col,
            )

            # Build the kwargs for this segment
            segment_kwargs = unchanging_kwargs.copy()
            segment_kwargs.update({
                'segment_params': segment_params,
                'productions': seg_productions,
                'attractions': seg_attractions,
            })

            # Get any other by_segment kwargs passed in
            segment_name = running_segmentation.get_segment_name(segment_params)
            segment_kwargs.update(by_segment_kwargs.get(segment_name, dict()))

            kwarg_list.append(segment_kwargs)

        # Multiprocess
        multiprocessing.multiprocess(
            fn=self.distribute_segment,
            kwargs=kwarg_list,
            pbar_kwargs=pbar_kwargs,
            process_count=0,
            # process_count=self.process_count,
        )


@enum.unique
class DistributionMethod(enum.Enum):
    GRAVITY = 'gravity'
    FURNESS3D = 'furness_3d'

    def get_distributor(self, **kwargs) -> AbstractDistributor:

        if self == DistributionMethod.GRAVITY:
            function = GravityDistributor

        elif self == DistributionMethod.FURNESS3D:
            raise NotImplementedError()

        else:
            raise nd.NormitsDemandError(
                "No definition exists for %s built in cost function"
                % self
            )

        return function(**kwargs)


class GravityDistributor(GravityModelExportPaths, AbstractDistributor):
    _log_fname = "Gravity_Model_log.log"

    _base_zone_col = "%s_zone_id"
    _pa_val_col = 'trips'

    _internal_only_suffix = 'int'

    def __init__(self,
                 year: int,
                 running_mode: nd.Mode,
                 zoning_system: nd.ZoningSystem,
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

        # Build the distributor
        AbstractDistributor.__init__(
            self,
            year=year,
            running_mode=running_mode,
            zoning_system=zoning_system,
            export_home=export_home,
            process_count=process_count,

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
        GravityModelExportPaths.__init__(
            self,
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

    def distribute_segment(self,
                           productions: pd.DataFrame,
                           attractions: pd.DataFrame,
                           running_segmentation: nd.SegmentationLevel,
                           overall_log_path: nd.PathLike,
                           **kwargs,
                           ):
        print(productions)
        print(attractions)
        print(running_segmentation)
        print(overall_log_path)
        print(kwargs)
        return

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
                      furness_tol: float = 0.1,
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
        cost_matrix = nd.read_df(path, find_similar=True, index_col=0).values
        if intrazonal_cost_infill is not None:
            cost_matrix = cost_utils.iz_infill_costs(
                cost_matrix,
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
            costs=cost_matrix,
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
            trip_lengths=cost_matrix,
            trips=calib.achieved_distribution,
        )

        tld_report['cell count'] = cost_utils.cells_in_bounds(
            min_bounds=tld_report['min (km)'].values,
            max_bounds=tld_report['max (km)'].values,
            cost=cost_matrix,
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
