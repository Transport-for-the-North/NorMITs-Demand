# -*- coding: utf-8 -*-
"""
Created on: 08/12/2021
Updated on:

Original author: Ben Taylor
Last update made by:
Other updates made by:
Originally based on code written by Chris Storey.

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

from normits_demand.cost import utils as cost_utils

from normits_demand.utils import timing
from normits_demand.utils import file_ops
from normits_demand.utils import math_utils
from normits_demand.utils import general as du
from normits_demand.utils import pandas_utils as pd_utils

from normits_demand.distribution import gravity_model
from normits_demand.distribution import furness

from normits_demand.validation import checks
from normits_demand.concurrency import multiprocessing

from normits_demand.pathing.distribution_model import DistributorExportPaths


class AbstractDistributor(abc.ABC, DistributorExportPaths):
    # Default class constants that can be overwritten
    _default_name = 'Distributor'

    # Internal variables for consistent naming
    _pa_val_col = 'trips'
    calibration_ignore_val = -1

    def __init__(self,
                 year: int,
                 running_mode: nd.Mode,
                 trip_origin: str,
                 zoning_system: nd.core.ZoningSystem,
                 running_zones: List[Any],
                 export_home: nd.PathLike,
                 process_count: Optional[int] = constants.PROCESS_COUNT,
                 zone_col: str = None,
                 name: str = None,
                 cost_name: str = None,
                 cost_units: str = None,
                 ):
        # Validate inputs
        if not isinstance(zoning_system, nd.core.zoning.ZoningSystem):
            raise ValueError(
                "Expected and instance of a normits_demand ZoningSystem. "
                "Got a %s instance instead."
                % type(zoning_system)
            )

        trip_origin = checks.validate_trip_origin(trip_origin)

        # Set default values where not set
        name = self._default_name if name is None else name
        zone_col = zoning_system.col_name if zone_col is None else zone_col

        # Assign attributes
        self.name = name
        self.year = year
        self.running_mode = running_mode
        self.trip_origin = trip_origin
        self.zoning_system = zoning_system
        self.running_zones = running_zones
        self.zone_col = zone_col
        self.cost_name = cost_name if cost_name is not None else 'Cost'
        self.cost_units = cost_units
        self.export_home = export_home
        self.process_count = process_count

        # Generate the graph label names
        self.y_label = "Band Share (%)"
        if cost_units is not None:
            self.x_label = f"{self.cost_name} ({self.cost_units})"
        else:
            self.x_label = self.cost_name

        # Build the output paths
        DistributorExportPaths.__init__(
            self,
            year=year,
            trip_origin=trip_origin,
            running_mode=running_mode,
            export_home=export_home,
        )

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

    @staticmethod
    def _check_segment_keys(running_segmentation: nd.SegmentationLevel,
                            check_dict: Dict[str, Any],
                            name: str = 'check_dict',
                            ) -> None:
        """
        Checks that check_dict contains all segment_names of
        running segmentation as keys

        Parameters
        ----------
        check_dict:
            The dictionary to check the keys of

        Raises
        ------
        ValueError:
            If any of the segment names don't exist in the keys
        """
        # Generate set of segment names
        segment_names = set()
        for segment_params in running_segmentation:
            segment_names.add(running_segmentation.get_segment_name(segment_params))

        # Check all exist
        dict_keys = set(check_dict.keys())
        missing = segment_names - dict_keys
        if len(missing) > 0:
            raise ValueError(
                "Not all segment names exist in %s. Missing segment names: %s"
                % (name, missing)
            )

    @abc.abstractmethod
    def distribute_segment(self,
                           segment_params: Dict[str, Any],
                           productions: pd.DataFrame,
                           attractions: pd.DataFrame,
                           cost_matrix: pd.DataFrame,
                           calibration_matrix: pd.DataFrame,
                           target_cost_distributions: Dict[Any, pd.DataFrame],
                           calibration_naming: Dict[Any, Any],
                           running_segmentation: nd.SegmentationLevel,
                           **kwargs,
                           ):
        pass

    def distribute(self,
                   productions: pd.DataFrame,
                   attractions: pd.DataFrame,
                   running_segmentation: nd.SegmentationLevel,
                   cost_matrices: Dict[str, pd.DataFrame],
                   calibration_matrix: pd.DataFrame,
                   target_cost_distributions: Dict[Any, Dict[str, pd.DataFrame]],
                   calibration_naming: Dict[Any, Any],
                   pa_val_col: str = 'val',
                   by_segment_kwargs: Dict[str, Dict[str, Any]] = None,
                   **kwargs,
                   ):
        # Validate inputs
        self._check_segment_keys(
            running_segmentation,
            cost_matrices,
            'cost_matrices',
        )
        for item in target_cost_distributions.values():
            self._check_segment_keys(
                running_segmentation,
                item,
                'target_cost_distributions',
            )

        # Validate calibration keys
        calib_keys = np.unique(calibration_matrix)
        calib_keys = du.list_safe_remove(list(calib_keys), [self.calibration_ignore_val])

        missing = set(calib_keys) - set(target_cost_distributions.keys())
        if len(missing) > 0:
            raise ValueError(
                "Target cost distributions can not be found for all key "
                "values found in the calibration matrix. No targets given for "
                "the following %s keys: %s"
                % (len(missing), missing)
            )

        # Add in any naming keys that don't exist
        for key in calib_keys:
            if key not in calibration_naming:
                calibration_naming[key] = key

        # Set defaults
        by_segment_kwargs = dict() if by_segment_kwargs is None else by_segment_kwargs

        # ## MULTIPROCESS ACROSS SEGMENTS ## #
        unchanging_kwargs = kwargs.copy()
        unchanging_kwargs.update({
            'running_segmentation': running_segmentation,
            'calibration_matrix': calibration_matrix,
            'calibration_naming': calibration_naming,
        })

        pbar_kwargs = {
            'desc': self.name,
            'unit': 'segment',
        }

        # Build a list of kwargs - one for each segment
        kwarg_list = list()
        for segment_params in running_segmentation:
            segment_name = running_segmentation.get_segment_name(segment_params)

            # Get productions, attractions
            seg_productions, seg_attractions = self._filter_productions_attractions(
                segment_params=segment_params,
                productions=productions,
                attractions=attractions,
                pa_val_col=pa_val_col,
            )

            # Get the cost distributions for this segment
            segment_target_costs = dict.fromkeys(calib_keys)
            for key in calib_keys:
                segment_target_costs[key] = target_cost_distributions[key][segment_name]

            # Build the kwargs for this segment
            segment_kwargs = unchanging_kwargs.copy()
            segment_kwargs.update({
                'segment_params': segment_params,
                'productions': seg_productions,
                'attractions': seg_attractions,
                'cost_matrix': cost_matrices[segment_name],
                'target_cost_distributions': segment_target_costs,
            })

            # Get any other by_segment kwargs passed in
            segment_kwargs.update(by_segment_kwargs.get(segment_name, dict()))

            kwarg_list.append(segment_kwargs)

        # Multiprocess
        multiprocessing.multiprocess(
            fn=self.distribute_segment,
            kwargs=kwarg_list,
            pbar_kwargs=pbar_kwargs,
            # process_count=0,
            process_count=self.process_count,
        )

    def generate_cost_distribution_report(
        self,
        min_bounds: np.ndarray,
        max_bounds: np.ndarray,
        target_ave_cost: np.ndarray,
        target_band_share: np.ndarray,
        achieved_band_share: np.ndarray,
        achieved_convergence: float,
        achieved_distribution: np.ndarray,
        cost_matrix: np.ndarray,
        report_cols: cost_utils.DistributionReportCols = None,
    ) -> pd.DataFrame:
        """Generates a report of the target and achieved cost distribution

        Parameters
        ----------
        min_bounds:
            The minimum bounds for each cost band. Corresponds to max_bounds.

        max_bounds:
            The maximum bounds for each cost band. Corresponds to min_bounds.

        target_ave_cost:
            The average cost achieved by each of the bands within min_bounds
            and max_bounds at the same index.

        target_band_share:
            The target band shares within min_bounds and max_bounds at the
            same index.

        achieved_band_share:
            A numpy array of the achieved band share values. Must
            correspond to target_band_share

        achieved_convergence:
            The value describing the convergence that was achieved between
            achieved_band_share and target_band_share

        achieved_distribution:
            The matrix of distributed values that produces achieved_band_share

        cost_matrix:
            The matrix of costs from zone to zone.

        report_cols:
            A DistributionReportCols object defining how to name the columns
            in the output report. If left as None, a new DistributionReportCols
            is made, and self.cost_units is given.

        Returns
        -------
        report:
            A report with column names as defined in report_cols.
        """
        # Init
        if report_cols is None:
            report_cols = cost_utils.DistributionReportCols(cost_units=self.cost_units)

        # Calculate remaining achieved values
        achieved_band_count = achieved_band_share * achieved_distribution.sum()
        achieved_ave_cost = cost_utils.calculate_average_cost_in_bounds(
            min_bounds=min_bounds,
            max_bounds=max_bounds,
            cost_matrix=cost_matrix,
            trips=achieved_distribution,
        )

        # Calculate cost distributions
        cell_count = cost_utils.cells_in_bounds(
            min_bounds=min_bounds,
            max_bounds=max_bounds,
            cost=cost_matrix,
        )
        cell_proportions = cell_count / cell_count.sum()

        # Create tld report
        report = pd.DataFrame({
            report_cols.min: min_bounds,
            report_cols.max: max_bounds,
            report_cols.mid: cost_utils.get_band_mid_points(min_bounds, max_bounds),
            report_cols.target_ave_cost: target_ave_cost,
            report_cols.achieved_ave_cost: achieved_ave_cost,
            report_cols.target_band_share: target_band_share,
            report_cols.achieved_band_share: achieved_band_share,
            report_cols.achieved_band_count: achieved_band_count,
            report_cols.cell_count: cell_count,
            report_cols.cell_proportion: cell_proportions,
            report_cols.convergence: achieved_convergence,
        })

        # Order columns for output
        return pd_utils.reindex_cols(report, report_cols.col_order())

    @staticmethod
    def generate_cost_distribution_graph(
        min_bounds: np.ndarray,
        max_bounds: np.ndarray,
        target_band_share: np.ndarray,
        achieved_band_share: np.ndarray,
        achieved_convergence: float,
        achieved_cost_params: Dict[str, float],
        plot_title: str,
        graph_path: nd.PathLike,
        x_label: str = 'Cost',
        y_label: str = 'Band Share (%)',
        **graph_kwargs,
    ) -> None:
        """Generates and writes out a graph of cost distributions

        Compares target and achieved cost distribution, generates a graph,
        and then writes the generated graph to disk

        Parameters
        ----------
        min_bounds:
            The minimum bounds for each cost band. Corresponds to max_bounds.

        max_bounds:
            The maximum bounds for each cost band. Corresponds to min_bounds.

        target_band_share:
            The target band shares within min_bounds and max_bounds at the
            same index.

        achieved_band_share:
            The achieved band shares within min_bounds and max_bounds at the
            same index.

        achieved_convergence:
            The value describing the convergence that was achieved between
            achieved_band_share and target_band_share

        achieved_cost_params:
            The cost parameters that were used to generate achieved_band_share.

        plot_title:
            The title to give to the generated plot

        x_label:
            The name to label the generated plot x axis with

        y_label:
            The name to label the generated plot y axis with

        graph_path:
            The path to write the generated plot out to. This will be passed
            to matplotlib.pyplot.savefig

        graph_kwargs:
            An further keyword arguments to pass to matplotlib.pyplot.savefig

        See Also
        --------
        `matplotlib.pyplot.savefig`
        """
        # Init
        if 'dpi' not in graph_kwargs:
            graph_kwargs['dpi'] = 300

        # Get the mid points of each band - more similar to calibration method
        mid_points = cost_utils.get_band_mid_points(
            min_bounds=min_bounds,
            max_bounds=max_bounds,
        )

        # Plot the graph and write out
        cost_utils.plot_cost_distribution(
            target_x=mid_points,
            target_y=target_band_share,
            achieved_x=mid_points,
            achieved_y=achieved_band_share,
            convergence=achieved_convergence,
            cost_params=achieved_cost_params,
            plot_title=plot_title,
            path=graph_path,
            x_label=x_label,
            y_label=y_label,
            **graph_kwargs,
        )


@enum.unique
class DistributionMethod(enum.Enum):
    GRAVITY = 'gravity'
    FURNESS3D = 'furness_3d'

    def get_distributor(self, **kwargs) -> AbstractDistributor:

        if self == DistributionMethod.GRAVITY:
            function = GravityDistributor
        elif self == DistributionMethod.FURNESS3D:
            function = Furness3dDistributor
        else:
            raise nd.NormitsDemandError(
                "No definition exists for %s distribution method"
                % self
            )

        return function(**kwargs)


class GravityDistributor(AbstractDistributor):
    _log_fname = "Gravity_Model_log.log"

    _base_zone_col = "%s_zone_id"
    _pa_val_col = 'trips'

    _internal_only_suffix = 'int'

    def __init__(self,
                 year: int,
                 running_mode: nd.Mode,
                 trip_origin: str,
                 zoning_system: nd.ZoningSystem,
                 running_zones: List[Any],
                 export_home: nd.PathLike,
                 zone_col: str = None,
                 cost_units: str = None,
                 cost_name: str = None,
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
        super().__init__(
            year=year,
            running_mode=running_mode,
            trip_origin=trip_origin,
            zoning_system=zoning_system,
            running_zones=running_zones,
            zone_col=zone_col,
            export_home=export_home,
            process_count=process_count,
            cost_units=cost_units,
            cost_name=cost_name,
        )

        # Create a logger
        logger_name = "%s.%s" % (nd.get_package_logger_name(), self.__class__.__name__)
        log_file_path = os.path.join(self.export_home, self._log_fname)
        self._logger = nd.get_logger(
            logger_name=logger_name,
            log_file_path=log_file_path,
            instantiate_msg="Initialised new Gravity Model Logger",
        )

    def _write_out_reports(
        self,
        segment_params: Dict[str, Any],
        running_segmentation: nd.SegmentationLevel,
        init_cost_params: Dict[str, Any],
        optimal_cost_params: Dict[str, Any],
        min_bounds: np.ndarray,
        max_bounds: np.ndarray,
        target_ave_cost: np.ndarray,
        target_band_share: np.ndarray,
        initial_convergence: float,
        achieved_convergence: float,
        achieved_band_share: np.ndarray,
        achieved_distribution: np.ndarray,
        cost_matrix: np.ndarray,
        overall_log_path: os.PathLike,
        subdir_name: str = None,
    ) -> None:
        """Generates and writes out standard gravity model reports

        Generated reports are written out to the following places:
        Reports: self.report_paths.tld_report_dir
        Matrices: self.export_paths.matrix_dir
        Overall log: self.report_paths.overall_log

        If subdir_name is defined, this name is append to each of the paths
        to create a unique place to store the generated reports.

        Parameters
        ----------
        segment_params:
            Passed into `running_segmentation.generate_file_name()`
            to generate filenames.

        running_segmentation:
            The currently running segmentation. Used to generate filenames
            alongside segment_params.

        init_cost_params:
            A dictionary of kwargs defining the initial cost params used
            to generate initial_convergence.

        optimal_cost_params:
            A dictionary of kwargs defining the optimal cost params used
            to generate achieved_band_share, achieved_convergence, and
            achieved_distribution.

        min_bounds:
            The minimum bounds for each cost band. Corresponds to max_bounds.

        max_bounds:
            The maximum bounds for each cost band. Corresponds to min_bounds.

        target_ave_cost:
            The average cost achieved by each of the bands within min_bounds
            and max_bounds at the same index.

        target_band_share:
            The target band shares within min_bounds and max_bounds at the
            same index.

        initial_convergence
            The value describing the convergence that was achieved between
            target_band_share, and the band share created when running a
            gravity model with init_cost_params.

        achieved_convergence:
            The value describing the convergence that was achieved between
            achieved_band_share and target_band_share

        achieved_band_share:
             A numpy array of the achieved band share values. Must
            correspond to target_band_share

        achieved_distribution:
            The matrix of distributed values that produces achieved_band_share

        cost_matrix:
            The matrix of costs from zone to zone.

        subdir_name:
            The optional name of the subdirectory to create and place the
            generated reports into.

        Returns
        -------
        None.
        """
        # Init
        if subdir_name is not None:
            # Add in the subdir_name to paths
            report_dir = os.path.join(self.report_paths.tld_report_dir, subdir_name)
            matrix_dir = os.path.join(self.export_paths.matrix_dir, subdir_name)

            # Create the paths if they don't already exist
            file_ops.create_folder(report_dir)
            file_ops.create_folder(matrix_dir)
        else:
            report_dir = self.report_paths.tld_report_dir
            matrix_dir = self.export_paths.matrix_dir

        # ## DISTRIBUTION REPORTS ## #
        # Generate the base filename
        fname = running_segmentation.generate_file_name(
            trip_origin=self.trip_origin,
            year=str(self.year),
            file_desc='tld_report',
            segment_params=segment_params,
        )

        report = self.generate_cost_distribution_report(
            min_bounds=min_bounds,
            max_bounds=max_bounds,
            target_ave_cost=target_ave_cost,
            target_band_share=target_band_share,
            achieved_band_share=achieved_band_share,
            achieved_convergence=achieved_convergence,
            achieved_distribution=achieved_distribution,
            cost_matrix=cost_matrix,
        )

        # Write out report
        csv_fname = fname + '.csv'
        path = os.path.join(report_dir, csv_fname)
        report.to_csv(path, index=False)

        # Convert to a graph and write out
        graph_fname = fname + '.png'
        graph_path = os.path.join(report_dir, graph_fname)
        self.generate_cost_distribution_graph(
            min_bounds=min_bounds,
            max_bounds=max_bounds,
            target_band_share=target_band_share,
            achieved_band_share=achieved_band_share,
            achieved_convergence=achieved_convergence,
            achieved_cost_params=optimal_cost_params,
            plot_title=fname,
            graph_path=graph_path,
            x_label=self.x_label,
            y_label=self.y_label,
        )

        # ## WRITE DISTRIBUTED DEMAND ## #
        # Put the demand into a df
        demand_df = pd.DataFrame(
            index=self.running_zones,
            columns=self.running_zones,
            data=achieved_distribution.astype(np.float32),
        )

        demand_df = demand_df.reindex(
            index=self.zoning_system.unique_zones,
            columns=self.zoning_system.unique_zones,
            fill_value=0,
        )

        # Generate path and write out
        fname = running_segmentation.generate_file_name(
            trip_origin=self.trip_origin,
            year=str(self.year),
            file_desc='synthetic_pa',
            segment_params=segment_params,
            compressed=True,
        )
        path = os.path.join(matrix_dir, fname)
        nd.write_df(demand_df, path)

        # ## ADD TO THE OVERALL LOG ## #
        # Generate the log
        log_dict = segment_params.copy()
        log_dict.update({"init_%s" % k: v for k, v in init_cost_params.items()})
        log_dict.update({'init_bs_con': initial_convergence})
        log_dict.update({"final_%s" % k: v for k, v in optimal_cost_params.items()})
        log_dict.update({'final_bs_con': achieved_convergence})

        # Append this iteration to log file
        file_ops.safe_dataframe_to_csv(
            pd.DataFrame(log_dict, index=[0]),
            overall_log_path,
            mode='a',
            header=(not os.path.exists(overall_log_path)),
            index=False,
        )

    def _single_area_distribution(
        self,
        segment_params: Dict[str, Any],
        np_productions: np.ndarray,
        np_attractions: np.ndarray,
        np_cost: np.ndarray,
        target_cost_distribution: pd.DataFrame,
        running_segmentation: nd.SegmentationLevel,
        **kwargs,
    ):
        # ## SET UP SEGMENT LOG ## #
        # Logging set up
        log_fname = running_segmentation.generate_file_name(
            trip_origin=self.trip_origin,
            file_desc='gravity_log',
            segment_params=segment_params,
            csv=True,
        )
        log_path = os.path.join(self.report_paths.model_log_dir, log_fname)

        # Replace the log if it already exists
        if os.path.isfile(log_path):
            os.remove(log_path)

        # ## CALIBRATE THE GRAVITY MODEL ## #
        calib = gravity_model.GravityModelCalibrator(
            row_targets=np_productions,
            col_targets=np_attractions,
            cost_matrix=np_cost,
            target_cost_distribution=target_cost_distribution,
            running_log_path=log_path,
            cost_function=kwargs.get('cost_function'),
            target_convergence=kwargs.get('target_convergence'),
            furness_max_iters=kwargs.get('furness_max_iters'),
            furness_tol=kwargs.get('furness_tol'),
            use_perceived_factors=kwargs.get('use_perceived_factors'),
        )

        optimal_cost_params = calib.calibrate(
            init_params=kwargs.get('init_params'),
            grav_max_iters=kwargs.get('grav_max_iters'),
            calibrate_params=kwargs.get('calibrate_params', True),
            ftol=kwargs.get('ftol', 1e-5),
            failure_tol=kwargs.get("failure_tol", 0),
            default_init_params=kwargs.get("default_init_params", None),
            verbose=kwargs.get('verbose', 2),
        )

        # ## GENERATE REPORTS AND WRITE OUT ## #
        self._write_out_reports(
            segment_params=segment_params,
            running_segmentation=running_segmentation,
            init_cost_params=kwargs.get('init_params'),
            optimal_cost_params=optimal_cost_params,
            min_bounds=target_cost_distribution['min'].values,
            max_bounds=target_cost_distribution['max'].values,
            target_ave_cost=target_cost_distribution['ave_km'].values,
            target_band_share=target_cost_distribution['band_share'].values,
            initial_convergence=calib.initial_convergence,
            achieved_convergence=calib.achieved_convergence,
            achieved_band_share=calib.achieved_band_share,
            achieved_distribution=calib.achieved_distribution,
            overall_log_path=self.report_paths.overall_log,
            cost_matrix=np_cost,
        )

    def _multi_area_distribution(
        self,
        segment_params: Dict[str, Any],
        np_productions: np.ndarray,
        np_attractions: np.ndarray,
        np_cost: np.ndarray,
        np_calibration_matrix: np.ndarray,
        calibration_naming: Dict[Any, str],
        target_cost_distributions: Dict[Any, pd.DataFrame],
        running_segmentation: nd.SegmentationLevel,
        **kwargs,
    ):
        # ## SET UP SEGMENT LOG ## #
        # Logging set up
        log_fname = running_segmentation.generate_file_name(
            trip_origin=self.trip_origin,
            file_desc='gravity_log',
            segment_params=segment_params,
            csv=True,
        )
        log_path = os.path.join(self.report_paths.model_log_dir, log_fname)

        # Replace the log if it already exists
        if os.path.isfile(log_path):
            os.remove(log_path)

        # ## CALIBRATE THE GRAVITY MODEL ## #
        calib = gravity_model.MultiAreaGravityModelCalibrator(
            row_targets=np_productions,
            col_targets=np_attractions,
            calibration_matrix=np_calibration_matrix,
            cost_matrix=np_cost,
            target_cost_distributions=target_cost_distributions,
            calibration_naming=calibration_naming,
            running_log_path=log_path,
            cost_function=kwargs.get('cost_function'),
            target_convergence=kwargs.get('target_convergence'),
            furness_max_iters=kwargs.get('furness_max_iters'),
            furness_tol=kwargs.get('furness_tol'),
            use_perceived_factors=kwargs.get('use_perceived_factors'),
            memory_optimised=kwargs.get('memory_optimised'),
        )

        optimal_cost_params = calib.calibrate(
            init_params=kwargs.get('init_params'),
            grav_max_iters=kwargs.get('grav_max_iters'),
            calibrate_params=kwargs.get('calibrate_params', True),
            ftol=kwargs.get('ftol', 1e-5),
            verbose=kwargs.get('verbose', 2),
        )

        # ## GENERATE REPORTS AND WRITE OUT ## #
        # Multiprocessing setup
        unchanging_kwargs = {
            'segment_params': segment_params,
            'running_segmentation': running_segmentation,
            'init_cost_params': kwargs.get('init_params'),
            'cost_matrix': np_cost,
        }

        # Need to generate reports for each calibration area
        kwarg_list = list()
        for calib_id, calib_name in calibration_naming.items():
            calib_kwargs = unchanging_kwargs.copy()

            # Create the overall log path
            stem, ext = os.path.splitext(self.report_paths.overall_log)
            overall_log_path = "{stem}_{id}{ext}".format(stem=stem, id=calib_name, ext=ext)

            # Replace the log if it already exists
            if os.path.isfile(overall_log_path):
                os.remove(overall_log_path)

            calib_kwargs.update({
                'optimal_cost_params': optimal_cost_params[calib_id],
                'min_bounds': target_cost_distributions[calib_id]['min'].values,
                'max_bounds': target_cost_distributions[calib_id]['max'].values,
                'target_ave_cost': target_cost_distributions[calib_id]['ave_km'].values,
                'target_band_share': target_cost_distributions[calib_id]['band_share'].values,
                'initial_convergence': calib.initial_convergence[calib_id],
                'achieved_convergence': calib.achieved_convergence[calib_id],
                'achieved_band_share': calib.achieved_band_share[calib_id],
                'achieved_distribution': calib.achieved_distribution[calib_id],
                'overall_log_path': overall_log_path,
                'subdir_name': calib_name,
            })
            kwarg_list.append(calib_kwargs)

        # Generate the reports
        # Multiprocess here only if we aren't already
        process_count = -2 if self.process_count == 0 else 0
        multiprocessing.multiprocess(
            fn=self._write_out_reports,
            kwargs=kwarg_list,
            process_count=process_count
        )

        # ## WRITE THE FULL DISTRIBUTED DEMAND ## #
        # Put the demand into a df
        demand_df = pd.DataFrame(
            index=self.running_zones,
            columns=self.running_zones,
            data=calib.achieved_full_distribution.astype(np.float32),
        )

        demand_df = demand_df.reindex(
            index=self.zoning_system.unique_zones,
            columns=self.zoning_system.unique_zones,
            fill_value=0,
        )

        # Generate path and write out
        fname = running_segmentation.generate_file_name(
            trip_origin=self.trip_origin,
            year=str(self.year),
            file_desc='synthetic_pa',
            segment_params=segment_params,
            compressed=True,
        )
        path = os.path.join(self.export_paths.matrix_dir, fname)
        nd.write_df(demand_df, path)

    def distribute_segment(
        self,
        segment_params: Dict[str, Any],
        productions: pd.DataFrame,
        attractions: pd.DataFrame,
        cost_matrix: pd.DataFrame,
        calibration_matrix: pd.DataFrame,
        target_cost_distributions: Dict[Any, pd.DataFrame],
        calibration_naming: Dict[Any, Any],
        running_segmentation: nd.SegmentationLevel,
        **kwargs,
    ):
        # Init
        seg_name = running_segmentation.generate_file_name(segment_params)
        self._logger.info("Running for %s", seg_name)
        use_multi_area = len(target_cost_distributions) > 1

        # ## MAKE SURE COST AND P/A ARE IN SAME ORDER ## #
        # Sort the cost
        cost_matrix = pd_utils.reindex_rows_and_cols(
            df=cost_matrix,
            columns=self.running_zones,
            index=self.running_zones,
            fill_value=0,
        )

        # Check the cost is something sensible
        if cost_matrix.values.sum() == 0:
            raise ValueError(
                f"In segment {segment_params}.\n"
                "The generated cost matrix sums to zero. The gravity model "
                "cannot calibrate if all costs are 0."
            )

        if np.isnan(cost_matrix.values).any():
            nan_report = math_utils.pandas_nan_report(
                df=cost_matrix,
                row_name='production',
                col_name='attraction',
            )
            raise ValueError(
                f"In segment {segment_params}.\n"
                "Found np.nan values in generated cost matrix. The gravity "
                "model cannot calibrate with NaN costs. NaN values found in "
                "the following places:\n"
                f"{nan_report}"
            )

        # TODO(BT): Fix this problem at the cost source,
        #  only do for upper model. cars, bus, active
        # # Fill any zero costs with 0.2
        # cost_matrix = cost_matrix.mask(cost_matrix == 0, 0.2)

        # sort the productions and attractions
        productions = productions.set_index(self.zone_col)
        productions = productions.reindex(self.running_zones).fillna(0)
        attractions = attractions.set_index(self.zone_col)
        attractions = attractions.reindex(self.running_zones).fillna(0)

        # Convert to numpy for gravity model
        np_cost = cost_matrix.values
        np_calibration_matrix = calibration_matrix.values
        np_productions = productions[self._pa_val_col].values
        np_attractions = attractions[self._pa_val_col].values

        # Pick which calibrator to use and run!
        if not use_multi_area:
            self._single_area_distribution(
                segment_params=segment_params,
                np_productions=np_productions,
                np_attractions=np_attractions,
                np_cost=np_cost,
                target_cost_distribution=target_cost_distributions[1],  # Default Key
                running_segmentation=running_segmentation,
                **kwargs,
            )
        else:
            self._multi_area_distribution(
                segment_params=segment_params,
                np_productions=np_productions,
                np_attractions=np_attractions,
                np_cost=np_cost,
                np_calibration_matrix=np_calibration_matrix,
                calibration_naming=calibration_naming,
                target_cost_distributions=target_cost_distributions,
                running_segmentation=running_segmentation,
                **kwargs,
            )

    def distribute(self, *args, **kwargs):
        # Make new log if one already exists
        if os.path.isfile(self.report_paths.overall_log):
            os.remove(self.report_paths.overall_log)

        # Run default distribution
        super().distribute(*args, **kwargs)


class Furness3dDistributor(AbstractDistributor):
    _log_fname = "3D_Furness_log.log"

    _base_zone_col = "%s_zone_id"
    _pa_val_col = 'trips'

    _internal_only_suffix = 'int'

    def __init__(self,
                 year: int,
                 running_mode: nd.Mode,
                 trip_origin: str,
                 zoning_system: nd.ZoningSystem,
                 running_zones: List[Any],
                 export_home: nd.PathLike,
                 zone_col: str = None,
                 cost_units: str = None,
                 cost_name: str = None,
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
        super().__init__(
            year=year,
            running_mode=running_mode,
            trip_origin=trip_origin,
            zoning_system=zoning_system,
            running_zones=running_zones,
            zone_col=zone_col,
            export_home=export_home,
            process_count=process_count,
            cost_units=cost_units,
            cost_name=cost_name,
        )

        # Create a logger
        logger_name = "%s.%s" % (nd.get_package_logger_name(), self.__class__.__name__)
        log_file_path = os.path.join(self.export_home, self._log_fname)
        self._logger = nd.get_logger(
            logger_name=logger_name,
            log_file_path=log_file_path,
            instantiate_msg="Initialised new 3D Furness Logger",
        )

    def distribute_segment(self,
                           segment_params: Dict[str, Any],
                           productions: pd.DataFrame,
                           attractions: pd.DataFrame,
                           cost_matrix: pd.DataFrame,
                           calibration_matrix: pd.DataFrame,
                           target_cost_distributions: Dict[Any, pd.DataFrame],
                           calibration_naming: Dict[Any, Any],
                           running_segmentation: nd.SegmentationLevel,
                           **kwargs,
                           ):
        # Init
        seg_name = running_segmentation.generate_file_name(segment_params)
        self._logger.info("Running for %s" % seg_name)
        calibration_keys = np.unique(calibration_matrix).tolist()

        # ## SET UP SEGMENT LOG ## #
        # Logging set up
        log_fname = running_segmentation.generate_file_name(
            trip_origin=self.trip_origin,
            file_desc='furness3d_log',
            segment_params=segment_params,
            csv=True,
        )
        log_path = os.path.join(self.report_paths.model_log_dir, log_fname)

        # Replace the log if it already exists
        if os.path.isfile(log_path):
            os.remove(log_path)

        # ## MAKE SURE INPUTS ARE IN SAME ORDER ## #
        # Sort the cost
        cost_matrix = cost_matrix.reindex(
            columns=self.running_zones,
            index=self.running_zones,
        ).fillna(0)

        calibration_matrix = calibration_matrix.reindex(
            columns=self.running_zones,
            index=self.running_zones,
        ).fillna(0)

        # TODO(BT): Fix this problem at the cost source
        # Fill any zero costs with 0.2
        cost_matrix = cost_matrix.mask(cost_matrix == 0, 0.2)

        # sort the productions and attractions
        productions = productions.set_index(self.zone_col)
        productions = productions.reindex(self.running_zones).fillna(0)
        attractions = attractions.set_index(self.zone_col)
        attractions = attractions.reindex(self.running_zones).fillna(0)

        # Convert things to numpy
        np_cost = cost_matrix.values
        np_calibration_matrix = calibration_matrix.values
        np_productions = productions[self._pa_val_col].values
        np_attractions = attractions[self._pa_val_col].values

        # ## RUN THE 3D FURNESS ## #
        calib = furness.Furness3D(
            row_targets=np_productions,
            col_targets=np_attractions,
            cost_matrix=np_cost,
            base_matrix=kwargs.get('base_matrix'),
            calibration_matrix=np_calibration_matrix,
            target_cost_distributions=target_cost_distributions,
            calibration_naming=calibration_naming,
            calibration_ignore_val=self.calibration_ignore_val,
            running_log_path=log_path,
            target_convergence=kwargs.get('target_convergence'),
            furness_max_iters=kwargs.get('furness_max_iters'),
            furness_tol=kwargs.get('furness_tol'),
        )

        # Run
        calib.fit(
            outer_max_iters=kwargs.get('outer_max_iters'),
            calibrate=kwargs.get('calibrate'),
        )

        # ## GENERATE REPORTS AND WRITE OUT ## #
        for calib_key in calibration_keys:
            # Filter down to this area
            area_mask = (np_calibration_matrix == calib_key)
            area_distribution = calib.achieved_distribution * area_mask
            area_cost = np_cost * area_mask

            # Generate report
            report = self.generate_cost_distribution_report(
                min_bounds=target_cost_distributions[calib_key]['min'].values,
                max_bounds=target_cost_distributions[calib_key]['max'].values,
                target_ave_cost=target_cost_distributions[calib_key]['ave_km'].values,
                target_band_share=target_cost_distributions[calib_key]['band_share'].values,
                achieved_band_share=calib.achieved_band_shares[calib_key],
                achieved_convergence=calib.achieved_convergences[calib_key],
                achieved_distribution=area_distribution,
                cost_matrix=area_cost,
            )

            # Write out report
            fname = running_segmentation.generate_file_name(
                trip_origin=self.trip_origin,
                year=str(self.year),
                file_desc='tld_report',
                segment_params=segment_params,
                suffix=calibration_naming[calib_key],
                csv=True,
            )
            path = os.path.join(self.report_paths.tld_report_dir, fname)
            report.to_csv(path, index=False)

        # ## WRITE DISTRIBUTED DEMAND ## #
        # Put the demand into a df
        demand_df = pd.DataFrame(
            index=self.running_zones,
            columns=self.running_zones,
            data=calib.achieved_distribution.astype(np.float32),
        )

        demand_df = demand_df.reindex(
            index=self.zoning_system.unique_zones,
            columns=self.zoning_system.unique_zones,
            fill_value=0,
        )

        # Generate path and write out
        fname = running_segmentation.generate_file_name(
            trip_origin=self.trip_origin,
            year=str(self.year),
            file_desc='synthetic_pa',
            segment_params=segment_params,
            compressed=True,
        )
        path = os.path.join(self.export_paths.matrix_dir, fname)
        nd.write_df(demand_df, path)

        # ## ADD TO THE OVERALL LOG ## #
        # Generate the log
        log_dict = segment_params.copy()

        # Add initial and final bs_con for each area
        for calib_key in calibration_keys:
            init_name = '%s_init_bs_con' % calibration_naming[calib_key]
            final_name = '%s_final_bs_con' % calibration_naming[calib_key]
            log_dict.update({
                init_name: calib.initial_convergences[calib_key],
                final_name: calib.achieved_convergences[calib_key],
            })

        # Append this iteration to log file
        file_ops.safe_dataframe_to_csv(
            pd.DataFrame(log_dict, index=[0]),
            self.report_paths.overall_log,
            mode='a',
            header=(not os.path.exists(self.report_paths.overall_log)),
            index=False,
        )

    def distribute(self, *args, **kwargs):
        # Make new log if one already exists
        if os.path.isfile(self.report_paths.overall_log):
            os.remove(self.report_paths.overall_log)

        # Run default distribution
        super().distribute(*args, **kwargs)

