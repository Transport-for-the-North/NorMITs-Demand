# -*- coding: utf-8 -*-
"""
Created on: 07/12/2021
Updated on:

Original author: Ben Taylor
Last update made by:
Other updates made by:

File purpose:

"""
from __future__ import annotations

# Built-Ins
import os
import abc
import collections

from typing import Any
from typing import Dict
from typing import List
from typing import Union

# Third Party
import pandas as pd
import tqdm

# Local Imports
import normits_demand as nd
from normits_demand import constants
from normits_demand.cost import utils as cost_utils
from normits_demand.utils import file_ops
from normits_demand.utils import general as du
from normits_demand.utils import pandas_utils as pd_utils

# ## DEFINE COLLECTIONS OF OUTPUT PATHS ## #
# Exports
_DM_ExportPaths_NT = collections.namedtuple(
    typename='_DM_ExportPaths_NT',
    field_names=[
        'home',
        'full_pa_dir',
        'compiled_pa_dir',
        'full_od_dir',
        'compiled_od_dir',
        'compiled_od_dir_pcu',
    ]
)


_GravityExportPaths_NT = collections.namedtuple(
    typename='_GravityExportPaths_NT',
    field_names=[
        'home',
        'distribution_dir',
    ]
)

# Reports
_DM_ReportPaths_NT = collections.namedtuple(
    typename='_DM_ReportPaths_NT',
    field_names=[
        'home',
        'pa_reports_dir',
        'od_reports_dir',
    ]
)

_Distributor_ReportPaths_NT = collections.namedtuple(
    typename='_Distributor_ReportPaths_NT',
    field_names=[
        'home',
        'segment_log_dir',
        'tld_report_dir',
    ]
)

_GravityReportPaths_NT = collections.namedtuple(
    typename='_GravityReportPaths_NT',
    field_names=[
        'home',
        'overall_log',
        'model_log_dir',
        'tld_report_dir',
    ]
)


# ## DEFINE IMPORT PATHS ## #
class DMArgumentBuilderBase(abc.ABC):
    """
    Abstract Class defining how the argument builder for the
    distribution model should look.

    If custom import paths are needed, then a new class needs to be made
    which inherits this abstract class. DistributionModel can then use the
    defined functions/properties to pick up new import files.
    """

    @abc.abstractmethod
    def build_upper_model_arguments(self) -> Dict[str, Any]:
        pass

    @abc.abstractmethod
    def build_lower_model_arguments(self) -> Dict[str, Any]:
        pass


class DistributionModelArgumentBuilder(DMArgumentBuilderBase):
    # Costs constants
    _modal_dir_name = 'modal'
    _cost_dir_name = 'costs'
    _cost_base_fname = "{zoning_name}_{cost_type}_costs.csv"

    # Initial Parameters consts
    _gravity_model_dir = 'gravity model'

    # Trip Length Distribution constants
    _tld_dir_name = 'trip_length_distributions'

    def __init__(self,
                 import_home: nd.PathLike,
                 trip_origin: str,
                 productions: nd.DVector,
                 attractions: nd.DVector,
                 running_mode: nd.Mode,
                 running_segmentation: nd.SegmentationLevel,
                 zoning_system: nd.ZoningSystem,
                 target_tld_name: str,
                 init_params_fname: str,
                 init_params_cols: List[str],
                 upper_model_method: nd.DistributionMethod,
                 upper_distributor_kwargs: Dict[str, Any],
                 lower_model_method: nd.DistributionMethod = None,
                 lower_distributor_kwargs: Dict[str, Any] = None,
                 intrazonal_cost_infill: float = None,
                 ):
        # Check paths exist
        file_ops.check_path_exists(import_home)

        # TODO(BT): Validate segments and zones are the correct types

        # Assign attributes
        self.import_home = import_home
        self.trip_origin = trip_origin
        self.productions = productions
        self.attractions = attractions

        self.running_mode = running_mode
        self.running_segmentation = running_segmentation
        self.zoning_system = zoning_system
        self.target_tld_name = target_tld_name

        self.init_params_fname = init_params_fname
        self.init_params_cols = init_params_cols

        self.upper_model_method = upper_model_method
        self.lower_model_method = lower_model_method
        self.upper_distributor_kwargs = upper_distributor_kwargs
        self.lower_distributor_kwargs = lower_distributor_kwargs

        self.intrazonal_cost_infill = intrazonal_cost_infill

    @staticmethod
    def _maybe_read_trip_end(trip_end: Union[nd.DVector, nd.PathLike]) -> nd.DVector:
        """Read in a Dvector if path was give"""
        # Just return if it's a DVector
        if isinstance(trip_end, nd.core.data_structures.DVector):
            return trip_end

        if not os.path.exists(trip_end):
            raise ValueError(
                "Expected either a DVector or a path to one. No path exists at "
                "%s" % trip_end
            )

        return nd.read_pickle(trip_end)

    def _get_cost(self, segment_params: Dict[str, Any]) -> pd.DataFrame:
        """Reads in the cost matrix for this segment"""
        # Generate the path to the segment file
        cost_dir = os.path.join(
            self.import_home,
            self._modal_dir_name,
            self.running_mode.value,
            self._cost_dir_name,
            self.zoning_system.name,
        )
        fname = self.running_segmentation.generate_file_name(
            trip_origin=self.trip_origin,
            file_desc="%s_cost" % self.zoning_system.name,
            segment_params=segment_params,
            csv=True,
        )
        path = os.path.join(cost_dir, fname)

        # Read in the costs and infill
        cost_matrix = nd.read_df(path, find_similar=True, index_col=0)
        if self.intrazonal_cost_infill is not None:
            cost_matrix = cost_utils.iz_infill_costs(
                cost_matrix,
                iz_infill=self.intrazonal_cost_infill,
            )

        return cost_matrix

    def _get_target_cost_distribution(self,
                                      segment_params: Dict[str, Any],
                                      ) -> pd.DataFrame:
        """Reads in the target cost distribution for this segment"""
        # Generate the path to the cost distribution file
        tcd_dir = os.path.join(
            self.import_home,
            self._tld_dir_name,
            self.target_tld_name,
        )
        fname = self.running_segmentation.generate_file_name(
            trip_origin=self.trip_origin,
            file_desc="tlb",
            segment_params=segment_params,
            csv=True,
        )
        path = os.path.join(tcd_dir, fname)

        # Convert to expected format
        target_cost_distribution = file_ops.read_df(path)

        rename = {'lower': 'min', 'upper': 'max'}
        target_cost_distribution = target_cost_distribution.rename(columns=rename)
        target_cost_distribution['min'] *= constants.MILES_TO_KM
        target_cost_distribution['max'] *= constants.MILES_TO_KM

        return target_cost_distribution

    def _get_init_params(self,
                         segment_params: Dict[str, Any],
                         init_params_df: pd.DataFrame,
                         ) -> Dict[str, int]:
        """Extracts the init params for this segment from init_params_df"""
        seg_init_params = pd_utils.filter_df(init_params_df, segment_params)

        if len(seg_init_params) > 1:
            seg_name = self.running_segmentation.generate_file_name(segment_params)
            raise ValueError(
                "%s rows found in init_params for segment %s. "
                "Expecting only 1 row."
                % (len(seg_init_params), seg_name)
            )

        # Make sure the columns we need do exist
        seg_init_params = pd_utils.reindex_cols(
            df=seg_init_params,
            columns=self.init_params_cols,
            dataframe_name='init_params',
        )

        return {x: seg_init_params[x].squeeze() for x in self.init_params_cols}

    def _build_gravity_by_segment_kwargs(self):
        """Build the dictionary of kwargs for each segment"""
        # Read in the init_params_df
        path = os.path.join(
            self.import_home,
            self._gravity_model_dir,
            self.init_params_fname,
        )
        init_params_df = file_ops.read_df(path)

        # Generate by segment kwargs
        by_segment_kwargs = dict()
        for segment_params in self.running_segmentation:
            # Get the needed kwargs
            segment_name = self.running_segmentation.get_segment_name(segment_params)
            init_params = self._get_init_params(segment_params, init_params_df)

            # Add to dictionary
            by_segment_kwargs[segment_name] = {
                'init_params': init_params,
            }

        return by_segment_kwargs

    def _build_furness3d_by_segment_kwargs(self):
        raise NotImplementedError

    def _build_by_segment_kwargs(self, method: nd.DistributionMethod):
        if method == nd.DistributionMethod.GRAVITY:
            fn = self._build_gravity_by_segment_kwargs
        elif method == nd.DistributionMethod.FURNESS3D:
            fn = self._build_furness3d_by_segment_kwargs
        else:
            raise NotImplementedError(
                "No function exists to build the by_segment_kwargs for "
                "DistributionMethod %s"
                % method.value
            )
        return fn()

    def _build_cost_matrices(self):
        """Build the dictionary of cost matrices for each segment"""
        # Generate by segment kwargs
        cost_matrices = dict()
        desc = "Reading in cost"
        # TODO: NOT KLUDGE
        count = 1
        for segment_params in tqdm.tqdm(self.running_segmentation, desc=desc):
            # Get the needed kwargs
            segment_name = self.running_segmentation.get_segment_name(segment_params)
            if count == 1:
                cost_matrix = self._get_cost(segment_params)
                import numpy as np
                print(np.count_nonzero(cost_matrix == 0))
                count += 1

            # Add to dictionary
            cost_matrices[segment_name] = cost_matrix

        return cost_matrices

    def _build_target_cost_distributions(self):
        """Build the dictionary of target_cost_distributions for each segment"""
        # Generate by segment kwargs
        target_cost_distributions = dict()
        for segment_params in self.running_segmentation:
            # Get the needed kwargs
            segment_name = self.running_segmentation.get_segment_name(segment_params)
            target_cost_distribution = self._get_target_cost_distribution(segment_params)

            # Add to dictionary
            target_cost_distributions[segment_name] = target_cost_distribution

        return target_cost_distributions

    def build_upper_model_arguments(self):
        # Read and validate trip ends
        productions = self._maybe_read_trip_end(self.productions)
        attractions = self._maybe_read_trip_end(self.attractions)

        cost_matrices = self._build_cost_matrices()
        target_cost_distributions = self._build_target_cost_distributions()
        by_segment_kwargs = self._build_by_segment_kwargs(self.upper_model_method)

        final_kwargs = self.upper_distributor_kwargs.copy()
        final_kwargs.update({
            'productions': productions.to_df(),
            'attractions': attractions.to_df(),
            'running_segmentation': self.running_segmentation,
            'cost_matrices': cost_matrices,
            'target_cost_distributions': target_cost_distributions,
            'by_segment_kwargs': by_segment_kwargs,
        })

        return final_kwargs

    def build_lower_model_arguments(self):
        pass


# ## DEFINE EXPORT PATHS ##
class DistributorExportPaths(abc.ABC):
    _reports_dirname = 'Logs & Reports'

    # Report dir names
    _overall_log_name = '{trip_origin}_overall_log.csv'
    _log_dir_name = 'Logs'
    _tld_report_dir = 'TLD Reports'

    def __init__(self,
                 year: int,
                 trip_origin: str,
                 running_mode: nd.Mode,
                 export_home: nd.PathLike,
                 ):
        # Init
        file_ops.check_path_exists(export_home)

        # Assign attributes
        self.year = year
        self.trip_origin = trip_origin
        self.running_mode = running_mode
        self.export_home = export_home
        self.report_home = os.path.join(self.export_home, self._reports_dirname)

        file_ops.create_folder(self.report_home)

        # Generate the paths
        self._create_export_paths()
        self._create_report_paths()

    @abc.abstractmethod
    def _create_export_paths(self) -> None:
        """Creates self.export_paths"""
        raise NotImplementedError

    @abc.abstractmethod
    def _create_report_paths(self) -> None:
        """Creates self.report_paths"""
        raise NotImplementedError


class GravityDistributorExportPaths(DistributorExportPaths):
    # Export dir names
    _dist_out_dir = 'Matrices'

    def _create_export_paths(self) -> None:
        """Creates self.export_paths"""

        # Build the matrix output path
        distribution_dir = os.path.join(self.export_home, self._dist_out_dir)

        # Make paths that don't exist
        dir_paths = [distribution_dir]
        for path in dir_paths:
            file_ops.create_folder(path)

        # Create the export_paths class
        self.export_paths = _GravityExportPaths_NT(
            home=self.export_home,
            distribution_dir=distribution_dir,
        )

    def _create_report_paths(self) -> None:
        """Creates self.report_paths"""
        # Build paths
        fname = self._overall_log_name.format(trip_origin=self.trip_origin)
        overall_log_path = os.path.join(self.report_home, fname)
        model_log_dir = os.path.join(self.report_home, self._log_dir_name)
        tld_report_dir = os.path.join(self.report_home, self._tld_report_dir)

        # Make paths that don't exist
        dir_paths = [self.report_home, model_log_dir, tld_report_dir]
        for path in dir_paths:
            file_ops.create_folder(path)

        # Create the export_paths class
        self.report_paths = _GravityReportPaths_NT(
            home=self.report_home,
            overall_log=overall_log_path,
            model_log_dir=model_log_dir,
            tld_report_dir=tld_report_dir,
        )


class DistributionModelExportPaths:

    # Define the names of the export dirs
    _upper_model_dir = 'Upper Model'
    _lower_model_dir = 'Lower Model'
    _final_outputs_dir = 'Final Outputs'

    # Export dir names
    _full_pa_out_dir = 'Full PA Matrices'
    compiled_pa_out_dir = 'Compiled PA Matrices'
    _full_od_out_dir = 'Full OD Matrices'
    _compiled_od_out_dir = 'Compiled OD Matrices'
    _compiled_od_out_dir_pcu = 'PCU'

    # Report dir names
    _reports_dirname = 'Reports'
    _pa_report_dir = 'PA Reports'
    _od_report_dir = 'OD Reports'

    def __init__(self,
                 year: int,
                 trip_origin: str,
                 iteration_name: str,
                 running_mode: nd.Mode,
                 upper_model_method: nd.DistributionMethod,
                 lower_model_method: nd.DistributionMethod,
                 export_home: nd.PathLike,
                 ):
        """
        Builds the export paths for all the distribution model

        Parameters
        ----------
        year:
            The year the distribution model is running for

        iteration_name:
            The name of this iteration of the distribution model.
            Will have 'iter' pre-pended to create the folder name. e.g. if
            iteration_name was set to '3i' the iteration folder would be
            called 'iter3i'.

        running_mode:
            The mode that the distribution model is running for. Only one
            mode can be run at a time.

        export_home:
            The home directory of all the export paths. A sub-directory will
            be made for the upper and lower models being run
        """
        # Init
        file_ops.check_path_exists(export_home)

        self.year = year
        self.trip_origin = trip_origin
        self.iteration_name = du.create_iter_name(iteration_name)
        self.running_mode = running_mode
        self.upper_model_method = upper_model_method
        self.lower_model_method = lower_model_method
        self.export_home = os.path.join(export_home, self.iteration_name, self.running_mode.value)
        file_ops.create_folder(self.export_home)

        # ## BUILD ALL MODEL PATHS ## #
        # Upper Model
        self.upper_export_home = os.path.join(self.export_home, self._upper_model_dir)
        file_ops.create_folder(self.upper_export_home)
        self.upper_exports = upper_model_method.get_export_paths(
            year=self.year,
            trip_origin=self.trip_origin,
            running_mode=self.running_mode,
            export_home=self.upper_export_home,
        )

        # Lower Model
        self.lower_export_home = os.path.join(self.export_home, self._lower_model_dir)
        file_ops.create_folder(self.lower_export_home)
        self.lower_exports = upper_model_method.get_export_paths(
            year=self.year,
            trip_origin=self.trip_origin,
            running_mode=self.running_mode,
            export_home=self.lower_export_home,
        )

        # Final Output Paths
        export_home = os.path.join(self.export_home, self._final_outputs_dir)
        report_home = os.path.join(export_home, self._reports_dirname)
        file_ops.create_folder(export_home)
        file_ops.create_folder(report_home)

        # Generate the paths
        self._create_export_paths(export_home)
        self._create_report_paths(report_home)

    def _create_export_paths(self, export_home: str) -> None:
        # Build the matrix output path
        full_pa_dir = os.path.join(export_home, self._full_pa_out_dir)
        compiled_pa_dir = os.path.join(export_home, self.compiled_pa_out_dir)
        full_od_dir = os.path.join(export_home, self._full_od_out_dir)
        compiled_od_dir = os.path.join(export_home, self._compiled_od_out_dir)
        compiled_od_dir_pcu = os.path.join(compiled_od_dir, self._compiled_od_out_dir_pcu)

        # Create the export_paths class
        self.export_paths = _DM_ExportPaths_NT(
            home=export_home,
            full_pa_dir=full_pa_dir,
            compiled_pa_dir=compiled_pa_dir,
            full_od_dir=full_od_dir,
            compiled_od_dir=compiled_od_dir,
            compiled_od_dir_pcu=compiled_od_dir_pcu,
        )

        # Make all paths that don't exist
        dir_paths = [
            full_pa_dir,
            compiled_pa_dir,
            full_od_dir,
            compiled_od_dir,
            compiled_od_dir_pcu,
        ]
        for path in dir_paths:
            file_ops.create_folder(path)

    def _create_report_paths(self,
                             report_home: str,
                             ) -> None:
        """Creates self.report_paths"""
        # Create the overall report paths
        self.report_paths = _DM_ReportPaths_NT(
            home=report_home,
            pa_reports_dir=os.path.join(report_home, self._pa_report_dir),
            od_reports_dir=os.path.join(report_home, self._od_report_dir),
        )

        # Make paths that don't exist
        for path in self.report_paths:
            file_ops.create_folder(path)
