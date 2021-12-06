# -*- coding: utf-8 -*-
"""
Created on: 09/09/2021
Updated on:

Original author: Ben Taylor
Last update made by:
Other updates made by:

File purpose:
Classes which build all the paths for TMS model inputs outputs
"""
# Built-Ins
import os
import abc
import collections

from typing import Any
from typing import Dict
from typing import Union
from typing import Optional

# Third Party
import pandas as pd

# Local Imports
import normits_demand as nd

from normits_demand.utils import file_ops
from normits_demand.utils import general as du
from normits_demand.utils import pandas_utils as pd_utils


class ExternalModelArgumentBuilderBase(abc.ABC):
    """Abstract Class defining how the argument builder for the external model should look.

    If custom import paths are needed, then a new class needs to be made
    which inherits this abstract class. TMS can then use the defined
    functions to pick up new import files.
    """

    @abc.abstractmethod
    def build_hb_arguments(self) -> Dict[str, nd.PathLike]:
        """
        Definition of function that needs to be overwritten

        Returns
        -------
        kwargs:
            This function should return a keyword argument dictionary to be
            passed straight into the production model. Specifically, this
            function needs to produce a dictionary with the following keys:
                # TODO (BT) Define how this should look
        """
        pass

    @abc.abstractmethod
    def build_nhb_arguments(self) -> Dict[str, nd.PathLike]:
        """
        Definition of function that needs to be overwritten

        Returns
        -------
        kwargs:
            This function should return a keyword argument dictionary to be
            passed straight into the production model. Specifically, this
            function needs to produce a dictionary with the following keys:
                # TODO (BT) Define how this should look
        """
        pass


class GravityModelArgumentBuilderBase(abc.ABC):
    """Abstract Class defining how the argument builder for the gravity model should look.

    If custom import paths are needed, then a new class needs to be made
    which inherits this abstract class. TMS can then use the defined
    functions to pick up new import files.
    """

    @abc.abstractmethod
    def build_hb_arguments(self) -> Dict[str, nd.PathLike]:
        """
        Definition of function that needs to be overwritten

        Returns
        -------
        kwargs:
            This function should return a keyword argument dictionary to be
            passed straight into the production model. Specifically, this
            function needs to produce a dictionary with the following keys:
                # TODO (BT) Define how this should look
        """
        pass

    @abc.abstractmethod
    def build_nhb_arguments(self) -> Dict[str, nd.PathLike]:
        """
        Definition of function that needs to be overwritten

        Returns
        -------
        kwargs:
            This function should return a keyword argument dictionary to be
            passed straight into the production model. Specifically, this
            function needs to produce a dictionary with the following keys:
                # TODO (BT) Define how this should look
        """
        pass


class TMSArgumentBuilderBase(abc.ABC):
    """Abstract Class defining how the argument builder for TMS should look.

    If custom import paths are needed, then a new class needs to be made
    which inherits this abstract class. TMS can then use the defined
    functions/properties to pick up new import files.
    """

    @property
    @abc.abstractmethod
    def external_model_arg_builder(self) -> ExternalModelArgumentBuilderBase:
        pass

    @property
    @abc.abstractmethod
    def gravity_model_arg_builder(self) -> GravityModelArgumentBuilderBase:
        pass

    @abc.abstractmethod
    def build_pa_to_od_arguments(self) -> Dict[str, Any]:
        pass


# ## EXTERNAL MODEL CLASSES ## #
class ExternalModelArgumentBuilder(ExternalModelArgumentBuilderBase):
    # Costs constants
    _modal_dir_name = 'modal'
    _cost_dir_name = 'costs'
    _cost_base_fname = "{zoning_name}_{cost_type}_costs.csv"

    # CJTW constants
    _cjtw_infill = 1e-7
    _cjtw_dir_name = 'cjtw'
    _cjtw_base_fname = 'cjtw_{zoning_name}.csv'

    # Trip Length Distribution constants
    _tld_dir_name = 'trip_length_distributions'

    def __init__(self,
                 import_home: nd.PathLike,
                 running_mode: nd.core.enumerations.Mode,
                 hb_running_segmentation: nd.core.segments.SegmentationLevel,
                 nhb_running_segmentation: nd.core.segments.SegmentationLevel,
                 zoning_system: nd.core.zoning.ZoningSystem,
                 internal_tld_name: str,
                 external_tld_name: str,
                 hb_cost_type: str,
                 nhb_cost_type: str,
                 hb_productions: nd.DVector,
                 hb_attractions: nd.DVector,
                 nhb_productions: nd.DVector,
                 nhb_attractions: nd.DVector,
                 intrazonal_cost_infill: float = 0.5,
                 convergence_target: float = 0.9,
                 external_iters: int = 50,
                 furness_tol: float = 0.1,
                 furness_max_iters: int = 5000,
                 **kwargs,
                 ):
        # Check paths exist
        file_ops.check_path_exists(import_home)

        # TODO(BT): Validate segments and zones are the correct types

        # Assign attributes
        self.import_home = import_home
        self.running_mode = running_mode
        self.hb_running_segmentation = hb_running_segmentation
        self.nhb_running_segmentation = nhb_running_segmentation
        self.zoning_system = zoning_system
        self.internal_tld_name = internal_tld_name
        self.external_tld_name = external_tld_name
        self.hb_cost_type = hb_cost_type
        self.nhb_cost_type = nhb_cost_type
        self.intrazonal_cost_infill = intrazonal_cost_infill
        self.convergence_target = convergence_target
        self.external_iters = external_iters
        self.furness_tol = furness_tol
        self.furness_max_iters = furness_max_iters
        self.kwargs = kwargs

        # Validate and assign trip ends
        te_names = ['hb_production', 'hb_attraction', 'nhb_production', 'nhb_attraction']
        te_vals = [hb_productions, hb_attractions, nhb_productions, nhb_attractions]
        segs = [hb_running_segmentation] * 2 + [nhb_running_segmentation] * 2
        for name, trip_end, running_seg in zip(te_names, te_vals, segs):
            if trip_end.zoning_system != self.zoning_system:
                raise ValueError(
                    "The given '%s' trip ends were not in the correct zoning "
                    "system, they need to be in the same zoning system "
                    "that the external model is running at.\n"
                    "External zoning: %s\n"
                    "%s zoning: %s\n"
                    % (name, self.zoning_system, name, trip_end.zoning_system)
                )

            if trip_end.segmentation != running_seg:
                raise ValueError(
                    "The given '%s' trip ends were not in the correct "
                    "segmentation, they need to be in the same segmentation "
                    "that the external model is running at.\n"
                    "External segmentation: %s\n"
                    "%s segmentation: %s\n"
                    % (name, running_seg, name, trip_end.segmentation)
                )

        # Must all be fine if we're here. Assign.
        self.hb_productions = hb_productions
        self.hb_attractions = hb_attractions
        self.nhb_productions = nhb_productions
        self.nhb_attractions = nhb_attractions

    def _read_and_convert_cjtw(self):
        # Build a path to the input
        fname = self._cjtw_base_fname.format(zoning_name=self.zoning_system.name)
        path = os.path.join(self.import_home, self._cjtw_dir_name, fname)

        # Read and convert to zoning system format
        cjtw = read_cjtw(
            file_path=path,
            zoning_name=self.zoning_system.name,
            subset=None,
            reduce_to_pa_factors=False,
        )

        # Aggregate mode
        p_col = list(cjtw)[0]
        a_col = list(cjtw)[1]
        cjtw = cjtw[cjtw['mode'] == self.running_mode.get_mode_num()].copy()
        cjtw = cjtw.reindex([p_col, a_col, 'trips'], axis=1)
        cjtw = cjtw.groupby([p_col, a_col]).sum().reset_index()

        # Convert to a wide matrix
        return pd_utils.long_to_wide_infill(
            df=cjtw,
            index_col=p_col,
            columns_col=a_col,
            values_col='trips',
            index_vals=self.zoning_system.unique_zones,
            column_vals=self.zoning_system.unique_zones,
            infill=self._cjtw_infill,
        )

    def build_arguments(self, trip_origin: str) -> Dict[str, nd.PathLike]:
        # Init
        trip_origin = trip_origin.lower()

        # Set trip origin specific stuff
        if trip_origin == 'hb':
            productions = self.hb_productions
            attractions = self.hb_attractions
            running_segmentation = self.hb_running_segmentation
            cost_type = self.hb_cost_type
        elif trip_origin == 'nhb':
            productions = self.nhb_productions
            attractions = self.nhb_attractions
            running_segmentation = self.nhb_running_segmentation
            cost_type = self.nhb_cost_type
        else:
            raise ValueError(
                "Received an unexpected value for trip origin. Expected one of "
                "'hb' or 'nhb'. Got %s"
                % trip_origin
            )

        # Build TLD directory paths
        base_tld_path = os.path.join(
            self.import_home,
            self._tld_dir_name,
        )
        internal_tld_path = os.path.join(base_tld_path, self.internal_tld_name)
        external_tld_path = os.path.join(base_tld_path, self.external_tld_name)

        # Build costs path
        fname = self._cost_base_fname.format(
            zoning_name=self.zoning_system.name,
            cost_type=cost_type,
        )
        costs_path = os.path.join(
            self.import_home,
            self._modal_dir_name,
            self.running_mode.value,
            self._cost_dir_name,
            fname,
        )

        # Check paths exist
        paths = [
            internal_tld_path,
            external_tld_path,
            costs_path,
        ]

        for path in paths:
            if not os.path.exists(path):
                raise IOError(
                    "Generated path doesn't exist!\nGenerated the following "
                    "path that does not exist while building External "
                    "Model arguments.\n %s"
                    % path
                )

        # Return the generated arguments
        final_kwargs = self.kwargs.copy()
        final_kwargs.update({
            'productions': productions.to_df(),
            'attractions': attractions.to_df(),
            'running_segmentation': running_segmentation,
            'seed_matrix': self._read_and_convert_cjtw(),
            'costs_path': costs_path,
            'internal_tld_dir': internal_tld_path,
            'external_tld_dir': external_tld_path,
            'intrazonal_cost_infill': self.intrazonal_cost_infill,
            'convergence_target': self.convergence_target,
            'external_iters': self.external_iters,
            'furness_tol': self.furness_tol,
            'furness_max_iters': self.furness_max_iters,
        })
        return final_kwargs

    def build_hb_arguments(self) -> Dict[str, nd.PathLike]:
        return self.build_arguments(trip_origin='hb')

    def build_nhb_arguments(self) -> Dict[str, nd.PathLike]:
        return self.build_arguments(trip_origin='nhb')


_EM_ExportPaths_NT = collections.namedtuple(
    typename='_EM_ExportPaths_NT',
    field_names=[
        'home',
        'hb_internal_productions',
        'nhb_internal_productions',
        'hb_internal_attractions',
        'nhb_internal_attractions',
        'external_distribution_dir',
        'full_distribution_dir',
    ]
)

_EM_ReportPaths_NT = collections.namedtuple(
    typename='_EM_ReportPaths_NT',
    field_names=[
        'home',
        'model_log_dir',
        'tld_report_dir',
        'ie_report_dir',
    ]
)


class ExternalModelExportPaths:
    _reports_dirname = 'Logs & Reports'

    _productions_dir_name = 'Productions'
    _attractions_dir_name = 'Attractions'
    _external_dist_dir_name = 'External Matrices'
    _full_dist_dir_name = 'Full PA Matrices'

    _int_productions_base_name = '{trip_origin}_{mode}_{year}_internal_productions.pkl'
    _int_attractions_base_name = '{trip_origin}_{mode}_{year}_internal_attractions.pkl'

    # Report dir names
    _log_dir_name = 'Logs'
    _tld_report_dir = 'TLD Reports'
    _ie_report_dir = 'IE Reports'

    def __init__(self,
                 year: int,
                 running_mode: nd.Mode,
                 export_home: nd.PathLike,
                 ):
        # Init
        file_ops.check_path_exists(export_home)

        # Assign attributes
        self.year = year
        self.running_mode = running_mode
        self.export_home = export_home  # Something like I:\NorMITs Demand\noham\TMS\iter8\External Model
        self.report_home = os.path.join(self.export_home, self._reports_dirname)

        file_ops.create_folder(self.report_home)

        # Generate the paths
        self._create_export_paths()
        self._create_report_paths()

    def _create_export_paths(self) -> None:
        """Creates self.export_paths"""
        # Init
        kwargs = {'mode': self.running_mode.value, 'year': self.year}

        # Generate production and paths
        production_out = os.path.join(self.export_home, self._productions_dir_name)
        file_ops.create_folder(production_out)

        fname = self._int_productions_base_name.format(trip_origin='hb', **kwargs)
        hb_internal_productions = os.path.join(production_out, fname)

        fname = self._int_productions_base_name.format(trip_origin='nhb', **kwargs)
        nhb_internal_productions = os.path.join(production_out, fname)

        # Generate attraction and paths
        attraction_out = os.path.join(self.export_home, self._attractions_dir_name)
        file_ops.create_folder(attraction_out)

        fname = self._int_attractions_base_name.format(trip_origin='hb', **kwargs)
        hb_internal_attractions = os.path.join(attraction_out, fname)

        fname = self._int_attractions_base_name.format(trip_origin='nhb', **kwargs)
        nhb_internal_attractions = os.path.join(attraction_out, fname)

        # Generate external distribution path
        external_distribution_dir = os.path.join(
            self.export_home,
            self._external_dist_dir_name,
        )
        file_ops.create_folder(external_distribution_dir)

        # Generate full distribution path
        full_distribution_dir = os.path.join(
            self.export_home,
            self._full_dist_dir_name,
        )
        file_ops.create_folder(full_distribution_dir)

        # Create the export_paths class
        self.export_paths = _EM_ExportPaths_NT(
            home=self.export_home,
            hb_internal_productions=hb_internal_productions,
            nhb_internal_productions=nhb_internal_productions,
            hb_internal_attractions=hb_internal_attractions,
            nhb_internal_attractions=nhb_internal_attractions,
            external_distribution_dir=external_distribution_dir,
            full_distribution_dir=full_distribution_dir,
        )

    def _create_report_paths(self) -> None:
        """Creates self.report_paths"""

        # Create the export_paths class
        self.report_paths = _EM_ReportPaths_NT(
            home=self.report_home,
            model_log_dir=os.path.join(self.report_home, self._log_dir_name),
            tld_report_dir=os.path.join(self.report_home, self._tld_report_dir),
            ie_report_dir=os.path.join(self.report_home, self._ie_report_dir),
        )

        # Make paths that don't exist
        for path in self.report_paths:
            file_ops.create_folder(path)


# ## GRAVITY MODEL CLASSES ## #
class GravityModelArgumentBuilder(GravityModelArgumentBuilderBase):
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
                 target_tld_name: str,
                 cost_function: str,
                 running_mode: nd.core.enumerations.Mode,
                 hb_running_segmentation: nd.core.segments.SegmentationLevel,
                 nhb_running_segmentation: nd.core.segments.SegmentationLevel,
                 zoning_system: nd.core.zoning.ZoningSystem,
                 hb_cost_type: str,
                 nhb_cost_type: str,
                 hb_init_params_fname: str,
                 nhb_init_params_fname: str,
                 hb_productions: Union[nd.DVector, nd.PathLike],
                 hb_attractions: nd.DVector,
                 nhb_productions: nd.DVector,
                 nhb_attractions: nd.DVector,
                 intrazonal_cost_infill: float = 0.5,
                 pa_val_col: Optional[str] = 'val',
                 convergence_target: Optional[float] = 0.95,
                 fitting_loops: Optional[int] = 100,
                 furness_max_iters: Optional[int] = 5000,
                 furness_tol: Optional[float] = 0.1,
                 **kwargs
                 ):
        # Check paths exist
        file_ops.check_path_exists(import_home)

        # TODO(BT): Validate segments and zones are the correct types

        # Assign attributes
        self.import_home = import_home
        self.target_tld_name = target_tld_name
        self.cost_function = cost_function
        self.running_mode = running_mode
        self.hb_running_segmentation = hb_running_segmentation
        self.nhb_running_segmentation = nhb_running_segmentation
        self.zoning_system = zoning_system
        self.hb_cost_type = hb_cost_type
        self.nhb_cost_type = nhb_cost_type
        self.hb_init_params_fname = hb_init_params_fname
        self.nhb_init_params_fname = nhb_init_params_fname

        self.intrazonal_cost_infill = intrazonal_cost_infill
        self.pa_val_col = pa_val_col
        self.convergence_target = convergence_target
        self.fitting_loops = fitting_loops
        self.furness_max_iters = furness_max_iters
        self.furness_tol = furness_tol
        self.kwargs = kwargs

        # Assign the trip ends
        self.hb_productions = hb_productions
        self.hb_attractions = hb_attractions
        self.nhb_productions = nhb_productions
        self.nhb_attractions = nhb_attractions

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

    def build_arguments(self, trip_origin: str) -> Dict[str, nd.PathLike]:
        # Init
        trip_origin = trip_origin.lower()

        # Set trip origin specific stuff
        if trip_origin == 'hb':
            productions = self.hb_productions
            attractions = self.hb_attractions
            running_segmentation = self.hb_running_segmentation
            cost_type = self.hb_cost_type
            init_params_fname = self.hb_init_params_fname
        elif trip_origin == 'nhb':
            productions = self.nhb_productions
            attractions = self.nhb_attractions
            running_segmentation = self.nhb_running_segmentation
            cost_type = self.nhb_cost_type
            init_params_fname = self.nhb_init_params_fname
        else:
            raise ValueError(
                "Received an unexpected value for trip origin. Expected one of "
                "'hb' or 'nhb'. Got %s"
                % trip_origin
            )

        # Read and validate trip ends
        productions = self._maybe_read_trip_end(productions)
        attractions = self._maybe_read_trip_end(attractions)

        # Validate and assign trip ends
        te_names = ['%s_production' % trip_origin, '%s_attraction' % trip_origin]
        te_vals = [productions, attractions]
        segs = [running_segmentation] * 2
        for name, trip_end, running_seg in zip(te_names, te_vals, segs):
            if trip_end.zoning_system != self.zoning_system:
                raise ValueError(
                    "The given '%s' trip ends were not in the correct zoning "
                    "system, they need to be in the same zoning system "
                    "that the gravity model is running at.\n"
                    "Gravity zoning: %s\n"
                    "%s zoning: %s\n"
                    % (name, self.zoning_system, name, trip_end.zoning_system)
                )

            if trip_end.segmentation != running_seg:
                raise ValueError(
                    "The given '%s' trip ends were not in the correct "
                    "segmentation, they need to be in the same segmentation "
                    "that the gravity model is running at.\n"
                    "Gravity segmentation: %s\n"
                    "%s segmentation: %s\n"
                    % (name, running_seg, name, trip_end.segmentation)
                )

        # Build TLD directory paths
        base_tld_path = os.path.join(
            self.import_home,
            self._tld_dir_name,
        )
        target_tld_dir = os.path.join(base_tld_path, self.target_tld_name)

        # Build costs path
        cost_dir = os.path.join(
            self.import_home,
            self._modal_dir_name,
            self.running_mode.value,
            self._cost_dir_name,
            self.zoning_system.name,
        )

        # Load in the initial parameters for gravity model
        path = os.path.join(
            self.import_home,
            self._gravity_model_dir,
            init_params_fname,
        )
        init_params = file_ops.read_df(path)

        # Check paths exist
        paths = [
            target_tld_dir,
            cost_dir,
        ]

        for path in paths:
            if not os.path.exists(path):
                raise IOError(
                    "Generated path doesn't exist!\nGenerated the following "
                    "path that does not exist while building Gravity "
                    "Model arguments.\n %s"
                    % path
                )

        # Return the built arguments
        final_kwargs = self.kwargs.copy()
        final_kwargs.update({
            'productions': productions.to_df(),
            'attractions': attractions.to_df(),
            'running_segmentation': running_segmentation,
            'init_params': init_params,
            'target_tld_dir': target_tld_dir,
            'cost_dir': cost_dir,
            'cost_function': self.cost_function,
            'intrazonal_cost_infill': self.intrazonal_cost_infill,
            'pa_val_col': self.pa_val_col,
            'convergence_target': self.convergence_target,
            'fitting_loops': self.fitting_loops,
            'furness_max_iters': self.furness_max_iters,
            'furness_tol': self.furness_tol,
        })
        return final_kwargs

    def build_hb_arguments(self) -> Dict[str, nd.PathLike]:
        return self.build_arguments(trip_origin='hb')

    def build_nhb_arguments(self) -> Dict[str, nd.PathLike]:
        return self.build_arguments(trip_origin='nhb')


_GM_ExportPaths_NT = collections.namedtuple(
    typename='_GM_ExportPaths_NT',
    field_names=[
        'home',
        'distribution_dir',
    ]
)

_GM_ReportPaths_NT = collections.namedtuple(
    typename='_GM_ReportPaths_NT',
    field_names=[
        'home',
        'hb_overall_log',
        'nhb_overall_log',
        'model_log_dir',
        'tld_report_dir',
    ]
)


class GravityModelExportPaths:
    _reports_dirname = 'Logs & Reports'

    # Export dir names
    _dist_out_dir = 'Internal Matrices'

    # Report dir names
    _hb_overall_log_name = 'hb_overall_log.csv'
    _nhb_overall_log_name = 'nhb_overall_log.csv'
    _log_dir_name = 'Logs'
    _tld_report_dir = 'TLD Reports'

    def __init__(self,
                 year: int,
                 running_mode: nd.Mode,
                 export_home: nd.PathLike,
                 ):
        # Init
        file_ops.check_path_exists(export_home)

        # Assign attributes
        self.year = year
        self.running_mode = running_mode
        self.export_home = export_home
        self.report_home = os.path.join(self.export_home, self._reports_dirname)

        file_ops.create_folder(self.report_home)

        # Generate the paths
        self._create_export_paths()
        self._create_report_paths()

    def _create_export_paths(self) -> None:
        """Creates self.export_paths"""

        # Build the matrix output path
        distribution_dir = os.path.join(self.export_home, self._dist_out_dir)

        # Make paths that don't exist
        dir_paths = [distribution_dir]
        for path in dir_paths:
            file_ops.create_folder(path)

        # Create the export_paths class
        self.export_paths = _GM_ExportPaths_NT(
            home=self.export_home,
            distribution_dir=distribution_dir,
        )

    def _create_report_paths(self) -> None:
        """Creates self.report_paths"""
        # Build paths
        hb_overall_log = os.path.join(self.report_home, self._hb_overall_log_name)
        nhb_overall_log = os.path.join(self.report_home, self._nhb_overall_log_name)
        model_log_dir = os.path.join(self.report_home, self._log_dir_name)
        tld_report_dir = os.path.join(self.report_home, self._tld_report_dir)

        # Make paths that don't exist
        dir_paths = [self.report_home, model_log_dir, tld_report_dir]
        for path in dir_paths:
            file_ops.create_folder(path)

        # Create the export_paths class
        self.report_paths = _GM_ReportPaths_NT(
            home=self.report_home,
            hb_overall_log=hb_overall_log,
            nhb_overall_log=nhb_overall_log,
            model_log_dir=model_log_dir,
            tld_report_dir=tld_report_dir,
        )


# ## TMS CLASSES ## #
_TMS_ExportPaths_NT = collections.namedtuple(
    typename='_TMS_ExportPaths_NT',
    field_names=[
        'home',
        'full_pa_dir',
        'compiled_pa_dir',
        'full_od_dir',
        'compiled_od_dir',
        'compiled_od_dir_pcu',
    ]
)

_TMS_ReportPaths_NT = collections.namedtuple(
    typename='_TMS_ReportPaths_NT',
    field_names=[
        'home',
        'pa_reports_dir',
        'od_reports_dir',
    ]
)


class TMSExportPaths:

    # Define the names of the export dirs
    _external_model_dir = 'External Model'
    _gravity_model_dir = 'Gravity Model'
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
                 iteration_name: str,
                 running_mode: nd.core.enumerations.Mode,
                 export_home: nd.PathLike,
                 ):
        """
        Builds the export paths for all the TMS sub-models

        Parameters
        ----------
        iteration_name:
            The name of this iteration of the TMS models. Will have 'iter'
            pre-pended to create the folder name. e.g. if iteration_name was
            set to '3i' the iteration folder would be called 'iter3i'.

        export_home:
            The home directory of all the export paths. A sub-directory will
            be made for each of the TMS sub models and outputs.
        """
        # Init
        file_ops.check_path_exists(export_home)

        self.year = year
        self.iteration_name = du.create_iter_name(iteration_name)
        self.running_mode = running_mode
        self.export_home = os.path.join(export_home, self.iteration_name, self.running_mode.value)
        file_ops.create_folder(self.export_home)

        # ## BUILD ALL MODEL PATHS ## #
        # External Model
        em_export_home = os.path.join(self.export_home, self._external_model_dir)
        file_ops.create_folder(em_export_home)

        self.external_model = ExternalModelExportPaths(
            year=self.year,
            running_mode=self.running_mode,
            export_home=em_export_home,
        )

        # Gravity Model
        gm_export_home = os.path.join(self.export_home, self._gravity_model_dir)
        file_ops.create_folder(gm_export_home)

        self.gravity_model = GravityModelExportPaths(
            year=self.year,
            running_mode=self.running_mode,
            export_home=gm_export_home,
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
        """Creates self.export_paths"""

        # Build the matrix output path
        full_pa_dir = os.path.join(export_home, self._full_pa_out_dir)
        compiled_pa_dir = os.path.join(export_home, self.compiled_pa_out_dir)
        full_od_dir = os.path.join(export_home, self._full_od_out_dir)
        compiled_od_dir = os.path.join(export_home, self._compiled_od_out_dir)
        compiled_od_dir_pcu = os.path.join(compiled_od_dir, self._compiled_od_out_dir_pcu)

        # Make paths that don't exist
        dir_paths = [
            full_pa_dir,
            compiled_pa_dir,
            full_od_dir,
            compiled_od_dir,
            compiled_od_dir_pcu,
        ]
        for path in dir_paths:
            file_ops.create_folder(path)

        # Create the export_paths class
        self.export_paths = _TMS_ExportPaths_NT(
            home=export_home,
            full_pa_dir=full_pa_dir,
            compiled_pa_dir=compiled_pa_dir,
            full_od_dir=full_od_dir,
            compiled_od_dir=compiled_od_dir,
            compiled_od_dir_pcu=compiled_od_dir_pcu,
        )

    def _create_report_paths(self, report_home: str) -> None:
        """Creates self.report_paths"""

        # Create the export_paths class
        self.report_paths = _TMS_ReportPaths_NT(
            home=report_home,
            pa_reports_dir=os.path.join(report_home, self._pa_report_dir),
            od_reports_dir=os.path.join(report_home, self._od_report_dir),
        )

        # Make paths that don't exist
        for path in self.report_paths:
            file_ops.create_folder(path)


class TMSArgumentBuilder(TMSArgumentBuilderBase):
    """
    Builds an ExternalModelArgumentBuilder and GravityModelArgumentBuilder
    """

    # Folder constants
    _modal_dir_name = 'modal'
    _tour_props_dir_name = 'pre_me_tour_proportions'
    _fh_th_factors_dir_name = 'fh_th_factors'

    def __init__(self,
                 import_home: nd.PathLike,
                 running_mode: nd.core.enumerations.Mode,
                 hb_running_segmentation: nd.core.segments.SegmentationLevel,
                 nhb_running_segmentation: nd.core.segments.SegmentationLevel,
                 zoning_system: nd.core.zoning.ZoningSystem,
                 hb_cost_type: str,
                 nhb_cost_type: str,
                 hb_productions: nd.DVector,
                 hb_attractions: nd.DVector,
                 nhb_productions: nd.DVector,
                 nhb_attractions: nd.DVector,
                 intrazonal_cost_infill: float = 0.5,
                 run_external_model: bool = True,

                 # EM Specific
                 tms_exports: TMSExportPaths = None,

                 external_kwargs: Optional[Dict[str, Any]] = None,
                 gravity_kwargs: Optional[Dict[str, Any]] = None,
                 ):
        # Check paths exist
        file_ops.check_path_exists(import_home)

        # TODO(BT): Validate segments and zones are the correct types

        # Assign needed attributes
        self.import_home = import_home
        self.running_mode = running_mode
        self.hb_running_segmentation = hb_running_segmentation

        # Trip End Paths
        self._hb_productions = hb_productions
        self._hb_attractions = hb_attractions
        self._nhb_productions = nhb_productions
        self._nhb_attractions = nhb_attractions

        # Other attributes
        external_kwargs = dict() if external_kwargs is None else external_kwargs
        gravity_kwargs = dict() if gravity_kwargs is None else gravity_kwargs
        self._external_kwargs = external_kwargs
        self._gravity_kwargs = gravity_kwargs

        # Set up a dictionary of the arguments shared by both
        self._common_kwargs = {
            'import_home': import_home,
            'running_mode': running_mode,
            'hb_running_segmentation': hb_running_segmentation,
            'nhb_running_segmentation': nhb_running_segmentation,
            'zoning_system': zoning_system,
            'hb_cost_type': hb_cost_type,
            'nhb_cost_type': nhb_cost_type,
            'intrazonal_cost_infill': intrazonal_cost_infill,
        }

        # Pick the arguments based on whether we're running the external model
        if run_external_model:
            if tms_exports is None:
                raise ValueError(
                    "tms_exports must be set if run_external_model id True."
                )
            self._build_external_model_run(tms_exports)
        else:
            self._build_gravity_model_only_run()

    @property
    def external_model_arg_builder(self):
        return self._em_arg_builder

    @property
    def gravity_model_arg_builder(self):
        return self._gm_arg_builder

    def _build_external_model_run(self, tms_exports: TMSExportPaths) -> None:
        """Build arguments for an external+gravity run"""

        self._em_arg_builder = ExternalModelArgumentBuilder(
            **self._common_kwargs,
            **self._external_kwargs,
            hb_productions=self._hb_productions,
            hb_attractions=self._hb_attractions,
            nhb_productions=self._nhb_productions,
            nhb_attractions=self._nhb_attractions,
        )

        em_paths = tms_exports.external_model.export_paths
        self._gm_arg_builder = GravityModelArgumentBuilder(
            **self._common_kwargs,
            **self._gravity_kwargs,
            hb_productions=em_paths.hb_internal_productions,
            hb_attractions=em_paths.hb_internal_attractions,
            nhb_productions=em_paths.nhb_internal_productions,
            nhb_attractions=em_paths.nhb_internal_attractions,
        )

    def _build_gravity_model_only_run(self):
        """Build arguments for a gravity only run"""
        self._em_arg_builder = None

        self._gm_arg_builder = GravityModelArgumentBuilder(
            **self._common_kwargs,
            **self._gravity_kwargs,
            hb_productions=self._hb_productions,
            hb_attractions=self._hb_attractions,
            nhb_productions=self._nhb_productions,
            nhb_attractions=self._nhb_attractions,
        )

    def build_pa_to_od_arguments(self) -> Dict[str, Any]:

        # TODO(BT): UPDATE build_od_from_fh_th_factors() to use segmentation levels
        seg_level = 'tms'
        seg_params = {
            'p_needed': self.hb_running_segmentation.segments['p'].unique(),
            'm_needed': self.hb_running_segmentation.segments['m'].unique(),
        }
        if 'ca' in self.hb_running_segmentation.segment_names:
            seg_params.update({
                'ca_needed': self.hb_running_segmentation.segments['ca'].unique(),
            })

        # Build the factors dir
        fh_th_factors_dir = os.path.join(
            self.import_home,
            self._modal_dir_name,
            self.running_mode.value,
            self._tour_props_dir_name,
            self._fh_th_factors_dir_name,
        )

        return {
            'seg_level': seg_level,
            'seg_params': seg_params,
            'fh_th_factors_dir': fh_th_factors_dir,
        }


# ## FUNCTIONS ## #
def read_cjtw(file_path: nd.PathLike,
              zoning_name: str,
              subset=None,
              reduce_to_pa_factors: bool = True,
              ) -> pd.DataFrame:
    """
    This function imports census journey to work and converts types
    to ntem journey types

    Parameters
    ----------
    file_path:
        Takes a model folder to look for a cjtw zonal conversion

    zoning_name:
        The name of the zoning system the cjtw file is in

    subset:
        Takes a vector of model zones to filter by. Mostly for test model runs.

    reduce_to_pa_factors:
        ???

    Returns
    ----------
    census_journey_to_work:
        A census journey to work distribution in the required zonal format.
    """
    # TODO(BT, CS): Re-write this to be more generic
    # Init
    zoning_name = zoning_name.lower()

    # Read in the file
    if not os.path.isfile(file_path):
        raise ValueError("No file exists at %s" % file_path)
    cjtw = pd.read_csv(file_path)

    # CTrip End Categories
    # 1 Walk
    # 2 Cycle
    # 3 Car driver
    # 4 Car passenger
    # 5 Bus
    # 6 Rail / underground

    if subset is not None:
        sub_col = list(subset)
        sub_zones = subset[sub_col].squeeze()
        cjtw = cjtw[cjtw['1_' + zoning_name + 'Areaofresidence'].isin(sub_zones)]
        cjtw = cjtw[cjtw['2_' + zoning_name + 'Areaofworkplace'].isin(sub_zones)]

    method_to_mode = {'4_Workmainlyatorfromhome': '1_walk',
                      '5_Undergroundmetrolightrailtram': '6_rail_ug',
                      '6_Train': '6_rail_ug',
                      '7_Busminibusorcoach': '5_bus',
                      '8_Taxi': '3_car',
                      '9_Motorcyclescooterormoped': '2_cycle',
                      '10_Drivingacarorvan': '3_car',
                      '11_Passengerinacarorvan': '3_car',
                      '12_Bicycle': '2_cycle',
                      '13_Onfoot': '1_walk',
                      '14_Othermethodoftraveltowork': '1_walk'}
    mode_cols = list(method_to_mode.keys())

    for col in mode_cols:
        cjtw = cjtw.rename(columns={col: method_to_mode.get(col)})

    cjtw = cjtw.drop('3_Allcategories_Methodoftraveltowork', axis=1)
    cjtw = cjtw.groupby(cjtw.columns, axis=1).sum()
    cjtw = cjtw.reindex(['1_' + zoning_name + 'Areaofresidence',
                         '2_' + zoning_name + 'Areaofworkplace',
                         '1_walk', '2_cycle', '3_car',
                         '5_bus', '6_rail_ug'], axis=1)

    # Pivot
    cjtw = pd.melt(
        cjtw,
        id_vars=['1_' + zoning_name + 'Areaofresidence', '2_' + zoning_name + 'Areaofworkplace'],
        var_name='mode',
        value_name='trips',
    )
    cjtw['mode'] = cjtw['mode'].str[0]
    cjtw['mode'] = cjtw['mode'].astype(int)

    # Build distribution factors
    hb_totals = cjtw.drop('2_' + zoning_name + 'Areaofworkplace', axis=1)
    hb_totals = hb_totals.groupby(['1_' + zoning_name + 'Areaofresidence', 'mode'])
    hb_totals = hb_totals.sum().reset_index()

    hb_totals = hb_totals.rename(columns={'trips': 'zonal_mode_total_trips'})
    hb_totals = hb_totals.reindex(
        ['1_' + zoning_name + 'Areaofresidence', 'mode', 'zonal_mode_total_trips'],
        axis=1
    )

    cjtw = cjtw.merge(
        hb_totals,
        how='left',
        on=['1_' + zoning_name + 'Areaofresidence', 'mode'],
    )

    # Divide by total trips to get distribution factors
    if reduce_to_pa_factors:
        cjtw['distribution'] = cjtw['trips'] / cjtw['zonal_mode_total_trips']
        cjtw = cjtw.drop(['trips', 'zonal_mode_total_trips'], axis=1)
    else:
        cjtw = cjtw.drop(['zonal_mode_total_trips'], axis=1)

    return cjtw
