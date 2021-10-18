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

from typing import Dict
from typing import Optional

# Third Party
import numpy as np
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


# ## EXTERNAL MODEL CLASSES ## #
class ExternalModelArgumentBuilder(ExternalModelArgumentBuilderBase):
    # Costs constants
    _cost_dir_name = 'costs'
    _cost_base_fname = "{zoning_name}_{cost_type}_costs.csv"

    # CJTW constants
    _cjtw_infill = 0.1
    _cjtw_dir_name = 'cjtw'
    _cjtw_base_fname = 'cjtw_{zoning_name}.csv'

    # Trip Length Distribution constants
    _tld_dir_name = 'trip_length_distributions'
    _tld_area_dir_name = 'gb'

    def __init__(self,
                 import_home: nd.PathLike,
                 base_year: int,
                 scenario: str,
                 running_mode: nd.core.enumerations.Mode,
                 zoning_system: nd.core.zoning.ZoningSystem,
                 internal_tld_name: str,
                 external_tld_name: str,
                 hb_cost_type: str,
                 nhb_cost_type: str,
                 notem_iteration_name: str,
                 notem_export_home: str,
                 cache_path: nd.PathLike = None,
                 intrazonal_cost_infill: float = 0.5,
                 convergence_target: float = 0.9,
                 furness_tol: float = 0.1,
                 furness_max_iters: int = 5000,
                 **kwargs,
                 ):
        # Check paths exist
        file_ops.check_path_exists(import_home)

        # TODO(BT): Validate segments and zones are the correct types

        # Assign attributes
        self.import_home = import_home
        self.base_year = base_year
        self.running_mode = running_mode
        self.zoning_system = zoning_system
        self.internal_tld_name = internal_tld_name
        self.external_tld_name = external_tld_name
        self.hb_cost_type = hb_cost_type
        self.nhb_cost_type = nhb_cost_type
        self.cache_path = cache_path
        self.intrazonal_cost_infill = intrazonal_cost_infill
        self.convergence_target = convergence_target
        self.furness_tol = furness_tol
        self.furness_max_iters = furness_max_iters
        self.kwargs = kwargs

        # Generate the NoTEM export paths
        self.notem_exports = nd.pathing.NoTEMExportPaths(
            path_years=[base_year],
            scenario=scenario,
            iteration_name=notem_iteration_name,
            export_home=notem_export_home,
        )

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
        exports = self.notem_exports
        if trip_origin == 'hb':
            productions_path = exports.hb_production.export_paths.notem_segmented[self.base_year]
            attractions_path = exports.hb_attraction.export_paths.notem_segmented[self.base_year]
            cost_type = self.hb_cost_type
        elif trip_origin == 'nhb':
            productions_path = exports.nhb_production.export_paths.notem_segmented[self.base_year]
            attractions_path = exports.nhb_attraction.export_paths.notem_segmented[self.base_year]
            cost_type = self.nhb_cost_type
        else:
            raise ValueError(
                "Received an unexpected value for trip origin. Expected one of "
                "'hb' or 'nhb'. Got %s"
                % trip_origin
            )

        # read in productions and attractions
        # TODO(BT): Tidy up this function - no cache!
        productions, attractions = import_pa(
            production_import_path=productions_path,
            attraction_import_path=attractions_path,
            model_zone=self.zoning_system.name,
            trip_origin=trip_origin,
            cache_path=self.cache_path,
        )

        # Build TLD directory paths
        base_tld_path = os.path.join(
            self.import_home,
            self._tld_dir_name,
            self._tld_area_dir_name,
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
            'productions': productions,
            'attractions': attractions,
            'seed_matrix': self._read_and_convert_cjtw(),
            'costs_path': costs_path,
            'internal_tld_dir': internal_tld_path,
            'external_tld_dir': external_tld_path,
            'intrazonal_cost_infill': self.intrazonal_cost_infill,
            'convergence_target': self.convergence_target,
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

    _int_productions_base_name = '{trip_origin}_{mode}_{year}_internal_productions.csv'
    _int_attractions_base_name = '{trip_origin}_{mode}_{year}_internal_attractions.csv'

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
    _cost_dir_name = 'costs'
    _cost_base_fname = "{zoning_name}_{cost_type}_costs.csv"

    # Initial Parameters consts
    _gravity_model_dir = 'gravity model'

    # Trip Length Distribution constants
    _tld_dir_name = 'trip_length_distributions'
    _tld_area_dir_name = 'gb'

    def __init__(self,
                 import_home: nd.PathLike,
                 target_tld_name: str,
                 cost_function: str,
                 running_mode: nd.core.enumerations.Mode,
                 zoning_system: nd.core.zoning.ZoningSystem,
                 hb_cost_type: str,
                 nhb_cost_type: str,
                 hb_init_params_fname: str,
                 nhb_init_params_fname: str,
                 external_model_exports: ExternalModelExportPaths,
                 intrazonal_cost_infill: float = 0.5,
                 pa_val_col: Optional[str] = 'val',
                 apply_k_factoring: Optional[bool] = True,
                 convergence_target: Optional[float] = 0.95,
                 fitting_loops: Optional[int] = 100,
                 furness_max_iters: Optional[int] = 2000,
                 furness_tol: Optional[float] = 1.0,
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
        self.zoning_system = zoning_system
        self.hb_cost_type = hb_cost_type
        self.nhb_cost_type = nhb_cost_type
        self.hb_init_params_fname = hb_init_params_fname
        self.nhb_init_params_fname = nhb_init_params_fname
        self.external_model_exports = external_model_exports

        self.intrazonal_cost_infill = intrazonal_cost_infill
        self.pa_val_col = pa_val_col
        self.apply_k_factoring = apply_k_factoring
        self.convergence_target = convergence_target
        self.fitting_loops = fitting_loops
        self.furness_max_iters = furness_max_iters
        self.furness_tol = furness_tol
        self.kwargs = kwargs

    def build_arguments(self, trip_origin: str) -> Dict[str, nd.PathLike]:
        # Init
        trip_origin = trip_origin.lower()

        # Set trip origin specific stuff
        exports = self.external_model_exports
        if trip_origin == 'hb':
            productions_path = exports.export_paths.hb_internal_productions
            attractions_path = exports.export_paths.hb_internal_attractions
            cost_type = self.hb_cost_type
            init_params_fname = self.hb_init_params_fname
        elif trip_origin == 'nhb':
            productions_path = exports.export_paths.nhb_internal_productions
            attractions_path = exports.export_paths.nhb_internal_attractions
            cost_type = self.nhb_cost_type
            init_params_fname = self.nhb_init_params_fname
        else:
            raise ValueError(
                "Received an unexpected value for trip origin. Expected one of "
                "'hb' or 'nhb'. Got %s"
                % trip_origin
            )

        # Make sure the productions and attractions exist
        if not os.path.isfile(productions_path):
            raise nd.NormitsDemandError(
                "Cannot find any internal productions at '%s'.\n"
                "Has the External Model been run?"
                % productions_path
            )

        if not os.path.isfile(attractions_path):
            raise nd.NormitsDemandError(
                "Cannot find any internal productions at '%s'.\n"
                "Has the External Model been run?"
                % attractions_path
            )

        # Build TLD directory paths
        base_tld_path = os.path.join(
            self.import_home,
            self._tld_dir_name,
            self._tld_area_dir_name,
        )
        target_tld_dir = os.path.join(base_tld_path, self.target_tld_name)

        # Build costs path
        fname = self._cost_base_fname.format(
            zoning_name=self.zoning_system.name,
            cost_type=cost_type,
        )
        costs_path = os.path.join(
            self.import_home,
            self.running_mode.value,
            self._cost_dir_name,
            fname,
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
            costs_path,
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
            'productions': file_ops.read_df(productions_path),
            'attractions': file_ops.read_df(attractions_path),
            'init_params': init_params,
            'target_tld_dir': target_tld_dir,
            'costs_path': costs_path,
            'cost_function': self.cost_function,
            'intrazonal_cost_infill': self.intrazonal_cost_infill,
            'pa_val_col': self.pa_val_col,
            'apply_k_factoring': self.apply_k_factoring,
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
        'model_log_dir',
        'tld_report_dir',
    ]
)


class GravityModelExportPaths:
    _reports_dirname = 'Logs & Reports'

    # Export dir names
    _dist_out_dir = 'Internal Matrices'

    # Report dir names
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
        self.export_home = export_home  # Something like I:\NorMITs Demand\noham\TMS\iter8\Gravity Model
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

        # Create the export_paths class
        self.report_paths = _GM_ReportPaths_NT(
            home=self.report_home,
            model_log_dir=os.path.join(self.report_home, self._log_dir_name),
            tld_report_dir=os.path.join(self.report_home, self._tld_report_dir),
        )

        # Make paths that don't exist
        for path in self.report_paths:
            file_ops.create_folder(path)


# ## TMS CLASSES ## #
_TMS_ExportPaths_NT = collections.namedtuple(
    typename='_TMS_ExportPaths_NT',
    field_names=[
        'home',
        'full_pa_dir',
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
        full_od_dir = os.path.join(export_home, self._full_od_out_dir)
        compiled_od_dir = os.path.join(export_home, self._compiled_od_out_dir)
        compiled_od_dir_pcu = os.path.join(compiled_od_dir, self._compiled_od_out_dir_pcu)

        # Make paths that don't exist
        dir_paths = [full_pa_dir, full_od_dir, compiled_od_dir, compiled_od_dir_pcu]
        for path in dir_paths:
            file_ops.create_folder(path)

        # Create the export_paths class
        self.export_paths = _TMS_ExportPaths_NT(
            home=export_home,
            full_pa_dir=full_pa_dir,
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


# ## FUNCTIONS ## #
def import_pa(production_import_path,
              attraction_import_path,
              model_zone,
              trip_origin,
              cache_path=None,
              ):
    """
    This function imports productions and attractions from given paths.

    Parameters
    ----------
    production_import_path:
        Path to import productions from.

    attraction_import_path:
        Path to import attractions from.

    model_zone:
        Type of model zoning system. norms or noham

    Returns
    ----------
    [0] productions:
        Mainland GB productions.

    [1] attractions:
        Mainland GB attractions.
    """
    # Init
    p_cache = None
    a_cache = None

    # handle cache
    if cache_path is not None:
        p_fname = "%s_%s_productions.csv" % (trip_origin, model_zone)
        a_fname = "%s_%s_attractions.csv" % (trip_origin, model_zone)

        p_cache = os.path.join(cache_path, p_fname)
        a_cache = os.path.join(cache_path, a_fname)

        if os.path.exists(p_cache) and os.path.exists(a_cache):
            return pd.read_csv(p_cache), pd.read_csv(a_cache)

    # Reading pickled Dvector
    prod_dvec = nd.from_pickle(production_import_path)

    # Aggregate to the required segmentation
    if trip_origin == 'hb':
        if model_zone == 'noham':
            agg_seg = nd.get_segmentation_level('hb_p_m_6tp')
        elif model_zone == 'norms':
            agg_seg = nd.get_segmentation_level('hb_p_m_ca_6tp')
        else:
            raise ValueError("Invalid model name")
    elif trip_origin == 'nhb':
        if model_zone == 'noham':
            agg_seg = nd.get_segmentation_level('nhb_p_m_6tp')
        elif model_zone == 'norms':
            agg_seg = nd.get_segmentation_level('nhb_p_m_ca_6tp')
        else:
            raise ValueError("Invalid model name")
    else:
        raise ValueError("Invalid trip origin")

    # Aggregate and translate for norms/noham
    prod_dvec_agg = prod_dvec.aggregate(out_segmentation=agg_seg)
    model_zoning = nd.get_zoning_system(model_zone)
    prod_dvec = prod_dvec_agg.translate_zoning(model_zoning, "population")

    # Weekly trips to weekday trips conversion
    prod_df = prod_dvec.to_df()
    prod_wd = weekly_to_weekday(prod_df, trip_origin, model_zone)

    # Reading pickled Dvector
    attr_dvec = nd.from_pickle(attraction_import_path)

    # Aggregate and translate for norms/noham
    attr_dvec_agg = attr_dvec.aggregate(out_segmentation=agg_seg)
    model_zoning = nd.get_zoning_system(model_zone)
    attr_dvec = attr_dvec_agg.translate_zoning(model_zoning, "employment")

    # Weekly trips to weekday trips conversion
    attr_df = attr_dvec.to_df()
    attr_wd = weekly_to_weekday(attr_df, trip_origin, model_zone)

    # TODO(BT): Sort zoning system into order
    if p_cache is not None and a_cache is not None:
        prod_wd.to_csv(p_cache, index=False)
        attr_wd.to_csv(a_cache, index=False)

    return prod_wd, attr_wd


def weekly_to_weekday(df, trip_origin, model_zone) -> pd.DataFrame:
    """
    Convert weekly trips to weekday trips.

    Removes tp5 and tp6 from the time period column and
    divides trips by 5 to convert them from weekly to weekday.

    Parameters
    ----------
    df:
    Dataframe (either productions or attractions) containing notem segmented weekly trips.

    trip_origin:
    Whether the trip origin is hb or nhb.

    Return
    ----------
    df:
    Dataframe (either productions or attractions) containing notem segmented weekday trips.
    """
    if model_zone == 'norms':
        df[["p", "m", "ca", "tp"]] = df[["p", "m", "ca", "tp"]].apply(pd.to_numeric)
    else:
        df[["p", "m", "tp"]] = df[["p", "m", "tp"]].apply(pd.to_numeric)
    df = df.drop(df[df.tp >= 5].index)
    df['val'] = df['val'] / 5
    df_index_cols = list(df)
    df_index_cols.remove('tp')
    df_group_cols = df_index_cols.copy()
    df_group_cols.remove('val')

    # Time period removed for hb based trips
    if trip_origin == 'hb':
        df = df.reindex(df_index_cols, axis=1).groupby(df_group_cols).sum().reset_index()
    return df


def read_cjtw(file_path: nd.PathLike,
              zoning_name: str,
              subset: bool = None,
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
    modeCols = list(method_to_mode.keys())

    for col in modeCols:
        cjtw = cjtw.rename(columns={col: method_to_mode.get(col)})

    cjtw = cjtw.drop('3_Allcategories_Methodoftraveltowork', axis=1)
    cjtw = cjtw.groupby(cjtw.columns, axis=1).sum()
    cjtw = cjtw.reindex(['1_' + zoning_name + 'Areaofresidence',
                         '2_' + zoning_name + 'Areaofworkplace',
                         '1_walk', '2_cycle', '3_car',
                         '5_bus', '6_rail_ug'], axis=1)
    # Redefine mode cols for new aggregated modes
    modeCols = ['1_walk', '2_cycle', '3_car', '5_bus', '6_rail_ug']
    # Pivot
    cjtw = pd.melt(cjtw, id_vars=['1_' + zoning_name + 'Areaofresidence',
                                  '2_' + zoning_name + 'Areaofworkplace'],
                   var_name='mode', value_name='trips')
    cjtw['mode'] = cjtw['mode'].str[0]

    # Build distribution factors
    hb_totals = cjtw.drop(
        '2_' + zoning_name + 'Areaofworkplace',
        axis=1
    ).groupby(
        ['1_' + zoning_name + 'Areaofresidence', 'mode']
    ).sum().reset_index()

    hb_totals = hb_totals.rename(columns={'trips': 'zonal_mode_total_trips'})
    hb_totals = hb_totals.reindex(
        ['1_' + zoning_name + 'Areaofresidence', 'mode', 'zonal_mode_total_trips'],
        axis=1
    )

    cjtw = cjtw.merge(hb_totals,
                      how='left',
                      on=['1_' + zoning_name + 'Areaofresidence', 'mode'])

    # Divide by total trips to get distribution factors

    if reduce_to_pa_factors:
        cjtw['distribution'] = cjtw['trips'] / cjtw['zonal_mode_total_trips']
        cjtw = cjtw.drop(['trips', 'zonal_mode_total_trips'], axis=1)
    else:
        cjtw = cjtw.drop(['zonal_mode_total_trips'], axis=1)

    return cjtw
