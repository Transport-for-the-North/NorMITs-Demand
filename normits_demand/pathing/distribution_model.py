# -*- coding: utf-8 -*-
"""
Created on: 07/12/2021
Updated on:

Original author: Ben Taylor
Last update made by:
Other updates made by:

File purpose:

"""
# Built-Ins
import os
import abc
import collections

# Third Party

# Local Imports
import normits_demand as nd

from normits_demand.utils import file_ops
from normits_demand.utils import general as du

# ## DEFINE COLLECTIONS OF OUTPUT PATHS ## #
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


# ## DEFINE IMPORT PATHS ## #
class DMArgumentBuilderBase(abc.ABC):
    """Abstract Class defining how the argument builder for TMS should look.

    If custom import paths are needed, then a new class needs to be made
    which inherits this abstract class. TMS can then use the defined
    functions/properties to pick up new import files.
    """

    # @property
    # @abc.abstractmethod
    # def external_model_arg_builder(self) -> ExternalModelArgumentBuilderBase:
    #     pass
    #
    # @property
    # @abc.abstractmethod
    # def gravity_model_arg_builder(self) -> GravityModelArgumentBuilderBase:
    #     pass
    #
    # @abc.abstractmethod
    # def build_pa_to_od_arguments(self) -> Dict[str, Any]:
    #     pass

# ## DEFINE EXPORT PATHS ##
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
                 iteration_name: str,
                 running_mode: nd.Mode,
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
        self.iteration_name = du.create_iter_name(iteration_name)
        self.running_mode = running_mode
        self.export_home = os.path.join(export_home, self.iteration_name, self.running_mode.value)
        file_ops.create_folder(self.export_home)

        # ## BUILD ALL MODEL PATHS ## #
        # Upper Model
        upper_export_home = os.path.join(self.export_home, self._upper_model_dir)
        file_ops.create_folder(upper_export_home)

        # Lower Model
        lower_export_home = os.path.join(self.export_home, self._lower_model_dir)
        file_ops.create_folder(lower_export_home)

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
