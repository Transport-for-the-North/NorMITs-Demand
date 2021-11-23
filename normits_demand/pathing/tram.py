# -*- coding: utf-8 -*-
"""
Created on: Mon November 18 2021
Updated on: Tue November 23 2021

Original author: Nirmal Kumar
Last update made by: Ben Taylor
Other updates made by:

File purpose:
Classes which build all the paths for Tram model inputs and outputs
"""
# Builtins
import os
import collections
from abc import ABC
from abc import abstractmethod

from typing import List
from typing import Dict
from typing import Tuple

# Third Party

# Local imports
import normits_demand as nd

from normits_demand.pathing import NoTEMExportPaths

from normits_demand.utils import file_ops
from normits_demand.utils import general as du


class TramImportPathsBase(ABC):
    """Abstract Class defining how the import paths class for Tram should look.

    If custom import paths are needed, then a new class needs to be made
    which inherits this abstract class. Tram can then use the defined
    functions to pick up new import files.
    """

    @abstractmethod
    def generate_hb_production_imports(self) -> Dict[str, nd.PathLike]:
        """
        Definition of function that needs to be overwritten

        Returns
        -------
        kwargs:
            This function should return a keyword argument dictionary to be
            passed straight into the production model. Specifically, this
            function needs to produce a dictionary with the following keys:
                'tram_import'
                'trip_origin'
                'dvec_imports'
        """
        raise NotImplementedError(
            "generate_hb_production_imports() has not been implemented! If "
            "you have inherited this class, you need to make sure this "
            "function is overwritten by the child class.\n"
            "See the method documentation for information on how the function "
            "call and return should look."
        )

    @abstractmethod
    def generate_nhb_production_imports(self) -> Dict[str, nd.PathLike]:
        """
        Definition of function that needs to be overwritten

        Returns
        -------
        kwargs:
            This function should return a keyword argument dictionary to be
            passed straight into the production model. Specifically, this
            function needs to produce a dictionary with the following keys:
                'tram_import'
                'trip_origin'
                'dvec_imports'
        """
        raise NotImplementedError(
            "generate_nhb_production_imports() has not been implemented! If "
            "you have inherited this class, you need to make sure this "
            "function is overwritten by the child class.\n"
            "See the method documentation for information on how the function "
            "call and return should look."
        )

    @abstractmethod
    def generate_hb_attraction_imports(self) -> Dict[str, nd.PathLike]:
        """
        Definition of function that needs to be overwritten

        Returns
        -------
        kwargs:
            This function should return a keyword argument dictionary to be
            passed straight into the attraction model. Specifically, this
            function needs to produce a dictionary with the following keys:
                'tram_import'
                'trip_origin'
                'dvec_imports'
        """
        raise NotImplementedError(
            "generate_hb_attraction_imports() has not been implemented! If "
            "you have inherited this class, you need to make sure this "
            "function is overwritten by the child class.\n"
            "See the method documentation for information on how the function "
            "call and return should look."
        )

    @abstractmethod
    def generate_nhb_attraction_imports(self) -> Dict[str, nd.PathLike]:
        """
        Definition of function that needs to be overwritten

        Returns
        -------
        kwargs:
            This function should return a keyword argument dictionary to be
            passed straight into the attraction model. Specifically, this
            function needs to produce a dictionary with the following keys:
                'tram_import'
                'trip_origin'
                'dvec_imports'
        """
        raise NotImplementedError(
            "generate_nhb_attraction_imports() has not been implemented! If "
            "you have inherited this class, you need to make sure this "
            "function is overwritten by the child class.\n"
            "See the method documentation for information on how the function "
            "call and return should look."
        )


class TramImportPaths(TramImportPathsBase):
    """The default Tram Import paths class.

    Defines the default input paths for Tram. All Attributes are as
    passed in to the constructor.

    Attributes
    ----------
    tram_import_home:
            The home for all of the tram imports.

    notem_import_home:
        The home for all of the notem outputs.

    scenario:
        The name of the scenario to run for.

    years:
        List of years to run Tram for. Will assume that the smallest
        year is the base year.

    hb_prod_tram_import_version:
        The version of inputs to use for the HB production tram model.
        e.g. '2.0'

    hb_attr_tram_import_version:
        The version of inputs to use for the HB attraction tram model.
        e.g. '2.0'

    nhb_production_data_version:
        The version of inputs to use for the NHB production tram model.
        e.g. '2.0'

    nhb_attraction_data_version:
        The version of inputs to use for the NHB attraction tram model.
        e.g. '2.0'
    """
    # Constant
    _current_base_year = 2018
    _hb_trip_origin = 'hb'
    _nhb_trip_origin = 'nhb'

    # Define input notem fnames
    _hb_notem_fname = "hb_msoa_notem_segmented_%d_dvec.pkl"
    _nhb_notem_fname = "nhb_msoa_notem_segmented_%d_dvec.pkl"

    # HB Productions
    _hb_productions_dir = "hb_productions"
    _hbp_tram_fname = "tram_hb_productions_v{version}.csv"

    # HB Attractions
    _hb_attractions_dir = "hb_attractions"
    _hba_tram_fname = "tram_hb_attractions_v{version}.csv"

    # NHB Productions
    _nhb_productions_dir = "nhb_productions"
    _nhbp_tram_fname = "tram_nhb_productions_v{version}.csv"

    # NHB Attractions
    _nhb_attractions_dir = "nhb_attractions"
    _nhba_tram_fname = "tram_nhb_attractions_v{version}.csv"

    def __init__(self,
                 years: List[int],
                 notem_exports: NoTEMExportPaths,
                 tram_import_home: nd.PathLike,
                 hb_production_data_version: str,
                 hb_attraction_data_version: str,
                 nhb_production_data_version: str,
                 nhb_attraction_data_version: str,
                 ):
        """
        Assigns and validates attributes.

        Parameters
        ----------
        tram_import_home:
            The home for all of the tram imports.

        notem_exports:
            An instance of NoTEMExportPaths. Used to generate the paths to
            the NoTEM Ouputs to use as inputs for the Tram Model

        years:
            List of years to run Tram for. Will assume that the smallest
            year is the base year.

        hb_production_data_version:
            The version of inputs to use for the HB production tram model.
            e.g. '2.0'

        hb_attraction_data_version:
            The version of inputs to use for the HB attraction tram model.
            e.g. '2.0'

        nhb_production_data_version:
            The version of inputs to use for the NHB production tram model.
            e.g. '2.0'

        nhb_attraction_data_version:
            The version of inputs to use for the NHB attraction tram model.
            e.g. '2.0'
        """

        # Validate inputs
        file_ops.check_path_exists(tram_import_home)

        # Assign attributes
        self.years = years
        self.notem_exports = notem_exports
        self.tram_import_home = tram_import_home

        self.hb_production_data_version = hb_production_data_version
        self.hb_attraction_data_version = hb_attraction_data_version
        self.nhb_production_data_version = nhb_production_data_version
        self.nhb_attraction_data_version = nhb_attraction_data_version

    def generate_hb_production_imports(self) -> Dict[str, nd.PathLike]:
        """
        Generates input paths for home based production tram model.

        See TramImportPathsBase.generate_hb_production_imports() for further
        documentation
        """
        # Generate the tram data fname
        tram_data = self._hbp_tram_fname.format(version=self.hb_production_data_version)

        # Get the NoTEM paths
        hbp_paths = self.notem_exports.hb_production
        hb_production_paths = {x: hbp_paths.export_paths.notem_segmented[x] for x in self.years}

        # Format in dictionary
        return {
            'tram_import': os.path.join(self.tram_import_home, tram_data),
            'trip_origin': self._hb_trip_origin,
            'dvec_import_paths': hb_production_paths,
        }

    def generate_hb_attraction_imports(self) -> Dict[str, nd.PathLike]:
        """
        Generates input paths for home based attraction tram model

        See TramImportPathsBase.generate_hb_attraction_imports() for further
        documentation
        """
        # Generate the tram data fname
        tram_data = self._hba_tram_fname.format(version=self.hb_attraction_data_version)

        # Get the NoTEM paths
        hba_paths = self.notem_exports.hb_attraction
        hb_attraction_paths = {x: hba_paths.export_paths.notem_segmented[x] for x in self.years}

        # Format in dictionary
        return {
            'tram_import': os.path.join(self.tram_import_home, tram_data),
            'trip_origin': self._hb_trip_origin,
            'dvec_import_paths': hb_attraction_paths,
        }

    def generate_nhb_production_imports(self) -> Dict[str, nd.PathLike]:
        """
        Generates input paths for non-home based production tram model

        See TramImportPathsBase.generate_nhb_production_imports() for further
        documentation
        """
        # Generate the tram data fname
        tram_data = self._nhbp_tram_fname.format(version=self.nhb_production_data_version)

        # Get the NoTEM paths
        nhbp_paths = self.notem_exports.hb_production
        nhb_production_paths = {x: nhbp_paths.export_paths.notem_segmented[x] for x in self.years}

        # Format in dictionary
        return {
            'tram_import': os.path.join(self.tram_import_home, tram_data),
            'trip_origin': self._nhb_trip_origin,
            'dvec_import_paths': nhb_production_paths,
        }

    def generate_nhb_attraction_imports(self) -> Dict[str, nd.PathLike]:
        """
        Generates input paths for non-home based attraction tram model

        Note that no import paths are currently needed for this model

        See TramImportPathsBase.generate_hb_attraction_imports() for further
        documentation
        """
        # Generate the tram data fname
        tram_data = self._nhba_tram_fname.format(version=self.nhb_attraction_data_version)

        # Get the NoTEM paths
        nhba_paths = self.notem_exports.hb_attraction
        nhb_attraction_paths = {x: nhba_paths.export_paths.notem_segmented[x] for x in self.years}

        # Format in dictionary
        return {
            'tram_import': os.path.join(self.tram_import_home, tram_data),
            'trip_origin': self._nhb_trip_origin,
            'dvec_import_paths': nhb_attraction_paths,
        }


class TramExportPaths:
    """Path Class for the Tram Model.

    This class defines and builds the export and reporting paths for
    all Tram sub-models. It creates and stores an instance of:
    HBProductionModelPaths, NHBProductionModelPaths,
    HBAttractionModelPaths, NHBAttractionModelPaths.
    If the outputs of Tram are needed, create an instance of this
    class to generate all paths.

    Attributes
    ----------
    path_years:
        A list of the years the models are running for. As passed into the
        constructor.

    scenario:
        The name of the scenario to run for. As passed in to the constructor.

    iteration_name:
        The name of this iteration of the NoTEM models. Constructor argument
        of the same name will have 'iter' prepended to create this name.
        e.g. if '9.2' was passed in, this would become 'iter9.2'.

    export_home:
        The home directory of all the export paths. Nested folder of the
        passed in export_home, iteration_name, and scenario

    hb_production:
        An instance of HBProductionModelPaths. See docs for more info on how
        to access paths.

    nhb_production:
        An instance of NHBProductionModelPaths. See docs for more info on how
        to access paths.

    hb_attraction:
        An instance of HBAttractionModelPaths. See docs for more info on how
        to access paths.

    nhb_attraction:
        An instance of NHBAttractionModelPaths. See docs for more info on how
        to access paths.
    """

    # Define the names of the export dirs
    _hb_productions_dir = 'hb_productions'
    _nhb_productions_dir = 'nhb_productions'
    _hb_attractions_dir = 'hb_attractions'
    _nhb_attractions_dir = 'nhb_attractions'

    _reports_dir = 'reports'

    def __init__(self,
                 path_years: List[int],
                 scenario: str,
                 iteration_name: str,
                 export_home: nd.PathLike,
                 ):
        """
        Builds the export paths for all the Tram sub-models

        Parameters
        ----------
        path_years:
            A list of the years the models are running for.

        scenario:
            The name of the scenario to run for.

        iteration_name:
            The name of this iteration of the NoTEM models. Will have 'iter'
            prepended to create the folder name. e.g. if iteration_name was
            set to '9.2' the iteration folder would be called 'iter9.2'.

        export_home:
            The home directory of all the export paths. A sub-directory will
            be made for each of the Tram sub models.
        """
        # Init
        file_ops.check_path_exists(export_home)

        self.path_years = path_years
        self.scenario = scenario
        self.iteration_name = du.create_iter_name(iteration_name)
        self.export_home = os.path.join(export_home, self.iteration_name, self.scenario)
        file_ops.create_folder(self.export_home)

        # ## BUILD ALL MODEL PATHS ## #
        # hb productions
        hb_p_export_home = os.path.join(self.export_home, self._hb_productions_dir)
        hb_p_report_home = os.path.join(hb_p_export_home, self._reports_dir)
        file_ops.create_folder(hb_p_export_home)
        file_ops.create_folder(hb_p_report_home)

        self.hb_production = HBProductionModelPaths(
            path_years=path_years,
            export_home=hb_p_export_home,
            report_home=hb_p_report_home,
        )

        # nhb productions
        nhb_p_export_home = os.path.join(self.export_home, self._nhb_productions_dir)
        nhb_p_report_home = os.path.join(nhb_p_export_home, self._reports_dir)
        file_ops.create_folder(nhb_p_export_home)
        file_ops.create_folder(nhb_p_report_home)

        self.nhb_production = NHBProductionModelPaths(
            path_years=path_years,
            export_home=nhb_p_export_home,
            report_home=nhb_p_report_home,
        )

        # hb attractions
        hb_a_export_home = os.path.join(self.export_home, self._hb_attractions_dir)
        hb_a_report_home = os.path.join(hb_a_export_home, self._reports_dir)
        file_ops.create_folder(hb_a_export_home)
        file_ops.create_folder(hb_a_report_home)

        self.hb_attraction = HBAttractionModelPaths(
            path_years=path_years,
            export_home=hb_a_export_home,
            report_home=hb_a_report_home,
        )

        # nhb attractions
        nhb_a_export_home = os.path.join(self.export_home, self._nhb_attractions_dir)
        nhb_a_report_home = os.path.join(nhb_a_export_home, self._reports_dir)
        file_ops.create_folder(nhb_a_export_home)
        file_ops.create_folder(nhb_a_report_home)

        self.nhb_attraction = NHBAttractionModelPaths(
            path_years=path_years,
            export_home=nhb_a_export_home,
            report_home=nhb_a_report_home,
        )


class TramModelPaths:
    """Base Path Class for all Tram models.

    This class forms the base path class that all Tram model path classes
    are built off of. It defines a number of constants to ensure all
    Tram models follow the same output structure and naming conventions

    Attributes
    ----------
    path_years: List[int]
        A list of years that paths will be generated for.

    export_home: nd.PathLike
        The home directory of all exports. Is used as a basis for
        all export path building.

    report_home: nd.PathLike
        The home directory of all reports. Is used as a basis for
        all report path building.
    """
    # Export fname params
    _trip_origin = None
    _zoning_system = 'msoa'

    # Segmentation names
    _notem_segmented = 'notem_segmented'

    # Report names
    _segment_totals_report_name = "segment_totals"
    _ca_sector_report_name = "ca_sector_totals"
    _ie_sector_report_name = "ie_sector_totals"

    # Output Path Classes
    ExportPaths = collections.namedtuple(
        typename='ExportPaths',
        field_names='home, notem_segmented',
    )

    ReportPaths = collections.namedtuple(
        typename='ReportPaths',
        field_names='segment_total, ca_sector, ie_sector',
    )

    # Define output fnames
    _base_output_fname = '%s_%s_%s_%d_dvec.pkl'
    _base_report_fname = '%s_%s_%d_%s.csv'

    def __init__(self,
                 path_years: List[int],
                 export_home: nd.PathLike,
                 report_home: nd.PathLike,
                 ):
        """Validates input attributes and builds class

        Parameters
        ----------
        path_years:
            A list of the years the models are running for.

        export_home:
            The home directory of all the export paths.

        report_home:
            The home directory of all the model reports paths.
        """
        # Assign attributes
        self.path_years = path_years
        self.export_home = export_home
        self.report_home = report_home

        # Make sure paths exist
        try:
            file_ops.check_path_exists(export_home)
            file_ops.check_path_exists(report_home)
        except IOError as e:
            raise type(e)(
                "Got the following error while checking if the export_home and "
                "report_home paths exist:\n%s"
                % str(e)
            )

        # Make sure variables that need to be overwritten, are
        if self._trip_origin is None:
            raise nd.PathingError(
                "When inheriting TramModelPaths the class variable "
                "_trip_origin needs to be set. This is usually set to "
                "'hb', or 'nhb' to reflect the type of model being run."
            )

    def _generate_report_paths(self,
                               report_name: str,
                               ) -> Tuple[Dict[int, str], Dict[int, str], Dict[int, str]]:
        """
        Creates report file paths for each of years

        Parameters
        ----------
        report_name:
            The name to use in the report filename. Filenames will be named
            as: [report_name, year, report_type], joined with '_'.

        Returns
        -------
        segment_total_paths:
            A dictionary of paths where the key is the year and the value is
            the path to the segment total reports for year.

        ca_sector_total_paths:
            A dictionary of paths where the key is the year and the value is
            the path to the ca sector segment total reports for year.

        ie_sector_total_paths:
            A dictionary of paths where the key is the year and the value is
            the path to the IE sector segment total reports for year.
        """
        # Init
        base_fname = self._base_report_fname
        fname_parts = [self._trip_origin, report_name]

        segment_total_paths = dict()
        ca_sector_paths = dict()
        ie_sector_paths = dict()

        # Create the paths for each year
        for year in self.path_years:
            # Segment totals
            fname = base_fname % (*fname_parts, year, self._segment_totals_report_name)
            segment_total_paths[year] = os.path.join(self.report_home, fname)

            # CA sector totals
            fname = base_fname % (*fname_parts, year, self._ca_sector_report_name)
            ca_sector_paths[year] = os.path.join(self.report_home, fname)

            # IE sector totals
            fname = base_fname % (*fname_parts, year, self._ie_sector_report_name)
            ie_sector_paths[year] = os.path.join(self.report_home, fname)

        return self.ReportPaths(
            segment_total=segment_total_paths,
            ca_sector=ca_sector_paths,
            ie_sector=ie_sector_paths,
        )


class HBProductionModelPaths(TramModelPaths):
    """Path Class for the Tram HB Production Model.

    This class defines and builds the export and reporting paths for
    the TramModelPaths. If the outputs of HBProductionModel are needed,
    create an instance of this class to generate all paths.

    Attributes
    ----------
    export_paths: nd.PathLike
        A namedtuple object (TramModelPaths.ExportPaths) with the following
        attributes (dictionary keys are path_years):
        - home: The home directory of all exports
        - notem_segmented: A dictionary of export paths for notem_segmented DVectors

    report_paths: nd.PathLike
        A namedtuple object (TramModelPaths.ExportPaths) with the following
        attributes (dictionary keys are path_years):
        - home: The home directory of all exports
        - notem_segmented: A TramModelPaths.ReportPaths object

    See TramModelPaths for documentation on:
    path_years, export_home, report_home
    """
    # Export fname params
    _trip_origin = 'hb'

    def __init__(self, *args, **kwargs):
        """Generates the export and report paths

        See super for more detail
        """
        # Set up superclass
        super().__init__(*args, **kwargs)

        # Generate the paths
        self._create_export_paths()
        self._create_report_paths()

    def _create_export_paths(self) -> None:
        """
        Creates self.export_paths
        """
        # Init
        base_fname = self._base_output_fname
        fname_parts = [self._trip_origin, self._zoning_system]

        notem_segmented_paths = dict()

        for year in self.path_years:
            # NoTEM Segmented path
            fname = base_fname % (*fname_parts, self._notem_segmented, year)
            notem_segmented_paths[year] = os.path.join(self.export_home, fname)

        # Create the export_paths class
        self.export_paths = self.ExportPaths(
            home=self.export_home,
            notem_segmented=notem_segmented_paths,
        )

    def _create_report_paths(self) -> None:
        """
        Creates self.report_paths
        """
        self.report_paths = self.ExportPaths(
            home=self.report_home,
            notem_segmented=self._generate_report_paths(self._notem_segmented),
        )


class HBAttractionModelPaths(TramModelPaths):
    """Path Class for the Tram HB Attraction Model.

    This class defines and builds the export and reporting paths for
    the TramModelPaths. If the outputs of HBAttractionModel are needed,
    create an instance of this class to generate all paths.

    Attributes
    ----------
    export_paths: nd.PathLike
        A namedtuple object (TramModelPaths.ExportPaths) with the following
        attributes (dictionary keys are path_years):
        - home: The home directory of all exports
        - notem_segmented: A dictionary of export paths for notem_segmented DVectors

    report_paths: nd.PathLike
        A namedtuple object (TramModelPaths.ExportPaths) with the following
        attributes (dictionary keys are path_years):
        - home: The home directory of all exports
        - notem_segmented: A TramModelPaths.ReportPaths object

    See TramModelPaths for documentation on:
    path_years, export_home, report_home
    """
    # Export fname params
    _trip_origin = 'hb'

    def __init__(self, *args, **kwargs):
        """Generates the export and report paths

        See super for more detail
        """
        # Set up superclass
        super().__init__(*args, **kwargs)

        # Generate the paths
        self._create_export_paths()
        self._create_report_paths()

    def _create_export_paths(self) -> None:
        """
        Creates self.export_paths
        """
        # Init
        base_fname = self._base_output_fname
        fname_parts = [self._trip_origin, self._zoning_system]

        notem_segmented_paths = dict()

        for year in self.path_years:
            # NoTEM Segmented path
            fname = base_fname % (*fname_parts, self._notem_segmented, year)
            notem_segmented_paths[year] = os.path.join(self.export_home, fname)

        # Create the export_paths class
        self.export_paths = self.ExportPaths(
            home=self.export_home,
            notem_segmented=notem_segmented_paths,
        )

    def _create_report_paths(self) -> None:
        """
        Creates self.report_paths
        """
        self.report_paths = self.ExportPaths(
            home=self.report_home,
            notem_segmented=self._generate_report_paths(self._notem_segmented),
        )


class NHBProductionModelPaths(TramModelPaths):
    """Path Class for the Tram NHB Production Model.

    This class defines and builds the export and reporting paths for
    the TramModelPaths. If the outputs of NHBProductionModel are needed,
    create an instance of this class to generate all paths.

    Attributes
    ----------
    export_paths: nd.PathLike
        A namedtuple object (TramModelPaths.ExportPaths) with the following
        attributes (dictionary keys are path_years):
        - home: The home directory of all exports
        - notem_segmented: A dictionary of export paths for notem_segmented DVectors

    report_paths: nd.PathLike
        A namedtuple object (TramModelPaths.ExportPaths) with the following
        attributes (dictionary keys are path_years):
        - home: The home directory of all exports
        - notem_segmented: A TramModelPaths.ReportPaths object

    See TramModelPaths for documentation on:
    path_years, export_home, report_home
    """
    # Export fname params
    _trip_origin = 'nhb'

    def __init__(self, *args, **kwargs):
        """Generates the export and report paths

        See super for more detail
        """
        # Set up superclass
        super().__init__(*args, **kwargs)

        # Generate the paths
        self._create_export_paths()
        self._create_report_paths()

    def _create_export_paths(self) -> None:
        """
        Creates self.export_paths
        """
        # Init
        base_fname = self._base_output_fname
        fname_parts = [self._trip_origin, self._zoning_system]

        notem_segmented_paths = dict()

        for year in self.path_years:
            # NoTEM Segmented path
            fname = base_fname % (*fname_parts, self._notem_segmented, year)
            notem_segmented_paths[year] = os.path.join(self.export_home, fname)

        # Create the export_paths class
        self.export_paths = self.ExportPaths(
            home=self.export_home,
            notem_segmented=notem_segmented_paths,
        )

    def _create_report_paths(self) -> None:
        """
        Creates self.report_paths
        """
        self.report_paths = self.ExportPaths(
            home=self.report_home,
            notem_segmented=self._generate_report_paths(self._notem_segmented),
        )


class NHBAttractionModelPaths(TramModelPaths):
    """Path Class for the Tram NHB Attraction Model.

    This class defines and builds the export and reporting paths for
    the TramModelPaths. If the outputs of NHBAttractionModel are needed,
    create an instance of this class to generate all paths.

    Attributes
    ----------
    export_paths: nd.PathLike
        A namedtuple object (TramModelPaths.ExportPaths) with the following
        attributes (dictionary keys are path_years):
        - home: The home directory of all exports
        - notem_segmented: A dictionary of export paths for notem_segmented DVectors

    report_paths: nd.PathLike
        A namedtuple object (TramModelPaths.ExportPaths) with the following
        attributes (dictionary keys are path_years):
        - home: The home directory of all exports
        - notem_segmented: A TramModelPaths.ReportPaths object

    See TramModelPaths for documentation on:
    path_years, export_home, report_home
    """
    # Export fname params
    _trip_origin = 'nhb'

    def __init__(self, *args, **kwargs):
        """Generates the export and report paths

        See super for more detail
        """
        # Set up superclass
        super().__init__(*args, **kwargs)

        # Generate the paths
        self._create_export_paths()
        self._create_report_paths()

    def _create_export_paths(self) -> None:
        """
        Creates self.export_paths
        """
        # Init
        base_fname = self._base_output_fname
        fname_parts = [self._trip_origin, self._zoning_system]

        notem_segmented_paths = dict()

        for year in self.path_years:
            # NoTEM Segmented path
            fname = base_fname % (*fname_parts, self._notem_segmented, year)
            notem_segmented_paths[year] = os.path.join(self.export_home, fname)

        # Create the export_paths class
        self.export_paths = self.ExportPaths(
            home=self.export_home,
            notem_segmented=notem_segmented_paths,
        )

    def _create_report_paths(self) -> None:
        """
        Creates self.report_paths
        """
        self.report_paths = self.ExportPaths(
            home=self.report_home,
            notem_segmented=self._generate_report_paths(self._notem_segmented),
        )
