# -*- coding: utf-8 -*-
"""
Created on: Mon August 2 2021
Updated on:

Original author: Ben Taylor
Last update made by:
Other updates made by:

File purpose:
Classes which build all the paths for NoTEM model outputs
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
from normits_demand.utils import file_ops
from normits_demand.utils import general as du


class NoTEMImportPathsBase(ABC):

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
                'population_paths'
                'trip_rates_path'
                'mode_time_splits_path'
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
                'population_paths'
                'trip_rates_path'
                'time_splits_path'
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
            passed straight into the production model. Specifically, this
            function needs to produce a dictionary with the following keys:
                'employment_paths'
                'trip_weights_path'
                'mode_splits_path'
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
            passed straight into the production model. Specifically, this
            function needs to produce a dictionary with the following keys:
               None
        """
        raise NotImplementedError(
            "generate_nhb_attraction_imports() has not been implemented! If "
            "you have inherited this class, you need to make sure this "
            "function is overwritten by the child class.\n"
            "See the method documentation for information on how the function "
            "call and return should look."
        )


class NoTEMImportPaths(NoTEMImportPathsBase):

    # Constant
    _current_base_year = 2018

    # Land Use
    _normits_land_use = "NorMITs Land Use"
    _by_lu_dir = "base_land_use"
    _fy_lu_dir = "future_land_use"

    _by_pop_fname = "land_use_output_msoa.csv"
    _fy_pop_fname = "land_use_{year}_pop.csv"
    _fy_emp_fname = "land_use_{year}_emp.csv"

    # HB Productions
    _hb_productions_dname = "hb_productions"
    _hbp_trip_rate_fname = "hb_trip_rates_v{version}.csv"
    _hbp_m_tp_split_fname = "hb_mode_time_split_v{version}.csv"

    # HB Attractions
    _hb_attractions_dname = "hb_attractions"
    _hba_trip_weight_fname = "hb_trip_weights_v{version}.csv"
    _hba_mode_split_fname = "hb_mode_splits_v{version}.csv"

    # NHB Productions
    _nhb_productions_dname = "nhb_productions"
    _nhbp_trip_rates_fname = "nhb_trip_rates_v{version}.csv"
    _nhbp_time_split_fname = "nhb_time_split_v{version}.csv"

    # NHB Attractions
    # None currently needed

    def __init__(self,
                 import_home: nd.PathLike,
                 scenario: str,
                 years: List[int],
                 land_use_import_home: nd.PathLike,
                 by_land_use_iter: str,
                 fy_land_use_iter: str,
                 hb_production_import_version: str,
                 hb_attraction_import_version: str,
                 nhb_production_import_version: str,
                 ):
        """
        Assigns and validates attributes.

        Parameters
        ----------
        import_home:
            The home for all of the imports. This is usually a drive letter.
            Expects to find a folder titled
            NoTEMImportPath._normits_land_use in there.

        scenario:
            The name of the scenario to run for.

        years:
            List of years to run NoTEM for. Will assume that the smallest
            year is the base year.

        land_use_import_home:
            Path to the base directory of land use outputs.

        by_land_use_iter:
            String containing base year land use iteration name Eg: 'iter3b'.

        fy_land_use_iter:
            String containing future year land use iteration name Eg: 'iter3b'.
        """
        # Validate inputs
        file_ops.check_path_exists(import_home)
        file_ops.check_path_exists(land_use_import_home)

        # Assign attributes
        self.import_home = import_home
        self.hb_production_import_version = hb_production_import_version
        self.hb_attraction_import_version = hb_attraction_import_version
        self.nhb_production_import_version = nhb_production_import_version

        # Generate Land Use paths - needed later
        self._generate_land_use_paths(
            scenario=scenario,
            years=years,
            land_use_import_home=land_use_import_home,
            by_land_use_iter=by_land_use_iter,
            fy_land_use_iter=fy_land_use_iter,
        )

    def _generate_land_use_paths(self,
                                 scenario: str,
                                 years: List[int],
                                 land_use_import_home: nd.PathLike,
                                 by_land_use_iter: str,
                                 fy_land_use_iter: str,
                                 ) -> None:
        """
        Creates population and employment data dictionaries.

        Generated dictionaries are based on NorMITs Land Use paths.
        """
        # ## GENERATE THE HOME PATHS ## #
        by_home = os.path.join(
            land_use_import_home,
            self._normits_land_use,
            self._by_lu_dir,
            by_land_use_iter,
            'outputs',
        )

        # future year land use home path
        fy_home = os.path.join(
            land_use_import_home,
            self._normits_land_use,
            self._fy_lu_dir,
            fy_land_use_iter,
            'outputs',
            'scenarios',
            scenario,
        )

        # Because of the way land use is written, we have this weird legacy
        # now where NoTEM needs to base year to run correctly. If the smallest
        # number given isn't the defined base year, we just can't run.
        base_year, _ = du.split_base_future_years(years)
        if base_year != self._current_base_year:
            raise ValueError(
                "The assumed base year (the smallest of all inputs years) is "
                "not the same as the defined base year in this model.\n"
                "Assumed: %s\n"
                "Defined: %s\n"
                % (base_year, self._current_base_year)
            )

        # ## GENERATE FULL PATHS ## #
        self.population_paths = dict()
        self.employment_paths = dict()

        for year in years:
            pop_fname = self._fy_pop_fname.format(year=str(year))
            emp_fname = self._fy_emp_fname.format(year=str(year))

            # Land use gives a different pop fname in base year
            if year == base_year:
                year_pop = os.path.join(by_home, self._by_pop_fname)
                year_emp = os.path.join(by_home, emp_fname)
            else:
                year_pop = os.path.join(fy_home, pop_fname)
                year_emp = os.path.join(fy_home, emp_fname)

            self.population_paths[year] = year_pop
            self.employment_paths[year] = year_emp

    def generate_hb_production_imports(self) -> Dict[str, nd.PathLike]:
        """
        Generates input paths for home based production model.

        See NoTEMImportPathsBase.generate_hb_production_imports() for further
        documentation
        """
        # Generate the paths
        import_home = os.path.join(self.import_home, self._hb_productions_dname)
        trip_rates = self._hbp_trip_rate_fname.format(version=self.hb_production_import_version)
        m_tp_split = self._hbp_m_tp_split_fname.format(version=self.hb_production_import_version)

        # Format in dictionary
        return {
            'population_paths': self.population_paths,
            'trip_rates_path': os.path.join(import_home, trip_rates),
            'mode_time_splits_path': os.path.join(import_home, m_tp_split),
        }

    def generate_hb_attraction_imports(self) -> Dict[str, nd.PathLike]:
        """
        Generates input paths for home based attraction model

        See NoTEMImportPathsBase.generate_hb_attraction_imports() for further
        documentation
        """
        # Generate the paths
        import_home = os.path.join(self.import_home, self._hb_attractions_dname)
        trip_weights = self._hba_trip_weight_fname.format(version=self.hb_attraction_import_version)
        mode_split = self._hba_mode_split_fname.format(version=self.hb_attraction_import_version)

        return {
            'employment_paths': self.employment_paths,
            'trip_weights_path': os.path.join(import_home, trip_weights),
            'mode_splits_path': os.path.join(import_home, mode_split),
        }

    def generate_nhb_production_imports(self) -> Dict[str, nd.PathLike]:
        """
        Generates input paths for non-home based production model

        See NoTEMImportPathsBase.generate_nhb_production_imports() for further
        documentation
        """
        # Generate the paths
        import_home = os.path.join(self.import_home, self._nhb_productions_dname)
        trip_rates = self._nhbp_trip_rates_fname.format(version=self.nhb_production_import_version)
        time_split = self._nhbp_time_split_fname.format(version=self.nhb_production_import_version)

        return {
            'population_paths': self.population_paths,
            'trip_rates_path': os.path.join(import_home, trip_rates),
            'time_splits_path': os.path.join(import_home, time_split),
        }

    def generate_nhb_attraction_imports(self) -> Dict[str, nd.PathLike]:
        """
        Generates input paths for non-home based attraction model

        Note that no import paths are currently needed for this model

        See NoTEMImportPathsBase.generate_hb_attraction_imports() for further
        documentation
        """
        return dict()


class NoTEMExportPaths:
    # Define the names of the export dirs
    _hb_productions_dir = 'hb_productions'
    _nhb_productions_dir = 'nhb_productions'
    _hb_attractions_dir = 'hb_attractions'
    _nhb_attractions_dir = 'nhb_attractions'

    _reports_dir = 'reports'

    def __init__(self,
                 path_years: List[int],
                 export_home: nd.PathLike
                 ):
        """
        Builds the export paths for all the NoTEM sub-models

        Parameters
        ----------
        path_years:
            A list of the years the models are running for.

        export_home:
            The home directory of all the export paths. A sub-directory will
            be made for each of the NoTEM sub models.
        """
        # Init
        file_ops.check_path_exists(export_home)

        self.export_home = export_home
        self.path_years = path_years

        # ## BUILD ALL MODEL PATHS ## #
        # hb productions
        hb_p_export_home = os.path.join(export_home, self._hb_productions_dir)
        hb_p_report_home = os.path.join(hb_p_export_home, self._reports_dir)
        file_ops.create_folder(hb_p_report_home)
        self.hb_production = HBProductionModelPaths(
            path_years=path_years,
            export_home=hb_p_export_home,
            report_home=hb_p_report_home,
        )

        # nhb productions
        nhb_p_export_home = os.path.join(export_home, self._nhb_productions_dir)
        nhb_p_report_home = os.path.join(nhb_p_export_home, self._reports_dir)
        file_ops.create_folder(nhb_p_report_home)
        self.nhb_production = NHBProductionModelPaths(
            path_years=path_years,
            export_home=nhb_p_export_home,
            report_home=nhb_p_report_home,
        )

        # hb attractions
        hb_a_export_home = os.path.join(export_home, self._hb_attractions_dir)
        hb_a_report_home = os.path.join(hb_a_export_home, self._reports_dir)
        file_ops.create_folder(hb_a_report_home)

        self.hb_attraction = HBAttractionModelPaths(
            path_years=path_years,
            export_home=hb_a_export_home,
            report_home=hb_a_report_home,
        )

        # nhb attractions
        nhb_a_export_home = os.path.join(export_home, self._nhb_attractions_dir)
        nhb_a_report_home = os.path.join(nhb_a_export_home, self._reports_dir)
        file_ops.create_folder(nhb_a_report_home)

        self.nhb_attraction = NHBAttractionModelPaths(
            path_years=path_years,
            export_home=nhb_a_export_home,
            report_home=nhb_a_report_home,
        )


class NoTEMPaths(NoTEMExportPaths, NoTEMImportPaths):

    def __init__(self,
                 years: List[int],
                 export_home: nd.PathLike,
                 import_home: nd.PathLike,
                 scenario: str,
                 *args,
                 **kwargs,
                 ):
        NoTEMExportPaths.__init__(
            self,
            path_years=years,
            export_home=export_home,
        )
        NoTEMImportPaths.__init__(
            self,
            import_home=import_home,
            scenario=scenario,
            years=years,
            *args,
            **kwargs,
        )


class NoTEMModelPaths:
    """Base Path Class for all NoTEM models.

    This class forms the base path class that all NoTEM model path classes
    are built off of. It defines a number of constants to ensure all
    NoTEM models follow the same output structure and naming conventions

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
    _pure_demand = 'pure_demand'
    _fully_segmented = 'fully_segmented'
    _notem_segmented = 'notem_segmented'

    # Report names
    _segment_totals_report_name = "segment_totals"
    _ca_sector_report_name = "ca_sector_totals"
    _ie_sector_report_name = "ie_sector_totals"

    # Output Path Classes
    ExportPaths = collections.namedtuple(
        typename='ExportPaths',
        field_names='home, pure_demand, fully_segmented, notem_segmented',
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
                "When inheriting NoTEMModelPaths the class variable "
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


class HBProductionModelPaths(NoTEMModelPaths):
    """Path Class for the NoTEM HB Production Model.

    This class defines and builds the export and reporting paths for
    the NoTEMModelPaths. If the outputs of HBProductionModel are needed,
    create an instance of this class to generate all paths.

    Attributes
    ----------
    export_paths: nd.PathLike
        A namedtuple object (NoTEMModelPaths.ExportPaths) with the following
        attributes (dictionary keys are path_years):
        - home: The home directory of all exports
        - pure_demand: A dictionary of export paths for pure_demand DVectors
        - fully_segmented: A dictionary of export paths for fully_segmented DVectors
        - notem_segmented: A dictionary of export paths for notem_segmented DVectors

    report_paths: nd.PathLike
        A namedtuple object (NoTEMModelPaths.ExportPaths) with the following
        attributes (dictionary keys are path_years):
        - home: The home directory of all exports
        - pure_demand: A NoTEMModelPaths.ReportPaths object
        - fully_segmented: A NoTEMModelPaths.ReportPaths object
        - notem_segmented: A NoTEMModelPaths.ReportPaths object

    See NoTEMModelPaths for documentation on:
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

        pure_demand_paths = dict()
        fully_segmented_paths = dict()
        notem_segmented_paths = dict()

        for year in self.path_years:
            # Pure demand path
            fname = base_fname % (*fname_parts, self._pure_demand, year)
            pure_demand_paths[year] = os.path.join(self.export_home, fname)

            # Fully Segmented path
            fname = base_fname % (*fname_parts, self._fully_segmented, year)
            fully_segmented_paths[year] = os.path.join(self.export_home, fname)

            # NoTEM Segmented path
            fname = base_fname % (*fname_parts, self._notem_segmented, year)
            notem_segmented_paths[year] = os.path.join(self.export_home, fname)

        # Create the export_paths class
        self.export_paths = self.ExportPaths(
            home=self.export_home,
            pure_demand=pure_demand_paths,
            fully_segmented=fully_segmented_paths,
            notem_segmented=notem_segmented_paths,
        )

    def _create_report_paths(self) -> None:
        """
        Creates self.report_paths
        """
        self.report_paths = self.ExportPaths(
            home=self.report_home,
            pure_demand=self._generate_report_paths(self._pure_demand),
            fully_segmented=None,
            notem_segmented=self._generate_report_paths(self._notem_segmented),
        )


class HBAttractionModelPaths(NoTEMModelPaths):
    """Path Class for the NoTEM HB Attraction Model.

    This class defines and builds the export and reporting paths for
    the NoTEMModelPaths. If the outputs of HBAttractionModel are needed,
    create an instance of this class to generate all paths.

    Attributes
    ----------
    export_paths: nd.PathLike
        A namedtuple object (NoTEMModelPaths.ExportPaths) with the following
        attributes (dictionary keys are path_years):
        - home: The home directory of all exports
        - pure_demand: A dictionary of export paths for pure_demand DVectors
        - fully_segmented: A dictionary of export paths for fully_segmented DVectors
        - notem_segmented: A dictionary of export paths for notem_segmented DVectors

    report_paths: nd.PathLike
        A namedtuple object (NoTEMModelPaths.ExportPaths) with the following
        attributes (dictionary keys are path_years):
        - home: The home directory of all exports
        - pure_demand: A NoTEMModelPaths.ReportPaths object
        - fully_segmented: A NoTEMModelPaths.ReportPaths object
        - notem_segmented: A NoTEMModelPaths.ReportPaths object

    See NoTEMModelPaths for documentation on:
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

        pure_demand_paths = dict()
        fully_segmented_paths = dict()
        notem_segmented_paths = dict()

        for year in self.path_years:
            # Pure demand path
            fname = base_fname % (*fname_parts, self._pure_demand, year)
            pure_demand_paths[year] = os.path.join(self.export_home, fname)

            # Fully Segmented path
            fname = base_fname % (*fname_parts, self._fully_segmented, year)
            fully_segmented_paths[year] = os.path.join(self.export_home, fname)

            # NoTEM Segmented path
            fname = base_fname % (*fname_parts, self._notem_segmented, year)
            notem_segmented_paths[year] = os.path.join(self.export_home, fname)

        # Create the export_paths class
        self.export_paths = self.ExportPaths(
            home=self.export_home,
            pure_demand=pure_demand_paths,
            fully_segmented=fully_segmented_paths,
            notem_segmented=notem_segmented_paths,
        )

    def _create_report_paths(self) -> None:
        """
        Creates self.report_paths
        """
        self.report_paths = self.ExportPaths(
            home=self.report_home,
            pure_demand=self._generate_report_paths(self._pure_demand),
            fully_segmented=None,
            notem_segmented=self._generate_report_paths(self._notem_segmented),
        )


class NHBProductionModelPaths(NoTEMModelPaths):
    """Path Class for the NoTEM NHB Production Model.

    This class defines and builds the export and reporting paths for
    the NoTEMModelPaths. If the outputs of NHBProductionModel are needed,
    create an instance of this class to generate all paths.

    Attributes
    ----------
    export_paths: nd.PathLike
        A namedtuple object (NoTEMModelPaths.ExportPaths) with the following
        attributes (dictionary keys are path_years):
        - home: The home directory of all exports
        - pure_demand: A dictionary of export paths for pure_demand DVectors
        - fully_segmented: A dictionary of export paths for fully_segmented DVectors
        - notem_segmented: A dictionary of export paths for notem_segmented DVectors

    report_paths: nd.PathLike
        A namedtuple object (NoTEMModelPaths.ExportPaths) with the following
        attributes (dictionary keys are path_years):
        - home: The home directory of all exports
        - pure_demand: A NoTEMModelPaths.ReportPaths object
        - fully_segmented: A NoTEMModelPaths.ReportPaths object
        - notem_segmented: A NoTEMModelPaths.ReportPaths object

    See NoTEMModelPaths for documentation on:
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

        pure_demand_paths = dict()
        fully_segmented_paths = dict()
        notem_segmented_paths = dict()

        for year in self.path_years:
            # Pure demand path
            fname = base_fname % (*fname_parts, self._pure_demand, year)
            pure_demand_paths[year] = os.path.join(self.export_home, fname)

            # Fully Segmented path
            fname = base_fname % (*fname_parts, self._fully_segmented, year)
            fully_segmented_paths[year] = os.path.join(self.export_home, fname)

            # NoTEM Segmented path
            fname = base_fname % (*fname_parts, self._notem_segmented, year)
            notem_segmented_paths[year] = os.path.join(self.export_home, fname)

        # Create the export_paths class
        self.export_paths = self.ExportPaths(
            home=self.export_home,
            pure_demand=pure_demand_paths,
            fully_segmented=fully_segmented_paths,
            notem_segmented=notem_segmented_paths,
        )

    def _create_report_paths(self) -> None:
        """
        Creates self.report_paths
        """
        self.report_paths = self.ExportPaths(
            home=self.report_home,
            pure_demand=self._generate_report_paths(self._pure_demand),
            fully_segmented=None,
            notem_segmented=self._generate_report_paths(self._notem_segmented),
        )


class NHBAttractionModelPaths(NoTEMModelPaths):
    """Path Class for the NoTEM NHB Attraction Model.

    This class defines and builds the export and reporting paths for
    the NoTEMModelPaths. If the outputs of NHBAttractionModel are needed,
    create an instance of this class to generate all paths.

    Attributes
    ----------
    export_paths: nd.PathLike
        A namedtuple object (NoTEMModelPaths.ExportPaths) with the following
        attributes (dictionary keys are path_years):
        - home: The home directory of all exports
        - pure_demand: A dictionary of export paths for pure_demand DVectors
        - fully_segmented: A dictionary of export paths for fully_segmented DVectors
        - notem_segmented: A dictionary of export paths for notem_segmented DVectors

    report_paths: nd.PathLike
        A namedtuple object (NoTEMModelPaths.ExportPaths) with the following
        attributes (dictionary keys are path_years):
        - home: The home directory of all exports
        - pure_demand: A NoTEMModelPaths.ReportPaths object
        - fully_segmented: A NoTEMModelPaths.ReportPaths object
        - notem_segmented: A NoTEMModelPaths.ReportPaths object

    See NoTEMModelPaths for documentation on:
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

        pure_demand_paths = dict()
        fully_segmented_paths = dict()
        notem_segmented_paths = dict()

        for year in self.path_years:
            # Pure demand path
            fname = base_fname % (*fname_parts, self._pure_demand, year)
            pure_demand_paths[year] = os.path.join(self.export_home, fname)

            # Fully Segmented path
            fname = base_fname % (*fname_parts, self._fully_segmented, year)
            fully_segmented_paths[year] = os.path.join(self.export_home, fname)

            # NoTEM Segmented path
            fname = base_fname % (*fname_parts, self._notem_segmented, year)
            notem_segmented_paths[year] = os.path.join(self.export_home, fname)

        # Create the export_paths class
        self.export_paths = self.ExportPaths(
            home=self.export_home,
            pure_demand=pure_demand_paths,
            fully_segmented=None,
            notem_segmented=notem_segmented_paths,
        )

    def _create_report_paths(self) -> None:
        """
        Creates self.report_paths
        """
        self.report_paths = self.ExportPaths(
            home=self.report_home,
            pure_demand=self._generate_report_paths(self._pure_demand),
            fully_segmented=None,
            notem_segmented=self._generate_report_paths(self._notem_segmented),
        )


