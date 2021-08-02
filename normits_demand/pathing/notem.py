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

from typing import List
from typing import Dict
from typing import Tuple

# Third Party

# Local imports
import normits_demand as nd

from normits_demand.utils import file_ops


class NoTEMModelPaths:
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
        # Assign attributes
        self.path_years = path_years
        self.export_home = export_home
        self.report_home = report_home

        # Make sure paths exist
        file_ops.check_path_exists(export_home)
        file_ops.check_path_exists(report_home)

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
        report_path:
           The home path (directory) where all the reports should go

        years:
           A list of years to generate report paths for

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
    # Export fname params
    _trip_origin = 'hb'
    
    def __init__(self, *args, **kwargs):
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
