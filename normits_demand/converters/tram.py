# -*- coding: utf-8 -*-
"""
Created on: 15/03/2022
Updated on:

Original author: Ben Taylor
Last update made by:
Other updates made by:

File purpose:

"""
# Built-Ins

import pathlib

# Third Party

# Local Imports
from normits_demand import core as nd_core
from normits_demand.converters import notem as notem_converters
from normits_demand.pathing import tram as tram_pathing


class TramToDistributionModel(notem_converters.ToDistributionModel):
    """Helper class to convert NoTEM outputs into Distribution Model inputs"""

    def __init__(
        self,
        output_zoning: nd_core.ZoningSystem,
        base_year: int,
        scenario: nd_core.Scenario,
        notem_iteration_name: str,
        export_home: pathlib.Path,
        cache_dir: pathlib.Path = None,
        time_format: nd_core.TimeFormat = nd_core.TimeFormat.AVG_DAY,
    ):
        """
        Parameters
        ----------
        output_zoning:
            The zoning system the output DVectors should be in

        base_year:
            The year the Distribution Model is running for. Needed for
            compatibility with nd.pathing.TramExportPaths

        scenario:
            The name of the scenario to run for. Needed for
            compatibility with nd.pathing.TramExportPaths

        notem_iteration_name:
            The name of this iteration of the NoTEM models. Will be passed to
            nd.pathing.TramExportPaths to generate the NoTEM paths.

        export_home:
            The home directory of all the export paths. Will be passed to
            nd.pathing.TramExportPaths to generate the NoTEM paths.

        cache_dir:
            A path to the directory to store the cached DVectors.

        time_format:
            The nd_core.TimeFormat to use in the output DVectors.
        """
        # Get the path to the NoTEM Exports
        tram_exports = tram_pathing.TramExportPaths(
            path_years=[base_year],
            scenario=scenario,
            iteration_name=notem_iteration_name,
            export_home=export_home,
        )

        # Extract shorthand paths
        hbp_path = tram_exports.hb_production.export_paths.notem_segmented[base_year]
        hba_path = tram_exports.hb_attraction.export_paths.notem_segmented[base_year]
        nhbp_path = tram_exports.nhb_production.export_paths.notem_segmented[base_year]
        nhba_path = tram_exports.nhb_attraction.export_paths.notem_segmented[base_year]

        # Pass the generated paths over to the converter
        super().__init__(
            output_zoning=output_zoning,
            hb_productions_path=pathlib.Path(hbp_path),
            hb_attractions_path=pathlib.Path(hba_path),
            nhb_productions_path=pathlib.Path(nhbp_path),
            nhb_attractions_path=pathlib.Path(nhba_path),
            cache_dir=cache_dir,
            time_format=time_format,
        )
