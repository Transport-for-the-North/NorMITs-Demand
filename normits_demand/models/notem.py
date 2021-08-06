# -*- coding: utf-8 -*-
"""
Created on: Tues May 25th 2021
Updated on:

Original author: Ben Taylor
Last update made by: Nirmal Kumar
Other updates made by: Nirmal Kumar

File purpose:
NoTEM Class Frontend for calling all production and attraction models
"""
# Builtins
import os.path
from typing import List
from typing import Dict

# Third Party

# Local Imports
import normits_demand as nd

from normits_demand.models import HBProductionModel
from normits_demand.models import NHBProductionModel
from normits_demand.models import HBAttractionModel
from normits_demand.models import NHBAttractionModel

from normits_demand.pathing import NoTEMPaths

from normits_demand.utils import file_ops
from normits_demand.utils import general as du


class NoTEM(NoTEMPaths):
    # Constants
    _hb_attr = "HB_Attractions"
    _nhb_prod = "NHB_Productions"
    _nhb_attr = "NHB_Attractions"
    _hb_attr_trip_rate_fname = "sample_attraction_trip_rate.csv"
    _hb_attr_mode_split_fname = "attraction_mode_split_new_infill.csv"
    _nhb_prod_trip_rate_fname = "nhb_ave_wday_enh_trip_rates_v1.5.csv"
    _nhb_prod_time_split_fname = "tfn_nhb_ave_week_time_split_18_v1.5.csv"

    _hb_prod = "HB_Productions"

    def __init__(self,
                 years: List[int],
                 scenario: str,
                 import_home: nd.PathLike,
                 export_home: nd.PathLike,

                 hb_production_import_version: str,

                 *args,
                 **kwargs,
                 ):
        """
        Assigns the attributes needed for NoTEM model.

        Parameters
        ----------
        years:
            List of years to run NoTEM for. Will assume that the smallest
            year is the base year.

        scenario:
            The name of the scenario to run for.

        land_use_import_home:
            Path to the base directory of land use outputs.

        by_land_use_iter:
            String containing base year land use iteration Eg: 'iter3b'.

        fy_land_use_iter:
            String containing future year land use iteration Eg: 'iter3b'.
        """
        # Validate inputs
        file_ops.check_path_exists(import_home)

        # Assign
        self.years = years
        self.scenario = scenario

        self.hb_production_import_version = hb_production_import_version

        # Generate the import and export paths
        super().__init__(
            path_years=self.years,
            export_home=export_home,
            import_home=import_home,
            scenario=scenario,
            years=years,
            *args,
            **kwargs,
        )

    def run(self,
            generate_all: bool = False,
            generate_hb: bool = False,
            generate_hb_production: bool = False,
            generate_hb_attraction: bool = False,
            generate_nhb: bool = False,
            generate_nhb_production: bool = False,
            generate_nhb_attraction: bool = False,
            verbose: bool = True,
            ) -> None:
        """
        Runs the notem trip end models based on the criteria given.

        Parameters
        ----------
        generate_all:
            Runs both home based and non home based trip end models.

        generate_hb:
            Runs the home based trip end models only.

        generate_hb_production:
            Runs the home based production trip end model only.

        generate_hb_attraction:
            Runs the home based attraction trip end model only.

        generate_nhb:
            Runs the non home based trip end models only.

        generate_nhb_production:
            Runs the non home based production trip end model only.

        generate_nhb_attraction:
            Runs the non home based attraction trip end model only.

        verbose:
            Whether to print progress updates to the terminal while running
            or not.

        Returns
        -------
        None
        """
        # Determine which models to run
        if generate_all:
            generate_hb = True
            generate_nhb = True

        if generate_hb:
            generate_hb_production = True
            generate_hb_attraction = True

        if generate_nhb:
            generate_nhb_production = True
            generate_nhb_attraction = True

        # Run the models
        if generate_hb_production:
            self._generate_hb_production(verbose)

        if generate_hb_attraction:
            self._generate_hb_attraction(verbose)

        if generate_nhb_production:
            self._generate_nhb_production(verbose)

        if generate_nhb_attraction:
            self._generate_nhb_attraction(verbose)

    def _generate_hb_production(self, verbose: bool) -> None:
        """
        Runs home based Production trip end models
        """
        import_files = self.generate_hb_production_imports(
            version=self.hb_production_import_version,
        )

        # Runs the home based Production model
        hb_prod = HBProductionModel(
            **import_files,
            constraint_paths=None,
            export_home=self.hb_production.export_paths.home,
        )

        hb_prod.run(
            export_pure_demand=True,
            export_fully_segmented=True,
            export_notem_segmentation=True,
            export_reports=True,
            verbose=verbose,
        )

    def _generate_hb_attraction(self, verbose: bool) -> None:
        """
        Runs the home based Attraction trip end model
        """
        # ## GENERATE THE NEEDED PATHS ## #
        # Runs the module to create import dictionary
        imports = self.generate_hb_attraction_imports()

        # Get the hb productions
        export_paths = self.hb_production.export_paths
        control_production_paths = {y: export_paths.notem_segmented[y] for y in self.years}

        # ## INSTANTIATE AND RUN THE MODEL ## #
        hb_attr = HBAttractionModel(
            land_use_paths=self.emp_land_use_path,
            control_production_paths=control_production_paths,
            attraction_trip_rates_path=imports['trip_rate'],
            mode_splits_path=imports['mode_split'],
            constraint_paths=None,
            export_home=imports['export_path'],
        )

        hb_attr.run(
            export_pure_attractions=True,
            export_fully_segmented=False,
            export_notem_segmentation=True,
            export_reports=True,
            verbose=verbose,
        )

    def _generate_nhb_production(self, verbose: bool) -> None:
        """
        Runs the non-home based Production trip end model
        """
        # ## GENERATE THE NEEDED PATHS ## #
        # Runs the module to create import dictionary
        imports = self.generate_nhb_production_imports()

        # Get the hb attractions
        export_paths = self.hb_attraction.export_path
        hb_attraction_paths = {y: export_paths.notem_segmented[y] for y in self.years}

        # ## INSTANTIATE AND RUN THE MODEL ## #
        nhb_prod = NHBProductionModel(
            hb_attraction_paths=hb_attraction_paths,
            land_use_paths=self.pop_land_use_path,
            nhb_trip_rates_path=imports['nbh_trip_rate'],
            nhb_time_splits_path=imports['nbh_time_split_rate'],
            export_home=imports['export_path'],
            constraint_paths=None,
        )

        nhb_prod.run(
            export_nhb_pure_demand=True,
            export_fully_segmented=False,
            export_notem_segmentation=True,
            export_reports=True,
            verbose=verbose,
        )

    def _generate_nhb_attraction(self, verbose: bool) -> None:
        """
        Runs non home based Attraction trip end models.
        """
        # ## GENERATE THE NEEDED PATHS ## #
        # Runs the module to create import dictionary
        imports = self.generate_nhb_attraction_imports()

        # Get the hb attractions
        export_paths = self.hb_attraction.export_path
        hb_attraction_paths = {y: export_paths.notem_segmented[y] for y in self.years}

        # Get the nhb productions
        export_paths = self.nhb_production.export_paths
        nhb_production_paths = {y: export_paths.notem_segmented[y] for y in self.years}

        # ## INSTANTIATE AND RUN THE MODEL ## #
        nhb_attr = NHBAttractionModel(
            hb_attraction_paths=hb_attraction_paths,
            nhb_production_paths=nhb_production_paths,
            export_home=imports['export_path'],
            constraint_paths=None
        )

        nhb_attr.run(
            export_nhb_pure_attractions=True,
            export_notem_segmentation=False,
            export_reports=True,
            verbose=verbose,
        )

    def generate_hb_attraction_imports(self) -> Dict[str, nd.PathLike]:
        """
        Creates inputs required for home based attraction trip ends.

        Creates dictionary containing import parameter and corresponding
        file path as keys and values respectively for home based attraction
        trip ends.

        Returns
        -------
        imports_hb_attr:
            A dictionary containing home based attraction input parameters
            and the corresponding file path.
        """

        # Creates inputs required for HB Attractions
        trip_rates_path = os.path.join(self.import_home, self._hb_attr, self._hb_attr_trip_rate_fname)
        mode_split_path = os.path.join(self.import_home, self._hb_attr, self._hb_attr_mode_split_fname)

        imports_hb_attr = {
            'trip_rate': trip_rates_path,
            'mode_split': mode_split_path,

        }
        return imports_hb_attr

    def generate_nhb_production_imports(self) -> Dict[str, nd.PathLike]:
        """
        Creates inputs required for non home based production trip ends.

        Creates dictionary containing import parameter and corresponding
        file path as keys and values respectively for non home based production
        trip ends.

        Returns
        -------
        imports_nhb_prod:
            A dictionary containing non home based production input parameters
            and the corresponding file path.
        """
        # Creates inputs required for NHB Productions

        nhb_trip_rates_path = os.path.join(self.import_home, self._nhb_prod, self._nhb_prod_trip_rate_fname)
        nhb_time_split_path = os.path.join(self.import_home, self._hb_prod,
                                           self._nhb_prod_time_split_fname)

        imports_nhb_prod = {
            'nhb_trip_rate': nhb_trip_rates_path,
            'nhb_time_split': nhb_time_split_path,
            'export_path': self.nhb_production.export_paths.home
        }
        return imports_nhb_prod

    def generate_nhb_attraction_imports(self) -> Dict[str, nd.PathLike]:
        """
        Creates inputs required for non home based attraction trip ends.

        Creates dictionary containing import parameter and corresponding
        file path as keys and values respectively for non home based attraction
        trip ends.

        Returns
        -------
        imports_nhb_attr:
            A dictionary containing non home based attraction input parameters
            and the corresponding file path.
        """
        imports_nhb_attr = {
            'export_path': self.nhb_attraction.export_paths.home
        }
        return imports_nhb_attr
