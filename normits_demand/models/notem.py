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
from typing import Tuple

# Third Party

# Local Imports
import normits_demand as nd

from normits_demand.models import notem_production_model as notem
from normits_demand.models import notem_attraction_model as notem_attr


class NoTEM:
    # Constants
    _by_pop_file_name = "land_use_output_msoa.csv"
    _lu_pop_file_name = "land_use_%s_pop.csv"
    _lu_emp_file_name = "land_use_%s_emp.csv"
    _base_year = 2018
    _import_home = r"I:\NorMITs Demand\import\NoTEM"
    _export_home = r"C:\Data\Nirmal_Atkins"

    def __init__(self,
                 years: List[int],
                 scenario: str,
                 land_use_import_drive: nd.PathLike,
                 by_land_use_iter: str,
                 fy_land_use_iter: str,
                 ):
        """
        Assigns the attributes needed for NoTEM model.

        Parameters
        ----------
        years:
            List containing years for which the notem model is to be run.

        scenario:
            String containing the scenario to run.

        land_use_import_drive:
            Directory containing landuse outputs.

        by_land_use_iter:
            String containing base year landuse iteration Eg: 'iter3b'.

        fy_land_use_iter:
            String containing future year landuse iteration Eg: 'iter3b'.
        """
        # Init
        self.years = years
        self.scenario = scenario
        self.land_use_import_drive = land_use_import_drive
        self.by_land_use_iter = by_land_use_iter
        self.fy_land_use_iter = fy_land_use_iter

        self.pop_land_use_path, self.emp_land_use_path = self.generate_land_use_inputs()

    def run(self,
            generate_all_trip_ends: bool,
            generate_hb_trip_ends: bool,
            generate_nhb_trip_ends: bool,
            ) -> None:
        """
        Runs the notem trip end models based on the criteria given.

        Parameters
        ----------
        generate_all_trip_ends:
            Runs both home based and non home based trip end models.

        generate_hb_trip_ends:
            Runs the home based trip end models only.

        generate_nhb_trip_ends:
            Runs the non home based trip end models only.

        Returns
        -------
        None
        """
        if generate_all_trip_ends:
            self.generate_all_trip_ends()

        if generate_hb_trip_ends:
            self.generate_hb_trip_ends()

        if generate_nhb_trip_ends:
            self.generate_nhb_trip_ends()

    def generate_all_trip_ends(self,
                               ) -> None:
        """
        Runs both home based and non home based trip end models

        Returns
        -------
        None
        """
        # Runs hb and nhb trip ends
        self.generate_hb_trip_ends()
        self.generate_nhb_trip_ends()

    def generate_hb_trip_ends(self,
                              ) -> None:
        """
        Runs home based Production and Attraction trip end models

        Returns
        -------
        None
        """
        # Runs the module to create import dictionary
        imports_hb_prod = self.create_pop_imports("HB_Productions")
        imports_hb_attr = self.create_pop_imports("HB_Attractions")

        # Runs the home based Production model
        hb_prod = notem.HBProductionModel(
            land_use_paths=self.pop_land_use_path,
            trip_rates_path=imports_hb_prod['trip_rate'],
            mode_time_splits_path=imports_hb_prod['mode_time_split'],
            constraint_paths=self.pop_land_use_path,
            export_path=imports_hb_prod['export_path']
        )

        hb_prod.run(
            export_pure_demand=True,
            export_fully_segmented=True,
            export_notem_segmentation=True,
            export_reports=False,
            verbose=True,
        )

        # Path to read pure demand productions
        notem_seg_prod_fname = "hb_msoa_notem_segmented_2018_dvec.pkl"
        notem_seg_production = os.path.join(imports_hb_prod['export_path'], notem_seg_prod_fname)

        # Runs the home based attraction model
        hb_attr = notem_attr.HBAttractionModel(
            land_use_paths=self.emp_land_use_path,
            notem_segmented_productions=notem_seg_production,
            trip_attraction_rates_path=imports_hb_attr['trip_rate'],
            mode_controls_path=imports_hb_attr['mode_split'],
            constraint_paths=self.emp_land_use_path,
            export_path=imports_hb_attr['export_path']
        )

        hb_attr.run(
            export_pure_attractions=True,
            export_fully_segmented=True,
            export_reports=False,
            verbose=True,
        )

    def generate_nhb_trip_ends(self,
                               ) -> None:
        """
        Runs non home based Production and Attraction trip end models.

        Returns
        -------
        None
        """
        # TODO(NK) : Add the NHB trip end models

        raise NotImplementedError

    def generate_land_use_inputs(self,
                                 ) -> Tuple[Dict[int, str], Dict[int, str]]:
        """
        Creates dictionaries containing year and the path to the
        corresponding land use file as keys and values
        respectively for population and employment data

        Returns
        -------
        pop_land_use_path:
        A dictionary containing year and the corresponding land use
        file path for population data {year:path}

        emp_land_use_path:
        A dictionary containing year and the corresponding land use
        file path for employment data {year:path}
        """
        # Create base year land use home path
        by_land_use_home = os.path.join(
            self.land_use_import_drive,
            'NorMITs Land Use',
            'base_land_use',
            self.by_land_use_iter,
            'outputs',
        )
        # Create future year land use home path
        fy_land_use_home = os.path.join(
            self.land_use_import_drive,
            'NorMITs Land Use',
            'future_land_use',
            self.fy_land_use_iter,
            'outputs',
            'scenarios',
            self.scenario,
        )

        pop_land_use_path = dict()
        emp_land_use_path = dict()

        for year in self.years:
            pop_fname = self._lu_pop_file_name % str(year)
            emp_fname = self._lu_emp_file_name % str(year)

            if year == self._base_year:
                year_pop = os.path.join(by_land_use_home, self._by_pop_file_name)
                year_emp = os.path.join(by_land_use_home, emp_fname)
            else:
                # Build the path to this years data
                year_pop = os.path.join(fy_land_use_home, pop_fname)
                year_emp = os.path.join(fy_land_use_home, emp_fname)

            pop_land_use_path[year] = year_pop
            emp_land_use_path[year] = year_emp

        return pop_land_use_path, emp_land_use_path

    def create_pop_imports(self,
                           trip_end: str
                           ):
        """
        Creates dictionary containing import parameter and corresponding
        file path as keys and values respectively based on trip end input.

        Parameters
        ---------
        trip_end:
            String containing trip end input. The import dictionary
            varies based on the trip end input.
            Eg: "HB_Productions", "HB_Attractions" etc

        Returns
        -------
        imports_hb_prod:
            A dictionary containing home based production input parameters
            and the corresponding file path.

        imports_hb_attr:
            A dictionary containing home based attraction input parameters
            and the corresponding file path.
        """
        # Creates inputs required for HB Productions
        if trip_end == "HB_Productions":
            trip_rate_fname = "hb_trip_rates_v1.9.csv"
            trip_rates_path = os.path.join(self._import_home, trip_end, trip_rate_fname)

            mode_time_split_fname = "hb_mode_time_split_v1.9.csv"
            mode_time_split_path = os.path.join(self._import_home, trip_end, mode_time_split_fname)

            export_path = os.path.join(self._export_home, trip_end)

            imports_hb_prod = {
                'trip_rate': trip_rates_path,
                'mode_time_split': mode_time_split_path,
                'export_path': export_path
            }
            return imports_hb_prod

        # Creates inputs required for HB Attractions
        if trip_end == "HB_Attractions":
            trip_end = "Attractions" # Will be removed later
            trip_rate_fname = "sample_attraction_trip_rate.csv"
            trip_rates_path = os.path.join(self._import_home, trip_end, trip_rate_fname)

            mode_split_fname = "attraction_mode_split_new_infill.csv"
            mode_split_path = os.path.join(self._import_home, trip_end, mode_split_fname)

            export_path = os.path.join(self._export_home, trip_end)

            imports_hb_attr = {
                'trip_rate': trip_rates_path,
                'mode_split': mode_split_path,
                'export_path': export_path
            }
            return imports_hb_attr
