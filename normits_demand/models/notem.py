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

from normits_demand.models import notem_production_model as notem
from normits_demand.models import notem_attraction_model as notem_attr
from normits_demand.utils import file_ops as ops


class NoTEM:
    # Constants
    _by_pop_file_name = "land_use_output_msoa.csv"
    _lu_pop_file_name = "land_use_%s_pop.csv"
    _lu_emp_file_name = "land_use_%s_emp.csv"
    _base_year = 2018
    _normits_land_use = "NorMITs Land Use"
    _hb_prod = "HB_Productions"
    _hb_attr = "HB_Attractions"
    _nhb_prod = "NHB_Productions"
    _nhb_attr = "NHB_Attractions"
    _hb_prod_trip_rate_fname = "hb_trip_rates_v1.9.csv"
    _hb_prod_mode_time_split_fname = "hb_mode_time_split_v1.9.csv"
    _notem_seg_prod_fname = "hb_msoa_notem_segmented_%d_dvec.pkl"
    _notem_seg_attr_fname = "hb_msoa_notem_segmented_%d_dvec.pkl"
    _hb_attr_trip_rate_fname = "sample_attraction_trip_rate.csv"
    _hb_attr_mode_split_fname = "attraction_mode_split_new_infill.csv"
    _nhb_prod_trip_rate_fname = "nhb_ave_wday_enh_trip_rates_v1.5.csv"
    _nhb_prod_time_split_fname = "tfn_nhb_ave_week_time_split_18_v1.5.csv"

    def __init__(self,
                 years: List[int],
                 scenario: str,
                 land_use_import_home: nd.PathLike,
                 by_land_use_iter: str,
                 fy_land_use_iter: str,
                 notem_import_home: nd.PathLike,
                 notem_export_home: nd.PathLike,
                 ):
        """
        Assigns the attributes needed for NoTEM model.

        Parameters
        ----------
        years:
            List of years to run NoTEM for. Will assume that the smallest
            year is the base year for the purposes of reading in Land Use
            data.

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
        ops.check_path_exists(land_use_import_home)
        ops.check_path_exists(notem_import_home)
        ops.check_path_exists(notem_export_home)
        if self._base_year not in years:
            raise ValueError(
                "Base year %d not found in years list"
                % self._base_year
            )

        # Assign
        self.years = years
        self.scenario = scenario
        self.land_use_import_home = land_use_import_home
        self.notem_import_home = notem_import_home
        self.notem_export_home = notem_export_home
        self.by_land_use_iter = by_land_use_iter
        self.fy_land_use_iter = fy_land_use_iter

        # Create paths
        self._generate_land_use_inputs()

    def run(self,
            generate_all: bool = False,
            generate_hb: bool = False,
            generate_hb_production: bool = False,
            generate_hb_attraction: bool = False,
            generate_nhb: bool = False,
            generate_nhb_production: bool = False,
            generate_nhb_attraction: bool = False,
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

        Returns
        -------
        None
        """
        if generate_all:
            generate_hb = True
            generate_nhb = True

        if generate_hb:
            generate_hb_production = True
            generate_hb_attraction = True

        if generate_hb_production:
            self.generate_hb_production()

        if generate_hb_attraction:
            self.generate_hb_attraction()

        if generate_nhb:
            generate_nhb_production = True
            generate_nhb_attraction = True

        if generate_nhb_production:
            self.generate_nhb_production()

        if generate_nhb_attraction:
            self.generate_nhb_attraction()

    def generate_hb_production(self,
                               ) -> None:
        """
        Runs home based Production trip end models

        Returns
        -------
        None
        """
        # Runs the module to create import dictionary
        imports_hb_prod = self.generate_hb_production_imports()

        # Runs the home based Production model
        hb_prod = notem.HBProductionModel(
            land_use_paths=self.pop_land_use_path,
            trip_rates_path=imports_hb_prod['trip_rate'],
            mode_time_splits_path=imports_hb_prod['mode_time_split'],
            constraint_paths=None,
            export_path=imports_hb_prod['export_path'],
        )

        hb_prod.run(
            export_pure_demand=True,
            export_fully_segmented=True,
            export_notem_segmentation=True,
            export_reports=True,
            verbose=True,
        )

    def generate_hb_attraction(self,
                               ) -> None:
        """
        Runs home based Attraction trip end models

        Returns
        -------
        None
        """
        # Runs the module to create import dictionary
        imports_hb_attr = self.generate_hb_attraction_imports()

        # Runs the home based attraction model
        hb_attr = notem_attr.HBAttractionModel(
            land_use_paths=self.emp_land_use_path,
            control_production_paths=self._generate_notem_seg_prod(),
            attraction_trip_rates_path=imports_hb_attr['trip_rate'],
            mode_splits_path=imports_hb_attr['mode_split'],
            constraint_paths=None,
            export_path=imports_hb_attr['export_path'],
        )

        hb_attr.run(
            export_pure_attractions=True,
            export_fully_segmented=True,
            export_notem_segmentation=True,
            export_reports=True,
            verbose=True,
        )

    def generate_nhb_production(self,
                                ) -> None:
        """
        Runs non home based Production trip end models.

        Returns
        -------
        None
        """

        # Runs the module to create import dictionary
        imports_nhb_prod = self.generate_nhb_production_imports()

        nhb_prod = notem.NHBProductionModel(
            hb_attractions=self._generate_notem_seg_attr(),
            land_use_paths=self.pop_land_use_path,
            nhb_trip_rates_path=imports_nhb_prod['nbh_trip_rate'],
            nhb_time_splits_path=imports_nhb_prod['nbh_time_split_rate'],
            export_path=imports_nhb_prod['export_path'],
            constraint_paths=None,
        )

        nhb_prod.run(
            export_nhb_pure_demand=True,
            export_fully_segmented=True,
            export_notem_segmentation=True,
            export_reports=True,
            verbose=True,
        )

    def generate_nhb_attraction(self,
                                ) -> None:
        """
        Runs non home based Attraction trip end models.

        Returns
        -------
        None
        """

        # Runs the module to create import dictionary
        imports_nhb_prod = self.generate_nhb_attraction_imports()

        nhb_attr = notem_attr.NHBAttractionModel(
            hb_attraction_paths=self._generate_notem_seg_attr(),
            nhb_production_paths=self._generate_notem_seg_nhb_prod(),
            export_path=imports_nhb_prod['export_path'],
            constraint_paths=None
        )

        nhb_attr.run(
            export_nhb_pure_attractions=True,
            export_notem_segmentation=True,
            export_reports=True,
            verbose=True,
        )

    def _generate_land_use_inputs(self,
                                  ) -> None:
        """
        Creates the land use import paths

        Creates dictionaries containing year and the path to the
        corresponding land use file as keys and values
        respectively for population and employment data

        Returns
        -------
        None
        """
        # Create base year land use home path
        by_land_use_home = os.path.join(
            self.land_use_import_home,
            self._normits_land_use,
            'base_land_use',
            self.by_land_use_iter,
            'outputs',
        )
        # Create future year land use home path
        fy_land_use_home = os.path.join(
            self.land_use_import_home,
            self._normits_land_use,
            'future_land_use',
            self.fy_land_use_iter,
            'outputs',
            'scenarios',
            self.scenario,
        )
        self.pop_land_use_path = dict()
        self.emp_land_use_path = dict()

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

            self.pop_land_use_path[year] = year_pop
            self.emp_land_use_path[year] = year_emp

    def generate_hb_production_imports(self,
                                       ) -> Dict[str, nd.PathLike]:
        """
        Creates inputs required for home based production trip ends.

        Creates dictionary containing import parameter and corresponding
        file path as keys and values respectively for home based production
        trip ends.

        Returns
        -------
        imports_hb_prod:
            A dictionary containing home based production input parameters
            and the corresponding file path.
        """
        # Creates inputs required for HB Productions

        trip_rates_path = os.path.join(self.notem_import_home, self._hb_prod, self._hb_prod_trip_rate_fname)
        mode_time_split_path = os.path.join(self.notem_import_home, self._hb_prod,
                                            self._hb_prod_mode_time_split_fname)
        export_path = os.path.join(self.notem_export_home, self._hb_prod)

        imports_hb_prod = {
            'trip_rate': trip_rates_path,
            'mode_time_split': mode_time_split_path,
            'export_path': export_path
        }
        return imports_hb_prod

    def generate_hb_attraction_imports(self,
                                       ) -> Dict[str, nd.PathLike]:
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
        trip_rates_path = os.path.join(self.notem_import_home, self._hb_attr, self._hb_attr_trip_rate_fname)
        mode_split_path = os.path.join(self.notem_import_home, self._hb_attr, self._hb_attr_mode_split_fname)
        export_path = os.path.join(self.notem_export_home, self._hb_attr)

        imports_hb_attr = {
            'trip_rate': trip_rates_path,
            'mode_split': mode_split_path,
            'export_path': export_path
        }
        return imports_hb_attr

    def generate_nhb_production_imports(self,
                                        ) -> Dict[str, nd.PathLike]:
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

        nhb_trip_rates_path = os.path.join(self.notem_import_home, self._nhb_prod, self._nhb_prod_trip_rate_fname)
        nhb_time_split_path = os.path.join(self.notem_import_home, self._hb_prod,
                                           self._nhb_prod_time_split_fname)
        export_path = os.path.join(self.notem_export_home, self._nhb_prod)

        imports_nhb_prod = {
            'nhb_trip_rate': nhb_trip_rates_path,
            'nhb_time_split': nhb_time_split_path,
            'export_path': export_path
        }
        return imports_nhb_prod

    def generate_nhb_attraction_imports(self,
                                        ) -> Dict[str, nd.PathLike]:
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

        # Creates inputs required for NHB Attractions
        export_path = os.path.join(self.notem_export_home, self._nhb_attr)

        imports_nhb_attr = {
            'export_path': export_path
        }
        return imports_nhb_attr

    def _generate_notem_seg_prod(self,
                                 ) -> Dict[int, nd.PathLike]:
        """
        Creates the notem segmented production paths.

        Creates dictionary of {year: notem segmented production paths} pairs.

        Returns
        -------
        notem_seg_prod:
            A dictionary containing {year: notem segmented production paths} pairs
        """
        notem_seg_prod = dict()

        for year in self.years:
            # Path to read notem segmentation productions
            notem_seg_production = os.path.join(self.notem_export_home, self._hb_prod,
                                                self._notem_seg_prod_fname % year)
            notem_seg_prod[year] = notem_seg_production

        return notem_seg_prod

    def _generate_notem_seg_nhb_prod(self,
                                     ) -> Dict[int, nd.PathLike]:
        """
        Creates the notem segmented NHB production paths.

        Creates dictionary of {year: notem segmented NHB production paths} pairs.

        Returns
        -------
        notem_seg_nhb_prod:
            A dictionary containing {year: notem segmented NHB production paths} pairs
        """
        notem_seg_nhb_prod = dict()

        for year in self.years:
            # Path to read notem segmentation productions
            notem_seg_nhb_production = os.path.join(self.notem_export_home, self._nhb_prod,
                                                    self._notem_seg_prod_fname % year)
            notem_seg_nhb_prod[year] = notem_seg_nhb_production

        return notem_seg_nhb_prod

    def _generate_notem_seg_attr(self,
                                 ) -> Dict[int, nd.PathLike]:
        """
        Creates the notem segmented attraction paths.

        Creates dictionary of {year: notem segmented attraction paths} pairs.

        Returns
        -------
        notem_seg_attr:
            A dictionary containing {year: notem segmented attraction paths} pairs
        """
        notem_seg_attr = dict()

        for year in self.years:
            # Path to read notem segmentation attractions
            notem_seg_attraction = os.path.join(self.notem_export_home, self._hb_attr,
                                                self._notem_seg_attr_fname % year)
            notem_seg_attr[year] = notem_seg_attraction

        return notem_seg_attr
