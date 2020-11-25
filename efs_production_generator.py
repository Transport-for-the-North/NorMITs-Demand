# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 12:13:07 2019

@author: Sneezy
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd

from typing import List
from typing import Dict

from functools import reduce

from tqdm import tqdm

import efs_constants as consts
from efs_constrainer import ForecastConstrainer
from demand_utilities import utils as du
# TODO: Move functions that can be static elsewhere.
#  Maybe utils?

# TODO: Tidy up the no longer needed functions -
#  Production model was re-written to use TMS method


class EFSProductionGenerator:
    
    def __init__(self,
                 tag_certainty_bounds=consts.TAG_CERTAINTY_BOUNDS):
        """
        #TODO
        """
        self.efs_constrainer = ForecastConstrainer()
        self.tag_certainty_bounds = tag_certainty_bounds
    
    def run(self,
            base_year: str,
            future_years: List[str],

            # Population growth
            population_growth: pd.DataFrame,
            population_constraint: pd.DataFrame,

            # Build import paths
            import_home: str,
            msoa_conversion_path: str,

            # Alternate population/production creation files
            lu_import_path: str = None,
            trip_rates_path: str = None,
            time_splits_path: str = None,
            mean_time_splits_path: str = None,
            mode_share_path: str = None,

            # Production control file
            ntem_control_dir: str = None,
            lad_lookup_dir: str = None,
            control_productions: bool = True,
            control_fy_productions: bool = True,

            # D-Log
            d_log: pd.DataFrame = None,
            d_log_split: pd.DataFrame = None,

            # Population constraints
            constraint_required: List[bool] = consts.DEFAULT_PRODUCTION_CONSTRAINTS,
            constraint_method: str = "Percentage",  # Percentage, Average
            constraint_area: str = "Designated",  # Zone, Designated, All
            constraint_on: str = "Growth",  # Growth, All
            constraint_source: str = "Grown Base",  # Default, Grown Base, Model Grown Base
            designated_area: pd.DataFrame = None,

            # Segmentation Controls
            m_needed: List[int] = consts.MODES_NEEDED,
            segmentation_cols: List[str] = None,
            external_zone_col: str = 'model_zone_id',
            lu_year: int = 2018,
            no_neg_growth: bool = True,
            population_infill: float = 0.001,

            # Handle outputs
            audits: bool = True,
            out_path: str = None,
            recreate_productions: bool = True,
            population_metric: str = "Population",  # Households, Population
            ) -> pd.DataFrame:
        """
        Production model for the external forecast system. This has been
        written to align with TMS production generation, with the addition of
        future year population growth and production generation.

        Performs the following functions:
            - Reads in the base year land use data to create the base year
              population numbers
            - Grows the base year population by pop_growth factors,
              resulting in future year population numbers.
            - Combines base and future year population numbers with trip_rates
              (currently the same across all years) to produce the base and
              future year production values (for all modes).
            - Finally, splits the produced productions to only return the
              desired mode. This dataframe is then returned.

        Parameters
        ----------
        base_year:
            The base year of the forecast.

        future_years:
            The future years to forecast.

        population_growth:
            dataframe containing the future year growth values for
            growing the base year population. Must be segmented by the same
            zoning system (at least) as land use data (usually msoa_zone_id).

        population_constraint:
            TODO: Need to clarify if population constrain is still needed,
             where the values come from, and how exactly the constrainer works.

        import_home:
            The home directory to find all the production imports. Usually
            Y:/NorMITs Demand/import

        msoa_conversion_path:
            Path to the file containing the conversion from msoa integer
            identifiers to the msoa string code identifiers. Hoping to remove
            this in a future update and align all of EFS to use msoa string
            code identifiers.

        lu_import_path:
            The path to alternate land use import data. If left as None, the
            production model will use the default land use data.

        trip_rates_path:
            The path to alternate trip rates data. If left as None, the
            production model will use the default trip rates data.

        time_splits_path:
            The path to alternate time splits data. If left as None, the
            production model will use the default time splits data.

        mean_time_splits_path:
            The path to alternate mean time splits data. If left as None, the
            production model will use the default mean time splits data.

        mode_share_path:
            The path to alternate mode share data. If left as None, the
            production model will use the default mode share data.

        ntem_control_dir:
            The path to alternate ntem control directory. If left as None, the
            production model will use the default land use data.

        lad_lookup_dir:
            The path to alternate lad to msoa import data. If left as None, the
            production model will use the default land use data.

        control_productions:
            Whether to control the generated production to the constraints
            given in ntem_control_dir or not.

        control_fy_productions:
            Whether to control the generated future year productions to the
            constraints given in ntem_control_dir or not. When running for
            scenarios other than the base NTEM, this should be False.

        d_log:
            TODO: Clarify what format D_log data comes in as

        d_log_split:
            See d_log

        constraint_required:
            See efs_constrainer.ForecastConstrainer()

        constraint_method:
            See efs_constrainer.ForecastConstrainer()

        constraint_area:
            See efs_constrainer.ForecastConstrainer()

        constraint_on:
            See efs_constrainer.ForecastConstrainer()

        constraint_source:
            See efs_constrainer.ForecastConstrainer()

        designated_area:
            See efs_constrainer.ForecastConstrainer()

        m_needed:
            Which mode to return productions for.

        segmentation_cols:
            The levels of segmentation that exist in the land use data. If
            not defined, will default to: ['area_type', 'traveller_type',
            'soc', 'ns', 'ca'].

        external_zone_col:
            The name of the zone column, as used externally to this production
            model. This is used to make sure this model can translate to the
            zoning name used internally in land_use and trip_rates data.

        lu_year:
            Which year the land_use data has been generated for. At the moment,
            if this is different to the base year and error is thrown. Used as
            a safety measure to make sure the user is warned if the base year
            changes without the land use.

        no_neg_growth:
            Whether to ensure there is no negative growth. If True, any growth
            values below 0 will be replaced with population_infill.

        population_infill:
            If no_neg_growth is True, this value will be used to replace all
            values that are less than 0.

        audits:
            Whether to output audits to the terminal during running. This can
            be used to monitor the population and production numbers being
            generated and constrained.

        out_path:
            Path to the directory to output the population and productions
            dataframes.

        recreate_productions:
            Whether to recreate the productions or not. If False, it will
            look in out_path for previously produced productions and return
            them. If none can be found, they will be generated.

        population_metric:
            No longer used - kept for now to retain all information from
            previous EFS. Will be removed in future.

        Returns
        -------
        Segmented_productions:
            Productions for mode m_needed, segmented by all segments possible
            in the input data.
        """
        # Return previously created productions if we can
        fname = 'MSOA_aggregated_productions.csv'
        final_output_path = os.path.join(out_path, fname)

        if not recreate_productions and os.path.isfile(final_output_path):
            print("Found some already produced productions. Using them!")
            return pd.read_csv(final_output_path)

        # Init
        internal_zone_col = 'msoa_zone_id'
        all_years = [str(x) for x in [base_year] + future_years]
        integrate_d_log = d_log is not None and d_log_split is not None
        if integrate_d_log:
            d_log = d_log.copy()
            d_log_split = d_log_split.copy()

        # TODO: Make this more adaptive
        # Set the level of segmentation being used
        if segmentation_cols is None:
            segmentation_cols = [
                'area_type',
                'traveller_type',
                'soc',
                'ns',
                'ca'
            ]

        # Fix column naming if different
        if external_zone_col != internal_zone_col:
            population_growth = population_growth.copy().rename(
                columns={external_zone_col: internal_zone_col}
            )
            designated_area = designated_area.copy().rename(
                columns={external_zone_col: internal_zone_col}
            )
            population_constraint = population_constraint.rename(
                columns={external_zone_col: internal_zone_col}
            )

        # TODO: Deal with case where land use year and base year don't match
        if str(lu_year) != str(base_year):
            raise ValueError("The base year and land use year are not the "
                             "same. Don't know how to deal with that at the"
                             "moment.")

        # Build paths to the needed files
        imports = build_production_imports(
            import_home=import_home,
            lu_import_path=lu_import_path,
            trip_rates_path=trip_rates_path,
            time_splits_path=time_splits_path,
            mean_time_splits_path=mean_time_splits_path,
            mode_share_path=mode_share_path,
            ntem_control_dir=ntem_control_dir,
            lad_lookup_dir=lad_lookup_dir,
            set_controls=control_productions
        )

        if population_metric == "households":
            raise ValueError("Production Model has changed. Households growth "
                             "is not currently supported.")

        # ## BASE YEAR POPULATION ## #
        print("Loading the base year population data...")
        base_year_pop = get_land_use_data(imports['land_use'],
                                          msoa_path=msoa_conversion_path,
                                          segmentation_cols=segmentation_cols)
        base_year_pop = base_year_pop.rename(columns={'people': base_year})

        # Audit population numbers
        du.print_w_toggle("Base Year Population: %d" % base_year_pop[base_year].sum(),
                          echo=audits)

        # ## FUTURE YEAR POPULATION ## #
        print("Generating future year population data...")
        population = du.grow_to_future_years(
            base_year_df=base_year_pop,
            growth_df=population_growth,
            base_year=base_year,
            future_years=future_years,
            no_neg_growth=no_neg_growth,
            infill=population_infill
        )

        # ## CONSTRAIN POPULATION ## #
        if constraint_required[0] and (constraint_source != "model grown base"):
            print("Performing the first constraint on population...")
            population = self.efs_constrainer.run(
                population,
                constraint_method,
                constraint_area,
                constraint_on,
                population_constraint,
                base_year,
                all_years,
                designated_area,
                internal_zone_col
            )
        elif constraint_source == "model grown base":
            print("Generating model grown base constraint for use on "
                  "development constraints...")
            population_constraint = population.copy()

        # ## INTEGRATE D-LOG ## #
        if integrate_d_log:
            print("Integrating the development log...")
            raise NotImplementedError("D-Log population integration has not "
                                      "yet been implemented.")
        else:
            # If not integrating, no need for another constraint
            constraint_required[1] = False

        # ## POST D-LOG CONSTRAINT ## #
        if constraint_required[1]:
            print("Performing the post-development log constraint on population...")
            population = self.efs_constrainer.run(
                population,
                constraint_method,
                constraint_area,
                constraint_on,
                population_constraint,
                base_year,
                all_years,
                designated_area,
                internal_zone_col
            )

        # Reindex and sum
        group_cols = [internal_zone_col] + segmentation_cols
        index_cols = group_cols.copy() + all_years
        population = population.reindex(index_cols, axis='columns')
        population = population.groupby(group_cols).sum().reset_index()

        # Population Audit
        if audits:
            print('\n', '-'*15, 'Population Audit', '-'*15)
            for year in all_years:
                print('. Total population for year %s is: %.4f'
                      % (year, population[year].sum()))
            print('\n')

        # Convert back to MSOA codes for output and productions
        population = du.convert_msoa_naming(
            population,
            msoa_col_name=internal_zone_col,
            msoa_path=msoa_conversion_path,
            to='string'
        )

        # Write the produced population to file
        if out_path is None:
            print("WARNING! No output path given. "
                  "Not writing populations to file.")
        else:
            print("Writing population to file...")
            population.to_csv(os.path.join(out_path, "MSOA_population.csv"),
                              index=False)

        # ## CREATE PRODUCTIONS ## #
        print("Population generated. Converting to productions...")
        productions = generate_productions(
            population=population,
            group_cols=group_cols,
            base_year=base_year,
            future_years=future_years,
            trip_rates_path=imports['trip_rates'],
            time_splits_path=imports['time_splits'],
            mean_time_splits_path=imports['mean_time_splits'],
            mode_share_path=imports['mode_share'],
            audit_dir=out_path,
            ntem_control_dir=imports['ntem_control'],
            lad_lookup_dir=imports['lad_lookup'],
            control_fy_productions=control_fy_productions,
        )

        # Write productions to file
        if out_path is None:
            print("WARNING! No output path given. "
                  "Not writing productions to file.")
        else:
            print("Writing productions to file...")
            fname = 'MSOA_production_trips.csv'
            productions.to_csv(os.path.join(out_path, fname), index=False)

        # ## CONVERT TO OLD EFS FORMAT ## #
        # Make sure columns are the correct data type
        productions['area_type'] = productions['area_type'].astype(int)
        productions['m'] = productions['m'].astype(int)
        productions['p'] = productions['p'].astype(int)
        productions['ca'] = productions['ca'].astype(int)
        productions.columns = productions.columns.astype(str)

        # Aggregate tp
        index_cols = list(productions)
        index_cols.remove('tp')

        group_cols = index_cols.copy()
        for year in all_years:
            group_cols.remove(year)

        # Group and sum
        productions = productions.reindex(index_cols, axis='columns')
        productions = productions.groupby(group_cols).sum().reset_index()

        # Extract just the needed mode
        mask = productions['m'].isin(m_needed)
        productions = productions[mask]
        productions = productions.drop('m', axis='columns')

        # Rename columns so output of this function call is the same
        # as it was before the re-write
        productions = du.convert_msoa_naming(
            productions,
            msoa_col_name=internal_zone_col,
            msoa_path=msoa_conversion_path,
            to='int'
        )

        productions = productions.rename(
            columns={
                internal_zone_col: external_zone_col,
                'ca': 'car_availability_id',
                'p': 'purpose_id',
                'area_type': 'area_type_id'
            }
        )

        fname = 'MSOA_aggregated_productions.csv'
        productions.to_csv(os.path.join(out_path, fname), index=False)

        return productions

    def grow_population(self,
                        population_values: pd.DataFrame,
                        population_growth: pd.DataFrame,
                        base_year: str = None,
                        future_years: List[str] = List[None],
                        growth_merge_col: str = 'msoa_zone_id'
                        ) -> pd.DataFrame:
        # TODO: Write grow_population() doc
        # Init
        all_years = [base_year] + future_years

        print("Adjusting population growth to base year...")
        population_growth = du.convert_growth_off_base_year(
            population_growth,
            base_year,
            future_years
        )

        print("Growing population from base year...")
        grown_population = du.get_growth_values(
            population_values,
            population_growth,
            base_year,
            future_years,
            merge_col=growth_merge_col
        )

        # Make sure there is no minus growth
        for year in all_years:
            mask = (grown_population[year] < 0)
            grown_population.loc[mask, year] = self.pop_infill

        # Add base year back in to get full grown values
        grown_population = du.growth_recombination(
            grown_population,
            base_year_col=base_year,
            future_year_cols=future_years,
            drop_base_year=False
        )

        return grown_population

    def grow_by_population(self,
                           population_growth: pd.DataFrame,
                           population_values: pd.DataFrame,
                           population_constraint: pd.DataFrame,
                           d_log: pd.DataFrame = None,
                           d_log_split: pd.DataFrame = None,
                           minimum_development_certainty: str = "MTL",  # "NC", "MTL", "RF", "H"
                           constraint_required: List[bool] = consts.DEFAULT_PRODUCTION_CONSTRAINTS,
                           constraint_method: str = "Percentage",  # Percentage, Average
                           constraint_area: str = "Designated",  # Zone, Designated, All
                           constraint_on: str = "Growth",  # Growth, All
                           constraint_source: str = "Grown Base",  # Default, Grown Base, Model Grown Base
                           designated_area: pd.DataFrame = None,
                           base_year_string: str = None,
                           model_years: List[str] = List[None]
                           ) -> pd.DataFrame:
        """
        TODO: Write grow_by_population() doc
        """
        # ## GROW POPULATION
        grown_population = self.population_grower(
            population_growth,
            population_values,
            base_year_string,
            model_years
        )

        # ## initial population metric constraint
        if constraint_required[0] and (constraint_source != "model grown base"):
            print("Performing the first constraint on population...")
            grown_population = self.efs_constrainer.run(
                grown_population,
                constraint_method,
                constraint_area,
                constraint_on,
                population_constraint,
                base_year_string,
                model_years,
                designated_area
            )
        elif constraint_source == "model grown base":
            print("Generating model grown base constraint for use on "
                  + "development constraints...")
            population_constraint = grown_population.copy()

        # ## D-LOG INTEGRATION
        if d_log is not None and d_log_split is not None:
            print("Including development log...")
            development_households = self.development_log_house_generator(
                d_log,
                d_log_split,
                minimum_development_certainty,
                model_years
            )
            raise NotImplementedError("D-Log pop generation not "
                                      "yet implemented.")
            # TODO: Generate population
        else:
            print("Not including development log...")

        # ## post-development log constraint
        if constraint_required[1]:
            print("Performing the post-development log on population...")
            grown_population = self.efs_constrainer.run(
                grown_population,
                constraint_method,
                constraint_area,
                constraint_on,
                population_constraint,
                base_year_string,
                model_years,
                designated_area
            )
        return grown_population

    def grow_by_households(self,
                           population_constraint: pd.DataFrame,
                           population_split: pd.DataFrame,
                           households_growth: pd.DataFrame,
                           households_values: pd.DataFrame,
                           households_constraint: pd.DataFrame,
                           housing_split: pd.DataFrame,
                           housing_occupancy: pd.DataFrame,
                           d_log: pd.DataFrame = None,
                           d_log_split: pd.DataFrame = None,
                           minimum_development_certainty: str = "MTL",  # "NC", "MTL", "RF", "H"
                           constraint_required: List[bool] = consts.DEFAULT_PRODUCTION_CONSTRAINTS,
                           constraint_method: str = "Percentage",  # Percentage, Average
                           constraint_area: str = "Designated",  # Zone, Designated, All
                           constraint_on: str = "Growth",  # Growth, All
                           constraint_source: str = "Grown Base",  # Default, Grown Base, Model Grown Base
                           designated_area: pd.DataFrame = None,
                           base_year_string: str = None,
                           model_years: List[str] = List[None]
                           ) -> pd.DataFrame:
        # ## GROW HOUSEHOLDS
        grown_households = self.households_grower(
            households_growth,
            households_values,
            base_year_string,
            model_years
        )

        # ## initial population metric constraint
        if constraint_required[0] and (constraint_source != "model grown base"):
            print("Performing the first constraint on households...")
            grown_households = self.efs_constrainer.run(
                grown_households,
                constraint_method,
                constraint_area,
                constraint_on,
                households_constraint,
                base_year_string,
                model_years,
                designated_area
            )
        elif constraint_source == "model grown base":
            print("Generating model grown base constraint for use on "
                  + "development constraints...")
            households_constraint = grown_households.copy()

        # ## SPLIT HOUSEHOLDS
        split_households = self.split_housing(
            grown_households,
            housing_split,
            base_year_string,
            model_years
        )

        # ## D-LOG INTEGRATION
        if d_log is not None and d_log_split is not None:
            print("Including development log...")
            # ## DEVELOPMENT SPLIT
            split_development_households = self.development_log_house_generator(
                d_log,
                d_log_split,
                minimum_development_certainty,
                model_years
            )

            # ## COMBINE BASE + DEVELOPMENTS
            split_households = self.combine_households_and_developments(
                split_households,
                split_development_households
            )

        else:
            print("Not including development log...")

        # ## post-development log constraint
        if constraint_required[1]:
            print("Performing the post-development log constraint on "
                  "households...")
            split_households = self.efs_constrainer.run(
                split_households,
                constraint_method,
                constraint_area,
                constraint_on,
                households_constraint,
                base_year_string,
                model_years,
                designated_area
            )

        # ## POPULATION GENERATION
        population = self.generate_housing_occupancy(
            split_households,
            housing_occupancy,
            base_year_string,
            model_years
        )

        # ## ENSURE WE HAVE NO MINUS POPULATION
        for year in model_years:
            mask = (population[year] < 0)
            population.loc[mask, year] = self.pop_infill

        # ## secondary post-development constraint
        # (used for matching HH pop)
        if constraint_required[2] and (constraint_source != "model grown base"):
            print("Constraining to population on population in "
                  "households...")
            population = self.efs_constrainer.run(
                population,
                constraint_method,
                constraint_area,
                constraint_on,
                population_constraint,
                base_year_string,
                model_years,
                designated_area
            )
        elif constraint_source == "model grown base":
            print("Population constraint in households metric selected.")
            print("No way to control population using model grown base "
                  + "constraint source.")

        # ## SPLIT POP (On traveller type)
        split_population = self.split_population(
            population,
            population_split,
            base_year_string,
            model_years
        )

        # ## RECOMBINING POP
        final_population = self.growth_recombination(split_population,
                                                     "base_year_population",
                                                     model_years)

        final_population.sort_values(
            by=[
                "model_zone_id",
                "property_type_id",
                "traveller_type_id"
            ],
            inplace=True
        )
        return final_population

    def production_generation(self,
                              population: pd.DataFrame,
                              area_type: pd.DataFrame,
                              trip_rate: pd.DataFrame,
                              year_list: List[str]
                              ) -> pd.DataFrame:
        """
        #TODO
        """
        population = population.copy()
        area_type = area_type.copy()
        trip_rate = trip_rate.copy()

        # Multiple Population zones belong to each MSOA area  type
        population = pd.merge(
            population,
            area_type,
            on=["model_zone_id"]
        )

        # Calculate the trips of each traveller in an area, based on
        # their trip rates
        trips = pd.merge(
            population,
            trip_rate,
            on=["traveller_type_id", "area_type_id"],
            suffixes=("", "_trip_rate")
        )
        for year in year_list:
            trips.loc[:, year] = trips[year] * trips[year + "_trip_rate"]

        # Extract just the needed columns
        group_by_cols = [
            "model_zone_id",
            "purpose_id",
            "traveller_type_id",
            "soc",
            "ns",
            "area_type_id"
        ]
        needed_columns = group_by_cols.copy()
        needed_columns.extend(year_list)
        trips = trips[needed_columns]

        productions = trips.groupby(
            by=group_by_cols,
            as_index=False
        ).sum()

        return productions

    def convert_to_average_weekday(self,
                                   production_dataframe: pd.DataFrame,
                                   all_years: List[str]
                                   ) -> pd.DataFrame:
        """
        #TODO
        """
        output_dataframe = production_dataframe.copy()

        for year in all_years:
            output_dataframe.loc[:, year] = output_dataframe.loc[:, year] / 5

        return output_dataframe

    def population_grower(self,
                          population_growth: pd.DataFrame,
                          population_values: pd.DataFrame,
                          base_year: str,
                          year_string_list: List[str]
                          ) -> pd.DataFrame:
        # get population growth from base year
        print("Adjusting population growth to base year...")
        population_growth = du.convert_growth_off_base_year(
                population_growth,
                base_year,
                year_string_list
                )
        print("Adjusted population growth to base year!")
        
        
        print("Growing population from base year...")
        grown_population = du.get_growth_values(population_values,
                                                population_growth,
                                                base_year,
                                                year_string_list)
        print("Grown population from base year!")
        
        return grown_population
            
    def households_grower(self,
                         households_growth: pd.DataFrame,
                         households_values: pd.DataFrame,
                         base_year: str,
                         year_string_list: List[str]
                         ) -> pd.DataFrame:
        
        households_growth = households_growth.copy()
        households_values = households_values.copy()
        print("Adjusting households growth to base year...")
        households_growth = du.convert_growth_off_base_year(
                households_growth,
                base_year,
                year_string_list
                )
        print("Adjusted households growth to base year!")
        
        print("Growing households from base year...")
        grown_households = self.get_grown_values(households_values,
                                                 households_growth,
                                                 "base_year_households",
                                                 year_string_list)
        print("Grown households from base year!")
        
        return grown_households

    
    def split_housing(self,
                      households_dataframe: pd.DataFrame,
                      housing_split_dataframe: pd.DataFrame,
                      base_year_string: str,
                      all_years: List[str]
                      ) -> pd.DataFrame:
        households_dataframe = households_dataframe.copy()
        housing_split_dataframe = housing_split_dataframe.copy()
        
        split_households_dataframe = pd.merge(
                households_dataframe,
                housing_split_dataframe,
                on = ["model_zone_id"],
                suffixes = {"", "_split"}
                )

        # Calculate the number of houses belonging to each split
        split_households_dataframe["base_year_households"] = (
                split_households_dataframe["base_year_households"]
                *
                split_households_dataframe[base_year_string + "_split"]
                )
        
        for year in all_years:
            # we iterate over each zone
            split_households_dataframe[year] = (
                    split_households_dataframe[year]
                    *
                    split_households_dataframe[year + "_split"]
                    )

        # extract just the needed columns
        required_columns = [
                "model_zone_id",
                "property_type_id",
                "base_year_households"
                ]
        required_columns.extend(all_years)
        split_households_dataframe = split_households_dataframe[required_columns]

        return split_households_dataframe
    
    def split_population(self,
                      population_dataframe: pd.DataFrame,
                      population_split_dataframe: pd.DataFrame,
                      base_year_string: str,
                      all_years: List[str]
                      ) -> pd.DataFrame:
        population_dataframe = population_dataframe.copy()
        population_split_dataframe = population_split_dataframe.copy()
        
        split_population_dataframe = pd.merge(
                population_dataframe,
                population_split_dataframe,
                on = ["model_zone_id", "property_type_id"],
                suffixes = {"", "_split"}
                )
                
        split_population_dataframe["base_year_population"] = (
                split_population_dataframe["base_year_population"]
                *
                split_population_dataframe[base_year_string + "_split"]
                )
        
        for year in all_years:
            # we iterate over each zone
            # create zone mask
            split_population_dataframe[year] = (
                    split_population_dataframe[year]
                    *
                    split_population_dataframe[year + "_split"]
                    )

        # Extract the required columns
        required_columns = [
                "model_zone_id",
                "property_type_id",
                "traveller_type_id",
                "base_year_population"
                ]
        required_columns.extend(all_years)
        split_population_dataframe = split_population_dataframe[required_columns]

        return split_population_dataframe
    
    def generate_housing_occupancy(self,
                                   households_dataframe: pd.DataFrame,
                                   housing_occupancy_dataframe: pd.DataFrame,
                                   base_year_string: str,
                                   all_years: List[str]
                                   ) -> pd.DataFrame:
        """
        #TODO
        """
        households_dataframe = households_dataframe.copy()
        housing_occupancy_dataframe = housing_occupancy_dataframe.copy()
        
        households_population = pd.merge(
                households_dataframe,
                housing_occupancy_dataframe,
                on = ["model_zone_id", "property_type_id"],
                suffixes = {"", "_occupancy"}
                )
        
        households_population["base_year_population"] = (
                households_population["base_year_households"]
                *
                households_population[base_year_string + "_occupancy"]
                )
        
        pop_columns = [
                "model_zone_id",
                "property_type_id",
                "base_year_population"
                ]
        pop_dictionary = {}
        
        for year in all_years:
            households_population[year + "_population"] = (
                    households_population[year]
                    *
                    households_population[year + "_occupancy"]
                    )
            pop_columns.append(year + "_population")
            pop_dictionary[year + "_population"] = year

        # Extract just the needed columns and rename to years
        households_population = households_population[pop_columns]
        households_population = households_population.rename(columns=pop_dictionary)
        
        return households_population
    
    def development_log_house_generator(self,
                                        development_log: pd.DataFrame,
                                        development_log_split: pd.DataFrame,
                                        minimum_development_certainty: str,
                                        all_years: List[str]
                                        ) -> pd.DataFrame:
        """
        #TODO
        """
        development_log = development_log.copy()
        development_log_split = development_log_split.copy()
        
        development_log = development_log[
                development_log["dev_type"] == "Housing"
                ]
        
        webtag_certainty_bounds = self.tag_certainty_bounds[
                minimum_development_certainty
                ]
        
        development_log["used"] = False 
        
        for webtag_certainty in webtag_certainty_bounds:
            mask = (development_log["webtag_certainty"] == webtag_certainty)
            development_log.loc[
                    mask,
                    "used"
                    ] = True
        
        required_columns = [
                "model_zone_id"
                ]
        
        required_columns.extend(all_years)
        
        development_log = development_log[
                development_log["used"] == True
                ]
        
        development_log = development_log[required_columns]
        
        development_households = pd.merge(
                development_log,
                development_log_split,
                on = ["model_zone_id"],
                suffixes = {"", "_split"}
                )
        
        for year in all_years:
            # we iterate over each zone
            development_households[year] = (
                    development_households[year]
                    *
                    development_households[year + "_split"]
                    )
        
        required_columns = [
                "model_zone_id",
                "property_type_id"
                ]
        
        required_columns.extend(all_years)        
        
        development_households = development_households[required_columns]
        
        development_households = development_households.groupby(
                by = [
                        "model_zone_id",
                        "property_type_id"
                        ],
                as_index = False
                ).sum()
        
        return development_households
    
    def combine_households_and_developments(self,
                                            split_households: pd.DataFrame,
                                            split_developments: pd.DataFrame
                                            ) -> pd.DataFrame:
        """
        #TODO
        """
        split_households = split_households.copy()
        split_developments = split_developments.copy()
        
        combined_households = split_households.append(
                split_developments,
                ignore_index = True
                )
        
        combined_households = combined_households.groupby(
                by = [
                        "model_zone_id",
                        "property_type_id"
                        ],
                as_index = False
                ).sum()
        
        return combined_households


class NhbProductionModel:

    def __init__(self,
                 import_home: str,
                 export_home: str,
                 model_name: str,
                 seg_level: str = 'tfn',

                 all_years: List[str] = consts.ALL_YEARS_STR,

                 # Alternate input paths
                 hb_prods_path: str = None,
                 hb_attrs_path: str = None,
                 trip_rates_path: str = None,
                 mode_splits_path: str = None,
                 time_splits_path: str = None,

                 zoning_system: str = 'msoa',
                 ):
        # TODO: Write NhbProductionModel docs
        # Validate inputs
        zoning_system = du.validate_zoning_system(zoning_system)
        model_name = du.validate_model_name(model_name)
        seg_level = du.validate_seg_level(seg_level)

        # Assign
        self.model_name = model_name
        self.seg_level = seg_level
        self.all_years = all_years

        self.zoning_system = zoning_system
        self.zone_col = '%s_zone_id' % zoning_system

        self.imports = self._build_paths(
            import_home=import_home,
            export_home=export_home,
            hb_prods_path=hb_prods_path,
            hb_attrs_path=hb_attrs_path,
            trip_rates_path=trip_rates_path,
            mode_splits_path=mode_splits_path,
            time_splits_path=time_splits_path
        )

        if seg_level == 'tfn':
            self.segments = ['area_type', 'p', 'soc', 'ns', 'ca']
        else:
            raise ValueError(
                "'%s' is a valid segmentation level, but I don't have a way "
                "of determining which segments to use for it. You should add "
                "one!" % seg_level
            )

    def _build_paths(self,
                     import_home: str,
                     export_home: str,
                     hb_prods_path: str,
                     hb_attrs_path: str,
                     trip_rates_path: str,
                     mode_splits_path: str,
                     time_splits_path: str,
                     ) -> Dict[str, str]:
        """
        Builds a dictionary of import paths, forming a standard calling
        procedure for imports. Arguments allow default paths to be replaced.
        """
        # Set all unset import paths to default values
        if hb_prods_path is None:
            fname = consts.HB_PRODS_FNAME % (self.zoning_system, 'hb')
            hb_prods_path = os.path.join(export_home,
                                         consts.PRODUCTIONS_DIRNAME,
                                         fname)

        if hb_attrs_path is None:
            fname = consts.HB_ATTRS_FNAME % (self.zoning_system, 'hb')
            hb_attrs_path = os.path.join(export_home,
                                         consts.ATTRACTIONS_DIRNAME,
                                         fname)

        if trip_rates_path is None:
            trip_rates_path = os.path.join(import_home,
                                           consts.NHB_PARAMS_DIRNAME,
                                           'nhb_ave_wday_enh_trip_rates.csv')

        if mode_splits_path is None:
            mode_splits_path = os.path.join(import_home,
                                            consts.NHB_PARAMS_DIRNAME,
                                            'nhb_ave_wday_mode_split.csv')

        if time_splits_path is None:
            time_splits_path = os.path.join(import_home,
                                            consts.NHB_PARAMS_DIRNAME,
                                            'nhb_ave_wday_time_split.csv')

        imports = {
            'productions': hb_prods_path,
            'attractions': hb_attrs_path,
            'trip_rates': trip_rates_path,
            'mode_splits': mode_splits_path,
            'time_splits': time_splits_path
        }

        return imports

    def apply_mode_splits(self,
                          nhb_prods: pd.DataFrame,
                          ) -> pd.DataFrame:
        # TODO: Write NhbProductionModel.apply_mode_splits() docs
        # Init
        mode_splits = pd.read_csv(self.imports['mode_splits'])

    def apply_time_splits(self,
                          nhb_prods: pd.DataFrame,
                          ) -> pd.DataFrame:
        # TODO: Write NhbProductionModel.apply_time_splits() docs
        raise NotImplementedError

    def run(self,
            soc_col: str = 'soc',
            ns_col: str = 'ns',
            nhb_p_col: str = 'nhb_p',
            trip_rate_col: str = 'trip_rate',
            verbose: bool = True
            ) -> pd.DataFrame:
        # TODO: Write NhbProductionModel.run() docs

        # Read in files
        prods = pd.read_csv(self.imports['productions'])
        attrs = pd.read_csv(self.imports['attractions'])
        nhb_trip_rates = pd.read_csv(self.imports['trip_rates'])

        # Ensure correct column types
        if soc_col in prods:
            prods[soc_col] = prods[soc_col].astype('str')

        if ns_col in prods:
            prods[ns_col] = prods[ns_col].astype('str')

        # Determine all unique segments
        unq_segs = dict()
        for segment in self.segments:
            if segment not in prods:
                raise ValueError(
                    "Cannot get segment '%s' from the productions as it isn't "
                    "in there!" % segment
                )

            unq_segs[segment] = prods[segment].unique().tolist()

        # Calculate the nhb productions per segment
        nhb_trips_ph = list()
        loop_gen = du.segment_loop_generator(unq_segs)
        total = reduce(lambda x, y: x * y, [len(i) for i in unq_segs])
        desc = "Calculating NHB Productions"
        for seg_vals in tqdm(loop_gen, total=total, desc=desc, disable=not verbose):
            print(seg_vals)

            # ## PSEUDO DISTRIBUTE EACH SEGMENT ## #
            # We do this to retain segments from productions

            # Filter the productions and attractions
            p_subset = du.filter_by_segmentation(prods, seg_vals, fit=True)
            a_subset = du.filter_by_segmentation(attrs, seg_vals, fit=True)

            # Remove all segmentation from the attractions
            group_cols = [self.zone_col]
            index_cols = group_cols.copy() + self.all_years
            a_subset = a_subset.reindex(index_cols, axis='columns')
            a_subset = a_subset.groupby(group_cols).sum().reset_index()

            # Balance P/A to pseudo distribute
            a_subset = du.balance_a_to_p(
                productions=p_subset,
                attractions=a_subset,
                unique_cols=self.all_years,
            )

            # ## APPLY NHB TRIP RATES ## #
            # Subset the trip_rates
            tr_index = [nhb_p_col, trip_rate_col]
            tr_subset = du.filter_by_segmentation(nhb_trip_rates, seg_vals, fit=True)
            tr_subset = tr_subset.reindex(tr_index, axis='columns')

            # Validate
            if len(tr_subset) > len(consts.ALL_NHB_P):
                raise du.NormitsDemandError(
                    "We have more than %d rows after filtering the nhb trip "
                    "rates. There are probably duplicates in the filter "
                    "somehow" % len(consts.ALL_NHB_P)
                )

            # Convert to a dictionary for speed
            tr_dict = dict(zip(tr_subset[nhb_p_col].values,
                               tr_subset[trip_rate_col].values))
            del tr_subset

            # Build the trip rates data for this segment
            for p, trip_rate in tr_dict.items():
                # Get the actual productions
                nhb_prods_df = a_subset.copy()
                for year in self.all_years:
                    nhb_prods_df[year] *= trip_rate

                # Store for compile later
                seg_nhb_prods = seg_vals.copy()
                seg_nhb_prods.update({
                    'nhb_p': p,
                    'df': nhb_prods_df,
                })
                nhb_trips_ph.append(seg_nhb_prods)

            # break
        # End of calculate the nhb productions per segment

        # Compile segmented
        # TODO: functionalise compile
        concat_ph = list()
        for seg_prod in nhb_trips_ph:
            # Add all segmentation cols back into df
            df = seg_prod.pop('df')
            for seg_name, seg_val in seg_prod.items():
                df[seg_name] = seg_val
            concat_ph.append(df)
        nhb_prods = pd.concat(concat_ph)

        # Reindex and tidy
        group_cols = [self.zone_col] + self.segments + ['nhb_p']
        index_cols = group_cols.copy() + self.all_years
        nhb_prods = nhb_prods.reindex(index_cols, axis='columns')
        nhb_prods = nhb_prods.groupby(group_cols).sum().reset_index()

        # Apply further splits to the productions
        nhb_prods = self.apply_mode_splits(nhb_prods)
        nhb_prods = self.apply_time_splits(nhb_prods)

        # Contrain!
        print(nhb_prods)
        print(list(nhb_prods))

        return nhb_prods



def build_production_imports(import_home: str,
                             lu_import_path: str = None,
                             trip_rates_path: str = None,
                             time_splits_path: str = None,
                             mean_time_splits_path: str = None,
                             mode_share_path: str = None,
                             ntem_control_dir: str = None,
                             lad_lookup_dir: str = None,
                             set_controls: bool = True
                             ) -> Dict[str, str]:
    """
    Builds a dictionary of production import paths, forming a standard calling
    procedure for production imports. Arguments allow default paths to be
    replaced.

    Parameters
    ----------
    import_home:
        The base path to base all of the other import paths from. This
        should usually be "Y:/NorMITs Demand/import" for business as usual.

    lu_import_path:
        An alternate land use import path to use. File will need to follow the
        same format as default file.

    trip_rates_path:
        An alternate trip rates import path to use. File will need to follow the
        same format as default file.

    time_splits_path:
        An alternate time splits import path to use. File will need to follow
        the same format as default file.

    mean_time_splits_path:
        An alternate mean time splits import path to use. File will need to
        follow the same format as default file.

    mode_share_path:
        An alternate mode share import path to use. File will need to follow
        the same format as default file.

    ntem_control_dir:
        An alternate ntem control directory to use. File will need to follow
        the same format as default files.

    lad_lookup_dir:
        An alternate lad lookup directory to use. File will need to follow
        the same format as default file.

    set_controls:
        If False 'ntem_control' and 'lad_lookup' outputs will be set to None,
        regardless of any other inputs.

    Returns
    -------
    import_dict:
        A dictionary of paths with the following keys:
        'land_use',
        'trip_rates',
        'time_splits',
        'mean_time_splits',
        'mode_share_path',
        'ntem_control',
        'lad_lookup',

    """
    # Set all unset import paths to default values
    if lu_import_path is None:
        path = 'land use\land_use_output_msoa.csv'
        lu_import_path = os.path.join(import_home, path)

    if trip_rates_path is None:
        path = 'tfn_segment_production_params\hb_trip_rates.csv'
        trip_rates_path = os.path.join(import_home, path)

    if time_splits_path is None:
        path = 'tfn_segment_production_params\hb_time_split.csv'
        time_splits_path = os.path.join(import_home, path)

    if mean_time_splits_path is None:
        path = 'tfn_segment_production_params\hb_ave_time_split.csv'
        mean_time_splits_path = os.path.join(import_home, path)

    if mode_share_path is None:
        path = 'tfn_segment_production_params\hb_mode_split.csv'
        mode_share_path = os.path.join(import_home, path)

    if set_controls and ntem_control_dir is None:
        path = 'ntem_constraints'
        ntem_control_dir = os.path.join(import_home, path)

    if set_controls and lad_lookup_dir is None:
        lad_lookup_dir = import_home

    # Assign to dict
    imports = {
        'land_use': lu_import_path,
        'trip_rates': trip_rates_path,
        'time_splits': time_splits_path,
        'mean_time_splits': mean_time_splits_path,
        'mode_share': mode_share_path,
        'ntem_control': ntem_control_dir,
        'lad_lookup': lad_lookup_dir
    }

    return imports


def get_land_use_data(land_use_path: str,
                      msoa_path: str = None,
                      segmentation_cols: List[str] = None,
                      lu_zone_col: str = 'msoa_zone_id'
                      ) -> pd.DataFrame:
    """
    Reads in land use outputs and aggregates up to land_use_cols

    Parameters
    ----------
    land_use_path:
        Path to the land use output file to import

    msoa_path:
        Path to the msoa file for converting from msoa string ids to
        integer ids. If left as None, then MSOA codes are returned instead

    segmentation_cols:
        The columns to keep in the land use data. Must include. If None,
        defaults to:
         [
            'area_type',
            'traveller_type',
            'soc',
            'ns',
        ]

    lu_zone_col:
        The name of the column in the land use data that refers to the zones.

    Returns
    -------
    population:
        Population data segmented by segmentation_cols. Will also include
        lu_zone_col and people cols from land use.
    """
    # Init
    if segmentation_cols is None:
        # Assume full segmentation if not told otherwise
        segmentation_cols = [
            'area_type',
            'traveller_type',
            'soc',
            'ns',
            'ca'
        ]
    land_use_cols = [lu_zone_col] + segmentation_cols + ['people']

    # Set up the columns to keep
    group_cols = land_use_cols.copy()
    group_cols.remove('people')

    # Read in Land use
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore')
        land_use = pd.read_csv(land_use_path)

    # Drop a lot of columns, group and sum the remaining

    land_use = land_use.reindex(land_use_cols, axis=1).groupby(
        group_cols
    ).sum().reset_index().sort_values(land_use_cols).reset_index()
    del group_cols

    # Convert to msoa zone numbers if needed
    if msoa_path is not None:
        land_use = du.convert_msoa_naming(
            land_use,
            msoa_col_name=lu_zone_col,
            msoa_path=msoa_path,
            to='int'
        )

    return land_use


def merge_pop_trip_rates(population: pd.DataFrame,
                         group_cols: List[str],
                         trip_rates_path: str,
                         time_splits_path: str,
                         mean_time_splits_path: str,
                         mode_share_path: str,
                         audit_out: str,
                         control_path: str = None,
                         lad_lookup_dir: str = None,
                         lad_lookup_name: str = consts.DEFAULT_LAD_LOOKUP,
                         tp_needed: List[int] = consts.TP_NEEDED,
                         traveller_type_col: str = 'traveller_type',
                         ) -> pd.DataFrame:
    """
    Converts a single year of population into productions

    Carries out the following actions:
        - Calculates the weekly trip rates
        - Convert to average weekday trip rate, and split by time period
        - Further splits the productions by mode
        - Optionally constrains to the values in control_path

    Parameters
    ----------
    population:
        Dataframe containing the segmented population values for this year.

    group_cols:
        A list of all non-unique columns in population. This will be used to
        retain any already given segmentation.

    trip_rates_path:
        Path to the file of trip rates data. Will try to merge with the
        population on all possible columns.

    time_splits_path:
        Path to the file of time splits by ['area_type', 'traveller_type', 'p'].

    mean_time_splits_path:
        Path to the file of mean time splits by 'p'

    mode_share_path:
        Path to the file of mode shares by ['area_type', 'ca', 'p']

    audit_out:
        The directory to write out any audit files produced.

    control_path:
        Path to the file containing the data to control the produced
        productions to. If left as None, no control will be carried out.


    lad_lookup_dir:
        Path to the file containing the conversion from msoa zoning to LAD
        zoning, to be used for controlling the productions. If left as None, no
        control will be carried out.

    lad_lookup_name:
        The name of the file in lad_lookup_dir that contains the msoa zoning
        to LAD zoning conversion.

    tp_needed:
        A list of the time periods to split the productions by.

    traveller_type_col:
        The name of the column in population that contains the traveller type
        information.

    Returns
    -------
    Productions:
        The population converted to productions for this year. Will try to keep
        all segmentation in the given population if possible, and add more.
    """
    # Init
    do_ntem_control = control_path is not None and lad_lookup_dir is not None

    group_cols = group_cols.copy()
    group_cols.insert(2, 'p')

    index_cols = group_cols.copy()
    index_cols.append('trips')

    # ## GET WEEKLY TRIP RATE FROM POPULATION ## #
    # Init
    trip_rates = pd.read_csv(trip_rates_path)

    # TODO: Make the production model more adaptable
    # Merge on all possible columns
    tr_cols = list(trip_rates)
    pop_cols = list(population)
    tr_merge_cols = [x for x in tr_cols if x in pop_cols]

    purpose_ph = dict()
    all_purposes = trip_rates['p'].drop_duplicates().reset_index(drop=True)
    desc = "Building trip rates by purpose"
    for p in tqdm(all_purposes, desc=desc):
        trip_rate_subset = trip_rates[trip_rates['p'] == p].copy()
        ph = population.copy()

        if p in consts.SOC_P:
            # Update ns with none
            ph['ns'] = 'none'
            ph['soc'] = ph['soc'].astype(int)
            # Insurance policy
            trip_rate_subset['ns'] = 'none'
            trip_rate_subset['soc'] = trip_rate_subset['soc'].astype(int)

        elif p in consts.NS_P:
            # Update soc with none
            ph['soc'] = 'none'
            ph['ns'] = ph['ns'].astype(int)
            # Insurance policy
            trip_rate_subset['soc'] = 'none'
            trip_rate_subset['ns'] = trip_rate_subset['ns'].astype(int)

        # Merge and calculate productions
        ph = ph[ph['people'] > 0].copy()
        ph = pd.merge(ph, trip_rate_subset, on=tr_merge_cols)

        ph['trips'] = ph['trip_rate'] * ph['people']
        ph = ph.drop(['trip_rate'], axis=1)

        # Group and sum
        ph = ph.reindex(index_cols, axis='columns')
        ph = ph.groupby(group_cols).sum().reset_index()

        # Update dictionary
        purpose_ph[p] = ph
    del trip_rates
    # Results in weekly trip rates by purpose and segmentation

    # ## SPLIT WEEKLY TRIP RATES BY TIME PERIOD ## #
    # Also converts to average weekday trips!
    # Init
    time_splits = pd.read_csv(time_splits_path)
    mean_time_splits = pd.read_csv(mean_time_splits_path)
    tp_merge_cols = ['area_type', 'traveller_type', 'p']

    # Convert tp nums to strings
    tp_needed = ['tp' + str(x) for x in tp_needed]

    tp_ph = dict()
    desc = 'Splitting trip rates by time period'
    for tp in tqdm(tp_needed, desc=desc):
        needed_cols = tp_merge_cols.copy() + [tp]
        tp_subset = time_splits.reindex(needed_cols, axis='columns').copy()
        tp_mean_subset = mean_time_splits.reindex(['p', tp], axis='columns').copy()

        for p, p_df in purpose_ph.items():
            # Get mean for infill
            tp_mean = tp_mean_subset[tp_mean_subset['p'] == p][tp]

            # Merge and infill
            tp_mat = p_df.copy()
            tp_mat = pd.merge(
                tp_mat,
                tp_subset,
                how='left',
                on=tp_merge_cols
            )
            tp_mat[tp] = tp_mat[tp].fillna(tp_mean)

            # Apply tp split and divide by 5 to get average weekday by tp
            tp_mat['trips'] = (tp_mat['trips'] * tp_mat[tp]) / 5

            # Group and sum
            tp_mat = tp_mat.reindex(index_cols, axis='columns')
            tp_mat = tp_mat.groupby(group_cols).sum().reset_index()

            # Add to compilation dict
            tp_ph[(p, tp)] = tp_mat
    del time_splits
    del mean_time_splits
    del purpose_ph
    # Results in average weekday trips by purpose, tp, and segmentation

    # Quick Audit
    approx_tp_totals = []
    for key, dat in tp_ph.items():
        total = dat['trips'].sum()
        approx_tp_totals.append(total)
    ave_wday = sum(approx_tp_totals)
    print('. Average weekday productions: %.2f' % ave_wday)

    # ## SPLIT AVERAGE WEEKDAY TRIP RATES BY MODE ## #
    # TODO: Apply at MSOA level rather than area type
    # Init
    mode_share = pd.read_csv(mode_share_path)
    m_merge_cols = ['area_type', 'ca', 'p']
    target_modes = ['m1', 'm2', 'm3', 'm5', 'm6']

    # Can get rid of traveller type now - too much detail
    # If we keep it we WILL have memory problems
    group_cols.remove(traveller_type_col)
    index_cols.remove(traveller_type_col)

    m_ph = dict()
    desc = 'Applying mode share splits'
    for m in tqdm(target_modes, desc=desc):
        needed_cols = m_merge_cols.copy() + [m]
        m_subset = mode_share.reindex(needed_cols, axis='columns').copy()

        for (p, tp), dat in tp_ph.items():
            m_mat = dat.copy()

            # Would merge all purposes, but left join should pick out target mode
            m_mat = pd.merge(
                m_mat,
                m_subset,
                how='left',
                on=m_merge_cols
            )

            # Apply m split
            m_mat['trips'] = (m_mat['trips'] * m_mat[m])

            # Reindex cols for efficiency
            m_mat = m_mat.reindex(index_cols, axis='columns')
            m_mat = m_mat.groupby(group_cols).sum().reset_index()

            m_mat = m_mat[m_mat['trips'] > 0]

            m_ph[(p, tp, m)] = m_mat
    del mode_share
    del tp_ph
    # Results in average weekday trips by purpose, tp, mode, and segmentation

    print("Writing topline audit...")
    approx_mode_totals = []
    for key, dat in m_ph.items():
        total = dat['trips'].sum()
        approx_mode_totals.append([key, total])

    # Build topline report
    topline = pd.DataFrame(approx_mode_totals, columns=['desc', 'total'])
    # Split key into components
    topline['p'], topline['tp'], topline['m'] = list(zip(*topline['desc']))
    topline = topline.reindex(['p', 'tp', 'm', 'total'], axis=1)
    topline = topline.groupby(['p', 'tp', 'm']).sum().reset_index()
    topline.to_csv(os.path.join(audit_out), index=False)

    # ## COMPILE ALL MATRICES INTO ONE ## #
    output_ph = list()
    desc = 'Compiling productions'
    for (p, tp, m), dat in tqdm(m_ph.items(), desc=desc):
        dat['p'] = p
        dat['tp'] = tp
        dat['m'] = m
        output_ph.append(dat)
    msoa_output = pd.concat(output_ph)

    # We now need to deal with tp and mode in one big matrix
    group_cols = group_cols + ['tp', 'm']
    index_cols = group_cols.copy()
    index_cols.append('trips')

    # Ensure matrix is in the correct format
    msoa_output = msoa_output.reindex(index_cols, axis='columns')
    msoa_output = msoa_output.groupby(group_cols).sum().reset_index()
    msoa_output['m'] = [int(m[1:]) for m in msoa_output['m']]
    msoa_output['tp'] = [int(tp[2:]) for tp in msoa_output['tp']]
    msoa_output['p'] = msoa_output['p'].astype(int)
    msoa_output['m'] = msoa_output['m'].astype(int)

    if do_ntem_control:
        # Get ntem totals
        # TODO: Depends on the time period - but this is fixed for now
        ntem_totals = pd.read_csv(control_path)
        ntem_lad_lookup = pd.read_csv(os.path.join(lad_lookup_dir,
                                                   lad_lookup_name))

        print("Performing NTEM constraint...")
        msoa_output, *_, = du.control_to_ntem(
            msoa_output,
            ntem_totals,
            ntem_lad_lookup,
            group_cols=['p', 'm'],
            base_value_name='trips',
            ntem_value_name='Productions',
            purpose='hb'
        )

    return msoa_output


def generate_productions(population: pd.DataFrame,
                         group_cols: List[str],
                         base_year: str,
                         future_years: List[str],
                         trip_rates_path: str,
                         time_splits_path: str,
                         mean_time_splits_path: str,
                         mode_share_path: str,
                         audit_dir: str,
                         ntem_control_dir: str = None,
                         lad_lookup_dir: str = None,
                         control_fy_productions: bool = True
                         ) -> pd.DataFrame:
    # TODO: write generate_productions() docs
    # Init
    all_years = [base_year] + future_years
    audit_base_fname = 'yr%s_production_topline.csv'
    ntem_base_fname = 'ntem_pa_ave_wday_%s.csv'

    # TODO: Multiprocess yearly productions
    # Generate Productions for each year
    yr_ph = dict()
    for year in all_years:
        # Only only set the control path if we need to constrain
        if not control_fy_productions and year != base_year:
            ntem_control_path = None
        elif ntem_control_dir is not None:
            ntem_fname = ntem_base_fname % year
            ntem_control_path = os.path.join(ntem_control_dir, ntem_fname)
        else:
            ntem_control_path = None

        audit_out = os.path.join(audit_dir, audit_base_fname % year)

        yr_pop = population.copy().reindex(group_cols + [year], axis='columns')
        yr_pop = yr_pop.rename(columns={year: 'people'})
        yr_prod = merge_pop_trip_rates(
            yr_pop,
            group_cols=group_cols,
            trip_rates_path=trip_rates_path,
            time_splits_path=time_splits_path,
            mean_time_splits_path=mean_time_splits_path,
            mode_share_path=mode_share_path,
            audit_out=audit_out,
            control_path=ntem_control_path,
            lad_lookup_dir=lad_lookup_dir
        )
        yr_ph[year] = yr_prod

    # Join all productions into one big matrix
    productions = du.combine_yearly_dfs(
        yr_ph,
        unique_col='trips'
    )

    return productions


def _nhb_production_internal(hb_pa_import,
                             nhb_trip_rates,
                             year,
                             purpose,
                             mode,
                             segment,
                             car_availability):
    """
      The internals of nhb_production(). Useful for making the code more
      readable due to the number of nested loops needed
    """
    hb_dist = du.get_dist_name(
        'hb',
        'pa',
        str(year),
        str(purpose),
        str(mode),
        str(segment),
        str(car_availability),
        csv=True
    )

    # Seed the nhb productions with hb values
    hb_pa = pd.read_csv(
        os.path.join(hb_pa_import, hb_dist)
    )
    hb_pa = du.expand_distribution(
        hb_pa,
        year,
        purpose,
        mode,
        segment,
        car_availability,
        id_vars='p_zone',
        var_name='a_zone',
        value_name='trips'
    )

    # Aggregate to destinations
    nhb_prods = hb_pa.groupby([
        "a_zone",
        "purpose_id",
        "mode_id"
    ])["trips"].sum().reset_index()

    # join nhb trip rates
    nhb_prods = pd.merge(nhb_trip_rates,
                         nhb_prods,
                         on=["purpose_id", "mode_id"])

    # Calculate NHB productions
    nhb_prods["nhb_dt"] = nhb_prods["trips"] * nhb_prods["nhb_trip_rate"]

    # aggregate nhb_p 11_12
    nhb_prods.loc[nhb_prods["nhb_p"] == 11, "nhb_p"] = 12

    # Remove hb purpose and mode by aggregation
    nhb_prods = nhb_prods.groupby([
        "a_zone",
        "nhb_p",
        "nhb_m",
    ])["nhb_dt"].sum().reset_index()

    return nhb_prods


def old_nhb_production(hb_pa_import,
                   nhb_export,
                   required_purposes,
                   required_modes,
                   required_soc,
                   required_ns,
                   required_car_availabilities,
                   years_needed,
                   nhb_factor_import,
                   out_fname=consts.NHB_PRODUCTIONS_FNAME
                   ):
    """
    This function builds NHB productions by
    aggregates HB distribution from EFS output to destination

    TODO: Update to use the TMS method - see backlog

    Parameters
    ----------
    required lists:
        to loop over TfN segments

    Returns
    ----------
    nhb_production_dictionary:
        Dictionary containing NHB productions by year
    """
    # Init
    nhb_production_dictionary = dict()

    # Get nhb trip rates
    # Might do the other way - This emits CA segmentation
    nhb_trip_rates = pd.read_csv(
        os.path.join(nhb_factor_import, "IgammaNMHM.csv")
    ).rename(
        columns={"p": "purpose_id", "m": "mode_id"}
    )

    # For every: Year, purpose, mode, segment, ca
    for year in years_needed:
        loop_gen = list(du.segmentation_loop_generator(
            required_purposes,
            required_modes,
            required_soc,
            required_ns,
            required_car_availabilities
        ))
        yearly_nhb_productions = list()
        desc = 'Generating NHB Productions for yr%s' % year
        for purpose, mode, segment, car_availability in tqdm(loop_gen, desc=desc):
            nhb_productions = _nhb_production_internal(
                hb_pa_import,
                nhb_trip_rates,
                year,
                purpose,
                mode,
                segment,
                car_availability
            )
            yearly_nhb_productions.append(nhb_productions)

        # ## Output the yearly productions ## #
        # Aggregate all productions for this year
        yr_nhb_productions = pd.concat(yearly_nhb_productions)
        yr_nhb_productions = yr_nhb_productions.groupby(
            ["a_zone", "nhb_p", "nhb_m"]
        )["nhb_dt"].sum().reset_index()

        # Rename columns from NHB perspective
        yr_nhb_productions = yr_nhb_productions.rename(
            columns={
                'a_zone': 'p_zone',
                'nhb_p': 'p',
                'nhb_m': 'm',
                'nhb_dt': 'trips'
            }
        )

        # Output
        nhb_productions_fname = '_'.join(["yr" + str(year), out_fname])
        yr_nhb_productions.to_csv(
            os.path.join(nhb_export, nhb_productions_fname),
            index=False
        )

        # save to dictionary by year
        nhb_production_dictionary[year] = yr_nhb_productions

    return nhb_production_dictionary
