# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 14:32:53 2020

@author: Sneezy
"""

import os
import numpy as np
import pandas as pd

from typing import List

import efs_constants as consts
from efs_constrainer import ForecastConstrainer

from demand_utilities import error_management as err_check


class EFSAttractionGenerator:
    # infill statics
    POPULATION_INFILL = 0.001
    
    def __init__(self,
                 tag_certainty_bounds=consts.TAG_CERTAINTY_BOUNDS):
        """
        #TODO
        """
        self.efs_constrainer = ForecastConstrainer()
        self.tag_certainty_bounds = tag_certainty_bounds
    
    def run(self,
            worker_growth: pd.DataFrame,
            worker_values: pd.DataFrame,
            worker_constraint: pd.DataFrame,
            worker_split: pd.DataFrame,
            development_log: pd.DataFrame = None,
            development_log_split: pd.DataFrame = None,
            integrating_development_log: bool = False,
            minimum_development_certainty: str = "MTL",  # "NC", "MTL", "RF", "H"
            constraint_required: List[bool] = consts.DEFAULT_ATTRACTION_CONSTRAINTS,
            constraint_method: str = "Percentage",  # Percentage, Average
            constraint_area: str = "Designated",  # Zone, Designated, All
            constraint_on: str = "Growth",  # Growth, All
            constraint_source: str = "Grown Base",  # Default, Grown Base, Model Grown Base
            designated_area: pd.DataFrame = None,
            base_year_string: str = None,
            model_years: List[str] = List[None],
            attraction_weights=None,
            output_path=None):
        """
        #TODO
        """
        # TODO: Attractions don't use the D-Log
        if integrating_development_log:
            if development_log is not None:
                development_log = development_log.copy()
            else:
                print("No development_log dataframe passed to worker "
                      + "generator but development_log is indicated to be "
                      + "required. Process will not function correctly.")
                sys.exit(1)
                 
        # ## GROW WORKERS
        grown_workers = self.worker_grower(
                worker_growth,
                worker_values,
                base_year_string,
                model_years
                ) 
    
        # ## ## initial worker metric constraint
        if constraint_required[4] and (constraint_source != "model grown base"):
            print("Performing the first constraint on workers...")
            grown_workers = self.efs_constrainer.run(
                    grown_workers,
                    constraint_method,
                    constraint_area,
                    constraint_on,
                    worker_constraint,
                    base_year_string,
                    model_years,
                    designated_area
                    )

        elif constraint_source == "model grown base":
            print("Generating model grown base constraint for use on "
                  "development constraints...")
            worker_constraint = grown_workers.copy()
        
        # ## D-LOG INTEGRATION
        if integrating_development_log:
            print("Including development log...")
            # TODO: Generate workers
        else:
            print("Not including development log...")
        
        # ## SPLIT WORKERS
        split_workers = self.split_workers(
                grown_workers,
                worker_split,
                base_year_string,
                model_years
                )
        
        # ## post-development log constraint
        # (secondary worker metric constraint)
        if constraint_required[5]:
            print("Performing the post-development log constraint on workers...")
            split_workers = self.efs_constrainer.run(
                    split_workers,
                    constraint_method,
                    constraint_area,
                    constraint_on,
                    worker_constraint,
                    base_year_string,
                    model_years,
                    designated_area
                    )
        
        print("Adding all worker category...")
        final_workers = self.add_all_worker_category(
                split_workers,
                "E01",
                model_years
                )
        print("Added all worker category!")
            
        # ## RECOMBINING WORKERS
        print("Recombining workers...")
        final_workers = self.growth_recombination(
                 final_workers,
                 "base_year_workers",
                 model_years
                 )
        
        final_workers.sort_values(
            by=["model_zone_id", "employment_class"],
            inplace=True
        )
        print("Recombined workers!")

        if output_path is not None:
            final_workers.to_csv(
                os.path.join(output_path, "MSOA_workers.csv"),
                index=False)

        if attraction_weights is None:
            return final_workers

        print("Workers generated. Converting to attractions...")
        attractions = self.attraction_generation(
            final_workers,
            attraction_weights,
            model_years
        )

        return attractions

    def attraction_generation(self,
                              worker_dataframe: pd.DataFrame,
                              attraction_weight: pd.DataFrame,
                              year_list: List[str]
                              ) -> pd.DataFrame:
        """
        #TODO
        """
        worker_dataframe = worker_dataframe.copy()
        attraction_weight = attraction_weight.copy()

        attraction_dataframe = pd.merge(
            worker_dataframe,
            attraction_weight,
            on=["employment_class"],
            suffixes=("", "_weights")
        )

        for year in year_list:
            attraction_dataframe.loc[:, year] = (
                attraction_dataframe[year]
                *
                attraction_dataframe[year + "_weights"]
            )

        group_by_cols = ["model_zone_id", "purpose_id"]
        needed_columns = group_by_cols.copy()
        needed_columns.extend(year_list)

        attraction_dataframe = attraction_dataframe[needed_columns]
        attraction_dataframe = attraction_dataframe.groupby(
            by=group_by_cols,
            as_index=False
        ).sum()

        return attraction_dataframe



    def worker_grower(self,
                      worker_growth: pd.DataFrame,
                      worker_values: pd.DataFrame,
                      base_year: str,
                      year_string_list: List[str]
                      ) -> pd.DataFrame:
        # get workers growth from base year
        print("Adjusting workers growth to base year...")
        worker_growth = self.convert_growth_off_base_year(
                worker_growth,
                base_year,
                year_string_list
                )
        print("Adjusted workers growth to base year!")
        
        
        print("Growing workers from base year...")
        grown_workers = self.get_grown_values(
                worker_values,
                worker_growth,
                "base_year_workers",
                year_string_list
                )
        print("Grown workers from base year!")
        
        return grown_workers
  
    def convert_growth_off_base_year(self,
                                     growth_dataframe: pd.DataFrame,
                                     base_year: str,
                                     all_years: List[str]
                                     ) -> pd.DataFrame:
        """
        #TODO
        """
        growth_dataframe = growth_dataframe.copy()
        growth_dataframe.loc[
        :,
        all_years
        ] = growth_dataframe.apply(
            lambda x,
            columns_required = all_years,
            base_year = base_year:
                x[columns_required] / x[base_year],
                axis = 1)
        
        return growth_dataframe
    
    def get_grown_values(self,
                       base_year_dataframe: pd.DataFrame,
                       growth_dataframe: pd.DataFrame,
                       base_year_string: str,
                       all_years: List[str]
                       ) -> pd.DataFrame:
        """
        #TODO
        """
        base_year_dataframe = base_year_dataframe.copy()
        growth_dataframe = growth_dataframe.copy()
        
        # CREATE GROWN DATAFRAME
        grown_dataframe = base_year_dataframe.merge(
                growth_dataframe,
                on = "model_zone_id"
                )
        
        grown_dataframe.loc[
        :,
        all_years
        ] = grown_dataframe.apply(
            lambda x,
            columns_required = all_years,
            base_year = base_year_string:
                (x[columns_required] - 1) * x[base_year],
                axis = 1)
        
        return grown_dataframe
    
    def growth_recombination(self,
                             metric_dataframe: pd.DataFrame,
                             metric_column_name: str,
                             all_years: List[str]
                             ) -> pd.DataFrame:
        """
        #TODO
        """
        metric_dataframe = metric_dataframe.copy()
        
        ## combine together dataframe columns to give full future values
        ## f.e. base year will get 0 + base_year_population        
        for year in all_years:
            metric_dataframe[year] = (
                    metric_dataframe[year]
                    +
                    metric_dataframe[metric_column_name]
                    )
        
        ## drop the unnecessary metric column
        metric_dataframe = metric_dataframe.drop(
                labels = metric_column_name,
                axis = 1
                )
        
        return metric_dataframe
    
    def split_workers(self,
                      workers_dataframe: pd.DataFrame,
                      workers_split_dataframe: pd.DataFrame,
                      base_year_string: str,
                      all_years: List[str]
                      ) -> pd.DataFrame:
        workers_dataframe = workers_dataframe.copy()
        workers_split_dataframe = workers_split_dataframe.copy()
        
        split_workers_dataframe = pd.merge(
                workers_dataframe,
                workers_split_dataframe,
                on = ["model_zone_id"],
                suffixes = {"", "_split"}
                )
                
        split_workers_dataframe["base_year_workers"] = (
                split_workers_dataframe["base_year_workers"]
                *
                split_workers_dataframe[base_year_string + "_split"]
                )
        
        for year in all_years:
            # we iterate over each zone
            # create zone mask
            split_workers_dataframe[year] = (
                    split_workers_dataframe[year]
                    *
                    split_workers_dataframe[year + "_split"]
                    )
        
        required_columns = [
                "model_zone_id",
                "employment_class",
                "base_year_workers"
                ]
        
        required_columns.extend(all_years)
        split_workers_dataframe = split_workers_dataframe[
                required_columns
                ]
        return split_workers_dataframe
    
    def add_all_worker_category(self,
                                workers_dataframe: pd.DataFrame,
                                all_worker_category_id: str,
                                all_years: List[str]
                                ) -> pd.DataFrame:
        workers_dataframe = workers_dataframe.copy()
        zones = workers_dataframe["model_zone_id"].unique()
        
        for zone in zones:
            total_line = workers_dataframe[
                    workers_dataframe["model_zone_id"] == zone
                    ].sum()
            
            year_data = {
                    "base_year_workers": [total_line["base_year_workers"]]
                    }            
            for year in all_years:
                year_data[year] = [total_line[year]]
                
            data = {
                    "model_zone_id": [zone],
                    "employment_class": [all_worker_category_id]
                    }
            
            data.update(year_data)
            
            total_line = pd.DataFrame(data)
            
            workers_dataframe = workers_dataframe.append(total_line)
            
        workers_dataframe = workers_dataframe.sort_values(
                by = [
                        "model_zone_id",
                        "employment_class"
                        ]
                )
        
        return workers_dataframe