# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 14:32:53 2020

@author: Sneezy
"""

import os
import sys
import numpy as np
import pandas as pd

from typing import List

import efs_constants as consts
from efs_constrainer import ForecastConstrainer

import demand_utilities.utils as du


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
            base_year: str,
            future_years: List[str],

            # Employment Growth
            employment_growth: pd.DataFrame,
            employment_constraint: pd.DataFrame,
            worker_values: pd.DataFrame,  # Probs remove
            worker_split: pd.DataFrame,  # Probs remove

            # Build import paths
            import_home: str,
            msoa_conversion_path: str,

            # Alternate population/attraction creation files
            attraction_weights=None,

            # Production control file
            ntem_control_dir: str = None,
            lad_lookup_dir: str = None,
            control_attractions: bool = True,

            # D-Log
            d_log: pd.DataFrame = None,
            d_log_split: pd.DataFrame = None,

            # Employment constraints
            constraint_required: List[bool] = consts.DEFAULT_ATTRACTION_CONSTRAINTS,
            constraint_method: str = "Percentage",  # Percentage, Average
            constraint_area: str = "Designated",  # Zone, Designated, All
            constraint_on: str = "Growth",  # Growth, All
            constraint_source: str = "Grown Base",  # Default, Grown Base, Model Grown Base
            designated_area: pd.DataFrame = None,

            # Segmentation controls
            m_needed: List[int] = consts.MODES_NEEDED,
            external_zone_col: str = 'model_zone_id',

            # Handle outputs
            audits: bool = True,
            out_path: str = None,
            recreate_attractions: bool = True,
            ) -> pd.DataFrame:
        """
        TODO: Write attraction model documentation
        """
        # Init
        internal_zone_col = 'msoa_zone_id'
        all_years = [str(x) for x in [base_year] + future_years]
        integrate_d_log = d_log is not None and d_log_split is not None
        if integrate_d_log:
            d_log = d_log.copy()
            d_log_split = d_log_split.copy()

        # Fix column naming if different
        if external_zone_col != internal_zone_col:
            employment_growth = employment_growth.copy().rename(
                columns={external_zone_col: internal_zone_col}
            )
            designated_area = designated_area.copy().rename(
                columns={external_zone_col: internal_zone_col}
            )
            employment_constraint = employment_constraint.rename(
                columns={external_zone_col: internal_zone_col}
            )

        # TODO: REMOVE THIS ONCE ATTRACTION DEV OVER
        # Replace with path builder
        soc_to_sic_path = r"Y:\NorMITs Synthesiser\import\attraction_data\soc_2_digit_sic_2018.csv"
        employment_path = r"Y:\NorMITs Synthesiser\import\attraction_data\non_freight_msoa_2018.csv"

        # ## BASE YEAR EMPLOYMENT ## #
        print("Loading the base year employment data...")
        base_year_emp = get_employment_data(import_path=employment_path)

        # Audit employment numbers
        du.print_w_toggle("Base Year Employment: %d" % base_year_emp.values.sum(),
                          echo=audits)

        # ## FUTURE YEAR EMPLOYMENT ## #
        # Write this!!!!!!!
        print("Generating future year employment data...")
        # employment = self.grow_employment(
        #     base_year_emp,
        #     employment_growth,
        #     base_year,
        #     future_years
        # )

        print(base_year_emp)
        print(employment_growth)

        # DO we need these? TfN segmentation!
        # soc_weights = get_soc_weights(soc_to_sic_path=soc_to_sic_path)


        exit()

        # OLD ATTRACTION MODEL BELOW HERE

        # TODO: Attractions don't use the D-Log
        if integrate_d_log:
            if d_log is not None:
                d_log = d_log.copy()
            else:
                print("No development_log dataframe passed to worker "
                      + "generator but development_log is indicated to be "
                      + "required. Process will not function correctly.")
                sys.exit(1)
                 
        # ## GROW WORKERS
        grown_workers = self.worker_grower(
                employment_growth,
                worker_values,
                base_year,
                all_years
                ) 
    
        # ## ## initial worker metric constraint
        if constraint_required[4] and (constraint_source != "model grown base"):
            print("Performing the first constraint on workers...")
            grown_workers = self.efs_constrainer.run(
                    grown_workers,
                    constraint_method,
                    constraint_area,
                    constraint_on,
                    employment_constraint,
                    base_year,
                    all_years,
                    designated_area
                    )

        elif constraint_source == "model grown base":
            print("Generating model grown base constraint for use on "
                  "development constraints...")
            employment_constraint = grown_workers.copy()
        
        # ## D-LOG INTEGRATION
        if integrate_d_log:
            print("Including development log...")
            # TODO: Generate workers
        else:
            print("Not including development log...")
        
        # ## SPLIT WORKERS
        split_workers = self.split_workers(
                grown_workers,
                worker_split,
                base_year,
                all_years
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
                    employment_constraint,
                    base_year,
                    all_years,
                    designated_area
                    )
        
        print("Adding all worker category...")
        final_workers = self.add_all_worker_category(
                split_workers,
                "E01",
                all_years
                )
        print("Added all worker category!")
            
        # ## RECOMBINING WORKERS
        print("Recombining workers...")
        final_workers = du.growth_recombination(
                 final_workers,
                 "base_year_workers",
                 all_years
                 )
        
        final_workers.sort_values(
            by=["model_zone_id", "employment_class"],
            inplace=True
        )
        print("Recombined workers!")

        if out_path is not None:
            final_workers.to_csv(
                os.path.join(out_path, "MSOA_workers.csv"),
                index=False)

        if attraction_weights is None:
            return final_workers

        print("Workers generated. Converting to attractions...")
        attractions = self.attraction_generation(
            final_workers,
            attraction_weights,
            all_years
        )

        # Write attractions out
        out_fname = 'MSOA_attractions.csv'
        attractions.to_csv(
            os.path.join(out_path, out_fname),
            index=False
        )

        print(attractions)
        print(list(attractions))
        print(attractions.dtypes)

        exit()

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
        worker_growth = du.convert_growth_off_base_year(
                worker_growth,
                base_year,
                year_string_list
                )
        print("Adjusted workers growth to base year!")
        
        
        print("Growing workers from base year...")
        grown_workers = du.get_grown_values(
                worker_values,
                worker_growth,
                "base_year_workers",
                year_string_list
                )
        print("Grown workers from base year!")
        
        return grown_workers
    
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


def get_employment_data(import_path: str
                        ) -> pd.DataFrame:
    """
    Reads in employment data from file and returns dataframe.

    Can be updated in future to accept different types of inputs,
    and return all in the same format.

    Parameters
    ----------
    import_path:
        The path to the employment data to import

    Returns
    -------
    employment_data:
        Dataframe with msoa_zone_ids as the index, and employment categories
        as the columns

    """
    emp_data = pd.read_csv(import_path)
    emp_data = emp_data.rename(columns={'geo_code': 'msoa_zone_id'})
    emp_data = emp_data.set_index('msoa_zone_id')

    return emp_data


def get_soc_weights(soc_to_sic_path: str,
                    zone_col: str = 'msoa_zone_id',
                    soc_col: str = 'soc_class',
                    jobs_col: str = 'seg_jobs'
                    ) -> pd.DataFrame:
    """
    TODO: Write get_soc_weights() docs

    Parameters
    ----------
    soc_to_sic_path
    zone_col
    soc_col
    jobs_col

    Returns
    -------

    """
    # Init
    soc_weighted_jobs = pd.read_csv(soc_to_sic_path)

    # Convert soc numbers to names (to differentiate from ns)
    soc_weighted_jobs[soc_col] = soc_weighted_jobs[soc_col].astype(int).astype(str)
    soc_weighted_jobs[soc_col] = 'soc' + soc_weighted_jobs[soc_col]

    # Calculate Zonal weights for socs
    # This give us the benefit of model purposes in HSL data
    group_cols = [zone_col, soc_col]
    index_cols = group_cols.copy()
    index_cols.append(jobs_col)

    soc_weights = soc_weighted_jobs.reindex(index_cols, axis='columns')
    soc_weights = soc_weights.groupby(group_cols).sum().reset_index()
    soc_weights = soc_weights.pivot(
        index=zone_col,
        columns=soc_col,
        values=jobs_col
    )

    # Convert to factors
    soc_segments = soc_weighted_jobs[soc_col].unique()
    soc_weights['total'] = soc_weights[soc_segments].sum(axis='columns')

    for soc in soc_segments:
        soc_weights[soc] /= soc_weights['total']

    soc_weights = soc_weights.drop('total', axis='columns')

    return soc_weights
