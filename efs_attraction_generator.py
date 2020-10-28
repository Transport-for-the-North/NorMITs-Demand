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
from typing import Dict

from tqdm import tqdm

import efs_constants as consts
from efs_constrainer import ForecastConstrainer

import demand_utilities.utils as du


class EFSAttractionGenerator:
    
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
            segmentation_cols: List[str] = None,
            external_zone_col: str = 'model_zone_id',
            emp_cat_col: str = 'employment_cat',
            no_neg_growth: bool = True,
            employment_infill: float = 0.001,

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

        # TODO: Make this more adaptive
        # Set the level of segmentation being used
        if segmentation_cols is None:
            segmentation_cols = [emp_cat_col]

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
        mode_splits_path = r"Y:\NorMITs Synthesiser\import\attraction_data\attraction_mode_split.csv"

        ntem_control_dir = r'Y:\NorMITs Demand\import\ntem_constraints'
        lad_lookup_dir = r'Y:\NorMITs Demand\import'

        # Watch out - this changes depending on the model!!!
        attraction_weights_path = r"Y:\NorMITs Synthesiser\Noham\Model Zone Lookups\attraction_weights.csv"

        # ## BASE YEAR EMPLOYMENT ## #
        print("Loading the base year employment data...")
        base_year_emp = get_employment_data(
            import_path=employment_path,
            zone_col=internal_zone_col,
            emp_cat_col=emp_cat_col,
            msoa_path=msoa_conversion_path,
            return_format='long',
            value_col=base_year,
        )

        # DO we need these? TfN segmentation!
        # soc_weights = get_soc_weights(soc_to_sic_path=soc_to_sic_path)

        # Audit employment numbers
        mask = (base_year_emp[emp_cat_col] == 'E01')
        total_base_year_emp = base_year_emp.loc[mask, base_year].sum()
        du.print_w_toggle("Base Year Employment: %d" % total_base_year_emp,
                          echo=audits)

        # ## FUTURE YEAR EMPLOYMENT ## #
        print("Generating future year employment data...")
        employment = du.grow_to_future_years(
            base_year_df=base_year_emp,
            growth_df=employment_growth,
            base_year=base_year,
            future_years=future_years,
            growth_merge_col=internal_zone_col,
            no_neg_growth=no_neg_growth,
            infill=employment_infill
        )

        # ## CONSTRAIN POPULATION ## #
        if constraint_required[3] and (constraint_source != "model grown base"):
            print("Performing the first constraint on employment...")
            employment = self.efs_constrainer.run(
                employment,
                constraint_method,
                constraint_area,
                constraint_on,
                employment_constraint,
                base_year,
                all_years,
                designated_area,
                internal_zone_col
            )
        elif constraint_source == "model grown base":
            print("Generating model grown base constraint for use on "
                  "development constraints...")
            employment_constraint = employment.copy()

        # ## INTEGRATE D-LOG ## #
        if integrate_d_log:
            print("Integrating the development log...")
            raise NotImplementedError("D-Log population integration has not "
                                      "yet been implemented.")
        else:
            # If not integrating, no need for another constraint
            constraint_required[4] = False

        # ## POST D-LOG CONSTRAINT ## #
        if constraint_required[4]:
            print("Performing the post-development log constraint on employment...")
            employment = self.efs_constrainer.run(
                employment,
                constraint_method,
                constraint_area,
                constraint_on,
                employment_constraint,
                base_year,
                all_years,
                designated_area,
                internal_zone_col
            )

        # Reindex and sum
        group_cols = [internal_zone_col] + segmentation_cols
        index_cols = group_cols.copy() + all_years
        employment = employment.reindex(index_cols, axis='columns')
        employment = employment.groupby(group_cols).sum().reset_index()

        # Population Audit
        if audits:
            print('\n', '-'*15, 'Employment Audit', '-'*15)
            mask = (employment[emp_cat_col] == 'E01')
            for year in all_years:
                total_emp = employment.loc[mask, year].sum()
                print('. Total population for year %s is: %.4f'
                      % (str(year), total_emp))
            print('\n')

        # Convert back to MSOA codes for output and attractions
        employment = du.convert_msoa_naming(
            employment,
            msoa_col_name=internal_zone_col,
            msoa_path=msoa_conversion_path,
            to='string'
        )

        # Write the produced employment to file
        if out_path is None:
            print("WARNING! No output path given. "
                  "Not writing employment to file.")
        else:
            print("Writing employment to file...")
            fname = "MSOA_employment.csv"
            employment.to_csv(os.path.join(out_path, fname), index=False)

        # ## CREATE ATTRACTIONS ## #
        # Index by as much segmentation as possible
        idx_cols = list(employment.columns)
        for unq_col in all_years:
            idx_cols.remove(unq_col)

        attractions = generate_attractions(
            employment=employment,
            all_years=all_years,
            attraction_weights_path=attraction_weights_path,
            mode_splits_path=mode_splits_path,
            idx_cols=idx_cols,
            emp_cat_col=emp_cat_col,
            ntem_control_dir=ntem_control_dir,
            lad_lookup_dir=lad_lookup_dir
        )

        # Write attractions to file
        if out_path is None:
            print("WARNING! No output path given. "
                  "Not writing attractions to file.")
        else:
            print("Writing productions to file...")
            fname = 'MSOA_attractions.csv'
            attractions.to_csv(os.path.join(out_path, fname), index=False)

        exit()










        # OLD ATTRACTION MODEL BELOW HERE

        
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


def get_employment_data(import_path: str,
                        msoa_path: str = None,
                        zone_col: str = 'msoa_zone_id',
                        emp_cat_col: str = 'employment_cat',
                        value_col: str = 'jobs',
                        return_format: str = 'wide',
                        add_all_commute: bool = True,
                        all_commute_col: str = 'E01'
                        ) -> pd.DataFrame:
    """
    Reads in employment data from file and returns dataframe.

    Can be updated in future to accept different types of inputs,
    and return all in the same format.

    Parameters
    ----------
    import_path:
        The path to the employment data to import

    msoa_path:
        Path to the msoa file for converting from msoa string ids to
        integer ids. If left as None, then MSOA codes are returned instead

    zone_col:
        The name of the column in the employment data that refers to the zones.

    emp_cat_col:
        The name to give to the employment category column when converting to
        long format.

    value_col:
        The name to give to the values in the wide matrix when converting to
        long format.

    return_format:
        What format to return the employment data in. Can take either 'wide' or
        'long'

    add_all_commute:
        Whether to add an additional employment category that covers commuting.
        The new category will be a sum of all other employment categories.
        If added, the column will be named all_commute_col

    all_commute_col:
        If add_all_commute is True, the added column will be given this name

    Returns
    -------
    employment_data:
        Dataframe with zone_col as the index, and employment categories
        as the columns

    """
    # Error check
    valid_return_formats = ['wide', 'long']
    return_format = return_format.strip().lower()
    if return_format not in valid_return_formats:
        raise ValueError(
            "'%s' is not a valid return format. Expected one of: %s"
            % (str(return_format), str(valid_return_formats))
        )

    # Read in raw data
    emp_data = pd.read_csv(import_path)
    emp_data = emp_data.rename(columns={'geo_code': zone_col})

    # Add in commute category if required
    if add_all_commute:
        # Commute is a sum of all other employment categories
        emp_cats = list(emp_data.columns)
        emp_cats.remove(zone_col)

        emp_data[all_commute_col] = emp_data[emp_cats].sum(axis='columns')

    # Convert to long format if needed
    if return_format == 'long':
        emp_data = emp_data.melt(
            id_vars=zone_col,
            var_name=emp_cat_col,
            value_name=value_col
        )

    # Convert to msoa zone numbers if needed
    if msoa_path is not None:
        emp_data = du.convert_msoa_naming(
            emp_data,
            msoa_col_name=zone_col,
            msoa_path=msoa_path,
            to='int'
        )


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


def combine_yearly_attractions(year_dfs: Dict[str, pd.DataFrame],
                               zone_col: str = 'msoa_zone_id',
                               p_col: str = 'purpose',
                               purposes: List[int] = None
                               ) -> pd.DataFrame:
    """
    Efficiently concatenates the yearly dataframes in year_dfs.

    Parameters
    ----------
    year_dfs:
        Dictionary, with keys as the years, and the attractions data for that
        year as a value. Expects the attractions data to be in wide format with
        an index of zone_id (plus any segmentation) and  a column for each
        purpose.

    zone_col:
        The name of the column containing the zone ids

    p_col:
        The name to give to the created purpose column during melting.

    purposes:
        A list of the purposes to keep. If left as None, all purposes are
        kept

    Returns
    -------
    combined_attractions:
        A dataframe of all combined data in year_dfs. There will be a separate
        column for each year of data.
    """
    # Init
    keys = list(year_dfs.keys())

    if purposes is None:
        purposes = list(year_dfs[keys[0]].columns)

    # ## CONVERT THE MATRICES TO LONG FORMAT ## #
    for year, mat in year_dfs.items():
        year_dfs[year] = mat.reset_index().melt(
            id_vars=zone_col,
            var_name=p_col,
            value_name=year
        )

    # The merge cols will be everything but the years
    year = list(year_dfs.keys())[0]
    merge_cols = list(year_dfs[year].columns)
    merge_cols.remove(year)

    # ## SPLIT MATRICES AND JOIN BY PURPOSE ## #
    attraction_ph = list()
    desc = "Merging attractions by purpose"
    for p in tqdm(purposes, desc=desc):

        # Get all the matrices that belong to this purpose
        yr_p_dfs = list()
        for year, df in year_dfs.items():
            temp_df = df[df[p_col] == p].copy()
            yr_p_dfs.append(temp_df)

        # Iteratively merge all matrices into one
        merged_df = yr_p_dfs[0]
        for df in yr_p_dfs[1:]:
            merged_df = pd.merge(
                merged_df,
                df,
                on=merge_cols
            )
        attraction_ph.append(merged_df)
        del yr_p_dfs

    # ## CONCATENATE ALL MERGED MATRICES ## #
    return pd.concat(attraction_ph)


def merge_attraction_weights(employment: pd.DataFrame,
                             attraction_weights_path: str,
                             mode_splits_path: str,
                             idx_cols: List[str] = None,
                             p_col: str = 'purpose',
                             m_col: str = 'mode',
                             m_split_col: str = 'mode_share',
                             unique_col: str = 'trips',
                             control_path: str = None,
                             lad_lookup_dir: str = None,
                             lad_lookup_name: str = consts.DEFAULT_LAD_LOOKUP,
                             ) -> pd.DataFrame:
    """
    Combines employment numbers with attractions weights to produce the
    attractions per purpose.

    Carries out the following actions:
        - Applies attraction weights to employment to produce attractions
        - Splits the attractions by mode
        - Optionally constrains to the values in control_path

    Parameters
    ----------
    employment:
        Wide dataframe containing the employment data. The index should be the
        model_zone (plus any further segmentation), and the columns should be
        the employment categories.

    attraction_weights_path:
        Path the the attraction weights file. This file should contain a wide
        matrix, with the purposes as the index, and the employment categories
        as the columns.

    mode_splits_path:
        Path to the file of mode splits by 'p'

    idx_cols:
        The column names used to index the wide employment df. This should
        cover all segmentation in the employment

    p_col:
        The name of the column containing the purpose values.

    m_col:
        The name of the column in mode_splits_path containing the mode values.

    m_split_col
        The name of the column in mode_splits_path containing the mode share
        values.

    unique_col:
        The name to give to the unique column for each year

    control_path:
        Path to the file containing the data to control the produced
        attractions to. If left as None, no control will be carried out.


    lad_lookup_dir:
        Path to the file containing the conversion from msoa zoning to LAD
        zoning, to be used for controlling the attractions. If left as None, no
        control will be carried out.

    lad_lookup_name:
        The name of the file in lad_lookup_dir that contains the msoa zoning
        to LAD zoning conversion.

    Returns
    -------
    Attractions:
        A wide dataframe containing the attraction numbers. The index will
        match the index from employment, the columns will be the purposes
        given in p_col of attractions_weight_path.

    """
    # Init
    do_ntem_control = control_path is not None and lad_lookup_dir is not None
    idx_cols = ['msoa_zone_id'] if idx_cols is None else idx_cols
    emp_cats = list(employment.columns.unique())

    # Read in the attraction weights
    attraction_weights = pd.read_csv(attraction_weights_path)
    purposes = list(attraction_weights[p_col])

    # Apply purpose weights to the employment
    a_ph = list()
    for p in purposes:
        # Init loop
        p_attractions = employment.copy()

        # Get the weights and broadcast across all model zones
        p_weights = attraction_weights[attraction_weights[p_col] == p]
        p_weights = np.broadcast_to(
            p_weights[emp_cats].values,
            (len(p_attractions.index), len(emp_cats))
        )

        # Calculate the total attractions per zone for this purpose
        p_attractions[emp_cats] *= p_weights
        p_attractions[p] = p_attractions[emp_cats].sum(axis='columns')
        p_attractions = p_attractions.drop(emp_cats, axis='columns')

        a_ph.append(p_attractions)

    # Stick the attractions by purpose together, and return
    attractions = pd.concat(a_ph, axis='columns')

    # Convert the matrix back to long format
    attractions = attractions.reset_index().melt(
        id_vars=idx_cols,
        var_name=p_col,
        value_name=unique_col
    )

    # TODO: Do we need to add in further segmentation here?
    # I think it should happen earlier so we can grow employment by segments
    # Will ask Chris when he is back

    # ## SPLIT THE ATTRACTIONS BY MODE ## #
    # Need to convert the str purposes into int
    attractions[p_col] = attractions[p_col].apply(lambda row: consts.P_STR2INT[row])

    # Read in and apply mode splits
    mode_splits = pd.read_csv(mode_splits_path)
    mode_splits = mode_splits.rename(columns={'p': p_col, 'm': m_col})

    attractions = attractions.merge(
        mode_splits,
        how='left',
        on=idx_cols + [p_col]
    )
    attractions[unique_col] = attractions[unique_col] * attractions[m_split_col]
    attractions = attractions.drop(m_split_col, axis='columns')

    # ## TIDY UP AND SORT ## #
    group_cols = idx_cols + [p_col, m_col]
    index_cols = group_cols.copy()
    index_cols.append(unique_col)

    attractions = attractions.reindex(index_cols, axis='columns')
    attractions = attractions.groupby(group_cols).sum().reset_index()
    attractions = attractions.sort_values(group_cols).reset_index(drop=True)

    attractions[m_col] = attractions[m_col].astype(int)
    attractions[p_col] = attractions[p_col].astype(int)

    # Control if required
    if do_ntem_control is not None:
        # Get ntem totals
        ntem_totals = pd.read_csv(control_path)
        ntem_lad_lookup = pd.read_csv(os.path.join(lad_lookup_dir,
                                                   lad_lookup_name))

        print("Performing NTEM constraint...")
        # TODO: Allow control_to_ntem() to take flexible col names
        attractions = attractions.rename(columns={p_col: 'p', m_col: 'm'})
        attractions, *_ = du.control_to_ntem(
            attractions,
            ntem_totals,
            ntem_lad_lookup,
            group_cols=['p', 'm'],
            base_value_name='trips',
            ntem_value_name='Attractions',
            purpose='hb'
        )
        attractions = attractions.rename(columns={'p': p_col, 'm': m_col})

    return attractions


def generate_attractions(employment: pd.DataFrame,
                         all_years: List[str],
                         attraction_weights_path: str,
                         mode_splits_path: str,
                         idx_cols: List[str],
                         emp_cat_col: str = 'employment_cat',
                         p_col: str = 'purpose',
                         m_col: str = 'mode',
                         m_split_col: str = 'mode_share',
                         ntem_control_dir: str = None,
                         lad_lookup_dir: str = None
                         ) -> pd.DataFrame:
    """
    Converts employment to attractions using attraction_weights

    Parameters
    ----------
    employment:
        Dataframe containing the employment data. This should contain all
        segmentation, and then a separate column for each year

    all_years:
        A list of all the year columns to be converted

    attraction_weights_path:
        Path the the attraction weights file. This file should contain a wide
        matrix, with the purposes as the index, and the employment categories
        as the columns.

    mode_splits_path:
        Path to the file of mode splits by 'p'

    idx_cols:
        The column names used to index the wide employment df. This should
        cover all segmentation in the employment

    emp_cat_col:
        The name of the column containing the employment categories

    p_col:
        The name of the column in attraction weights containing the purpose
        names.

    m_col:
        The name of the column in mode_splits_path containing the mode values.

    m_split_col
        The name of the column in mode_splits_path containing the mode share
        values.

    ntem_control_dir:
        Path to the file containing the data to control the produced
        attractions to. If left as None, no control will be carried out.

    lad_lookup_dir:
        Path to the file containing the conversion from msoa zoning to LAD
        zoning, to be used for controlling the attractions. If left as None, no
        control will be carried out.

    Returns
    -------
    attractions:
        A copy of the employment dataframe, with the yearly values converted
        to attractions.
    """
    # Init
    unique_col = 'trips'
    idx_cols = idx_cols.copy()
    idx_cols.remove(emp_cat_col)
    ntem_base_fname = 'ntem_pa_ave_wday_%s.csv'

    # Generate attractions per year
    yr_ph = dict()
    for year in all_years:
        print("\nConverting year %s to attractions..." % str(year))

        # Figure out the ntem control path
        if ntem_control_dir is not None:
            ntem_fname = ntem_base_fname % year
            ntem_control_path = os.path.join(ntem_control_dir, ntem_fname)
        else:
            ntem_control_path = None

        # Convert to wide format, for this single year
        yr_emp = employment.pivot_table(
            index=idx_cols,
            columns=emp_cat_col,
            values=year
        )

        # Convert employment to attractions for this year
        yr_ph[year] = merge_attraction_weights(
            employment=yr_emp,
            attraction_weights_path=attraction_weights_path,
            mode_splits_path=mode_splits_path,
            idx_cols=idx_cols,
            p_col=p_col,
            m_col=m_col,
            m_split_col=m_split_col,
            unique_col=unique_col,
            control_path=ntem_control_path,
            lad_lookup_dir=lad_lookup_dir

        )

    # Get all the attractions into a single df, efficiently
    attractions = du.combine_yearly_dfs(
        yr_ph,
        unique_col=unique_col,
        p_col=p_col
    )
    return attractions
