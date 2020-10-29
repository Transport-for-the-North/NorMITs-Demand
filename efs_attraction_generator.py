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

from itertools import product

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
            attraction_weights_path: str,
            employment_path: str = None,
            mode_splits_path: str = None,
            soc_to_sic_path: str = None,

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
        # Return previously created productions if we can
        fname = 'MSOA_aggregated_attractions.csv'
        final_output_path = os.path.join(out_path, fname)

        if not recreate_attractions and os.path.isfile(final_output_path):
            print("Found some already produced attractions. Using them!")
            return pd.read_csv(final_output_path)

        # Init
        internal_zone_col = 'msoa_zone_id'
        a_weights_p_col = 'purpose'
        mode_split_m_col = 'mode'
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

        # Build paths to the needed files
        imports = build_attraction_imports(
            import_home=import_home,
            base_year=base_year,
            attraction_weights_path=attraction_weights_path,
            employment_path=employment_path,
            mode_splits_path=mode_splits_path,
            soc_to_sic_path=soc_to_sic_path,
            ntem_control_dir=ntem_control_dir,
            lad_lookup_dir=lad_lookup_dir,
            set_controls=control_attractions
        )

        # ## BASE YEAR EMPLOYMENT ## #
        print("Loading the base year employment data...")
        base_year_emp = get_employment_data(
            import_path=imports['base_employment'],
            zone_col=internal_zone_col,
            emp_cat_col=emp_cat_col,
            msoa_path=msoa_conversion_path,
            return_format='long',
            value_col=base_year,
        )

        # TODO: TfN segmentation in attractions?
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
            attraction_weights_path=imports['weights'],
            mode_splits_path=imports['mode_splits'],
            idx_cols=idx_cols,
            emp_cat_col=emp_cat_col,
            p_col=a_weights_p_col,
            m_col=mode_split_m_col,
            ntem_control_dir=imports['ntem_control'],
            lad_lookup_dir=imports['lad_lookup']
        )

        # Write attractions to file
        if out_path is None:
            print("WARNING! No output path given. "
                  "Not writing attractions to file.")
        else:
            print("Writing productions to file...")
            fname = 'MSOA_attractions.csv'
            attractions.to_csv(os.path.join(out_path, fname), index=False)

        # ## CONVERT TO OLD EFS FORMAT ## #
        # Make sure columns are the correct data type
        attractions[a_weights_p_col] = attractions[a_weights_p_col].astype(int)
        attractions[mode_split_m_col] = attractions[mode_split_m_col].astype(int)
        attractions.columns = attractions.columns.astype(str)

        # Extract just the needed mode
        mask = attractions[mode_split_m_col].isin(m_needed)
        attractions = attractions[mask]
        attractions = attractions.drop(mode_split_m_col, axis='columns')

        # Rename columns so output of this function call is the same
        # as it was before the re-write
        attractions = du.convert_msoa_naming(
            attractions,
            msoa_col_name=internal_zone_col,
            msoa_path=msoa_conversion_path,
            to='int'
        )

        attractions = attractions.rename(
            columns={
                internal_zone_col: external_zone_col,
                a_weights_p_col: 'purpose_id',
            }
        )

        fname = 'MSOA_aggregated_attractions.csv'
        attractions.to_csv(os.path.join(out_path, fname), index=False)

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


def get_mode_splits(mode_splits_path: str,
                    zone_col: str = 'msoa_zone_id',
                    p_col: str = 'p',
                    m_col: str = 'm',
                    m_split_col: str = 'mode_share',
                    ret_zone_col: str = None,
                    ret_p_col: str = None,
                    ret_m_col: str = None,
                    infill: float = 0.0
                    ) -> pd.DataFrame:
    """
    Reads the mode splits from file

    Will make sure

    Parameters
    ----------
    mode_splits_path
    zone_col
    p_col
    m_col
    m_split_col
    ret_zone_col
    ret_p_col
    ret_m_col
    infill

    Returns
    -------

    """
    # Init
    ret_zone_col = zone_col if ret_zone_col is None else ret_zone_col
    ret_p_col = p_col if ret_p_col is None else ret_p_col
    ret_m_col = m_col if ret_m_col is None else ret_m_col
    mode_splits = pd.read_csv(mode_splits_path)

    # Aggregate any duplicates
    group_cols = list(mode_splits.columns)
    group_cols.remove(m_split_col)
    mode_splits = mode_splits.groupby(group_cols).sum().reset_index()

    # Get all unique values for our unique columns
    zones = mode_splits[zone_col].unique()
    ps = mode_splits[p_col].unique()
    ms = mode_splits[m_col].unique()

    # Create a new placeholder df containing every combination of the
    # unique columns
    ph = dict()
    ph[zone_col], ph[p_col], ph[m_col] = zip(*product(zones, ps, ms))
    ph = pd.DataFrame(ph)

    # Where a combination of segmentation does not exist, infill
    mode_splits = ph.merge(
        mode_splits,
        how='left',
        on=[zone_col, p_col, m_col]
    ).fillna(infill)

    # rename the return
    mode_splits = mode_splits.rename(
        columns={
            zone_col: ret_zone_col,
            p_col: ret_p_col,
            m_col: ret_m_col
        }
    )

    return mode_splits


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
                             zone_col: str = 'msoa_zone_id',
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

    zone_col:
        The name of the column in employment containing the zone data.

    p_col:
        The name to give to the purpose values column.

    m_col:
        The name to give to the mode values column.

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
    mode_splits = get_mode_splits(
        mode_splits_path,
        ret_zone_col=zone_col,
        ret_p_col=p_col,
        ret_m_col=m_col
    )

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
    if do_ntem_control:
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
        names. This is also the name that will be given to the column
        containing purpose data in the return df.

    m_col:
        The name to give to the column containing the mode values

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


def build_attraction_imports(import_home: str,
                             base_year: str,
                             attraction_weights_path: str,
                             employment_path: str = None,
                             mode_splits_path: str = None,
                             soc_to_sic_path: str = None,
                             ntem_control_dir: str = None,
                             lad_lookup_dir: str = None,
                             set_controls: bool = True
                             ) -> Dict[str, str]:
    """
    Builds a dictionary of attraction import paths, forming a standard calling
    procedure for attraction imports. Arguments allow default paths to be
    replaced.

    Parameters
    ----------
    import_home:
        The base path to base all of the other import paths from. This
        should usually be "Y:/NorMITs Demand/import" for business as usual.

    base_year:
        The base year the model is being run at. This is used to determine the
        correct default employment_path and soc_to_sic path

    attraction_weights_path:
        The path to the attractions weights. Unable to give a default value for
        this as it changes depending on the mode.

    employment_path:
        An alternate base year employment import path to use. File will need to
        follow the same format as default file.

    mode_splits_path:
        An alternate mode splits import path to use. File will need to follow
        the same format as default file.

    soc_to_sic_path:
        An alternate soc to sic import path to use. File will need to follow
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
        'weights'
        'base_employment'
        'mode_splits'
        'soc_to_sic'
        'ntem_control'
        'lad_lookup'
    """
    if employment_path is None:
        path = 'attractions/non_freight_msoa_%s.csv' % base_year
        employment_path = os.path.join(import_home, path)

    if mode_splits_path is None:
        path = 'attractions/attraction_mode_split.csv'
        mode_splits_path = os.path.join(import_home, path)

    if soc_to_sic_path is None:
        path = 'attractions/soc_2_digit_sic_%s.csv' % base_year
        soc_to_sic_path = os.path.join(import_home, path)

    if set_controls and ntem_control_dir is None:
        path = 'ntem_constraints'
        ntem_control_dir = os.path.join(import_home, path)

    if set_controls and lad_lookup_dir is None:
        lad_lookup_dir = import_home

    # Assign to dict
    imports = {
        'weights': attraction_weights_path,
        'base_employment': employment_path,
        'mode_splits': mode_splits_path,
        'soc_to_sic': soc_to_sic_path,
        'ntem_control': ntem_control_dir,
        'lad_lookup': lad_lookup_dir
    }

    return imports
