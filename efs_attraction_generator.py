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
from typing import Tuple
from typing import Union

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
            soc_weights_path: str = None,

            # Production control file
            ntem_control_dir: str = None,
            lad_lookup_dir: str = None,
            control_attractions: bool = True,
            control_fy_attractions: bool = True,

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
            zoning_system: str = 'msoa',
            no_neg_growth: bool = True,
            employment_infill: float = 0.001,

            # Handle outputs
            audits: bool = True,
            out_path: str = None,
            recreate_attractions: bool = True,
            aggregate_nhb_tp: bool = True
            ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Attraction model for the external forecast system. This has been
        written to align with TMS attraction generation, with the addition of
        future year employment growth and attraction generation.

        Performs the following functions:
            - Reads in the base year employment data to create the base year
              employment numbers
            - Grows the base year employment by employment_growth factors,
              resulting in future year employment numbers.
            - Combines base and future year employment numbers with
              attraction_weights (currently the same across all years) to
              produce the base and future year attraction values (for all
              modes).
            - Finally, splits the produced attractions to only return the
              desired mode. This dataframe is then returned.

        Parameters
        ----------
        base_year:
            The base year of the forecast.

        future_years:
            The future years to forecast.

        employment_growth:
            dataframe containing the future year growth values for
            growing the base year employment. Must be segmented by the same
            zoning system (at least) as employment_path data (usually
            msoa_zone_id).

        employment_constraint:
            Values to constrain the employment numbers to. See
            efs_constrainer.ForecastConstrainer() for further information.

        import_home:
            The home directory to find all the attraction imports. Usually
            Y:/NorMITs Demand/import

        msoa_conversion_path:
            Path to the file containing the conversion from msoa integer
            identifiers to the msoa string code identifiers. Hoping to remove
            this in a future update and align all of EFS to use msoa string
            code identifiers.

        attraction_weights_path:
            The path to alternate attraction weights import data. If left as
            None, the attraction model will use the default land use data.

        employment_path:
            The path to alternate employment import data. If left as None, the
            attraction model will use the default land use data.

        mode_splits_path:
            The path to alternate mode splits import data. If left as None, the
            attraction model will use the default land use data.

        soc_weights_path:
            The path to alternate soc weights import data. If left as None, the
            attraction model will use the default land use data.

        ntem_control_dir:
            The path to alternate ntem control directory. If left as None, the
            attraction model will use the default land use data.

        lad_lookup_dir:
            The path to alternate lad to msoa import data. If left as None, the
            attraction model will use the default land use data.

        control_attractions:
            Whether to control the generated attractions to the constraints
            given in ntem_control_dir or not.

        control_fy_attractions:
            Whether to control the generated future year attractions to the
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
            The levels of segmentation that exist in the employment_path data.
            If not defined, will default to: [emp_cat_col].

        external_zone_col:
            The name of the zone column, as used externally to this attraction
            model. This is used to make sure this model can translate to the
            zoning name used internally in employment_path and
            attraction_weights data.

        no_neg_growth:
            Whether to ensure there is no negative growth. If True, any growth
            values below 0 will be replaced with employment_infill.

        employment_infill:
            If no_neg_growth is True, this value will be used to replace all
            values that are less than 0.

        audits:
            Whether to output print_audits to the terminal during running. This can
            be used to monitor the employment and attraction numbers being
            generated and constrained.

        out_path:
            Path to the directory to output the employment and attractions
            dataframes.

        recreate_attractions:
            Whether to recreate the attractions or not. If False, it will
            look in out_path for previously produced attractions and return
            them. If none can be found, they will be generated.

        aggregate_nhb_tp:
            Whether to aggregate the time period before writing to disk for
            nhb attractions or not

        Returns
        -------
        segmented_attractions:
            Attractions for mode m_needed, segmented by all segments possible
            in the input data.

        segmented_nhb_attractions:
            NHB attractions for mode m_needed, segmented by all segments
            possible in the input data.
        """
        # Return previously created productions if we can
        fname = consts.ATTRS_FNAME % (zoning_system, 'hb')
        nhb_fname = consts.ATTRS_FNAME % (zoning_system, 'nhb')
        final_output_path = os.path.join(out_path, fname)
        nhb_output_path = os.path.join(out_path, nhb_fname)

        if (not recreate_attractions
                and os.path.isfile(final_output_path)
                and os.path.isfile(nhb_output_path)):
            print("Found some already produced attractions. Using them!")
            return pd.read_csv(final_output_path), pd.read_csv(nhb_output_path)

        # Init
        internal_zone_col = 'msoa_zone_id'
        zoning_system = du.validate_zoning_system(zoning_system)
        a_weights_p_col = 'purpose'
        mode_split_m_col = 'mode'
        emp_cat_col = 'employment_cat'
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
            soc_weights_path=soc_weights_path,
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
            return_format='long',
            value_col=base_year,
        )

        # Audit employment numbers
        mask = (base_year_emp[emp_cat_col] == 'E01')
        total_base_year_emp = base_year_emp.loc[mask, base_year].sum()
        du.print_w_toggle("Base Year Employment: %d" % total_base_year_emp,
                          echo=audits)

        # ## FUTURE YEAR EMPLOYMENT ## #
        print("Generating future year employment data...")
        # If soc splits in the growth factors, we have a few extra steps
        if 'soc' in employment_growth:
            # Add Soc splits into the base year
            base_year_emp = split_by_soc(
                df=base_year_emp,
                soc_weights=get_soc_weights(imports['soc_weights']),
                unique_col=base_year
            )

            # Aggregate the growth factors to remove extra segmentation
            group_cols = [internal_zone_col, 'soc']
            index_cols = group_cols.copy() + all_years

            employment_growth = employment_growth.reindex(columns=index_cols)
            employment_growth = employment_growth.groupby(group_cols).mean().reset_index()

        # Merge on all possible segmentations - not years
        merge_cols = du.intersection(list(base_year_emp), list(employment_growth))
        merge_cols = du.list_safe_remove(merge_cols, all_years)

        employment = du.grow_to_future_years(
            base_year_df=base_year_emp,
            growth_df=employment_growth,
            base_year=base_year,
            future_years=future_years,
            growth_merge_cols=merge_cols,
            no_neg_growth=no_neg_growth,
            infill=employment_infill
        )

        # Now need te remove soc splits
        if 'soc' in employment_growth:
            pass

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
                print('. Total jobs for year %s is: %.4f'
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
            path = os.path.join(out_path, consts.EMP_FNAME % zoning_system)
            employment.to_csv(path, index=False)

        # ## CREATE ATTRACTIONS ## #
        # Index by as much segmentation as possible
        idx_cols = list(employment.columns)
        for unq_col in all_years:
            idx_cols.remove(unq_col)

        attractions, nhb_att = generate_attractions(
            employment=employment,
            base_year=base_year,
            future_years=future_years,
            attraction_weights_path=imports['weights'],
            mode_splits_path=imports['mode_splits'],
            soc_weights_path=imports['soc_weights'],
            idx_cols=idx_cols,
            emp_cat_col=emp_cat_col,
            p_col=a_weights_p_col,
            m_col=mode_split_m_col,
            ntem_control_dir=imports['ntem_control'],
            lad_lookup_dir=imports['lad_lookup'],
            control_fy_attractions=control_fy_attractions
        )

        # Aggregate nhb trips if needed
        if aggregate_nhb_tp:
            reindex_cols = list(nhb_att)
            reindex_cols.remove('tp')
            group_cols = [x for x in reindex_cols.copy() if x not in all_years]

            nhb_att = nhb_att.reindex(reindex_cols, axis='columns')
            nhb_att = nhb_att.groupby(group_cols).sum().reset_index()

        # Align purpose and mode columns to standards
        p_col = 'p'
        m_col = 'm'
        columns = {a_weights_p_col: p_col, mode_split_m_col: m_col}
        attractions = attractions.rename(columns=columns)
        nhb_att = nhb_att.rename(columns=columns)

        # Write attractions to file
        if out_path is None:
            print("WARNING! No output path given. "
                  "Not writing attractions to file.")
        else:
            print("Writing attractions to file...")
            fname = consts.ATTRS_FNAME % (zoning_system, 'raw_hb')
            nhb_fname = consts.ATTRS_FNAME % (zoning_system, 'raw_nhb')
            attractions.to_csv(os.path.join(out_path, fname), index=False)
            nhb_att.to_csv(os.path.join(out_path, nhb_fname), index=False)

        # TODO: functionalise conversion to old efs
        # ## CONVERT TO OLD EFS FORMAT ## #
        # Make sure columns are the correct data type
        attractions[p_col] = attractions[p_col].astype(int)
        attractions[m_col] = attractions[m_col].astype(int)
        attractions.columns = attractions.columns.astype(str)

        nhb_att[p_col] = nhb_att[p_col].astype(int)
        nhb_att[m_col] = nhb_att[m_col].astype(int)
        nhb_att.columns = nhb_att.columns.astype(str)

        # Extract just the needed mode
        mask = attractions[m_col].isin(m_needed)
        attractions = attractions[mask]
        attractions = attractions.drop(m_col, axis='columns')

        mask = nhb_att[m_col].isin(m_needed)
        nhb_att = nhb_att[mask]
        nhb_att = nhb_att.drop(m_col, axis='columns')

        # Rename columns so output of this function call is the same
        # as it was before the re-write
        attractions = du.convert_msoa_naming(
            attractions,
            msoa_col_name=internal_zone_col,
            msoa_path=msoa_conversion_path,
            to='int'
        )

        nhb_att = du.convert_msoa_naming(
            nhb_att,
            msoa_col_name=internal_zone_col,
            msoa_path=msoa_conversion_path,
            to='int'
        )

        # Re-align col names for returning
        columns = {internal_zone_col: external_zone_col, p_col: 'purpose_id'}
        attractions = attractions.rename(columns=columns)
        nhb_att = nhb_att.rename(columns=columns)

        fname = consts.ATTRS_FNAME % (zoning_system, 'hb')
        nhb_fname = consts.ATTRS_FNAME % (zoning_system, 'nhb')
        attractions.to_csv(os.path.join(out_path, fname), index=False)
        nhb_att.to_csv(os.path.join(out_path, nhb_fname), index=False)

        return attractions, nhb_att

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
                    ret_m_split_col: str = None,
                    infill: float = 0.0
                    ) -> pd.DataFrame:
    """
    Reads the mode splits from file

    Will make sure all combinations of purposes and modes exist, and aggregate
    any duplicate entries.

    Parameters
    ----------
    mode_splits_path:
        Path to the mode splits file to read in

    zone_col:
        Name of the column containing the zone data

    p_col:
        Name of the columns containing the purpose data

    m_col:
        Name of the column containing the mode data

    m_split_col:
        Name of the column containing the mode split factors

    ret_zone_col:
        The name to give to zone_col on return.

    ret_p_col:
        The name to give to p_col on return.

    ret_m_col
        The name to give to m_col on return.

    ret_m_split_col
        The name to give to m_split_col on return.

    infill:
        The value to infill any missing p/m combinations

    Returns
    -------
    mode_splits:
        Dataframe containing the following columns [ret_zone_col, ret_p_col,
        ret_m_col, ret_m_split_col]
    """
    # Init
    ret_zone_col = zone_col if ret_zone_col is None else ret_zone_col
    ret_p_col = p_col if ret_p_col is None else ret_p_col
    ret_m_col = m_col if ret_m_col is None else ret_m_col
    ret_m_split_col = m_split_col if ret_m_split_col is None else ret_m_split_col
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
            m_col: ret_m_col,
            m_split_col: ret_m_split_col
        }
    )

    return mode_splits


def get_soc_weights(soc_weights_path: str,
                    zone_col: str = 'msoa_zone_id',
                    soc_col: str = 'soc_class',
                    jobs_col: str = 'seg_jobs',
                    str_cols: bool = False
                    ) -> pd.DataFrame:
    """
    Converts the input file into soc weights by zone

    Parameters
    ----------
    soc_weights_path:
        Path to the soc weights file. Must contain at least the following
        column names [zone_col, soc_col, jobs_col]

    zone_col:
        The column name in soc_weights_path that contains the zone data.

    soc_col:
        The column name in soc_weights_path that contains the soc categories.

    jobs_col:
        The column name in soc_weights_path that contains the number of jobs
        data.

    str_cols:
        Whether the return dataframe columns should be as [soc1, soc2, ...]
        (if True), or [1, 2, ...] (if False).

    Returns
    -------
    soc_weights:
        a wide dataframe with zones from zone_col as the column names, and
        soc categories from soc_col as columns. Each row of soc weights will
        sum to 1.
    """
    # Init
    soc_weighted_jobs = pd.read_csv(soc_weights_path)

    # Convert soc numbers to names (to differentiate from ns)
    soc_weighted_jobs[soc_col] = soc_weighted_jobs[soc_col].astype(int).astype(str)

    if str_cols:
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


def split_by_soc(df: pd.DataFrame,
                 soc_weights: pd.DataFrame,
                 zone_col: str = 'msoa_zone_id',
                 p_col: str = 'p',
                 unique_col: str = 'trips',
                 soc_col: str = 'soc'
                 ) -> pd.DataFrame:
    """
    Splits df purposes by the soc_weights given.

    Parameters
    ----------
    df:
        Dataframe to add soc splits too. Must contain the following columns
        [zone_col, p_col, unique_col]

    soc_weights:
        Wide dataframe containing the soc splitting weights. Must have a
        zone_col columns, and all other columns are the soc categories to split
        by.

    zone_col:
        The name of the column in df and soc_weights that contains the
        zone data.

    p_col:
        Name of the column in df that contains purpose data.

    unique_col:
        Name of the column in df that contains the unique data (usually the
        number of trips at that row of segmentation)

    soc_col:
        The name to give to the added soc column in the return dataframe.

    Returns
    -------
    soc_split_df:
        df with an added soc_col. Unique_col will be split by the weights
        given
    """
    # Init
    soc_cats = list(soc_weights.columns)

    # Figure out which rows need splitting
    if p_col in df:
        mask = (df[p_col].isin(consts.SOC_P))
        split_df = df[mask].copy()
        retain_df = df[~mask].copy()
        id_cols = [zone_col, p_col]
    else:
        # Split on all data
        split_df = df.copy()
        retain_df = None
        id_cols = [zone_col]

    # Split by soc weights
    split_df = pd.merge(
        split_df,
        soc_weights,
        on=zone_col
    )

    for soc in soc_cats:
        split_df[soc] *= split_df[unique_col]

    # Tidy up the split dataframe ready to re-merge
    split_df = split_df.drop(unique_col, axis='columns')
    split_df = split_df.melt(
        id_vars=id_cols,
        value_vars=soc_cats,
        var_name=soc_col,
        value_name=unique_col,
    )

    # Don't need to stick back together
    if retain_df is None:
        return split_df

    # Add the soc col to the retained values to match
    retain_df[soc_col] = 0

    # Finally, stick the two back together
    return pd.concat([split_df, retain_df])


def merge_attraction_weights(employment: pd.DataFrame,
                             attraction_weights_path: str,
                             mode_splits_path: str,
                             soc_weights: pd.DataFrame = None,
                             idx_cols: List[str] = None,
                             zone_col: str = 'msoa_zone_id',
                             p_col: str = 'purpose',
                             m_col: str = 'mode',
                             m_split_col: str = 'mode_share',
                             unique_col: str = 'trips',
                             control_path: str = None,
                             lad_lookup_dir: str = None,
                             lad_lookup_name: str = consts.DEFAULT_LAD_LOOKUP,
                             ) -> Tuple[pd.DataFrame, pd.DataFrame]:
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

    soc_weights:
        dataframe containing the soc weights by model zone. This dataframe
        should follow the same format as that returned from get_soc_weights().

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
    idx_cols = ['msoa_zone_id'] if idx_cols is None else idx_cols.copy()
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
    idx_cols.append(p_col)

    # Need to convert the str purposes into int
    attractions[p_col] = attractions[p_col].apply(lambda row: consts.P_STR2INT[row])

    # Split by soc categories if needed
    if soc_weights is not None:
        soc_col = 'soc'
        attractions = split_by_soc(
            attractions,
            soc_weights,
            zone_col=zone_col,
            p_col=p_col,
            unique_col=unique_col,
            soc_col=soc_col
        )
        idx_cols.append(soc_col)

    # ## SPLIT THE ATTRACTIONS BY MODE ## #
    # Read in and apply mode splits
    mode_splits = get_mode_splits(
        mode_splits_path,
        ret_zone_col=zone_col,
        ret_p_col=p_col,
        ret_m_col=m_col
    )

    # Merge on all possible columns
    merge_cols = du.intersection(list(mode_splits), idx_cols)
    attractions = attractions.merge(
        mode_splits,
        how='left',
        on=merge_cols
    )
    idx_cols.append(m_col)

    attractions[unique_col] = attractions[unique_col] * attractions[m_split_col]
    attractions = attractions.drop(m_split_col, axis='columns')

    # ## TIDY UP AND SORT ## #
    group_cols = idx_cols
    index_cols = group_cols.copy()
    index_cols.append(unique_col)

    attractions = attractions.reindex(index_cols, axis='columns')
    attractions = attractions.groupby(group_cols).sum().reset_index()
    attractions = attractions.sort_values(group_cols).reset_index(drop=True)

    attractions[m_col] = attractions[m_col].astype(int)
    attractions[p_col] = attractions[p_col].astype(int)

    # ## GENERATE NHB ATTRACTIONS ## #
    # Copy and rename Hb attractions
    nhb_attractions = attractions.copy()
    nhb_attractions = nhb_attractions[nhb_attractions[p_col] != 1]
    nhb_attractions = nhb_attractions[nhb_attractions[p_col] != 7]
    nhb_attractions[p_col] += 10

    # In the meantime hack infill time period in nhb
    tp_infill = pd.DataFrame({'ph': [1, 1, 1, 1],
                              'tp': [1, 2, 3, 4]})
    nhb_attractions['ph'] = 1
    nhb_attractions = pd.merge(
        nhb_attractions,
        tp_infill,
        how='left',
        on='ph'
    ).drop('ph', axis='columns')
    del tp_infill

    # Control if required
    if do_ntem_control:
        # Get ntem totals
        ntem_totals = pd.read_csv(control_path)
        ntem_lad_lookup = pd.read_csv(os.path.join(lad_lookup_dir,
                                                   lad_lookup_name))

        print("Performing HB NTEM constraint...")
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

        print("Performing NHB NTEM constraint...")
        nhb_attractions = nhb_attractions.rename(columns={p_col: 'p', m_col: 'm'})
        nhb_attractions, *_ = du.control_to_ntem(
            nhb_attractions,
            ntem_totals,
            ntem_lad_lookup,
            group_cols=['p', 'm', 'tp'],
            base_value_name='trips',
            ntem_value_name='Attractions',
            purpose='nhb'
        )
        nhb_attractions = nhb_attractions.rename(columns={'p': p_col, 'm': m_col})

    return attractions, nhb_attractions


def generate_attractions(employment: pd.DataFrame,
                         base_year: str,
                         future_years: List[str],
                         attraction_weights_path: str,
                         mode_splits_path: str,
                         soc_weights_path: str,
                         idx_cols: List[str],
                         emp_cat_col: str = 'employment_cat',
                         p_col: str = 'purpose',
                         m_col: str = 'mode',
                         m_split_col: str = 'mode_share',
                         ntem_control_dir: str = None,
                         lad_lookup_dir: str = None,
                         soc_split: bool = True,
                         control_fy_attractions: bool = True
                         ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Converts employment to attractions using attraction_weights

    Parameters
    ----------
    employment:
        Dataframe containing the employment data. This should contain all
        segmentation, and then a separate column for each year

    base_year:
        The base year of this model run. This will be prepended to future_years
        to run attraction generation for all years

    future_years:
        A list of all the future year columns to be converted

    attraction_weights_path:
        Path the the attraction weights file. This file should contain a wide
        matrix, with the purposes as the index, and the employment categories
        as the columns.

    mode_splits_path:
        Path to the file of mode splits by 'p'

    soc_weights_path:
        Path to the file of soc weights by zone. This file does not
        specifically need to be weights, as the reader will do the conversion
        on data ingestion.

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

    soc_split:
        Whether to apply the soc splits from soc_weights_path to the
        attractions or not.

    control_fy_attractions:
        Whether to control the generated future year attractions to the
        constraints given in ntem_control_dir or not. When running for
        scenarios other than the base NTEM, this should be False.

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
    all_years = [base_year] + future_years

    # Get the soc weights per zone - may want to move this into each year
    # in future
    soc_weights = None
    if soc_split:
        soc_weights = get_soc_weights(soc_weights_path)

    # Generate attractions per year
    yr_ph = dict()
    yr_ph_nhb = dict()
    for year in all_years:
        print("\nConverting year %s to attractions..." % str(year))

        # Only only set the control path if we need to constrain
        if not control_fy_attractions and year != base_year:
            ntem_control_path = None
        elif ntem_control_dir is not None:
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
        yr_ph[year], yr_ph_nhb[year] = merge_attraction_weights(
            employment=yr_emp,
            attraction_weights_path=attraction_weights_path,
            mode_splits_path=mode_splits_path,
            soc_weights=soc_weights,
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
    nhb_attractions = du.combine_yearly_dfs(
        yr_ph_nhb,
        unique_col=unique_col,
        p_col=p_col
    )

    return attractions, nhb_attractions


def build_attraction_imports(import_home: str,
                             base_year: str,
                             attraction_weights_path: str,
                             employment_path: str = None,
                             mode_splits_path: str = None,
                             soc_weights_path: str = None,
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

    soc_weights_path:
        An alternate soc weights import path to use. File will need to follow
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

    if soc_weights_path is None:
        path = 'attractions/soc_2_digit_sic_%s.csv' % base_year
        soc_weights_path = os.path.join(import_home, path)

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
        'soc_weights': soc_weights_path,
        'ntem_control': ntem_control_dir,
        'lad_lookup': lad_lookup_dir
    }

    return imports
