# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 14:32:53 2020

@author: Sneezy
"""

import os
import numpy as np
import pandas as pd

from typing import List
from typing import Dict
from typing import Tuple

from itertools import product

from tqdm import tqdm

# self imports
import normits_demand as nd
from normits_demand import efs_constants as consts
from normits_demand.utils import general as du

from normits_demand.constraints import ntem_control as ntem

from normits_demand.d_log import processor as dlog_p


# TODO: Align attraction model class to NHB Production Model

# BACKLOG: Integrate EFS Attraction Model into TMS trip end models
#  labels: EFS, demand merge


class EFSAttractionGenerator:

    def __init__(self,
                 model_name: str,
                 seg_level: str = 'tfn',
                 zoning_system: str = 'msoa',
                 tag_certainty_bounds=consts.TAG_CERTAINTY_BOUNDS
                 ):
        """
        #TODO
        """
        # Validate inputs
        seg_level = du.validate_seg_level(seg_level)
        model_name = du.validate_model_name(model_name)
        zoning_system = du.validate_zoning_system(zoning_system)

        # Assign
        self.tag_certainty_bounds = tag_certainty_bounds

        self.model_name = model_name

        self.zoning_system = zoning_system
        self.zone_col = '%s_zone_id' % zoning_system
        self.emp_cat_col = 'employment_cat'

        self.emp_fname = consts.EMP_FNAME % zoning_system

        # Define the segmentation we're using
        if seg_level == 'tfn':
            self.emp_segments = [self.emp_cat_col, 'soc']
            self.return_segments = [self.zone_col, 'p', 'soc']
        else:
            raise ValueError(
                "'%s' is a valid segmentation level, but I don't have a way "
                "of determining which segments to use for it. You should add "
                "one!" % seg_level
            )

    def run(self,
            out_path: str,
            base_year: str,
            future_years: List[str],

            # Build import paths
            import_home: str,
            export_home: str,

            # Employment data
            by_emp_import_path: nd.PathLike,
            emp_constraint: pd.DataFrame,
            fy_emp_import_dir: nd.PathLike = None,

            # Alternate population/attraction creation files
            attraction_weights_path: str = None,
            mode_splits_path: str = None,
            soc_weights_path: str = None,
            msoa_lookup_path: str = None,

            # Alternate output paths
            audit_write_dir: str = None,

            # Production control file
            ntem_control_dir: str = None,
            lad_lookup_dir: str = None,
            control_attractions: bool = True,
            control_fy_attractions: bool = True,

            # D-Log
            dlog: str = None,

            # Employment constraints
            pre_dlog_constraint: bool = False,
            post_dlog_constraint: bool = None,
            designated_area: pd.DataFrame = None,

            # Segmentation controls
            m_needed: List[int] = consts.MODES_NEEDED,
            segmentation_cols: List[str] = None,
            external_zone_col: str = 'model_zone_id',

            # Handle outputs
            audits: bool = True,
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
        out_path:
            Path to the directory to output the employment and attractions
            dataframes.

        base_year:
            The base year of the forecast.

        future_years:
            The future years to forecast.

        employment_import_path:
            Path to the directory containing the NorMITs Land Use outputs
            for future year employment estimates. The filenames will
            be automatically generated based on efs_consts.LU_EMP_FNAME

        emp_constraint:
            Values to constrain the employment numbers to.

        import_home:
            The home directory to find all the attraction imports. Usually
            Y:/NorMITs Demand/import

        export_home:
            Path to the export home of this instance of outputs. This is
            usually related to a specific run of the ExternalForecastSystem,
            and should be gotten from there using generate_output_paths().
            e.g. 'E:/NorMITs Demand/norms_2015/v2_3-EFS_Output/iter1'

        attraction_weights_path:
            The path to alternate attraction weights import data. If left as
            None, the attraction model will use the default land use data.

        mode_splits_path:
            The path to alternate mode splits import data. If left as None, the
            attraction model will use the default land use data.

        soc_weights_path:
            The path to alternate soc weights import data. If left as None, the
            attraction model will use the default land use data.

        msoa_lookup_path:
            The path to alternate msoa lookup import data. If left as None,
            the attraction model will use the default msoa lookup path.

        audit_write_dir:
            Alternate path to write the audits. If left as None, the default
            location is used.

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

        dlog:
            TODO: Clarify what format D_log data comes in as

        pre_dlog_constraint:
            Whether to constrain the population before applying the dlog or
            not.

        post_dlog_constraint
            Whether to constrain the population after applying the dlog or
            not.

        designated_area:
            TODO cladifiy what the designated_area is

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

        audits:
            Whether to output print_audits to the terminal during running. This can
            be used to monitor the employment and attraction numbers being
            generated and constrained.

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
        fname = consts.ATTRS_FNAME % (self.zoning_system, 'hb')
        nhb_fname = consts.ATTRS_FNAME % (self.zoning_system, 'nhb')
        final_output_path = os.path.join(out_path, fname)
        nhb_output_path = os.path.join(out_path, nhb_fname)

        if (not recreate_attractions
                and os.path.isfile(final_output_path)
                and os.path.isfile(nhb_output_path)):
            print("Found some already produced attractions. Using them!")
            return pd.read_csv(final_output_path), pd.read_csv(nhb_output_path)

        # Init
        a_weights_p_col = 'purpose'
        mode_split_m_col = 'mode'
        all_years = [str(x) for x in [base_year] + future_years]

        # If not set, perform the post_dlog_constrain if dlog is on
        if post_dlog_constraint is None:
            post_dlog_constraint = dlog is not None

        # TODO: Make this more adaptive
        # Set the level of segmentation being used
        if segmentation_cols is None:
            segmentation_cols = self.emp_segments

        # Fix column naming if different
        if external_zone_col != self.zone_col:
            designated_area = designated_area.copy().rename(
                columns={external_zone_col: self.zone_col}
            )
            emp_constraint = emp_constraint.rename(
                columns={external_zone_col: self.zone_col}
            )

        # Build paths to the needed files
        imports = build_attraction_imports(
            import_home=import_home,
            base_year=base_year,
            attraction_weights_path=attraction_weights_path,
            mode_splits_path=mode_splits_path,
            soc_weights_path=soc_weights_path,
            msoa_lookup_path=msoa_lookup_path,
            ntem_control_dir=ntem_control_dir,
            lad_lookup_dir=lad_lookup_dir,
            set_controls=control_attractions
        )

        exports = build_attraction_exports(
            export_home=export_home,
            audit_write_dir=audit_write_dir
        )

        # # ## READ IN EMPLOYMENT DATA ## #
        employment = get_emp_data_from_land_use(
            by_emp_import_path=by_emp_import_path,
            fy_emp_import_dir=fy_emp_import_dir,
            base_year=base_year,
            future_years=future_years,
            segmentation_cols=segmentation_cols,
        )

        # Remove E01 for constraints / dlog
        employment = du.remove_all_commute_cat(
            df=employment,
            emp_cat_col=self.emp_cat_col
        )

        # ## CONSTRAIN POPULATION ## #
        if pre_dlog_constraint:
            print("Performing the first constraint on employment...")
            print(". Pre Constraint:\n%s" % employment[future_years].sum())
            constraint_segments = du.intersection(segmentation_cols,
                                                  emp_constraint)

            employment = dlog_p.constrain_forecast(
                employment,
                emp_constraint,
                designated_area,
                base_year,
                future_years,
                self.zone_col,
                segment_cols=constraint_segments
            )
            print(". Post Constraint:\n%s" % employment[future_years].sum())

        # ## INTEGRATE D-LOG ## #
        if dlog is not None:
            print("Integrating the development log...")

            dlog_segments = ["employment_cat"]
            if "soc" in list(employment):
                dlog_segments.append("soc")

            employment, hg_zones = dlog_p.apply_d_log(
                pre_dlog_df=employment,
                base_year=base_year,
                future_years=future_years,
                dlog_path=dlog,
                msoa_conversion_path=imports['msoa_lookup'],
                constraints_zone_equivalence=designated_area,
                segment_cols=dlog_segments,
                dlog_conversion_factor=1.0,
                dlog_data_column_key="employees",
                perform_constraint=False,
                audit_location=out_path
            )

            # Save High Growth (Exceptional) zones to file
            hg_zones.to_csv(os.path.join(out_path, consts.EG_FNAME),
                            index=False)

        # ## POST D-LOG CONSTRAINT ## #
        if post_dlog_constraint:
            pd.set_option('display.float_format', str)
            print("Performing the post-development log constraint on employment...")
            print(". Pre Constraint:\n%s" % employment[future_years].sum())
            print(". Constraint:\n%s" % emp_constraint[future_years].sum())
            constraint_segments = du.intersection(segmentation_cols,
                                                  emp_constraint)

            employment = dlog_p.constrain_forecast(
                employment,
                emp_constraint,
                designated_area,
                base_year,
                future_years,
                self.zone_col,
                segment_cols=constraint_segments
            )
            print(". Post Constraint:\n%s" % employment[future_years].sum())

        # D-Log and constraints done. Need to add E01 back in
        employment = du.add_all_commute_cat(
            df=employment,
            emp_cat_col=self.emp_cat_col,
            unique_data_cols=all_years
        )

        # Write the produced employment to file
        # Earlier than previously to also save the soc segmentation
        if out_path is None:
            print("WARNING! No output path given. "
                  "Not writing employment to file.")
        else:
            print("Writing employment to file...")
            path = os.path.join(out_path, consts.EMP_FNAME % self.zone_col)
            employment.to_csv(path, index=False)

        if 'soc' not in list(employment):
            self.emp_segments = du.list_safe_remove(self.emp_segments, ['soc'])

        # Reindex and sum
        group_cols = [self.zone_col] + self.emp_segments
        index_cols = group_cols.copy() + all_years
        employment = employment.reindex(index_cols, axis='columns')
        employment = employment.groupby(group_cols).sum().reset_index()

        # Population Audit
        if audits:
            print('\n', '-'*15, 'Employment Audit', '-'*15)
            mask = (employment[self.emp_cat_col] == 'E01')
            for year in all_years:
                total_emp = employment.loc[mask, year].sum()
                print('. Total jobs for year %s is: %.4f'
                      % (str(year), total_emp))
            print('\n')

        # Write the produced employment to file
        print("Writing employment to file...")
        employment_output = os.path.join(out_path, self.emp_fname)
        employment.to_csv(employment_output, index=False)

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
            emp_cat_col=self.emp_cat_col,
            p_col=a_weights_p_col,
            m_col=mode_split_m_col,
        )

        # Aggregate nhb trips if needed
        if aggregate_nhb_tp:
            reindex_cols = list(nhb_att)
            reindex_cols.remove('tp')
            group_cols = [x for x in reindex_cols.copy() if x not in all_years]

            nhb_att = nhb_att.reindex(reindex_cols, axis='columns')
            nhb_att = nhb_att.groupby(group_cols).sum().reset_index()

        # ## TIDY UP THE VECTORS ## #
        p_col = 'p'
        m_col = 'm'
        soc_col = 'soc'
        columns = {a_weights_p_col: p_col, mode_split_m_col: m_col}
        attractions = attractions.rename(columns=columns)
        nhb_att = nhb_att.rename(columns=columns)

        # ## OPTIONALLY CONTROL TO NTEM ## #
        lad_lookup_path = os.path.join(imports['lad_lookup'],
                                       consts.DEFAULT_LAD_LOOKUP)

        attractions = ntem.control_vector_to_ntem(
            vector=attractions,
            vector_type='attractions',
            trip_origin='hb',
            ntem_dir=imports['ntem_control'],
            lad_lookup_path=lad_lookup_path,
            base_year=base_year,
            future_years=future_years,
            control_base_year=control_attractions,
            control_future_years=control_fy_attractions,
            reports_dir=exports['audits'],
        )

        nhb_att = ntem.control_vector_to_ntem(
            vector=nhb_att,
            vector_type='attractions',
            trip_origin='nhb',
            ntem_dir=imports['ntem_control'],
            lad_lookup_path=lad_lookup_path,
            base_year=base_year,
            future_years=future_years,
            control_base_year=control_attractions,
            control_future_years=control_fy_attractions,
            reports_dir=exports['audits']
        )

        # Write attractions to file
        print("Writing attractions to disk...")
        fname = consts.ATTRS_FNAME % (self.zoning_system, 'raw_hb')
        nhb_fname = consts.ATTRS_FNAME % (self.zoning_system, 'raw_nhb')
        attractions.to_csv(os.path.join(out_path, fname), index=False)
        nhb_att.to_csv(os.path.join(out_path, nhb_fname), index=False)

        # TODO: functionalise conversion to old efs
        # ## CONVERT TO OLD EFS FORMAT ## #
        # Make sure columns are the correct data type
        attractions[p_col] = attractions[p_col].astype(int)
        attractions[m_col] = attractions[m_col].astype(int)
        attractions[soc_col] = attractions[soc_col].astype(int)
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

        # Reindex to just the wanted return cols
        group_cols = self.return_segments
        index_cols = group_cols.copy() + all_years

        attractions = attractions.reindex(index_cols, axis='columns')
        attractions = attractions.groupby(group_cols).sum().reset_index()
        nhb_att = nhb_att.reindex(index_cols, axis='columns')
        nhb_att = nhb_att.groupby(group_cols).sum().reset_index()

        # Output the final attractions
        fname = consts.ATTRS_FNAME % (self.zoning_system, 'hb')
        nhb_fname = consts.ATTRS_FNAME % (self.zoning_system, 'nhb')
        attractions.to_csv(os.path.join(out_path, fname), index=False)
        nhb_att.to_csv(os.path.join(out_path, nhb_fname), index=False)

        return attractions, nhb_att


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


def add_soc0(df: pd.DataFrame,
             data_cols: List[str],
             p_col: str = 'p',
             soc_col: str = 'soc'
             ) -> pd.DataFrame:
    """
    Removes soc splits for the purposes that don't use them, and replaces with
    soc0

    Parameters
    ----------
    df:
        The dataframe to add soc0 into

    data_cols:
        A list of the column names that contain data values. I.e. not
        segmentation variables

    p_col:
        Name of the column in df that contains purpose data.

    soc_col:
        Name of the column in df that contains soc data.

    Returns
    -------
    soc_df:
        The given df with soc0 added in where needed
    """
    # Init
    df = df.copy()
    index_cols = list(df)

    # Figure out which rows need combining to soc0
    mask = (df[p_col].isin(consts.SOC_P))
    retain_df = df[mask].copy()
    combine_df = df[~mask].copy()

    # Return early if we can
    if combine_df.empty:
        return retain_df

    # Remove the soc segmentation
    index_cols.remove(soc_col)
    group_cols = du.list_safe_remove(index_cols.copy(), data_cols)

    combine_df = combine_df.reindex(columns=index_cols)
    combine_df = combine_df.groupby(group_cols).sum().reset_index()

    # Re-add in soc col, all set to 0
    combine_df[soc_col] = 0

    # Finally, stick the two back together
    return pd.concat([retain_df, combine_df])


def split_by_soc(df: pd.DataFrame,
                 soc_weights: pd.DataFrame,
                 zone_col: str = 'msoa_zone_id',
                 p_col: str = 'p',
                 unique_col: str = 'trips',
                 soc_col: str = 'soc',
                 split_cols: str = None
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

    split_cols:
        Which columns are being split by soc. If left as None, only zone_col
        is used.

    Returns
    -------
    soc_split_df:
        df with an added soc_col. Unique_col will be split by the weights
        given
    """
    # Init
    soc_cats = list(soc_weights.columns)
    split_cols = [zone_col] if split_cols is None else split_cols

    # Figure out which rows need splitting
    if p_col in df:
        mask = (df[p_col].isin(consts.SOC_P))
        split_df = df[mask].copy()
        retain_df = df[~mask].copy()
        id_cols = split_cols + [p_col]
    else:
        # Split on all data
        split_df = df.copy()
        retain_df = None
        id_cols = split_cols

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

    Returns
    -------
    hb_attractions:
        A wide dataframe containing the attraction numbers. The index will
        match the index from employment, the columns will be the purposes
        given in p_col of attractions_weight_path.

    nhb_attractions:
        A wide dataframe containing the attraction numbers. The index will
        match the index from employment, the columns will be the purposes
        given in p_col of attractions_weight_path.
    """
    # Init
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

    # Sort out soc categories as needed
    soc_col = 'soc'
    if soc_col in list(attractions):
        # Remove soc splits in p3-8
        attractions = add_soc0(
            attractions,
            data_cols=[unique_col],
            p_col=p_col,
            soc_col=soc_col
        )

    elif soc_weights is not None:
        # Add in soc if needed
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
                         soc_split: bool = True,
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

    audit_dir:
        Path to the directory to write NTEM control audits out to during
        attraction generation.

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

    soc_split:
        Whether to apply the soc splits from soc_weights_path to the
        attractions or not.

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
                             mode_splits_path: str = None,
                             soc_weights_path: str = None,
                             msoa_lookup_path: str = None,
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

    mode_splits_path:
        An alternate mode splits import path to use. File will need to follow
        the same format as default file.

    soc_weights_path:
        An alternate soc weights import path to use. File will need to follow
        the same format as default file.

    msoa_lookup_path:
        An alternate msoa lookup import path to use. File will need to follow
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
    # Set all unset paths
    if mode_splits_path is None:
        path = 'attractions/attraction_mode_split.csv'
        mode_splits_path = os.path.join(import_home, path)

    if soc_weights_path is None:
        path = 'attractions/soc_2_digit_sic_%s.csv' % base_year
        soc_weights_path = os.path.join(import_home, path)

    if msoa_lookup_path is None:
        path = "zone_translation\msoa_zones.csv"
        msoa_lookup_path = os.path.join(import_home, path)

    if set_controls and ntem_control_dir is None:
        path = 'ntem_constraints'
        ntem_control_dir = os.path.join(import_home, path)

    if set_controls and lad_lookup_dir is None:
        path = os.path.join('zone_translation', 'no_overlap')
        lad_lookup_dir = os.path.join(import_home, path)

    # Assign to dict
    imports = {
        'weights': attraction_weights_path,
        'mode_splits': mode_splits_path,
        'soc_weights': soc_weights_path,
        'msoa_lookup': msoa_lookup_path,
        'ntem_control': ntem_control_dir,
        'lad_lookup': lad_lookup_dir,
    }

    # Make sure all import paths exit
    for key, path in imports.items():
        # BACKLOG: Fix cross model inputs
        #  labels: demand merge
        if key == 'weights' and path is None:
            continue

        if not os.path.exists(path):
            raise FileNotFoundError(
                "Attraction Model Imports: The path for %s does not "
                "exist.\nFull path: %s" % (key, path)
            )

    return imports


def build_attraction_exports(export_home: str,
                             audit_write_dir: str = None
                             ) -> Dict[str, str]:
    """
    Builds a dictionary of attraction export paths, forming a standard calling
    procedure for attraction efs_exports. Arguments allow default paths to be
    replaced.


    Parameters
    ----------
    export_home:
        Usually the export home for this run of the EFS. Can be automatically
        generated using du.build_io_paths()

    audit_write_dir:
        An alternate export path for the audits. By default this will be:
        audits/productions/

    Returns
    -------
    export_dict:
        A dictionary of paths with the following keys:
        'audits'

    """
    # Set all unset export paths to default values
    if audit_write_dir is None:
        audit_write_dir = os.path.join(export_home,
                                       consts.AUDITS_DIRNAME,
                                       'Attractions')
    du.create_folder(audit_write_dir, chDir=False)

    # Build the efs_exports dictionary
    exports = {
        'audits': audit_write_dir
    }

    # Make sure all export paths exit
    for key, path in exports.items():
        if not os.path.exists(path):
            raise IOError(
                "Attraction Model Exports: The path for %s does not "
                "exist.\nFull path: %s" % (key, path)
            )

    return exports


def get_emp_data_from_land_use(by_emp_import_path: nd.PathLike,
                               base_year: str,
                               fy_emp_import_dir: nd.PathLike = None,
                               future_years: List[str] = None,
                               segmentation_cols: List[str] = None,
                               lu_zone_col: str = 'msoa_zone_id',
                               base_year_data_col: str = '2018',
                               dtype: Dict[str, np.dtype] = None,
                               soc_col: str = 'soc',
                               ignore_missing_soc: bool = True,
                               ) -> pd.DataFrame:
    """
    Reads in land use outputs and aggregates up to segmentation_cols.

    Combines all the dataframe from each into a single dataframe.

    Parameters
    ----------
    by_emp_import_path:
        Path to the file containing base year population data.

    base_year:
        The base year. The year the data in by_pop_import_path was created
        for.

    fy_emp_import_dir:
        Path to the land use directory containing population data for future years

    future_years:
        The future years of population data to read in.

    segmentation_cols:
        The columns to keep in the land use data. If None, defaults to:
         [
            'area_type',
            'traveller_type',
            'soc',
            'ns',
        ]

    lu_zone_col:
        The name of the column in the land use data that refers to the zones.

    base_year_data_col:
        The name of the column in by_pop_import_path that contains the
        base year population figures.

    dtype:
        The data types to assign to columns in the read in data. Follows the
        same format as dtypes argument in pd.read_csv()

    soc_col:
        The name of the column containing soc data. If this column doesn't
        exist, this argument can be safely ignored

    ignore_missing_soc:
        If the given segmentation_cols contain soc_col and this is set to
        True an error will not be thrown if no soc col exists in the data.
        Instead soc_col will be removed from segmentation_cols

    Returns
    -------
    population:
        A dataframe of population data for all years segmented by
        segmentation_cols. Will also include lu_zone_col and year cols
        from land use.
    """
    # Init
    future_years = list() if future_years is None else future_years
    all_years = [base_year] + future_years

    if dtype is None:
        dtype = {'soc': int, 'ns': int}

    if segmentation_cols is None:
        segmentation_cols = ['employment_cat']
    group_cols = [lu_zone_col] + segmentation_cols

    # We can use the future years to determine if we should keep soc in the
    # base year. Do the years backwards!
    all_emp_ph = list()
    for year in reversed(all_years):

        # Read in the dataframe - different if base year
        if year == base_year:
            year_emp = pd.read_csv(by_emp_import_path, dtype=dtype)
            year_emp = year_emp.rename(columns={base_year_data_col: base_year})

        else:
            # Build the path to this years data
            fname = consts.LU_EMP_FNAME % str(year)
            lu_path = os.path.join(fy_emp_import_dir, fname)
            year_emp = pd.read_csv(lu_path, dtype=dtype)

        # ## CHECK IF WE SHOULD IGNORE SOC ## #
        if soc_col in segmentation_cols and soc_col not in list(year_emp):
            # We'll catch the error lower down, so don't need to here
            if ignore_missing_soc:
                segmentation_cols.remove(soc_col)
                group_cols.remove(soc_col)

        # ## FILTER TO JUST THE DATA WE NEED ## #
        # Set up the columns to keep
        index_cols = group_cols.copy() + [year]

        # Check all columns exist
        year_emp_cols = list(year_emp)
        for col in index_cols:
            if col not in year_emp_cols:
                raise nd.NormitsDemandError(
                    "Tried to read in population data from NorMITs Land Use "
                    "for year %s. Cannot find all the needed columns in the "
                    "data. Specifically, column %s does not exist."
                    % (year, col)
                )

        # Filter down
        year_emp = year_emp.reindex(columns=index_cols)
        year_emp = year_emp.groupby(group_cols).sum().reset_index()

        all_emp_ph.append(year_emp)

    # Can't merge if there is only one dataframe!
    if len(all_emp_ph) == 1:
        return all_emp_ph[0]

    return du.merge_df_list(all_emp_ph, on=group_cols)
