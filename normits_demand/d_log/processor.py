
import os

from typing import List

import pandas as pd

from tqdm import tqdm

# Local imports
from normits_demand.utils import general as du


def identify_exceptional_zones(pre_df: pd.DataFrame,
                               post_df: pd.DataFrame,
                               growth_cutoff: float,
                               base_year: str,
                               future_years: List[str],
                               absolute_cutoff: float = None,
                               cutoff_method: str = "growth",
                               zone_col: str = 'msoa_zone_id'
                               ) -> pd.DataFrame:
    """Returns a dataframe of the MSOAs identified to have exceptional growth.
    The criteria can be based on the difference in growth between the pre and 
    post D-Log forecast data, an absolute difference, or both. The default uses
    the difference in growth.

    Parameters
    ----------
    pre_df : pd.DataFrame
        Forecast population / employment dataframe before application of the
        D-Log
    post_df : pd.DataFrame
        Forecast population / employment dataframe after application of the 
        D-Log
    growth_cutoff : float
        The growth difference to identify exceptional zones. E.g. 0.75 will
        flag any zones that have 75% difference in percentage points -> 
        101% growth pre D-Log and 176% growth post D-Log would be flagged.
    base_year : str
        The Base year of the forecast
    future_years : List[str]
        All forecast years
    absolute_cutoff : float, optional
        Absolute difference if using the "absolute" cutoff_method,
        by default None
    cutoff_method : str, optional
        Can be "growth", "absolute", or "both", by default "growth"
    zone_col : str, optional
        Column in pre/post df containing the zones, by default 'msoa_zone_id'

    Returns
    -------
    pd.DataFrame
        Dataframe containing all flagged zones.

    Raises
    ------
    AttributeError
        Raised if pre/post dataframes are not comparable.
    ValueError
        Raised on invalid cutoff method.
    """
    
    # Return a dataframe of the highest growth zones and their associated 
    # growth

    # Check that both dataframes have the same index
    if not pre_df.index.identical(post_df.index):
        raise AttributeError("Pre and Post dataframe index do not align")

    # Empty dataframe to store which zones meet the cutoff
    growth_diff = pd.DataFrame(columns=future_years, index=pre_df.index)
    absolute_cols = []
    # Loop over each forecast year and calculate the difference in growth 
    # factor between the pre adjustment and post adjustment dataframes - store 
    # as a column in growth_diff.
    # If this value exceeds the growth criteria, flag the relevant zone.
    # (Also do the same for the growth in absolute values if required)
    for year in future_years:
        # Calculate growth difference pre -> post
        growth_diff[year] = (
            (post_df[year] / post_df[base_year])
            - (pre_df[year] / pre_df[base_year])
        )
        # Change to a boolean series
        growth_diff[year] = growth_diff[year] > growth_cutoff
        
        # Calculate absolute differences pre -> post
        abs_diff_col = f"{year}_abs"
        absolute_cols.append(abs_diff_col)
        growth_diff[abs_diff_col] = (
            (post_df[year] - post_df[base_year])
            - (pre_df[year] - pre_df[base_year])
        )
        # Change to a boolean series
        if absolute_cutoff is not None:
            growth_diff[abs_diff_col] = (
                growth_diff[abs_diff_col] > absolute_cutoff
            )

    # If the criteria is met in any year, extract this zone as "exceptional"
    if cutoff_method == "growth":
        e_zones = growth_diff.loc[growth_diff[future_years].any(axis=1)].index
    elif cutoff_method == "absolute":
        e_zones = growth_diff.loc[growth_diff[absolute_cols].any(axis=1)].index
    elif cutoff_method == "both":
        e_zones = growth_diff.loc[growth_diff.any(axis=1)].index
    else:
        raise ValueError("Cutoff method must be 'growth', 'absolute', "
                         "or 'both'")

    return pd.DataFrame(e_zones, columns=[zone_col])


def set_datetime_types(df: pd.DataFrame,
                       cols: List[str],
                       **kwargs
                       ) -> pd.DataFrame:
    """Set all columns in cols as datetime types.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing columns given in cols
    cols : List[str]
        Cols that need to be converted.

    Returns
    -------
    pd.DataFrame
        Dataframe df, with all cols converted to datetime
    """
    for col in cols:
        df[col] = pd.to_datetime(
            df[col],
            **kwargs
        )
    return df


def constrain_post_dlog(df: pd.DataFrame,
                        constraint: pd.DataFrame,
                        la_equivalence: pd.DataFrame,
                        base_year: str,
                        year: str,
                        msoa_column: str,
                        segment_cols: List[str],
                        grouping_column: str = "grouping_id"
                        ):
    """Perform a constraint on the post d-log data. Constraint is on the 
    total growth at sector level.

    Parameters
    ----------
    df : pd.DataFrame
        The post d-log dataframe. Containing data for the base year and 
        future year
    constraint : pd.DataFrame
        Constraint data. Should contain absolute values for the base year and
        future year. Zoning system should be the same as df (MSOA)
    la_equivalence : pd.DataFrame
        Equivalence lookup table of MSOA to sector (LA). Should contain 
        msoa_zone_id and grouping_id
    base_year : str
        The base year column containing pop / emp
    year : str
        The future year column containing pop / emp
    msoa_column : str
        MSOA zone id column (msoa_zone_id)

    Returns
    -------
    pd.DataFrame
        The constrained dataframe
    """

    grouping_columns = [grouping_column] + segment_cols

    # Calculate pre-constraint sector growth
    df["growth"] = (
        df[year]
        / df[base_year]
    )
    df["sector_growth"] = (
        df.groupby(grouping_columns)[year].transform("sum")
        /
        df.groupby(grouping_columns)[base_year].transform("sum")
    )
    # Calculate the constraint growth
    pop_constraint = constraint.copy()
    pop_constraint = pop_constraint.merge(
        la_equivalence,
        on=msoa_column,
        how="left"
    )[[msoa_column] + grouping_columns + [base_year, year]]

    # Keep grouping_id as index for mapping later
    pop_constraint = pop_constraint.groupby(
        grouping_columns,
        as_index=False
    )[[base_year, year]].sum()

    pop_constraint["constraint"] = (
        pop_constraint[year]
        / pop_constraint[base_year]
    )
    pop_constraint = pop_constraint[grouping_columns + ["constraint"]]
    # Merge to the df data
    df = pd.merge(
        df,
        pop_constraint,
        on=grouping_columns,
        how="left"
    )
    # Adjust segment growth factors
    df["growth"] *= (
        df["constraint"]
        / df["sector_growth"]
    )
    # Adjust forecast year by new growth
    df[year] = (
        df[base_year]
        * df["growth"]
    )
    # Handle NaN values that are reintroduced when calculating growth
    df[year].fillna(0.0, inplace=True)

    return df


# BACKLOG: Move constraining function
#  labels: demand merge
def constrain_forecast(pre_constraint_df: pd.DataFrame,
                       constraint_df: pd.DataFrame,
                       constraint_zone_equivalence: pd.DataFrame,
                       base_year: str,
                       future_years: List[str],
                       zone_column: str,
                       segment_cols: List[str] = None
                       ) -> pd.DataFrame:
    """Constrains a population / employment dataframe by growth values.
    Constrain can be done using a sector system given by 
    constraint_zone_equivalence
    Note that if the base absolute values are not the same as the constraint
    values, the constrained values will not match the NTEM totals.

    Parameters
    ----------
    pre_constraint_df : pd.DataFrame
        The dataframe to be constrained. Should contain all columns: base_year,
        future_years, zone_column, and segment_cols.
    constraint_df : pd.DataFrame
        Contains absolute values for each of the future_years and base_year. 
        Should be at the same segmentation as pre_constraint_df.
    constraint_zone_equivalence : pd.DataFrame
        The zone - sector equivalence. Must contain model_zone_id and 
        grouping_id columns.
    base_year : str
        The base year to calculate growth from.
    future_years : List[str]
        The forecast years.
    zone_column : str
        Name of the column containing zone IDs
    segment_cols : List[str], optional
        The segmentation columns contained in the data. E.g. [purpose, soc, ns,
        ca] or [purpose, soc] generally for population and employment, 
        by default None

    Returns
    -------
    pd.DataFrame
        The constrained dataframe. Growth from the base year will have been
        constrained at the given sector level.
    """

    df = pre_constraint_df.copy()
    constraint = constraint_df.copy()
    constraint_seg = segment_cols or []

    sector_equivalence = constraint_zone_equivalence.copy()
    sector_equivalence.rename(
        {"model_zone_id": zone_column},
        axis=1,
        inplace=True
    )
    sector_equivalence = sector_equivalence.set_index(
        zone_column
    )["grouping_id"]
    df["grouping_id"] = df[zone_column].map(sector_equivalence)

    for year in tqdm(future_years, desc="Constraining by year"):
        year_df = df.drop(
            [col for col in future_years if col != year],
            axis=1
        )
        year_constrained = constrain_post_dlog(
            year_df,
            constraint,
            sector_equivalence.reset_index(),
            base_year,
            year,
            zone_column,
            constraint_seg,
            grouping_column="grouping_id"
        )

        df[year] = year_constrained[year]

    # Drop the temporary grouping id column
    df.drop("grouping_id", axis=1, inplace=True)

    return df


def estimate_dlog_build_out(dlog: pd.DataFrame,
                            year: str,
                            start_date_col: str,
                            end_date_col: str,
                            data_key: str,
                            ) -> pd.DataFrame:
    """Uses the start and end dates within the D-Log to estimate the 
    additional units provided in the given year. Uses a linear build out 
    profile.

    Parameters
    ----------
    dlog : pd.DataFrame
        Development Log data, either residential or non-residential
    year : str
        The year that the number of units will be estimated for.
    start_date_col : str
        Column providing the development start date.
    end_date_col : str
        Column providing the development end date.
    data_key : str
        String that identifies the units of data in the D-Log. Likely either
        population or employment.

    Returns
    -------
    pd.DataFrame
        The dlog dataframe with an additional column called year (variable)
        that contains the estimated population / employment.
    """
    
    parsed_dlog = dlog.copy()
    
    # Get the total development impact (population or employees)
    data_columns = [col for col in parsed_dlog.columns if data_key in col]
    parsed_dlog["total"] = parsed_dlog[data_columns].sum(axis=1)
    
    # Add a column to represent the forecast date
    parsed_dlog["future_year"] = pd.to_datetime(
        f"{year}-01-01",
        format="%Y-%m-%d"
    )
    
    # Handle forecast year inside of development dates
    parsed_dlog[year] = (
        (parsed_dlog["future_year"] - parsed_dlog[start_date_col]).dt.days
        / (parsed_dlog[end_date_col] - parsed_dlog[start_date_col]).dt.days
        * parsed_dlog["total"]
    )
    
    # Handle forecast year before development dates
    mask = parsed_dlog["future_year"] <= parsed_dlog[start_date_col]
    parsed_dlog.loc[mask, year] = 0.0
    
    # Handle forecast year after development dates
    mask = parsed_dlog[end_date_col] <= parsed_dlog["future_year"]
    parsed_dlog.loc[mask, year] = parsed_dlog.loc[mask, "total"]
    
    # Check for errors
    errors = parsed_dlog.loc[parsed_dlog.isna().any(axis=1)]
    
    return parsed_dlog, errors
    

def apply_d_log(pre_dlog_df: pd.DataFrame,
                base_year:  str,
                future_years: List[str],
                dlog_path: str,
                msoa_conversion_path: str,
                constraints_zone_equivalence: pd.DataFrame,
                segment_cols: List[str],
                segment_groups: List[str] = None,
                dlog_conversion_factor: float = 1.0,
                msoa_column: str = "msoa_zone_id",
                emp_cat_col: str = 'employment_cat',
                min_growth_limit: float = 0.25,
                dlog_data_column_key: str = "population",
                perform_constraint: bool = True,
                constraints: pd.DataFrame = None,
                exceptional_zone_cutoff: float = 0.75,
                exceptional_zone_method: str = "growth",
                audit_outputs: bool = False,
                audit_location: str = None
                ) -> pd.DataFrame:
    """Alters the forecast population or employment vectors in the EFS by
    distributing data from the development log (residential or non-residential)
    across the required segmentation.
    
    Segment splits are calculated at sector level for the required EFS
    segmentation.
    
    Development log data is estimated for each year using 
    estimate_dlog_build_out().
    
    D-Log data is added to the existing data for each segmentation and new
    adjusted growth factors are calculated.
    

    Parameters
    ----------
    pre_dlog_df : pd.DataFrame
        Forecast population or employment as produced in 
        "efs_production_generator" or "efs_attraction_generator". Should 
        contain columns representing the base and future years, msoa zones.
    base_year : str
        The base year provided by the EFS
    future_years : List[str]
        The forecast years provided by the EFS
    dlog_path : str
        The path to the relevant D-Log data
    msoa_conversion_path : str
        Conversion data from MSOA zone IDs to Codes
    constraints : pd.DataFrame, optional
        The constraint data if being applied, by default None
    constraints_zone_equivalence : pd.DataFrame
        Dataframe containing a zone to sector lookup that is used when 
        distributing the D-Log data across segments
    segment_cols : List[str]
        All segmentation columns present in the input dataframe.
    segment_groups : List[str], optional
        Segment columns to use when calculating the segment splits for 
        distributing the D-Log data. For population, this will not include
        area_type or traveller_type, by default None
    dlog_conversion_factor : float, optional
        Conversion factor to use for the D-Log data to convert to 
        population or employment. Should be 1.0 if already in the required 
        units, by default 1.0
    msoa_column : str, optional
        Column containing MSOA zone ids, by default "msoa_zone_id"
    emp_cat_col : str, optional
        Name of the employment category column if present, by default 
        'employment_cat'
    min_growth_limit : float, optional
        Minimum factor that the development log data can change the input data 
        by. This stops large reductions in population/employment if this is
        what the D-Log suggests, by default 0.25
    dlog_data_column_key : str, optional
        String that identifies the columns in the D-Log to use as population 
        / employment, by default "population"
    perform_constraint : bool, optional
        Flag is the built in constraint should be used. Now replaced by 
        constraint in production/attraction generators, by default True
    exceptional_zone_cutoff : float, optional
        The growth difference to identify exceptional zones. E.g. 0.75 will
        flag any zones that have 75% difference in percentage points -> 
        101% growth pre D-Log and 176% growth post D-Log would be flagged,
        by default 0.75
    exceptional_zone_method : str, optional
        Can be "growth", "absolute", or "both", by default "growth"
    audit_outputs : bool, optional
        If audit outputs should be provided showing the splits used, and the 
        additions from the D-Log, by default False
    audit_location : str, optional
        Location to save the audits, by default None

    Returns
    -------
    pd.DataFrame
        The adjusted input dataframes, with all future years adjusted to 
        reflect the D-Log

    Raises
    ------
    ValueError
        If columns are missing from the DataFrame
    """

    # TODO: remove the conversions to and from msoa id numbers in apply_d_log()

    # Use segment columns if no value provided for segment groups
    segment_groups = segment_groups or segment_cols

    post_dlog_df = pre_dlog_df.copy()[
        [msoa_column, base_year] + segment_cols + future_years
    ]
    if msoa_conversion_path is not None:
        post_dlog_df = du.convert_msoa_naming(
            post_dlog_df,
            msoa_col_name=msoa_column,
            msoa_path=msoa_conversion_path,
            to='int'
        )
    # If we are applying to the employment data - remove the totals category
    # (will be added back in afterwards)
    re_add_all_commute_cat = False
    if emp_cat_col in segment_cols:
        # Check if the all_commute_Cat exists
        if 'E01' in post_dlog_df[emp_cat_col].unique():
            post_dlog_df = du.remove_all_commute_cat(post_dlog_df, emp_cat_col)
            re_add_all_commute_cat = True

    # Save the initial growth by MSOA to use in identifying exceptional zones
    pre_dlog_growth = post_dlog_df.groupby(msoa_column)[
        [base_year] + future_years
    ].sum()
    post_dlog_growth = pre_dlog_growth.copy()

    # Calculate the base year segment shares over each sector
    print("Calculating Segment Share by Sector")
    sector_equivalence = constraints_zone_equivalence.copy()
    sector_equivalence.rename(
        {"model_zone_id": msoa_column},
        axis=1,
        inplace=True
    )
    # Use map instead of merge as MSOA to sector will be 1 to 1 and likely
    # faster than merge
    post_dlog_df["sector_id"] = post_dlog_df[msoa_column].map(
        sector_equivalence.set_index(msoa_column)["grouping_id"]
    )
    # Calculate the split for each sector
    segment_split = post_dlog_df.groupby(
        [msoa_column, "sector_id"] + segment_groups,
        as_index=False
    )[[base_year] + future_years].sum()
    segment_split["seg_s"] = (
        segment_split.groupby(
            ["sector_id"] + segment_groups
        )[base_year].transform("sum")
        / segment_split.groupby("sector_id")[base_year].transform("sum")
    )

    # # Read in development log data
    # print(du.safe_read_csv(dlog_path))
    # print(msoa_conversion_path)
    #
    # dlog_data = du.convert_msoa_naming(
    #     du.safe_read_csv(dlog_path),
    #     msoa_col_name=msoa_column,
    #     msoa_path=msoa_conversion_path,
    #     to='str'
    # )
    dlog_data = du.safe_read_csv(dlog_path)

    # Define column names
    start_date = "start_date"
    end_date = "expected_completion_date"
    # Set the start and end as datetime type
    # Ignore errors for now - these will be caught later
    dlog_data = set_datetime_types(
        dlog_data,
        [start_date, end_date],
        format="%Y-%m-%d",
        errors="coerce"
    )

    # Check that the dataframe contains the correct columns
    pop_required_cols = ["msoa_zone_id", base_year] + future_years
    if not all(col in pre_dlog_df.columns for col in pop_required_cols):
        raise ValueError("Dataframe does not contain the required"
                         " columns")

    # Save dlog_processor rows where data is missing
    dlog_missing_data = pd.DataFrame()
    # Identify zones where estimated growth was invalid
    invalid_growth_zones = pd.DataFrame()

    # Calculate the growth factors for each MSOA and year
    metric_columns = [base_year] + future_years
    dlog_additions = post_dlog_df.groupby(msoa_column)[metric_columns].sum()

    # Calculate a new column with the estimated new households for each year
    for year in future_years:

        print(f"Replacing D-LOG data for {year}")

        dlog_subset = dlog_data.copy()
        dlog_subset, dlog_missing_data = estimate_dlog_build_out(
            dlog=dlog_subset,
            year=year,
            start_date_col=start_date,
            end_date_col=end_date,
            data_key=dlog_data_column_key
        )
        print(f"D-LOG data contains {dlog_subset[year].sum()} units")

        # Convert to population / employment
        dlog_subset[year] *= dlog_conversion_factor

        dlog_subset = dlog_subset.groupby(
            msoa_column,
            as_index=False
        )[year].sum()

        # Calculate the new growth as base year absolute + data from dlog_processor
        dlog_data_col = f"dlog_add_{year}"
        dlog_subset.rename({year: dlog_data_col}, axis=1, inplace=True)
        dlog_additions = dlog_additions.merge(
            dlog_subset,
            on=msoa_column,
            how="left"
        )
        dlog_additions[dlog_data_col].fillna(0.0, inplace=True)

        # Join to dataframe and replace old values
        
        print(f"Previous {year} total = {post_dlog_df[year].sum()}")
        # Calculate the new growth factors by segment share
        post_dlog = f"dlog_{year}"
        adj_growth = f"adj_growth_{year}"
        segment_split[post_dlog] = (
            segment_split[year]
            + (
                segment_split["seg_s"]
                * segment_split[msoa_column].map(
                    dlog_additions.set_index(msoa_column)[dlog_data_col]
                )
            )
        )
        # Set default to zero
        segment_split[adj_growth] = (
            segment_split[post_dlog] 
            / segment_split[base_year]
        )
        segment_split.loc[segment_split[base_year] == 0, adj_growth] = 0
        
        # Handle cases where addition of the dlog_processor data (negative) leaves
        # a zone with negative or very low population / employment
        # Use the minimimum growth limit from the function arguments as the
        # minimum possible value for the adjusted growth

        # Flag zones where the adjusted growth is invalid
        invalid_growth = segment_split[adj_growth] < min_growth_limit
        audit_zones = segment_split[invalid_growth][
            [msoa_column] + segment_groups + [adj_growth]
        ].copy()
        # Remove zones where the original growth was zero
        audit_zones = audit_zones.loc[audit_zones[adj_growth] != 0]
        audit_zones["year"] = year
        audit_zones.rename({adj_growth: "growth_factor"}, axis=1, inplace=True)
        # Reset the invalid adjusted growth to the minimum growth limit
        segment_split.loc[invalid_growth, adj_growth] = min_growth_limit
        # Save the invalid zones to output in audits if required
        if invalid_growth_zones.empty:
            invalid_growth_zones = audit_zones.drop_duplicates()
        else:
            invalid_growth_zones = invalid_growth_zones.append(
                audit_zones.drop_duplicates()
            )
        
        # Update original data with the new calculated segment growth factors
        merge_cols = [msoa_column] + segment_groups
        post_dlog_df = pd.merge(
            post_dlog_df,
            segment_split[merge_cols + [adj_growth]],
            on=merge_cols,
            how="left"
        )
        post_dlog_df[year] = (
            post_dlog_df[base_year]
            * post_dlog_df[adj_growth]
        )

        print(f"New {year} total = {post_dlog_df[year].sum()}")

        # Save the pre-constraint values to identify exceptional zones
        post_dlog_growth[year] = post_dlog_df.groupby(msoa_column)[year].sum()

        if perform_constraint:
            post_dlog_df = constrain_post_dlog(
                df=post_dlog_df,
                constraint=constraints,
                la_equivalence=sector_equivalence,
                base_year=base_year,
                year=year,
                msoa_column=msoa_column,
                segment_cols=segment_cols
            )
            print(f"Post constraint total = {post_dlog_df[year].sum()}")

            # Drop any temporary columns
            post_dlog_df.drop(
                ["growth", "constraint", "sector_growth"],
                axis=1,
                inplace=True
            )

    # Use pre and post dlog_processor growth to identify exceptional zones
    # Default growth cutoff here is 75%
    e_zones = identify_exceptional_zones(
        pre_df=pre_dlog_growth,
        post_df=post_dlog_growth,
        growth_cutoff=exceptional_zone_cutoff,
        cutoff_method=exceptional_zone_method,
        base_year=base_year,
        future_years=future_years
    )

    # Drop any temporary columns
    drop_cols = ["sector_id"]
    adjusted_growth_cols = [
        col for col in post_dlog_df.columns if "adj_growth" in col
    ]
    drop_cols.extend(adjusted_growth_cols)
    post_dlog_df.drop(
        drop_cols,
        axis=1,
        inplace=True
    )
    # If we are applying to the employment data - add the totals category
    # back in
    if re_add_all_commute_cat:
        post_dlog_df = du.add_all_commute_cat(
            df=post_dlog_df,
            emp_cat_col=emp_cat_col,
            unique_data_cols=[base_year] + future_years,
        )

    # Convert back to string - as expected by the next steps
    if msoa_conversion_path is not None:
        post_dlog_df = du.convert_msoa_naming(
            post_dlog_df,
            msoa_col_name=msoa_column,
            msoa_path=msoa_conversion_path,
            to="string"
        )
        e_zones = du.convert_msoa_naming(
            e_zones,
            msoa_col_name=msoa_column,
            msoa_path=msoa_conversion_path,
            to="string"
        )

    if audit_outputs:
        # Save outputs for sense checks
        dlog_additions.to_csv(
            os.path.join(audit_location, "dlog_extra.csv"),
            index=False
        )
        post_dlog_df.to_csv(
            os.path.join(audit_location, "post_dlog_data.csv"),
            index=False
        )
        dlog_missing_data.to_csv(
            os.path.join(audit_location, "dlog_errors.csv"),
            index=False
        )
        segment_split.to_csv(
            os.path.join(audit_location, "segment_splits.csv"),
            index=False
        )
        invalid_growth_zones.to_csv(
            os.path.join(audit_location, "invalid_growth_zones.csv"),
            index=False
        )

    return post_dlog_df, e_zones

