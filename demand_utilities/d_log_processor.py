
import os

from typing import List, Tuple

import pandas as pd

from demand_utilities import utils as du

def identify_exceptional_zones(pre_df: pd.DataFrame,
                               post_df: pd.DataFrame,
                               growth_cutoff: float,
                               base_year: str,
                               future_years: List[str],
                               absolute_cutoff: float = None,
                               cutoff_method: str = "growth"
                               ) -> pd.DataFrame:
    
    # Return a dataframe of the highest growth zones and their associated 
    # growth

    # Check that both dataframes have the same index
    if not pre_df.index.identical(post_df.index):
        raise AttributeError("Pre and Post dataframe index do not align")

    # Empty dataframe to store which zones meat the cutoff
    growth_diff = pd.DataFrame(columns=future_years, index=pre_df.index)
    absolute_cols = []
    for year in future_years:
        # Calculate growth difference pre -> post
        growth_diff[year] = (
            (post_df[year] / post_df[base_year])
            - (pre_df[year] / pre_df[base_year])
        )
        print(growth_diff)
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

    if cutoff_method == "growth":
        e_zones = growth_diff.loc[growth_diff[future_years].any(axis=1)].index
    elif cutoff_method == "absolute":
        e_zones = growth_diff.loc[growth_diff[abs_diff_col].any(axis=1)].index
    elif cutoff_method == "both":
        e_zones = growth_diff.loc[growth_diff.any(axis=1)].index
    else:
        raise ValueError("Cutoff method must be 'growth', 'absolute', "
                         "or 'both'")

    return pd.Series(e_zones)


def set_datetime_types(df: pd.DataFrame,
                       cols: List[str],
                       **kwargs
                       ) -> pd.DataFrame:
    for col in cols:
        df[col] = pd.to_datetime(
            df[col],
            **kwargs
        )
    return df


def parse_dlog_build_out(dlog: pd.DataFrame,
                         year: str,
                         dlog_columns: List[str]
                         ) -> pd.DataFrame:
    """Parses the data in the build out columns of the development log.
    TODO Currently assumes that the first year in the d-log is the base
    Does contain some repitition if this is done for all years but only small
    tasks.

    Parameters
    ----------
    dlog : pd.DataFrame
        The subset of the dlog (excess columns removed)
    year : str
        Future year that the build-out information will be calculated to
    dlog_columns : List[str]
        All required columns to be present in the returned dataframe

    Returns
    -------
    pd.DataFrame
        The same dataframe as dlog, with only the columns in 
        dlog_columns + a column with the name given in year that contains
        the estimated development
    """

    dlog_subset = dlog.copy()

    # Extract the build out columns
    build_out_cols = [
        col for col in dlog_subset.columns
        if "units_" in col and any([c for c in col if c.isdigit()])
    ]
    # Extract the first year in the range
    build_out_years = [int(x.replace("units_", "").split("_")[0])
                       for x in build_out_cols]
    # Ensure these are sorted
    build_out_cols = [
        col for col, year in sorted(zip(build_out_cols, build_out_years),
                                    key=lambda pair: pair[1])
    ]
    build_out_years = [year for year in sorted(build_out_years)]

    # Get the position in the build out profile
    for i, band_start in enumerate(build_out_years[:-1]):
        next_band_start = build_out_years[i+1]
        future_year = int(year)
        if future_year > band_start and future_year <= next_band_start:

            band_width = next_band_start - band_start
            build_out_factor = (future_year - band_start) / band_width

            build_out_idx = i
            break

    # Calculate the number of additional households from the base year
    final_year_col = build_out_cols[build_out_idx]
    required_year_cols = build_out_cols[:build_out_idx+1]
    subset_columns = dlog_columns + required_year_cols
    dlog_subset = dlog_subset[subset_columns]

    # Any missing build out can be infilled as 0
    dlog_subset.fillna(
        {col: 0.0 for col in required_year_cols},
        inplace=True
    )

    # Assume a linear build out for the final band using previously
    # calculated factor
    dlog_subset[final_year_col] *= build_out_factor
    dlog_subset[year] = dlog_subset[required_year_cols].sum(axis=1)

    return dlog_subset


def infill_dlog_build_out(dlog: pd.DataFrame,
                          year: str,
                          start_date_col: str,
                          end_date_col: str,
                          development_id_col: str,
                          total_column: str
                          ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fill in gaps where no build out data was provided in the development
    log. This is done using the entire development phase total and the start /
    end dates. A linear build out is assumed.
    TODO This does not return anything currently as more than half the 
    developments are missing some of this data.
    Fix the date check - only checks if between

    Parameters
    ----------
    dlog : pd.DataFrame
        Development Log with estimated values already calculated from the 
        build out years.
    year : str
        Future year column that contains the known data. (from build out)
    start_date_col : str
        Column containing the development start date
    end_date_col : str
        Column containing the development end date
    development_id_col : str
        Column containing the ID of the development. Used to report 
        developments with errors.
    total_column : str
        Column containing the total number of units during the development
        phase.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        A dataframe containing only developments where the data can be infilled
        A dataframe listing the developments where data cannot be estimated 
        (missing dates or missing totals)
    """
    start_date = start_date_col
    end_date = end_date_col
    development_id = development_id_col
    i_year = int(year)

    no_build_out = dlog.copy()
    # Empty dataframe to store d-log ids with insufficient data
    dlog_errors = pd.DataFrame(columns=[development_id, year])
    dlog_errors[development_id] = dlog[development_id]

    # TODO check that this is correct way to identify no data
    no_build_out = no_build_out.loc[no_build_out[year] == 0]
    # Just flag the number of errors for now
    print(f"No build out data for {no_build_out.shape[0]}")
    # Check for missing totals
    missing_totals = no_build_out.loc[
        no_build_out[total_column] == 0
    ][development_id]
    dlog_errors[year] = ""
    dlog_errors.loc[
        dlog_errors[development_id].isin(missing_totals), year
    ] = "missing_total"
    # Remove missing data from no_build_out
    no_build_out = no_build_out.loc[
        ~no_build_out[development_id].isin(missing_totals)
    ]
    # Check for missing start / end dates
    missing_dates = no_build_out.loc[
        (no_build_out[start_date].isna())
        | (no_build_out[end_date].isna())
    ][development_id]
    dlog_errors.loc[
        dlog_errors[development_id].isin(missing_dates), year
    ] = "missing_date"
    # Remove missing data from no_build_out
    no_build_out = no_build_out.loc[
        ~no_build_out[development_id].isin(missing_dates)
    ]
    # Ignore data where the year is outside of the start / end range
    # these will be zero for this year
    no_build_out[start_date] = no_build_out[start_date].dt.year
    no_build_out[end_date] = no_build_out[end_date].dt.year
    no_build_out = no_build_out.loc[
        (i_year >= no_build_out[start_date])
        & (i_year <= no_build_out[end_date])
    ]
    # Estimate the build out for the year - linear between start
    # and end dates
    no_build_out[year] = (
        no_build_out[total_column]
        * (
            (i_year - no_build_out[start_date])
            / (no_build_out[end_date] - no_build_out[start_date])
        )
    )
    # Merge back to the subset
    no_build_out.rename({year: "filled_data"}, axis=1, inplace=True)

    return no_build_out, dlog_errors


def constrain_post_dlog(df: pd.DataFrame,
                        constraint: pd.DataFrame,
                        la_equivalence: pd.DataFrame,
                        base_year: str,
                        year: str,
                        msoa_column: str
                        ):
    """Perform a constraint on the post d-log data. Constraint is on the 
    total growth at sector level.
    TODO Check that this is correct. Sometimes will result in zones decreasing
    in later years.

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

    # Calculate pre-constraint sector growth
    df["growth"] = (
        df[year]
        / df[base_year]
    )
    df["sector_growth"] = (
        df.groupby("sector_id")[year].transform("sum")
        /
        df.groupby("sector_id")[base_year].transform("sum")
    )
    # Calculate the constraint growth
    pop_constraint = constraint.copy()
    pop_constraint = pop_constraint.merge(
        la_equivalence,
        on=msoa_column,
        how="left"
    )[[msoa_column, "grouping_id", base_year, year]]

    # Keep grouping_id as index for mapping later
    pop_constraint = pop_constraint.groupby(
        "grouping_id"
    )[[base_year, year]].sum()

    pop_constraint["constraint"] = (
        pop_constraint[year]
        / pop_constraint[base_year]
    )
    # Merge to the DLOG data
    df["constraint"] = df["sector_id"].map(
        pop_constraint["constraint"]
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


def estimate_dlog_build_out(dlog: pd.DataFrame,
                            year: str,
                            start_date_col: str,
                            end_date_col: str,
                            data_key: str,
                            ) -> pd.DataFrame:
    
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
                constraints: pd.DataFrame,
                constraints_zone_equivalence: pd.DataFrame,
                segment_cols: List[str],
                dlog_conversion_factor: float = 1.0,
                msoa_column: str = "msoa_zone_id",
                min_growth_limit: float = 0.25,
                dlog_data_column_key: str = "population",
                perform_constraint: bool = True,
                audit_outputs: bool = False,
                audit_location: str = None
                ) -> pd.DataFrame:
    """TODO - Write docs when complete
    Should be done as base year population is read in - before growth is applied

    Requires:
    base_year_population - from nelum inputs, tfn segmentation
    dlog_data - as extracted from the database, currently in households
              - eastings / northings must be converted to MSOA
    ntem_growth - the current factors applied
    household_factors - data to convert households to population
    constrainer - data for constraining population 
                - MSOA level to be aggregated to LA

    Returns a DataFrame of future year population
    """

    post_dlog_df = pre_dlog_df.copy()[
        [msoa_column, base_year] + segment_cols + future_years
    ]
    # If we are applying to the employment data - remove the totals category
    # (will be added back in afterwards)
    if "employment_cat" in segment_cols:
        post_dlog_df = post_dlog_df.loc[
            post_dlog_df["employment_cat"] != "E01"
        ]
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
        [msoa_column, "sector_id"] + segment_cols,
        as_index=False
    )[[base_year] + future_years].sum()
    segment_split["seg_s"] = (
        segment_split.groupby(
            ["sector_id"] + segment_cols
        )[base_year].transform("sum")
        / segment_split.groupby("sector_id")[base_year].transform("sum")
    )

    # Read in development log data
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

    # Save dlog rows where data is missing
    dlog_missing_data = pd.DataFrame()
    # Identify zones where estimated growth was invalid
    invalid_growth_zones = pd.DataFrame()

    # Calculate the growth factors for each MSOA and year
    # May be faster to just get the first entry for each MSOA as the factors
    # are identical
    metric_columns = [base_year] + future_years
    dlog_additions = pre_dlog_df.groupby(msoa_column)[metric_columns].sum()

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

        # Calculate the new growth as base year absolute + data from dlog
        dlog_data_col = f"dlog_add_{year}"
        dlog_subset.rename({year: dlog_data_col}, axis=1, inplace=True)
        dlog_additions = dlog_additions.merge(
            dlog_subset,
            on=msoa_column,
            how="left"
        )
        dlog_additions[dlog_data_col].fillna(0.0, inplace=True)

        # Join to population dataframe and replace old values
        
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
        
        # Handle cases where addition of the dlog data (negative) leaves 
        # a zone with negative or very low population / employment
        # TODO check that this is correct - could instead adjust all segments
        # in the zone to preserver spatial distribution
        invalid_growth = segment_split[adj_growth] < min_growth_limit
        audit_zones = segment_split[invalid_growth][
            [msoa_column] + segment_cols + [adj_growth]
        ].copy()
        audit_zones = audit_zones.loc[audit_zones[adj_growth] != 0]
        audit_zones["year"] = year
        audit_zones.rename({adj_growth: "growth_factor"}, axis=1, inplace=True)
        segment_split.loc[invalid_growth, adj_growth] = min_growth_limit
        if invalid_growth_zones.empty:
            invalid_growth_zones = audit_zones.drop_duplicates()
        else:
            invalid_growth_zones = invalid_growth_zones.append(
                audit_zones.drop_duplicates()
            )
        
        # Update original data with new segment growth
        merge_cols = [msoa_column] + segment_cols
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
                msoa_column=msoa_column
            )
            print(f"Post constraint total = {post_dlog_df[year].sum()}")

        # Drop any temporary columns
        post_dlog_df.drop(
            ["growth", "constraint", "sector_growth"],
            axis=1,
            inplace=True
        )

    # Use pre and post dlog growth to identify exceptional zones
    e_zones = identify_exceptional_zones(
        pre_df=pre_dlog_growth,
        post_df=post_dlog_growth,
        growth_cutoff=0.75,
        base_year=base_year,
        future_years=future_years
    )
    
    # Drop any temporary columns
    post_dlog_df.drop(
        ["sector_id"] + [col for col in post_dlog_df.columns 
                         if "adj_growth" in col],
        axis=1,
        inplace=True
    )
    # If we are applying to the employment data - add the totals category
    # back in
    if "employment_cat" in segment_cols:
        # Calculate the totals for each zone
        emp_cat1 = post_dlog_df.groupby(msoa_column, as_index=False)[
            [base_year] + future_years
        ].sum()
        emp_cat1["employment_cat"] = "E01"
        # Append back to the post_dlog dataframe and sort by zone/segment
        post_dlog_df = post_dlog_df.append(emp_cat1)
        post_dlog_df.sort_values(by=[msoa_column, "employment_cat"],
                                 inplace=True)
        post_dlog_df.reset_index(drop=True, inplace=True)


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
        e_zones.to_csv(
            os.path.join(audit_location, "exceptional_zones.csv"),
            index=False
        )

    
    return post_dlog_df, e_zones

