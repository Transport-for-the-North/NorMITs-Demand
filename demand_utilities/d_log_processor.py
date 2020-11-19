
import os

from typing import List, Tuple

import pandas as pd

from demand_utilities import utils as du


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

    return df


def apply_d_log(pre_dlog_df: pd.DataFrame,
                base_year:  str,
                future_years: List[str],
                dlog_path: str,
                constraints: pd.DataFrame,
                constraints_zone_equivalence: str,
                segment_cols: List[str],
                dlog_conversion_factor: float,
                development_zone_lookup: str,
                msoa_zones: str,
                msoa_column: str = "msoa_zone_id",
                total_column: str = "units_of_properties",
                perform_constraint: bool = True,
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

    post_dlog_df = pre_dlog_df.copy()
    print(post_dlog_df)
    # Calculate the base year segment shares over each sector
    print("Calculating Segment Share by Sector")
    la_equivalence = du.safe_read_csv(constraints_zone_equivalence)
    la_equivalence.rename(
        {"model_zone_id": msoa_column},
        axis=1,
        inplace=True
    )
    # Use map instead of merge as MSOA to sector will be 1 to 1 and likely
    # faster than merge
    post_dlog_df["sector_id"] = post_dlog_df[msoa_column].map(
        la_equivalence.set_index(msoa_column)["grouping_id"]
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
    development_id = "development_site_id"
    start_date = "start_date"
    end_date = "expected_completion_date"
    # Set the start and end as datetime type
    # Ignore errors for now - these will be caught later
    dlog_data = set_datetime_types(
        dlog_data,
        [start_date, end_date],
        format="%d/%m/%Y",
        errors="coerce"
    )

    # Read in development to msoa zone lookup
    zone_lookup = du.safe_read_csv(development_zone_lookup)
    zone_lookup = zone_lookup[[development_id, "msoa11cd"]]

    # Check that the dataframe contains the correct columns
    pop_required_cols = ["msoa_zone_id", base_year] + future_years
    if not all(col in pre_dlog_df.columns for col in pop_required_cols):
        raise ValueError("Dataframe does not contain the required"
                         " columns")

    # Convert d-log data to the required format
    # TODO check the format of d-log data after the update
    dlog_missing_data = pd.DataFrame()

    # Calculate the growth factors for each MSOA and year
    # May be faster to just get the first entry for each MSOA as the factors
    # are identical
    metric_columns = [base_year] + future_years
    dlog_additions = pre_dlog_df.groupby(msoa_column)[metric_columns].sum()

    # Calculate a new column with the estimated new households for each year
    for year in future_years:

        print(f"Replacing D-LOG data for {year}")

        dlog_subset = dlog_data.copy()
        dlog_columns = [development_id,
                        total_column,
                        start_date,
                        end_date]

        dlog_subset = parse_dlog_build_out(
            dlog=dlog_subset,
            year=year,
            dlog_columns=dlog_columns
        )
        print(f"D-LOG data contains {dlog_subset[year].sum()} units")

        # Where the build out profile contains nothing, check the start
        # and end dates, if relevant then use these
        if (dlog_subset[year] == 0).sum() > 0:

            infill_data, dlog_errors = infill_dlog_build_out(
                dlog=dlog_subset,
                year=year,
                start_date_col=start_date,
                end_date_col=end_date,
                development_id_col=development_id,
                total_column=total_column
            )

            dlog_subset = dlog_subset.merge(
                infill_data[[development_id, "filled_data"]],
                on=development_id,
                how="left"
            )
            dlog_subset["filled_data"].fillna(dlog_subset[year], inplace=True)
            dlog_subset = dlog_subset.drop(
                year, axis=1
            ).rename(
                {"filled_data": year}, axis=1
            )
            print(f"Overriding {infill_data.shape[0]} developments")
            # Collate errors
            if dlog_missing_data.empty:
                dlog_missing_data = dlog_errors
            else:
                dlog_missing_data = dlog_missing_data.merge(
                    dlog_errors,
                    on=development_id
                )

            print(f"Infill data contains {infill_data['filled_data'].sum()} units")
            print(f"D-LOG data contains {dlog_subset[year].sum()} units")

        # Convert to population / employment
        dlog_subset[year] *= dlog_conversion_factor

        # Map to MSOA zones
        dlog_subset = dlog_subset.merge(zone_lookup, on=development_id)
        dlog_subset = du.convert_msoa_naming(
            dlog_subset,
            msoa_col_name="msoa11cd",
            msoa_path=msoa_zones,
            to="int"
        )
        dlog_subset.rename({"msoa11cd": msoa_column}, axis=1, inplace=True)
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

        if perform_constraint:
            post_dlog_df = constrain_post_dlog(
                df=post_dlog_df,
                constraint=constraints,
                la_equivalence=la_equivalence,
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

    # Drop any temporary columns
    post_dlog_df.drop(
        ["sector_id"],
        axis=1,
        inplace=True
    )

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

    return post_dlog_df

