
import os

from typing import List

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


def apply_d_log_population(population: pd.DataFrame,
                           base_year:  str,
                           future_years: List[str],
                           dlog_path: str,
                           constraints: pd.DataFrame,
                           constraints_zone_equivalence: str,
                           household_factors: float,
                           development_zone_lookup: str,
                           msoa_zones: str,
                           msoa_column: str = "msoa_zone_id",
                           perform_constraint: bool = True
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
    # TODO musch of this should probably be in separate functions to
    # increase readability

    updated_population = population.copy()
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
    updated_population["sector_id"] = updated_population[msoa_column].map(
        la_equivalence.set_index(msoa_column)["grouping_id"]
    )
    updated_population["seg_s"] = (
        updated_population.groupby(
            ["sector_id", "traveller_type", "soc", "ns", "ca"]
        )[base_year].transform("sum")
        / updated_population.groupby(
            "sector_id"
        )[base_year].transform("sum")
    )
    # Adjust to handle zeroes in the base for some MSOAS within sectors
    updated_population["adj_seg_s"] = updated_population["seg_s"]
    updated_population.loc[
        updated_population[base_year] == 0,
        "adj_seg_s"
    ] = 0.0
    updated_population.loc[
        updated_population[base_year] != 0.0,
        "adj_seg_s"
    ] /= updated_population.loc[
        updated_population[base_year] != 0.0].groupby(
            msoa_column
    )["adj_seg_s"].transform("sum")
    print(updated_population.head(10))

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

    # Check that the population dataframe contains the correct columns
    pop_required_cols = ["msoa_zone_id", base_year] + future_years
    if not all(col in population.columns for col in pop_required_cols):
        raise ValueError("Population Dataframe does not contain the required"
                         " columns")

    # Convert d-log data to the required format
    # TODO check the format of d-log data after the update

    # Empty dataframe to store d-log ids with insufficient data
    dlog_errors = pd.DataFrame(columns=[development_id, "errors"])
    dlog_errors[development_id] = dlog_data[development_id]

    # Extract the build out columns
    build_out_cols = [
        col for col in dlog_data.columns
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

    # Calculate the growth factors for each MSOA and year
    # May be faster to just get the first entry for each MSOA as the factors
    # are identical
    metric_columns = [base_year] + future_years
    population_factors = population.groupby(msoa_column)[metric_columns].sum()

    # Calculate a new column with the estimated new households for each year
    for year in future_years:

        print(f"Replacing D-LOG data for {year}")

        i_year = int(year)

        dlog_subset = dlog_data.copy()
        dlog_columns = [development_id,
                        "units_of_properties",
                        start_date,
                        end_date]

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

        # Where the build out profile contains nothing, check the start
        # and end dates, if relevant then use these
        # TODO implement this method as a fallback
        if (dlog_subset[year] == 0).sum() > 0:
            no_build_out = dlog_subset.copy()
            no_build_out = no_build_out.loc[no_build_out[year] == 0]
            # Just flag the number of errors for now
            print(f"No build out data for {no_build_out.shape[0]}")
            # Check for missing totals
            missing_totals = no_build_out.loc[
                no_build_out["units_of_properties"] == 0
            ][development_id]
            dlog_errors["errors"] = ""
            dlog_errors.loc[
                dlog_errors[development_id].isin(missing_totals), "errors"
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
                dlog_errors[development_id].isin(missing_dates), "errors"
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
                no_build_out["units_of_properties"]
                * (
                    (i_year - no_build_out[start_date])
                    / (no_build_out[end_date] - no_build_out[start_date])
                )
            )
            # Merge back to the subset
            no_build_out.rename({year: "filled_data"}, axis=1, inplace=True)
            dlog_subset = dlog_subset.merge(
                no_build_out[[development_id, "filled_data"]],
                on=development_id,
                how="left"
            )
            dlog_subset["filled_data"].fillna(dlog_subset[year], inplace=True)
            dlog_subset = dlog_subset.drop(
                year, axis=1
            ).rename(
                {"filled_data": year}, axis=1
            )
            print(f"Overriding {no_build_out.shape[0]} developments")
            # Write Errors to file
            dlog_errors.to_csv("dlog_errors.csv")

        # Convert to population
        dlog_subset[year] *= household_factors

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

        # Calculate the new growth as base year pop + population from dlog
        # TODO should this be done over all segments?
        dlog_subset.rename({year: "dlog_data"}, axis=1, inplace=True)
        population_factors = population_factors.merge(
            dlog_subset,
            on=msoa_column,
            how="left"
        )
        population_factors["dlog_data"].fillna(0.0, inplace=True)

        print(population_factors)
        print(population_factors.loc[population_factors["dlog_data"] != 0.0])

        # Join to population dataframe and replace old values
        print(f"Previous {year} total = {updated_population[year].sum()}")
        updated_population[year] += (
            updated_population["adj_seg_s"]
            * updated_population[msoa_column].map(
                population_factors.set_index(msoa_column)["dlog_data"]
            ).fillna(0.0)
        )
        print(f"New {year} total = {updated_population[year].sum()}")
        print(updated_population)

        # Constrain to LA TODO Change to growth constraint

        # Calculate pre-constraint sector growth
        updated_population["growth"] = (
            updated_population[year]
            / updated_population[base_year]
        )
        updated_population["sector_growth"] = (
            updated_population.groupby("sector_id")[year].transform("sum")
            /
            updated_population.groupby("sector_id")[base_year].transform("sum")
        )
        # Calculate the constraint growth
        pop_constraint = constraints.copy()
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
        updated_population["constraint"] = updated_population["sector_id"].map(
            pop_constraint["constraint"]
        )
        # Adjust segment growth factors
        updated_population["growth"] *= (
            updated_population["constraint"]
            / updated_population["sector_growth"]
        )
        # Adjust forecast year by new growth
        updated_population[year] = (
            updated_population[base_year]
            * updated_population["growth"]
        )
        
        # Drop any temporary tables
        updated_population.drop(
            ["growth", "constraint", "sector_growth"],
            axis=1,
            inplace=True
        )
        population_factors.drop(
            ["dlog_data"],
            axis=1,
            inplace=True
        )

    # Drop any temporary tables
    updated_population.drop(
        ["sector_id", "seg_s", "adj_seg_s"],
        axis=1,
        inplace=True
    )

    population_factors.to_csv("test.csv")
    updated_population.to_csv("post_dlog_pop.csv")

    return updated_population

