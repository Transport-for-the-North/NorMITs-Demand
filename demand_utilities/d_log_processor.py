
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

    # Store d-log ids with insufficient data
    dlog_errors = pd.DataFrame(columns=[development_id, "error"])

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
    # May be faster to just get the first entry for each MSOA
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
        dlog_subset = dlog_subset.groupby(msoa_column, as_index=False)[year].sum()

        # Calculate the original growth factors for this year
        old_factor = f"{year}_old_factor"
        new_factor = f"{year}_new_factor"
        population_factors[old_factor] = (
            population_factors[year] / population_factors[base_year]
        )
        population_factors["abs_growth"] = (
            population_factors[year] - population_factors[base_year]
        )

        # Calculate the new growth as base year pop + population from dlog
        # TODO should this be done over all segments?
        dlog_subset.rename({year: "filled_data"}, axis=1, inplace=True)
        population_factors = population_factors.merge(
            dlog_subset,
            on=msoa_column,
            how="left"
        )
        population_factors["filled_data"].fillna(
            population_factors["abs_growth"],
            inplace=True
        )
        population_factors["filled_data"] += population_factors[base_year]
        population_factors = population_factors.drop(
            year, axis=1
        ).rename(
            {"filled_data": year}, axis=1
        )
        population_factors[new_factor] = (
            population_factors[year]
            / population_factors[base_year]
        )

        # Join to population dataframe and replace old values
        updated_population[year] = (
            updated_population[base_year]
            + updated_population[msoa_column].map(
                population_factors.set_index(msoa_column)[new_factor]
            )
        )

        # Constrain to LA TODO Change to growth constraint
        # pop_constraint = pd.read_csv(constraints)
        pop_constraint = constraints.copy()
        la_equivalence = pd.read_csv(constraints_zone_equivalence)
        la_equivalence.rename(
            {"model_zone_id": msoa_column},
            axis=1,
            inplace=True
        )
        print(pop_constraint)
        print(la_equivalence)
        pop_constraint = pop_constraint.merge(
            la_equivalence,
            on=msoa_column,
            how="left"
        )[[msoa_column, "grouping_id", year]]
        pop_constraint["total"] = pop_constraint.groupby(
            "grouping_id"
        )[year].transform("sum")
        # Calculate Pre constraint totals
        # By MSOA
        pre_constraint_total = updated_population.groupby(
            msoa_column,
            as_index=False
        )[year].sum()
        pre_constraint_total[msoa_column] = (
            pre_constraint_total[msoa_column].map(
                la_equivalence.set_index(msoa_column)["grouping_id"]
            )
        )
        print("MSOA")
        print(pre_constraint_total)
        # By LA
        pre_constraint_total = pre_constraint_total.groupby(
            msoa_column,
            as_index=False
        )[year].sum()
        pop_constraint["pre_constraint"] = pop_constraint["grouping_id"].map(
            pre_constraint_total.set_index(msoa_column)[year]
        )
        print("LA")
        print(pre_constraint_total)
        # Calculate adjustment factor
        pop_constraint["factor"] = (
            pop_constraint["total"]
            / pop_constraint["pre_constraint"]
        )
        print("FACTORS")
        print(pop_constraint)
        # Merge to updated population and adjust
        updated_population["factor"] = updated_population[msoa_column].map(
            pop_constraint.set_index(msoa_column)["factor"]
        )
        updated_population[year] *= updated_population["factor"]
        updated_population.drop("factor", axis=1, inplace=True)
        
    population_factors.to_csv("test.csv")
    updated_population.to_csv("dlog_pop.csv")

    return updated_population
