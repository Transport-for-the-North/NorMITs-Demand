import os
from typing import List
from typing import Tuple
from typing import Iterable
from itertools import product

import numpy as np
import pandas as pd
from tqdm import tqdm

# Import attraction generator to access soc splits
import efs_attraction_generator as attr

from zone_translator import ZoneTranslator
from demand_utilities import utils as du
import efs_constants as consts


def load_exceptional_zones(productions_export: str,
                           attractions_export: str
                           ) -> pd.DataFrame:
    """Load all identified exceptional zones. 
    As these are generated at the population level, the returned dataframe
    will contain MSOA zones.

    Parameters
    ----------
    productions_export : str
        Export folder supplied to the efs_production_generator.
        Used to retrieve exceptional population growth zones.
    attractions_export : str
        Export folder supplied to efs_attraction_generator.
        Used to retrieve exceptional employment growth zones.

    Returns
    -------
    pd.DataFrame
        A dataframe that should have the column "msoa_zone_id", containing all 
        identified exceptional zones.
    """

    e_zones = pd.read_csv(
        os.path.join(productions_export, "exceptional_zones.csv")
    )
    e_zones = pd.concat(
        [e_zones,
         pd.read_csv(
             os.path.join(attractions_export, "exceptional_zones.csv")
         )],
        axis=0
    )

    return e_zones.drop_duplicates()


def segment_employment(employment: pd.DataFrame,
                       soc_weights_path: str,
                       zone_column: str,
                       data_col: str,
                       msoa_lookup_path: str
                       ) -> pd.DataFrame:
    """Takes a dataframe containing employment data and a path to soc_weights,
    returns the employment data segmented by skill level (soc).

    Parameters
    ----------
    employment : pd.DataFrame
        Employment data containing a zone_column, data_col, and 
        "employment_cat" (employment category). Zoning should be at MSOA.
    soc_weights_path : str
        Path to a file containing soc weights. Should contain a column called
        "msoa_zone_id".
    zone_column : str
        Usually "msoa_zone_id", should contain MSOA zones.
    data_col : str
        Column in employment that contains the employee data.
    msoa_lookup_path : str
        File used to convert MSOA codes to zone ids

    Returns
    -------
    pd.DataFrame
        Returns employment segmented by soc.
    """

    soc_weights = attr.get_soc_weights(
        soc_weights_path
    )

    # Use dummy values of purpose id to get the employment split by soc
    emp_soc = employment.copy()
    emp_soc["p"] = consts.SOC_P[0]
    emp_ns = employment.copy()
    emp_ns["p"] = consts.NS_P[0]
    soc_employment = pd.concat(
        [emp_soc, emp_ns]
    )

    soc_weights.reset_index(inplace=True)
    soc_weights.rename({"msoa_zone_id": zone_column}, axis=1, inplace=True)

    soc_weights = du.convert_msoa_naming(
        soc_weights,
        msoa_col_name=zone_column,
        msoa_path=msoa_lookup_path,
        to='int'
    )
    soc_weights.set_index(zone_column, inplace=True)

    soc_employment = attr.split_by_soc(
        soc_employment,
        soc_weights,
        zone_col=zone_column,
        p_col="p",
        unique_col=data_col,
        soc_col="soc"
    )

    return soc_employment


def attraction_exceptional_trip_rate(observed_base: pd.DataFrame,
                                     land_use: pd.DataFrame,
                                     e_zones: pd.DataFrame,
                                     base_year: str,
                                     segment_cols: List[str],
                                     zone_column: str,
                                     sector_lookup: pd.Series = None,
                                     purpose_column: str = "purpose_id",
                                     soc_weights_path: str = None,
                                     msoa_lookup_path: str = None
                                     ) -> pd.DataFrame:
    """Calculates the sector level trip rates for attractions using the 
    observed base attractions and land use.

    Parameters
    ----------
    observed_base : pd.DataFrame
        Observed base attractions. Must contain all columns in segment_cols
        and be at the same zone level as the land_use.
    land_use : pd.DataFrame
        Base year land use data. Should contain the column base_year and all 
        columns in segment_cols, with the exception of the purpose_column.
    e_zones : pd.DataFrame
        Unused
    base_year : str
        The base year that will be used to calculate the trip rates. Must be a 
        column in bothobserved_base and land_use.
    segment_cols : List[str]
        The segment columns present in observed_base. Generally will just be 
        "purpose_id".
    zone_column : str
        Column containing zone ids.
    sector_lookup : pd.Series, optional
        A pd.Series with an index containing all zone ids in zone_column that 
        map to the sector in the series. If None, the zone system will be used 
        as the sector system, by default None
    purpose_column : str, optional
        Column containing purpose ids in observed_base, by default "purpose_id"
    soc_weights_path : str, optional
        Path to the skill level weights. Required is attractions are needed 
        with soc segmentation - not required after attractions generation 
        update, by default None

    Returns
    -------
    pd.DataFrame
        Dataframe with the columns "sector_id", all columns in segment_cols, 
        and "trip_rate". This format is required by the function
        handle_exceptional_growth.
    """

    emp_group_cols = ["sector_id"] + segment_cols
    if sector_lookup is None:
        def sector_map(x): return x
    else:
        sector_map = sector_lookup.copy()

    tr_e = None

    print("Calculating Attraction Weights")

    # Add sector column to observed data
    observed = observed_base.copy()
    observed["sector_id"] = observed[zone_column].map(sector_map)

    # Filter employment to only the commute category
    emp_sub = land_use.loc[land_use["employment_cat"] == "E01"]
    emp_sub = emp_sub[[zone_column, base_year]]

    # If required, convert employment to soc segmentation
    if soc_weights_path is not None:
        emp = segment_employment(
            emp_sub,
            soc_weights_path=soc_weights_path,
            zone_column=zone_column,
            data_col=base_year,
            msoa_lookup_path=msoa_lookup_path
        )
        observed["soc"] = observed["soc"].astype("int")
    else:
        emp = emp_sub.copy()

    # Group and sum the required segmentation
    emp["sector_id"] = emp[zone_column].map(sector_map)
    if soc_weights_path is not None:
        emp = emp.groupby(
            ["sector_id", "soc"],
            as_index=False
        )[base_year].sum()
        emp["soc"] = emp["soc"].astype("int")
    else:
        emp = emp.groupby(["sector_id"], as_index=False)[base_year].sum()
    emp.rename({base_year: "land_use"}, axis=1, inplace=True)

    print("Split Employment")
    print(emp)

    observed.to_csv("observed_pre_group.csv")

    observed = observed.groupby(
        emp_group_cols,
        as_index=False
    )[base_year].sum()

    print("Observed Attractions")
    print(observed)

    emp_group_cols.remove(purpose_column)

    # Merge data and calculate the trip rate
    tr_e = observed.merge(
        emp,
        on=emp_group_cols
    )

    emp_group_cols.insert(1, purpose_column)
    tr_e.set_index(emp_group_cols, inplace=True)
    tr_e.sort_index(inplace=True)
    tr_e.reset_index(inplace=True)

    tr_e["trip_rate"] = tr_e[base_year] / tr_e["land_use"]

    tr_e = tr_e[emp_group_cols + ["trip_rate"]]

    print(tr_e)

    tr_e.to_csv("tr_e.csv")

    return tr_e


def convert_pop_segmentation(population: pd.DataFrame,
                             grouping_cols: List[str],
                             value_cols: List[str],
                             sector_map: pd.Series = None,
                             zone_column: str = "model_zone_id",
                             sector_column: str = "sector_id"
                             ) -> pd.DataFrame:
    """Convert a given population dataframe to the segmentation format required
    by the productions. Converts soc and ns columns from floats to a mix of
    integers and "none".

    Parameters
    ----------
    population : pd.DataFrame
        Dataframe containing population data, Requires the columns in 
        grouping_cols, value_cols, and "soc" and "ns".
    grouping_cols : List[str]
        Columns to group by. If a sector_map is supplied, this should contain
        "sector_id". Should also contain "soc" and "ns".
    value_cols : List[str]
        Columns to sum when grouping. 
    sector_map : pd.Series, optional
        A pd.Series with an index containing all zone ids in zone_column that 
        map to the sector in the series. If None, the zone system will be used 
        as the sector system, by default None
    zone_column : str, optional
        The column in population that contains the zone ids, by default
        "model_zone_id"
    sector_column : str, optional
        Column name for the new column containing sector ids, by default 
        "sector_id"

    Returns
    -------
    pd.DataFrame
        Dataframe with (optional) new column sector_column. Columns "soc" and 
        "ns" have their types converted to the same as in EFS productions.
    """

    pop = population.copy()
    pop["ns"] = pop["ns"].astype("int")
    pop["soc"] = pop["soc"].astype("int")
    pop["ns"] = "none"
    pop["ns"] = pop["ns"].astype("str")
    pop["soc"] = pop["soc"].astype("str")
    if sector_map is not None:
        pop[sector_column] = pop[zone_column].map(sector_map)
    pop_g = pop.groupby(grouping_cols)[value_cols].sum()
    pop = population.copy()
    pop["ns"] = pop["ns"].astype("int")
    pop["soc"] = pop["soc"].astype("int")
    pop["soc"] = "none"
    pop["soc"] = pop["soc"].astype("str")
    pop["ns"] = pop["ns"].astype("str")
    if sector_map is not None:
        pop[sector_column] = pop[zone_column].map(sector_map)
    pop_g = pop_g.append(pop.groupby(grouping_cols)[value_cols].sum())

    pop_g = pop_g.sort_index()
    pop_g = pop_g.reset_index()

    return pop_g


def production_exceptional_trip_rate(observed_base: pd.DataFrame,
                                     land_use: pd.DataFrame,
                                     e_zones: pd.DataFrame,
                                     base_year: str,
                                     segment_cols: List[str],
                                     zone_column: str,
                                     sector_lookup: pd.Series = None,
                                     purpose_column: str = "purpose_id"
                                     ) -> pd.DataFrame:
    """Calculates the sector level trip rates for productions using the 
    observed base productions and land use.

    Parameters
    ----------
    observed_base : pd.DataFrame
        Observed base productions. Must contain all columns in segment_cols
        and be at the same zone level as the land_use.
    land_use : pd.DataFrame
        Base year land use data. Should contain the column base_year and all 
        columns in segment_cols, with the exception of the purpose_column.
    e_zones : pd.DataFrame
        Unused
    base_year : str
        The base year that will be used to calculate the trip rates. Must be a 
        column in bothobserved_base and land_use.
    segment_cols : List[str]
        The segment columns present in observed_base. Generally will just be 
        "purpose_id".
    zone_column : str
        Column containing zone ids.
    sector_lookup : pd.Series, optional
        A pd.Series with an index containing all zone ids in zone_column that 
        map to the sector in the series. If None, the zone system will be used 
        as the sector system, by default None
    purpose_column : str, optional
        Column containing purpose ids in observed_base, by default "purpose_id"
    soc_weights_path : str, optional
        Path to the skill level weights. Required is productions are needed 
        with soc segmentation - not required after productions generation 
        update, by default None

    Returns
    -------
    pd.DataFrame
        Dataframe with the columns "sector_id", all columns in segment_cols, 
        and "trip_rate". This format is required by the function
        handle_exceptional_growth.
    """

    # Add sector column to observed data
    pop_group_cols = ["sector_id"] + segment_cols

    pop_group_cols.remove(purpose_column)

    if sector_lookup is None:
        def sector_map(x): return x
    else:
        sector_map = sector_lookup.copy()

    tr_p = None

    print("Caclulating Production Trip Rates")

    # Convert to the trip segementation dealing with mismatched types
    pop_g = convert_pop_segmentation(
        land_use,
        pop_group_cols,
        value_cols=base_year,
        sector_map=sector_map,
        zone_column=zone_column
    )

    pop_g = pop_g.rename({base_year: "land_use"}, axis=1)

    # Group and sum the required segmentation
    observed = observed_base.copy()
    observed["sector_id"] = observed[zone_column].map(sector_map)
    observed = observed.groupby(
        pop_group_cols + [purpose_column],
        as_index=False
    )[base_year].sum()

    # Merge data and calculate the trip rate
    tr_p = observed.merge(
        pop_g,
        on=pop_group_cols
    )

    tr_p["trip_rate"] = tr_p[base_year] / tr_p["land_use"]

    pop_group_cols.insert(1, purpose_column)
    tr_p.set_index(pop_group_cols, inplace=True)
    tr_p.sort_index(inplace=True)
    tr_p.reset_index(inplace=True)

    print("Saving to files")
    observed.to_csv("observed_base_prod.csv", index=False)
    pop_g.to_csv("pop_g.csv", index=False)

    tr_p = tr_p[pop_group_cols + ["trip_rate"]]

    print(tr_p)

    return tr_p


def handle_exceptional_growth(synth_future: pd.DataFrame,
                              synth_base: pd.DataFrame,
                              observed_base: pd.DataFrame,
                              zone_column: str,
                              segment_columns: List[str],
                              value_column: str,
                              base_year: str,
                              exceptional_zones: pd.DataFrame,
                              land_use: pd.DataFrame,
                              trip_rates: pd.DataFrame,
                              sector_lookup: pd.Series,
                              force_soc_type: bool = False
                              ) -> pd.DataFrame:
    """Applies the following growth criteria depending on the zone type 
    (normal growth or exceptional) -
    normal = observed_base * synthetic_forecast / synthetic_base
    exceptional = land_use_future * sector_trip_rate
    Outputs files containing the changes made to exceptional zones.
    Returns a dataframe containing the forecast trips after applying the 
    growth criteria.

    Parameters
    ----------
    synth_future : pd.DataFrame
        Synthetic forecast data.
        Must contain zone_column, segment_columns, and value_column
    synth_base : pd.DataFrame
        Synthetic base year data.
        Must contain zone_column, segment_columns, and base_year
    observed_base : pd.DataFrame
        Observed base year data.
        Must contain zone_column, segment_columns, and base_year
    zone_column : str
        Column containing zone ids. Zone system must be common to all input
        dataframes.
    segment_columns : List[str]
        Segment columns present in the synth_future/synth_base/observed_base
        and trip_rates dataframes. Used as the merge keys for these dataframes.
    value_column : str
        The column containing data relevant to this forecast year. 
    base_year : str
        The column containing base year data.
    exceptional_zones : pd.DataFrame
        Supply any zones that are to be considered exceptional. Contains a list 
        of the "exceptional" zones
    land_use : pd.DataFrame
        Contains land use data for each zone, with the same segmentation as 
        the synthetic/observed data (except for purpose_id).
    trip_rates : pd.DataFrame
        Contains trip rates for each sector in sector_lookup, with the same 
        segmentation as the synthetic/observed data. 
    sector_lookup : pd.Series
        A pd.Series with an index containing all zone ids in zone_column that 
        map to the sector in the series. Maps the zones in land_use to the
        sectors in trip_rates.
    force_soc_type : bool, optional
        If the type of the soc columns should be forced to integer. Used for
        attractions if skill segmentation is needed in the outputs, by default 
        True

    Returns
    -------
    pd.DataFrame
        The base observed data grown according to the growth criteria.
        Contains all segmentation columns, the intermediate synthetic and 
        observed data, and the final grown trips.
    """
    # TODO Note this could result in lower trips generated than before. See
    # the audit output for the changes

    sector_map = sector_lookup.copy()

    # Combine all dataframes into one, renameing the columns for 
    # better readability
    merge_cols = [zone_column] + segment_columns
    forecast_vector = pd.merge(
        synth_base.rename({base_year: "s_b"}, axis=1),
        synth_future.rename({value_column: "s_f"}, axis=1),
        how="outer",
        on=merge_cols
    )
    forecast_vector = pd.merge(
        forecast_vector,
        observed_base.rename({base_year: "b_c"}, axis=1),
        how="outer",
        on=merge_cols
    )

    # Normal Zones Growth Calculation
    forecast_vector[value_column] = (
        forecast_vector["b_c"]
        * forecast_vector["s_f"]
        / forecast_vector["s_b"]
    )

    # Handle Exceptional Zones
    # Get the relevant land use data
    e_land_use = land_use.loc[
        land_use[zone_column
                 ].isin(exceptional_zones[zone_column])].copy()
    # Map the land use to the sector used for the trip rates
    e_land_use["sector_id"] = e_land_use[zone_column].map(sector_map)

    print("Exceptional Land Use")
    print(e_land_use)
    e_land_use.to_csv("e_land_use.csv")
    print("Trip Rates")
    print(trip_rates)
    trip_rate_merge_cols = merge_cols.copy()
    trip_rate_merge_cols.remove(zone_column)
    trip_rate_merge_cols.remove("purpose_id")
    trip_rate_merge_cols.insert(0, "sector_id")

    # (Required to ensure a complete merge on soc column -
    # only for attractions)
    if force_soc_type:
        e_land_use["soc"] = e_land_use["soc"].astype("int")

    # Merge on the common segmentation and re-calculate the synthetic forecast 
    # using the new sector level trip rates
    print("Merging on ", trip_rate_merge_cols)
    e_infill = pd.merge(
        e_land_use,
        trip_rates,
        how="left",
        on=trip_rate_merge_cols
    )
    e_infill["s_f_exceptional"] = (
        e_infill[value_column]
        * e_infill["trip_rate"]
    )
    # Tidy up unused columns
    e_infill.drop(["trip_rate", "sector_id", value_column],
                  axis=1, inplace=True)
    print("Exceptional Forecast")
    print(e_infill)

    # Merge to the final forecast vector and overwrite with s_f_exceptional 
    # where necessary
    forecast_vector = pd.merge(
        forecast_vector,
        e_infill,
        how="left",
        on=merge_cols
    )

    mask = forecast_vector[zone_column].isin(exceptional_zones[zone_column])
    forecast_vector.loc[mask, value_column] = (
        forecast_vector.loc[mask, "s_f_exceptional"]
    )
    forecast_vector.drop(["s_f_exceptional"], axis=1, inplace=True)

    print("Final Forecast")
    print(forecast_vector)
    forecast_vector.to_csv("test.csv")
    # TODO Add constraining?

    return forecast_vector


def growth_criteria(synth_productions: pd.DataFrame,
                    synth_attractions: pd.DataFrame,
                    observed_prod_path: str,
                    observed_attr_path: str,
                    population_path: str,
                    employment_path: str,
                    msoa_lookup_path: str,
                    segments: dict,
                    future_years: List[str],
                    base_year: str,
                    zone_translator: ZoneTranslator = None,
                    zone_translator_args: dict = None,
                    exceptional_zones: pd.DataFrame = None,
                    trip_rate_sectors: str = None
                    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Processes the PA vectors and other data from the main EFS and returns 
    the forecast vectors.

    Parameters
    ----------
    synth_productions : pd.DataFrame
        Synthetic production data for base and future years.
    synth_attractions : pd.DataFrame
        Synthetic attraction data for base and future years.
    observed_prod_path : str
        Path to the base observed productions.
        TODO Change the processing of this depending on the format of the final
        base observed.
    observed_attr_path : str
        Path to the base observed attractions.
        TODO Change the processing of this depending on the format of the final
        base observed.
    population_path : str
        Path to the population data as output by the efs_production_generator.
        At MSOA zone level.
    employment_path : str
        Path to the employment data as output by the efs_attraction_generator.
        At MSOA zone level.
    msoa_lookup_path : str
        Path to the MSOA code to id conversion file.
    segments : dict
        Dictionary of segments required for each dataframe.
        Contains:
            "prod" : List[str] Production segments required
            "attr" : List[str] Attraction segments required
            "pop" : List[str] Population segments required
            "emp" : List[str] Employment segments required
    future_years : List[str]
        List of all future years to calculate the forecast for. Must be a 
        column for each of these in the synthetic/observed/land use.
    base_year : str
        Base year column that must be present in the observed/synthetic/land 
        use data.
    zone_translator : ZoneTranslator, optional
        Instance of a zone translator - supplied by the main EFS process, by
        default None
    zone_translator_args : dict, optional
        Arguments to provide to the zone_translator.run method. Should contain
        the translation dataframe, start and end zoning, by default None
    exceptional_zones : pd.DataFrame, optional
        List of all exceptional zones, identified by the development log in 
        the production and attraction generators, by default None
    trip_rate_sectors : pd.Series, optional
        A pd.Series with an index containing all zone ids in the zone_columns
        that map to the sector in the series, by default None

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Returns the grown productions and attractions data.
    """

    # Load files and format inputs as necessary
    print("Loading Files")
    population = du.safe_read_csv(population_path)
    employment = du.safe_read_csv(employment_path)
    observed_productions = du.safe_read_csv(observed_prod_path)
    observed_attractions = du.safe_read_csv(observed_attr_path)

    pop_segments = [x for x in segments["pop"] if x != "model_zone_id"]
    emp_segments = [x for x in segments["emp"] if x != "model_zone_id"]
    prod_segments = [x for x in segments["prod"] if x != "model_zone_id"]
    attr_segments = [x for x in segments["attr"] if x != "model_zone_id"]

    # Rename zone and segment columns to the standard EFS naming
    print("Population")
    print(population)
    population.rename(
        {"msoa_zone_id": "model_zone_id",
         "area_type": "area_type_id",
         "ca": "car_availability_id"},
        axis=1,
        inplace=True)
    population = du.convert_msoa_naming(
        population,
        msoa_col_name="model_zone_id",
        msoa_path=msoa_lookup_path,
        to='int'
    )
    print("Employment")
    print(employment)
    employment.rename(
        {"msoa_zone_id": "model_zone_id"},
        axis=1,
        inplace=True)
    employment = du.convert_msoa_naming(
        employment,
        msoa_col_name="model_zone_id",
        msoa_path=msoa_lookup_path,
        to='int'
    )

    # If the zone translator has been supplied, need to change zone system
    if zone_translator is not None:

        population = zone_translator.run(
            population,
            non_split_columns=segments["pop"],
            **zone_translator_args
        )

        employment = zone_translator.run(
            employment,
            non_split_columns=segments["emp"],
            **zone_translator_args
        )

        if exceptional_zones is not None:
            converted_e_zones = zone_translator.run(
                exceptional_zones,
                non_split_columns=["model_zone_id"],
                **zone_translator_args
            )
        else:
            converted_e_zones = pd.DataFrame(columns=["model_zone_id"])

    print("Converted Population")
    print(population)
    print("Employment")
    print(employment)
    print("Obs Productions")
    print(observed_productions)
    print("Obs Attractions")
    print(observed_attractions)
    print("Exceptional Zones")
    print(converted_e_zones)

    # ## Calculate Trip Rates ## #
    observed_prod_base = observed_productions[segments["prod"] + [base_year]]
    observed_attr_base = observed_attractions[segments["attr"] + [base_year]]

    prod_trip_rates = production_exceptional_trip_rate(
        observed_base=observed_prod_base,
        land_use=population,
        e_zones=pd.DataFrame,
        base_year=base_year,
        segment_cols=prod_segments,
        zone_column="model_zone_id",
        purpose_column="purpose_id",
        sector_lookup=trip_rate_sectors
    )
    attr_trip_rates = attraction_exceptional_trip_rate(
        observed_base=observed_attr_base,
        land_use=employment,
        e_zones=exceptional_zones,
        base_year=base_year,
        segment_cols=attr_segments,
        zone_column="model_zone_id",
        purpose_column="purpose_id",
        sector_lookup=trip_rate_sectors,
        msoa_lookup_path=msoa_lookup_path
    )

    # Setup population segmentation for growth criteria
    population = convert_pop_segmentation(
        population,
        grouping_cols=segments["pop"],
        value_cols=future_years
    )

    # ## Apply Growth Criteria ## #
    grown_productions = {}
    grown_attractions = {}

    synth_prod_base = synth_productions[segments["prod"] + [base_year]]
    synth_attr_base = synth_attractions[segments["attr"] + [base_year]]
    
    grown_productions[base_year] = synth_prod_base
    grown_attractions[base_year] = synth_attr_base

    # Calculate separately for each year and combine at the end
    for year in future_years:
        synth_prod_subset = synth_productions[segments["prod"] + [year]]
        synth_attr_subset = synth_attractions[segments["attr"] + [year]]

        pop_subset = population[segments["pop"] + [year]]
        emp_subset = employment[segments["emp"] + [year]]
        emp_subset = emp_subset.loc[
            emp_subset["employment_cat"] == "E01"
        ].drop("employment_cat", axis=1)

        grown_productions[year] = handle_exceptional_growth(
            synth_future=synth_prod_subset,
            synth_base=synth_prod_base,
            observed_base=observed_prod_base,
            zone_column="model_zone_id",
            segment_columns=prod_segments,
            value_column=year,
            base_year=base_year,
            exceptional_zones=converted_e_zones,
            land_use=pop_subset,
            trip_rates=prod_trip_rates,
            sector_lookup=trip_rate_sectors
        )
        grown_attractions[year] = handle_exceptional_growth(
            synth_future=synth_attr_subset,
            synth_base=synth_attr_base,
            observed_base=observed_attractions,
            zone_column="model_zone_id",
            segment_columns=attr_segments,
            value_column=year,
            base_year=base_year,
            exceptional_zones=converted_e_zones,
            land_use=emp_subset,
            trip_rates=attr_trip_rates,
            sector_lookup=trip_rate_sectors
        )

    # Combine forecast vectors
    converted_productions = pd.DataFrame()
    converted_pure_attractions = pd.DataFrame()
    for year in grown_productions.keys():
        prod = grown_productions[year][segments["prod"] + [year]]
        attr = grown_attractions[year][segments["attr"] + [year]]
        if converted_productions.empty:
            converted_productions = prod
            converted_pure_attractions = attr
        else:
            converted_productions = pd.merge(
                converted_productions,
                prod,
                on=segments["prod"]
            )
            converted_pure_attractions = pd.merge(
                converted_pure_attractions,
                attr,
                on=segments["attr"]
            )
    converted_productions.to_csv("grown_productions.csv", index=False)
    converted_pure_attractions.to_csv("grown_attractions.csv", index=False)

    return (converted_productions, converted_pure_attractions)


def extract_donor_totals(matrix_path: str,
                         sectors: pd.DataFrame,
                         tour_proportions: pd.DataFrame = None
                         ) -> Tuple[pd.DataFrame, pd.DataFrame]:

    df = pd.read_csv(matrix_path, index_col=0)
    agg_tour_props = pd.DataFrame()
    
    # Convert to OD if tour proportions are supplied
    if tour_proportions is not None:
        # Check that the shapes match
        if tour_proportions.shape != df.shape:
            raise ValueError("Shape Mismatch in tour proportions")
        # Tour proportions are the ratio of from home to to home
        # If these are 0.5, equal amounts of both will be used
        od_df = (
            (2 * df.values * tour_proportions.values)
            + (2 * df.values.T * (1 - tour_proportions.values))
        )
        od_df = pd.DataFrame(
            od_df,
            index=df.index,
            columns=df.columns
        )
        # Extract the aggregated tour proportions for the sectors
        tp_o = tour_proportions.mean(axis=1)
        tp_d = tour_proportions.T.mean(axis=1)
        tp_d.index = tp_d.index.astype("int")
        tp_d.index.name = tp_o.index.name

        agg_tour_props = pd.DataFrame({"origins": tp_o,
                                       "dests": tp_d})
        agg_tour_props = pd.merge(
            sectors,
            agg_tour_props,
            left_on="Zone",
            right_index=True
        )
        agg_tour_props = agg_tour_props.groupby(
            "Sector ID"
        )[["origins", "dests"]].mean()
        agg_tour_props.rename(
            {"Sector ID": "Donor Sector ID"},
            axis=1,
            inplace=True
        )
    else:
        od_df = df.copy()
    
    origins = od_df.sum(axis=1)
    destinations = od_df.T.sum(axis=1)
    destinations.index = destinations.index.astype("int")
    destinations.index.name = origins.index.name

    totals = pd.DataFrame({"origins": origins,
                           "dests": destinations})

    donor_totals = pd.merge(
        sectors,
        totals,
        left_on="Zone",
        right_index=True
    )

    donor_totals = donor_totals.groupby(
        "Sector ID"
    )[["origins", "dests"]].sum()
    donor_totals.index.name = "Donor Sector ID"

    return donor_totals, agg_tour_props


def calculate_tour_proportions(od_matrix_base: str,
                               fill_val: float = 0.5
                               ) -> pd.DataFrame:
    
    from_24 = pd.DataFrame()
    to_24 = pd.DataFrame()
    
    for tp in consts.TP_NEEDED:
        from_matrix_path = od_matrix_base.format("from", tp)
        to_matrix_path = od_matrix_base.format("to", tp)
        
        from_df = pd.read_csv(from_matrix_path, index_col=0)
        to_df = pd.read_csv(to_matrix_path, index_col=0)
        
        if from_24.empty:
            from_24 = from_df
            to_24 = to_df
        else:
            from_24 += from_df
            to_24 += to_df

    # Ignore errors where the numerator and denominator are both 0 (raise 
    # exception if only denominator is 0) - keeps console tidy
    with np.errstate(invalid="ignore", divide="raise"):
        tour_props = from_24.values / (from_24.values + to_24.T.values)
    tour_props = pd.DataFrame(
        tour_props,
        index=from_24.index,
        columns=from_24.columns
    )
    # Remove NaN values where from_24 + to_24.T was 0
    tour_props.fillna(fill_val, inplace=True)

    return tour_props


def get_donor_zone_data(sectors: pd.DataFrame,
                        export_paths: dict,
                        nhb_segmented: bool,
                        ca_needed: bool
                        ) -> Tuple[pd.DataFrame, pd.DataFrame]:

    pa_path = export_paths["pa_24"]
    od_path = export_paths["od"]
    
    # Check if OD matrices are available for tour proportions
    ods_available = len(os.listdir(od_path)) > 0

    # ## Build HB totals ## #
    hb_purps = consts.PURPOSES_NEEDED
    socs = consts.SOC_NEEDED
    ns = consts.NS_NEEDED
    # For Noham, assume CA_NEEDED should be None
    cas = consts.CA_NEEDED if ca_needed else ["None"]
    modes = consts.MODES_NEEDED
    year = consts.BASE_YEAR

    hb_donor_data = pd.DataFrame()
    agg_tour_props = pd.DataFrame()

    desc = "Getting HB donor zone data"
    message = ""
    iter_hb = tqdm(list(product(hb_purps, modes, cas)), desc=desc)
    for purp, mode, ca in iter_hb:
        if purp in consts.SOC_P:
            segments = socs
        elif purp in consts.NS_P:
            segments = ns
        for segment in segments:
            desc_string = f"p_{purp}, m_{mode}, ca_{ca}, seg_{segment}"
            iter_hb.set_description(
                desc_string + " : " + message
            )
            matrix_name = du.get_dist_name(
                trip_origin="hb",
                matrix_format="pa",
                year=str(year),
                purpose=str(purp),
                mode=str(mode),
                segment=str(segment),
                car_availability=str(ca),
                tp=None,
                csv=True
            )
            matrix_path = os.path.join(pa_path, matrix_name)
            if ods_available:
                # Get 24hr OD tour proportions
                od_matrix_base = du.get_dist_name(
                    trip_origin="hb",
                    matrix_format="od_{}",
                    year=str(year),
                    purpose=str(purp),
                    mode=str(mode),
                    segment=str(segment),
                    car_availability=str(ca),
                    tp="{}",
                    csv=True
                )
                od_matrix_path = os.path.join(od_path, od_matrix_base)
                tour_props = calculate_tour_proportions(od_matrix_path)
                message = "Calculating Tour Proportions"
            else:
                # If OD matrices are not available - give a warning and use 
                # default 0.5 for all
                tour_props = pd.read_csv(matrix_path, index_col=0)
                for col in tour_props.columns:
                    tour_props[col].values[:] = 0.5
                message = "Warning: Using default Tour Proportions of 0.5"
            # Extract the origin and destinations for each donor sector
            donor_totals, agg_tp = extract_donor_totals(
                matrix_path,
                sectors,
                tour_proportions=tour_props
            )
            # Add segmentation columns
            donor_totals["Purpose"] = purp
            donor_totals["segment"] = segment
            donor_totals["mode"] = mode
            donor_totals["ca"] = ca

            agg_tp["Purpose"] = purp
            agg_tp["segment"] = segment
            agg_tp["mode"] = mode
            agg_tp["ca"] = ca

            if hb_donor_data.empty:
                hb_donor_data = donor_totals
                agg_tour_props = agg_tp
            else:
                hb_donor_data = hb_donor_data.append(donor_totals)
                agg_tour_props = agg_tour_props.append(agg_tp)

    # ## Build HB totals ## #
    nhb_purps = consts.ALL_NHB_P

    nhb_donor_data = pd.DataFrame()

    desc = "Getting NHB donor zone data"
    if segment_employment:
        iter_nhb = tqdm(list(product(nhb_purps, modes, cas)), desc=desc)
    else:
        iter_nhb = tqdm(list(product(nhb_purps, modes, [None])), desc=desc)
    for purp, mode, ca in iter_nhb:
        if not segment_employment:
            segments = [999]
        elif purp in consts.SOC_P:
            segments = socs
        # Not elif as NS_P not defined for NHB
        else:
            segments = ns
        for segment in segments:
            iter_nhb.set_description(
                f"p_{purp}, m_{mode}, ca_{ca}, seg_{segment}"
            )
            matrix_name = du.get_dist_name(
                trip_origin="nhb",
                matrix_format="pa",
                year=str(year),
                purpose=str(purp),
                mode=str(mode),
                segment=str(segment) if segment_employment else None,
                car_availability=str(ca),
                tp=None,
                csv=True
            )
            matrix_path = os.path.join(pa_path, matrix_name)
            # Extract the origin and destinations for each donor sector
            donor_totals, _ = extract_donor_totals(matrix_path, sectors)
            # Add segmentation columns
            donor_totals["Purpose"] = purp
            donor_totals["segment"] = segment
            donor_totals["mode"] = mode
            donor_totals["ca"] = ca

            if nhb_donor_data.empty:
                nhb_donor_data = donor_totals
            else:
                nhb_donor_data = nhb_donor_data.append(donor_totals)

    hb_donor_data["trip_origin"] = "hb"
    nhb_donor_data["trip_origin"] = "nhb"
    donor_data = hb_donor_data.append(nhb_donor_data)

    return donor_data, agg_tour_props


def _replace_generation_segments(generation_data: pd.DataFrame,
                                 purpose_data: pd.DataFrame,
                                 segmented_nhb: bool,
                                 ca_needed: bool):
    
    gen_data = generation_data.copy()
    
    if not ca_needed:
        # Combine any ca segmentation in the generation data 
        # - likely not needed for most inputs
        group_cols = [
            col for col in gen_data.columns 
            if col not in ["Volume", "TfN Segmentation - ca"]
        ]
        gen_data = gen_data.groupby(
            group_cols,
            as_index=False
        )["Volume"].sum()
        # Add the dummy value all ca segmentation
        gen_data["TfN Segmentation - ca"] = 999
    
    # Convert given purpose to EFS purposes
    try:
        gen_data = pd.merge(
            gen_data,
            purpose_data,
            on="Purpose ID",
            validate="m:m"
        )
        # Split purposes
    except pd.errors.MergeError:
        print("Purposes are already in EFS format")
        gen_data["Purpose"] = gen_data["Purpose ID"]
    
    # Add the segmentation if required
    # 999 is used in the inputs to represent the aggregated segment
    socs = pd.DataFrame(
        [[999, p, seg] for p, seg in product(consts.SOC_P, consts.SOC_NEEDED)],
        columns=["TfN Segmentation - soc", "Purpose", "soc"]
    )
    gen_data = pd.merge(
        gen_data,
        socs,
        on=["TfN Segmentation - soc", "Purpose"],
        how="left"
    )
    hb_nhb_ns_p = [p for p in consts.ALL_P if p not in consts.SOC_P]
    ns = pd.DataFrame(
        [[999, p, seg] for p, seg in product(hb_nhb_ns_p, consts.NS_NEEDED)],
        columns=["TfN Segmentation - ns", "Purpose", "ns"]
    )
    gen_data = pd.merge(
        gen_data,
        ns,
        on=["TfN Segmentation - ns", "Purpose"],
        how="left"
    )
    if ca_needed:
        cas = pd.DataFrame(
            [[999,  seg] for seg in consts.CA_NEEDED],
            columns=["TfN Segmentation - ca", "ca"]
        )
        gen_data = pd.merge(
            gen_data,
            cas,
            on=["TfN Segmentation - ca"],
            how="left"
        )
        # Replace 999 values with the given segmentation
        gen_data["ca"] = gen_data["ca"].fillna(
            gen_data["TfN Segmentation - ca"]
        ).astype("int")
    else:
        gen_data["ca"] = "None"
    # Build the segment column using the hierarchy of aggregated first
    gen_data["segment"] = gen_data["soc"].fillna(
        gen_data["ns"]).fillna(
            gen_data["TfN Segmentation - soc"]).fillna(
                gen_data["TfN Segmentation - ns"]
            ).astype("int")
    # Replace the values for nhb purposes with 999
    if not segmented_nhb:
        gen_data.loc[
            gen_data["Purpose"].isin(consts.NHB_PURPOSES_NEEDED), "segment"
            ] = 999
        gen_data.loc[
            gen_data["Purpose"].isin(consts.NHB_PURPOSES_NEEDED), "ca"
            ] = 999
    gen_data.drop(
        ["soc",
         "ns"],
        axis=1,
        inplace=True
    )
    gen_data.drop_duplicates(inplace=True)
    
    return gen_data

def _apply_underlying_segment_splits(generation_data: pd.DataFrame,
                                     donor_data: pd.DataFrame
                                     ) -> pd.DataFrame:
    
    ca_needed = "None" not in generation_data["ca"].unique()
    
    df = generation_data.copy()
    
    # Merge generation data to the donor_data to split where required
    split_data = pd.merge(
        df,
        donor_data,
        on=["Donor Sector ID", "Purpose", "segment", "ca"],
        how="left"
    )
    segment_cols = [
        "TfN Segmentation - soc",
        "TfN Segmentation - ns",
        "TfN Segmentation - ca"
    ]
    if not ca_needed:
        segment_cols.remove("TfN Segmentation - ca")
    group_cols = ["Year",
                  "Purpose ID",
                  "Direction"] + segment_cols
    split_data["o_totals"] = split_data.groupby(
        group_cols,
        as_index=False
    )["origins"].transform(sum)
    split_data["d_totals"] = split_data.groupby(
        group_cols,
        as_index=False
    )["dests"].transform(sum)
    split_data.loc[split_data["Direction"] == 1, "proportion"] = (
        split_data["origins"] / split_data["o_totals"]
    )
    split_data.loc[split_data["Direction"] == 2, "proportion"] = (
        split_data["dests"] / split_data["d_totals"]
    )
    split_data["split_volume"] = split_data["Volume"] * split_data["proportion"]
    
    # Drop intermediate columns
    split_data.drop(
        ["Purpose ID",
         "origins",
         "dests",
         "o_totals",
         "d_totals",
         "proportion"],
        axis=1,
        inplace=True
    )
    split_data.drop(
        segment_cols,
        axis=1,
        inplace=True
    )
    
    return split_data


def _apply_sector_distribution(segment_split_data: pd.DataFrame,
                               distribution_data: pd.DataFrame
                               ) -> pd.DataFrame:
    
    split_data = segment_split_data.copy()
    
    # Assign a unique ID to each row
    split_data["dist_id"] = split_data.reset_index().index.values
    # Apply sector distribution from Distribution ID
    sector_dist = pd.merge(
        split_data,
        distribution_data,
        on=["Distribution ID"],
        how="left"
    )
    # Use the Proportion column to get the distribution splits
    sector_dist["dist_volume"] = (
        sector_dist["split_volume"]
        * sector_dist["Proportion"]
        / sector_dist.groupby(["dist_id"])["Proportion"].transform(sum)
    )
    # Drop intermediate columns
    sector_dist.drop(
        ["dist_id", "Proportion", "split_volume", "Distribution ID"],
        axis=1,
        inplace=True
    )
    
    return sector_dist


def _convert_od_to_trips(sector_distributed_data: pd.DataFrame,
                         aggregated_tour_proportions: pd.DataFrame
                         ) -> pd.DataFrame:
    
    ca_needed = "None" not in sector_distributed_data["ca"].unique()
    
    sector_dist = sector_distributed_data.copy()
    agg_tour_props = aggregated_tour_proportions.copy()
    
    # Split from home / to home
    # Reset index to set Sector ID as a column
    agg_tour_props.reset_index(inplace=True)
    id_vars = ["Sector ID", "Purpose", "segment", "mode", "ca"]
    if not ca_needed:
        id_vars.remove("ca")
    agg_tour_props = agg_tour_props.melt(
        id_vars=["Sector ID", "Purpose", "segment", "mode", "ca"],
        value_vars=["origins", "dests"],
        var_name="Direction",
        value_name="tour_proportion"
    )
    agg_tour_props["Direction"].replace(
        {"origins": 1,
            "dests": 2},
        inplace=True
    )
    agg_tour_props.rename(
        {"Sector ID": "Donor Sector ID"},
        axis=1,
        inplace=True
    )
    # Join to just the relevant rows - HB purposes
    converted_trips = pd.merge(
        sector_dist,
        agg_tour_props,
        on=["Donor Sector ID", "Direction", "Purpose", 
            "segment", "mode", "ca"],
        how="inner"
    )
    # Calculation to convert to productions and attractions
    converted_trips["prod"] = (
        converted_trips["dist_volume"]
        * converted_trips["tour_proportion"]
        / 2
    )
    converted_trips["attr"] = (
        converted_trips["dist_volume"]
        * (1 - converted_trips["tour_proportion"])
        / 2
    )
    merge_cols = ["Donor Sector ID", "Sector ID", "Direction", "Year", 
                  "Purpose", "segment", "mode", "ca"]
    converted_trips = pd.merge(
        sector_dist,
        converted_trips[merge_cols + ["prod", "attr"]],
        on=merge_cols,
        how="left"
    )
    
    return converted_trips


def _build_addition_matrix(filtered_trips: pd.DataFrame,
                           sector_data: pd.DataFrame,
                           old_trips: np.array,
                           purpose: int
                           ) -> np.array:

    # Build new matrix to combine with existing
    add_trips = np.zeros_like(old_trips)
    # Add the new volumes to the relevant zones
    for row_dict in filtered_trips.to_dict(orient="records"):
        # Check if intrazonals should be included
        include_intrazonals = row_dict["Include / Exclude Intrazonals"] == 1
        # Get the zone IDs
        bespoke_zone = row_dict["Zone ID"]
        zones = sector_data.loc[
            sector_data["Sector ID"] == row_dict["Sector ID"]
        ]["Zone"].values
        # Remove the intrazonal if necessary
        if not include_intrazonals:
            zones = zones[zones != bespoke_zone]
        # Raise an error if this means there is nowhere to distribute
        if zones.size == 0:
            raise ValueError("Error: Intrazonals excluded when "
                             "distributing to self")
        # Convert the zones to matrix indices (offset by one)
        bespoke_zone_idx = bespoke_zone - 1
        zone_idxs = zones - 1
        # Distribute using the underlying distribution in that sector
        row_dist = old_trips[bespoke_zone_idx, zone_idxs]
        col_dist = old_trips[zone_idxs, bespoke_zone_idx]
        if purpose in consts.ALL_HB_P:
            # For HB - add both productions and attractions
            add_trips[bespoke_zone_idx, zone_idxs] += (
                row_dict["prod"]
                * row_dist
                / row_dist.sum()
            )
            add_trips[zone_idxs, bespoke_zone_idx] += (
                row_dict["attr"]
                * col_dist
                / col_dist.sum()
            )
        elif row_dict["Direction"] == 1:
            # For NHB - use the direction and add the ODs
            add_trips[bespoke_zone_idx, zone_idxs] += (
                row_dict["dist_volume"]
                * row_dist
                / row_dist.sum()
            )
        elif row_dict["Direction"] == 2:
            add_trips[zone_idxs, bespoke_zone_idx] += (
                row_dict["dist_volume"]
                * col_dist
                / col_dist.sum()
            )
        else:
            raise ValueError("Invalid Purpose or Direction")
    return add_trips


def _constrain_to_sector_total(trip_matrix: np.array,
                               bespoke_trip_matrix: np.array,
                               constraint_zones: np.array,
                               bespoke_zone: int,
                               minimum_reduction: float = 0.25,
                               constrain: str = "origin"
                               ) -> np.array:
    
    final_trip_matrix = trip_matrix.copy()
    
    # Create masks to access the constraint area
    sector_mask = np.zeros_like(trip_matrix, dtype=bool)
    if constrain == "origin":
        sector_mask[bespoke_zone, constraint_zones] = True
    elif constrain == "destination":
        sector_mask[constraint_zones, bespoke_zone] = True
    c_mask = (bespoke_trip_matrix == 0) & sector_mask
    add_mask = bespoke_trip_matrix != 0
    
    # Calculate the target (sector total - original trips - bespoke trips)
    target = (
        trip_matrix[sector_mask].sum()
        - trip_matrix[add_mask].sum()
        - bespoke_trip_matrix[add_mask].sum()
    )
    # Get the original number of trips in the constraint area to check that 
    # they are not being reduced too much
    start_trips = trip_matrix[c_mask].sum()
    if target <= minimum_reduction * start_trips:
        target = minimum_reduction * start_trips
    
    # Factor all zones that have not had bespoke trips added so that the 
    # totals are consistent
    final_trip_matrix[c_mask] = target * trip_matrix[c_mask] / start_trips
    
    # Finally, add the bespoke trips
    final_trip_matrix[add_mask] += bespoke_trip_matrix[add_mask]
    
    return final_trip_matrix
    

def _apply_to_bespoke_zones(converted_trips: pd.DataFrame,
                            sector_data: pd.DataFrame,
                            export_path: str,
                            segmented_nhb: bool
                            ) -> Tuple[pd.DataFrame, pd.DataFrame]:

    ca_needed = "None" not in converted_trips["ca"].unique()

    additions = []
    skipped = []
    year_list = converted_trips["Year"].unique()
    socs = consts.SOC_NEEDED
    ns = consts.NS_NEEDED
    # Use all CA values - will need to skip for NOHAM
    cas = consts.CA_NEEDED if ca_needed else ["None"]
    purps = converted_trips["Purpose"].unique()
    modes = converted_trips["mode"].unique()
    gen_zones = converted_trips["Zone ID"].unique()
    if len(modes) != 1:
        raise ValueError("Only one mode is supported at once")
    segment_combs = tqdm(list(product(year_list, purps, cas, modes, gen_zones)))
    for year, purp, ca, mode, gen_zone in segment_combs:
        if purp in consts.ALL_NHB_P and not segmented_nhb:
            if ca == 2:
                continue
            segments = [999]
            ca = 999
        elif purp in consts.SOC_P:
            segments = socs
        elif purp in consts.NS_P:
            segments = ns
        for segment in segments:
            # Load the original matrix
            trip_origin = "hb" if purp in consts.ALL_HB_P else "nhb"
            segment_str = str(segment) if segmented_nhb else None
            ca_str = str(ca) if segmented_nhb else None
            matrix_name = du.get_dist_name(
                trip_origin=trip_origin,
                matrix_format="pa",
                year=str(year),
                purpose=str(purp),
                mode=str(mode),
                segment=segment_str,
                car_availability=ca_str,
                tp=None,
                csv=True
            )
            matrix_path = os.path.join(export_path, matrix_name)
            try:
                trips_df = pd.read_csv(matrix_path, index_col=0)
            except FileNotFoundError:
                # Skip if data was provided for matrices that don't exist
                skipped.append([year, purp, mode, ca, segment])
                continue
            trips = trips_df.values
            # Build dictionary of the additional productions / attractions
            filter_str = (
                "`Zone ID` == @gen_zone and "
                "Year == @year and "
                "Purpose == @purp and "
                "segment == @segment and "
                "ca == @ca"
            )
            filtered_trips = converted_trips.query(
                filter_str
            )
            filtered_trips = filtered_trips[
                ["Zone ID", "Sector ID", "Include / Exclude Intrazonals", 
                 "Direction", "Constraint Type", "Constraint Sector ID",
                 "dist_volume", "prod", "attr"]
            ]
            if filtered_trips.empty:
                raise ValueError("Fatal Error: No generators found")

            constraint_type = filtered_trips["Constraint Type"].unique()
            if len(constraint_type) > 1:
                print(filtered_trips)
                raise ValueError("Error: Inconsistent Constraint Type")
            constraint_type = constraint_type[0]
            
            add_trips = _build_addition_matrix(
                filtered_trips,
                sector_data,
                trips,
                purp
            )
            
            additions.append([
                year, purp, mode, ca, segment, trips.sum(), add_trips.sum()
            ])
            segment_combs.set_description(
                f"yr_{year}, p_{purp}, m_{mode}, ca_{ca}, seg_{segment} "
                f"- Added {round(add_trips.sum(), 2)}"
            )
            # Constraint Type 0 - Add Trips to existing
            if constraint_type == 0:
                trips += add_trips
            # Constraint Type 1 - Replace existing trips
            elif constraint_type == 1:
                mask = add_trips != 0
                trips[mask] = add_trips[mask]
            # Constraint Type 2 - Constrain to zone / sector total
            elif constraint_type == 2:
                # TODO Test this further - is constraint method correct?
                # There should be just one constraint sector id
                c_sector = filtered_trips["Constraint Sector ID"].unique()
                if len(c_sector) > 1:
                    raise ValueError("Error: Inconsistent Constraint Sector")
                c_sector = c_sector[0]
                # Get an array of the zones used in the constraint
                c_zones = sector_data.loc[sector_data["Sector ID"] == c_sector]
                c_zones = c_zones["Zone"].values - 1
                bespoke_zone = gen_zone - 1
                trips = _constrain_to_sector_total(trips, add_trips, c_zones, 
                                                   bespoke_zone)
            else:
                raise ValueError("Invalid Constraint Type, ", constraint_type)
            
            # Save to new path or overwrite
            new_matrix_path = matrix_path.replace(".csv", "_bespoke.csv")
            new_trips_df = pd.DataFrame(
                trips,
                index=trips_df.index,
                columns=trips_df.columns
            )
            new_trips_df.to_csv(new_matrix_path)
    additions = pd.DataFrame(
        additions,
        columns=["Year", "Purp", "Mode", "CA", "Segment", "Old Trips", 
                 "Additional Trips"]
    )
    skipped = pd.DataFrame(
        skipped,
        columns=["Year", "Purp", "Mode", "CA", "Segment"]
    )
    return additions, skipped


def check_bespoke_input(gen_data: pd.DataFrame,
                        purp_data: pd.DataFrame,
                        sector_data: pd.DataFrame,
                        sector_def_data: pd.DataFrame,
                        dist_data: pd.DataFrame
                        ) -> None:
    def check_valid(check_vals: pd.Series,
                    valid_vals: Iterable,
                    message: str = None):
        """
        Checks if any of check_vals are not in valid_vals
        - return True if not valid
        """
        check = check_vals.unique()
        if isinstance(valid_vals, pd.Series):
            valid = valid_vals.unique()
        else:
            valid = valid_vals
        if any([x not in valid for x in check]):
            if message is not None:
                raise ValueError(message)
            else:
                return True
        return False

    # Check for duplicates
    if gen_data.duplicated().any():
        raise ValueError("Error: Duplicate Rows Exist in Generation Sheet")

    # Check each zone ID exists in norms/noham
    check_valid(
        gen_data["Zone ID"],
        sector_data["Zone"],
        "Error: Bespoke Zone IDs do not exist in Model Zones"
    )

    # Check origin and destinations are both defined for each zone
    group_cols = [col for col in gen_data.columns
                  if "TfN" in col or
                  col in ["Purpose ID", "Generator ID", "Zone ID", "Year"]]
    direction_counts = gen_data.groupby(group_cols)["Direction"].count().values
    if any([d_count != 2 for d_count in direction_counts]):
        raise ValueError("Error: Supply both directions for each generator")

    # Check all unique purpose ids exist in lookup - check they are in the
    # same group e.g. <100, <200. Detect if splits will need to be done
    check_valid(
        gen_data["Purpose ID"],
        purp_data["Purpose ID"],
        "Error: Undefined Purpose ID supplied"
    )
    u_purps = gen_data[["Purpose ID"]].drop_duplicates()
    if u_purps.merge(purp_data)["Purpose"].duplicated().any():
        print("Warning: Overlapping purpose IDs supplied ")

    # Check that each of soc, ns, ca are valid, either all explicitly defined
    # (none missing) or based on underlying data. Check that they are valid
    # combinations e.g. purpose 1 only has soc
    soc_not_defined = check_valid(gen_data["TfN Segmentation - soc"],
                                  consts.SOC_NEEDED + [999])
    ns_not_defined = check_valid(gen_data["TfN Segmentation - ns"],
                                 consts.NS_NEEDED + [999])
    ca_not_defined = check_valid(gen_data["TfN Segmentation - ca"],
                                 consts.CA_NEEDED + [999])
    if (soc_not_defined and ns_not_defined and ca_not_defined):
        raise ValueError("Error: Segmentation is not valid")

    # Check sector ID exists in sector system
    check_valid(
        gen_data["Donor Sector ID"],
        sector_data["Sector ID"],
        "Error: Define all donor sectors"
    )
    constraint_secs = gen_data.loc[gen_data["Constraint Type"] != 0]
    check_valid(
        constraint_secs["Constraint Sector ID"],
        sector_data["Sector ID"],
        "Error: Define all constraint sectors"
    )

    # Check all distribution ids exist
    check_valid(
        gen_data["Distribution ID"],
        dist_data["Distribution ID"],
        "Error: Define all Distribution IDs"
    )

    # Check for same distribution - sector systems
    dest_sectors_check = dist_data.merge(sector_def_data, on="Sector ID")
    dest_sectors_check = dest_sectors_check.groupby("Distribution ID")
    if not all(dest_sectors_check["Sector System ID"].nunique() == 1):
        raise ValueError("Error: Distributions must use a single "
                         "sector system")

    # Check intrazonal ids are valid
    check_valid(
        gen_data["Include / Exclude Intrazonals"],
        [1, 2],
        "Error: Intrazonal ID must be 1 or 2"
    )

    # Check constraint Ids are valid
    check_valid(
        gen_data["Constraint Type"],
        [0, 1, 2],
        "Error: Constraint Type must be 0, 1, or 2"
    )
    
def test_bespoke_zones(gen_path: str,
                       exports_path: str,
                       model_name: str,
                       audit_path: str,
                       recreate_donor: bool = True,
                       nhb_segmented: bool = True
                       ):
    
    if model_name == "norms_2015" or model_name == "norms":
        model_suffix = "NoRMS"
        ca_needed = True
    elif model_name == "noham":
        model_suffix = "NoHAM"
        ca_needed = False
    else:
        raise ValueError(f"Model Type {model_name} is not supported")
    
    # Load Generation Data
    bespoke_dict = pd.read_excel(gen_path, engine="openpyxl", sheet_name=None)
    gen_data = bespoke_dict[f"Generation Data {model_suffix}"]
    purp_data = bespoke_dict[f"Purpose Data"]
    sector_data = bespoke_dict[f"Sector Data {model_suffix}"]
    sector_def_data = bespoke_dict[f"Sector Definition {model_suffix}"]
    dist_data = bespoke_dict[f"Distribution Data {model_suffix}"]

    # ## Error Checking ## #
    print("Checking for input errors")
    check_bespoke_input(
        gen_data,
        purp_data,
        sector_data,
        sector_def_data,
        dist_data
    )
    print("Input ok")

    # ## Prepare and Infill data ## #
    # Fetch matrix data at max segmentation for all donor sectors
    if recreate_donor:
        print("Calculating donor sector splits")
        donor_sectors = gen_data["Donor Sector ID"].unique()
        sector_lookup = sector_data.loc[
            sector_data["Sector ID"].isin(donor_sectors)
        ]
        donor_data, agg_tour_props = get_donor_zone_data(
            sector_lookup,
            exports_path,
            nhb_segmented,
            ca_needed
        )
        
        # donor_data.to_csv(os.path.join(audit_path, "donor_test.csv"))
        # agg_tour_props.to_csv(os.path.join(audit_path, "tp_test.csv"))
    else:
        donor_data = pd.read_csv(os.path.join(audit_path, "donor_test.csv"))
        agg_tour_props = pd.read_csv(os.path.join(audit_path, "tp_test.csv"))
    
    # Convert the segmentation to the EFS segments to split the bespoke 
    # zone data
    print("Splitting bespoke segments")
    gen_data = _replace_generation_segments(
        gen_data, 
        purp_data, 
        nhb_segmented,
        ca_needed
    )
    
    # gen_data.to_csv(os.path.join(audit_path, "gen_test.csv"))
    
    # Apply the underlying segment splits where required
    print("Applying donor sector splits")
    split_data = _apply_underlying_segment_splits(
        gen_data,
        donor_data
    )
    
    # split_data.to_csv(os.path.join(audit_path, "split_test.csv"), index=False)
    
    # ## Distribution ## #
    print("Applying user distribution")
    sector_dist = _apply_sector_distribution(split_data, dist_data)
    
    sector_dist.to_csv(os.path.join(audit_path, "dist_test.csv"), index=False)
    
    # Convert HB purposes into productions/attractions using tour proportions
    print("Converting to PAs")
    converted_trips = _convert_od_to_trips(
        sector_dist,
        agg_tour_props
    )
    
    # converted_trips.to_csv(os.path.join(audit_path, "converted_test.csv"))
    
    # ## Combine with existing matrices ## #
    # Build list of all segmentations
    print("Combining with existing matrices")
    additions, skipped = _apply_to_bespoke_zones(
        converted_trips,
        sector_data,
        exports_path["pa_24"],
        nhb_segmented
    )
    
    print(f"Skipped {skipped.shape[0]} matrices - see log file")
    
    bespoke_audit_path = os.path.join(
        audit_path, "Bespoke Zones"
    )
    if not os.path.isdir(bespoke_audit_path):
        os.mkdir(bespoke_audit_path)
    additions.to_csv(
        os.path.join(bespoke_audit_path, "bespoke_zone_additions.csv")
    )
    skipped.to_csv(
        os.path.join(bespoke_audit_path, "bespoke_zone_skipped_matrices.csv")
    )


def test_growth_criteria():
    # Test productions

    print("Loading files")

    zt = ZoneTranslator()

    population = None
    employment = None
    synthetic_p = None
    synthetic_e = None
    obs_base_e = None
    obs_base_p = None

    seg_cols_p = ["purpose_id", "soc", "ns", "car_availability_id"]
    seg_cols_e = ["purpose_id", "soc"]
    zone_col = "model_zone_id"
    base_year = "2018"

    synthetic_p = pd.read_csv(
        r"C:\NorMITs Demand\norms\v2_3-EFS_Output\iter0\Productions\MSOA"
        r"_aggregated_productions.csv"
    )

    population = pd.read_csv(r"C:\NorMITs Demand\norms\v2_3-EFS_Output\iter0"
                             r"\Productions\MSOA_population.csv")

    population.rename(
        {"msoa_zone_id": "model_zone_id",
         "ca": "car_availability_id"},
        axis=1,
        inplace=True)

    population = du.convert_msoa_naming(
        population,
        msoa_col_name=zone_col,
        msoa_path=r"Y:\NorMITs Demand\inputs\default\zoning\msoa_zones.csv",
        to='int'
    )

    obs_base_p = synthetic_p[[zone_col] + seg_cols_p + [base_year]]

    synthetic_e = pd.read_csv(
        r"C:\NorMITs Demand\norms\v2_3-EFS_Output\iter1\Attractions"
        r"\MSOA_aggregated_attractions.csv"
    )
    employment = pd.read_csv(r"C:\NorMITs Demand\norms\v2_3-EFS_Output\iter1"
                             r"\Attractions\MSOA_employment.csv")
    employment.rename(
        {"msoa_zone_id": "model_zone_id"},
        axis=1,
        inplace=True)
    employment = du.convert_msoa_naming(
        employment,
        msoa_col_name=zone_col,
        msoa_path=r"Y:\NorMITs Demand\inputs\default\zoning\msoa_zones.csv",
        to='int'
    )

    exceptional_zones = pd.read_csv(
        r"C:\NorMITs Demand\norms\v2_3-EFS_Output\iter1\Productions\exceptional_zones.csv"
    ).rename({"msoa_zone_id": "model_zone_id"}, axis=1)

    output_path = os.path.join(
        r"Y:\NorMITs Demand\inputs\default\zoning\norms_2015.csv")
    translation_dataframe = pd.read_csv(output_path)

    exceptional_zones = zt.run(
        exceptional_zones,
        translation_dataframe,
        "MSOA",
        "norms_2015",
        non_split_columns=["model_zone_id"]
    )

    population = zt.run(
        population,
        translation_dataframe,
        "MSOA",
        "norms_2015",
        non_split_columns=["model_zone_id",
                           "car_availability_id",
                           "soc",
                           "ns",
                           "area_type",
                           "traveller_type"]
    )
    employment = zt.run(
        employment,
        translation_dataframe,
        "MSOA",
        "norms_2015",
        non_split_columns=["model_zone_id",
                           "employment_cat"]
    )
    synthetic_p = zt.run(
        synthetic_p,
        translation_dataframe,
        "MSOA",
        "norms_2015",
        non_split_columns=[
            "model_zone_id",
            "purpose_id",
            "car_availability_id",
            "soc",
            "ns"
        ]
    )
    obs_base_p = zt.run(
        obs_base_p,
        translation_dataframe,
        "MSOA",
        "norms_2015",
        non_split_columns=[
            "model_zone_id",
            "purpose_id",
            "car_availability_id",
            "soc",
            "ns"
        ]
    )
    synthetic_e = zt.run(
        synthetic_e,
        translation_dataframe,
        "MSOA",
        "norms_2015",
        non_split_columns=["model_zone_id", "purpose_id", "soc"]
    )

    obs_base_e = synthetic_e[[zone_col] + seg_cols_e + [base_year]]

    soc_weights_path = (r"Y:\NorMITs Demand\import\attractions"
                        r"\soc_2_digit_sic_2018.csv")

    print("Finished loading")

    tr_p = production_exceptional_trip_rate(
        observed_base=obs_base_p,
        land_use=population,
        e_zones=pd.DataFrame,
        base_year=base_year,
        segment_cols=seg_cols_p,
        zone_column=zone_col,
        purpose_column="purpose_id",
        sector_lookup=r"Y:\NorMITs Demand\import\zone_translation\norms_2015_to_tfn_sectors.csv",
        zone_sys_name="norms_zone_id",
        sector_sys_name="tfn_sectors_zone_id"
    )

    tr_e = attraction_exceptional_trip_rate(
        observed_base=obs_base_e,
        land_use=employment,
        e_zones=pd.DataFrame,
        base_year=base_year,
        segment_cols=seg_cols_e,
        zone_column=zone_col,
        soc_weights_path=soc_weights_path,
        purpose_column="purpose_id",
        sector_lookup=r"Y:\NorMITs Demand\import\zone_translation\norms_2015_to_tfn_sectors.csv",
        zone_sys_name="norms_zone_id",
        sector_sys_name="tfn_sectors_zone_id"
    )

    print("Productions")

    year = "2033"
    base_year = "2018"
    synth_future = synthetic_p.rename({year: "value"}, axis=1)
    synth_future = synth_future[[zone_col] + seg_cols_p + ["value"]]
    synth_base = synthetic_p.rename({base_year: "value"}, axis=1)
    synth_base = synth_base[[zone_col] + seg_cols_p + ["value"]]
    observed_base = obs_base_p.rename({base_year: "value"}, axis=1)
    sector_lookup = pd.read_csv(
        r"Y:\NorMITs Demand\import\zone_translation\norms_2015_to_tfn_sectors.csv")
    sector_lookup.rename({"norms_zone_id": "model_zone_id",
                          "tfn_sectors_zone_id": "grouping_id"},
                         axis=1,
                         inplace=True)

    population = convert_pop_segmentation(
        population,
        grouping_cols=["model_zone_id", "soc", "ns", "car_availability_id"],
        value_cols="2033"
    )
    population.rename({"2033": "value"}, axis=1, inplace=True)

    print("Synthetic Future")
    print(synth_future)
    print("Synthetic Base")
    print(synth_base)
    print("Observed Base")
    print(observed_base)
    print("Land Use")
    print(population)
    print("Trip Rates")
    print(tr_p)
    tr_p.to_csv("tr.csv")

    handle_exceptional_growth(
        synth_future=synth_future,
        synth_base=synth_base,
        observed_base=observed_base,
        zone_column="model_zone_id",
        segment_columns=["purpose_id", "soc", "ns", "car_availability_id"],
        value_column="value",
        exceptional_zones=exceptional_zones,
        land_use=population,
        trip_rates=tr_p,
        sector_lookup=sector_lookup
    )

    print("Attractions")

    year = "2033"
    base_year = "2018"
    synth_future = synthetic_e.rename({year: "value"}, axis=1)
    synth_future = synth_future[[zone_col] + seg_cols_e + ["value"]]
    synth_base = synthetic_e.rename({base_year: "value"}, axis=1)
    synth_base = synth_base[[zone_col] + seg_cols_e + ["value"]]
    observed_base = obs_base_e.rename({base_year: "value"}, axis=1)
    sector_lookup = pd.read_csv(
        r"Y:\NorMITs Demand\import\zone_translation\norms_2015_to_tfn_sectors.csv")
    sector_lookup.rename({"norms_zone_id": "model_zone_id",
                          "tfn_sectors_zone_id": "grouping_id"},
                         axis=1,
                         inplace=True)

    employment = segment_employment(
        employment.loc[employment["employment_cat"] == "E01"],
        soc_weights_path=soc_weights_path,
        zone_column=zone_col,
        data_col="2033"
    )
    employment.drop(
        ["p", "employment_cat", base_year],
        axis=1,
        inplace=True
    )
    employment.rename({"2033": "value"}, axis=1, inplace=True)

    print("Synthetic Future")
    print(synth_future)
    print("Synthetic Base")
    print(synth_base)
    print("Observed Base")
    print(observed_base)
    print("Land Use")
    print(employment)
    print("Trip Rates")
    print(tr_e)
    tr_e.to_csv("tr.csv")

    handle_exceptional_growth(
        synth_future=synth_future,
        synth_base=synth_base,
        observed_base=observed_base,
        zone_column="model_zone_id",
        segment_columns=["purpose_id", "soc"],
        value_column="value",
        exceptional_zones=exceptional_zones,
        land_use=employment,
        trip_rates=tr_e,
        sector_lookup=sector_lookup,
        force_soc_type=True
    )
