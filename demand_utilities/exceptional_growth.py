import os
from typing import List
from typing import Tuple

import pandas as pd

# Import attraction generator to access soc splits
import efs_attraction_generator as am

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
    exceptional_production_zones:
        A dataframe with the column "msoa_zone_id", containing all
        identified exceptional zones for productions.

    exceptional_attraction_zones:
        A dataframe with the column "msoa_zone_id", containing all
        identified exceptional zones for attractions.
    """
    prod_ez_path = os.path.join(productions_export, consts.EG_FNAME)
    attr_ez_path = os.path.join(attractions_export, consts.EG_FNAME)

    return pd.read_csv(prod_ez_path), pd.read_csv(attr_ez_path)


def segment_employment(employment: pd.DataFrame,
                       soc_weights_path: str,
                       zone_column: str,
                       data_cols: List[str],
                       emp_cat_col: str = 'employment_cat',
                       ) -> pd.DataFrame:
    """Takes a dataframe containing employment data and a path to soc_weights,
    returns the employment data segmented by skill level (soc).

    Parameters
    ----------
    employment : pd.DataFrame
        Employment data containing a zone_column, col, and
        "employment_cat" (employment category). Zoning should be at MSOA.
    soc_weights_path : str
        Path to a file containing soc weights. Should contain a column called
        "msoa_zone_id".
    zone_column : str
        Usually "msoa_zone_id", should contain MSOA zones.
    data_cols : List[str]
        Columns in employment that contains the employee data.
    msoa_lookup_path : str
        File used to convert MSOA codes to zone ids

    Returns
    -------
    pd.DataFrame
        Returns employment segmented by soc.
    """
    # Init
    soc_weights = am.get_soc_weights(soc_weights_path)

    # Split by soc for each data col
    segmented_emp = pd.DataFrame()
    non_data_cols = du.list_safe_remove(list(employment), data_cols)
    for col in data_cols:
        # Extract just this data col
        index_cols = non_data_cols + [col]
        filtered_emp = employment.reindex(columns=index_cols)

        # split and make sure data is sorted
        filtered_emp = am.split_by_soc(
            filtered_emp,
            soc_weights,
            unique_col=col,
            split_cols=[zone_column, emp_cat_col]
        ).sort_values(by=[zone_column, emp_cat_col])

        # Build the return df
        if segmented_emp.empty:
            segmented_emp = filtered_emp
        else:
            segmented_emp[col] = filtered_emp[col]

    return segmented_emp


def calculate_attraction_weights(observed_base: pd.DataFrame,
                                 land_use: pd.DataFrame,
                                 base_year: str,
                                 emp_segment_cols: List[str],
                                 attr_segment_cols: List[str],
                                 zone_column: str,
                                 sector_lookup: pd.Series = None,
                                 purpose_column: str = "p",
                                 soc_weights_path: str = None
                                 ) -> pd.DataFrame:
    """Calculates the sector level attraction weights using the 
    observed base attractions and land use.
    These are used to convert the employment for exceptional zones into 
    attractions that replace the standard attractions for these few zones.

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
    # Init
    attr_group_cols = ["sector_id"] + attr_segment_cols
    emp_group_cols = ["sector_id"] + attr_segment_cols
    emp_group_cols.remove(purpose_column)

    if sector_lookup is None:
        def sector_map(x): return x
    else:
        sector_map = sector_lookup.copy()

    # tr_e will contain the sector level trip rates for each purpose
    tr_e = None

    print("Calculating Attraction Weights")

    # Add sector column to observed data
    observed = observed_base.copy()
    observed["sector_id"] = observed[zone_column].map(sector_map)

    # Filter employment to only the commute category - drop the column
    index_cols = [zone_column] + emp_segment_cols + [base_year]
    emp = land_use.loc[land_use["employment_cat"] == "E01"]
    emp = emp.reindex(columns=index_cols)

    # If required, convert employment to soc segmentation
    if soc_weights_path is not None and 'soc' not in emp:
        emp = segment_employment(
            emp,
            soc_weights_path=soc_weights_path,
            zone_column=zone_column,
            data_cols=[base_year],
        )
        observed["soc"] = observed["soc"].astype("int")

    # Group and sum employment by the required segmentation (likely sector 
    # and soc) - adding in the column for soc segmentation if necessary
    emp["sector_id"] = emp[zone_column].map(sector_map)
    if soc_weights_path is not None:
        emp = emp.groupby(
            ["sector_id", "soc"],
            as_index=False
        )[base_year].sum()
        emp["soc"] = emp["soc"].astype("int")
    else:
        emp = emp.groupby(emp_group_cols, as_index=False)[base_year].sum()
    # Rename base year employment column to avoid conflicts when merging
    emp.rename({base_year: "land_use"}, axis=1, inplace=True)

    # print("Split Employment")
    # print(emp)

    # observed.to_csv("observed_pre_group.csv")

    # Convert the base observed attractions to sector totals
    observed = observed.groupby(
        attr_group_cols,
        as_index=False
    )[base_year].sum()

    # print("Observed Attractions")
    # print(observed)

    if "soc" in emp.columns and 0 not in emp["soc"].unique():
        # Add in soc segmentation for non-soc purposes (sum 1, 2, 3)
        soc_0 = emp.groupby(
            [col for col in emp_group_cols if col != "soc"],
            as_index=False
        )["land_use"].sum()
        soc_0["soc"] = 0
        emp = emp.append(
            soc_0
        )

    # Merge is done on sector_id and employment segmentation (soc) so that the 
    # same land_use data is joined to each purpose in the attractions
    tr_e = pd.merge(
        observed,
        emp,
        on=emp_group_cols
    )

    # emp_group_cols.insert(1, purpose_column)
    tr_e.set_index(attr_group_cols, inplace=True)
    tr_e.sort_index(inplace=True)
    tr_e.reset_index(inplace=True)

    # Trip Rate (attraction weight) = 
    # Base year attractions / Base year employment
    tr_e["trip_rate"] = tr_e[base_year] / tr_e["land_use"]

    tr_e = tr_e[attr_group_cols + ["trip_rate"]]

    # print(tr_e)

    # tr_e.to_csv("tr_e.csv")

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
    If a sector map is provided, the returned datframe will also be grouped by 
    these sectors.

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


def calculate_production_trip_rate(observed_base: pd.DataFrame,
                                   land_use: pd.DataFrame,
                                   e_zones: pd.DataFrame,
                                   base_year: str,
                                   segment_cols: List[str],
                                   zone_column: str,
                                   sector_lookup: pd.Series = None,
                                   purpose_column: str = "p"
                                   ) -> pd.DataFrame:
    """Calculates the sector level trip rates for productions using the 
    observed base productions and land use population data.

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

    print("Calculating Production Trip Rates")

    # Convert to the trip segmentation dealing with mismatched soc and ns types
    # Dataframe pop_g is also grouped by sector here
    pop_g = convert_pop_segmentation(
        land_use,
        pop_group_cols,
        value_cols=base_year,
        sector_map=sector_map,
        zone_column=zone_column
    )

    # Rename to land_use to prevent name conflicts on merges
    pop_g = pop_g.rename({base_year: "land_use"}, axis=1)

    # print(sector_lookup)
    # print(observed_base)

    # Group and sum the required segmentation
    # - likely [sector_id, purpose, soc, ns, ca]
    observed = observed_base.copy()
    observed["sector_id"] = observed[zone_column].map(sector_map)
    observed = observed.groupby(
        pop_group_cols + [purpose_column],
        as_index=False
    )[base_year].sum()

    # Merge data on all common segmentation between land_use and productions
    tr_p = observed.merge(
        pop_g,
        on=pop_group_cols
    )

    # Trip Rate = Base year productions / Base year population
    tr_p["trip_rate"] = tr_p[base_year] / tr_p["land_use"]

    pop_group_cols.insert(1, purpose_column)
    tr_p.set_index(pop_group_cols, inplace=True)
    tr_p.sort_index(inplace=True)
    tr_p.reset_index(inplace=True)

    # print("Saving to files")
    # observed.to_csv("observed_base_prod.csv", index=False)
    # pop_g.to_csv("pop_g.csv", index=False)

    tr_p.to_csv("tr_p.csv", index=False)

    tr_p = tr_p[pop_group_cols + ["trip_rate"]]

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
                              force_soc_type: bool = False,
                              purpose_col: str = 'p',
                              audit_location: str = None
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
    # Note this could result in lower trips generated than before. See
    # the audit output for the changes

    sector_map = sector_lookup.copy()

    # Combine all dataframes into one, renaming the columns for
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
    growth_factors = forecast_vector["s_f"] / forecast_vector["s_b"]
    forecast_vector[value_column] = forecast_vector["b_c"] * growth_factors

    # Handle Exceptional Zones
    # Get the relevant land use data
    e_land_use = land_use.loc[
        land_use[zone_column
                 ].isin(exceptional_zones[zone_column])].copy()
    # Map the land use to the sector used for the trip rates
    e_land_use["sector_id"] = e_land_use[zone_column].map(sector_map)
    # If a sector id was not found for any zone - print the errors and assume 
    # that it is an external zone and can be ignored
    no_sector_id = e_land_use.loc[e_land_use["sector_id"].isna()]
    if not no_sector_id.empty:
        print("Could not find a match for the following zones - ignoring "
              "them:")
        invalid_zones = no_sector_id[zone_column].unique()
        print(invalid_zones)
        e_land_use = e_land_use.loc[~e_land_use["sector_id"].isna()]
        valid_e_zones = exceptional_zones.loc[
            ~exceptional_zones[zone_column].isin(invalid_zones)
        ]
    else:
        valid_e_zones = exceptional_zones

    trip_rate_merge_cols = merge_cols.copy()
    trip_rate_merge_cols.remove(zone_column)
    trip_rate_merge_cols.remove(purpose_col)
    trip_rate_merge_cols.insert(0, "sector_id")

    # (Required to ensure a complete merge on soc column -
    # only for attractions)
    if force_soc_type:
        e_land_use["soc"] = e_land_use["soc"].astype("int")
        # Add soc = 0 segmentation (sum of the others) for use in 
        # non-soc purposes
        if 0 not in e_land_use["soc"].unique():
            # Extract the soc total at sector level
            soc_group_cols = [
                col for col in merge_cols 
                if col not in ["soc", "purpose_id"]
            ] + ["sector_id"]
            soc_0 = e_land_use.groupby(
                soc_group_cols,
                as_index=False
            )[value_column].sum()
            # Define as soc 0 and add to the end of the existing data
            soc_0["soc"] = 0
            e_land_use = e_land_use.append(
                soc_0
            )

    # print("Exceptional Land Use")
    # print(e_land_use)
    if audit_location:
        e_land_use.to_csv(
            os.path.join(audit_location, "exceptional_land_use.csv"),
            index=False
        )
    # print("Trip Rates")
    # print(trip_rates)

    # This looks like a mini production/attraction model?

    # Merge on the common segmentation and re-calculate the synthetic forecast 
    # using the new sector level trip rates
    # print("Merging on ", trip_rate_merge_cols)
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
    # print("Exceptional Forecast")
    # print(e_infill)

    # Merge to the final forecast vector and overwrite with s_f_exceptional 
    # where necessary

    forecast_vector = pd.merge(
        forecast_vector,
        e_infill,
        how="left",
        on=merge_cols
    )

    e_zone_mask = forecast_vector[zone_column].isin(valid_e_zones[zone_column])
    # If the calculated trip rate/attraction weight is too low and results in 
    # a smaller value for the P/A, use the original P/A
    s_f_increase = (
        forecast_vector["s_f_exceptional"] >= forecast_vector[value_column]
    )
    forecast_vector.loc[e_zone_mask & s_f_increase, value_column] = (
        forecast_vector.loc[e_zone_mask & s_f_increase, "s_f_exceptional"]
    )
    forecast_vector.drop(["s_f_exceptional"], axis=1, inplace=True)

    # Tidy up the forecast vector
    group_cols = [zone_column] + segment_columns
    index_cols = group_cols.copy() + [value_column]

    forecast_vector = forecast_vector.reindex(columns=index_cols)
    forecast_vector = forecast_vector.groupby(group_cols).sum().reset_index()

    # Is this how long it should be? Are we dropping anything?
    # Should be 98800 long?
    # - Currently ignores missing values in the input P/A's, results in a 
    # vector with the same shape as the one provided
    # norms_zones * ca * p * soc/ns
    # (1300*2*2*4)
    # + (1300*2*6*5)
    # print("Final Forecast")
    # print(forecast_vector)
    forecast_vector.to_csv("test.csv")

    return forecast_vector


def growth_criteria(synth_productions: pd.DataFrame,
                    synth_attractions: pd.DataFrame,
                    observed_prod_path: str,
                    observed_attr_path: str,
                    population_path: str,
                    employment_path: str,
                    model_name: str,
                    base_year: str,
                    future_years: List[str],
                    prod_exceptional_zones: pd.DataFrame = None,
                    attr_exceptional_zones: pd.DataFrame = None,
                    zone_translator: ZoneTranslator = None,
                    zt_from_zone: str = None,
                    zt_pop_df: pd.DataFrame = None,
                    zt_emp_df: pd.DataFrame = None,
                    trip_rate_sectors: str = None,
                    soc_weights_path: str = None,
                    purpose_col: str = 'p',
                    prod_audits: str = None,
                    attr_audits: str = None
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
            "am" : List[str] Attraction segments required
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
    # Init
    all_years = [base_year] + future_years

    zt_from_zone_col = zt_from_zone.lower() + '_zone_id'
    model_zone_col = model_name.lower() + '_zone_id'

    # Load files and format inputs as necessary
    print("Loading Files")
    population = du.safe_read_csv(population_path)
    employment = du.safe_read_csv(employment_path)
    observed_productions = du.safe_read_csv(observed_prod_path)
    observed_attractions = du.safe_read_csv(observed_attr_path)

    # Infer the P/A and pop/emp segmentation
    non_seg_cols = [zt_from_zone_col, model_zone_col] + all_years
    segments = {
        "pop": du.list_safe_remove(list(population), non_seg_cols),
        "emp": du.list_safe_remove(list(employment), non_seg_cols),
        "prod": du.list_safe_remove(list(synth_productions), non_seg_cols),
        "attr": du.list_safe_remove(list(synth_attractions), non_seg_cols)
    }

    # Set up the grouping columns
    prod_group_cols = [model_zone_col] + segments["prod"]
    attr_group_cols = [model_zone_col] + segments["attr"]
    pop_group_cols = [model_zone_col] + segments["pop"]
    emp_group_cols = [model_zone_col] + segments["emp"]

    # Ensure correct column types
    population.columns = population.columns.astype(str)
    employment.columns = employment.columns.astype(str)

    # print("Population")
    # print(population)

    # print("Employment")
    # print(employment)

    # If there is no soc segmentation in the employment - need to add
    if "soc" not in employment.columns:
        employment = segment_employment(
            employment,
            soc_weights_path,
            "msoa_zone_id",
            [base_year] + future_years
        )

    # If the zone translator has been supplied, need to change zone system
    # TODO: Use pop/emp translation file depending on what's being translated
    if zone_translator is not None:
        print("Translating zone system")

        # Make sure we have all the arguments for the zone translation
        zt_inputs = [zt_from_zone, zt_emp_df, zt_emp_df]
        if any([x is None for x in zt_inputs]):
            raise ValueError(
                "Was given a zone translator, but not all the arguments. "
                "Please make sure all of the following are being passed into "
                "the function call:\n"
                "'[zt_from_zone, zt_to_zone, zt_emp_df, zt_emp_df]'"
            )

        # Convert the pop/emp to the required zoning system
        non_split_columns = list(population)
        non_split_columns = du.list_safe_remove(non_split_columns, all_years)
        population = zone_translator.run(
            population,
            translation_df=zt_pop_df,
            from_zoning=zt_from_zone,
            to_zoning=model_name,
            non_split_cols=non_split_columns
        )

        non_split_columns = list(employment)
        non_split_columns = du.list_safe_remove(non_split_columns, all_years)
        employment = zone_translator.run(
            employment,
            translation_df=zt_emp_df,
            from_zoning=zt_from_zone,
            to_zoning=model_name,
            non_split_cols=non_split_columns
        )

        # Convert the exceptional zones if they exist
        if prod_exceptional_zones is not None:
            non_split_columns = list(prod_exceptional_zones)
            non_split_columns = du.list_safe_remove(non_split_columns, all_years)
            prod_exceptional_zones = zone_translator.run(
                prod_exceptional_zones,
                translation_df=zt_pop_df,
                from_zoning=zt_from_zone,
                to_zoning=model_name,
                non_split_cols=non_split_columns
            )
        else:
            prod_exceptional_zones = pd.DataFrame(columns=["model_zone_id"])

        if attr_exceptional_zones is not None:
            non_split_columns = list(attr_exceptional_zones)
            non_split_columns = du.list_safe_remove(non_split_columns, all_years)
            attr_exceptional_zones = zone_translator.run(
                attr_exceptional_zones,
                translation_df=zt_pop_df,
                from_zoning=zt_from_zone,
                to_zoning=model_name,
                non_split_cols=non_split_columns
            )
        else:
            attr_exceptional_zones = pd.DataFrame(columns=["model_zone_id"])

    # Stick exceptional zones together now they've been translated
    exceptional_zones = [prod_exceptional_zones, attr_exceptional_zones]
    exceptional_zones = pd.concat(exceptional_zones, axis=0)

    # print("Converted Population")
    # print(population)
    # print("Employment")
    # print(employment)
    # print("Obs Productions")
    # print(observed_productions)
    # print("Obs Attractions")
    # print(observed_attractions)
    # print("Exceptional Zones")
    # print(exceptional_zones)

    # ## Calculate Trip Rates ## #

    # Extract just the base year data from observed data
    index_cols = prod_group_cols + [base_year]
    observed_prod_base = observed_productions.reindex(columns=index_cols)

    index_cols = attr_group_cols + [base_year]
    observed_attr_base = observed_attractions.reindex(columns=index_cols)

    prod_trip_rates = production_exceptional_trip_rate(
        observed_base=observed_prod_base,
        land_use=population,
        e_zones=pd.DataFrame,
        base_year=base_year,
        segment_cols=segments['prod'],
        zone_column=model_zone_col,
        purpose_column=purpose_col,
        sector_lookup=trip_rate_sectors
    )
    attr_trip_rates = calculate_attraction_weights(
        observed_base=observed_attr_base,
        land_use=employment,
        base_year=base_year,
        emp_segment_cols=segments['emp'],
        attr_segment_cols=segments['attr'],
        zone_column=model_zone_col,
        purpose_column=purpose_col,
        sector_lookup=trip_rate_sectors
    )
    
    if prod_audits:
        prod_trip_rates.to_csv(
            os.path.join(prod_audits, "exc_production_triprate.csv"),
            index=False
        )
    if attr_audits:
        attr_trip_rates.to_csv(
            os.path.join(attr_audits, "exc_attraction_triprate.csv"),
            index=False
        )

    # Setup population segmentation for growth criteria
    population = convert_pop_segmentation(
        population,
        grouping_cols=du.intersection(pop_group_cols, prod_group_cols),
        value_cols=future_years
    )

    # ## Apply Growth Criteria ## #

    # Grab just the base year P/A
    index_cols = prod_group_cols + [base_year]
    synth_prod_base = synth_productions.reindex(columns=index_cols)

    index_cols = attr_group_cols + [base_year]
    synth_attr_base = synth_attractions.reindex(columns=index_cols)

    # Initialise loop output
    grown_productions = list()
    grown_attractions = list()
    grown_productions.append(synth_prod_base)
    grown_attractions.append(synth_attr_base)

    # Calculate separately for each year and combine at the end
    for year in future_years:
        # Grab the data for this year
        index_cols = prod_group_cols + [year]
        synth_prod_subset = synth_productions.reindex(columns=index_cols)

        index_cols = attr_group_cols + [year]
        synth_attr_subset = synth_attractions.reindex(columns=index_cols)

        pop_subset = population.reindex(columns=pop_group_cols + [year])
        emp_subset = employment.reindex(columns=emp_group_cols + [year])

        # Drop the commute column
        emp_subset = emp_subset.loc[emp_subset["employment_cat"] == "E01"]
        emp_subset = emp_subset.drop("employment_cat", axis=1)

        year_productions = handle_exceptional_growth(
            synth_future=synth_prod_subset,
            synth_base=synth_prod_base,
            observed_base=observed_prod_base,
            zone_column=model_zone_col,
            segment_columns=segments['prod'],
            value_column=year,
            base_year=base_year,
            exceptional_zones=prod_exceptional_zones,
            land_use=pop_subset,
            trip_rates=prod_trip_rates,
            sector_lookup=trip_rate_sectors,
            audit_location=prod_audits
        )
        grown_productions.append(year_productions)

        year_attractions = handle_exceptional_growth(
            synth_future=synth_attr_subset,
            synth_base=synth_attr_base,
            observed_base=observed_attractions,
            zone_column=model_zone_col,
            segment_columns=segments['attr'],
            value_column=year,
            base_year=base_year,
            exceptional_zones=attr_exceptional_zones,
            land_use=emp_subset,
            trip_rates=attr_trip_rates,
            sector_lookup=trip_rate_sectors,
            force_soc_type="soc" in segments['emp'],
            audit_location=attr_audits
        )
        grown_attractions.append(year_attractions)

    # Combine forecast vectors for each year
    converted_productions = pd.concat(
        [x.set_index(prod_group_cols) for x in grown_productions],
        axis=1
    ).reset_index()
    converted_pure_attractions = pd.concat(
        [x.set_index(attr_group_cols) for x in grown_attractions],
        axis=1
    ).reset_index()

    return converted_productions, converted_pure_attractions

