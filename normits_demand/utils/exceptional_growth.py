import os
import shutil
import warnings
from typing import List
from typing import Tuple
from typing import Iterable
from itertools import product

import numpy as np
import pandas as pd
from tqdm import tqdm

# BACKLOG: Exceptional growth probably doesn't belong in utils!
#  labels: EFS, demand merge

# Local imports
from normits_demand import efs_constants as consts
from normits_demand.utils import general as du

from normits_demand.models import efs_attraction_model as am
from normits_demand.models import efs_zone_translator as zt


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
                                 employment: pd.DataFrame,
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
    employment : pd.DataFrame
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
    emp = employment.loc[employment["employment_cat"] == "E01"]
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
    if "soc" in emp.columns:
        emp = emp.groupby(
            ["sector_id", "soc"],
            as_index=False
        )[base_year].sum()
        emp["soc"] = emp["soc"].astype("int")
    else:
        emp = emp.groupby(emp_group_cols, as_index=False)[base_year].sum()
    # Rename base year employment column to avoid conflicts when merging
    emp.rename({base_year: "land_use"}, axis=1, inplace=True)

    # Convert the base observed attractions to sector totals
    observed = observed.groupby(
        attr_group_cols,
        as_index=False
    )[base_year].sum()

    # Convert the observed data to the correct format for a merge
    observed["soc"] = observed["soc"].replace("none", 0)
    observed["soc"] = observed["soc"].astype(int)

    # Add in soc0 if not in there
    if "soc" in emp.columns and 0 not in emp["soc"].unique():
        # Sum all columns except soc to get soc0 values
        group_cols = [col for col in emp_group_cols if col != "soc"]
        soc_0 = emp.groupby(group_cols)["land_use"].sum().reset_index()
        soc_0["soc"] = 0
        emp = emp.append(soc_0)

        # Make sure the dtype matches the observed data
        emp["soc"] = emp["soc"].astype(int)

    # Merge is done on sector_id and employment segmentation (soc) so that the
    # same land_use data is joined to each purpose in the attractions
    tr_e = pd.merge(
        observed,
        emp,
        on=emp_group_cols
    )

    # emp_group_cols.insert(1, purpose_column)
    tr_e = tr_e.set_index(attr_group_cols).sort_index().reset_index()

    # Trip Rate (attraction weight) =
    # Base year attractions / Base year employment
    tr_e["trip_rate"] = tr_e[base_year] / tr_e["land_use"]
    tr_e = tr_e[attr_group_cols + ["trip_rate"]]

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

    # The productions expect the soc ns segmentation to contain "none" for
    # purposes where that segmentation is not relevant. Population needs to
    # be converted to this format
    # Format change is:
    #  soc, ns     soc, ns
    #  1  , 1      1  , none
    #  1  , 2      2  , none
    #  1  , 3  ->
    #  1  , 4
    #  1  , 5
    #  2  , 1
    pop = population.copy()

    pop["ns"] = pop["ns"].astype("int")
    pop["soc"] = pop["soc"].astype("int")
    pop["ns"] = "none"
    # Also need to ensure consistent types (object) for these columns as they
    # will contain a mixture of integers and "none"
    pop["ns"] = pop["ns"].astype("str")
    pop["soc"] = pop["soc"].astype("str")
    # Map to sectors if necessary and group by the segmentation columns
    if sector_map is not None:
        pop[sector_column] = pop[zone_column].map(sector_map)
    pop_g = pop.groupby(grouping_cols)[value_cols].sum()

    pop = population.copy()
    # Repeat for soc segmentation
    pop["ns"] = pop["ns"].astype("int")
    pop["soc"] = pop["soc"].astype("int")
    pop["soc"] = "none"

    pop["soc"] = pop["soc"].astype("str")
    pop["ns"] = pop["ns"].astype("str")

    if sector_map is not None:
        pop[sector_column] = pop[zone_column].map(sector_map)

    # Combine the two dataframes
    pop_g = pop_g.append(pop.groupby(grouping_cols)[value_cols].sum())

    pop_g = pop_g.sort_index()
    pop_g = pop_g.reset_index()

    return pop_g


def calculate_production_trip_rate(observed_base: pd.DataFrame,
                                   population: pd.DataFrame,
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
        and be at the same zone level as the population.
    population : pd.DataFrame
        Base year land use data. Should contain the column base_year and all 
        columns in segment_cols, with the exception of the purpose_column.
    e_zones : pd.DataFrame
        Unused
    base_year : str
        The base year that will be used to calculate the trip rates. Must be a 
        column in both observed_base and population.
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
        population,
        pop_group_cols,
        value_cols=base_year,
        sector_map=sector_map,
        zone_column=zone_column
    )

    # Rename to land_use to prevent name conflicts on merges
    pop_g = pop_g.rename({base_year: "land_use"}, axis=1)

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
        Contains either the population or employment  data for each zone, 
        with the same segmentation as the synthetic/observed data (except
        for purpose_id).
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

    # None is left in soc data for attractions as a residual from getting
    # observed data from PA matrices. Set soc0 where soc==None to remove this
    # and making it match synthetic attractions.
    if forecast_vector['soc'].dtype != object:
        # Cast observed data to match dtype
        forecast_dtype = forecast_vector['soc'].dtype
        observed_base["soc"] = observed_base["soc"].replace("none", 0)
        observed_base["soc"] = observed_base["soc"].astype(forecast_dtype)

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
    # Get the relevant land use data and save in e_land_use (exceptional)
    mask = land_use[zone_column].isin(exceptional_zones[zone_column])
    e_land_use = land_use.loc[mask].copy()
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
                if col not in ["soc", "p"]
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

    if audit_location:
        e_land_use.to_csv(
            os.path.join(audit_location, "exceptional_land_use.csv"),
            index=False
        )
    # This looks like a mini production/attraction model?

    # Merge on the common segmentation and re-calculate the synthetic forecast
    # using the new sector level trip rates
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

    return forecast_vector


def load_seed_dists(mat_folder: str,
                    segments_needed: List[str],
                    base_year: str,
                    zone_column: str,
                    trip_type: str = "productions",
                    ) -> pd.DataFrame:

    # Define the columns needed
    group_cols = [zone_column] + segments_needed
    required_cols = group_cols + [base_year]

    # Get the list of available files in the seed dist folder
    files = du.parse_mat_output(
        os.listdir(mat_folder),
        mat_type="pa"
    )
    # Filter to just HB matrices
    hb_files = files.loc[files["trip_origin"] == "hb"]
    # Define dataframe to store the observed trip ends
    all_obs = pd.DataFrame()

    iterator = tqdm(
        hb_files.to_dict(orient="records"),
        desc=f"Loading Base Observed {trip_type}"
    )
    # Loop through each matrix in the path and add to the overall dataframe
    for row in iterator:
        file_name = row.pop("file")
        file_path = os.path.join(mat_folder, file_name)

        obs = pd.read_csv(file_path, index_col=0)
        # Sum along columns for productions and rows for attractions
        if trip_type == "productions":
            obs = obs.sum(axis=1)
        elif trip_type == "attractions":
            obs = obs.sum(axis=0)
        else:
            raise ValueError("Invalid Trip Type supplied")

        # Set column names
        obs = obs.reset_index()
        obs.columns = [zone_column, base_year]
        obs[zone_column] = obs[zone_column].astype("int")

        # Extract segments from the file names
        for segment in segments_needed:
            obs[segment] = row[segment]

        # Add to the overall dataframe
        if all_obs.empty:
            all_obs = obs
        else:
            all_obs = all_obs.append(obs)

    # Change data types for all integer columns
    for col in ["p", "ca"]:
        if col in all_obs.columns:
            all_obs[col] = all_obs[col].astype("int")

    # Finally group and sum the dataframe
    all_obs = all_obs.groupby(
        group_cols,
        as_index=False
    )[base_year].sum()

    return all_obs[required_cols]


def growth_criteria(synth_productions: pd.DataFrame,
                    synth_attractions: pd.DataFrame,
                    observed_pa_path: str,
                    population_path: str,
                    employment_path: str,
                    model_name: str,
                    base_year: str,
                    future_years: List[str],
                    prod_exceptional_zones: pd.DataFrame = None,
                    attr_exceptional_zones: pd.DataFrame = None,
                    zone_translator: zt.ZoneTranslator = None,
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
    observed_pa_path : str
        Path to the base observed productions/attractions. Seed distributions
        used in the distribution process
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

    # Infer the P/A and pop/emp segmentation
    non_seg_cols = [zt_from_zone_col, model_zone_col] + all_years
    segments = {
        "pop": du.list_safe_remove(list(population), non_seg_cols),
        "emp": du.list_safe_remove(list(employment), non_seg_cols),
        "prod": du.list_safe_remove(list(synth_productions), non_seg_cols),
        "attr": du.list_safe_remove(list(synth_attractions), non_seg_cols)
    }
    # Load the base observed Production/Attractions
    observed_productions = load_seed_dists(
        observed_pa_path,
        segments_needed=segments["prod"],
        base_year=base_year,
        zone_column=model_zone_col,
        trip_type="productions"
    )
    observed_attractions = load_seed_dists(
        observed_pa_path,
        segments_needed=segments["attr"],
        base_year=base_year,
        zone_column=model_zone_col,
        trip_type="attractions"
    )

    # Set up the grouping columns
    prod_group_cols = [model_zone_col] + segments["prod"]
    attr_group_cols = [model_zone_col] + segments["attr"]
    pop_group_cols = [model_zone_col] + segments["pop"]
    emp_group_cols = [model_zone_col] + segments["emp"]

    # Ensure correct column types
    population.columns = population.columns.astype(str)
    employment.columns = employment.columns.astype(str)

    # If there is no soc segmentation in the employment - need to add
    if "soc" not in employment.columns:
        employment = segment_employment(
            employment,
            soc_weights_path,
            "msoa_zone_id",
            [base_year] + future_years
        )
        # Add back into the employment segmentation
        segments["emp"].append("soc")
        emp_group_cols.append("soc")

    # If the zone translator has been supplied, need to change zone system
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
            non_split_columns = du.list_safe_remove(
                non_split_columns, all_years)
            prod_exceptional_zones = zone_translator.run(
                prod_exceptional_zones,
                translation_df=zt_pop_df,
                from_zoning=zt_from_zone,
                to_zoning=model_name,
                non_split_cols=non_split_columns
            )
        else:
            prod_exceptional_zones = pd.DataFrame(columns=[model_zone_col])

        if attr_exceptional_zones is not None:
            non_split_columns = list(attr_exceptional_zones)
            non_split_columns = du.list_safe_remove(
                non_split_columns, all_years)
            attr_exceptional_zones = zone_translator.run(
                attr_exceptional_zones,
                translation_df=zt_pop_df,
                from_zoning=zt_from_zone,
                to_zoning=model_name,
                non_split_cols=non_split_columns
            )
        else:
            attr_exceptional_zones = pd.DataFrame(columns=[model_zone_col])

    # Stick exceptional zones together now they've been translated
    exceptional_zones = [prod_exceptional_zones, attr_exceptional_zones]
    exceptional_zones = pd.concat(exceptional_zones, axis=0)

    # ## Calculate Trip Rates ## #

    # Extract just the base year data from observed data
    index_cols = prod_group_cols + [base_year]
    observed_prod_base = observed_productions.reindex(columns=index_cols)

    index_cols = attr_group_cols + [base_year]
    observed_attr_base = observed_attractions.reindex(columns=index_cols)

    prod_trip_rates = calculate_production_trip_rate(
        observed_base=observed_prod_base,
        population=population,
        e_zones=pd.DataFrame,
        base_year=base_year,
        segment_cols=segments['prod'],
        zone_column=model_zone_col,
        purpose_column=purpose_col,
        sector_lookup=trip_rate_sectors
    )
    attr_trip_rates = calculate_attraction_weights(
        observed_base=observed_attr_base,
        employment=employment,
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

    # Set indexes to make concat faster
    grown_productions = [x.set_index(prod_group_cols) for x in grown_productions]
    grown_attractions = [x.set_index(attr_group_cols) for x in grown_attractions]

    # Combine production forecast vectors for each year
    grown_productions = pd.concat(grown_productions, axis=1, sort=False)
    grown_productions = grown_productions.reset_index()

    # Combine attraction forecast vectors for each year
    grown_attractions = pd.concat(grown_attractions, axis=1, sort=False)
    grown_attractions = grown_attractions.reset_index()

    return grown_productions, grown_attractions


def extract_donor_totals(matrix_path: str,
                         sectors: pd.DataFrame,
                         tour_proportions: pd.DataFrame = None
                         ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Extracts the matrix trip ends for a given HB/NHB matrix, for the 
    required Donor Sectors.
    If tour proportions are provided, PA matrices are converted to OD
    first, so that the totals are comparable with each other.

    Parameters
    ----------
    matrix_path : str
        Path to the matrix file
    sectors : pd.DataFrame
        Donor sectors that the totals will be extracted for.
    tour_proportions : pd.DataFrame, optional
        Tour proportions - wide format. Should be provided if the matrix is
        HB productions/Attractions, by default None

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Tuple containing the donor sector totals and the average tour 
        proportions for the donor sectors.

    Raises
    ------
    ValueError
        If the tour proportions shape does not match the matrix shape.
    """

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

    # Calculate the origin and destination trip ends and combine into one
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
    donor_totals = donor_totals.reset_index()

    return donor_totals, agg_tour_props


def calculate_tour_proportions(od_matrix_base: str,
                               fill_val: float = 0.5
                               ) -> pd.DataFrame:
    """Calculate tour proportions from the "from_home" and "to_home" OD
    matrices produced by the EFS. Combines all time periods into a 24hr value.
    If no values are available, the tour proportions default to fill_val.

    Parameters
    ----------
    od_matrix_base : str
        The base form of the OD matrix name. Curly braces should be included
        where the from/to strings should be added.
    fill_val : float, optional
        The default value for tour proportions, by default 0.5

    Returns
    -------
    pd.DataFrame
        A wide format dataframe with the proportion of "from home" to "to home"
        trips.
    """

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
                        ca_seg_needed: bool,
                        base_year: str,
                        p_needed: List[int] = consts.ALL_PURPOSES_NEEDED,
                        m_needed: List[int] = consts.MODES_NEEDED,
                        soc_needed: List[int] = consts.SOC_NEEDED,
                        ns_needed: List[int] = consts.NS_NEEDED,
                        ca_needed: List[int] = consts.CA_NEEDED
                        ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Extract matrix totals for each donor zone, for each segmentation.
    If available, also calculate the tour proportions for each of these.

    Parameters
    ----------
    sectors : pd.DataFrame
        Dataframe of all donor sectors that are required.
    export_paths : dict
        EFS export path dictionary. Used to locate the output matrices.
    nhb_segmented : bool
        Flag if the NHB matrices contain the full segmentation like the HB 
        matrices.
    ca_needed : bool
        Flag if Car Availability segmentation is needed.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Tuple containing the donor sector totals for each segmentation, and 
        the tour proportions extracted for each donor zone.
    """

    pa_path = export_paths["pa_24"]
    od_path = export_paths["od"]

    # Check if OD matrices are available for tour proportions
    ods_available = len(os.listdir(od_path)) > 0

    # ## Build HB totals ## #
    hb_purps = du.intersection(p_needed, consts.ALL_HB_P)
    # For Noham, assume CA_NEEDED should be None
    cas = ca_needed if ca_seg_needed else ["None"]
    year = base_year

    hb_donor_data = pd.DataFrame()
    agg_tour_props = pd.DataFrame()

    desc = "Getting HB donor zone data"
    message = ""

    # Iterate through all segmentation available
    iter_hb = tqdm(list(product(hb_purps, m_needed, cas)), desc=desc)
    for purp, mode, ca in iter_hb:
        if purp in consts.SOC_P:
            segments = soc_needed
        elif purp in consts.NS_P:
            segments = ns_needed
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
                warnings.warn("Warning: Using default Tour Proportions of 0.5")
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
    nhb_purps = du.intersection(p_needed, consts.ALL_NHB_P)

    nhb_donor_data = pd.DataFrame()

    desc = "Getting NHB donor zone data"
    if segment_employment:
        iter_nhb = tqdm(list(product(nhb_purps, m_needed, cas)), desc=desc)
    else:
        iter_nhb = tqdm(list(product(nhb_purps, m_needed, [None])), desc=desc)
    for purp, mode, ca in iter_nhb:
        if not segment_employment:
            segments = [999]
        elif purp in consts.SOC_P:
            segments = soc_needed
        # Not elif as NS_P not defined for NHB
        else:
            segments = ns_needed
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

    # Combine to a single Dataframe with all segmentation
    hb_donor_data["trip_origin"] = "hb"
    nhb_donor_data["trip_origin"] = "nhb"
    donor_data = hb_donor_data.append(nhb_donor_data)

    return donor_data, agg_tour_props


def _replace_generation_segments(generation_data: pd.DataFrame,
                                 purpose_data: pd.DataFrame,
                                 segmented_nhb: bool,
                                 ca_seg_needed: bool,
                                 p_needed: List[int] = consts.ALL_PURPOSES_NEEDED,
                                 soc_needed: List[int] = consts.SOC_NEEDED,
                                 ns_needed: List[int] = consts.NS_NEEDED,
                                 ca_needed: List[int] = consts.CA_NEEDED):
    """Splits the input generation data to the same level as TfN enhanced 
    segmentation. E.g. where dummy values of 999 are used, these are split 
    to represent all available segmentation.
    Note: at this point the underlying donor sector splits are not applied.

    Parameters
    ----------
    generation_data : pd.DataFrame
        The user input generation data
    purpose_data : pd.DataFrame
        The user input purpose definitions.
    segmented_nhb : bool
        Flag that is True if NHB matrices contain all segmentation that HB does
    ca_needed : bool
        Flag that is True if CA segmentation is needed. E.g. False for NoHam

    Returns
    -------
    pd.DataFrame
        The generation data dataframe, with new segmentation columns at TfN 
    enhanced level.
    """

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

    # If the "TfN Segmentation - soc" column is "999", merge with the
    # following dataframe to ensure all possible TfN soc values are considered
    soc_purps = du.intersection(p_needed, consts.SOC_P)
    socs = pd.DataFrame(
        [[999, p, seg] for p, seg in product(soc_purps, soc_needed)],
        columns=["TfN Segmentation - soc", "Purpose", "soc"]
    )
    gen_data = pd.merge(
        gen_data,
        socs,
        on=["TfN Segmentation - soc", "Purpose"],
        how="left"
    )
    # If the "TfN Segmentation - ns" column is "999", merge with the
    # following dataframe to ensure all possible TfN ns values are considered
    ns_purps = du.intersection(p_needed, consts.NS_P)
    ns = pd.DataFrame(
        [[999, p, seg] for p, seg in product(ns_purps, ns_needed)],
        columns=["TfN Segmentation - ns", "Purpose", "ns"]
    )
    gen_data = pd.merge(
        gen_data,
        ns,
        on=["TfN Segmentation - ns", "Purpose"],
        how="left"
    )
    if ca_seg_needed:
        cas = pd.DataFrame(
            [[999,  seg] for seg in ca_needed],
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
    # Build the segment column using the hierarchy of disaggregated first
    # If the values supplied were 999, use the TfN values just added by
    # the merge
    gen_data["segment"] = gen_data["soc"].fillna(gen_data["ns"])
    # If any values are still NA, the values in "TfN Segmentation - x" must be
    # valid segmentation values and we can use these instead
    gen_data["segment"] = gen_data["segment"].fillna(
        gen_data["TfN Segmentation - soc"]).fillna(
        gen_data["TfN Segmentation - ns"]
    ).astype("int")
    # Replace the values for nhb purposes with 999
    if not segmented_nhb:
        nhb_purps = du.intersection(p_needed, consts.ALL_NHB_P)
        gen_data.loc[
            gen_data["Purpose"].isin(nhb_purps), "segment"
        ] = 999
        gen_data.loc[
            gen_data["Purpose"].isin(nhb_purps), "ca"
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
    """Apply the underlying donor sector splits, as calculated from 
    get_donor_zone_totals().
    Generation data trip volumes are split across all the required segmentation
    if aggregated values are provided.

    Parameters
    ----------
    generation_data : pd.DataFrame
        The generation data as output from _replace_generation_segments()
    donor_data : pd.DataFrame
        Donor zone data as output from get_donor_zone_totals()

    Returns
    -------
    pd.DataFrame
        The generation data with trip volumes split to the required 
        segmentation
    """

    ca_needed = ~generation_data["ca"].isin(["None"]).any()

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

    # Convert the donor data to segmentation splits for distributing the trip
    # volumes
    group_cols = ["Year",
                  "Purpose ID",
                  "Direction"] + segment_cols
    # Calculate totals
    split_data["o_totals"] = split_data.groupby(
        group_cols,
        as_index=False
    )["origins"].transform(sum)
    split_data["d_totals"] = split_data.groupby(
        group_cols,
        as_index=False
    )["dests"].transform(sum)
    # Calculate segment proportions
    split_data.loc[split_data["Direction"] == 1, "proportion"] = (
        split_data["origins"] / split_data["o_totals"]
    )
    split_data.loc[split_data["Direction"] == 2, "proportion"] = (
        split_data["dests"] / split_data["d_totals"]
    )

    # Distribute the volume totals across the new segmentation
    split_data["split_volume"] = split_data["Volume"] * \
        split_data["proportion"]

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
    """Distributed the bespoke zone generation data across all sectors in the 
    user-defined distribution.

    Parameters
    ----------
    segment_split_data : pd.DataFrame
        Generaion data at TfN level segmentation, as output by 
        _apply_underlying_segment_splits()
    distribution_data : pd.DataFrame
        The user defined sector distribution data.

    Returns
    -------
    pd.DataFrame
        The input dataframe with a new column, "dist_volume", where the trips
        for each segment have been distributed to the required sectors.
    """

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
    """Uses the aggregated tour proportions to convert OD trips to PA where
    required - for HB matrices.

    Parameters
    ----------
    sector_distributed_data : pd.DataFrame
        Generation data as output by _apply_sector_distribution()
    aggregated_tour_proportions : pd.DataFrame
        Tour proportions for each donor sector, used to convert trips in 
        HB purposes into productions and attractions to be compatible with the
        24hr PA matrices.

    Returns
    -------
    pd.DataFrame
        The input sector_distributed_data dataframe with the trips converted 
        to the relevant type for each purpose.
    """

    ca_needed = ~sector_distributed_data["ca"].isin(["None"]).any()

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
                           purpose: int,
                           keep_zeros: bool = False
                           ) -> Tuple[np.array, str]:
    """Builds the matrix of trips to be added to the existing EFS output for 
    a given segmentation.

    Handles the cases for both HB and NHB matrices (PA and OD trips)

    Extracts the zonal distribution using the old_trips matrix.

    Loops through each entry in the generation data, distributing each 
    sector individually

    Parameters
    ----------
    filtered_trips : pd.DataFrame
        The filtered generation data provided by _apply_to_bespoke_zones. 
        Contains just the data for the required segmentation.
    sector_data : pd.DataFrame
        User defined lookup of sector ID to model zone
    old_trips : np.array
        Numpy array of the original trips matrix. The returned matrix will
        have the same shape as this and will use the underlying distribution
        by zone contained in this matrix.
    purpose : int
        Used to determine if we need to apply as PAs or ODs
    keep_zeros : bool
        How to handle instances where the underlying zonal distribution 
        contains only 0s - introduced due to rounding of matrices.
        If False, the trips will be evenly distributed in that sector (all 
        zones in the sector receive an equal proportion).
        If True, the zeros will remain and no bespoke trips will be distributed
        in this sector. By default, False

    Returns
    -------
    Tuple[np.array, str]
        The matrix of bespoke zone data that will be applied to the original 
        matrix for this segmentation.
        Warning message if relevant or None

    Raises
    ------
    ValueError
        If invalid purpose supplied or combination of IntraZonal distribution
        and sector definition is invalid.
    """

    # Build new matrix to combine with existing
    add_trips = np.zeros_like(old_trips)
    # Variable to store warning message
    warning_message = None
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
        # Extract the trip that need to be distributed
        trips_tot = {
            "prods": row_dict["prod"],
            "attrs": row_dict["attr"],
            "ods_from": row_dict["dist_volume"],
            "ods_to": row_dict["dist_volume"]
        }
        
        # Handle case where all zero in the distribution data
        keep_warning = (
            "Not distributing some trips as 0 trips in original matrix"
        )
        replace_warning = (
            "Evenly distributing some trips where there were 0 in the original"
            " matrix"
        )
        if np.count_nonzero(row_dist) == 0:
            # Replace wil all ones to remove any errors and distribute evenly
            # if needed
            row_dist = np.ones_like(row_dist)
            # If we do not want to distribute these trips
            if keep_zeros:
                warning_message = keep_warning
                trips_tot["prods"] = 0
                trips_tot["ods_from"] = 0
            else:
                warning_message = replace_warning
        if np.count_nonzero(col_dist) == 0:
            # Replace wil all ones to remove any errors and distribute evenly
            # if needed
            col_dist = np.ones_like(col_dist)
            # If we do not want to distribute these trips
            if keep_zeros:
                warning_message = keep_warning
                trips_tot["attrs"] = 0
                trips_tot["ods_to"] = 0
            else:
                warning_message = replace_warning
                
        if purpose in consts.ALL_HB_P:
            # For HB - add both productions and attractions
            add_trips[bespoke_zone_idx, zone_idxs] += (
                trips_tot["prods"]
                * row_dist
                / row_dist.sum()
            )
            add_trips[zone_idxs, bespoke_zone_idx] += (
                trips_tot["attrs"]
                * col_dist
                / col_dist.sum()
            )
        elif row_dict["Direction"] == 1:
            # For NHB - use the direction and add the ODs
            add_trips[bespoke_zone_idx, zone_idxs] += (
                trips_tot["ods_from"]
                * row_dist
                / row_dist.sum()
            )
        elif row_dict["Direction"] == 2:
            add_trips[zone_idxs, bespoke_zone_idx] += (
                trips_tot["ods_to"]
                * col_dist
                / col_dist.sum()
            )
        else:
            raise ValueError("Invalid Purpose or Direction")
    return add_trips, warning_message


def _constrain_to_sector_total(trip_matrix: np.array,
                               bespoke_trip_matrix: np.array,
                               constraint_zones: np.array,
                               bespoke_zone: int,
                               minimum_reduction: float = 0.25,
                               constrain: str = "origin"
                               ) -> np.array:
    """Constraint method for user-defined constraint type 2.

    Constrains the new bespoke_trip_matrix to the original trip_matrix, for 
    the given constraint_zones

    Parameters
    ----------
    trip_matrix : np.array
        [description]
    bespoke_trip_matrix : np.array
        [description]
    constraint_zones : np.array
        [description]
    bespoke_zone : int
        [description]
    minimum_reduction : float, optional
        [description], by default 0.25
    constrain : str, optional
        [description], by default "origin"

    Returns
    -------
    np.array
        [description]
    """

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
                            export_dict: dict,
                            segmented_nhb: bool,
                            p_needed: List[int] = consts.ALL_PURPOSES_NEEDED,
                            soc_needed: List[int] = consts.SOC_NEEDED,
                            ns_needed: List[int] = consts.NS_NEEDED,
                            ca_needed: List[int] = consts.CA_NEEDED
                            ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loops through all segmentation in the EFS output matrices, applying the
    generation data to the bespoke zones.

    The existing matrices are first copied to the new directory and bespoke 
    matrices are written to disk as we loop through them - overwriting the
    copied matrices.

    Sub functions are used to create the addition matrix of the generation
    data that will be added to the original data, and to constrain to sector 
    total if required.

    Parameters
    ----------
    converted_trips : pd.DataFrame
        Dataframe returned by _convert_od_to_trips()
    sector_data : pd.DataFrame
        Sector definitions for the model zones.
    export_dict : str
        EFS dictionary of export paths. Used to locate the relevant matrices 
        for each segmentation.
    segmented_nhb : bool
        Flag that is True if NHB matrices contain all possible segmentation

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Tuple containing a dataframe of the matrix totals for each segmentation
        and the changes made to them, and the matrices that were skipped where 
        they could not be located or an error occurred.

    Raises
    ------
    ValueError
        If inconsistencies are found in the input data.
    """

    # Copy the existing matrices to the bespoke folder - some will be
    # overwritten by the new bespoke versions
    for file_name in os.listdir(export_dict["pa_24"]):
        old_file_path = os.path.join(export_dict["pa_24"], file_name)
        dest_file_path = os.path.join(export_dict["pa_24_bespoke"], file_name)
        shutil.copy(old_file_path, dest_file_path)

    # Initialise empty lists to hold the audit outputs
    # For storing the totals added to each matrix
    additions = []
    # For storing info on any matrices that needed to be skipped
    skipped = []
    # Store any warning message here
    warning_message = None
    # Extract the required segmentation from the generation data
    year_list = converted_trips["Year"].unique()
    purps = converted_trips["Purpose"].unique()
    modes = converted_trips["mode"].unique()
    # Use CA values - only if CA segmentation is needed
    ca_seg_needed = ~converted_trips["ca"].isin(["None"]).any()
    cas = ca_needed if ca_seg_needed else ["None"]
    # Extract the unique generation zones that need to be considered
    gen_zones = converted_trips["Zone ID"].unique()

    if len(modes) != 1:
        raise ValueError("Only one mode is supported at once")

    # Setup iteration using tqdm
    segment_combs = tqdm(
        list(product(year_list, purps, cas, modes, gen_zones))
    )
    for year, purp, ca, mode, gen_zone in segment_combs:
        # Extract the soc/ns segmentation required for this purpose
        if purp in consts.ALL_NHB_P and not segmented_nhb:
            # Handle the case where NHB is not segmented
            if ca == 2:
                continue
            segments = [999]
            ca = 999
        elif purp in consts.SOC_P:
            segments = soc_needed
        elif purp in consts.NS_P:
            segments = ns_needed
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
            matrix_path = os.path.join(
                export_dict["pa_24_bespoke"],
                matrix_name
            )
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

            add_trips, warning_returned = _build_addition_matrix(
                filtered_trips,
                sector_data,
                trips,
                purp
            )
            if warning_returned is not None:
                warning_message = warning_returned

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

            # Overwrite the matrix with the new bespoke zone adjustments
            new_matrix_path = matrix_path
            new_trips_df = pd.DataFrame(
                trips,
                index=trips_df.index,
                columns=trips_df.columns
            )
            new_trips_df.to_csv(new_matrix_path)
    # Raise the warning on replacing zeros
    if warning_message is not None:
        warnings.warn(warning_message)
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
    """Checks the input bespoke zone data for errors or inconsistencies.

    Raises a ValueError if any errors are found.

    Parameters
    ----------
    gen_data : pd.DataFrame
        Generation Data Table
    purp_data : pd.DataFrame
        Purpose Definition Table
    sector_data : pd.DataFrame
        Sector Lookup Table
    sector_def_data : pd.DataFrame
        Sector System Definition Table
    dist_data : pd.DataFrame
        Distribution Data Table
    """
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


def adjust_bespoke_zones(gen_path: str,
                         exports_dict: str,
                         model_name: str,
                         audit_path: str,
                         base_year: str,
                         recreate_donor: bool = True,
                         nhb_segmented: bool = True
                         ):
    """Applies the entire bespoke zone process given the path to the user input
    data for bespoke generation zones.

    Parameters
    ----------
    gen_path : str
        Path to the Excel Workbook containing all definitions and data for 
        handling bespoke zone generation.
    exports_dict : str
        EFS dictionary of export paths.
    model_name : str
        Model name (norms, noham)
    audit_path : str
        Path to save intermediate/audit outputs to.
    recreate_donor : bool, optional
        Flag set to True if the underlying splits need to be calculated, 
        by default True
    nhb_segmented : bool, optional
        Flag set to True if the NHB matrices contain the full segmentation, 
        by default True

    Raises
    ------
    ValueError
        On invalid model type supplied
    """

    if model_name == "norms_2015" or model_name == "norms":
        model_suffix = "NoRMS"
        ca_needed = True
    elif model_name == "noham":
        model_suffix = "NoHAM"
        ca_needed = False
    else:
        raise ValueError(f"Model Type {model_name} is not supported")

    bespoke_audit_path = os.path.join(
        audit_path, "Bespoke Zones"
    )
    if not os.path.isdir(bespoke_audit_path):
        os.mkdir(bespoke_audit_path)

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
            exports_dict,
            nhb_segmented,
            ca_seg_needed=ca_needed,
            base_year=base_year
        )

        donor_data.to_csv(os.path.join(bespoke_audit_path, "donor_data.csv"))
        agg_tour_props.to_csv(os.path.join(bespoke_audit_path, "tp_data.csv"))
    else:
        donor_data = pd.read_csv(
            os.path.join(bespoke_audit_path, "donor_data.csv")
        )
        agg_tour_props = pd.read_csv(
            os.path.join(bespoke_audit_path, "tp_data.csv")
        )

    # Convert the segmentation to the EFS segments to split the bespoke
    # zone data
    print("Splitting bespoke segments")
    gen_data = _replace_generation_segments(
        gen_data,
        purp_data,
        nhb_segmented,
        ca_needed
    )

    # Apply the underlying segment splits where required
    print("Applying donor sector splits")
    split_data = _apply_underlying_segment_splits(
        gen_data,
        donor_data
    )

    # ## Distribution ## #
    print("Applying user distribution")
    sector_dist = _apply_sector_distribution(split_data, dist_data)

    # Convert HB purposes into productions/attractions using tour proportions
    print("Converting to PAs")
    converted_trips = _convert_od_to_trips(
        sector_dist,
        agg_tour_props
    )

    # ## Combine with existing matrices ## #
    # Build list of all segmentations
    print("Combining with existing matrices")
    additions, skipped = _apply_to_bespoke_zones(
        converted_trips,
        sector_data,
        exports_dict,
        nhb_segmented
    )

    print(f"Skipped {skipped.shape[0]} matrices - see log file")

    additions.to_csv(
        os.path.join(bespoke_audit_path, "bespoke_zone_additions.csv")
    )
    skipped.to_csv(
        os.path.join(bespoke_audit_path, "bespoke_zone_skipped_matrices.csv")
    )
