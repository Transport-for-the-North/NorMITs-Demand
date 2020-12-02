import os
from typing import List

import pandas as pd

# Import attraction generator to access soc splits
import efs_attraction_generator as attr
from zone_translator import ZoneTranslator
from demand_utilities import utils as du
import efs_constants as consts


def load_exceptional_zones(productions_export: str,
                           attractions_export: str
                           ):
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

    return e_zones


def segment_employement(employment: pd.DataFrame,
                        soc_weights_path: str,
                        zone_column: str,
                        data_col: str
                        ) -> pd.DataFrame:
    
    soc_weights = attr.get_soc_weights(
        soc_weights_path
    )
    
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
            msoa_path=r"Y:\NorMITs Demand\inputs\default\zoning\msoa_zones.csv",
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
                                     sector_lookup: str = None,
                                     purpose_column: str = "purpose_id",
                                     zone_sys_name: str = "model_zone_id",
                                     sector_sys_name: str = "grouping_id",
                                     soc_weights_path: str = None
                                     ) -> pd.DataFrame:
    
    
    emp_group_cols = ["sector_id"] + segment_cols
    if sector_lookup is not None:
        sector_map = pd.read_csv(sector_lookup)
        sector_map = sector_map.set_index(zone_sys_name)[sector_sys_name]
    else:
        sector_map = lambda x: x

    tr_e = None
    
    print("Calculating Attraction Weights")
    
    observed = observed_base.copy()
    observed["sector_id"] = observed[zone_column].map(sector_map)
    
    emp_sub = land_use.loc[land_use["employment_cat"] == "E01"]
    emp_sub = emp_sub[[zone_column, base_year]]
    
    if soc_weights_path is not None:
        emp = segment_employement(
            emp_sub,
            soc_weights_path=soc_weights_path,
            zone_column=zone_column,
            data_col=base_year
        )
        observed["soc"] = observed["soc"].astype("int")
    else:
        emp = emp_sub.copy()
    
    emp["sector_id"] = emp[zone_column].map(sector_map)
    if soc_weights_path is not None:
        emp = emp.groupby(["sector_id", "soc"], as_index=False)[base_year].sum()
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
                             sector_map: pd.DataFrame = None,
                             zone_column: str = "model_zone_id",
                             sector_column: str = "sector_id"
                             ):
    
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
                                     sector_lookup: str = None,
                                     purpose_column: str = "purpose_id",
                                     zone_sys_name: str = "model_zone_id",
                                     sector_sys_name: str = "grouping_id"
                                     ) -> pd.DataFrame:
    # observed_base dataframes are vectors of productions and attractions at
    # TfN enhanced segmentation and model zone system

    # Convert population segmentation to production segmentation

    pop_group_cols = ["sector_id"] + segment_cols

    pop_group_cols.remove(purpose_column)
    
    if sector_lookup is not None:
        sector_map = pd.read_csv(sector_lookup)
        sector_map = sector_map.set_index(zone_sys_name)[sector_sys_name]
    else:
        sector_map = lambda x: x

    tr_p = None

    print("Caclulating Production Trip Rates")

    # Convert to the trip segementation
    # Deal with mismatched types
    pop_g = convert_pop_segmentation(
        land_use,
        pop_group_cols,
        value_cols=base_year,
        sector_map=sector_map,
        zone_column=zone_column
    )

    # Tidy up
    pop_g = pop_g.rename({base_year: "land_use"}, axis=1)
    
    # Group observed data
    observed = observed_base.copy()
    observed["sector_id"] = observed[zone_column].map(sector_map)
    observed = observed.groupby(
        pop_group_cols + [purpose_column],
        as_index=False
    )[base_year].sum()

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
                              exceptional_zones: pd.DataFrame = None,
                              land_use: pd.DataFrame = None, # model_zone level
                              trip_rates: pd.DataFrame = None, # Sector level
                              sector_lookup: pd.DataFrame = None, # Contains zone_column and grouping_id
                              force_soc_type: bool = False
                              ) -> pd.DataFrame:
    
    # TODO This could result in lower trips generated than before. Should this 
    # be handled
    
    # Setup sector lookup
    sector_map = sector_lookup.set_index(zone_column)["grouping_id"]

    # Join and calculate growth - will likely need changing
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

    # Setup default growth values
    forecast_vector[value_column] = (
        forecast_vector["b_c"]
        * forecast_vector["s_f"]
        / forecast_vector["s_b"]
    )

    # Handle Exceptional Zones
    if trip_rates is None:
        # TODO Calculate trip rates here?
        pass
    # Get the relevant land use data
    e_land_use = land_use.loc[
        land_use[zone_column
                 ].isin(exceptional_zones[zone_column])]
    # Map the land use to the sector used for the trip rates
    e_land_use["sector_id"] = e_land_use[zone_column].map(sector_map)
    # Join to the relevant trip rate
    print("Exceptional Land Use")
    print(e_land_use)
    e_land_use.to_csv("e_land_use.csv")
    print("Trip Rates")
    print(trip_rates)
    trip_rate_merge_cols = merge_cols.copy()
    trip_rate_merge_cols.remove(zone_column)
    trip_rate_merge_cols.remove("purpose_id")
    trip_rate_merge_cols.insert(0, "sector_id")
    
    # Required to ensure a complete merge on soc column - only for attractions
    if force_soc_type:
        e_land_use["soc"] = e_land_use["soc"].astype("int")
        
    print("Merging on ", trip_rate_merge_cols)
    e_infill = pd.merge(
        e_land_use,
        trip_rates,
        how="left",
        on=trip_rate_merge_cols
    )
    e_infill["s_f_exceptional"] = e_infill[value_column] * e_infill["trip_rate"]
    e_infill.drop(["trip_rate", "sector_id", value_column], 
                  axis=1, inplace=True)
    print("Exceptional Forecast")
    print(e_infill)
    
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
    # Constrain?

    return forecast_vector


def test():
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
    
    output_path = os.path.join(r"Y:\NorMITs Demand\inputs\default\zoning\norms_2015.csv")
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
    sector_lookup = pd.read_csv(r"Y:\NorMITs Demand\import\zone_translation\norms_2015_to_tfn_sectors.csv")
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
    sector_lookup = pd.read_csv(r"Y:\NorMITs Demand\import\zone_translation\norms_2015_to_tfn_sectors.csv")
    sector_lookup.rename({"norms_zone_id": "model_zone_id",
                          "tfn_sectors_zone_id": "grouping_id"},
                         axis=1,
                         inplace=True)
    
    employment = segment_employement(
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
    
