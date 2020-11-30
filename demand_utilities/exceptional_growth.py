import os
from typing import List

import pandas as pd

# Import attraction generator to access soc splits
import efs_attraction_generator as attr
from demand_utilities import sector_reporter_v2 as sr
from demand_utilities import utils as du
import efs_constants as consts


def load_exceptional_zones(productions_export: str,
                           attractions_export: str,
                           zone_conversion_path: str
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
                                     soc_weights_path: str,
                                     base_year: str,
                                     segment_cols: List[str],
                                     zone_column: str,
                                     sector_lookup: str = None,
                                     purpose_column: str = "purpose_id"
                                     ) -> pd.DataFrame:
    
    
    emp_group_cols = ["sector_id"] + segment_cols
    if sector_lookup is not None:
        sector_map = pd.read_csv(sector_lookup)
        sector_map = sector_map.set_index("model_zone_id")["grouping_id"]
    else:
        sector_map = lambda x: x

    tr_e = None
    
    print("Calculating Attraction Weights")
    emp_sub = land_use.loc[land_use["employment_cat"] == "E01"]
    emp_sub = emp_sub[[zone_column, base_year]]
    emp = segment_employement(
        emp_sub,
        soc_weights_path=soc_weights_path,
        zone_column=zone_column,
        data_col=base_year
    )
    emp["sector_id"] = emp[zone_column].map(sector_map)
    emp = emp.groupby(["sector_id", "soc"], as_index=False)[base_year].sum()
    emp["soc"] = emp["soc"].astype("int")
    emp.rename({base_year: "land_use"}, axis=1, inplace=True)
    
    observed = observed_base.copy()
    observed["sector_id"] = observed[zone_column].map(sector_map)
    observed = observed.groupby(
        emp_group_cols,
        as_index=False
    )[base_year].sum()
    
    emp_group_cols.remove(purpose_column)
    
    tr_e = observed.merge(
        emp,
        on=emp_group_cols
    )
    
    emp_group_cols.insert(1, purpose_column)
    tr_e.set_index(emp_group_cols, inplace=True)
    tr_e.sort_index(inplace=True)
    tr_e.reset_index(inplace=True)
    
    tr_e["a_tr"] = tr_e[base_year] / tr_e["land_use"]
    
    print(tr_e)
    
    emp.to_csv("emp.csv")
    
    return tr_e


def production_exceptional_trip_rate(observed_base: pd.DataFrame,
                                     land_use: pd.DataFrame,
                                     e_zones: pd.DataFrame,
                                     base_year: str,
                                     segment_cols: List[str],
                                     zone_column: str,
                                     sector_lookup: str = None,
                                     purpose_column: str = "purpose_id"
                                     ) -> pd.DataFrame:
    # observed_base dataframes are vectors of productions and attractions at
    # TfN enhanced segmentation and model zone system

    # Convert population segmentation to production segmentation

    pop_group_cols = ["sector_id"] + segment_cols

    pop_group_cols.remove(purpose_column)
    
    if sector_lookup is not None:
        sector_map = pd.read_csv(sector_lookup)
        sector_map = sector_map.set_index("model_zone_id")["grouping_id"]
    else:
        sector_map = lambda x: x

    tr_p = None

    print("Caclulating Production Trip Rates")

    # Convert to the trip segementation
    # Deal with mismatched types
    pop = land_use.copy()
    pop["ns"] = pop["ns"].astype("int")
    pop["soc"] = pop["soc"].astype("int")
    pop["ns"] = "none"
    pop["ns"] = pop["ns"].astype("str")
    pop["soc"] = pop["soc"].astype("str")
    pop["sector_id"] = pop[zone_column].map(sector_map)
    pop_g = pop.groupby(pop_group_cols)[base_year].sum()
    pop = land_use.copy()
    pop["ns"] = pop["ns"].astype("int")
    pop["soc"] = pop["soc"].astype("int")
    pop["soc"] = "none"
    pop["soc"] = pop["soc"].astype("str")
    pop["ns"] = pop["ns"].astype("str")
    pop["sector_id"] = pop[zone_column].map(sector_map)
    pop_g = pop_g.append(pop.groupby(pop_group_cols)[base_year].sum())

    # Tidy up
    pop_g = pop_g.sort_index()
    pop_g = pop_g.reset_index()
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

    tr_p["p_tr"] = tr_p[base_year] / tr_p["land_use"]

    pop_group_cols.insert(1, purpose_column)
    tr_p.set_index(pop_group_cols, inplace=True)
    tr_p.sort_index(inplace=True)
    tr_p.reset_index(inplace=True)

    print("Saving to files")
    observed.to_csv("observed_base_prod.csv", index=False)
    pop_g.to_csv("pop_g.csv", index=False)
    
    print(tr_p)

    return tr_p


def handle_exceptional_growth(synth_future: pd.DataFrame,
                              synth_base: pd.DataFrame,
                              observed_base: pd.DataFrame,
                              zone_column: str,
                              segment_columns: List[str],
                              value_column: str,
                              exceptional_zones: pd.DataFrame = None,
                              trip_rate_path: str = None,
                              trip_type: str = "productions"
                              ) -> pd.DataFrame:

    # Read in data on exceptional zones
    e_zones = exceptional_zones or pd.DataFrame()

    # Join and calculate growth - will likely need changing
    merge_cols = [zone_column] + segment_columns
    forecast_vector = pd.merge(
        synth_base.rename({value_column: "s_b"}, axis=1),
        synth_future.rename({value_column: "s_f"}, axis=1),
        how="outer",
        on=merge_cols
    )
    forecast_vector = pd.merge(
        forecast_vector,
        calib_base.rename({value_column: "b_c"}, axis=1),
        how="outer",
        on=merge_cols
    )

    forecast_vector[value_column] = (
        forecast_vector["b_c"]
        * forecast_vector["s_f"]
        / forecast_vector["s_b"]
    )

    # Handle Exceptional Zones
    
    if trip_rate_path is None:
        # Calculate trip rates
        pass
    
    mask = forecast_vector[zone_column].isin(e_zones[zone_column])
    forecast_vector.loc[mask, value_column] = (
        forecast_vector.loc[mask, "s_f"]
    )

    # Constrain?

    return forecast_vector


def test():
    # Test productions

    print("Loading files")

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
        sector_lookup=r"Y:\NorMITs Demand\inputs\default\zoning\lad_msoa_grouping.csv"
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
        sector_lookup=r"Y:\NorMITs Demand\inputs\default\zoning\lad_msoa_grouping.csv"
    )
    
    
