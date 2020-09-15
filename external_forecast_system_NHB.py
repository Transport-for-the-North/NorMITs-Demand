# -*- coding: utf-8 -*-
"""
Created on: Wed Jun 17 11:50:10 2020
Updated on: Wed Sep 09 14:07:24 2020

Original author: tyu2
Last Update Made by: Ben Taylor

File purpose:
NHB production and distribution + PA 2 to OD conversion
returned from Systra phase 2 contract
"""

import os

import numpy as np
import pandas as pd
import itertools

import pa_to_od as pa2od
import efs_constants as consts
from demand_utilities import utils as du

# Global paths
home_path = 'Y:/NorMITs Demand/'
lookup_path = os.path.join(home_path, 'import')
import_path = os.path.join(home_path, 'inputs/default/tp_pa')
export_path = "C:/Users/Sneezy/Desktop/NorMITs Demand/nhb_dev"
seed_distributions_path = os.path.join(home_path,
                                       'inputs',
                                       'distributions',
                                       'tms',
                                       'PA Matrices 24hr')

MODEL_NAME = 'norms'


def _nhb_production_internal(hb_pa_import,
                             nhb_trip_rates,
                             year,
                             purpose,
                             mode,
                             segment,
                             car_availability):
    """
      The internals of nhb_production(). Useful for making the code more
      readable du to the number of nested loops needed
    """
    hb_dist = du.get_dist_name(
        'hb',
        'pa',
        str(year),
        str(purpose),
        str(mode),
        str(segment),
        str(car_availability),
        csv=True
    )

    # Seed the nhb productions with hb values
    hb_pa = pd.read_csv(
        os.path.join(hb_pa_import, hb_dist)
    )
    hb_pa = du.expand_distribution(
        hb_pa,
        year,
        purpose,
        mode,
        segment,
        car_availability,
        id_vars='p_zone',
        var_name='a_zone',
        value_name='trips'
    )

    # Aggregate to destinations
    nhb_prods = hb_pa.groupby([
        "a_zone",
        "purpose_id",
        "mode_id",
        "car_availability_id",
        "soc_id",
        "ns_id"
    ])["trips"].sum().reset_index()

    # join nhb trip rates
    nhb_prods = pd.merge(nhb_trip_rates,
                         nhb_prods,
                         on=["purpose_id", "mode_id"])

    # Calculate NHB productions
    nhb_prods["nhb_dt"] = nhb_prods["trips"] * nhb_prods["nhb_trip_rate"]

    # aggregate nhb_p 11_12
    nhb_prods.loc[nhb_prods["nhb_p"] == 11, "nhb_p"] = 12

    # Remove hb purpose and mode by aggregation
    nhb_prods = nhb_prods.groupby([
        "a_zone",
        "nhb_p",
        "nhb_m",
        "car_availability_id",
        "soc_id",
        "ns_id"
    ])["nhb_dt"].sum().reset_index()

    return nhb_prods


def nhb_production(hb_pa_import,
                   nhb_export,
                   required_purposes,
                   required_modes,
                   required_soc,
                   required_ns,
                   required_car_availabilities,
                   year_string_list,
                   lookup_folder,
                   nhb_productions_fname='internal_nhb_productions.csv'):
    """
    This function builds NHB productions by
    aggregates HB distribution from EFS output to destination

    TODO: Does this need updating to use the TMS method?

    Parameters
    ----------
    required lists:
        to loop over TfN segments

    Returns
    ----------
    nhb_production_dictionary:
        Dictionary containing NHB productions by year
    """
    # Init
    yearly_nhb_productions = list()
    nhb_production_dictionary = dict()

    # Get nhb trip rates
    # Might do the other way - This emits CA segmentation
    nhb_trip_rates = pd.read_csv(
        os.path.join(lookup_folder, "IgammaNMHM.csv")
    ).rename(
        columns={"p": "purpose_id", "m": "mode_id"}
    )

    # For every: Year, purpose, mode, segment, ca
    for year in year_string_list:
        loop_gen = du.segmentation_loop_generator(required_purposes,
                                                  required_modes,
                                                  required_soc,
                                                  required_ns,
                                                  required_car_availabilities)
        for purpose, mode, segment, car_availability in loop_gen:
            nhb_productions = _nhb_production_internal(
                hb_pa_import,
                nhb_trip_rates,
                year,
                purpose,
                mode,
                segment,
                car_availability
            )
            yearly_nhb_productions.append(nhb_productions)

        # ## Output the yearly productions ## #
        # Aggregate all productions for this year
        print("INFO: NHB Productions for yr%d complete!" % year)
        yr_nhb_productions = pd.concat(yearly_nhb_productions)
        yearly_nhb_productions.clear()

        # Rename columns from NHB perspective
        yr_nhb_productions = yr_nhb_productions.rename(
            columns={
                'a_zone': 'p_zone',
                'nhb_p': 'p',
                'nhb_m': 'm',
                'nhb_dt': 'trips'
            }
        )

        # Create year fname
        nhb_productions_fname = '_'.join(
            ["yr" + str(year), nhb_productions_fname]
        )

        # Output disaggregated
        da_fname = du.add_fname_suffix(nhb_productions_fname, '_disaggregated')
        yr_nhb_productions.to_csv(
            os.path.join(nhb_export, da_fname),
            index=False
        )

        # Aggregate productions up to p/m level
        yr_nhb_productions = yr_nhb_productions.groupby(
            ["p_zone", "p", "m"]
        )["trips"].sum().reset_index()

        # Rename cols and output to file
        # Output at p/m aggregation
        yr_nhb_productions.to_csv(
            os.path.join(nhb_export, nhb_productions_fname),
            index=False
        )

        # save to dictionary by year
        nhb_production_dictionary[year] = yr_nhb_productions
   
    return nhb_production_dictionary

                        
def nhb_furness(p_import,
                seed_nhb_dist_dir,
                od_export,
                required_purposes,
                required_modes,
                year_string_list,
                replace_zero_vals,
                zero_infill,
                nhb_productions_fname='internal_nhb_productions.csv',
                use_zone_id_subset=False):

    """
    Provides a one-iteration Furness constrained on production
    with options whether to replace zero values on the seed

    Essentially distributes the Productions based on the seed nhb dist
    TODO: Actually add in some furnessing

    Return:
    ----------
    None
    """
    # TODO: Add in file exists checks

    # For every year, purpose, mode
    yr_p_m_iter = itertools.product(year_string_list,
                                    required_purposes,
                                    required_modes)
    for year, purpose, mode in yr_p_m_iter:
        # ## Read in Files ## #
        # Create year fname
        year_p_fname = '_'.join(
            ["yr" + str(year), nhb_productions_fname]
        )

        # Read in productions
        p_path = os.path.join(p_import, year_p_fname)
        productions = pd.read_csv(p_path)

        # select needed productions
        productions = productions.loc[productions["p"] == purpose]
        productions = productions.loc[productions["m"] == mode]

        # read in nhb_seeds
        seed_fname = du.get_dist_name(
            'nhb',
            'pa',
            purpose=str(purpose),
            mode=str(mode),
            csv=True
        )
        nhb_seeds = pd.read_csv(os.path.join(seed_nhb_dist_dir, seed_fname))

        # convert from wide to long format
        nhb_seeds = nhb_seeds.melt(
            id_vars=['p_zone'],
            var_name='a_zone',
            value_name='seed_vals'
        )

        # Need to make sure they are the correct types
        nhb_seeds['a_zone'] = nhb_seeds['a_zone'].astype(float).astype(int)
        productions['p_zone'] = productions['p_zone'].astype(int)

        if use_zone_id_subset:
            zone_subset = [259, 267, 268, 270, 275, 1171, 1173]
            nhb_seeds = du.get_data_subset(
                nhb_seeds, 'p_zone', zone_subset)
            nhb_seeds = du.get_data_subset(
                nhb_seeds, 'a_zone', zone_subset)

        # Check the productions and seed zones match
        p_zones = set(productions["p_zone"].tolist())
        seed_zones = set(nhb_seeds["p_zone"].tolist())

        # Skip check if we're using a subset
        if use_zone_id_subset:
            print("WARNING! Using a zone subset. Can't check seed "
                  "zones are valid!")
        else:
            if p_zones != seed_zones:
                raise ValueError("Production and seed attraction zones "
                                 "do not match.")

        # Infill zero values
        if replace_zero_vals:
            mask = (nhb_seeds["seed_vals"] == 0)
            nhb_seeds.loc[mask, "seed_vals"] = zero_infill

        # Calculate seed factors by zone
        # (The sum of zone seed factors should equal 1)
        unq_zone = nhb_seeds['p_zone'].drop_duplicates()
        for zone in unq_zone:
            zone_mask = (nhb_seeds['p_zone'] == zone)
            nhb_seeds.loc[zone_mask, 'seed_factor'] = (
                    nhb_seeds[zone_mask]['seed_vals'].values
                    /
                    nhb_seeds[zone_mask]['seed_vals'].sum()
            )
        nhb_seeds = nhb_seeds.reindex(
            ['p_zone', 'a_zone', 'seed_factor'],
            axis=1
        )

        # Use the seed factors to Init P-A trips
        init_pa = pd.merge(
            nhb_seeds,
            productions,
            on=["p_zone"])
        init_pa["trips"] = init_pa["seed_factor"] * init_pa["trips"]

        # TODO: Some actual furnessing should happen here!
        final_pa = init_pa

        # ## Output the furnessed PA matrix to file ## #
        # Generate path
        nhb_dist_fname = du.get_dist_name(
            'nhb',
            'od',
            str(year),
            str(purpose),
            str(mode),
            csv=True
        )
        out_path = os.path.join(od_export, nhb_dist_fname)

        # Convert from long to wide format and output
        # TODO: Generate output name based on model name
        du.long_to_wide_out(
            final_pa.rename(columns={'p_zone': 'norms_zone_id'}),
            v_heading='norms_zone_id',
            h_heading='a_zone',
            values='trips',
            out_path=out_path
        )
        print("NHB Distribution %s complete!" % nhb_dist_fname)


def main():
    # TODO: Integrate into TMS and EFS proper

    # Say what to run
    run_build_tp_pa = False
    run_build_od = False
    run_nhb_production = False
    run_nhb_furness = True
    run_nhb_build_tp_pa = False

    # TODO: Properly integrate this
    # How much should we print?
    echo = False
    use_zone_id_subset = True

    # TODO: Create output folders
    # du.create_folder(pa_export, chDir=False)

    if run_build_tp_pa:
        pa2od.efs_build_tp_pa(
            tp_import=import_path,
            pa_import=os.path.join(export_path, '24hr PA Matrices'),
            pa_export=os.path.join(export_path, 'PA Matrices'),
            year_string_list=consts.NHB_FUTURE_YEARS,
            required_purposes=consts.PURPOSES_NEEDED,
            required_modes=consts.MODES_NEEDED,
            required_soc=consts.SOC_NEEDED,
            required_ns=consts.NS_NEEDED,
            required_ca=consts.CA_NEEDED
        )
        print('Transposed HB PA to tp PA\n')

    if run_build_od:
        pa2od.efs_build_od(
            pa_import=os.path.join(export_path, "PA Matrices"),
            od_export=os.path.join(export_path, "OD Matrices"),
            required_purposes=consts.PURPOSES_NEEDED,
            required_modes=consts.MODES_NEEDED,
            required_soc=consts.SOC_NEEDED,
            required_ns=consts.NS_NEEDED,
            required_car_availabilities=consts.CA_NEEDED,
            year_string_list=consts.NHB_FUTURE_YEARS,
            phi_type='fhp_tp',
            aggregate_to_wday=True,
            echo=echo)
        print('Transposed HB tp PA to OD\n')

    # TODO: Create 24hr OD for HB

    if run_nhb_production:
        nhb_production(
            hb_pa_import=os.path.join(export_path, "24hr PA Matrices"),
            nhb_export=os.path.join(export_path, "Productions"),
            required_purposes=consts.PURPOSES_NEEDED,
            required_modes=consts.NHB_MODES_NEEDED,
            required_soc=consts.SOC_NEEDED,
            required_ns=consts.NS_NEEDED,
            required_car_availabilities=consts.CA_NEEDED,
            year_string_list=consts.NHB_FUTURE_YEARS,
            lookup_folder=lookup_path)
        print('Generated NHB productions\n')

    if run_nhb_furness:
        nhb_furness(
            p_import=os.path.join(export_path, "Productions"),
            seed_nhb_dist_dir=seed_distributions_path,
            od_export=os.path.join(export_path, "24hr OD Matrices"),
            required_purposes=consts.NHB_PURPOSES_NEEDED,
            required_modes=consts.NHB_MODES_NEEDED,
            year_string_list=consts.NHB_FUTURE_YEARS,
            replace_zero_vals=True,
            zero_infill=0.01,
            use_zone_id_subset=use_zone_id_subset)
        print('"Furnessed" NHB Productions\n')

    if run_nhb_build_tp_pa:
        pa2od.efs_build_tp_pa(
            tp_import=import_path,
            pa_import=os.path.join(export_path, '24hr OD Matrices'),
            pa_export=os.path.join(export_path, 'OD Matrices'),
            matrix_format='od',
            year_string_list=consts.NHB_FUTURE_YEARS,
            required_purposes=consts.NHB_PURPOSES_NEEDED,
            required_modes=consts.NHB_MODES_NEEDED
        )
        print('Transposed NHB OD to tp OD\n')


if __name__ == '__main__':
    main()
