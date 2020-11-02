# -*- coding: utf-8 -*-
"""
Created on: Mon Sep 30 09:42:25 2019
Updated on: Tues Sep 1 08:56:35 2020

Original Author: Robert Farrar
Last update made by: Ben Taylor

File Purpose:
Module of all distribution functions for EFS
"""
import os

import pandas as pd
import numpy as np

from typing import List

import efs_constants as consts
from demand_utilities import utils as du


def doubly_constrained_furness(seed_vals: np.array,
                               row_targets: np.array,
                               col_targets: np.array,
                               tol: float = 1e-9,
                               max_iters: int = 5000
                               ) -> np.array:
    """
    Performs a doubly constrained furness for max_iters or until tol is met

    Parameters
    ----------
    seed_vals:
        Initial values for the furness. Must be of shape
        (len(n_rows), len(n_cols)).

    row_targets:
        The target values for the sum of each row

    col_targets:
        The target values for the sum of each column

    tol:
        The maximum difference between the achieved and the target values
        to tolerate before exiting early. R^2 is used to calculate the
        difference.

    max_iters:
        The maximum number of iterations to complete before exiting.

    Returns
    -------
    furnessed_matrix:
        The final furnessed matrix
    """
    # Error check
    if seed_vals.shape != (len(row_targets), len(col_targets)):
        raise ValueError(
            "The shape of the seed values given does not match the row "
            "and col targets. Seed_vals are shape %s. Expected shape (%d, %d)."
            % (str(seed_vals.shape), len(row_targets), len(col_targets))
        )

    # Init
    furnessed_mat = seed_vals.copy()

    for i in range(max_iters):
        # ## ROW CONSTRAIN ## #
        # Calculate difference factor
        row_ach = np.sum(furnessed_mat, axis=1)
        row_ach = np.where(row_ach == 0, 1, row_ach)
        diff_factor = row_targets / row_ach

        # adjust rows
        furnessed_mat = (furnessed_mat.T * diff_factor).T

        # ## COL CONSTRAIN ## #
        # Calculate difference factor
        col_ach = np.sum(furnessed_mat, axis=0)
        col_ach = np.where(col_ach == 0, 1, col_ach)
        diff_factor = col_targets / col_ach

        # adjust cols
        furnessed_mat = furnessed_mat * diff_factor

        # Calculate the diff - leave early if met
        row_diff = (row_targets - np.sum(furnessed_mat, axis=1)) ** 2
        col_diff = (col_targets - np.sum(furnessed_mat, axis=0)) ** 2
        cur_diff = np.sum(row_diff + col_diff) ** 0.5
        if cur_diff < tol:
            break

        if np.isnan(cur_diff):
            return np.zeros(furnessed_mat.shape)

    return furnessed_mat


def _distribute_pa_internal(productions,
                            attraction_weights,
                            seed_dist_dir,
                            trip_origin,
                            year,
                            p,
                            m,
                            seg,
                            ca,
                            tp,
                            zone_col,
                            p_col,
                            m_col,
                            seg_col,
                            ca_col,
                            tp_col
                            ):
    """
    Internal function of distribute_pa(). See that for full documentation.
    """
    # Init
    productions = productions.copy()
    attraction_weights = attraction_weights.copy()

    # Read in the seed distribution
    seed_fname = du.get_dist_name(
        trip_origin=trip_origin,
        matrix_format='pa',
        purpose=str(p),
        mode=str(m),
        segment=str(seg),
        car_availability=str(ca),
        tp=str(tp),
        csv=True
    )
    seed_dist = pd.read_csv(os.path.join(seed_dist_dir, seed_fname))

    # TODO: Make sure the seed values are the same size as P/A
    # TODO: Add in zone subset stuff?
    # if self.use_zone_id_subset:
    #     zone_subset = [259, 267, 268, 270, 275, 1171, 1173]
    #     synth_dists = du.get_data_subset(
    #         synth_dists, 'p_zone', zone_subset)
    #     synth_dists = du.get_data_subset(
    #         synth_dists, 'a_zone', zone_subset)

    # ## FILTER P/A TO SEGMENTATION ## #
    str_seg = str(seg) if seg is not None else None
    base_filter = {
        p_col: [p],
        m_col: [m],
        seg_col: [str_seg],
        ca_col: [ca],
        tp_col: [tp]
    }

    productions = du.filter_by_segmentation(productions,
                                            df_filter=base_filter,
                                            fit=True)
    attraction_weights = du.filter_by_segmentation(attraction_weights,
                                                   df_filter=base_filter,
                                                   fit=True)

    # Rename columns for furness
    productions = productions.rename(columns={str(year): "production_forecast"})
    attraction_weights = attraction_weights.rename(columns={str(year): "attraction_forecast"})

    # ## MATCH P/A ZONES ## #
    if productions.empty:
        raise ValueError("Something has gone wrong. I "
                         "have no productions.")

    if attraction_weights.empty:
        raise ValueError("Something has gone wrong. I "
                         "have no productions.")

    # TODO: Fix how this is done - might introduce an error like this
    #  Use same method developed for mode shares in attractions

    # Match the production and attraction zones
    prod_cols = list(productions)
    att_cols = list(attraction_weights)

    # Outer join makes sure all model zones will get values
    pa_input = pd.merge(
        productions,
        attraction_weights,
        on=zone_col,
        how='outer'
    ).fillna(0)

    productions = pa_input.reindex(prod_cols, axis='columns').copy()
    attraction_weights = pa_input.reindex(att_cols, axis='columns').copy()

    # TODO: write this properly after dev
    target_percentage = 0.975
    max_iters = 5000
    constrain_on_production = True
    constrain_on_attraction = True
    zero_replacement_value = 0.001
    echo = True

    seed_dist = seed_dist.rename(columns={'norms_zone_id': 'p_zone'}).melt(
        id_vars=['p_zone'],
        var_name='a_zone',
        value_name='seed_values'
    )

    # Furness the productions and attractions
    final_distribution = furness(
        productions=productions,
        attractions=attraction_weights,
        distributions=seed_dist,
        max_iters=max_iters,
        constrain_on_production=constrain_on_production,
        constrain_on_attraction=constrain_on_attraction,
        zero_replacement_value=zero_replacement_value,
        target_percentage=target_percentage,
        echo=echo
    )

    # sort out and save furnessed val

    exit()
    return

    ########### OLD DISTRBUTION BELOW #########

    # TODO: Output files while it runs, instead of at the end!
    productions = productions.copy()
    attraction_weights = attraction_weights.copy()
    final_distribution_dictionary = {}

    # Make sure the soc and ns columns are strings
    productions['soc'] = productions['soc'].astype(str)
    productions['ns'] = productions['ns'].astype(str)

    # TODO: Move inside of all nested loops into function (stops the
    #  indentation from making difficult to read code)
    # TODO: Move mode out to nested loops
    # TODO: Tidy this up
    # TODO: Generate synth_dists path based on segmentation
    #  and file location given
    for year in years_needed:
        for purpose in p_needed:
            # ns/soc depends on purpose
            if purpose in [1, 2]:
                required_segments = soc_needed
            else:
                required_segments = ns_needed

            for segment in required_segments:
                car_availability_dataframe = pd.DataFrame
                first_iteration = True
                for car_availability in ca_needed:

                    # for tp in required_times:
                    dist_path = os.path.join(
                        distribution_file_location,
                        distribution_dataframe_dict[purpose][segment][car_availability]
                    )

                    # Convert from wide to long format
                    # (needed for furnessing)
                    synth_dists = pd.read_csv(dist_path)
                    synth_dists = pd.melt(
                        synth_dists,
                        id_vars=['norms_zone_id'],
                        var_name='a_zone',
                        value_name='seed_values'
                    ).rename(
                        columns={"norms_zone_id": "p_zone"})

                    # convert column object to int
                    synth_dists['a_zone'] = synth_dists['a_zone'].astype(int)
                    synth_dists = synth_dists.groupby(
                        by=["p_zone", "a_zone"],
                        as_index=False
                    ).sum()

                    if self.use_zone_id_subset:
                        zone_subset = [259, 267, 268, 270, 275, 1171, 1173]
                        synth_dists = du.get_data_subset(
                            synth_dists, 'p_zone', zone_subset)
                        synth_dists = du.get_data_subset(
                            synth_dists, 'a_zone', zone_subset)

                    # Generate productions input
                    if purpose in [1, 2]:
                        segment_mask = (
                            (productions["purpose_id"] == purpose)
                            & (productions["car_availability_id"] == car_availability)
                            & (productions["soc"] == str(segment))
                        )
                    else:
                        segment_mask = (
                            (productions["purpose_id"] == purpose)
                            & (productions["car_availability_id"] == car_availability)
                            & (productions["ns"] == str(segment))
                        )

                    production_input = productions[segment_mask][
                        ["model_zone_id", str(year)]
                    ].rename(columns={str(year): "production_forecast"})

                    # Generate attractions input
                    mask = attraction_weights["purpose_id"] == purpose
                    attraction_input = attraction_weights[mask][
                        ["model_zone_id", str(year)]
                    ].rename(columns={str(year): "attraction_forecast"})

                    # ## MATCH P/A ZONES ## #
                    if production_input.empty:
                        raise ValueError("Something has gone wrong. I "
                                         "have no productions.")

                    if attraction_input.empty:
                        raise ValueError("Something has gone wrong. I "
                                         "have no productions.")

                    # Match the production and attraction zones
                    p_cols = list(production_input)
                    a_cols = list(attraction_input)

                    # Outer join makes sure all model zones will get values
                    pa_input = pd.merge(
                        production_input,
                        attraction_input,
                        on='model_zone_id',
                        how='outer'
                    ).fillna(0)

                    production_input = pa_input.reindex(p_cols, axis='columns').copy()
                    attraction_input = pa_input.reindex(a_cols, axis='columns').copy()

                    # Furness the productions and attractions
                    target_percentage = 0.7 if self.use_zone_id_subset else 0.975
                    final_distribution = fp.furness(
                        productions=production_input,
                        attractions=attraction_input,
                        distributions=synth_dists,
                        max_iters=max_iters,
                        constrain_on_production=constrain_on_production,
                        constrain_on_attraction=constrain_on_attraction,
                        zero_replacement_value=zero_replacement_value,
                        target_percentage=target_percentage,
                        echo=echo
                    )

                    final_distribution["purpose_id"] = purpose
                    final_distribution["car_availability_id"] = car_availability
                    final_distribution[year] = year

                    final_distribution["mode_id"] = required_mode
                    final_distribution = final_distribution[[
                        "p_zone",
                        "a_zone",
                        "mode_id",
                        "dt"
                     ]]

                    # Rename to the common output names
                    final_distribution = final_distribution.rename(columns={
                        "mode_id": "m",
                        "dt": "trips"
                    })

                    # TODO: Make sure this works for NHB trips too

                    final_distribution_mode = final_distribution.copy()
                    final_distribution_mode = final_distribution_mode[[
                        'p_zone', 'a_zone', 'trips'
                    ]]

                    dict_string = du.get_dist_name(
                        str(trip_origin),
                        'pa',
                        str(year),
                        str(purpose),
                        str(required_mode),
                        str(segment),
                        str(car_availability)
                    )

                    final_distribution_dictionary[dict_string] = final_distribution_mode

                    print("Distribution " + dict_string + " complete!")
                    if first_iteration:
                        car_availability_dataframe = final_distribution_mode
                        first_iteration = False
                    else:
                        car_availability_dataframe = car_availability_dataframe.append(
                            final_distribution_mode
                            )

    return final_distribution_dictionary


def distribute_pa(productions: pd.DataFrame,
                  attraction_weights: pd.DataFrame,
                  zone_areatype_lookup: pd.DataFrame,
                  seed_dist_dir: str,
                  years_needed: List[str],
                  p_needed: List[int],
                  m_needed: List[int],
                  soc_needed: List[int] = None,
                  ns_needed: List[int] = None,
                  ca_needed: List[int] = None,
                  tp_needed: List[int] = None,
                  zone_col: str = 'model_zone_id',
                  p_col: str = 'purpose_id',
                  m_col: str = 'mode_id',
                  soc_col: str = 'soc',
                  ns_col: str = 'ns',
                  ca_col: str = 'car_availability_id',
                  tp_col: str = 'tp',
                  trip_origin: str = 'hb',
                  max_iters: int = 5000,
                  constrain_on_production: bool = True,
                  constrain_on_attraction: bool = True,
                  zero_replacement_value: float = 0.00001,
                  echo: bool = False
                  ) -> None:
    """
    # TODO: Write distribute_pa() docs
    """
    # Init
    productions = productions.copy()
    attraction_weights = attraction_weights.copy()
    soc_needed = [None] if soc_needed is None else soc_needed
    ns_needed = [None] if ns_needed is None else ns_needed
    ca_needed = [None] if ca_needed is None else ca_needed
    tp_needed = [None] if tp_needed is None else tp_needed

    # Make sure the soc and ns columns are strings
    productions['soc'] = productions['soc'].astype(str)
    productions['ns'] = productions['ns'].astype(str)

    # Get P/A columns
    p_cols = list(productions.columns)
    for year in years_needed:
        p_cols.remove(year)

    a_cols = list(attraction_weights.columns)
    for year in years_needed:
        a_cols.remove(year)

    # TODO: Fix area_type_id in production model
    if 'area_type_id' in p_cols:
        p_cols.remove('area_type_id')
        productions = productions.drop('area_type_id', axis='columns')
        productions = productions.groupby(p_cols).sum().reset_index()

    # Distribute P/A per segmentation required
    for year in years_needed:
        # Filter P/A for this year
        p_index = p_cols.copy() + [year]
        productions = productions.reindex(p_index, axis='columns')

        a_index = a_cols.copy() + [year]
        attraction_weights = attraction_weights.reindex(a_index, axis='columns')

        # Loop through segmentations for this year
        loop_generator = du.segmentation_loop_generator(
            p_list=p_needed,
            m_list=m_needed,
            soc_list=soc_needed,
            ns_list=ns_needed,
            ca_list=ca_needed,
            tp_list=tp_needed,
        )

        # TODO: Multiprocess distributions
        for p, m, seg, ca, tp in loop_generator:
            if p in consts.SOC_P:
                seg_col = soc_col
            elif p in consts.NS_P:
                seg_col = ns_col,
            elif p in consts.ALL_NHB_P:
                seg_col = None
            else:
                raise ValueError("'%s' does not seem to be a valid soc, ns, or "
                                 "nhb purpose." % str(p))

            _distribute_pa_internal(
                productions=productions,
                attraction_weights=attraction_weights,
                seed_dist_dir=seed_dist_dir,
                trip_origin=trip_origin,
                year=year,
                p=p,
                m=m,
                seg=seg,
                ca=ca,
                tp=tp,
                zone_col=zone_col,
                p_col=p_col,
                m_col=m_col,
                seg_col=seg_col,
                ca_col=ca_col,
                tp_col=tp_col,
            )


def furness(productions: pd.DataFrame,
            attractions: pd.DataFrame,
            distributions: pd.DataFrame,
            max_iters: int = 1000,
            replace_zero_values: bool = True,
            constrain_on_production: bool = True,
            constrain_on_attraction: bool = True,
            zero_replacement_value: float = 0.01,
            target_percentage: float = 0.7,
            exit_early_tol: int = 40,
            audit_outputs: bool = False,
            echo=True,
            zone_col: str = 'model_zone_id'
            ) -> pd.DataFrame:
    """
    Run function for the Furness Process class.

    Provides a full Furness with options for number of
    iterations, whether to replace zero values on the
    seed, whether to constrain on production or attraction
    or both and what to replace zero values on the seed
    with.

    TODO: Make use of max_iters

    Parameters
    ----------
    productions:
        A dataframe with the columns "model_zone_id" and
        "production_forecast".

    attractions:
        A dataframe with the columns "model_zone_id" and
        "attraction_forecast".

    distributions:
        A dataframe with the columns "p_zone", "a_zone"
        and "seed_values".

    number_of_iterations:
        Amount of Furness iterations to be done.
        Default value is 1.

    replace_zero_values:
        Whether to replace zero values in the seed
        distribution dataframe.
        Default value is True.
        Possible values are True and False.

    constrain_on_production:
        Whether to constraining on production.
        Defaults to True.
        Possible values are True and False.

    constrain_on_attraction:
        Whether to constraining on attraction.
        Defaults to True.
        Possible values are True and False.

    zero_replacement_value:
        What value to replace zero values in the seed
        distribution dataframe with.
        Defaults to 0.01.

    target_percentage:
        TODO: Write this docstring

    audit_outputs:
        Whether to produce audit outputs.
        Defaults to True.
        Possible values are True and False.

    Return
    ----------
    furnessed_frame:
        The complete final furnessed frame with the
        columns "p_zone", "a_zone" and "dt".
    """
    # Grab only the necessary columns
    productions = productions[["model_zone_id", "production_forecast"]].copy()
    attractions = attractions[["model_zone_id", "attraction_forecast"]].copy()
    distributions = distributions[["p_zone", "a_zone", "seed_values"]].copy()

    # Ensure correct formats
    productions['model_zone_id'] = productions['model_zone_id'].astype(int)
    attractions['model_zone_id'] = attractions['model_zone_id'].astype(int)
    distributions['p_zone'] = distributions['p_zone'].astype(int)
    distributions['a_zone'] = distributions['a_zone'].astype(int)

    # Get a set of production and attraction zone ids (for checks)
    production_zones = set(productions["model_zone_id"].tolist())
    attraction_zones = set(attractions["model_zone_id"].tolist())

    # Get a set of distribution zone ids (for checks)
    distribution_zones = set(distributions["p_zone"].tolist())

    # ensure production and attraction zones match
    if production_zones != attraction_zones:
        raise ValueError("In FurnessProcess.run(): "
                         + "Production and attraction zones "
                         + "do not match.")

    # checking production total versus attraction total
    if (productions["production_forecast"].sum()
            !=
            attractions["attraction_forecast"].sum()):
        print("In FurnessProcess.run(): "
              + "Attraction forecast and production forecast do not match, "
              + "shifting attraction forecast to match to production "
              + "forecast.")

        # production and attraction do not match
        # need to scale attraction forecast to production forecast
        attractions["attraction_forecast"] /= (
            attractions["attraction_forecast"].sum()
            /
            productions["production_forecast"].sum()
        )

    if replace_zero_values:
        zero_seed_mask = (distributions["seed_values"] == 0)
        distributions.loc[zero_seed_mask, "seed_values"] = zero_replacement_value

    # Get percentage of productions to each a_zone from each p_zone
    ph = list()
    for zone in distribution_zones:
        temp_dists = distributions[distributions["p_zone"] == zone].copy()
        temp_dists['seed_values'] /= temp_dists['seed_values'].sum()
        ph.append(temp_dists)
    distributions = pd.concat(ph)

    # Loop Init
    furnessed_frame = pd.merge(
        distributions.copy(),
        productions,
        left_on='p_zone',
        right_on='model_zone_id'
    ).drop('model_zone_id', axis='columns')

    furnessed_frame = pd.merge(
        furnessed_frame,
        attractions,
        left_on='a_zone',
        right_on='model_zone_id'
    ).drop('model_zone_id', axis='columns')

    # Calculate trip distributions
    furnessed_frame["dt"] = 0
    furnessed_frame["dt"] = (
        (
            (
                furnessed_frame["seed_values"].values
                *
                furnessed_frame["production_forecast"].values
            )
            +
            (
                furnessed_frame["seed_values"].values
                *
                furnessed_frame["attraction_forecast"].values
            )
        )
        /
        2
    )

    # Calculate the total production of each zone
    production_zone_total = furnessed_frame[["p_zone", "dt"]].groupby(
        by=["p_zone"],
        as_index=False
    ).sum()
    production_zone_total.rename(
        columns={"dt": "production_zone_total"},
        inplace=True
    )

    production_zone_total = pd.merge(
        production_zone_total,
        furnessed_frame[["p_zone", "production_forecast"]].drop_duplicates(
            subset=["p_zone"]
        ),
        on="p_zone"
    )

    # Calculate production accuracy
    production_zone_total["production_zone_accuracy"] = (
        abs(
            (
                production_zone_total["production_forecast"].values
                /
                production_zone_total["production_zone_total"].values
            )
            -
            1
        )
    )

    # Calculate the total attraction of each zone
    attraction_zone_total = furnessed_frame[["a_zone", "dt"]].groupby(
        by=["a_zone"],
        as_index=False
    ).sum()
    attraction_zone_total.rename(
        columns={
            "dt": "attraction_zone_total"
        },
        inplace=True
    )

    attraction_zone_total = pd.merge(
        attraction_zone_total,
        furnessed_frame[["a_zone", "attraction_forecast"]].drop_duplicates(
            subset=["a_zone"]
        ),
        on="a_zone"
    )

    # Calculate attraction accuracy
    attraction_zone_total["attraction_zone_accuracy"] = (
        abs(
            (
                attraction_zone_total["attraction_forecast"].values
                /
                attraction_zone_total["attraction_zone_total"].values
            )
            -
            1
        )
    )

    pa_acc = (
        1
        -
        (
            (
                production_zone_total["production_zone_accuracy"].mean()
                +
                attraction_zone_total["attraction_zone_accuracy"].mean()
            )
            /
            2
        )
    )

    furnessed_frame = pd.merge(
        furnessed_frame,
        production_zone_total[["p_zone", "production_zone_total"]],
        on="p_zone"
    )
    furnessed_frame = pd.merge(
        furnessed_frame,
        attraction_zone_total[["a_zone", "attraction_zone_total"]],
        on="a_zone"
    )

    # TODO: Refactor this to avoid the code duplication before entering while loop
    i = 1

    best_pa_acc = (pa_acc, i)

    while pa_acc < target_percentage:
        du.print_w_toggle("Distribution iteration: %d" % i, echo=echo)
        du.print_w_toggle("Distribution iteration: %.9f" % pa_acc, echo=echo)
        if constrain_on_production and constrain_on_attraction:
            furnessed_frame["dt"] = (
                (
                    (
                        furnessed_frame["dt"].values
                        *
                        (
                            furnessed_frame["production_forecast"].values
                            /
                            furnessed_frame["production_zone_total"].values
                        )
                    )
                    +
                    (
                        furnessed_frame["dt"].values
                        *
                        (
                            furnessed_frame["attraction_forecast"].values
                            /
                            furnessed_frame["attraction_zone_total"].values
                        )
                    )
                )
                / 2
            )

            # Calculate the total production of each zone
            production_zone_total = furnessed_frame[["p_zone", "dt"]]
            production_zone_total = production_zone_total.groupby(
                by=["p_zone"],
                as_index=False
            ).sum()
            production_zone_total.rename(
                columns={
                    "dt": "production_zone_total"
                },
                inplace=True
            )

            production_zone_total = pd.merge(
                production_zone_total,
                furnessed_frame[["p_zone", "production_forecast"]].drop_duplicates(
                    subset=["p_zone"]
                ),
                on="p_zone"
            )

            # Calculate production accuracy
            production_zone_total["production_zone_accuracy"] = (
                abs(
                    (
                        production_zone_total["production_forecast"].values
                        /
                        production_zone_total["production_zone_total"].values
                    )
                    -
                    1
                )
            )

            # Calculate the total attraction of each zone
            attraction_zone_total = furnessed_frame[["a_zone", "dt"]]
            attraction_zone_total = attraction_zone_total.groupby(
                by=["a_zone"],
                as_index=False
            ).sum()
            attraction_zone_total.rename(
                columns={"dt": "attraction_zone_total"},
                inplace=True
            )

            attraction_zone_total = pd.merge(
                attraction_zone_total,
                furnessed_frame[["a_zone", "attraction_forecast"]].drop_duplicates(
                    subset=["a_zone"]
                ),
                on="a_zone"
            )

            # Calculate attraction accuracy
            attraction_zone_total["attraction_zone_accuracy"] = (
                abs(
                    (
                        attraction_zone_total["attraction_forecast"].values
                        /
                        attraction_zone_total["attraction_zone_total"].values
                    )
                    -
                    1
                )
            )

            pa_acc = (
                1
                -
                (
                    (
                        production_zone_total["production_zone_accuracy"].mean()
                        +
                        attraction_zone_total["attraction_zone_accuracy"].mean()
                    )
                    /
                    2
                )
            )

            furnessed_frame = furnessed_frame.drop(
                columns=["production_zone_total", "attraction_zone_total"],
                axis=1
            )
            furnessed_frame = pd.merge(
                furnessed_frame,
                production_zone_total[["p_zone", "production_zone_total"]],
                on="p_zone"
            )
            furnessed_frame = pd.merge(
                furnessed_frame,
                attraction_zone_total[["a_zone", "attraction_zone_total"]],
                on="a_zone"
            )

            # TODO: Write a log of pa_acc achieved for each distribution
            #  Warn user when the target cannot be met
            # Log the best pa_acc achieved
            if pa_acc > best_pa_acc[0]:
                best_pa_acc = (pa_acc, i)
            else:
                # Exit early if we have been stuck here a while
                if i - best_pa_acc[1] > exit_early_tol:
                    print("WARNING: Couldn't reach target accuracy! Exiting "
                          "furnessing early...")
                    break

            # TODO: Turn this into a for loop
            i = i + 1

    # if we're performing checks to see drift of production vs trips
    if audit_outputs:
        # do this for each zone
        for zone in production_zones:
            # pre-distribution predictions
            zone_production = productions[
                productions["model_zone_id"] == zone
                ][
                "production_forecast"
            ].values[0]

            zone_attraction = attractions[
                attractions["model_zone_id"] == zone
                ][
                "attraction_forecast"
            ].values[0]

            # post-distribution predictions
            zone_from = furnessed_frame[
                furnessed_frame["p_zone"] == zone
            ].sum()["dt"]
            zone_to = furnessed_frame[
                furnessed_frame["a_zone"] == zone
            ].sum()["dt"]

            # difference between post and pre distribution predictions
            production_difference = zone_from - zone_production
            attraction_difference = zone_to - zone_attraction

            # print outputs
            # TODO: Offer option to build and save to csv
            print("Audit outputs for Zone " + str(zone) + ":")
            print("----------------------------------")
            print("Predicted Production from Zone " + str(zone) + ": " + str(zone_production))
            print("Trips from Zone " + str(zone) + ": " + str(zone_from))
            print("Difference is: " + str(production_difference))
            print("----------------------------------")
            print("Predicted Attraction to Zone " + str(zone) + ": " + str(zone_attraction))
            print("Trips to Zone " + str(zone) + ": " + str(zone_to))
            print("Difference is: " + str(attraction_difference))
            print("----------------------------------")
            print("")

    # return the completed furnessed frame
    return furnessed_frame[["p_zone", "a_zone", "dt"]]
