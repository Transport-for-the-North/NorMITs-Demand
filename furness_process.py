# -*- coding: utf-8 -*-
"""
Created on: Mon Sep 30 09:42:25 2019
Updated on: Tues Sep 1 08:56:35 2020

Original Author: Robert Farrar
Last update made by: Ben Taylor

File Purpose:
Candidate for generalised Furness Process for Normits
Utilities.
"""

import pandas as pd
import numpy as np

def furness(productions: pd.DataFrame,
            attractions: pd.DataFrame,
            distributions: pd.DataFrame,
            max_iters: int = 1,
            replace_zero_values: bool = True,
            constrain_on_production: bool = True,
            constrain_on_attraction: bool = True,
            zero_replacement_value: float = 0.01,
            target_percentage: float = 0.7,
            exit_early_tol: int = 40,
            audit_outputs: bool = False,
            echo=True
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
    productions = productions[["model_zone_id", "production_forecast"]]
    attractions = attractions[["model_zone_id", "attraction_forecast"]]
    distributions = distributions[["p_zone", "a_zone", "seed_values"]]

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
        attractions["attraction_forecast"] = (
                attractions["attraction_forecast"].values
                /
                (
                    attractions["attraction_forecast"].sum()
                    /
                    productions["production_forecast"].sum()
                )
        )

    if replace_zero_values:
        zero_seed_mask = (distributions["seed_values"] == 0)
        distributions.loc[zero_seed_mask, "seed_values"] = zero_replacement_value

    # Get percentage of productions in each p_zone
    for zone in distribution_zones:
        zone_mask = (distributions["p_zone"] == zone)
        distributions.loc[zone_mask, "seed_values"] = (
                distributions[zone_mask]["seed_values"].values
                /
                distributions[zone_mask]["seed_values"].sum()
        )

    # Loop Init
    furnessed_frame = distributions.copy()
    furnessed_frame["production_forecast"] = 0
    furnessed_frame["attraction_forecast"] = 0

    for zone in production_zones:
        # Copy the production values into furnessed_frame
        p_zone_mask = (furnessed_frame["p_zone"] == zone)
        furnessed_frame.loc[p_zone_mask, "production_forecast"] = (
            productions[productions["model_zone_id"] == zone][
                "production_forecast"
            ].values[0]
        )

        # Copy the attraction values into furnessed_frame
        a_zone_mask = (furnessed_frame["a_zone"] == zone)
        furnessed_frame.loc[a_zone_mask, "attraction_forecast"] = (
            attractions[attractions["model_zone_id"] == zone][
            "attraction_forecast"
            ].values[0]
        )

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

    # print(furnessed_frame)
    # exit()

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


    # count iterations
    #### OLD VERSION
    #        for i in range(0, number_of_iterations):
    #
    #            # if we're constraining on production
    #            if (constrain_on_production):
    #                # we need to constrain on production axis
    #                if (first_iteration):
    #                    # if it's the first iteration/
    #                    # set up the initial values
    #                    furnessed_frame["dt"] =\
    #                         (
    #                                 furnessed_frame["production_forecast"].values
    #                                 *
    #                                 furnessed_frame["seed_values"].values
    #                                 )
    #                    # and uncheck the first iteration flag
    #                    first_iteration = False
    #                else:
    #                    # if it's the second iteration
    #                    # then we have the initial values
    #                    # we just need to constrain to production
    #                    # max
    #                    for zone in production_zones:
    #                        # we iterate over each zone
    #                        # create zone mask
    #                        furnessed_mask = furnessed_frame["p_zone"] == zone
    #
    #                        # change distributed trips values using zone mask
    #                        furnessed_frame.loc[
    #                                furnessed_mask,
    #                                "dt"
    #                                ] =\
    #                             (
    #                                     furnessed_frame[furnessed_mask]["dt"].values
    #                                     /
    #                                     (
    #                                             furnessed_frame[furnessed_mask]["dt"].sum()
    #                                             /
    #                                             furnessed_frame[furnessed_mask]["production_forecast"].values[0]
    #                                             )
    #                                     )
    #
    #            # if we're constraining on attraction
    #            if (constrain_on_attraction):
    #                # we need to constrain on attraction axis
    #                if (first_iteration):
    #                    # if it's the first iteration
    #                    # set up the initial values
    #                    furnessed_frame["dt"] =\
    #                         (
    #                                 furnessed_frame["attraction_forecast"].values
    #                                 *
    #                                 furnessed_frame["seed_values"].values
    #                                 )
    #                    # and uncheck the first iteration flag
    #                    first_iteration = False
    #                else:
    #                    # if it's not the first iteration
    #                    for zone in attraction_zones:
    #                        # we iterate over each zone
    #                        # create zone mask
    #                        furnessed_mask = furnessed_frame["a_zone"] == zone
    #
    #                        # change distributed trips values using zone mask
    #                        furnessed_frame.loc[
    #                                furnessed_mask,
    #                                "dt"
    #                                ] =\
    #                             (
    #                                     furnessed_frame[furnessed_mask]["dt"].values
    #                                     /
    #                                     (
    #                                             furnessed_frame[furnessed_mask]["dt"].sum()
    #                                             /
    #                                             furnessed_frame[furnessed_mask]["attraction_forecast"].values[0]
    #                                             )
    #                                     )

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


from demand_utilities import utils as du
