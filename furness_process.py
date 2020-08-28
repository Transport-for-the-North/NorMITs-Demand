# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 09:42:25 2019

Candidate for generalised Furness Process for Normits
Utilities.

@author: Robert Farrar
"""
import pandas as pd
import numpy as np


class FurnessProcess:
    def run(self,
            production_dataframe: pd.DataFrame,
            attraction_dataframe: pd.DataFrame,
            distribution_dataframe: pd.DataFrame,
            number_of_iterations: int = 1,
            replace_zero_values: bool = True,
            constrain_on_production: bool = True,
            constrain_on_attraction: bool = True,
            zero_replacement_value: float = 0.01,
            target_percentage: float = 0.7,
            audit_outputs: bool = False
            ) -> pd.DataFrame:
        """
        Run function for the Furness Process class.
        
        Provides a full Furness with options for number of
        iterations, whether to replace zero values on the
        seed, whether to constrain on production or attraction
        or both and what to replace zero values on the seed
        with.
        
        Parameters
        ----------
        production_dataframe:
            A dataframe with the columns "model_zone_id" and
            "production_forecast".
            
        attraction_dataframe:
            A dataframe with the columns "model_zone_id" and
            "attraction_forecast".
            
        distribution_dataframe:
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
        production_dataframe = production_dataframe[[
            "model_zone_id", "production_forecast"
        ]]

        attraction_dataframe = attraction_dataframe[[
            "model_zone_id", "attraction_forecast"
        ]]

        distribution_dataframe = distribution_dataframe[[
            "p_zone", "a_zone", "seed_values"
        ]]

        # Get a set of production and attraction zone ids (for checks)
        production_zones = set(production_dataframe["model_zone_id"].tolist())
        attraction_zones = set(attraction_dataframe["model_zone_id"].tolist())

        # Get a set of distribution zone ids (for checks)
        distribution_zones = set(distribution_dataframe["p_zone"].tolist())

        # ensure production and attraction zones match
        if production_zones != attraction_zones:
            raise ValueError("In FurnessProcess.run(): "
                             + "Production and attraction zones "
                             + "do not match.")

        # checking production total versus attraction total
        if (production_dataframe["production_forecast"].sum()
                !=
                attraction_dataframe["attraction_forecast"].sum()):
            print("In FurnessProcess.run(): "
                  + "Attraction forecast and production forecast do not match, "
                  + "shifting attraction forecast to match to production "
                  + "forecast.")

            # production and attraction do not match
            # need to scale attraction forecast to production forecast
            total_attraction_by_total_production = (
                    attraction_dataframe["attraction_forecast"].sum()
                    /
                    production_dataframe["production_forecast"].sum()
            )

            attraction_dataframe["attraction_forecast"] = (
                    attraction_dataframe["attraction_forecast"].values
                    /
                    total_attraction_by_total_production
            )

        # if we need to replace zero values
        if replace_zero_values:
            # fill zero values
            zero_seed_mask = (distribution_dataframe["seed_values"] == 0)
            distribution_dataframe.loc[zero_seed_mask, "seed_values"] = zero_replacement_value

        # for each zone
        for zone in distribution_zones:
            # divide seed values by total on p_zone to get
            # percentages
            zone_mask = (distribution_dataframe["p_zone"] == zone)
            distribution_dataframe.loc[zone_mask, "seed_values"] = (
                    distribution_dataframe[zone_mask]["seed_values"].values
                    /
                    distribution_dataframe[zone_mask]["seed_values"].sum()
            )

        # set up furnessed_frame and first_iteration flag
        furnessed_frame = distribution_dataframe.copy()

        furnessed_frame["production_forecast"] = 0
        furnessed_frame["attraction_forecast"] = 0

        for zone in production_zones:
            # Copy the production values for this zone into furnessed frame
            p_zone_mask = furnessed_frame["p_zone"] == zone
            furnessed_frame.loc[p_zone_mask, "production_forecast"] = (
                production_dataframe[
                    production_dataframe["model_zone_id"] == zone
                    ][
                    "production_forecast"
                ].values[0]
            )

            # Copy the attraction values for this zone into furnessed frame
            a_zone_mask = (furnessed_frame["a_zone"] == zone)
            furnessed_frame.loc[a_zone_mask, "attraction_forecast"] = (
                attraction_dataframe[
                    attraction_dataframe["model_zone_id"] == zone
                    ][
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

        mean_correct_percentage = (
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
        print()
        furnessed_frame = pd.merge(
            furnessed_frame,
            attraction_zone_total[["a_zone", "attraction_zone_total"]],
            on="a_zone"
        )
        print()

        # TODO: Refactor this to avoid the code duplication before entering while loop
        iteration_counter = 1

        while mean_correct_percentage < target_percentage:
            print("Current distribution iteration: " + str(iteration_counter))
            print("Current distribution iteration: " + str(round(mean_correct_percentage, 10)))
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

                mean_correct_percentage = (
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

                iteration_counter = iteration_counter + 1

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
                zone_production = production_dataframe[
                    production_dataframe["model_zone_id"] == zone
                ][
                    "production_forecast"
                ].values[0]

                zone_attraction = attraction_dataframe[
                    attraction_dataframe["model_zone_id"] == zone
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
