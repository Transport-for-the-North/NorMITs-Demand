# -*- coding: utf-8 -*-
"""
Created on: 30/09/2021
Updated on:

Original author: Ben Taylor
Last update made by:
Other updates made by:

File purpose:
Collection of standard enumerations
"""
# Built-Ins
import enum

# Third Party

# Local Imports


# ## CLASSES ## #
import normits_demand as nd


@enum.unique
class Mode(enum.Enum):
    WALK = 'walk'
    CYCLE = 'cycle'
    ACTIVE = 'walk_and_cycle'
    CAR = 'car_and_passenger'
    BUS = 'bus'
    RAIL = 'rail'
    TRAIN = 'train'
    TRAM = 'tram'

    def get_mode_values(self):
        """Conversion from enum to modes"""
        # Define conversion
        conversion = {
            Mode.WALK: [1],
            Mode.CYCLE: [2],
            Mode.ACTIVE: [1, 2],
            Mode.CAR: [3],
            Mode.BUS: [5],
            Mode.TRAIN: [6],
            Mode.TRAM: [7],
            Mode.RAIL: [6, 7],
        }

        if self not in conversion:
            raise nd.NormitsDemandError(
                "No definition exists for %s mode_values"
                % self
            )

        return conversion[self]

    def get_mode_num(self):
        """
        Get a single mode num for this mode

        If self.get_mode_values() would return more than one mode value,
        then this function will throw and error instead.
        """
        mode_vals = self.get_mode_values()

        if len(mode_vals) == 1:
            return mode_vals[0]

        if len(mode_vals) > 1:
            raise ValueError(
                "Mode %s has more than one mode value. If you want to return "
                "multiple mode values, use Mode.get_mode_values() instead."
                % self.value
            )

        # Must somehow have returned nothing?
        raise ValueError(
            "The call to self.get_mode_values() returned an item of len() < 1."
            "Check this function to make sure it is returning what it should "
            "be!"
        )

    def get_name(self):
        return self.value
