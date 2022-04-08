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
from __future__ import annotations

# Built-Ins
import enum

from typing import Dict
from typing import List

# Third Party

# Local Imports


# ## CLASSES ## #
import normits_demand as nd


class AutoName(enum.Enum):
    """Enum class to automatically use the Enum name for it's value."""

    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        del start, count, last_values  # Unused
        return name

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


@enum.unique
class Scenario(enum.Enum):
    NTEM = 'NTEM'
    SC01_JAM = 'SC01_JAM'
    SC02_PP = 'SC02_PP'
    SC03_DD = 'SC03_DD'
    SC04_UZC = 'SC04_UZC'

    def get_name(self):
        return self.value

    @staticmethod
    def tfn_scenarios():
        return [
            Scenario.SC01_JAM,
            Scenario.SC02_PP,
            Scenario.SC03_DD,
            Scenario.SC04_UZC,
        ]


@enum.unique
class TripOrigin(enum.Enum):
    HB = 'hb'
    NHB = 'nhb'

    def get_name(self):
        return self.value

    def get_purposes(self):
        """Returns a list of purposes for this TripOrigin"""
        p_dict = TripOrigin.get_purpose_dict()

        if self not in p_dict.keys():
            raise ValueError(
                "Internal error. There doesn't seem to be a purpose definition "
                "for TripOrigin %s"
                % self.value
            )

        return p_dict[self]

    @staticmethod
    def get_purpose_dict() -> Dict[TripOrigin, List[int]]:
        """Returns a dictionary of purposes for each TripOrigin"""
        return {
            TripOrigin.HB: [1, 2, 3, 4, 5, 6, 7, 8],
            TripOrigin.NHB: [12, 13, 14, 15, 16, 18],
        }

    @staticmethod
    def get_trip_origin(val: str):
        """Returns a TripOrigin object with value val"""
        valid_values = list()
        for to in TripOrigin:
            valid_values.append(to.value)
            if to.value == val:
                return to

        raise ValueError(
            "No TripOrigin exists with the value '%s'. Expected one of: %s"
            % (val, valid_values)
        )
