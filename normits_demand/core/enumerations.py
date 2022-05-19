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

from typing import Any
from typing import Dict
from typing import List

# Third Party

# Local Imports


# ## CLASSES ## #
import normits_demand as nd


class IsValidEnum(enum.Enum):
    """Enum with helper functions to check if a given value is valid"""

    @classmethod
    def to_list(cls):
        """Convert Enum into a list of Enums"""
        return list(cls)

    @classmethod
    def values_to_list(cls):
        """Convert Enum into a list of Enum values"""
        return [x.value for x in list(cls)]

    @classmethod
    def to_enum(cls, value: Any) -> enum.Enum:
        """Converts a value to a member of this enum class"""
        # NOTE: enum("value") does the same thing
        if isinstance(value, IsValidEnum):
            return value
        return cls(value)

    @classmethod
    def is_valid(cls, value: Any) -> bool:
        """Checks if a value is a valid member of this enum"""
        if isinstance(value, IsValidEnum):
            return value in cls.to_list()

        # Try convert to enum, if fails, it isn't valid
        success = True
        try:
            cls(value)
        except ValueError:
            success = False
        return success


@enum.unique
class Mode(IsValidEnum):
    """Collection of valid modes and their values/names"""
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
                f"No definition exists for {self} mode_values"
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
                f"Mode {self.value} has more than one mode value. "
                f"If you want to return multiple mode values, use "
                f"Mode.get_mode_values() instead."
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
class Scenario(IsValidEnum):
    """Collection of valid Scenario names"""
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
class TripOrigin(IsValidEnum):
    """Collection of valid trip origins"""
    HB = 'hb'
    NHB = 'nhb'

    def get_name(self):
        return self.value

    def get_purposes(self):
        """Returns a list of purposes for this TripOrigin"""
        p_dict = TripOrigin.get_purpose_dict()

        if self not in p_dict:
            raise ValueError(
                f"Internal error. There doesn't seem to be a purpose "
                f"definition for TripOrigin {self.value}"
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
            f"No TripOrigin exists with the value '{val}'. "
            f"Expected one of: {valid_values}"
        )


@enum.unique
class MatrixFormat(IsValidEnum):
    """Collection of valid matrix formats"""
    PA = 'pa'
    OD = 'od'
    OD_TO = 'od_to'
    OD_FROM = 'od_from'
