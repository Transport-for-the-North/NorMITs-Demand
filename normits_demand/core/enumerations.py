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
from typing import Any, Dict, List

# Local Imports
import normits_demand as nd


# ## CLASSES ## #
class AutoName(enum.Enum):
    """Enum class to automatically use the Enum name for it's value."""

    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        del start, count, last_values  # Unused
        return name


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


class IsValidEnumWithAutoNameLower(IsValidEnum):
    """Enum class to combine IsValidEnum and AutoName

    Must be a better way to do this, but inheriting both classes seems to not
    produce the expected results

    TODO(BT): Investigate a better way to combine these classes
    """

    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        del start, count, last_values  # Unused
        return name.lower()


@enum.unique
class Mode(IsValidEnum):
    """Collection of valid modes and their values/names"""

    WALK = "walk"
    CYCLE = "cycle"
    ACTIVE = "walk_and_cycle"
    CAR = "car_and_passenger"
    BUS = "bus"
    RAIL = "rail"
    TRAIN = "train"
    TRAM = "tram"

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
            raise nd.NormitsDemandError(f"No definition exists for {self} mode_values")

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
        """Gets the name of this mode"""
        return self.value


@enum.unique
class Scenario(IsValidEnum):
    """Collection of valid Scenario names"""

    NTEM = "NTEM"
    SC01_JAM = "SC01_JAM"
    SC02_PP = "SC02_PP"
    SC03_DD = "SC03_DD"
    SC04_UZC = "SC04_UZC"
    DLOG = "DLOG"

    @staticmethod
    def tfn_scenarios():
        """Gets a list of the TfN Future Travel Scenarios"""
        return [
            Scenario.SC01_JAM,
            Scenario.SC02_PP,
            Scenario.SC03_DD,
            Scenario.SC04_UZC,
        ]


@enum.unique
class TripOrigin(IsValidEnum):
    """Collection of valid trip origins"""

    HB = "hb"
    NHB = "nhb"

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
            f"No TripOrigin exists with the value '{val}'. " f"Expected one of: {valid_values}"
        )


@enum.unique
class UserClass(IsValidEnumWithAutoNameLower):
    """Collection of valid User Classes and linked purposes"""

    COMMUTE = enum.auto()
    BUSINESS = enum.auto()
    OTHER = enum.auto()

    HB_COMMUTE = enum.auto()
    HB_BUSINESS = enum.auto()
    HB_OTHER = enum.auto()
    NHB_BUSINESS = enum.auto()
    NHB_OTHER = enum.auto()

    def get_purposes(self):
        """Returns a list of purposes for this UserClass"""
        p_dict = UserClass.get_purpose_dict()

        if self not in p_dict:
            raise ValueError(
                f"Internal error. There doesn't seem to be a purpose "
                f"definition for UserClass {self.value!r}"
            )

        return p_dict[self]

    @staticmethod
    def get_purpose_dict() -> Dict[UserClass, List[int]]:
        """Returns a dictionary of purposes for each UserClass"""
        p_dict = {
            UserClass.HB_COMMUTE: [1],
            UserClass.HB_BUSINESS: [2],
            UserClass.HB_OTHER: [3, 4, 5, 6, 7, 8],
            UserClass.NHB_BUSINESS: [11, 12],
            UserClass.NHB_OTHER: [13, 14, 15, 16, 17, 18],
        }

        # Add combinations of other values
        b_val = p_dict[UserClass.HB_BUSINESS] + p_dict[UserClass.NHB_BUSINESS]
        p_dict[UserClass.BUSINESS] = b_val

        p_dict[UserClass.COMMUTE] = p_dict[UserClass.HB_COMMUTE]
        p_dict[UserClass.OTHER] = p_dict[UserClass.HB_OTHER] + p_dict[UserClass.NHB_OTHER]

        return p_dict


@enum.unique
class MatrixFormat(IsValidEnum):
    """Collection of valid matrix formats"""

    PA = "pa"
    OD = "od"
    OD_TO = "od_to"
    OD_FROM = "od_from"


class CostUnits(IsValidEnum):
    """Valid cost units for the TLD builder"""

    KM = "km"
    KILOMETRES = "km"
    KILOMETERS = "km"

    M = "m"
    METRES = "m"
    METERS = "m"

    MILES = "miles"

    def get_conversion_factor(self, to_units: CostUnits):
        """Calculates the conversion factors between cost units

        Returns the conversion factor to get to `to_units` CostUnits from
        self

        Parameters
        ----------
        to_units:
            The cost units to convert to.
        """
        if self == to_units:
            return 1
        if self == CostUnits.MILES and to_units == CostUnits.KM:
            return self._miles_to_km_factor()
        if self == CostUnits.MILES and to_units == CostUnits.M:
            return self._miles_to_m_factor()

        if self == CostUnits.KM and to_units == CostUnits.MILES:
            return self._km_to_miles_factor()
        if self == CostUnits.KM and to_units == CostUnits.M:
            return self._km_to_m_factor()

        if self == CostUnits.M and to_units == CostUnits.MILES:
            return self._m_to_miles_factor()
        if self == CostUnits.M and to_units == CostUnits.KM:
            return self._m_to_km_factor()

        raise nd.NormitsDemandError(
            f"No definition exits to convert from {self} to {to_units}"
        )

    @staticmethod
    def _miles_to_km_factor() -> float:
        return 1.609344

    @staticmethod
    def _km_to_m_factor() -> float:
        return 1000

    @staticmethod
    def _miles_to_m_factor() -> float:
        return CostUnits._miles_to_km_factor() * CostUnits._km_to_m_factor()

    @staticmethod
    def _m_to_km_factor() -> float:
        return 1 / CostUnits._km_to_m_factor()

    @staticmethod
    def _km_to_miles_factor() -> float:
        return 1 / CostUnits._miles_to_km_factor()

    @staticmethod
    def _m_to_miles_factor() -> float:
        return 1 / CostUnits._miles_to_m_factor()


@enum.unique
class AssignmentModel(IsValidEnum):
    """Network assignment models NorMITs demand is used with."""

    NOHAM = "NoHAM"
    NORMS = "NoRMS"
    # Models for Midlands Connect
    MIHAM = "MiHAM"
    MIRANDA = "MiRANDA"

    @classmethod
    def from_str(cls, model_name: str) -> AssignmentModel:
        """Parse string and return AssignmentModel if valid.

        Parameters
        ----------
        model_name : str
            Name of the assignment model.

        Returns
        -------
        AssignmentModel
            AssignmentModel enum with the given name.

        Raises
        ------
        ValueError
            If no assignment models exist with `model_name`.
        """
        name = model_name.strip().lower()
        for model in cls:
            if model.value.lower() == name:
                return model

        raise ValueError(f"'{model_name}' isn't a valid AssignmentModel")

    def get_name(self) -> str:
        """Return the model name."""
        return self.value

    def get_zoning_system(self) -> nd.ZoningSystem:
        """Return the zone system for the assignment model."""
        return nd.get_zoning_system(self.get_name().lower())

    @classmethod
    def mode_lookup(cls) -> Dict[AssignmentModel, Mode]:
        """Dictionary lookup for the assignment model modes."""
        return {
            cls.NOHAM: Mode.CAR,
            cls.NORMS: Mode.TRAIN,
            cls.MIHAM: Mode.CAR,
            cls.MIRANDA: Mode.TRAIN,
        }

    def get_mode(self) -> Mode:
        """Return the assignment model mode."""
        return self.mode_lookup()[self]

    @classmethod
    def tfn_models(cls) -> set[AssignmentModel]:
        """Transport for the North's assignment models."""
        return {cls.NOHAM, cls.NORMS}


class TripEndType(enum.StrEnum):
    """Defined trip end types."""

    HB_PRODUCTIONS = enum.auto()
    HB_ATTRACTIONS = enum.auto()
    NHB_PRODUCTIONS = enum.auto()
    NHB_ATTRACTIONS = enum.auto()

    @staticmethod
    def trip_origin_lookup() -> dict[TripEndType, TripOrigin]:
        return {
            TripEndType.HB_PRODUCTIONS: TripOrigin.HB,
            TripEndType.HB_ATTRACTIONS: TripOrigin.HB,
            TripEndType.NHB_PRODUCTIONS: TripOrigin.NHB,
            TripEndType.NHB_ATTRACTIONS: TripOrigin.NHB,
        }

    @property
    def trip_origin(self) -> TripOrigin:
        return self.trip_origin_lookup()[self]
    
    def formatted(self) -> str:
        """Format text for outputs and display purposes."""
        to, pa = self.value.split("_")
        return f"{to.upper()} {pa.title()}"
