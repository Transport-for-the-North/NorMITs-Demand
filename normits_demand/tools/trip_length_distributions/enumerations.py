# -*- coding: utf-8 -*-
"""
Created on: 01/06/2022
Updated on:

Original author: Ben Taylor
Last update made by:
Other updates made by:

File purpose:

"""
from __future__ import annotations

# Built-Ins
import enum

# Third Party

# Local Imports
import normits_demand as nd
from normits_demand.core import enumerations as nd_enum


# enumerations for geo areas, output paths etc.
# Might move some into general enumerations if they are needed too!
#   Externally needed if making a TLD class to handle these properly

_cost_units = ["km", "miles", "m"]


@enum.unique
class GeoArea(nd_enum.AutoName, nd_enum.IsValidEnum):
    """Valid geographical area filters for the TLD builder"""
    NORTH = enum.auto()
    NORTH_AND_MIDS = enum.auto()
    GB = enum.auto()


@enum.unique
class TripFilter(nd_enum.AutoName, nd_enum.IsValidEnum):
    """Valid trip filter types for the TLD builder"""
    TRIP_OD = enum.auto()
    TRIP_O = enum.auto()
    TRIP_D = enum.auto()


class CostUnits(nd_enum.IsValidEnum):
    """Valid cost units for the TLD builder"""
    KM = "KM"
    KILOMETRES = "KM"
    KILOMETERS = "KM"

    M = "M"
    METRES = "M"
    METERS = "M"

    MILES = "MILES"

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
class SampleTimePeriods(nd_enum.AutoName, nd_enum.IsValidEnum):
    """Valid time periods filters for the TLD builder"""
    FULL_WEEK = enum.auto()
    WEEK_DAYS = enum.auto()
    WEEK_ENDS = enum.auto()

    def get_time_periods(self) -> list[int]:
        """Gets the relevant time periods for this sample type"""
        if self == SampleTimePeriods.FULL_WEEK:
            return self._week_day_time_periods() + self._week_end_time_periods()
        if self == SampleTimePeriods.WEEK_DAYS:
            return self._week_day_time_periods()
        if self == SampleTimePeriods.WEEK_ENDS:
            return self._week_end_time_periods()

        raise nd.NormitsDemandError(
            f"No valid time period definition exists for {self}"
        )

    @staticmethod
    def _week_day_time_periods() -> list[int]:
        return [1, 2, 3, 4]

    @staticmethod
    def _week_end_time_periods() -> list[int]:
        return [5, 6]
