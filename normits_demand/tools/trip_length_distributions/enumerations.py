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


@enum.unique
class GeoArea(nd_enum.IsValidEnumWithAutoNameLower):
    """Valid geographical area filters for the TLD builder"""
    NORTH = enum.auto()
    NORTH_AND_MIDS = enum.auto()
    GB = enum.auto()

    def get_lads(self) -> list[str]:
        """Gets the Local Authority Districts relevant to this geo area"""
        if self == GeoArea.NORTH:
            return GeoArea._north_lads()
        if self == GeoArea.NORTH_AND_MIDS:
            return list(set(GeoArea._north_lads() + GeoArea._midlands_lads()))

        raise nd.NormitsDemandError(
            f"No definition exits for LADs for GeoArea `{self}`"
        )

    def get_gors(self) -> list[int]:
        """Gets the Government Office Regions relevant to this geo area"""
        if self == GeoArea.NORTH:
            return GeoArea._north_gors()
        if self == GeoArea.NORTH_AND_MIDS:
            return list(set(GeoArea._north_gors() + GeoArea._midlands_gors()))

        if self == GeoArea.GB:
            return GeoArea._gb_gors()

        raise nd.NormitsDemandError(
            f"No definition exits for GORs for GeoArea `{self}`"
        )

    @staticmethod
    def _north_gors() -> list[int]:
        return [1, 2, 3]

    @staticmethod
    def _midlands_gors() -> list[int]:
        return [4, 5]

    @staticmethod
    def _gb_gors() -> list[int]:
        return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    @staticmethod
    def _north_lads() -> list[str]:
        # fmt: off
        return [
            'E06000001', 'E06000002', 'E06000003', 'E06000004', 'E06000005', 'E06000006',
            'E06000007', 'E06000008', 'E06000009', 'E06000010', 'E06000011', 'E06000012',
            'E06000013', 'E06000014', 'E06000021', 'E06000047', 'E06000049', 'E06000050',
            'E06000057', 'E07000026', 'E07000027', 'E07000028', 'E07000029', 'E07000030',
            'E07000031', 'E07000033', 'E07000034', 'E07000035', 'E07000037', 'E07000038',
            'E07000117', 'E07000118', 'E07000119', 'E07000120', 'E07000121', 'E07000122',
            'E07000123', 'E07000124', 'E07000125', 'E07000126', 'E07000127', 'E07000128',
            'E07000137', 'E07000142', 'E07000163', 'E07000164', 'E07000165', 'E07000166',
            'E07000167', 'E07000168', 'E07000169', 'E07000170', 'E07000171', 'E07000174',
            'E07000175', 'E07000198', 'E08000001', 'E08000002', 'E08000003', 'E08000004',
            'E08000005', 'E08000006', 'E08000007', 'E08000008', 'E08000009', 'E08000010',
            'E08000011', 'E08000012', 'E08000013', 'E08000014', 'E08000015', 'E08000016',
            'E08000017', 'E08000018', 'E08000019', 'E08000021', 'E08000022', 'E08000023',
            'E08000024', 'E08000032', 'E08000033', 'E08000034', 'E08000035', 'E08000036',
            'E08000037', 'W06000001', 'W06000002', 'W06000003', 'W06000004', 'W06000005',
            'W06000006',
        ]
        # fmt: on

    @staticmethod
    def _midlands_lads() -> list[str]:
        # fmt: off
        return [
            'E06000015', 'E06000016', 'E06000017', 'E06000018', 'E07000032', 'E07000033',
            'E07000034', 'E07000035', 'E07000036', 'E07000037', 'E07000038', 'E07000039',
            'E07000129', 'E07000130', 'E07000131', 'E07000132', 'E07000133', 'E07000134',
            'E07000135', 'E07000136', 'E07000137', 'E07000138', 'E07000139', 'E07000140',
            'E07000141', 'E07000142', 'E07000150', 'E07000151', 'E07000152', 'E07000153',
            'E07000154', 'E07000155', 'E07000156', 'E07000170', 'E07000171', 'E07000172',
            'E07000173', 'E07000174', 'E07000175', 'E07000176', 'E06000019', 'E06000020',
            'E06000021', 'E06000051', 'E07000192', 'E07000193', 'E07000194', 'E07000195',
            'E07000196', 'E07000197', 'E07000198', 'E07000199', 'E07000234', 'E07000235',
            'E07000236', 'E07000237', 'E07000238', 'E07000239', 'E07000218', 'E07000219',
            'E07000220', 'E07000221', 'E07000222', 'E08000025', 'E08000026', 'E08000027',
            'E08000028', 'E08000029', 'E08000030', 'E08000031'
        ]
        # fmt: on


@enum.unique
class TripFilter(nd_enum.IsValidEnumWithAutoNameLower):
    """Valid trip filter types for the TLD builder"""
    TRIP_OD = enum.auto()
    TRIP_O = enum.auto()
    TRIP_D = enum.auto()

    def is_od_type(self) -> bool:
        return self in self._od_type()

    @staticmethod
    def _od_type() -> list[TripFilter]:
        return [TripFilter.TRIP_OD, TripFilter.TRIP_O, TripFilter.TRIP_D]


class CostUnits(nd_enum.IsValidEnum):
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
class SampleTimePeriods(nd_enum.IsValidEnumWithAutoNameLower):
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
