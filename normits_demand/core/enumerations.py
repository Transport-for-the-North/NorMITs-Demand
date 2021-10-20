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
@enum.unique
class Mode(enum.Enum):
    WALK = 'walk'
    CYCLE = 'cycle'
    ACTIVE = 'walk_and_cycle'
    CAR = 'car_and_passenger'
    BUS = 'bus'
    RAIL = 'rail'
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
            Mode.RAIL: [6],
            Mode.TRAM: [7],
        }

        # TODO(BT): Check that self is in conversion and error

        return conversion[self]
