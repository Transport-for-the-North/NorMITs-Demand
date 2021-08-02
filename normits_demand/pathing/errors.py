# -*- coding: utf-8 -*-
"""
Created on: Mon August 2 2021
Updated on:

Original author: Ben Taylor
Last update made by:
Other updates made by:

File purpose:
Module of errors for the pathing modules
"""
# Builtins

# Third Party

# Local imports
import normits_demand as nd


class PathingError(nd.NormitsDemandError):
    """
    Base Exception for all NorMITs Demand Pathing errors
    """

    def __init__(self, message=None):
        self.message = message
        super().__init__(self.message)
