# -*- coding: utf-8 -*-
"""
Created on: Tues May 25th 15:04:32 2021
Updated on:

Original author: Ben Taylor
Last update made by:
Other updates made by:

File purpose:
Holds custom normits_demand objects, such as DVector and its functions
"""
# Builtins

# Third Party

# Local Imports
from normits_demand import core


class DVector:

    def __init__(self,
                 zoning_system: core.ZoningSystem,
                 segmentation: core.SegmentationLevel,
                 ):
        # Init
        self.zoning_system = zoning_system
        self.segmentation = segmentation

        # Need to decide on a format to read the data in as. Maybe helper
        # functions in the module to convert df / np into this structure??

        raise NotImplementedError()
