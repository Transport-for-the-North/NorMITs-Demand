# -*- coding: utf-8 -*-
"""
Created on: Tues May 25th 15:04:32 2021
Updated on:

Original author: Ben Taylor
Last update made by:
Other updates made by:

File purpose:
Holds the SegmentationLevel Class which stores all information on segmentations
"""
# Builtins

# Third Party

# Local Imports


class SegmentationLevel:

    def __init__(self,
                 name: str,
                 ):
        # Init
        self.name = name

        # Should find data on this segmentation_level and save all to this object
        # May need to instantiate a location to look for data on ND import.

        raise NotImplementedError()


def get_segmentation_level(name: str) -> SegmentationLevel:
    # Should validate name as a segmentation_level, instantiate an object and return it

    raise NotImplementedError()
