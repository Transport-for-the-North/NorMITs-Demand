# -*- coding: utf-8 -*-
"""
Created on: Tues May 25th 15:04:32 2021
Updated on:

Original author: Ben Taylor
Last update made by:
Other updates made by:

File purpose:
Holds the ZoningSystem Class which stores all information on different zoning
systems
"""
# Builtins

# Third Party

# Local Imports


class ZoningSystem:

    def __init__(self,
                 name: str,
                 ):
        # Init
        self.name = name

        # Should find data on this zoning system and save all to this object
        # May need to instantiate a location to look for data on ND import.

        raise NotImplementedError()


def get_zoning_system(name: str) -> ZoningSystem:
    # Should validate name as a zoning system, instantiate an object and return it

    raise NotImplementedError()
