# -*- coding: utf-8 -*-
"""
Created on: Tues May 25th 15:04:32 2021
Updated on:

Original author: Ben Taylor
Last update made by:
Other updates made by:

File purpose:
Holds the NoTEM Class for calling all production and attraction models
"""
# Builtins

from typing import List

# Third Party

# Local Imports
import normits_demand as nd


class NoTEM:

    def __init__(self,
                 years: List[int],
                 scenario: str,
                 land_use_import_drive: nd.PathLike,
                 land_use_iter: str,
                 ):
        # Init
        self.years = years
        self.scenario = scenario
        self.land_use_import_drive = land_use_import_drive
        self.land_use_iter = land_use_iter

        raise NotImplementedError()
