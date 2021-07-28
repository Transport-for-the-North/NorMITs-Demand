# -*- coding: utf-8 -*-
"""
Created on: Wed July 28 2021
Updated on:

Original author: Ben Taylor
Last update made by:
Other updates made by:

File purpose:
Holds all the errors related to the elasticity model
"""

import normits_demand as nd


class ElasticityError(nd.NormitsDemandError):
    """
    Base Exception for all custom NorMITS demand errors
    """

    def __init__(self, message=None):
        self.message = message
        super().__init__(self.message)

