# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 16:43:09 2020

Wrapper script to run TMS start to finish.
# Steps as follows
# *Optional

# Import params sheet
# Check import folders
# Run required lookups
# Run production model
# Run attraction model
# Run cjtw translation
# Run distribution model
# Run external model
# Compile tp pa
# *Run pa to od
# Run nhb production model
# Compile tp nhb pa
# *Compile aggregate matrices
# *Translate trips to vehicles

"""

import os

import pandas as pd

import normits_demand.demand as dem

class TravelMarketSynthesiser( dem.NormitsDemand ):

    """
    """
    pass
