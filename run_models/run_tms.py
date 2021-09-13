# -*- coding: utf-8 -*-
"""
Created on: 09/09/2021
Updated on:

Original author: Ben Taylor
Last update made by:
Other updates made by:

File purpose:

"""
# Built-Ins
import sys

# Third Party

# Local Imports
sys.path.append("..")
from normits_demand.models import TravelMarketSynthesiser


def main():

    tms = TravelMarketSynthesiser()
    tms.run()


if __name__ == '__main__':
    main()
