"""
File purpose:
A file of constants to be used.
Keeps all constants in one place, for all files in the project to refer to,
and keeps the code more readable.
"""

import os

# Census Journey to Work parameters
CJTW_PATH = 'Y:/Data Strategy/Data/ct_stage/WU03UK_msoa_v3/'
CJTW_DAT = 'wu03uk_msoa_v3.csv'
CJTW_HEADER = 'WU03UK_msoa.txt'

_SHAPES_PATH = 'Y:/Data Strategy/GIS Shapefiles/'
Z_HYBRID_MSOA = os.path.join(
    _SHAPES_PATH,
    'UK MSOA 2011 IZ 2001 Hybrid/UK MSOA 2011 IZ 2001 Hybrid.shp')
