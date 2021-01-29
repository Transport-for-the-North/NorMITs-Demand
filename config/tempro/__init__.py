# -*- coding: utf-8 -*-

import pandas as pd

if __name__ == '__main__':
    code_to_zone = pd.read_csv('ntem_code_to_zone.csv')
    geo_lookup = pd.read_csv('tblLookupGeo76.csv')
