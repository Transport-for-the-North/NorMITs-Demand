# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 12:34:24 2020

@author: genie

Download access drivers:
https://www.microsoft.com/en-us/download/confirmation.aspx?id=54920
"""
# Builtins
import os

from typing import List

# Third Party
import pyodbc
import pandas as pd
import numpy as np
from tqdm import tqdm


class TemproParser:
    """
    """
    _access_driver = '{Microsoft Access Driver (*.mdb, *.accdb)}'
    _data_source = 'Y:\Data Strategy\Data\TEMPRO'
    _out_folder = r'I:\NorMITs Demand\import\ntem_constraints'
    _region_list = [
        'EAST_',
        'EM_',
        'LON_',
        'NE_',
        'NW_',
        'SCOTLAND_',
        'SE_',
        'SW_',
        'WALES_',
        'WM_',
        'YH_'
    ]
    _output_years = [2018, 2033, 2035, 2050]

    def __init__(self,
                 access_driver: str = None,
                 data_source: str = None,
                 region_list: List[str] = None,
                 output_years: List[int] = None,
                 out_folder: str = None,
                 ):
        """
        """
        print('Initialising tempro extractor...')

        # Set to default values if not passed in
        access_driver = self._access_driver if access_driver is None else access_driver
        data_source = self._data_source if data_source is None else data_source
        out_folder = self._out_folder if out_folder is None else out_folder
        region_list = self._region_list if region_list is None else region_list
        output_years = self._output_years if output_years is None else output_years

        if not os.path.exists(data_source):
            raise ValueError('Tempro not installed at' + data_source)

        # Assign variables
        self.access_driver = access_driver
        self.data_source = data_source
        self.region_list = region_list
        self.output_years = output_years
        self.out_folder = out_folder

        print('Tempro extractor running!')

    def parse_tempro(self,
                     trip_type: str = 'pa',
                     aggregate_car: bool = True):

        """
        trip_type = 'pa' or 'od'
        
        aggregate_car: Takes True or False
            Very important for car demand. If aggregate = False will take Tempro
            mode 3 only - ie. growth in car drivers.
            If False, will add Modes 3 & 4, so car driver and passenger.
        """
        if trip_type == 'pa':
            tt_q = '(1,2)'
            col_a = 'Productions'
            col_b = 'Attractions'
        else:
            tt_q = '(3,4)'
            col_a = 'Origin'
            col_b = 'Destination'

        # Good stuff
        ntem_zone_lookup = pd.read_csv(os.path.join(os.getcwd(),
                                                    'config',
                                                    'tempro',
                                                    'tblLookupGeo76.csv'))
        ntem_code_lookup = pd.read_csv(os.path.join(os.getcwd(),
                                                    'config',
                                                    'tempro',
                                                    'ntem_code_to_zone.csv'))
        # TODO: Move to code base
        gb_ntem_lookup = pd.read_csv(os.path.join(os.getcwd(),
                                                  'config',
                                                  'tempro',
                                                  'ntem_lad_pop_weighted_lookup.csv'))

        available_dbs = []
        db_list = [x for x in os.listdir(self.data_source) if '.mdb' in x]
        for db_fname in db_list:
            for region in self.region_list:
                if region in db_fname:
                    available_dbs.append(db_fname)
                    break

        if available_dbs == list():
            raise IOError("Couldn't find any dbs to load from.")

        # TODO: Check there's the full whack of regions here - say which aren't - error if any North missing

        # Iterate over all databases
        # TODO: multithread wrapper

        db_ph = []
        for db_fname in tqdm(available_dbs, desc="Extracting from DBs..."):
            # Connect
            db_path = os.path.join(self.data_source, db_fname)
            conn_string = (
                'Driver=' + self.access_driver + ';'
                'DBQ=' + db_path + ';'
            )
            conn = pyodbc.connect(conn_string)
            cursor = conn.cursor()

            # Get and unpack trip and and zone tables
            # run a query and get the results
            cursor.execute('select * from TripEndDataByDirection where TripType in ' + tt_q)
            trip_end_rows = cursor.fetchall()
        
            # Unpack trip end data
            trip_cols = [column[0] for column in cursor.description]

            trip_ends = []
            for row in trip_end_rows:
                trip_ends.append([x for x in row])

            trip_ends = pd.DataFrame(trip_ends)
            trip_ends.columns = trip_cols

            cursor.execute('select * from Zones')
            zone_rows = cursor.fetchall()

            # Unpack Zones Data
            zone_cols = [column[0] for column in cursor.description]
    
            zones = []
            for row in zone_rows:
                zones.append([x for x in row])

            zones = pd.DataFrame(zones)
            zones.columns = zone_cols

            # Close db_fname
            conn.close()

            # Get years
            av_years = [int(x) for x in list(trip_ends) if x.isdigit()]
            year_index = []
            year_dicts = []
            for year in self.output_years:
        
                if year > 2051:
                    print('Impossible to interpolate past 2051')
                    break
                else:
                    year_index.append(str(year))
                    if year in av_years:
                        year_dicts.append({'t_year': year,
                                           'start_year': year,
                                           'end_year': year})
                    else:
                        year_diff = np.array([year - x for x in av_years])
                        # Get lower than
                        ly = np.argmin(np.where(year_diff > 0, year_diff, 100))
                        ly = av_years[ly]
                        # Get greater than
                        hy = np.argmax(np.where(year_diff < 0, year_diff, -100))
                        hy = av_years[hy]

                        year_dicts.append({'t_year': year,
                                           'start_year': ly,
                                           'end_year': hy})

            # Interpolate mid point years if needed
            for year in year_dicts:
                print('Building ' + str(year))
                period_diff = year['end_year']-year['start_year']
                target_diff = year['t_year']-year['start_year']
                if target_diff > 0:
                    trip_ends['annual_growth'] = (trip_ends[
                        str(year['end_year'])] - trip_ends[
                            str(year['start_year'])])/period_diff
                    
                    trip_ends[str(year['t_year'])] = trip_ends[
                        str(year['start_year'])] + (
                            target_diff*trip_ends['annual_growth'])
                    trip_ends = trip_ends.drop('annual_growth',axis=1)

            # Add LA names
            trip_ends = trip_ends.merge(zones,
                                        how='left',
                                        on='ZoneID')

            # TODO: Join LA (as NTEM) to new LA (lookup)
            # Nightmare because NTEM zone id != NTEM_zone_id - have to go round the houses
            trip_ends = trip_ends.merge(ntem_zone_lookup,
                                        how='inner',
                                        on='ZoneName')

            trip_ends = trip_ends.merge(ntem_code_lookup,
                                        how='inner',
                                        on='ntem_id')

            trip_ends = trip_ends.merge(gb_ntem_lookup,
                                        how='inner',
                                        left_on='Zone_ID',
                                        right_on='ntem_zone_id')

            # Reindex
            group_cols = ['lad_zone_id', 'Purpose', 'Mode', 'TimePeriod', 'TripType']
            target_cols = group_cols.copy()
            for year in year_index:
                target_cols.append(year)

            # Compile segments (mode 3&4 == 3, purpose 11 & 12 == 12)
            # Weekdays only - ave weekday = weekday / 5 - see below
            trip_ends['Purpose'] = trip_ends['Purpose'].replace([11], 12)
        
            if aggregate_car:
                trip_ends['Mode'] = trip_ends['Mode'].replace(4, 3)

            trip_ends = trip_ends[
                    trip_ends['TimePeriod'].isin([1, 2, 3, 4])].reset_index(drop=True)

            # Aggregate @ LA
            trip_ends = trip_ends.reindex(target_cols, axis=1).groupby(
                    group_cols).sum().reset_index()

            # output constraint data.
            db_ph.append(trip_ends)

            # Subset by year & chunk out
            out_dat = pd.concat(db_ph)

        # Iterate over years, pivot out P/A - needs segments lookup like NTS script
        out_years = [int(x) for x in list(trip_ends) if x.isdigit()]

        # Test for unq zones
        for year in out_years:
            # Get year cols
            target_cols = group_cols.copy()
            target_cols.append(str(year))
            pivot_cols = group_cols.copy()
            pivot_cols.remove('TripType')

            # Reindex
            single_year = out_dat.reindex(target_cols, axis=1).groupby(
                    group_cols).sum().reset_index()
        
            # / 5 to get wday
            single_year[str(year)] = single_year[str(year)]/5

            # Pivot to PA
            single_year = single_year.pivot_table(
                    index=pivot_cols,
                    columns=['TripType'],
                    values=str(year)).reset_index()
            # Rename
            single_year = single_year.rename(
                    columns={1: col_a, 2: col_b,
                             3: col_a, 4: col_b})

            # Write to disk
            out_fname = "ntem_%s_ave_wday_%s.csv" % (trip_type, str(year))
            out_path = os.path.join(self.out_folder, out_fname)
            single_year.to_csv(out_path, index=False)

# TODO: Ask Nhan - about VO application here


if __name__ == '__main__':

    pa = TemproParser(out_folder=r'C:\Users\Genie\Documents\Tempro',
                      output_years=[2015])
    od = TemproParser(out_folder=r'C:\Users\Genie\Documents\Tempro',
                      output_years=[2015])

    pa.parse_tempro(trip_type='pa',
                    aggregate_car=True)

    od.parse_tempro(trip_type='od',
                    aggregate_car=True)
