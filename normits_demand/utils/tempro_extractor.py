# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 12:34:24 2020

@author: genie
"""

import pyodbc
import os

import pandas as pd
import numpy as np

from typing import List

class TemproParser:
    """
    """
    def __init__(self,
                 access_driver = '{Microsoft Access Driver (*.mdb, *.accdb)}',
                 data_source: str = 'C:/Program Files (x86)/TEMPRO7/DATA/',
                 region_list: list = ['EAST_', 'EM_', 'LON_', 'NE_', 'NW_',
                                'SCOTLAND_', 'SE_', 'SW_', 'WALES_',
                                'WM_', 'YH_'],
                 output_years: List[int] = [2018, 2033, 2035, 2050],
                 out_folder: str = 'Y:/NorMITs Synthesiser/import/ntem_constraints',
                 ):
        """
        """

        print('Tempro extractor running')
        
        self.access_driver = access_driver
        
        if not os.path.exists(data_source):
            raise ValueError('Tempro not installed at' + data_source)

        self.data_source = data_source
        self.region_list = region_list
        self.output_years = output_years
        self.out_folder = out_folder

    def parse_tempro(self,
                     trip_type = 'pa',  
                     aggregate_car = True):

        """
        trip_type = 'pa' or 'od'
        
        aggregate_car: Takes True or False
            Very important for car demand. If aggregate = False will take Tempro
            mode 3 only - ie. growth in car drivers.
            If False, will add Modes 3 & 4, so car driver and passenger.
        """
        access_driver = self.access_driver
        
        data_source = self.data_source
        region_list = self.region_list

        output_years = self.output_years
        out_folder = self.out_folder

        if trip_type == 'pa':
            tt_q = '(1,2)'
            col_a = 'Productions'
            col_b = 'Attractions'
        else:
            tt_q = '(3,4)'
            col_a = 'Origin'
            col_b = 'Destination'

        ## Good stuff
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

        read_dbs = []
        db_list = [x for x in os.listdir(data_source) if '.mdb' in x]
        for db in db_list:
            for region in region_list:
                if region in db:
                    read_dbs.append(db)
                    break

        # TODO: Check there's the full whack of regions here - say which aren't - error if any North missing

        # Iterate over all databases
        # TODO: multithread wrapper

        db_ph = []
        for db in read_dbs:
            print(db)
        
            # Connect
            conn_string = ('Driver=' + access_driver +
                           ';DBQ=' + data_source + db + ';')
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

            # Close db
            conn.close()

            # Get years
            av_years = [int(x) for x in list(trip_ends) if x.isdigit()]
            year_index = []
            year_dicts = []
            for year in output_years:
        
                if year > 2051:
                    print('Impossible to interpolate past 2051')
                    break
                else:
                    year_index.append(str(year))
                    if year in av_years:
                        year_dicts.append({'t_year':year,
                                           'start_year':year,
                                           'end_year':year})
                    else:
                        year_diff = np.array([year - x for x in av_years])
                        # Get lower than
                        ly = np.argmin(np.where(year_diff>0, year_diff, 100))
                        ly = av_years[ly]
                        # Get greater than
                        hy = np.argmax(np.where(year_diff<0, year_diff, -100))
                        hy = av_years[hy]

                        year_dicts.append({'t_year':year,
                                           'start_year':ly,
                                           'end_year':hy})

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
                                        how = 'left',
                                        on = 'ZoneID')

            # TODO: Join LA (as NTEM) to new LA (lookup)
            # Nightmare because NTEM zone id != NTEM_zone_id - have to go round the houses
            trip_ends = trip_ends.merge(ntem_zone_lookup,
                                        how = 'inner',
                                        on = 'ZoneName')

            trip_ends = trip_ends.merge(ntem_code_lookup,
                                        how = 'inner',
                                        on = 'ntem_id')

            trip_ends = trip_ends.merge(gb_ntem_lookup,
                                        how='inner',
                                        left_on = 'Zone_ID',
                                        right_on ='ntem_zone_id')

            # Reindex
            group_cols = ['lad_zone_id', 'Purpose', 'Mode', 'TimePeriod', 'TripType']
            target_cols = group_cols.copy()
            for year in year_index:
                target_cols.append(year)

            # Compile segments (mode 3&4 == 3, purpose 11 & 12 == 12)
            # Weekdays only - ave weekday = weekday / 5 - see below
            trip_ends['Purpose'] = trip_ends['Purpose'].replace([11],12)
        
            if aggregate_car:
                trip_ends['Mode'] = trip_ends['Mode'].replace(4,3)

            trip_ends = trip_ends[
                    trip_ends['TimePeriod'].isin([1,2,3,4])].reset_index(drop=True)

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
                    index = pivot_cols,
                    columns = ['TripType'],
                    values = str(year)).reset_index()
            # Rename
            single_year = single_year.rename(
                    columns={1:col_a, 2:col_b,
                             3:col_a, 4:col_b})

            # Build outname
            out_path = os.path.join(out_folder,
                                    'ntem_' +
                                    trip_type +
                                    '_ave_wday_' + str(year) + '.csv')

            # write
            single_year.to_csv(out_path, index=False)
            return('Done' + trip_type)

# TODO: Ask Nhan - about VO application here

if __name__ == '__main__':

    pa = TemproParser(out_folder=r'C:\Users\Genie\Documents\Tempro',
                      output_years = [2015])
    od = TemproParser(out_folder=r'C:\Users\Genie\Documents\Tempro',
                      output_years = [2015])

    pa.parse_tempro(trip_type = 'pa',
                    aggregate_car = True)

    od.parse_tempro(trip_type = 'od',
                    aggregate_car = True)