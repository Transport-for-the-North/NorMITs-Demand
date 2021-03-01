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
from typing import Tuple

# Third Party
import pyodbc
import pandas as pd
import numpy as np
from tqdm import tqdm

# local imports
from normits_demand import AuditError
from normits_demand import constants as consts
from normits_demand.models import efs_zone_translator as zt

from normits_demand.utils import general as du

# TODO(BT/CS): Functionalist query by DB and interpolate (separately)


class TemproParser:
    """
    """
    _access_driver = '{Microsoft Access Driver (*.mdb, *.accdb)}'
    _data_source = 'Y:\Data Strategy\Data\TEMPRO'
    _out_folder = r'I:\NorMITs Demand\import\ntem_constraints\ss'
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
    _output_years_str = [str(x) for x in _output_years]

    planning_data_types = {
        'under16': [1],
        '16-74': [2],
        '75+': [3],
        'total_pop': [1, 2, 3],
        'HHs': [5],
        'jobs': [6],
        'workers': [4],
    }
    
    co_data_types = {
        'no_car': [1],
        '1_car': [2],
        '2_cars': [3],
        '3+_cars': [4],
        'nca': [1],
        'ca': [2,3,4]}

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

        # Set up paths
        # TODO(BT): Update these paths to build more dynamically
        home_path = os.path.join(os.getcwd(), 'config', 'tempro')

        self.ntem_trans_path = os.path.join(home_path, 'tblLookupGeo76.csv')
        self.ntem_code_zone_trans_path = os.path.join(home_path, 'ntem_code_to_zone.csv')
        self.ntem_lad_trans_path = os.path.join(home_path, 'ntem_lad_pop_weighted_lookup.csv')
        self.ntem_to_msoa_path = r"I:\NorMITs Demand\import\zone_translation\weighted\ntem_msoa_pop_weighted_lookup.csv"

        print('Tempro extractor running!')

    def _get_co_data(self,
                     db_fname
                     ):
        
        # Init
        db_path = os.path.join(self.data_source, db_fname)
        conn_string = (
            'Driver=' + self.access_driver + ';'
            'DBQ=' + db_path + ';'
        )
        conn = None
        
        try:
            # Connect
            conn = pyodbc.connect(conn_string)
            cursor = conn.cursor()

            # Grab and unpack the planning data
            cursor.execute('select * from CarOwnership')

            co_ph = list()
            for row in cursor.fetchall():
                co_ph.append([x for x in row])

            co_data = pd.DataFrame(
                data=co_ph,
                columns=[column[0] for column in cursor.description]
            )

            print(list(co_data))

            # Grab and unpack local db zone translations
            cursor.execute('select * from Zones')

            zonal_ph = list()
            for row in cursor.fetchall():
                zonal_ph.append([x for x in row])

            zones = pd.DataFrame(
                data=zonal_ph,
                columns=[column[0] for column in cursor.description]
            )
        except BaseException as e:
            if conn is not None:
                conn.close()
            raise e
        finally:
            if conn is not None:
                conn.close()

        # Get years
        av_years = [int(x) for x in list(co_data) if x.isdigit()]
        year_index = list()
        year_dicts = list()
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
            period_diff = year['end_year'] - year['start_year']
            target_diff = year['t_year'] - year['start_year']
            if target_diff > 0:
                co_data['annual_growth'] = (
                    (
                        co_data[str(year['end_year'])]
                        - co_data[str(year['start_year'])]
                    )
                    / period_diff
                )

                co_data[str(year['t_year'])] = (
                    co_data[str(year['start_year'])]
                    + (target_diff * co_data['annual_growth'])
                )
                co_data = co_data.drop('annual_growth', axis=1)

        # ## SPLIT INTO POP AND JOBS ## #
        # population
        needed_types = self.co_data_types['nca']
        mask = co_data['CarOwnershipType'].isin(needed_types)
        nca = co_data[mask].copy()

        # jobs
        needed_types = self.co_data_types['ca']
        mask = co_data['CarOwnershipType'].isin(needed_types)
        ca = co_data[mask].copy()

        # ## ATTACH ZONE NAMES ## #
        nca = pd.merge(
            nca,
            zones,
            how='left',
            on='ZoneID'
        )

        ca = pd.merge(
            ca,
            zones,
            how='left',
            on='ZoneID'
        )
        return nca, ca

    def _get_pop_job_numbers(self,
                             db_fname
                             ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Init
        db_path = os.path.join(self.data_source, db_fname)
        conn_string = (
            'Driver=' + self.access_driver + ';'
            'DBQ=' + db_path + ';'
        )
        conn = None

        try:
            # Connect
            conn = pyodbc.connect(conn_string)
            cursor = conn.cursor()

            # Grab and unpack the planning data
            cursor.execute('select * from Planning')

            planning_ph = list()
            for row in cursor.fetchall():
                planning_ph.append([x for x in row])

            planning_data = pd.DataFrame(
                data=planning_ph,
                columns=[column[0] for column in cursor.description]
            )

            # Grab and unpack local db zone translations
            cursor.execute('select * from Zones')

            zonal_ph = list()
            for row in cursor.fetchall():
                zonal_ph.append([x for x in row])

            zones = pd.DataFrame(
                data=zonal_ph,
                columns=[column[0] for column in cursor.description]
            )
        except BaseException as e:
            if conn is not None:
                conn.close()
            raise e
        finally:
            if conn is not None:
                conn.close()

        # Get years
        av_years = [int(x) for x in list(planning_data) if x.isdigit()]
        year_index = list()
        year_dicts = list()
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
            period_diff = year['end_year'] - year['start_year']
            target_diff = year['t_year'] - year['start_year']
            if target_diff > 0:
                planning_data['annual_growth'] = (
                    (
                        planning_data[str(year['end_year'])]
                        - planning_data[str(year['start_year'])]
                    )
                    / period_diff
                )

                planning_data[str(year['t_year'])] = (
                    planning_data[str(year['start_year'])]
                    + (target_diff * planning_data['annual_growth'])
                )
                planning_data = planning_data.drop('annual_growth', axis=1)

        # ## SPLIT INTO POP AND JOBS ## #
        # population
        needed_types = self.planning_data_types['total_pop']
        mask = planning_data['PlanningDataType'].isin(needed_types)
        pop = planning_data[mask].copy()

        # jobs
        needed_types = self.planning_data_types['jobs']
        mask = planning_data['PlanningDataType'].isin(needed_types)
        emp = planning_data[mask].copy()

        # ## ATTACH ZONE NAMES ## #
        pop = pd.merge(
            pop,
            zones,
            how='left',
            on='ZoneID'
        )

        emp = pd.merge(
            emp,
            zones,
            how='left',
            on='ZoneID'
        )

        return pop, emp

    def _get_pa_trip_ends(self,
                          db_fname: str,
                          col_indices: str
                          ):
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
        cursor.execute('select * from TripEndDataByDirection where TripType in ' + col_indices)
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
            period_diff = year['end_year'] - year['start_year']
            target_diff = year['t_year'] - year['start_year']
            if target_diff > 0:
                trip_ends['annual_growth'] = (trip_ends[
                                                  str(year['end_year'])] - trip_ends[
                                                  str(year['start_year'])]) / period_diff

                trip_ends[str(year['t_year'])] = trip_ends[
                                                     str(year['start_year'])] + (
                                                         target_diff * trip_ends['annual_growth'])
                trip_ends = trip_ends.drop('annual_growth', axis=1)

        # Stick the zone names back on
        trip_ends = trip_ends.merge(zones,
                                    how='left',
                                    on='ZoneID')

        return trip_ends, year_index

    def _tempro_ntem_to_normits_ntem(self,
                                     df: pd.DataFrame,
                                     zone_col: str = 'ZoneName',
                                     return_zone_col: str = 'Zone_ID',
                                     val_cols: List[str] = None,
                                     tolerance: float = 0.05,
                                     ) -> pd.DataFrame:
        # Init
        val_cols = self._output_years_str if val_cols is None else val_cols

        ntem_trans = pd.read_csv(self.ntem_trans_path)
        ntem_code_trans = pd.read_csv(self.ntem_code_zone_trans_path)

        # validate
        for col in val_cols:
            if col not in df:
                raise ValueError(
                    "%s is not in the given df" % col
                )

        # Get starting totals
        start_totals = dict()
        for col in val_cols:
            start_totals[col] = df[col].sum()

        # Translate Tempro zone names to zone codes
        trip_ends = pd.merge(
            df,
            ntem_trans,
            how='left',
            on=zone_col
        )

        # Translate zone code to numbers
        trip_ends = pd.merge(
            trip_ends,
            ntem_code_trans,
            how='left',
            on='ntem_id'
        )

        # See if we're equal. Throw an error if not
        for col, val in start_totals.items():
            lower = val - (val*tolerance)
            upper = val + (val*tolerance)

            if not (lower < trip_ends[col].sum() < upper):
                raise AuditError(
                    "More than the tolerance of demand was dropped during zone "
                    "translation.\n"
                    "Column: %s\n"
                    "Demand before: %f\n"
                    "Demand after: %f\n"
                    % (col, val, trip_ends[col].sum())
                )

        return trip_ends.rename(columns={'Zone_ID': return_zone_col})

    def _get_growth_factors_internal(self,
                                     db_fname,
                                     col_indices,
                                     trip_origin: str = 'hb',
                                     pbar=None,
                                     ):
        """
        Grabs the trip ends from the database and converts to msoa zoning
        """
        # Init
        needed_purposes = list()
        trip_origin = trip_origin.strip().lower()
        if trip_origin == 'both' or trip_origin == 'hb':
            needed_purposes += consts.ALL_HB_P

        if trip_origin == 'both' or trip_origin == 'nhb':
            needed_purposes += consts.ALL_NHB_P

        if needed_purposes == list():
            raise ValueError(
                "Received an invalid trip origin. Expected one of ['hb', "
                "'nhb', 'both']. Got %s instead." % str(trip_origin)
            )

        # Grab the data from this database
        trip_ends, years = self._get_pa_trip_ends(db_fname, col_indices)
        trip_ends = self._tempro_ntem_to_normits_ntem(
            trip_ends,
            return_zone_col='ntem_zone_id',
        )

        # Filter down to just the needed time periods
        mask = (trip_ends['TimePeriod'].isin([1, 2, 3, 4]))
        trip_ends = trip_ends[mask].reset_index(drop=True)

        # Filter down to the needed purposes
        mask = (trip_ends['Purpose'].isin(needed_purposes))
        trip_ends = trip_ends[mask].reset_index(drop=True)

        # Aggregate up to just zones
        group_cols = ['ntem_zone_id', 'TripType']
        needed_cols = group_cols.copy() + [str(x) for x in self._output_years]
        trip_ends = trip_ends.reindex(columns=needed_cols)
        trip_ends = trip_ends.groupby(group_cols).sum().reset_index()

        # Translate to MSOA
        translator = zt.ZoneTranslator()
        trip_ends = translator.run(
            trip_ends,
            pd.read_csv(self.ntem_to_msoa_path),
            'ntem',
            'msoa',
            non_split_cols=group_cols,
        )

        # Tidy up and hold to join later
        group_cols = ['msoa_zone_id', 'TripType']
        needed_cols = group_cols.copy() + [str(x) for x in self._output_years]
        trip_ends = trip_ends.reindex(columns=needed_cols)
        trip_ends = trip_ends.groupby(group_cols).sum().reset_index()

        # Divide by 5 to get the average weekday
        for year in self._output_years_str:
            trip_ends[year] /= 5

        # ## SPLIT INTO P/A VECTORS ##
        # productions are trip type 1
        mask = (trip_ends['TripType'] == 1)
        productions = trip_ends[mask].copy()
        productions.drop(columns=['TripType'], inplace=True)

        # attractions are trip type 2
        mask = (trip_ends['TripType'] == 2)
        attractions = trip_ends[mask].copy()
        attractions.drop(columns=['TripType'], inplace=True)

        if pbar is not None:
            pbar.update(1)

        return productions, attractions

    def _get_pop_emp_factors_internal(self, db_fname, pbar=None):
        """
        Grabs the trip ends from the database and converts to msoa zoning
        """

        # Grab the data from this database
        pop, emp = self._get_pop_job_numbers(db_fname)

        # Translate local ntem to global ntem
        pop = self._tempro_ntem_to_normits_ntem(
            pop,
            return_zone_col='ntem_zone_id',
        )

        emp = self._tempro_ntem_to_normits_ntem(
            emp,
            return_zone_col='ntem_zone_id',
        )

        # ## AGGREGATE TO NEEDED DATA ONLY ## #
        group_cols = ['ntem_zone_id']
        needed_cols = group_cols.copy() + [str(x) for x in self._output_years]

        pop = pop.reindex(columns=needed_cols).groupby(group_cols).sum().reset_index()
        emp = emp.reindex(columns=needed_cols).groupby(group_cols).sum().reset_index()

        # ## TRANSLATE TO MSOA ## #
        translator = zt.ZoneTranslator()

        pop = translator.run(
            pop,
            pd.read_csv(self.ntem_to_msoa_path),
            'ntem',
            'msoa',
            non_split_cols=group_cols,
        )
        emp = translator.run(
            emp,
            pd.read_csv(self.ntem_to_msoa_path),
            'ntem',
            'msoa',
            non_split_cols=group_cols,
        )

        # Tidy up and hold to join later
        group_cols = ['msoa_zone_id']
        needed_cols = group_cols.copy() + [str(x) for x in self._output_years]

        pop = pop.reindex(columns=needed_cols).groupby(group_cols).sum().reset_index()
        emp = emp.reindex(columns=needed_cols).groupby(group_cols).sum().reset_index()

        if pbar is not None:
            pbar.update(1)

        return pop, emp

    def _get_co_growth_factors_internal(self,
                                        db_fname,
                                        pbar=None):
        """
        Get CO data from databse and convert to MSOA
        """
        nca, ca = self._get_co_data(db_fname)
        
        # Translate local ntem to global ntem
        nca = self._tempro_ntem_to_normits_ntem(
            nca,
            return_zone_col='ntem_zone_id',
        )
        
        ca = self._tempro_ntem_to_normits_ntem(
            ca,
            return_zone_col='ntem_zone_id',
        )
        
        # ## AGGREGATE TO NEEDED DATA ONLY ## #
        group_cols = ['ntem_zone_id']
        needed_cols = group_cols.copy() + [str(x) for x in self._output_years]

        nca = nca.reindex(
            columns=needed_cols).groupby(group_cols).sum().reset_index()
        ca = ca.reindex(
            columns=needed_cols).groupby(group_cols).sum().reset_index()

        # ## TRANSLATE TO MSOA ## #
        translator = zt.ZoneTranslator()

        nca = translator.run(
            nca,
            pd.read_csv(self.ntem_to_msoa_path),
            'ntem',
            'msoa',
            non_split_cols=group_cols,
        )
        ca = translator.run(
            ca,
            pd.read_csv(self.ntem_to_msoa_path),
            'ntem',
            'msoa',
            non_split_cols=group_cols,
        )

        # Tidy up and hold to join later
        group_cols = ['msoa_zone_id']
        needed_cols = group_cols.copy() + [str(x) for x in self._output_years]

        nca = nca.reindex(
            columns=needed_cols).groupby(group_cols).sum().reset_index()
        ca = ca.reindex(
            columns=needed_cols).groupby(group_cols).sum().reset_index()

        if pbar is not None:
            pbar.update(1)
        
        return nca, ca

    def get_available_dbs(self):
        available_dbs = []
        db_list = [x for x in os.listdir(self.data_source) if '.mdb' in x]
        for db_fname in db_list:
            for region in self.region_list:
                if region in db_fname:
                    available_dbs.append(db_fname)
                    break

        if available_dbs == list():
            raise IOError("Couldn't find any dbs to load from.")

        return available_dbs

    def get_pop_emp_growth_factors(self, verbose=True):
        # Init
        available_dbs = self.get_available_dbs()

        # Loop setup
        pop_ph = list()
        emp_ph = list()
        pbar = tqdm(
            total=len(available_dbs),
            desc="Extracting trip ends from DBs",
            disable=(not verbose),
        )

        for db_fname in available_dbs:
            pop, emp = self._get_pop_emp_factors_internal(
                db_fname,
                pbar,
            )
            pop_ph.append(pop)
            emp_ph.append(emp)

        # Stick all the partials together
        pop = pd.concat(pop_ph)
        emp = pd.concat(emp_ph)

        # Sort by zone_col
        pop = pop.sort_values(by=['msoa_zone_id']).reset_index(drop=True)
        emp = emp.sort_values(by=['msoa_zone_id']).reset_index(drop=True)

        # ## CALCULATE GROWTH FACTORS ## #
        # Need to know which is the base year
        base_year, future_years = du.split_base_future_years(self._output_years)
        base_year_col = str(base_year)
        future_year_cols = [str(x) for x in future_years]

        pop_df = pop.copy()
        emp_gf = emp.copy()

        for vector in [pop_df, emp_gf]:
            # Calculate growth factors
            for col in future_year_cols:
                vector[col] /= vector[base_year_col]
            vector[base_year_col] = 1

        return pop_df, emp_gf, pop, emp

    def get_growth_factors(self, verbose=True):
        # Init
        col_indices = '(1,2)'

        available_dbs = self.get_available_dbs()

        # Loop setup
        prod_ph = list()
        attr_ph = list()
        pbar = tqdm(
            total=len(available_dbs),
            desc="Extracting trip ends from DBs",
            disable=(not verbose),
        )

        for db_fname in available_dbs:
            prods, attrs = self._get_growth_factors_internal(
                db_fname,
                col_indices,
                pbar=pbar,
            )
            prod_ph.append(prods)
            attr_ph.append(attrs)

        # Stick all the partials together
        prods = pd.concat(prod_ph)
        attrs = pd.concat(attr_ph)

        # Sort by zone_col
        prods = prods.sort_values(by=['msoa_zone_id']).reset_index(drop=True)
        attrs = attrs.sort_values(by=['msoa_zone_id']).reset_index(drop=True)

        # ## CALCULATE GROWTH FACTORS ## #
        # Need to know which is the base year
        base_year, future_years = du.split_base_future_years(self._output_years)
        base_year_col = str(base_year)
        future_year_cols = [str(x) for x in future_years]

        prods_gf = prods.copy()
        attrs_gf = attrs.copy()

        for vector in [prods_gf, attrs_gf]:
            # Calculate growth factors
            for col in future_year_cols:
                vector[col] /= vector[base_year_col]
            vector[base_year_col] = 1

        return prods_gf, attrs_gf, prods, attrs

    def get_co_growth_factors(self,
                              verbose = True):
        """
        Get car ownership growth factors
        """
        
        # Init
        available_dbs = self.get_available_dbs()

        # Loop setup
        nca_ph = list()
        ca_ph = list()
        pbar = tqdm(
            total=len(available_dbs),
            desc="Extracting trip ends from DBs",
            disable=(not verbose),
        )

        for db_fname in available_dbs:
            nca, ca = self._get_co_growth_factors_internal(
                db_fname,
                pbar,
            )
            nca_ph.append(nca)
            ca_ph.append(ca)

        # Stick all the partials together
        nca = pd.concat(nca_ph)
        ca = pd.concat(ca_ph)

        # Sort by zone_col
        nca = nca.sort_values(by=['msoa_zone_id']).reset_index(drop=True)
        ca = ca.sort_values(by=['msoa_zone_id']).reset_index(drop=True)

        # ## CALCULATE GROWTH FACTORS ## #
        # Need to know which is the base year
        base_year, future_years = du.split_base_future_years(self._output_years)
        base_year_col = str(base_year)
        future_year_cols = [str(x) for x in future_years]

        nca_df = nca.copy()
        ca_df = ca.copy()

        for vector in [nca_df, ca_df]:
            # Calculate growth factors
            for col in future_year_cols:
                vector[col] /= vector[base_year_col]
            vector[base_year_col] = 1

        return nca_df, ca_df, nca, ca

    def get_trip_ends(self,
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
            col_indices = '(1,2)'
            col_a = 'Productions'
            col_b = 'Attractions'
        else:
            col_indices = '(3,4)'
            col_a = 'Origin'
            col_b = 'Destination'

        ntem_trans = pd.read_csv(self.ntem_trans_path)
        ntem_code_trans = pd.read_csv(self.ntem_code_zone_trans_path)
        ntem_lad_trans = pd.read_csv(self.ntem_lad_trans_path)

        available_dbs = self.get_available_dbs()

        # TODO: Check there's the full whack of regions here - say which aren't - error if any North missing

        # Iterate over all databases
        # TODO: multithread wrapper

        db_ph = []
        for db_fname in tqdm(available_dbs, desc="Extracting from DBs..."):
            
            trip_ends, years = self._get_pa_trip_ends(db_fname, col_indices)


            # TODO: Join LA (as NTEM) to new LA (lookup)
            # Nightmare because NTEM zone id != NTEM_zone_id - have to go round the houses
            trip_ends = trip_ends.merge(ntem_trans,
                                        how='inner',
                                        on='ZoneName')

            trip_ends = trip_ends.merge(ntem_code_trans,
                                        how='inner',
                                        on='ntem_id')

            trip_ends = trip_ends.merge(ntem_lad_trans,
                                        how='inner',
                                        left_on='Zone_ID',
                                        right_on='ntem_zone_id')

            # Reindex
            group_cols = ['lad_zone_id', 'Purpose', 'Mode', 'TimePeriod', 'TripType']
            target_cols = group_cols.copy()
            for year in years:
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
