# -*- coding: utf-8 -*-
"""

Download access drivers:
https://www.microsoft.com/en-us/download/confirmation.aspx?id=54920

This is in pretty good shape now, just need to use the functions
for all queries and tidy up the trip end queries.
"""
# Builtins
import os

from typing import List
from typing import Tuple
import warnings

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


class TemproParser:
    """
    """
    _access_driver = '{Microsoft Access Driver (*.mdb, *.accdb)}'
    _data_source = r'I:\Data\NTEM'
    _ntem_version = 8.0
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
    _scenario_list = [
        'High',
        'Low',
        'Regional',
        'Core',
        'Behavioural',
        'Technology'
    ]
    _output_years = [2011, 2018, 2033, 2035, 2040, 2050]

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
        'ca': [2, 3, 4]}

    trip_end_types = {
        'productions': [1],
        'attractions': [2],
        'origins': [3],
        'destinations': [4]
    }

    def __init__(self,
                 access_driver: str = None,
                 data_source: str = None,
                 ntem_version: float = None,
                 scenario: str = None,
                 region_list: List[str] = None,
                 output_years: List[int] = None,
                 out_folder: str = None,
                 ):
        """
        """
        print('Building a TEMPRO extractor...')

        # Set to default values if not passed in
        access_driver = self._access_driver if access_driver is None else access_driver
        data_source = self._data_source if data_source is None else data_source
        ntem_version = self._ntem_version if ntem_version is None else ntem_version
        out_folder = self._out_folder if out_folder is None else out_folder
        region_list = self._region_list if region_list is None else region_list
        output_years = self._output_years if output_years is None else output_years

        if not os.path.exists(data_source):
            raise ValueError('Tempro not installed at' + data_source)

        # Assign variables
        self.access_driver = access_driver
        self.data_source = os.path.join(data_source, str(ntem_version))
        self.ntem_version = ntem_version
        self.scenario = scenario
        self.region_list = region_list
        self.output_years = output_years
        self.output_years_str = [str(x) for x in output_years]
        self.out_folder = out_folder

        # Set up paths
        # TODO(CS/BT): Update these paths to search a bit
        home_path = os.path.normpath(os.getcwd())
        config_path = os.path.join(home_path, 'config', 'tempro')

        self.ntem_trans_path = os.path.join(config_path, 'tblLookupGeo76.csv')
        self.ntem_code_zone_trans_path = os.path.join(config_path, 'ntem_code_to_zone.csv')
        self.ntem_lad_trans_path = os.path.join(config_path, 'ntem_lad_pop_weighted_lookup.csv')
        self.ntem_to_msoa_path = r"I:\NorMITs Demand\import\ntem_extractor\zone translation\ntem_msoa_pop_weighted_lookup.csv"

    def get_available_dbs(self):
        """
        Returns
        ----------
        available_dbs: list
            List of dbs available for import
        """

        available_dbs = []
        db_list = [x for x in os.listdir(self.data_source) if '.mdb' in x]
        for db_fname in db_list:
            for region in self.region_list:
                if region in db_fname:
                    available_dbs.append(db_fname)
                    break

        if self.ntem_version > 7.2:
            if self.scenario is None:
                raise IOError('NTEM later than 7.2 requires scenario, none was passed')
            else:
                available_dbs = [x for x in available_dbs if self.scenario in x]

        if available_dbs == list():
            raise IOError("Couldn't find any dbs to load from.")

        return available_dbs

    def get_planning_data(self,
                          compile_planning_data=True,
                          verbose=True):
        """
        Function to get planning data from TEMPRO
        """
        # TODO: Rewrite everything in this style
        
        # Init
        available_dbs = self.get_available_dbs()
        
        # Loop setup
        plan_ph = list()
        
        pbar = tqdm(
            total=len(available_dbs),
            desc="Extracting trip ends from DBs",
            disable=(not verbose),
        )

        for db_fname in available_dbs:
            if verbose:
                print(db_fname)
            
            plan_dat = self._get_segmented_planning_data(db_fname,
                                                         pbar)

            plan_ph.append(plan_dat)

        # Compile segments dict label wise
        plan_dat = du.concat_df_dict(plan_ph,
                                     non_sum_cols=['msoa_zone_id'],
                                     sort=True)

        if compile_planning_data:
            plan_dat = pd.concat(
                plan_dat, names=('population', 'indo')).reset_index()
            ri_list = ['msoa_zone_id', 'population'] + list(map(
                str, self.output_years))
            plan_dat = plan_dat.reindex(ri_list, axis=1)

        return plan_dat

    def get_trip_ends(self,
                      trip_type=None,
                      all_commute_hb: bool = True,
                      aggregate_car: bool = True,
                      average_weekday: bool = False,
                      verbose=True):

        """
        trip_type = 'pa' or 'od':
            If none returns both

        compile_trip_end_data:
            True or false, compile dict to dataframe or no

        all_commute_hb: bool
            If true, compiles NHB commute in TEMPRO into HB commute

        aggregate_car: bool
            Very important for car demand. If aggregate = False will take Tempro
            mode 3 only - ie. growth in car drivers.
            If False, will add Modes 3 & 4, so car driver and passenger.

        average_weekday: bool
            If true, drops weekends and divides weekday trips by 5

        """
        # Init
        available_dbs = self.get_available_dbs()

        # Loop setup
        te_ph = list()

        pbar = tqdm(
            total=len(available_dbs),
            desc="Extracting trip ends from DBs",
            disable=(not verbose),
        )

        for db_fname in available_dbs:
            if verbose:
                print(db_fname)

            te_dat = self._get_trip_ends(db_fname,
                                         pbar,
                                         trip_type)

            te_ph.append(te_dat)

        # Compile segments dict label wise
        te_dat = du.concat_df_dict(te_ph,
                                   non_sum_cols=['msoa_zone_id'],
                                   sort=True)

        # Compile to 1 df by default
        te_dat = pd.concat(
            te_dat, names=('trip_end_type', 'indo')).reset_index()
        ri_list = ['msoa_zone_id', 'trip_end_type', 'Purpose',
                   'Mode', 'TimePeriod'] + list(map(str, self.output_years))
        te_dat = te_dat.reindex(ri_list, axis=1)

        if all_commute_hb:
            te_dat['Purpose'] = te_dat['Purpose'].replace([11], 12)

        if aggregate_car:
            te_dat['Mode'] = te_dat['Mode'].replace(4, 3)

        if average_weekday:
            te_dat = te_dat[te_dat['TimePeriod'].isin([1, 2, 3, 4])]
            for year in self.output_years:
                te_dat[str(year)] = te_dat[str(year)]/5

        # Final reindex and out
        final_groups = ['msoa_zone_id', 'trip_end_type', 'Purpose',
                        'Mode', 'TimePeriod']
        final_cols = final_groups.copy() + list(map(str, self.output_years))
        te_dat = te_dat.reindex(final_cols, axis=1)
        te_dat = te_dat.groupby(final_groups).sum().reset_index()
        te_dat = te_dat.sort_values(final_groups).reset_index(drop=True)

        return te_dat

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
        base_year, future_years = du.split_base_future_years(self.output_years)
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

    def get_household_co_data(self,
                              compile_co_data=True,
                              verbose=True):
        """
        """
        # Init
        available_dbs = self.get_available_dbs()

        # Loop setup
        co_ph = list()

        pbar = tqdm(
            total=len(available_dbs),
            desc="Extracting trip ends from DBs",
            disable=(not verbose),
        )

        for db_fname in available_dbs:
            if verbose:
                print(db_fname)

            co_dat = self._get_segmented_co_data(db_fname,
                                                 pbar)

            co_ph.append(co_dat)

        # Compile segments dict label wise
        co_dat = du.concat_df_dict(co_ph,
                                   non_sum_cols=['msoa_zone_id'],
                                   sort=True)

        if compile_co_data:
            co_dat = pd.concat(
                co_dat, names=('car_ownership', 'indo')).reset_index()
            ri_list = ['msoa_zone_id', 'car_ownership'] + list(map(
                str, self.output_years))
            co_dat = co_dat.reindex(ri_list, axis=1)

        return co_dat

    def get_co_data(self,
                    verbose: bool = True):
        """
        Get car ownership future year shares
        Calculates share of ca/nca in a given fy

        returns:
        fy_ca_share, fy_ca_growth, nca, ca

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

        # Build factor and share outputs
        # Get headings for base and future year cols
        base_year, future_years = du.split_base_future_years(self.output_years)
        base_year_col = str(base_year)
        future_year_cols = [str(x) for x in future_years]

        # Do future year growth calc
        nca_growth = nca.copy()
        ca_growth = ca.copy()
        # Work out growth factors
        for col in future_year_cols:
            nca_growth[col] /= nca_growth[base_year_col]
            ca_growth[col] /= ca_growth[base_year_col]

        nca_growth['ca'] = '1'
        ca_growth['ca'] = '2'

        fy_ca_growth = pd.concat([nca_growth, ca_growth])
        fy_ca_growth[base_year_col] = 1
        fy_ca_growth = fy_ca_growth.reindex(
            ['msoa_zone_id', 'ca', base_year_col] + future_year_cols, axis=1)

        # Do future year share calc
        nca_share = nca.copy()
        ca_share = ca.copy()
        
        total = pd.concat([nca_share, ca_share])
        total = total.groupby('msoa_zone_id').sum().reset_index()
        
        nca_share[[base_year_col]+future_year_cols] /= total[[base_year_col]+future_year_cols]
        print(nca_share)
        ca_share[[base_year_col]+future_year_cols] /= total[[base_year_col]+future_year_cols]
        print(ca_share)
        
        nca_share['ca'] = '1'
        ca_share['ca'] = '2'

        # Compile and reindex
        fy_ca_share = pd.concat(
            [nca_share, ca_share])
        fy_ca_share = fy_ca_share.reindex(
            ['msoa_zone_id', 'ca', base_year_col] + future_year_cols, axis=1)
        
        return fy_ca_share, fy_ca_growth, nca, ca

    def _get_trip_ends(self,
                       db_fname,
                       pbar=None,
                       trip_type=None) -> pd.DataFrame:
        """
        Get segmented planning data from Tempro DB.

        Parameters
        ----------
        db_fname : str
            Name of DB.
        pbar : std.tqdm
            Progress bar

        Returns
        -------
        trip_ends :
            Dictionary of trip ends
        """
        # Init
        if trip_type == 'pa':
            te = '(1,2)'
            segmented_trip_ends = {
                'productions': self.trip_end_types['productions'],
                'attractions': self.trip_end_types['attractions']
            }
        elif trip_type == 'od':
            te = '(3,4)'
            segmented_trip_ends = {
                'origins': self.trip_end_types['origins'],
                'destinations': self.trip_end_types['destinations']
            }
        else:
            te = '(1,2,3,4)'
            segmented_trip_ends = {
                'productions': self.trip_end_types['productions'],
                'attractions': self.trip_end_types['attractions'],
                'origins': self.trip_end_types['origins'],
                'destinations': self.trip_end_types['destinations']
            }

        query = 'select * from TripEndDataByDirection where TripType in ' + te
        target_table_attr = 'TripType'

        query_dat, zones = self._hit_ntem_db(
            db_fname,
            query)

        # Get years
        query_dat = self._select_years_and_interpolate(query_dat)

        # Split off data into nice usable pots

        for segment, indices in segmented_trip_ends.items():

            # Filter to target
            mask = query_dat[target_table_attr].isin(indices)
            query_out = query_dat[mask]

            # ## GET INTO NORMITS NTEM ## #
            # Attach the zones
            query_out = pd.merge(
                query_out,
                zones,
                how='left',
                on='ZoneID'
            )

            # Drop this GB zone thats repeated across DBs
            odd_zone = query_out['ZoneID'] == 9999
            query_out = query_out[~odd_zone]

            # Convert
            query_out = self._tempro_ntem_to_normits_ntem(
                query_out,
                return_zone_col='ntem_zone_id'
            )

            # ## FILTER DOWN TO NEEDED DATA ONLY ## #
            ri_list = ['ntem_zone_id', target_table_attr,
                       'Purpose', 'Mode', 'TimePeriod']
            for oy in self.output_years:
                ri_list.append(str(oy))
            query_out = query_out.reindex(ri_list, axis=1)

            # Translate to MSOA
            translator = zt.ZoneTranslator()
            query_out = translator.run(
                query_out,
                pd.read_csv(self.ntem_to_msoa_path),
                'ntem',
                'msoa',
                non_split_cols=['ntem_zone_id', 'Purpose', 'Mode', 'TimePeriod'])

            # Aggregate to required data only
            group_cols = ['msoa_zone_id', 'Purpose', 'Mode', 'TimePeriod']
            needed_cols = group_cols.copy(
            ) + [str(x) for x in self.output_years]
            query_out = query_out.reindex(
                columns=needed_cols).groupby(group_cols).sum().reset_index()

            segmented_trip_ends.update({segment: query_out})

        if pbar is not None:
            pbar.update(1)

        return segmented_trip_ends

    def _get_co_data(self,
                     db_fname
                     ):
        # TODO: Deprecate and remove, now in get_segmented_co_data
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

    def _get_pa_trip_ends(self,
                          db_fname: str,
                          col_indices: str
                          ):

        ## TODO: DELETE

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
        val_cols = self.output_years_str if val_cols is None else val_cols

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
        needed_cols = group_cols.copy() + [str(x) for x in self.output_years]
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
        needed_cols = group_cols.copy() + [str(x) for x in self.output_years]
        trip_ends = trip_ends.reindex(columns=needed_cols)
        trip_ends = trip_ends.groupby(group_cols).sum().reset_index()

        # Divide by 5 to get the average weekday
        for year in self.output_years_str:
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

    def _get_segmented_planning_data(self,
                                     db_fname,
                                     pbar=None,
                                     ) -> pd.DataFrame:
        """
        Get segmented planning data from Tempro DB.
        
        Parameters
        ----------
        db_fname : str
            Name of DB.
        pbar : std.tqdm
            Progress bar

        Returns
        -------
        segmented_population :
            Dictionary of planning data

        """
        # Init
        query = 'select * from Planning'
        target_table_attr = 'PlanningDataType'
        
        query_dat, zones = self._hit_ntem_db(
            db_fname,
            query)
        
        # Get years
        query_dat = self._select_years_and_interpolate(query_dat)

        # Split off data into nice usable pots
        segmented_population = {
            'under16': self.planning_data_types['under16'],
            '16-74': self.planning_data_types['16-74'],
            '75+': self.planning_data_types['75+'],
            'HHs': self.planning_data_types['HHs'],
            'jobs': self.planning_data_types['jobs'],
            'workers': self.planning_data_types['workers']
            }
        for segment, indices in segmented_population.items():
            # Filter to target
            mask = query_dat[target_table_attr].isin(indices)
            query_out = query_dat[mask]

            # ## GET INTO NORMITS NTEM ## #
            # Attach the zones
            query_out = pd.merge(
                query_out,
                zones,
                how='left',
                on='ZoneID'
            )

            # Drop this GB zone thats repeated across DBs
            odd_zone = query_out['ZoneID'] == 9999
            gb_zone = query_out[odd_zone]
            query_out = query_out[~odd_zone]

            # Convert
            query_out = self._tempro_ntem_to_normits_ntem(
                query_out,
                return_zone_col='ntem_zone_id'
            )

            # ## FILTER DOWN TO NEEDED DATA ONLY ## #
            ri_list = ['ntem_zone_id', target_table_attr]
            for oy in self.output_years:
                ri_list.append(str(oy))
            query_out = query_out.reindex(ri_list, axis=1)

            # Translate to MSOA
            translator = zt.ZoneTranslator()
            query_out = translator.run(
                query_out,
                pd.read_csv(self.ntem_to_msoa_path),
                'ntem',
                'msoa',
                non_split_cols=['ntem_zone_id'])
            
            # Aggregate to required data only
            group_cols = ['msoa_zone_id']
            needed_cols = group_cols.copy(
                ) + [str(x) for x in self.output_years]
            query_out = query_out.reindex(
                columns=needed_cols).groupby(group_cols).sum().reset_index()

            segmented_population.update({segment: query_out})

        if pbar is not None:
            pbar.update(1)
        
        return segmented_population

    def _get_segmented_co_data(self,
                               db_fname,
                               pbar=None,
                               ) -> pd.DataFrame:

        """
        Parameters
        ----------
        db_fname : str
            Name of DB.
        pbar : std.tqdm
            Progress bar

        Returns
        -------
        segmented_co :
            Dictionary of co data (households)
        """

        query = 'select * from CarOwnership'
        target_table_attr = 'CarOwnershipType'

        query_dat, zones = self._hit_ntem_db(
            db_fname,
            query)

        # Get years
        query_dat = self._select_years_and_interpolate(query_dat)

        # Split off data into nice usable pots
        segmented_co = {
            'no_car': self.co_data_types['no_car'],
            '1_car': self.co_data_types['1_car'],
            '2_cars': self.co_data_types['2_cars'],
            '3+_cars': self.co_data_types['3+_cars']
        }
        for segment, indices in segmented_co.items():
            # Filter to target
            mask = query_dat[target_table_attr].isin(indices)
            query_out = query_dat[mask]

            # Attach the zones
            query_out = pd.merge(
                query_out,
                zones,
                how='left',
                on='ZoneID'
            )

            # Drop this GB zone thats repeated across DBs
            odd_zone = query_out['ZoneID'] == 9999
            gb_zone = query_out[odd_zone]
            query_out = query_out[~odd_zone]

            # Convert
            query_out = self._tempro_ntem_to_normits_ntem(
                query_out,
                return_zone_col='ntem_zone_id'
            )

            # ## FILTER DOWN TO NEEDED DATA ONLY ## #
            ri_list = ['ntem_zone_id', target_table_attr]
            for oy in self.output_years:
                ri_list.append(str(oy))
            query_out = query_out.reindex(ri_list, axis=1)

            # Translate to MSOA
            translator = zt.ZoneTranslator()
            query_out = translator.run(
                query_out,
                pd.read_csv(self.ntem_to_msoa_path),
                'ntem',
                'msoa',
                non_split_cols=['ntem_zone_id'])

            # Aggregate to required data only
            group_cols = ['msoa_zone_id']
            needed_cols = group_cols.copy(
            ) + [str(x) for x in self.output_years]
            query_out = query_out.reindex(
                columns=needed_cols).groupby(group_cols).sum().reset_index()

            segmented_co.update({segment: query_out})

        if pbar is not None:
            pbar.update(1)

        return segmented_co

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
        needed_cols = group_cols.copy() + [str(x) for x in self.output_years]

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
        needed_cols = group_cols.copy() + [str(x) for x in self.output_years]

        nca = nca.reindex(
            columns=needed_cols).groupby(group_cols).sum().reset_index()
        ca = ca.reindex(
            columns=needed_cols).groupby(group_cols).sum().reset_index()

        if pbar is not None:
            pbar.update(1)
        
        return nca, ca
    
    def _hit_ntem_db(self,
                     db_fname: str,
                     query: str,
                     zone_query = 'select * from Zones'):
        """
        db_fname:
            Name of database
        query:
            Query to target table as Access SQL string
        zone_query:
            Query to zones table, mess about at your peril
        
        Returns
        ####
        query_dat: pd.Dataframe
            Query as df
        zones:
            Zones returned from same DB
        """
            
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
            cursor.execute(query)

            co_ph = list()
            for row in cursor.fetchall():
                co_ph.append([x for x in row])

            query_dat = pd.DataFrame(
                data=co_ph,
                columns=[column[0] for column in cursor.description]
            )

            print(list(query_dat))

            # Grab and unpack local db zone translations
            cursor.execute(zone_query)

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
        
        return query_dat, zones

    def _select_years_and_interpolate(self, query_dat: pd.DataFrame) -> pd.DataFrame:
        """
        Pick years from the run defined years
        Interpolate from them if you have to.

        Parameters
        ----------
        query_dat : pd.DataFrame
            Raw query from DB in df

        Returns
        -------
        query_dat: pd.Dataframe
            Same but with years added and/or taken away

        """
        av_years = sorted(int(x) for x in list(query_dat) if x.isdigit())
        if av_years[0] < 0:
            raise ValueError("negative year found in query data")

        for target_year in self.output_years:
            if target_year in av_years:
                continue # Year already exists and doesn't need creating

            if target_year > av_years[-1]:
                # Extrapolating past largest year, using growth from last 2 years
                warnings.warn(
                    f"Extrapolating TEMPro data past largest year ({av_years[-1]})",
                    RuntimeWarning
                )
                lower, upper = av_years[-2:]
                start_year = upper
            elif target_year < av_years[0]:
                raise ValueError("target year is less than the minimum TEMPro year available")
            else:
                # Interpolating between one year either side
                lower = max(i for i in av_years if i < target_year)
                upper = min(i for i in av_years if i > target_year)
                start_year = lower

            annual_growth = (query_dat[str(upper)] - query_dat[str(lower)]) / (upper - lower)
            year_diff = target_year - start_year
            query_dat[str(target_year)] = query_dat[str(start_year)] + (
                year_diff * annual_growth
            )

        return query_dat
