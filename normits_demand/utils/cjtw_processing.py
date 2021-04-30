# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 11:33:58 2019

Upgrade census journey to work to 2018
Resolve non standard MSOA zones
Output at model level

@author: cruella
"""

import os

import pandas as pd
import geopandas as gpd

from normits_demand import tms_constants
from normits_demand.utils import shapefiles


class CjtwTranslator:

    def __init__(self,
                 model_name,
                 model_folder=None,
                 cjtw_year=tms_constants.CJTW_YEAR,
                 cjtw_path=tms_constants.CJTW_PATH,
                 cjtw_dat_file=tms_constants.CJTW_DAT,
                 cjtw_header_file=tms_constants.CJTW_HEADER,
                 hybrid_msoa_ref=tms_constants.Z_HYBRID_MSOA,
                 tempro_path=tms_constants.MSOA_TEMPRO
                 ):
        """

        Params:
        model_folder:
            Path to output model folder - Model Zone Lookups or similar
        """
        self.model_name = model_name
        self.model_folder = model_folder

        self.cjtw_year = cjtw_year

        self.cjtw_path = cjtw_path
        self.cjtw_dat_file = cjtw_dat_file
        self.cjtw_header_file = cjtw_header_file
        self.hybrid_msoa_ref = hybrid_msoa_ref

        self.tempro_path = tempro_path

    # What will be function parameter defs
    def cjtw_to_model_zone(self,
                           target_year=None):
        """
        Build a 2011 cjtw set in MSOA, adjust to a future year and translate
        to model zoning.

        target_year:
            Target year if you want to rebase
        """

        # Build base year MSOA
        msoa_cjtw, audits = self._cjtw_to_msoa(write=False)

        # Adjust to NorMITs format
        msoa_cjtw = self._adjust_to_normits_format(msoa_cjtw)

        # Do adjustment to target year
        if target_year is not None:
            msoa_cjtw = self._adjust_to_target_year(msoa_cjtw,
                                                    target_year)

        # Translate to Model Zoning
        if self.model_name != 'msoa':
            zone_cjtw = self._translate_to_model_zoning(msoa_cjtw)

        # Pack up into nice dictionaries
        out_cjtw = dict()
        for key, dat in zone_cjtw.items():
            file_name = '%s_cjtw_yr%d_p1_m%d' % (self.model_name.lower(),
                                                 target_year,
                                                 key)

            out_cjtw.update({file_name: dat})

        return out_cjtw

    def _translate_to_model_zoning(self,
                                   msoa_cjtw,
                                   verbose=True):
        """
        msoa_cjtw:
            Interim census journey to work dictionary by NorMITs Mode
        """

        # TODO: Build better/consistent translation method for splits
        # TODO: Update to use new indices

        zonal_cjtw = msoa_cjtw.copy()

        # Init
        # Get msoa pop and emp lookups
        index_folder = os.listdir(self.model_folder)
        model_lookups = [x for x in index_folder if
                         'msoa' in x and self.model_name.lower() in x]
        pop_lookup = [x for x in model_lookups if 'pop' in x][0]
        emp_lookup = [x for x in model_lookups if 'emp' in x][0]

        print('Pop lookup from %s' % pop_lookup)
        print('Emp lookup from %s' % emp_lookup)

        # Import and simplify lookups
        pop_lookup = pd.read_csv(
            os.path.join(self.model_folder,
                         pop_lookup)
        )
        pop_lookup = pop_lookup.reindex(
        ['msoa_zone_id',
         self.model_name.lower() + '_zone_id',
         'overlap_msoa_split_factor'],
            axis=1)

        emp_lookup = pd.read_csv(
            os.path.join(self.model_folder,
                         emp_lookup)
        )
        emp_lookup = emp_lookup.reindex(
            ['msoa_zone_id',
             self.model_name.lower() + '_zone_id',
             'overlap_msoa_split_factor'],
            axis=1)

        # Into the iterator
        for key, dat in zonal_cjtw.items():
            dat_ph = dat.copy()
            demand_before = dat_ph['demand'].sum()

            # Do production end
            dat_ph = dat_ph.rename(columns={'p_zone': 'msoa_zone_id'})
            dat_ph = dat_ph.merge(
                pop_lookup,
                how='left',
                on='msoa_zone_id'
            )
            dat_ph['demand'] *= dat_ph['overlap_msoa_split_factor']
            dat_ph = dat_ph.rename(
                columns={self.model_name.lower() + '_zone_id': 'p_zone'})
            dat_ph = dat_ph.reindex(
                ['p_zone', 'a_zone', 'demand'], axis=1).groupby(
                ['p_zone', 'a_zone']
            ).sum().reset_index()

            # Do attraction end
            dat_ph = dat_ph.rename(columns={'a_zone': 'msoa_zone_id'})
            dat_ph = dat_ph.merge(
                emp_lookup,
                how='left',
                on='msoa_zone_id'
            )
            dat_ph['demand'] *= dat_ph['overlap_msoa_split_factor']
            dat_ph = dat_ph.rename(
                columns={self.model_name.lower() + '_zone_id': 'a_zone'})
            dat_ph = dat_ph.reindex(
                ['p_zone', 'a_zone', 'demand'], axis=1).groupby(
                ['p_zone', 'a_zone']
            ).sum().reset_index()
            demand_after = dat_ph['demand'].sum()

            if verbose:
                print('Translating mode %d' % key)
                print('%d before' % demand_before)
                print('%d after' % demand_after)

            zonal_cjtw.update({key: dat_ph})

        return zonal_cjtw

    @staticmethod
    def _adjust_to_normits_format(msoa_cjtw):
        """
        msoa_cjtw:
            Pandas dataframe with original format cjtw
        """

        # Init
        normits_cjtw = dict()

        # Define p/a cols
        pa_cols = ['1_msoaAreaofresidence',
                   '2_msoaAreaofworkplace']

        # Define key to mode bin
        mode_bins = {1: ['13_Onfoot'],
                     2: ['12_Bicycle'],
                     3: ['8_Taxi',
                         '9_Motorcyclescooterormoped',
                         '10_Drivingacarorvan',
                         '11_Passengerinacarorvan'],
                     4: ['5_Undergroundmetrolightrailtram'],
                     5: ['7_Busminibusorcoach'],
                     6: ['6_Train']}

        for key, cols in mode_bins.items():

            # Subset by mode, col wise
            mode_sub = msoa_cjtw.reindex(
                pa_cols + mode_bins[key], axis=1
            )
            mode_sub['demand'] = mode_sub[mode_bins[key]].sum(axis=1)
            mode_sub = mode_sub.drop(mode_bins[key], axis=1)
            mode_sub = mode_sub.rename(
                columns={'1_msoaAreaofresidence': 'p_zone',
                         '2_msoaAreaofworkplace': 'a_zone'})

            # Drop 0 cells
            mode_sub = mode_sub[mode_sub['demand'] > 0]

            normits_cjtw.update({key: mode_sub})

        return normits_cjtw

    def _adjust_to_target_year(self,
                               msoa_cjtw,
                               target_year,
                               verbose=True):
        """
        msoa_cjtw: pd.DataFrame:
            Dataframe of CJtW already translated to MSOA
        """
        # Init
        fy_cjtw = msoa_cjtw.copy()

        # Get tempro commute data
        tempro = pd.read_csv(self.tempro_path)
        # Filter to commute only
        tempro = tempro[tempro['Purpose'] == 1]
        # Drop Time Period
        tempro = tempro.drop('TimePeriod', axis=1)
        # Sum remainder - makes assumptions about col names
        group_cols = ['msoa_zone_id', 'trip_end_type', 'Purpose',
                      'Mode']
        tempro = tempro.groupby(group_cols).sum().reset_index()

        # Seperate p/a
        productions = tempro[tempro['trip_end_type'] == 'productions']
        attractions = tempro[tempro['trip_end_type'] == 'attractions']
        del tempro
        productions = productions.reset_index(drop=True)
        attractions = attractions.reset_index(drop=True)

        # TODO: Should balance P/A & furness
        # Attractions are already here

        # Get growth factor from NTEM
        productions['gf'] = productions[str(target_year)] / productions[str(self.cjtw_year)]

        # Grow prod wise
        for key, dat in fy_cjtw.items():
            tempro_sub = productions[productions['Mode'] == key]
            tempro_sub = tempro_sub.reindex(['msoa_zone_id', 'gf'], axis=1)
            tempro_sub = tempro_sub.reset_index(drop=True)

            future_cjtw = dat.copy()
            demand_before = future_cjtw['demand'].sum()

            future_cjtw = future_cjtw.merge(
                tempro_sub,
                how='left',
                left_on='p_zone',
                right_on='msoa_zone_id'
            ).fillna(1)
            future_cjtw['demand'] *= future_cjtw['gf']
            future_cjtw = future_cjtw.drop(
                ['msoa_zone_id', 'gf'], axis=1)
            demand_after = future_cjtw['demand'].sum()

            if verbose:
                print('Adjusting mode %d from %d to %d' % (key,
                                                           self.cjtw_year,
                                                           target_year))
                print('%d before' % demand_before)
                print('%d after' % demand_after)

            fy_cjtw.update({key: future_cjtw})

        return fy_cjtw

    def _cjtw_to_msoa(self,
                      write=True):

        """
        Translate demand from census journey to work into a target zoning system
        """

        msoa_hybrid_ref = self.hybrid_msoa_ref
        msoa_hybrid_shp = shapefiles.count_list_shp(
            shp=msoa_hybrid_ref,
            id_col='msoa11cd')
        unq_hybrid = msoa_hybrid_shp[1]
        msoa_hybrid = gpd.read_file(msoa_hybrid_ref).reindex(
            ['objectid', 'msoa11cd'], axis=1)

        cjtw_header = []
        with open((self.cjtw_path + self.cjtw_header_file), "r") as my_file:
            for columns in (raw.strip() for raw in my_file):
                cjtw_header.append(columns)

        cjtw_header = pd.Series(cjtw_header)
        cjtw_header = cjtw_header[7:21].reset_index(drop=True)
        cjtw_header = cjtw_header.str.replace(',', '').str.replace(' ', '').str.replace(':', '_')

        print('Importing 2011 census journey to work')
        cjtw = pd.read_csv((self.cjtw_path + self.cjtw_dat_file), names=cjtw_header)

        # Get total trip counts in hybrid area for comparison
        inclusive_zones = cjtw[cjtw['1_Areaofusualresidence'].isin(unq_hybrid)]
        inclusive_zones = inclusive_zones[inclusive_zones['2_Areaofworkplace'].isin(unq_hybrid)]
        total_trips1 = inclusive_zones['3_Allcategories_Methodoftraveltowork'].sum()
        print(total_trips1)
        del inclusive_zones

        # Import hybrid to MSOA translation
        hybrid_msoa_trans = pd.read_csv(tms_constants.MSOA_TO_HYBRID_POP)

        hmt_cols = ['msoa_hybrid_zone_id',
                    'msoa_zone_id',
                    'overlap_msoa_hybrid_split_factor',
                    'overlap_msoa_split_factor']

        # Append msoa11cd
        hybrid_msoa_trans = hybrid_msoa_trans.reindex(hmt_cols, axis=1)
        hybrid_msoa_trans = hybrid_msoa_trans.merge(msoa_hybrid,
                                                    how='inner',
                                                    left_on='msoa_hybrid_zone_id',
                                                    right_on='objectid').drop(
                                                            'objectid', axis=1)

        gb_cjtw = cjtw[cjtw['1_Areaofusualresidence'].isin(unq_hybrid)]
        gb_cjtw = gb_cjtw[gb_cjtw['2_Areaofworkplace'].isin(unq_hybrid)]
        total_trips2 = gb_cjtw['3_Allcategories_Methodoftraveltowork'].sum()

        # Join translation - msoa11cd is retained somehow - need to keep it.
        gb_cjtw = gb_cjtw.merge(hybrid_msoa_trans,
                                how='inner',
                                left_on='1_Areaofusualresidence',
                                right_on='msoa11cd').drop_duplicates(
                                        ).drop('msoa_hybrid_zone_id', axis=1)

        gb_cjtw = gb_cjtw.rename(
            columns={'msoa_zone_id': '1_msoaAreaofresidence'})
        total_trips3 = gb_cjtw['3_Allcategories_Methodoftraveltowork'].sum()
        del cjtw

        # TODO: define totals by column for audits
        # Reduce segments to factors
        factor_cols = ['4_Workmainlyatorfromhome',
                       '5_Undergroundmetrolightrailtram',
                       '6_Train',
                       '7_Busminibusorcoach',
                       '8_Taxi',
                       '9_Motorcyclescooterormoped',
                       '10_Drivingacarorvan',
                       '11_Passengerinacarorvan',
                       '12_Bicycle',
                       '13_Onfoot',
                       '14_Othermethodoftraveltowork']

        # Factor down columns for split adjustment
        for col in factor_cols:
            gb_cjtw = self.factor_col(gb_cjtw, method='Down', working_col=col)

        # Apply split adjustment
        # TODO: May be a bit more complicated than this - need to check
        gb_cjtw['3_Allcategories_Methodoftraveltowork'] = (
                gb_cjtw['3_Allcategories_Methodoftraveltowork']*
                gb_cjtw[hmt_cols[2]])

        total_trips4 = gb_cjtw['3_Allcategories_Methodoftraveltowork'].sum()

        # Factor up columns to resolve splits
        for col in factor_cols:
            gb_cjtw = self.factor_col(gb_cjtw, method='Up', working_col=col)

        zone_audit = shapefiles.hybrid_zone_counts(
            gb_cjtw['1_Areaofusualresidence'],
            unq_hybrid)
        audit_status = self.split_audit(gb_cjtw,
                                        unq_hybrid)
        print(audit_status)

        # Build reindex columns
        zone_cjtw_cols = ['1_msoaAreaofresidence',
                          '1_Areaofusualresidence',
                          '2_Areaofworkplace',
                          '3_Allcategories_Methodoftraveltowork']
        for col in factor_cols:
            zone_cjtw_cols.append(col)

        zone_cjtw = gb_cjtw.reindex(zone_cjtw_cols,
                                    axis=1)
        del gb_cjtw

        total_trips5 = zone_cjtw['3_Allcategories_Methodoftraveltowork'].sum()

        # Do the same with the commute attraction weightings for the attraction end
        hybrid_msoa_emp_trans = pd.read_csv(tms_constants.MSOA_TO_HYBRID_EMP)

        hybrid_msoa_emp_trans = hybrid_msoa_emp_trans.reindex(hmt_cols,
                                                              axis=1)
        hybrid_msoa_emp_trans = hybrid_msoa_emp_trans.merge(
                msoa_hybrid,
                how='inner',
                left_on='msoa_hybrid_zone_id',
                right_on='objectid').drop('objectid', axis=1)

        zone_cjtw = zone_cjtw.merge(hybrid_msoa_emp_trans,
                                    how='inner',
                                    left_on='2_Areaofworkplace',
                                    right_on='msoa11cd')

        # Audit it
        zone_audit = shapefiles.hybrid_zone_counts(
            zone_cjtw['2_Areaofworkplace'],
            unq_hybrid)
        zone_audit = shapefiles.hybrid_zone_counts(
            zone_cjtw['2_Areaofworkplace'],
            unq_hybrid)

        # Not dropping duplicates here, apparently they're required.
        zone_cjtw = zone_cjtw.rename(columns={
            'msoa_zone_id' : '2_msoaAreaofworkplace'})

        zone_audit = shapefiles.hybrid_zone_counts(zone_cjtw['2_Areaofworkplace'], unq_hybrid)
        print(zone_audit)

        total_trips6 = zone_cjtw['3_Allcategories_Methodoftraveltowork'].sum()

        # Factor down columns for split adjustment
        for col in factor_cols:
            zone_cjtw = self.factor_col(zone_cjtw, method='Down', working_col=col)

        # Apply split adjustment
        zone_cjtw['3_Allcategories_Methodoftraveltowork'] = (
                zone_cjtw['3_Allcategories_Methodoftraveltowork']*
                zone_cjtw['overlap_msoa_hybrid_split_factor'])

        total_trips7 = zone_cjtw['3_Allcategories_Methodoftraveltowork'].sum()

        # Factor up columns to resolve splits
        for col in factor_cols:
            zone_cjtw = self.factor_col(zone_cjtw, method='Up', working_col=col)

        zone_cjtw_cols = ['1_msoaAreaofresidence',
                          '2_msoaAreaofworkplace',
                          '3_Allcategories_Methodoftraveltowork']
        for col in factor_cols:
            zone_cjtw_cols.append(col)

        zone_cjtw = zone_cjtw.reindex(zone_cjtw_cols, axis=1)
        total_trips8 = zone_cjtw['3_Allcategories_Methodoftraveltowork'].sum()

        zone_cjtw = zone_cjtw.groupby(
                ['1_msoaAreaofresidence',
                 '2_msoaAreaofworkplace']).sum().reset_index()

        # TODO: write clean_cjtw
        # zone_cjtw = self.clean_cjtw(zone_cjtw)

        total_trips9 = zone_cjtw['3_Allcategories_Methodoftraveltowork'].sum()

        if write:
            zone_cjtw.to_csv('cjtw_msoa.csv',
                             index=False)
        # ?? Lookups

        # Get audit numbers
        audit_numbers = pd.DataFrame([total_trips1, total_trips2,
                                      total_trips3, total_trips4,
                                      total_trips5, total_trips6,
                                      total_trips7, total_trips8,
                                      total_trips9])

        return zone_cjtw, audit_numbers

    def split_audit(self,
                    gb_cjtw,
                    unq_hybrid):
        # Test split total is same as number of distinct zones
        # If so zone split has worked
        audit_set = gb_cjtw.reindex(['1_Areaofusualresidence', 'overlap_msoa_hybrid_pop_split_factor'], axis=1).drop_duplicates()
        split_sum = audit_set['overlap_msoa_hybrid_pop_split_factor'].sum()
        print('Total split:', split_sum)
        if split_sum == len(unq_hybrid):
            print('Total split factors match splits after joins')
            return True
        else:
            return False

    def factor_col(self,
                   df,
                   method=None,
                   total_col='3_Allcategories_Methodoftraveltowork',
                   working_col=None):
        # function to factor columns up or down by a factor in the same pandas row
        # method needs to be 'Up' or 'Down'
        if working_col is not None:
            if method == 'Up':
                print('factoring up', working_col)
                df[working_col] = df[working_col] * df[total_col]
            elif method == 'Down':
                print('factoring down', working_col)
                df[working_col] = df[working_col] / df[total_col]
            else:
                raise ValueError('No factoring method supplied: set method to \'Up\' or \'Down\'')
        else:
            print('No working column supplied')
        return df

    def _clean_cjtw(self,
                   cjtw):
        # TODO: Function to clean top end trips.
        # Iterate over by mode
        # Get zone to zone distance
        # Get sigma of distribution
        # Clean out any trips 3sigma over mean.
        return cjtw


