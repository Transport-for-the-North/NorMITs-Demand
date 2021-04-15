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
                 cjtw_path=tms_constants.CJTW_PATH,
                 cjtw_dat_file=tms_constants.CJTW_DAT,
                 cjtw_header_file=tms_constants.CJTW_HEADER,
                 hybrid_msoa_ref=tms_constants.Z_HYBRID_MSOA
                 ):
        """
        
        Params:
        model_folder:
            Path to output model folder - Model Zone Lookups or similar
        """
        
        self.model_folder = model_folder
        self.model_name = model_name
        self.cjtw_path = cjtw_path
        self.cjtw_dat_file = cjtw_dat_file
        self.cjtw_header_file = cjtw_header_file
        self.hybrid_msoa_ref = hybrid_msoa_ref

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

    def clean_cjtw(self,
                   cjtw):
        # TODO: Function to clean top end trips.
        # Iterate over by mode
        # Get zone to zone distance
        # Get sigma of distribution
        # Clean out any trips 3sigma over mean.
        return cjtw

    # What will be function parameter defs
    def cjtw_to_zone_translation(self,
                                 write=True):

        """
        Translate demand from census journey to work into a target zoning system
        """

        lookups = os.path.join(self.model_folder, 'Model Zone Lookups')

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

        # Look in model folder for 'hybrid pop' translation
        # Should be pathed to the model folder
        file_sys = os.listdir(lookups)
        msoa_hybrid_pop_lookup_path = [x for x in file_sys if 'pop_weighted' in x]
        msoa_hybrid_pop_lookup_path = [x for x in msoa_hybrid_pop_lookup_path
                                       if '_hybrid' in x][0]

        hybrid_msoa_trans = pd.read_csv(lookups +
                                        '/' +
                                        msoa_hybrid_pop_lookup_path)

        hmt_cols = ['msoa_hybrid_zone_id',
                   (self.model_name.lower() + '_zone_id'),
                   'overlap_msoa_hybrid_split_factor',
                   ('overlap_' + self.model_name.lower() + '_split_factor')]

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

        gb_cjtw = gb_cjtw.rename(columns={(
                self.model_name.lower() + '_zone_id'):(
                        '1_' + self.model_name.lower() +'Areaofresidence')})
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
                gb_cjtw[hmt_cols[2]]) # Overlap_msoa_hybrid_split_factor

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

        # Build reindex columns - cool method :D
        zone_cjtw_cols = [('1_' + self.model_name.lower() +'Areaofresidence'),
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
        msoa_hybrid_emp_lookup_path = [x for x in file_sys if (
                'emp_weighted') in x and 'hybrid' in x][0]
        hybrid_msoa_emp_trans = pd.read_csv(lookups +
                                            '/' +
                                            msoa_hybrid_emp_lookup_path)

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
        zone_audit = shapefiles.hybrid_zone_counts(zone_cjtw['2_Areaofworkplace'],
                                             unq_hybrid)
        zone_audit = shapefiles.hybrid_zone_counts(zone_cjtw['2_Areaofworkplace'],
                                             unq_hybrid)

        # Not dropping duplicates here, apparently they're required.
        zone_cjtw = zone_cjtw.rename(columns={(
                self.model_name.lower() + '_zone_id'):(
                        '2_' + self.model_name.lower() +'Areaofworkplace')})

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

        zone_cjtw_cols = [('1_' + self.model_name.lower() +'Areaofresidence'),
                          ('2_' + self.model_name.lower() + 'Areaofworkplace'),
                          '3_Allcategories_Methodoftraveltowork']
        for col in factor_cols:
            zone_cjtw_cols.append(col)

        zone_cjtw = zone_cjtw.reindex(zone_cjtw_cols,axis=1)
        total_trips8 = zone_cjtw['3_Allcategories_Methodoftraveltowork'].sum()

        # TODO: Upgrade CJTW to 2018!

        zone_cjtw = zone_cjtw.groupby(
                [('1_' +
                  self.model_name.lower() +
                  'Areaofresidence'),
            ('2_' +
             self.model_name.lower() +
             'Areaofworkplace')]).sum().reset_index()

        # TODO: write clean_cjtw
        zone_cjtw = self.clean_cjtw(zone_cjtw)

        total_trips9 = zone_cjtw['3_Allcategories_Methodoftraveltowork'].sum()

        zone_cjtw.to_csv(lookups + '/cjtw_' + self.model_name.lower() + '.csv',
                         index=False)

        # Segment Rail
        seg_cols = list(zone_cjtw)
        rail_segment = [seg_cols[0], seg_cols[1], '6_Train']
        zone_cjtw_rail = zone_cjtw.reindex(rail_segment, axis=1)

        zone_cjtw_rail.to_csv(lookups + '/cjtw_' + self.model_name.lower() + '_rail_only.csv', index=False)

        zone_cjtw_rail_excl = zone_cjtw.copy()
        zone_cjtw_rail_excl[
                '3_Allcategories_Methodoftraveltowork'] = (zone_cjtw_rail_excl[
                        '3_Allcategories_Methodoftraveltowork']-zone_cjtw_rail_excl['6_Train'])

        zone_cjtw_rail_excl = zone_cjtw_rail_excl.drop('6_Train', axis=1)

        zone_cjtw_rail_excl.to_csv(lookups + '/cjtw_' + self.model_name.lower() + '_rail_excl.csv', index=False)

        audit_numbers = pd.DataFrame([total_trips1, total_trips2,
                                      total_trips3, total_trips4,
                                      total_trips5, total_trips6,
                                      total_trips7, total_trips8,
                                      total_trips9])

        return zone_cjtw, audit_numbers
