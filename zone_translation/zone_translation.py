# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 12:56:14 2019

Calling examples for nest functions

"""

import os

import zone_translation.geo_utils as nf
import zone_translation.zt_constants as ztc

class ZoneTranslation:

    def __init__(self,
                 zoning_name1,
                 zone_shape1,
                 zoning_name2,
                 zone_shape2,
                 export=None):
        if export is None:
            export = ztc.EXPORT
        self.export = export
        self.zoning_name1 = zoning_name1
        self.zoning_name2 = zoning_name2
        self.zone_shape1 = zone_shape1
        self.zone_shape2 = zone_shape2
        self.lsoa_trans1 = self.run_lsoa_translation(zoning_name1)
        self.lsoa_trans2 = self.run_lsoa_translation(zoning_name2)

    def spatial_translation(self,
                            upper_tolerance=.85,
                            lower_tolerance=.1,
                            zone1_index=0,
                            zone2_index=0,
                            write = True):
        """
        index is the reference of the unique zone identifier.
        """
        spatial_translation = nf.zone_nest(
            self.zone_shape1,
            self.zone_shape2,
            zone_name1=self.zoning_name1,
            zone_name2=self.zoning_name2,
            upper_tolerance = upper_tolerance,
            lower_tolerance = lower_tolerance,
            zone1_index = zone1_index,
            zone2_index = zone2_index)

        if write:
            spatial_translation.to_csv(
                os.path.join(
                    self.export,
                    self.zoning_name1.lower() +
                    '_to_' +
                    self.zoning_name2.lower() +
                    '.csv'), index=False)

        return spatial_translation

    def run_lsoa_translation(self,
                             zone_to_translate,
                             upperTolerance=.95,
                             lowerTolerance=.05,
                             zone1_index=0,
                             zone2_index=0,
                             write=True):
        """
        """
        if zone_to_translate == self.zoning_name1:
            trans_shape = self.zone_shape1
            trans_name = self.zoning_name1
            trans_index = self.zone1_index
        elif zone_to_translate == self.zoning_name2:
            trans_shape = self.zone_shape2
            trans_name = self.zoning_name2
            trans_index = self.zone2_index

        lsoa_translation = nf.zone_nest(
            ztc.ZLSOA,
            trans_shape,
            zone_name1='lsoa',
            zone_name2=trans_name,
            upper_tolerance=.95,
            lower_tolerance=.05,
            zone1_index=0,
            zone2_index=zone2_index)

        if write:
            lsoa_translation.to_csv(
                os.path.join(
                    self.export,
                    trans_name.lower() +
                    '_to_' +
                    self.zoning_name2.lower() +
                    '.csv'), index=False)

        return lsoa_translation

    def find_lsoa_translation(self,
                              zoning_name):

        """
        TODO: Index the existing lookups to LSOA, return it.
        If there isn't one, run one using the method above.
        """

        lsoatp = [x for x in ztc.EXPORT if zoning_name in x]
        lsoatp = [x for x in lsoatp if 'lsoa' in x]

        print(lsoatp)

        lsoatp =lsoatp[0]

        return lsoatp

    def weighted_translation(self,
                             translation_path1,
                             translation_path2,
                             method = 'lsoa_pop',
                             write=True):
        """
        Method: 'lsoa_pop' or 'lsoa_emp'
        """
        if method == 'lsoa_pop':
            method_name = '_pop_weighted'
        elif method == 'lsoa_emp':
            method_name = '_emp_weighted'
        else:
            raise ValueError('Invalid weighting method')

        weighted_translation = nf.zone_split(
            translation_path1,
            translation_path2,
            splitMethod=method)

        if write:
            weighted_translation.to_csv(
                os.path.join(
                    self.export,
                    self.zoning_name1.lower() +
                    '_' +
                    self.zoning_name2.lower() +
                    method_name +
                    '.csv'),
                index=False
            )

        return weighted_translation
