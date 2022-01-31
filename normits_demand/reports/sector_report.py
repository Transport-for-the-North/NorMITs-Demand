# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 11:34:00 2020

@author: genie
"""

import os

import pandas as pd

from typing import List

import normits_demand as nd
from normits_demand.matrices import translate_matrices
import normits_demand.utils.file_ops as fo


class SectorReporter:
    """
    Class to build sector reports from an output demand vector
    """

    def __init__(self,
                 target_folder: nd.PathLike,
                 model_name: str,
                 output_folder: nd.PathLike,
                 input_type: str = None,
                 model_schema: nd.PathLike = None,
                 model_sectors: nd.PathLike = None,
                 target_file_types: List = ['.csv', '.bz2']
                 ):

        """
        target_folder: file or folder of files to do sector reports for
        model_name: name of the model to look for
        input_type: from 'vector', 'long_matrix', 'wide_matrix'
        model_schema: folder to look for correspondences. will search in
        default folder if nothing provided
        model_sectors = path to .csv correspondence to sectors
        """

        # Init
        _default_import_folder = 'I:/NorMITs Demand/import'

        self.target_folder = target_folder

        # Pass init params to object
        self.model_name = model_name
        self.zone_id = '%s_zone_id' % model_name.lower()

        # If no model schema folder - find one
        if model_schema is None:
            home_list = os.listdir(_default_import_folder)
            model_schema = [x for x in home_list if model_name in x][0]
            model_schema = os.path.join(
                _default_import_folder,
                model_schema,
                'model schema')

        # Figure out input type
        # TODO: If nothing provided, work it out
        self.input_type = input_type

        # Set model schema
        self.model_schema = model_schema

        # Find sector correspondence
        if model_sectors is None:
            schema_dir = os.listdir(model_schema)
            corr_name = '%s_sector_correspondence.csv' % model_name.lower()
            sector_file = [x for x in schema_dir if corr_name in x][0]
            model_sectors = os.path.join(model_schema, sector_file)

        self.sectors = pd.read_csv(model_sectors)

        self.target_file_types = target_file_types

    def sector_report(self,
                      ca_report: bool = True,
                      three_sector_report: bool = True,
                      ie_report: bool = True):

        """
        sector_type: report requested - takes
            'ca_sector', 'three_sector', 'ie_sector'
        export = False:
            Write to object output dir, or not
        """

        # Index folder
        # TODO: Pull imports and parsing into line with NorMITs standard
        target_mats = os.listdir(self.target_folder)
        # Filter down to target file types
        target_mats = [x for y in self.target_file_types for x in target_mats if y in x]

        # Subset sectors into ie and 3 sector reports
        ca_sectors_2d = translate_matrices.convert_correspondence_to_wide(
            long_correspondence=self.sectors,
            primary_key=self.zone_id,
            secondary_key='ca_sector_2020_zone_id',
        )
        three_sectors_2d = translate_matrices.convert_correspondence_to_wide(
            long_correspondence=self.sectors,
            primary_key=self.zone_id,
            secondary_key='three_sector_id',
        )
        ie_2d = translate_matrices.convert_correspondence_to_wide(
            long_correspondence=self.sectors,
            primary_key=self.zone_id,
            secondary_key='ie_id',
        )

        # Apply translation
        mat_sector_reports = dict()
        # TODO: Assumes square matrices - needs to be smarter and better integrated
        for tm in target_mats:

            matrix_comp = dict()
            print(tm)
            mat = fo.read_df(os.path.join(self.target_folder, tm), index_col=0)

            v_mat = mat.values

            if ca_report:
                ca_sr = translate_matrices.matrix_zone_translation(
                    mat=v_mat,
                    sector_trans_mat=ca_sectors_2d.values)
                ca_sr = pd.DataFrame(ca_sr,
                                     index=ca_sectors_2d.columns,
                                     columns=ca_sectors_2d.columns)
                matrix_comp.update({'ca_sectors': ca_sr})

            if three_sector_report:
                three_sr = translate_matrices.matrix_zone_translation(
                    mat=v_mat,
                    sector_trans_mat=three_sectors_2d.values)
                three_sr = pd.DataFrame(three_sr,
                                        index=three_sectors_2d.columns,
                                        columns=three_sectors_2d.columns)
                matrix_comp.update({'three_sectors': three_sr})

            if ie_report:
                ie_sr = translate_matrices.matrix_zone_translation(
                    mat=v_mat,
                    sector_trans_mat=ie_2d.values)
                ie_sr = pd.DataFrame(ie_sr,
                                     index=ie_2d.columns,
                                     columns=ie_2d.columns)

                matrix_comp.update({'ie_sectors': ie_sr})

            mat_sector_reports.update({tm: matrix_comp})

        return mat_sector_reports
