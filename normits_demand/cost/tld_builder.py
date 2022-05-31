# -*- coding: utf-8 -*-
"""
Created on: 26/08/2020
Updated on:

Original author: Chris Storey
Last update made by: Ben Taylor
Other updates made by:

File purpose:

"""
# Built-Ins
import os

# Third Party
import pandas as pd

# Local Imports
import normits_demand as nd
from normits_demand import constants
from normits_demand.utils import file_ops


class TripLengthDistributionBuilder:
    # Class constants

    _geo_areas = ['gb', 'north', 'north_incl_ie', 'north_and_mids', 'north_and_mids_incl_ie']
    _region_filter_types = ['household', 'trip_OD']

    # LA Definitions
    _north_las = [
        'E06000001', 'E06000002', 'E06000003', 'E06000004', 'E06000005',
        'E06000006', 'E06000007', 'E06000008', 'E06000009', 'E06000010',
        'E06000011', 'E06000012', 'E06000013', 'E06000014', 'E06000021',
        'E06000047', 'E06000049', 'E06000050', 'E06000057', 'E07000026',
        'E07000027', 'E07000028', 'E07000029', 'E07000030', 'E07000031',
        'E07000033', 'E07000034', 'E07000035', 'E07000037', 'E07000038',
        'E07000117', 'E07000118', 'E07000119', 'E07000120', 'E07000121',
        'E07000122', 'E07000123', 'E07000124', 'E07000125', 'E07000126',
        'E07000127', 'E07000128', 'E07000137', 'E07000142', 'E07000163',
        'E07000164', 'E07000165', 'E07000166', 'E07000167', 'E07000168',
        'E07000169', 'E07000170', 'E07000171', 'E07000174', 'E07000175',
        'E07000198', 'E08000001', 'E08000002', 'E08000003', 'E08000004',
        'E08000005', 'E08000006', 'E08000007', 'E08000008', 'E08000009',
        'E08000010', 'E08000011', 'E08000012', 'E08000013', 'E08000014',
        'E08000015', 'E08000016', 'E08000017', 'E08000018', 'E08000019',
        'E08000021', 'E08000022', 'E08000023', 'E08000024', 'E08000032',
        'E08000033', 'E08000034', 'E08000035', 'E08000036', 'E08000037',
        'W06000001', 'W06000002', 'W06000003', 'W06000004', 'W06000005',
        'W06000006',
    ]
    _mid_las = [
        'E06000015', 'E06000016', 'E06000017', 'E06000018',
        'E07000032', 'E07000033', 'E07000034', 'E07000035',
        'E07000036', 'E07000037', 'E07000038', 'E07000039',
        'E07000129', 'E07000130', 'E07000131', 'E07000132',
        'E07000133', 'E07000134', 'E07000135', 'E07000136',
        'E07000137', 'E07000138', 'E07000139', 'E07000140',
        'E07000141', 'E07000142', 'E07000150', 'E07000151',
        'E07000152', 'E07000153', 'E07000154', 'E07000155',
        'E07000156', 'E07000170', 'E07000171', 'E07000172',
        'E07000173', 'E07000174', 'E07000175', 'E07000176',
        'E06000019', 'E06000020', 'E06000021', 'E06000051',
        'E07000192', 'E07000193', 'E07000194', 'E07000195',
        'E07000196', 'E07000197', 'E07000198', 'E07000199',
        'E07000234', 'E07000235', 'E07000236', 'E07000237',
        'E07000238', 'E07000239', 'E07000218', 'E07000219',
        'E07000220', 'E07000221', 'E07000222', 'E08000025',
        'E08000026', 'E08000027', 'E08000028', 'E08000029',
        'E08000030', 'E08000031'
    ]
    _north_and_mid_las = list(set(_north_las + _mid_las))

    # GOR definitions
    _north_gors = [1, 2, 3]
    _mid_gors = [4, 5]
    _north_and_mid_gors = list(set(_north_gors + _mid_gors))

    # HB/NHB definitions
    _hb_purposes = [1, 2, 3, 4, 5, 6, 7, 8]
    _nhb_purposes = [11, 12, 13, 14, 15, 16, 18]

    def __init__(self,
                 tlb_folder: nd.PathLike,
                 nts_import: nd.PathLike,
                 output_home: nd.PathLike,
                 ):

        # TODO: Would be good if the options were parsed by the class first,
        #  then you could choose to run all instead of 1x1

        self.tlb_folder = tlb_folder
        self.nts_import = nts_import
        self.output_home = output_home

        # ## USER SELECTS WHICH BANDS TO USE ## #
        band_path = os.path.join(tlb_folder, 'config', 'bands')
        band_options = os.listdir(band_path)
        band_options = [x for x in band_options if '.csv' in x]
        if len(band_options) == 0:
            raise ValueError('no trip length bands in folder')

        bands_confirmed = False
        while not bands_confirmed:
            for (i, option) in enumerate(band_options, 0):
                print(i, option)
            selection_b = input('Choose bands to aggregate by (index): ')
            band_text = band_options[int(selection_b)]

            bands = pd.read_csv(os.path.join(band_path, band_text))
            print('%d bands in selected' % len(bands))
            print(bands)

            if input('Keep these bands y/n').lower() == 'y':
                bands_confirmed = True

        self.trip_length_bands = bands
        seg_path = os.path.join(tlb_folder, 'config', 'segmentations')

        # ## USER SELECTS SEGMENTATION TO USE ## #
        seg_options = os.listdir(seg_path)
        seg_options = [x for x in seg_options if '.csv' in x]
        if len(seg_options) == 0:
            raise ValueError('no target segmentations in folder')

        segments_confirmed = False
        while not segments_confirmed:
            for (i, option) in enumerate(seg_options, 0):
                print(i, option)
            selection_s = input('Choose segments to aggregate by (index): ')
            segments_text = seg_options[int(selection_s)]

            segments = pd.read_csv(os.path.join(seg_path, segments_text))
            print(segments)

            if input('Keep these bands y/n') == 'y':
                segments_confirmed = True

        self.target_segmentation = segments

        # ## USER SELECTS GEOGRAPHICAL AREA TO USE ## #
        for (i, option) in enumerate(self._geo_areas, 0):
            print(i, option)
        selection_g = input('Choose geo-area (index): ')
        self.geo_area = self._geo_areas[int(selection_g)]

        # ## USER SELECTS TRIP SELECTION CONSTRAINT TO USE ## #
        print('Constrain to geographical areas by location of:')
        for (i, option) in enumerate(self._region_filter_types, 0):
            print(i, option)
        selection_r = input('How to apply region filter: ')
        self.region_filter = self._region_filter_types[int(selection_r)]

        # ## GENERATE OUTPUT PATH AND CONFIRM SELECTION ## #
        band_label = band_text.replace(' ', '_')
        segments_label = segments_text.replace(' ', '_')

        for char_out in ['(', ')', '.csv']:
            band_label = band_label.replace(char_out, '')
            segments_label = segments_label.replace(char_out, '')

        self.export = os.path.join(
            self.output_home,
            self.geo_area,
            self.region_filter,
            band_label,
            segments_label,
        )

        print("\nOutput path generated: %s" % self.export)
        if input('Output here? y/n\n').strip().lower() != 'y':
            print("y/Y not detected. Exiting.")
            exit()

        file_ops.create_folder(self.export)

        print('Loading processed NTS data from %s' % nts_import)
        self.nts_import = pd.read_csv(nts_import)
        self.nts_import['weighted_trip'] = self.nts_import['W1'] * self.nts_import['W5xHH'] * self.nts_import['W2']

    def _apply_geo_filter(self,
                          output_dat):
        """
        output_dat: processed NTS data

        if region filter is based on home, filters on a UA subset
        if it's based on trip ends (gor) filters on trip O/D
        """
        # If region filters are home end, filter by LA
        if self.region_filter == 'household':

            if self.geo_area == 'north':
                output_dat = output_dat[
                    output_dat['HHoldOSLAUA_B01ID'].isin(
                        self._north_las)]
                output_dat = output_dat.reset_index(drop=True)
            elif self.geo_area == 'north_and_mids':
                output_dat = output_dat[
                    output_dat['HHoldOSLAUA_B01ID'].isin(
                        self._north_and_mid_las)]
                output_dat = output_dat.reset_index(drop=True)
            elif self.geo_area == 'north_incl_ie':
                raise ValueError('i/e filter not compatible with home end filter')

        elif self.region_filter == 'trip_OD':

            if self.geo_area == 'north':
                # From O/D filter
                output_dat = output_dat[
                    output_dat['TripOrigGOR_B02ID'].isin(
                        self._north_gors)]
                output_dat = output_dat.reset_index(drop=True)
                # To O/D filter
                output_dat = output_dat[
                    output_dat['TripDestGOR_B02ID'].isin(
                        self._north_gors)]
                output_dat = output_dat.reset_index(drop=True)
            elif self.geo_area == 'north_incl_ie':
                # From filter only
                output_dat = output_dat[
                    output_dat['TripOrigGOR_B02ID'].isin(
                        self._north_gors)]
                output_dat = output_dat.reset_index(drop=True)
            elif self.geo_area == 'north_and_mids':
                output_dat = output_dat[
                    output_dat['TripOrigGOR_B02ID'].isin(
                        self._north_and_mid_gors)]
                output_dat = output_dat.reset_index(drop=True)
                # To O/D filter
                output_dat = output_dat[
                    output_dat['TripDestGOR_B02ID'].isin(
                        self._north_and_mid_gors)]
                output_dat = output_dat.reset_index(drop=True)
            elif self.geo_area == 'north_and_mids_incl_ie':
                output_dat = output_dat[
                    output_dat['TripOrigGOR_B02ID'].isin(
                        self._north_and_mid_gors)]
        return output_dat


    def run_tlb_lookups(self,
                        weekdays=[1, 2, 3, 4, 5],
                        agg_purp=list(), #[13, 14, 15, 18]
                        write=True):
        """
        weekdays: list of ints to consider default 1:5:

        agg_purp: purposes to aggregate

        region_filter: how to do regional subsets
        """
        # TODO: Need smart aggregation based on sample size threshold

        # Set target cols
        target_cols = ['SurveyYear', 'TravelWeekDay_B01ID', 'HHoldOSLAUA_B01ID', 'CarAccess_B01ID', 'soc_cat',
                       'ns_sec', 'main_mode', 'hb_purpose', 'nhb_purpose', 'nhb_purpose_hb_leg', 'Sex_B01ID',
                       'trip_origin', 'start_time', 'TripDisIncSW', 'TripOrigGOR_B02ID',
                       'TripDestGOR_B02ID', 'tfn_area_type', 'weighted_trip']

        output_dat = self.nts_import.reindex(target_cols, axis=1)

        # Build a list to record how many records
        records = list()
        records.append(len(output_dat))

        # CA Map
        """
        1	Main driver of company car
        2	Other main driver
        3	Not main driver of household car
        4	Household car but non driver
        5	Driver but no car
        6	Non driver and no car
        7	NA
        """
        ca_map = pd.DataFrame({'CarAccess_B01ID': [1, 2, 3, 4, 5, 6, 7],
                               'ca': [2, 2, 2, 2, 1, 1, 1]})

        output_dat = output_dat.merge(ca_map,
                                      how='left',
                                      on='CarAccess_B01ID')

        # map agg gor
        a_gor_from_map = pd.DataFrame({'agg_gor_from': [1, 2, 3, 4, 4, 4, 4, 5, 5, 5, 6],
                                         'TripOrigGOR_B02ID': [1, 2, 3, 4, 6,
                                                               7, 8, 5, 9, 10, 11]})
        a_gor_to_map = pd.DataFrame({'agg_gor_to': [1, 2, 3, 4, 4, 4, 4, 5, 5, 5, 6],
                                       'TripDestGOR_B02ID': [1, 2, 3, 4, 6,
                                                             7, 8, 5, 9, 10, 11]})

        output_dat = output_dat.merge(a_gor_from_map,
                                      how='left',
                                      on='TripOrigGOR_B02ID')

        output_dat = output_dat.merge(a_gor_to_map,
                                      how='left',
                                      on='TripDestGOR_B02ID')

        # Aggregate area type application
        agg_at = pd.DataFrame({'tfn_area_type': [1, 2, 3, 4, 5, 6, 7, 8],
                               'agg_tfn_area_type': [1, 1, 2, 2, 3, 3, 4, 4]})

        output_dat = output_dat.merge(agg_at,
                                      how='left',
                                      on='tfn_area_type')

        records.append(len(output_dat))

        # Filter to weekdays only
        output_dat = output_dat[
            output_dat['TravelWeekDay_B01ID'].isin(weekdays)].reset_index(drop=True)
        records.append(len(output_dat))

        # Geo filter on self.region_filter and self.geo_area
        output_dat = self._apply_geo_filter(output_dat)

        out_mat = []
        for index, row in self.target_segmentation.iterrows():

            print(row)
            op_sub = output_dat.copy()

            # Establish if PA cost or OD TLD
            if 'cost_type' in list(row):
                cost_type = row['cost_type']
            else:
                if 'p' in row:
                    if int(row['p']) in self._hb_purposes:
                        cost_type = 'pa'
                    elif int(row['p']) in self._nhb_purposes:
                        cost_type = 'od'
                    else:
                        raise ValueError('%d non-recognised purpose' % row['p'])

            # Seed values so they can go MIA
            trip_origin, purpose, mode, tp, soc, ns = [0, 0, 0, 0, 0, 0]
            tfn_at, agg_at, g, ca, agg_gor_from, agg_gor_to = [0, 0, 0, 0, 0, 0]

            for subset, value in row.iteritems():
                if subset == 'trip_origin':
                    op_sub = op_sub[op_sub['trip_origin'] == value].reset_index(drop=True)
                    trip_origin = value
                if subset == 'p':
                    if trip_origin == 'hb':
                        if cost_type == 'pa':
                            op_sub = op_sub[
                                (op_sub['nhb_purpose_hb_leg'] == value) |
                                (op_sub['hb_purpose'] == value)
                            ]
                        elif cost_type == 'od':
                            op_sub = op_sub[op_sub['hb_purpose'] == value].reset_index(drop=True)
                    elif trip_origin == 'nhb':
                        op_sub = op_sub[op_sub['nhb_purpose'] == value].reset_index(drop=True)
                    purpose = value
                if subset == 'ca':
                    if value != 0:
                        op_sub = op_sub[op_sub['ca'] == value].reset_index(drop=True)
                    ca = value
                if subset == 'm':
                    if value != 0:
                        op_sub = op_sub[op_sub['main_mode'] == value].reset_index(drop=True)
                    mode = value
                if subset == 'tp':
                    tp = value
                    if value != 0:
                        # Filter around tp to aggregate
                        time_vec: list = [value]
                        if purpose in agg_purp:
                            time_vec = [3, 4]
                        op_sub = op_sub[
                            op_sub['start_time'].isin(
                                time_vec)].reset_index(drop=True)
                if subset == 'soc_cat':
                    soc = value
                    if value != 0:
                        op_sub = op_sub[
                            op_sub['soc_cat'] == value].reset_index(drop=True)
                if subset == 'ns_sec':
                    ns = value
                    if value != 0:
                        op_sub = op_sub[
                            op_sub['ns_sec'] == value].reset_index(drop=True)
                if subset == 'tfn_area_type':
                    tfn_at = value
                    if value != 0:
                        op_sub = op_sub[
                            op_sub[
                                'tfn_area_type'] == value].reset_index(drop=True)
                if subset == 'agg_tfn_area_type':
                    agg_at = value
                    if value != 0:
                        op_sub = op_sub[
                            op_sub[
                                'agg_tfn_area_type'] == value].reset_index(drop=True)
                if subset == 'g':
                    g = value
                    if value != 0:
                        op_sub = op_sub[
                            op_sub['Sex_B01ID'] == value].reset_index(drop=True)
                if subset == 'agg_gor_to':
                    agg_gor_to = value
                    if value != 0:
                        op_sub = op_sub[
                            op_sub['agg_gor_to'] == value].reset_index(drop=True)
                if subset == 'agg_gor_from':
                    agg_gor_from = value
                    if value != 0:
                        op_sub = op_sub[
                            op_sub['agg_gor_from'] == value].reset_index(drop=True)

            out = self.trip_length_bands.copy()

            out['ave_km'] = 0
            out['trips'] = 0

            for line, thres in self.trip_length_bands.iterrows():

                tlb_sub = op_sub.copy()

                lower = thres['lower']
                upper = thres['upper']

                tlb_sub = tlb_sub[
                    tlb_sub['TripDisIncSW'] >= lower].reset_index(drop=True)
                tlb_sub = tlb_sub[
                    tlb_sub['TripDisIncSW'] < upper].reset_index(drop=True)

                mean_val = (tlb_sub['TripDisIncSW'].mean() * 1.61)
                total_trips = (tlb_sub['weighted_trip'].sum())

                out.loc[line, 'ave_km'] = mean_val
                out.loc[line, 'trips'] = total_trips

                del mean_val
                del total_trips

            out['band_share'] = out['trips']/out['trips'].sum()

            name = (trip_origin + '_tlb' + '_p' +
                    str(purpose) + '_m' + str(mode))
            if ca != 0:
                name = name + '_ca' + str(ca)
            if tfn_at != 0:
                name = name + '_at' + str(tfn_at)
            if agg_at != 0:
                name = name + '_aat' + str(agg_at)
            if tp != 0:
                name = name + '_tp' + str(tp)
            if soc != 0 or (purpose in [1,2,12] and 'soc_cat' in list(self.target_segmentation)):
                name = name + '_soc' + str(soc)
            if ns != 0 and 'ns_sec' in list(self.target_segmentation):
                name = name + '_ns' + str(ns)
            if g != 0:
                name = name + '_g' + str(g)
            if agg_gor_to != 0:
                name = name + '_gort' + str(agg_gor_to)
            if agg_gor_from != 0:
                name = name + '_gorf' + str(agg_gor_from)
            name += '.csv'

            ex_name = os.path.join(self.export, name)

            out.to_csv(ex_name, index=False)

            out['mode'] = mode
            out['period'] = tp
            out['ca'] = ca
            out['purpose'] = purpose
            out['soc'] = soc
            out['ns'] = ns
            out['tfn_area_type'] = tfn_at
            out['agg_tfn_area_type'] = agg_at
            out['g'] = g
            out['agg_gor_from'] = agg_gor_from
            out['agg_gor_to'] = agg_gor_to

            out_mat.append(out)

        final = pd.concat(out_mat)

        full_name = os.path.join(self.export, 'full_export.csv')

        if write:
            final.to_csv(full_name, index=False)

        return out_mat, final
