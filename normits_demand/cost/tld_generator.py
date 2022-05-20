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
from normits_demand.utils import file_ops


class TripLengthDistributionGenerator:
    # Class constants

    _geo_areas = ['north', 'north_incl_ie', 'north_and_mids', 'north_and_mids_incl_ie', 'gb']
    _trip_filter_types = ['trip_OD']
    _sample_periods = ['weekday', 'week', 'weekend']
    _cost_units = ['km', 'miles', 'm']

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

    # Maps for non-classified categories
    _household_type_to_ca = {'hh_type': [1, 2, 3, 4, 5, 6, 7, 8],
                             'ca': [1, 2, 1, 2, 2, 1, 2, 2]}

    _a_gor_from = pd.DataFrame({'agg_gor_from': [1, 2, 3, 4, 4, 4, 4, 5, 5, 5, 6],
                                'TripOrigGOR_B02ID': [1, 2, 3, 4, 6,
                                                      7, 8, 5, 9, 10, 11]})
    _a_gor_to = pd.DataFrame({'agg_gor_to': [1, 2, 3, 4, 4, 4, 4, 5, 5, 5, 6],
                              'TripDestGOR_B02ID': [1, 2, 3, 4, 6,
                                                    7, 8, 5, 9, 10, 11]})

    _tfn_at_to_agg_at = {'tfn_at': [1, 2, 3, 4, 5, 6, 7, 8],
                         'agg_tfn_at': [1, 1, 2, 2, 3, 3, 4, 4]}

    # Define weekdays
    _weekday_tps = [1, 2, 3, 4]

    segment_treatment = {'trip_origin': 'trip_origin',
                         'p': 'int',
                         'm': 'int',
                         'tp': 'tp'
                         }

    segment_order = [key for key, value in segment_treatment.items()]

    # Correspondences between segment names and NTS data names
    tld_to_nts_names = {'m': 'main_mode',
                        'p': 'p',
                        'tp': 'start_time',
                        'trip_origin': 'trip_direction'}

    # Miles to other units conversion factors
    miles_to_other_distance = {
        'miles': 1,
        'km': 1.6093,
        'm': 1609.34,
    }

    def __init__(self,
                 tlb_folder: nd.PathLike,
                 tlb_version: nd.PathLike,
                 output_folder: nd.PathLike,
                 trip_miles_col: str = 'trip_mile',
                 trip_distance_col: str = 'TripDisIncSW',
                 ):
        """
        Define the environment for a set of trip length distribution runs.

        Parameters
        ----------
        tlb_folder: pd.DataFrame
            Path to folder containing TLD specific output from
            'NTS Processing' tool
        tlb_version: str
            Which version of the TLD export to pick up
        output_folder:
            NorMITs Demand config folder to export outputs to
        trip_miles_col:
            Which column to use as the trip miles in the import data
        trip_distance_col:
            Which column to use as the trip distance in the import data
        """

        self.tlb_folder = tlb_folder
        self.tlb_version = tlb_version
        self.tlb_import_path = os.path.join(tlb_folder, tlb_version)
        print('Loading processed NTS trip length data from %s' % self.tlb_import_path)
        self.nts_import = pd.read_csv(self.tlb_import_path)
        self.output_folder = output_folder

        if trip_miles_col in list(self.nts_import):
            self.trip_miles_col = trip_miles_col
        else:
            raise ValueError(
                'Given trip miles col %s not in NTS data' % trip_miles_col
            )

        if trip_distance_col in list(self.nts_import):
            self.trip_distance_col = trip_distance_col
        else:
            raise ValueError(
                'Given trip distance col %s not in NTS data' % trip_distance_col
            )

    def _apply_geo_filter(self,
                          output_dat: pd.DataFrame,
                          trip_filter_type: str,
                          geo_area: str):
        """
        This function defines how the origin and destination of trips are
        derived and also defines regional subsets
        If region filter is based on home, filters on a UA subset
        If it's based on trip ends (gor) filters on trip O/D

        Parameters
        ----------
        output_dat: pd.DataFrame
            Processed NTS data
        trip_filter_type: str
            How to filter the origin/destination of trips.
            this is defined here in the class, so improvement work here.
        geo_area:
            Target regional subset

        Returns
        ----------
        output_dat:
            Input dataframe modified to target geography
        """

        # TODO: Work for household trip_filter_type properly

        # If region filters are home end, filter by LA
        filter_orig = False
        filter_dest = False
        target_orig_gors = None
        target_dest_gors = None

        if trip_filter_type == 'trip_OD':

            if geo_area == 'north':
                filter_orig = True
                filter_dest = True
                # From O/D filter
                target_orig_gors = self._north_gors
                # To O/D filter
                target_dest_gors = self._north_gors

            elif geo_area == 'north_incl_ie':
                # From filter only
                filter_orig = True
                filter_dest = False
                target_orig_gors = self._north_gors

            elif geo_area == 'north_and_mids':
                filter_orig = True
                filter_dest = True
                target_orig_gors = self._north_and_mid_gors
                target_dest_gors = self._north_and_mid_gors

            elif geo_area == 'north_and_mids_incl_ie':
                filter_orig = True
                target_orig_gors = self._north_and_mid_gors

        if filter_orig:
            output_dat = output_dat[
                output_dat['TripOrigGOR_B02ID'].isin(
                    target_orig_gors)]
            output_dat = output_dat.reset_index(drop=True)
        if filter_dest:
            output_dat = output_dat[
                output_dat['TripDestGOR_B02ID'].isin(
                    target_dest_gors)]
            output_dat = output_dat.reset_index(drop=True)

        return output_dat

    @staticmethod
    def _map_dict(output_dat: pd.DataFrame,
                  map_dict: dict,
                  key: str):

        """
        Analogue of pd.map for filling out a category from a dictionary
        output_dat: a DataFrame of NTS dataset
        map: a dictionary in category: list format
        key: string for join, existing category in output_dat
        """

        map_frame = pd.DataFrame(map_dict)

        output_dat = output_dat.merge(map_frame,
                                      how='left',
                                      on=key)

        return output_dat

    def _filter_to_weekday(self,
                           output_dat):
        """
        Subset a NTS table to weekdays only using 'TravelWeekDay'
        Correct weekdays defined in class
        """
        w_d = self._weekday_tps
        output_dat = output_dat[output_dat['start_time'].isin(w_d)]
        output_dat = output_dat.reset_index(drop=True)

        return output_dat

    def _build_band_subset(self,
                           seg_sub: pd.DataFrame,
                           bands: pd.DataFrame,
                           cost_units: str = 'km'):
        """
        Take a set of NTS data and distribute it to a set of given bands,
        counting trips per band and mean trip length per band.

        Parameters
        ----------
        seg_sub: pd.DataFrame
            DataFrame of NTS data, sub refers to the fact this should have
            been filtered down by this point in the process
        bands: pd.DataFrame
            DataFrame of target bands
        cost_units: str:
            Units for outputs. Inputs always in miles so is used to fetch
            a constant for conversion

        Returns
        ----------
        out_frame:
            Import bands DataFrame with appended trip and distance totals
        """

        dist_constant = self.miles_to_other_distance[cost_units]

        loc_bands = bands.copy()

        for line, threshold in loc_bands.iterrows():
            tlb_sub = seg_sub.copy()

            lower = threshold['lower']
            upper = threshold['upper']

            tlb_sub = tlb_sub[
                tlb_sub[self.trip_miles_col] >= lower].reset_index(drop=True)
            tlb_sub = tlb_sub[
                tlb_sub[self.trip_miles_col] < upper].reset_index(drop=True)

            total_miles = tlb_sub[self.trip_miles_col].sum()
            total_trips = tlb_sub[self.trip_distance_col].sum()

            mean_miles = sum(tlb_sub[self.trip_miles_col]*tlb_sub[self.trip_distance_col])
            mean_miles /= total_trips
            # Value adjusted for target distance
            mean_val = mean_miles * dist_constant

            loc_bands.loc[line, ('mean_%s' % cost_units)] = mean_val
            loc_bands.loc[line, 'total_trips'] = total_trips

        loc_bands['dist'] = loc_bands['total_trips']/loc_bands['total_trips'].sum()
        loc_bands['lower'] *= dist_constant
        loc_bands['upper'] *= dist_constant

        return loc_bands

    def _filter_segment(self,
                        seg_sub: pd.DataFrame,
                        trip_filter_type: str,
                        segment_name: str,
                        filter_value,
                        method: str = 'int'
                        ):
        """
        Core of process, filters the NTS data down to include the target
        segmentation only, one segment per call.
        Takes its method from the class dictionary, this function will have
        to be expanded to handle other bespoke segments as they arrive.
        This function also transposes the return home leg if the user requests
        PA costs by defining trip filter type as trip_OD.

        Parameters
        ----------
        seg_sub: pd.DataFrame
            NTS data in frame
        trip_filter_type: str
            Origin of trip definition
        segment_name: str
            The name of the segment in the TLD definition. Is translated to
            the equivalent NTS name
        filter_value:
            The target value of the above segment
        method:
            How to filter, as some segments require bespoke handling
            'tp' - if value is 0 ignore and don't label segments
            'trip_origin' - if hb, filter trip types to from home but if 'trip_OD' is
            requested, also reverse the origin/destination of to home trips and retain them

        Returns
        ----------
        nts_sub:
            Subset of NTS filtered to match a single value

        """
        # TODO: Handle values differently by type for consistency
        # TODO: Fix slice write warning - probably discrete function for transposition

        hb_types = ['hb_fr', 'hb_to']

        nts_sub = seg_sub.copy()

        nts_seg = self.tld_to_nts_names[segment_name]

        if method == 'tp':
            if filter_value != 0:
                nts_sub = self._filter_segment(
                    seg_sub=nts_sub,
                    trip_filter_type=trip_filter_type,
                    segment_name=segment_name,
                    filter_value=filter_value,
                    method='int')

        elif method == 'trip_origin':
            if filter_value == 'hb':
                if trip_filter_type == 'trip_OD':
                    nts_sub = nts_sub[nts_sub[nts_seg].isin(hb_types)]
                    # Transpose from and to for OD trip ends
                    to_orig = nts_sub[nts_sub[nts_seg] == 'hb_to']['TripOrigGOR_B02ID'].copy()
                    to_dest = nts_sub[nts_sub[nts_seg] == 'hb_to']['TripDestGOR_B02ID'].copy()
                    nts_sub[nts_sub[nts_seg] == 'hb_to']['TripOrigGOR_B02ID'] = to_dest
                    nts_sub[nts_sub[nts_seg] == 'hb_to']['TripDestGOR_B02ID'] = to_orig

        elif method == 'int':
            nts_sub = nts_sub[nts_sub[nts_seg] == filter_value]

        return nts_sub

    @staticmethod
    def _append_segment_names(tld: pd.DataFrame,
                              seg_descs: dict()):
        """
        Add the segment descriptions and names back into the distribution,
        so they're readable in aggregate and auditable against what the
        folder says they are

        Parameters
        ----------
        tld:
            Run dataframe of a distribution
        seg_descs:
            Dictionary of segment names and classifications

        Returns
        -------
        tld: pd.DataFrame
            input TLD with description columns
        """

        # Append segment sub-categories to tld matrix
        # retain order
        index_order = ['lower', 'upper']
        end_cols = list(tld)[1:]
        for segment, seg_val in seg_descs.items():
            tld[segment] = seg_val
            index_order.append(segment)

        index_order += end_cols

        tld = tld.reindex(index_order, axis=1)

        return tld

    def _build_single_tld_name(
            self,
            seg_descs,
            cost_units):

        """
        Build single names for the distribution, using its definition
        takes a standard order of construction from class

        Parameters
        ----------
        seg_descs:
            Dictionary of segment descriptions
        cost_units:
            Units used in totals to be appended to tld name

        Returns
        -------
        tld_name: str
            Name of individual segment
        """

        tld_name = str()

        for valid_name in self.segment_order:
            if valid_name in list(seg_descs.keys()):
                seg_value = str(seg_descs[valid_name])

                method = self.segment_treatment[str(valid_name)]

                if method == 'trip_origin':
                    tld_name += seg_value

                elif method == 'tp':
                    if seg_value != 0:
                        tld_name += '_' + valid_name + seg_value
                else:
                    tld_name += '_' + valid_name + seg_value

        tld_name += '_' + cost_units

        return tld_name

    def _build_set_tld_name(
            self,
            segments):
        """
        Build a set name for the distribution, using its definition
        takes a standard order of construction from class

        Parameters
        ----------
        seg_descs:
            List of segment descriptions
        cost_units:
            Units used in totals to be appended to tld name

        Returns
        -------
        seg_output_name: str
            Name of distribution at large
        """

        seg_output_name = str()

        seg_descs = list(segments)

        for valid_name in self.segment_order:
            if valid_name in seg_descs:

                method = self.segment_treatment[str(valid_name)]

                if method == 'trip_origin':
                    # is there only 1 trip origin
                    origin_types = segments[valid_name].unique()
                    # if so append to names
                    if len(origin_types == 1):
                        seg_output_name += origin_types[0]

                elif method == 'tp':
                    # If all tps are 0, omit from name
                    if bool(segments[valid_name].unique() == 0):
                        seg_output_name += '_' + valid_name
                else:
                    seg_output_name += '_' + valid_name

        return seg_output_name

    def _handle_sample_period(self,
                              input_dat,
                              sample_period):
        """
        Function to subset whole dataset for time periods
        """
        # TODO: Needs to be expanded to work for other time periods

        if sample_period == 'weekday':
            input_dat = self._filter_to_weekday(input_dat)

        return input_dat

    def _correct_defaults(self,
                          segments):
        """
        Assume missing segment classifications from NTS are the same as the input label
        Append those labels to the dictionary to avoid key errors later on

        Parameters
        ----------
        segments:
            list of segments from target segmentation, as defined in input csv
        Returns
        ----------
        defaults:
            list of segments missing in current form
        """

        defaults = {x: x for x in list(segments.columns) if x not in self.tld_to_nts_names}
        self.tld_to_nts_names.update(defaults)

        return defaults

    def build_tld(self,
                  input_dat: pd.DataFrame,
                  trip_filter_type: str,
                  bands: pd.DataFrame,
                  segments: pd.DataFrame,
                  cost_units: str,
                  verbose: bool = True):

        """
        Build a set of trip length distributions

        Parameters
        ----------
        input_dat: pd.DataFrame:
            Dataframe of pre-processed trip length distribution data
        trip_filter_type: str:
            Method of isolation for location, household or regional OD
        bands: pd.DataFrame:
            Dataframe of bands with headings lower, upper
        segments: pd.DataFrame:
            dataframe of segments by individual row
        cost_units: str:
            Units of distance, or in theory other cost
        verbose: bool:
          Echo or no

        Returns
        ----------
        tld_dict: dict
            A dictionary with {description of tld: tld DataFrame}
        seg_output_name: str
            the name of the output segmentation as a whole.
            derived from segments used
        """

        tld_dict = dict()

        # Handle lookup exceptions
        self._correct_defaults(segments)

        # Iterate over each individual segment descriptions
        for row_num, row in segments.iterrows():
            # Clone data for sub-setting
            seg_sub = input_dat.copy()
            seg_descs = dict()
            # Iterate
            for segment, seg_value in row.items():
                method = self.segment_treatment[str(segment)]
                # filter based on method
                seg_sub = self._filter_segment(
                    seg_sub=seg_sub,
                    trip_filter_type=trip_filter_type,
                    segment_name=str(segment),
                    filter_value=seg_value,
                    method=method)
                # Break loop if len is 0
                if len(seg_sub) == 0:
                    break
                else:
                    seg_descs.update({segment: seg_value})

            if verbose:
                print('Filtered for %s' % row)
                print('Remaining records %d' % len(seg_sub))

            # build tld
            tld = self._build_band_subset(
                seg_sub=seg_sub,
                bands=bands,
                cost_units=cost_units
            )


            tld = self._append_segment_names(
                tld,
                seg_descs
            )

            # build single tld name
            tld_name = self._build_single_tld_name(
                seg_descs,
                cost_units=cost_units)

            tld_dict.update({tld_name: tld})

        return tld_dict

    def tld_generator(self,
                      geo_area: str,
                      bands_path: nd.PathLike,
                      segmentation_path: nd.PathLike,
                      sample_period: str = 'week',
                      trip_filter_type: str = 'trip_OD',
                      cost_units: str = 'km',
                      verbose: bool = True,
                      write=True):

        # TODO: Can most of these be defined as class types to limit inputs?
        """
        Generate a consistent set of trip length distributions

        Parameters
        ----------
        geo_area: str:
            how to do regional subsets
            'north', 'north_incl_ie', 'north_and_mids', 'north_and_mids_incl_ie'
            should be limited by type in future
        bands_path: nd.PathLike:
            Path to a .csv describing bands to be used for tlds
        segmentation_path: nd.PathLike:
            Path to a .csv describing segmentation to be used
            Where cols = segments and rows = segment values
        sample_period:
            'weekday', 'week', 'weekend' - time period filter for target tld
            currently only handles week as import data build week
        trip_filter_type: str = 'trip_OD':
            How to define the start and end of trips. Currently only works
            for trip_OD, i.e. filter on the start and end of trip, but will
            work for household i.e where a house is with small modification
        cost_units: str = 'km',
            Units of distance to be output. Essentially picks a constant
            to multiply the native NTS miles by
            'miles', 'm', 'km'
        verbose: bool = True,
            Echo to terminal or not
        write: bool = True:
            Write export to class export folder

        Returns
        ----------
        tld_dict: dict
            A dictionary with {description of tld: tld DataFrame}
        full_export: pd.DataFrame
            A compiled, concatenated version of the DataFrames in tld_dict

        Future Improvements
        ----------
        Add more functionality for time period handling.
        Add better error control and type limiting for inputs.
        """

        input_dat = self.nts_import.copy()
        records = list()
        records.append(len(input_dat))

        # Import bands
        bands = pd.read_csv(bands_path)

        # Import segments
        segments = pd.read_csv(segmentation_path)

        # Limited input data pre-processing, should all really happen R side

        # Map categories not classified in classified build
        # Car availability
        input_dat = self._map_dict(output_dat=input_dat,
                                   map_dict=self._household_type_to_ca,
                                   key='hh_type')

        records.append(len(input_dat))

        # Filter to weekdays only
        input_dat = self._handle_sample_period(
            input_dat,
            sample_period=sample_period)

        records.append(len(input_dat))

        # Geo filter on self.region_filter and self.geo_area
        input_dat = self._apply_geo_filter(input_dat,
                                           trip_filter_type,
                                           geo_area
                                           )
        records.append(len(input_dat))

        # Build tld dictionary, return a proper name for the distributions
        tld_dict = self.build_tld(
            input_dat=input_dat,
            trip_filter_type=trip_filter_type,
            bands=bands,
            segments=segments,
            cost_units=cost_units,
            verbose=verbose
        )

        seg_output_name = self._build_set_tld_name(segments)

        # Build output path
        tld_out_path = os.path.join(
            self.output_folder,
            geo_area,
            seg_output_name
        )

        # Build full export
        full_export = list()
        for desc, dat in tld_dict.items():
            full_export.append(dat)
        full_export = pd.concat(full_export)

        if write:
            # for csv in mat
            file_ops.create_folder(tld_out_path)

            # TODO: Archive anything that's in this folder already, to ss
            # Write final compiled tld
            full_export.to_csv(
                os.path.join(tld_out_path, 'full_export.csv'), index=False)

            # Write individual tlds
            for path, df in tld_dict.items():
                csv_path = path + '.csv'
                individual_file = os.path.join(tld_out_path, csv_path)
                df.to_csv(individual_file, index=False)

        return tld_dict, full_export
