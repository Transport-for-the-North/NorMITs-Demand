# -*- coding: utf-8 -*-
"""
Attraction Model for NorMITs Demand
"""
# Modules required
import os # File operations
from typing import List

import pandas as pd # Bread & butter
import numpy as np

from normits_demand.utils import utils as nup # Folder build utils
from normits_demand.utils import ntem_control as ntem
from normits_demand.utils.general import safe_dataframe_to_csv

_default_msoa_lad = ('Y:/NorMITs Synthesiser/import/lad_to_msoa.csv')
_default_attraction_weights = 'attraction_weights.csv'
_default_attraction_mode_split = 'attraction_mode_split.csv'
_default_ntem = ('Y:/NorMITs Synthesiser/import/' +
                 'ntem_constraints/ntem_pa_ave_wday_2018.csv')

class AttractionModel:
    
    """
    NorMITs Attraction model.
    Attraction end of NorMITs Trip End Model in the NorMITs Demand siote.
    """

    def __init__(self,
                 model_name: str = 'test',
                 build_folder: str = 'Y:/NorMITs Synthesiser',
                 iteration: str = 'iterx',
                 trip_origin: str = 'hb',
                 input_zones: str = 'msoa',
                 output_zones: str = 'test',
                 import_folder: str = 'Y:/NorMITs Synthesiser/import', # Add attraction data
                 model_folder: str = 'Y:/',
                 output_segments: List[str] = ['p','m'],
                 attractions_name: str = 'non_freight_msoa_2018.csv',
                 attraction_weights = _default_attraction_weights,
                 attraction_mode_split = _default_attraction_mode_split,
                 lad_path: str = _default_msoa_lad,
                 ntem_control: bool = True,
                 ntem_path: str = _default_ntem,
                 k_factor_control: bool = False,
                 k_factor_path = None,
                 export_msoa: bool = False,
                 export_lad: bool = False,
                 export_uncorrected: bool = False,
                 export_target: bool = True):

        # File Paths

        # Model setup variables
        self.model_name = model_name
        self.build_folder = build_folder
        self.iteration = iteration
        self.trip_origin = trip_origin
        self.input_zones = input_zones
        self.output_zones = output_zones
        # attraction data has its own folder in import
        self.import_folder = os.path.join(import_folder, 'attraction_data')
        self.model_folder = model_folder
        
        # Model option variables
        self.output_segments = output_segments
        self.attractions_name = attractions_name
        self.attraction_weights = attraction_weights
        self.ntem_control = ntem_control      
        if ntem_path == 'default':
            ntem_path = _default_ntem
        self.ntem_path = ntem_path
        # TODO: self.lad_path = lad_path Need this???
        self.k_factor_control = k_factor_control
        self.k_factor_path = k_factor_path
        self.export_msoa = export_msoa
        self.export_lad = export_lad
        self.export_uncorrected = export_uncorrected
        self.export_target = export_target
    
    def get_attractions(self,
                        source = 'flat',
                        ):
        """
        Imports raw attraction data from either a flat file on the the Y://
        or direct from the SQL server.
        
        Parameters
        ----------
        source:
            Where to look for the data. Will take 'flat' or 'sql', now just
            flat.
        
        flat_source:
            Path to flat source. Just points at global import path.
        
        Returns
        ----------
        all_msoa:
            All attractions by modelling employment types at msoa level.
        """
        flat_source = os.path.join(self.import_folder,
                                   self.attractions_name)

        # Blocking out most of this so it just imports everything.
        # BACKLOG: Adapt to handle the other 2 required datasets. Process if needed.
        all_msoa = pd.read_csv(flat_source)
    
        all_msoa = all_msoa.rename(columns={'geo_code':'msoa_zone_id'})
        
        return(all_msoa)
    
    def profile_attractions(self,
                            msoa_attractions,
                            attr_profile):
        """
        Takes an attraction profile defined by an import file in a model folder
        and creates attractions to those segments.
    
        Parameters
        ----------
        model_folder:
            Takes a model folder to look for an internal area definition.
    
        msoa_attractions:
            DataFrame containing msoa attractions by E segments, as imported from
            HSL data.
    
        Returns
        ----------
        a_list:
            A list of dataframe for each attraction type.    
        """
    
        # Build all job list for iteration
        all_jobs = list(msoa_attractions)
        all_jobs.remove('msoa_zone_id')
    
        # Count purposes
        a_count = list(attr_profile['purpose'])
    
        # Build placeholder
        a_list = []
    
        # Loop over and weight
        for attraction in a_count:
            # Make a copy of attractions as placeholder
            attr_data = msoa_attractions.copy()
    
            # Get unique weighting to apply
            weightings = attr_profile[attr_profile['purpose'] == attraction]
    
            for job in all_jobs:
                print(job)
                this_weight = weightings[job]
    
                attr_data.loc[:,job] = attr_data.loc[:,job]*float(this_weight)
                # TODO: Some audits to make sure nothing is falling out
    
            attr_data.loc[:,attraction] = attr_data[all_jobs].sum(axis=1)
            attr_data = attr_data.drop(all_jobs, axis=1)
    
            a_list.append(attr_data)
    
        return(a_list)
    
    def aggregate_to_model_zones_attr(self,
                                      attractions,
                                      model_zone_lookup_path,
                                      translation_name = None,
                                      max_level=True):
        """
        Function to aggregate to a target model zoning system. Translates from msoa
        base to whatever translation you give it.
        In future, will need control for splitting and custom zones included.
        
        Parameters
        ----------
        attractions:
            DataFrame, or a list of DataFrames describing attractions, likely to a 
            given attraction segmentation.
    
        model_zone_lookup_path:
            Flat file containing correspondence factors between msoa and target
            model zoning system.
    
        max_level = True:
            If True, drops columns used to calculate employment totals.
    
        Returns
        ----------
        attractions_w_zones:
            DataFrame or list of attractions as input, aggregated to model zones.
        """
    
        print(attractions['attractions'].sum())

        model_zone_conversion = pd.read_csv(model_zone_lookup_path)

        major_zone_name = list(model_zone_conversion)[0]
        # ia_name needs to pick up the minor zone name, should do usually
        minor_zone_name = list(model_zone_conversion)[1]
        # Count unique model zones in minor zoning system
        unq_minor_zones = model_zone_conversion[minor_zone_name].drop_duplicates()
        print(unq_minor_zones)

        # If there are zone splits - just go with the zone that overlaps the most
        # Slightly cheaty and I think can be fixed by multiplying out.
        # TODO: Replace with multiply out method

        # Reindex
        mzc_cols = [major_zone_name, minor_zone_name, translation_name]
        model_zone_conversion = model_zone_conversion.reindex(mzc_cols, axis=1)
        # Need another function to pick the biggest zone in a split & make sure it's not in the internal area
        # Print totals to topline.
    
        # Reapply zonal splits to each
        # Msoa is hard coded
    
        # Workaround for minor zone sometimes being major
        if 'msoa' in major_zone_name:
            group_col = minor_zone_name
        elif 'msoa' in minor_zone_name:
            group_col = major_zone_name
        
        # get group cols
        group_cols = [group_col]
        for col in list(attractions):
            if col != 'msoa_zone_id' and col != 'attractions':
                group_cols.append(col)
    
        # get sum cols
        sum_cols = group_cols.copy()
        sum_cols.append('attractions')
        
        attractions = attractions.merge(model_zone_conversion,
                                        how='inner',
                                        on='msoa_zone_id')
    
        attraction_w_zones = attractions.copy()
        
        attractions_w_zones = attractions[
                'attractions'] * attractions[translation_name]
    
        attractions_w_zones = attraction_w_zones.reindex(
                sum_cols,
                axis=1).groupby(
                        group_cols).sum(
                                ).reset_index()
    
        print(attractions_w_zones['attractions'].sum())
    
        return(attractions_w_zones)
    
    def run(self):

        """
        Function to run the attraction model. Takes a path to a lookup folder
        containing
    
        Parameters
        ----------
        model_folder = _zone_lookup_folder:
            Lookup folder for target model zoning system. Should contain the
            attraction splits required.
    
        input_zones_name:
            Name for aggregation in input data. Should be MSOA or LSOA.
    
        output_zones_name:
            Name for aggregation in output data. Should be the name of the target
            model zone.
    
        segmentation_type:
            'tfn' or 'ntem'. Defines which segments to write attractions for.
            If 'tfn', will build SOC splits into output attractions.
            If 'ntem', will build for 8 NTEM purposes only.
    
        model_zone_lookup_path:
            Path to area weighted correspondence between input and output zoning.
            Should be attraction weighting in future.
    
        write_output = True:
            Write csvs out or not.
    
        Returns
        ----------
        attractions:
            List of DataFrames containing attractions, split by attraction type.
            At ONS geography provided at input.
    
        all_attr:
            Single DataFrame with all attraction splits side by side. At ONS
            geography provided at input.
    
        zonal_attractions:
            List of DataFrames containing attractions, split by attraction type.
            At target model zoning system.
    
        all_zonal_attr:
            Single DataFrame with all attraction splits side by side.
            At target model zoning system.
    
        """
        # Define model zone lookup path
        model_zone_lookup_path = (self.model_folder +
                                  '/' +
                                  self.output_zones.lower() +
                                  '_msoa_emp_weighted_lookup.csv')

        # Set output params dict    
        if 'soc' in self.output_segments:
            print('Running soc weighting')
            # If there are TfN segments import the segmentation
            # BACKLOG: Could be more relative
            soc_weighted_jobs = pd.read_csv(
                os.path.join(
                    self.import_folder,
                    'soc_2_digit_sic_2018.csv'))

            # Make the soc integer a category
            soc_weighted_jobs['soc_class'] = 'soc' + soc_weighted_jobs[
                    'soc_class'].astype(int).astype(str)

            # Get uniq socs
            soc_segments = soc_weighted_jobs[
                    'soc_class'].drop_duplicates().reset_index(drop=True)

            # We should apply the SOC attractions as a zonal weight
            # not use it as a total jobs number
            # this will give us the benefit of model purposes in HSL data
            skill_index = ['msoa_zone_id', 'soc_class', 'seg_jobs']
            skill_group = skill_index.copy()
            skill_group.remove('seg_jobs')

            skill_weight = soc_weighted_jobs.reindex(
                    skill_index,
                    axis=1).groupby(skill_group).sum().reset_index()
            skill_weight = skill_weight.pivot(index='msoa_zone_id',
                                              columns='soc_class',
                                              values='seg_jobs').reset_index()
            skill_weight['total'] = skill_weight['soc1'] + skill_weight['soc2'] + skill_weight['soc3']

            # Reduce to factors
            for soc_seg in soc_segments:
                skill_weight[soc_seg] = skill_weight[soc_seg] / skill_weight['total']
        
            # Get rename total to soc0
            del(skill_weight['total'])
        
            # Rebuild soc segments with soc back in
            new_soc_segments = []
            for seg in soc_segments:
                new_soc_segments.append(seg)
            soc_segments = new_soc_segments.copy()
            del(new_soc_segments)
    
            # Define export segments
            group_segments = ['msoa_zone_id',
                              'p', 'm', 'soc']
            sum_segments = group_segments.copy()
            sum_segments.append('attractions')
    
        output_dir = os.path.join(self.build_folder,
                                  self.iteration,
                                  'Attraction Outputs')
        nup.create_folder(output_dir, chDir=False)

        print("Getting MSOA attractions")
        # Filters down to internal only here.
        # Uses global variables that aren't in the function call & shouldn't.
        msoa_attractions = self.get_attractions(source = 'flat')

        all_jobs = list(msoa_attractions)
        all_jobs.remove('msoa_zone_id')
    
        # Seed with all commute as E01
        msoa_attractions['E01'] = msoa_attractions[all_jobs].sum(axis=1)
    
        # Import attraction profile
        model_dir = os.listdir(self.model_folder)
        attr_profile = [x for x in model_dir if self.attraction_weights in x][0]
    
        # Profile attractions
        attr_profile = pd.read_csv(os.path.join(
            self.model_folder,
            attr_profile))

        # Profile and weight attractions
        attractions = self.profile_attractions(msoa_attractions, attr_profile)

        # BACKLOG: Move segmentation to after base is built
        if 'soc' in self.output_segments:
            
            # Split into segments to split and those to retain
            retain_ph = []
            split_ph = []
            
            soc_purposes = ['Commute', 'Business']
    
            for attr in attractions:
                attr_type = list(attr)[-1]
                print(attr_type)
                if attr_type in soc_purposes:
                    split_ph.append(attr)
                else:
                    retain_ph.append(attr)
    
            # soc_segments defined above
    
            tfn_attr = []
            for attr_frame in split_ph:
                purpose = list(attr_frame)[-1]
                for segment in soc_segments:
                    print('Building TfN segment ' +
                          purpose + '_' + segment)
                    # Copy skill weights
                    target_weights = skill_weight.copy()
                    
                    # Reindex to target cols
                    target_weights = target_weights.reindex(
                            ['msoa_zone_id', segment], axis=1)
    
                    # Join the SOC weighting factors
                    ph_frame = attr_frame.copy()
                    ph_frame = ph_frame.merge(target_weights,
                                              how='left',
                                              on='msoa_zone_id')
    
                    # Build weighted attractions with SOC
                    soc_seg_name = (purpose + '_' + segment)
                    ph_frame[soc_seg_name] = ph_frame[
                            purpose] * ph_frame[
                                    segment]
                    ph_frame = ph_frame.reindex(['msoa_zone_id', soc_seg_name], axis=1)
    
                    # Append to placeholder
                    tfn_attr.append(ph_frame)
                    # END
                # END

            # Append first list to end of TfN list
            for attr in retain_ph:
                tfn_attr.append(attr)
            del(retain_ph)
            del(attractions)
            attractions = tfn_attr
            del(tfn_attr)

        # Concatenate and consolidate attractions into single table.
        all_attr = pd.concat(attractions, sort=True)
        all_attr = all_attr.fillna(0)
        all_attr = all_attr.groupby('msoa_zone_id').sum().reset_index()
    
        col_heads = list(all_attr)
        col_heads.remove('msoa_zone_id')
    
        all_attr = pd.melt(all_attr, id_vars='msoa_zone_id', value_vars=col_heads,
                       value_name = 'attractions', var_name = 'ph')
    
        p_dist = pd.DataFrame({'ph' :
            ['Commute', 'Commute_soc1', 'Commute_soc2', 'Commute_soc3',
             'Business', 'Business_soc1', 'Business_soc2', 'Business_soc3',
             'Education', 'Shopping', 'Personal_business',
             'Recreation_social', 'Visiting_friends', 'Holiday_day_trip'],
             'p':[1,1,1,1,
                  2,2,2,2,
                  3,4,5,6,7,8],
                  'soc':['0', '1', '2', '3',
                         '0', '1', '2', '3',
                         '0','0','0','0','0','0']
        })
    
        if'soc' not in self.output_segments:
            p_dist = p_dist.drop('soc', axis=1)

        # Attach segments
        all_attr = all_attr.merge(p_dist,
                                  how='left',
                                  on='ph')
        all_attr = all_attr.drop('ph', axis=1)
    
        # Attach msoa mode_splits
        mode_splits = pd.read_csv(
            os.path.join(self.import_folder,
                         self.attraction_mode_split))

        all_attr = all_attr.merge(mode_splits,
                                  how='left',
                                  on = ['msoa_zone_id', 'p'])
    
        all_attr['attractions'] = all_attr['attractions'] * all_attr['mode_share']
        all_attr = all_attr.drop('mode_share', axis=1)
    
        all_attr = all_attr.reindex(sum_segments,
                                    axis=1).groupby(group_segments).sum(
              ).reset_index()
    
        all_attr = all_attr.sort_values(
                group_segments).reset_index(drop=True)
    
        hb_attr = all_attr.copy()
        nhb_attr = all_attr.copy()

        # Rename hb purpose to nhb
        nhb_attr = nhb_attr[nhb_attr['p']!=1]
        nhb_attr = nhb_attr[nhb_attr['p']!=7]
        nhb_attr['p'] = nhb_attr['p']+10

        # In the hack infill time period in nhb
        tp_infill = pd.DataFrame({'ph':[1,1,1,1],
                                  'tp':[1,2,3,4]})
        nhb_attr['ph'] = 1
        nhb_attr = nhb_attr.merge(tp_infill,
                                  how = 'left',
                                  on = 'ph')
        nhb_attr = nhb_attr.drop('ph', axis=1)
        del(tp_infill)
    
        ntem_lad_lookup = pd.read_csv('Y:/NorMITs Synthesiser/import/lad_to_msoa.csv')
    
        if self.export_uncorrected:
            safe_dataframe_to_csv(
                hb_attr,
                os.path.join(
                    self.output_dir,
                    'hb_attractions_uncorrected.csv'),
                index=False)
            safe_dataframe_to_csv(nhb_attr,
                os.path.join(
                    self.output_dir,
                    'nhb_attractions_uncorrected.csv'),
                index=False)
    
        if self.ntem_control:
            # Do an NTEM adjustment
    
            ntem_totals = pd.read_csv(self.ntem_path)

            hb_attr, hb_adj, hb_audit, hb_lad = ntem.control_to_ntem(
                hb_attr,
                ntem_totals,
                ntem_lad_lookup,
                group_cols = ['p', 'm'],
                base_value_name = 'attractions',
                ntem_value_name = 'Attractions',
                purpose = 'hb')
            print(hb_audit)

            nhb_attr, nhb_adj, nhb_audit, nhb_lad = ntem.control_to_ntem(
                nhb_attr,
                ntem_totals,
                ntem_lad_lookup,
                group_cols = ['p', 'm', 'tp'],
                base_value_name = 'attractions',
                ntem_value_name = 'Attractions',
                purpose = 'nhb')

            if self.export_lad:
                hb_lad.to_csv(
                    os.path.join(output_dir,
                                 'lad_hb_attractions.csv'),
                    index=False)
    
                nhb_lad.to_csv(
                    os.path.join(
                        output_dir,
                        'lad_nhb_attractions.csv'),
                    index=False)
            print(nhb_audit)
    
        # Control to k factors
        if self.k_factor_paths is not None:
            # Adjust to k factor for hb
            lad_lookup = pd.read_csv(self.lad_path)
            
            k_factors = os.listdir(self.k_factor_paths)[0]
            k_factors = pd.read_csv(
                os.path.join(self.k_factor_paths,
                             k_factors))
            k_factors = k_factors.reindex(['lad_zone_id','p','m','tp','attr_k'],
                                          axis=1)
    
            hb_purpose = [1,2,3,4,5,6,7,8]
            nhb_purpose = [12,13,14,15,16,18]
    
            hb_k_factors = k_factors[k_factors['p'].isin(hb_purpose)]
            hb_k_factors = hb_k_factors.drop('tp', axis=1)
            nhb_k_factors = k_factors[k_factors['p'].isin(nhb_purpose)]
    
            # Get hb_adjustment factors
            lad_lookup = lad_lookup.reindex(['lad_zone_id', 'msoa_zone_id'], axis=1)
            
            hb_attr = hb_attr.merge(lad_lookup,
                                    how = 'left',
                                    on = 'msoa_zone_id')
            # Seed zero infill
            hb_attr['attractions'] = hb_attr['attractions'].replace(0,0.001)
    
            # Build LA adjustment
            adj_fac = hb_attr.reindex(['lad_zone_id',
                                       'p',
                                       'm',
                                       'attractions'], axis=1).groupby(
            ['lad_zone_id',
             'p',
             'm']).sum().reset_index()
            adj_fac = adj_fac.merge(hb_k_factors,
                                    how = 'left',
                                    on = ['lad_zone_id',
                                          'p',
                                          'm'])
            adj_fac['adj_fac'] = adj_fac['attr_k']/adj_fac['attractions']
            adj_fac = adj_fac.reindex(['lad_zone_id',
                                       'p',
                                       'm',
                                       'adj_fac'], axis=1)
            adj_fac['adj_fac'] = adj_fac['adj_fac'].replace(np.nan, 1)
            
            hb_attr = hb_attr.merge(adj_fac,
                                    how = 'left',
                                    on = ['lad_zone_id',
                                          'p',
                                          'm'])
            hb_attr['attractions'] = hb_attr['attractions'] * hb_attr['adj_fac']
            
            hb_attr = hb_attr.drop(['lad_zone_id','adj_fac'], axis=1)
            
            # BACKLOG: TP attr
            nhb_attr = nhb_attr.merge(lad_lookup,
                                    how = 'left',
                                    on = 'msoa_zone_id')
            nhb_attr['attractions'] = nhb_attr['attractions'].replace(0,0.001)
            
            # Build LA adjustment
            adj_fac = nhb_attr.reindex(['lad_zone_id',
                                       'p',
                                       'm',
                                       'tp',
                                       'attractions'], axis=1).groupby(
            ['lad_zone_id',
             'p',
             'm',
             'tp']).sum().reset_index()
            adj_fac = adj_fac.merge(nhb_k_factors,
                                    how = 'left',
                                    on = ['lad_zone_id',
                                          'p',
                                          'm',
                                          'tp'])
            adj_fac['adj_fac'] = adj_fac['attr_k']/adj_fac['attractions']
            adj_fac = adj_fac.reindex(['lad_zone_id',
                                       'p',
                                       'm',
                                       'tp',
                                       'adj_fac'], axis=1)
            adj_fac['adj_fac'] = adj_fac['adj_fac'].replace(np.nan, 1)
            
            nhb_attr = nhb_attr.merge(adj_fac,
                                      how = 'left',
                                      on = ['lad_zone_id',
                                            'p',
                                            'm',
                                            'tp'])
            nhb_attr['attractions'] = nhb_attr['attractions'] * nhb_attr['adj_fac']
            
            nhb_attr = nhb_attr.drop(['lad_zone_id','adj_fac'], axis=1)
    
        # Write input attractions
        if self.export_msoa == True:
            hb_attr.to_csv(
                os.path.join(
                    self.output_dir,
                    self.input_zones.lower(),
                    '_hb_attractions.csv'),
                index=False)
            nhb_attr.to_csv(
                os.path.join(
                    self.output_dir,
                    self.input_zones.lower(),
                    '_nhb_attractions.csv'),
                index=False)

        # Aggregate input productions to model zones.
        zonal_hb_attr = self.aggregate_to_model_zones_attr(
            hb_attr,
            model_zone_lookup_path,
            translation_name = 'overlap_msoa_split_factor',
            max_level=True)

        zonal_nhb_attr = self.aggregate_to_model_zones_attr(
            nhb_attr,
            model_zone_lookup_path,
            translation_name = 'overlap_msoa_split_factor',
            max_level=True)
        
        hb_out_path = os.path.join(
            self.output_dir,
            self.output_zones.lower() +
            '_hb_attractions.csv')
        nhb_out_path = os.path.join(
            self.output_dir,
            self.output_zones.lower() +
            '_nhb_attractions.csv')

        # Write output totals
        if self.export_target == True:
            zonal_hb_attr.to_csv(
                hb_out_path,
                index=False)
            zonal_nhb_attr.to_csv(
                nhb_out_path,
                index=False)

        return [hb_out_path, nhb_out_path], [zonal_hb_attr, zonal_nhb_attr]