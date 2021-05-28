# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 10:38:23 2020

@author: cruella
"""

import distribution_model as dm
import os
import matrix_processing as mp
import pandas as pd

def build_fusion_tp_matrices(distribution_segments,
                             model_name,
                             internal_24hr_productions,
                             internal_24hr_nhb_productions,
                             external_pa,
                             external_nhb_pa,
                             initial_betas,
                             paths, # This is complicated
                             time_period_splits,
                             production_splits,
                             nhb_production_splits,
                             all_zone_movements,
                             fusion_only = True,
                             fusion_modes = [6], # List
                             write_24hr_internal_pa = False,
                             write_24hr_pa = False,
                             write_tp_pa = False,
                             write_arrivals = False
                             ):
    
    """
    Function to replace the compile and export loop.

    Takes:
        Everything

    """
    # TODO: Full generalise
    # TODO: Get time period splits in function
    # TODO: Get production splits in function

    unq_purpose = [1,2,3,4,5,6,7,8]
    unq_nhb_purpose = [12,13,14,15,16,18]

    unq_seg = distribution_segments.copy()
    unq_seg.remove('purpose')
    unq_seg.remove('mode')

    # Define temporary export list, don't overwrite
    t_el = paths[4]

    # Fusion handling
    # Set export folders
    pa_export = t_el[12]
    od_export = t_el[13]
    
    initial_betas = dm.get_initial_betas(paths[1],
                                         distribution_type = 'hb',
                                         model_name = model_name)
    
    initial_betas = initial_betas[initial_betas['mode'].isin(fusion_modes)]

    initial_betas['source'] = 'moira_fusion'

    # TODO: This doesn't work for Noham anymore becuase there's a column - BAD FIX
    # Something like if seg is just called seg
    if len(list(unq_seg)) > 0:
        unq_seg = initial_betas.reindex(
                unq_seg,
                axis=1).drop_duplicates().reset_index(drop=True)
        no_segments = False
    else:
        a = {'seg': ['']}
        unq_seg = pd.DataFrame(a)
        no_segments = True

    # Use row.index, where index is col no to get segment names
    for index, row in unq_seg.iterrows():
        print(row)

        # Filter out betas to segs
        try:
            seg_betas = initial_betas[
                    initial_betas[
                            row.index[0]]==row[0]].reset_index(drop=True)
        except:
            # If fail, there's no segments
            print('No segments, passing initial betas')
            # Copy betas straight across
            seg_betas = initial_betas.copy()
            # No seg subset required from 24hr productions

        # Define unq mode for mode subset GB tp pa
        unq_mode = fusion_modes.copy()

        for mode_subset in unq_mode:

            print(mode_subset)
            # Reimport full set of betas
            mode_betas = seg_betas[
                    seg_betas[
                            'mode']==mode_subset].reset_index(drop=True)

            # So far takes 'cjtw', 'synthetic', 'moira_fusion'
            # TODO: This is pretty handy now but need to optimise output
            reimport_dict = {}
            reimport_dict.update({'moira_fusion':t_el[1]})

            # Filter internal_24hr_productions to mode only

            # Should never balance if there are fusion modes
            if fusion_modes is not None:
                balance = False
            else:
                balance = True

            # Create full matrix from re-imports
            internal_pa = dm.compile_internal_pa(reimport_dict,
                                              mode_betas, # initial_betas = mode_betas
                                              (model_name.lower() + '_zone_id'), # ia_name = (model_name.lower() + '_zone_id')
                                              internal_24hr_productions, # internal_24hr_productions
                                              distribution_segments, # distribution_segments
                                              row, # current_segment = row
                                              balance)

            # Resum segments in case any duplication
            ipa_cols = ['p_zone', 'a_zone']
            for ds in distribution_segments:
                ipa_cols.append(ds)

            internal_pa = internal_pa.groupby(
                    ipa_cols).sum().reset_index()

            # TODO: production report or trip length audit call

            # Export 24hr PA matrix

            # If segment row exists add to path
            if row is not None:
                internal_pa_path = (os.getcwd() +
                                    pa_export +
                                    '/hb_pa_internal_24hr_export_mode' +
                                    str(mode_subset) +
                                    '_' +
                                    row.index[0] +
                                    str(row[0]) +
                                    '.csv')
            # Otherwise do path without segment
            else:
                internal_pa_path = (os.getcwd() +
                                    pa_export +
                                    '/hb_pa_internal_24hr_export_mode' +
                                    str(mode_subset) +
                                    '.csv')

            # internal_pa = pd.read_csv(internal_pa_path)
            if write_24hr_internal_pa:
                print('Rxporting 24hr HB PA matrix')
                internal_pa.to_csv(internal_pa_path,
                                   index=False)

            # REAPPLY TIME PERIOD SPLITS TO GET TIME PERIOD PA

            # Define purpose list
            for purpose in unq_purpose:
                print('Compiling purpose ' +
                      str(purpose) +
                      ' at time period PA')

                # Can name the full export now
                export_name = ('_mode' +
                               str(mode_subset) +
                               '_purpose' +
                               str(purpose))
                if no_segments == False:
                    export_name = row.index[0] + str(row[0]) + export_name

                # External to Internal PA
                subset_external_pa = external_pa[
                        external_pa['mode']==mode_subset]
                subset_external_pa = subset_external_pa[
                        subset_external_pa ['purpose'] == purpose]
                # Flexible segment detection and filter for externals
                if row is not None:
                    for seg_heading in list(unq_seg):
                        subset_external_pa = subset_external_pa[
                                subset_external_pa[seg_heading] == row[seg_heading]]
                        print(seg_heading)

                subset_internal_pa = internal_pa[
                        internal_pa['mode']==mode_subset]
                subset_internal_pa = subset_internal_pa[
                        subset_internal_pa['purpose'] == purpose]

                subset_pa = pd.concat([subset_internal_pa,
                                       subset_external_pa], sort=True)

                _24hr_pa_path = (os.getcwd() +
                                 pa_export +
                                 '/hb_pa_' +
                                 export_name +
                                 '.csv')

                if write_24hr_pa:
                    subset_pa.to_csv(_24hr_pa_path,
                                     index=False)

                # Get subset total for comparison
                s_pa_t = subset_pa['dt'].sum()
                print(s_pa_t)

                # Add model name to resplit 24hr
                # TODO: This is where the segment split loop should start - avoid directly handling unsplit
                subset_tp_pa = dm.resplit_24hr_pa(paths[1],
                                               (model_name.lower() + '_zone_id'),
                                               subset_pa,
                                               # Default - does this need to be GB or Int only?
                                               splits = production_splits,
                                               mode_subset = mode_subset,
                                               purpose_subset = [purpose],
                                               aggregation_cols = None)

                # Comparison total
                s_pa_tp_t = subset_tp_pa['dt'].sum()

                print('Total before: ' + str(s_pa_t))
                print('Total after: ' + str(s_pa_tp_t))

                # TODO: Can delete subset_pa now?
                del(subset_pa)

                # Reindex tp_pa for mode purpose and time only
                # Can drop mode now, if there's only ever one

                # Define export path, export tp pa by mode
                subset_tp_pa_path = (os.getcwd() +
                                     pa_export +
                                     '/hb_tp_pa_' +
                                     export_name +
                                     '.csv')

                if write_tp_pa:
                    subset_tp_pa.to_csv(subset_tp_pa_path,
                                        index=False)

                # Split to OD
                # You can't do them in one go.
                # TODO: Call this from multiprocessing instead of in this script
                subset_tp_od = dm.pa_to_od(subset_tp_pa)

                # Split from
                subset_tp_od_from = subset_tp_od[0].reindex(
                        ['o_zone', 'd_zone',
                         'time', 'dt'],
                         axis=1).groupby(
                                 ['o_zone', 'd_zone',
                                  'time']).sum().reset_index()

                for tp in subset_tp_od_from['time'].drop_duplicates(
                        ).reset_index(drop=True):
                    print(tp)
                    mat = subset_tp_od_from[
                            subset_tp_od_from[
                                    'time']==tp].reset_index(drop=True)

                    del(mat['time'])
                    mat = mp.matrix_long_to_wide(mat,
                                                 all_zone_movements,
                                                 merge_cols = ['o_zone',
                                                               'd_zone'])

                    # Export matrix format OD - leave index on!
                    od_from_path = (os.getcwd() +
                                    od_export +
                                    '/hb_od_tp' +
                                    str(tp) +
                                    '_' +
                                    export_name +
                                    '_from.csv')

                    mat.to_csv(od_from_path)

                # Split to
                subset_tp_od_to = subset_tp_od[1].reindex(
                        ['o_zone', 'd_zone',
                         'time', 'dt'], axis=1).groupby(
                         ['o_zone', 'd_zone', 'time']).sum().reset_index()

                for tp in subset_tp_od_to['time'].drop_duplicates().reset_index(drop=True):
                    print(tp)
                    mat = subset_tp_od_to[subset_tp_od_to['time']==tp].reset_index(drop=True)

                    del(mat['time'])
                    mat = mp.matrix_long_to_wide(mat,
                                                 all_zone_movements,
                                                 merge_cols = ['o_zone',
                                                               'd_zone'])

                    od_to_path = (os.getcwd() +
                                  od_export +
                                  '/hb_od_tp' +
                                  str(tp) +
                                  '_' +
                                  export_name +
                                  '_to.csv')

                    mat.to_csv(od_to_path)

    initial_nhb_betas = dm.get_initial_betas(paths[1],
                                          distribution_type = 'nhb',
                                          model_name = model_name)

    initial_nhb_betas = initial_nhb_betas[initial_nhb_betas['mode'].isin(fusion_modes)]

    initial_nhb_betas['source'] = 'moira_fusion'

    unq_seg = distribution_segments.copy()
    unq_seg.remove('purpose')
    unq_seg.remove('mode')

    # TODO: This doesn't work for Noham anymore becuase there's a column - BAD FIX
    # Something like if seg is just called seg
    if len(list(unq_seg)) > 0:
        unq_seg = initial_nhb_betas.reindex(
                unq_seg,
                axis=1).drop_duplicates().reset_index(drop=True)
        no_segments = False
    else:
        a = {'seg': ['']}
        unq_seg = pd.DataFrame(a)
        no_segments = True

    # Redefine unq seg for nhb
    unq_seg = distribution_segments.copy()
    unq_seg.remove('purpose')
    unq_seg.remove('mode')

    # Filter to unique segs. If no unq_segs returns an empty df with single
    # index to iterate over
    unq_seg = initial_nhb_betas.reindex(
            unq_seg,
            axis=1).drop_duplicates().reset_index(drop=True)

    # If no segments reduce index
    if len(list(unq_seg))==0:
        unq_seg = unq_seg[:1]

    # Use row.index, where index is col no to get segment names
    for index, row in unq_seg.iterrows():
        print(row)

        # Filter out betas to segs
        # TODO: Needs to be a loop to do multiple segments
        try:
            seg_betas = initial_nhb_betas[
                    initial_nhb_betas[
                            row.index[0]]==row[0]].reset_index(drop=True)
        except:
            # If fail, there's no segments
            print('No segments, passing initial betas')
            # Copy betas straight across
            seg_betas = initial_nhb_betas.copy()
            # No seg subset required from 24hr productions

        # Define unq mode for mode subset GB tp pa
        unq_mode = fusion_modes.copy()

        for mode_subset in unq_mode:

            print(mode_subset)
            # Reimport full set of betas
            mode_betas = seg_betas[
                    seg_betas[
                            'mode']==mode_subset].reset_index(drop=True)

            # So far takes 'cjtw', 'synthetic', 'moira_fusion'
            # TODO: This is pretty handy now but need to optimise output
            reimport_dict = {}
            reimport_dict.update({'moira_fusion':t_el[1]})

            # Filter internal_24hr_productions to mode only

            # Should never balance if there are fusion modes
            if fusion_modes is not None:
                balance = False
            else:
                balance = True

            # Create full matrix from re-imports
            # TODO: Hasn't quite taken to the segments yet.
            internal_nhb_pa = dm.compile_internal_pa(reimport_dict,
                                                     mode_betas, # initial_betas
                                                     'o_zone',
                                                     internal_24hr_nhb_productions, # internal_24hr_productions
                                                     distribution_segments, # distribution_segments
                                                     row, # current_segment
                                                     balance)

            # Resum segments in case any duplication
            ipa_cols = ['p_zone', 'a_zone']
            for ds in distribution_segments:
                ipa_cols.append(ds)

            internal_nhb_pa = internal_nhb_pa.groupby(
                    ipa_cols).sum().reset_index()

            # TODO: production report or trip length audit call

            # Export 24hr PA matrix
            print('exporting 24hr NHB PA matrix')

            # If segment row exists add to path
            if row is not None:
                internal_nhb_pa_path = (os.getcwd() +
                                        pa_export +
                                        '/nhb_pa_internal_24hr_export_mode' +
                                        str(mode_subset) +
                                        '_' +
                                        row.index[0] +
                                        str(row[0]) +
                                        '.csv')
            # Otherwise do path without segment
            else:
                internal_nhb_pa_path = (os.getcwd() +
                                        pa_export +
                                        '/nhb_pa_internal_24hr_export_mode' +
                                        str(mode_subset) +
                                        '.csv')

            # Export
            if write_24hr_internal_pa:
                print('exporting 24hr NHB PA matrix')
                internal_nhb_pa.to_csv(internal_nhb_pa_path,
                                       index=False)

            # REAPPLY TIME PERIOD SPLITS TO GET TIME PERIOD OD

            # Define purpose list
            for purpose in unq_nhb_purpose:
                print(purpose)

                print('Compiling purpose ' +
                      str(purpose) +
                      ' at time period OD')

                # Can name the full export now
                export_name = ('_mode' +
                               str(mode_subset) +
                               '_purpose' +
                               str(purpose))
                if len(list(unq_seg))>0:
                    export_name = row.index[0] + str(row[0]) + export_name

                # Filter down on mode and purpose segments
                subset_external_nhb_pa = external_nhb_pa[
                        external_nhb_pa['mode']==mode_subset]
                subset_external_nhb_pa = subset_external_nhb_pa[
                        subset_external_nhb_pa ['purpose'] == purpose]
                # Flexible segment detection and filter for externals
                if row is not None:
                    for seg_heading in list(unq_seg):
                        subset_external_nhb_pa = subset_external_nhb_pa[
                                subset_external_nhb_pa[seg_heading] == row[seg_heading]]
                        print(seg_heading)

                subset_internal_nhb_pa = internal_nhb_pa[
                        internal_nhb_pa['mode']==mode_subset]
                subset_internal_nhb_pa = subset_internal_nhb_pa[
                        subset_internal_nhb_pa['purpose'] == purpose]

                subset_nhb_pa = pd.concat([subset_internal_nhb_pa,
                                       subset_external_nhb_pa], sort=True)

                _24hr_nhb_pa_path = (os.getcwd() +
                                     pa_export +
                                     '/nhb_pa_' +
                                     export_name +
                                     '.csv')

                if write_24hr_pa:
                    subset_nhb_pa.to_csv(_24hr_nhb_pa_path,
                                         index=False)

                # Get subset total for comparison
                s_pa_nhb_t = subset_nhb_pa['dt'].sum()
                print(s_pa_nhb_t)

                # Add model name to resplit 24hr
                # TODO: This is where the segment split loop should start - avoid directly handling unsplit
                subset_tp_nhb_pa = dm.resplit_24hr_pa(paths[1],
                                                      'o_zone',
                                                      subset_nhb_pa,
                                                      # Default - does this need to be GB or Int only?
                                                      splits = nhb_production_splits,
                                                      mode_subset = mode_subset,
                                                      purpose_subset = [purpose],
                                                      aggregation_cols = None)

                # Comparison total
                s_pa_tp_nhb_t = subset_tp_nhb_pa['dt'].sum()

                print('Total before: ' + str(s_pa_nhb_t))
                print('Total after: ' + str(s_pa_tp_nhb_t))

                # TODO: Can delete subset_pa now?
                del(subset_nhb_pa)

                # Reindex tp_pa for mode purpose and time only
                # TODO: Will need to have segments too eventually.
                # TODO: Could be done fairly smartly with adds to list

                # Can drop mode now, if there's only ever one

                # Define export path, export tp pa by mode
                subset_tp_nhb_pa_path = (os.getcwd() +
                                         pa_export +
                                         '/nhb_tp_pa_' +
                                         export_name +
                                         '.csv')
                if write_tp_pa:
                    subset_tp_nhb_pa.to_csv(subset_tp_nhb_pa_path,
                                            index=False)

                # TODO: Check column names
                for tp in subset_tp_nhb_pa['time'].drop_duplicates().reset_index(drop=True):
                    print(tp)
                    mat = subset_tp_nhb_pa[subset_tp_nhb_pa['time']==tp].reset_index(drop=True)
                    print(mat['dt'].sum())

                    mat = mat.rename(columns={'p_zone':'o_zone',
                                              'a_zone':'d_zone'})

                    del(mat['time'], mat['purpose'], mat['mode'])

                    mat = mat.groupby(
                            ['o_zone', 'd_zone']).sum().reset_index()

                    mat = mp.matrix_long_to_wide(mat,
                                                 all_zone_movements,
                                                 merge_cols = ['o_zone',
                                                               'd_zone'])

                    # Define nhb OD path
                    nhb_od_path = (os.getcwd() +
                                   od_export +
                                   '/nhb_od_tp' +
                                   str(tp) +
                                   '_' +
                                   export_name +
                                   '.csv')

                    # Export matrix format OD - leave index on!
                    mat.to_csv(nhb_od_path)

    return(True)

def build_gb_matrices(distribution_segments,
                      model_name,
                      build_type = 'hb',
                      mode_subset = None,
                      purpose_subset = None,
                             internal_24hr_productions,
                             internal_24hr_nhb_productions,
                             external_pa,
                             external_nhb_pa,
                             initial_betas,
                             paths, # This is complicated
                             production_splits,
                             nhb_production_splits,
                             all_zone_movements,
                             fusion_only = True,
                             fusion_modes = [6], # List
                             write_24hr_internal_pa = False,
                             write_24hr_pa = False,
                             write_tp_pa = False,
                             write_arrivals = False
                             ):
    
    """
    Function to replace the compile and export loop.

    Takes:
        distribution_segments:
        
        model_name:
            
        mode_subset: list like mode subset
        
        purpose_subset: list like purpose subset
        
        internal_24hr productions: DF of productions with target subset

    """
    
    print('This function shouldnt run until pathing is sorted')
    
    if not build_type.isin(['hb','nhb','fusion_hb', 'fusion_nhb']):
        raise ValueError('Unrecognised build_type param')

    # Fix right inputs lower function params
    if build_type == 'hb':
        beta_source = 'hb'
        production_col = (model_name.lower() + '_zone_id')
    elif build_type == 'nhb':
        beta_source = 'nhb'
        production_col = 'o_zone'

    # Set build type params
    if build_type == 'fusion_hb' or build_type == 'fusion_nhb':
        fusion = True
        # Catch fusion being done for all modes
        if mode_subset == None:
            raise ValueError('Fusion specified but no mode subset supplied')
        if build_type == 'fusion_hb':
            # HB params
            beta_source = 'hb'
            production_col = (model_name.lower() + '_zone_id')
        else:
            # NHB params  
            beta_source = 'nhb'
            production_col = 'o_zone'

    # Get time period splits
    time_period_splits = get_time_period_splits(
            path = dm._default_time_period_splits_path,
            aggregate = True)

    # TODO: Full generalise

    # TODO: Get production splits in function

    unq_purpose = [1,2,3,4,5,6,7,8]
    unq_nhb_purpose = [12,13,14,15,16,18]

    unq_seg = distribution_segments.copy()
    unq_seg.remove('purpose')
    unq_seg.remove('mode')

    # Define temporary export list, don't overwrite
    t_el = paths[4]

    # Fusion handling
    # Set export folders
    pa_export = t_el[12]
    od_export = t_el[13]

    initial_betas = dm.get_initial_betas(paths[1],
                                         distribution_type = beta_source,
                                         model_name = model_name)
    # Subsets and beta handling
    if mode_subset:
        initial_betas[initial_betas['mode'].isin(mode_subset)]
    
    if purpose_subset:
        print('Applying purpose subset - are you sure?')
        initial_betas[initial_betas['mode'].isin(purpose_subset)]

    if fusion:
        # Fix betas to handle fusion
        initial_betas['source'] = 'moira_fusion'

    # TODO: This doesn't work for Noham anymore becuase there's a column - BAD FIX
    # Something like if seg is just called seg
    if len(list(unq_seg)) > 0:
        unq_seg = initial_betas.reindex(
                unq_seg,
                axis=1).drop_duplicates().reset_index(drop=True)
        no_segments = False
    else:
        a = {'seg': ['']}
        unq_seg = pd.DataFrame(a)
        no_segments = True

    # Use row.index, where index is col no to get segment names
    for index, row in unq_seg.iterrows():
        print(row)

        # Filter out betas to segs
        try:
            seg_betas = initial_betas[
                    initial_betas[
                            row.index[0]]==row[0]].reset_index(drop=True)
        except:
            # If fail, there's no segments
            print('No segments, passing initial betas')
            # Copy betas straight across
            seg_betas = initial_betas.copy()
            # No seg subset required from 24hr productions

        # Define unq mode for mode subset GB tp pa
        unq_mode = fusion_modes.copy()

        for mode_subset in unq_mode:

            print(mode_subset)
            # Reimport full set of betas
            mode_betas = seg_betas[
                    seg_betas[
                            'mode']==mode_subset].reset_index(drop=True)

            # So far takes 'cjtw', 'synthetic', 'moira_fusion'
            # TODO: This is pretty handy now but need to optimise output
            reimport_dict = {}
            reimport_dict.update({'moira_fusion':t_el[1]})

            # Filter internal_24hr_productions to mode only

            # Should never balance if there are fusion modes
            if fusion:
                balance = False
            else:
                balance = True

            # TODO: Define name on hb / nhb

            # Create full matrix from re-imports
            internal_pa = dm.compile_internal_pa(reimport_dict,
                                              mode_betas, # initial_betas = mode_betas
                                              production_col, # ia_name
                                              internal_24hr_productions, # internal_24hr_productions
                                              distribution_segments, # distribution_segments
                                              row, # current_segment = row
                                              balance)

            # Resum segments in case any duplication
            ipa_cols = ['p_zone', 'a_zone']
            for ds in distribution_segments:
                ipa_cols.append(ds)

            internal_pa = internal_pa.groupby(
                    ipa_cols).sum().reset_index()

            # TODO: production report or trip length audit call

            # Export 24hr PA matrix

            # If segment row exists add to path
            # TODO: Relativise export calls
            if row is not None:
                internal_pa_path = (os.getcwd() +
                                    pa_export +
                                    '/hb_pa_internal_24hr_export_mode' +
                                    str(mode_subset) +
                                    '_' +
                                    row.index[0] +
                                    str(row[0]) +
                                    '.csv')
            # Otherwise do path without segment
            else:
                internal_pa_path = (os.getcwd() +
                                    pa_export +
                                    '/hb_pa_internal_24hr_export_mode' +
                                    str(mode_subset) +
                                    '.csv')

            # internal_pa = pd.read_csv(internal_pa_path)
            if write_24hr_internal_pa:
                print('Rxporting 24hr HB PA matrix')
                internal_pa.to_csv(internal_pa_path,
                                   index=False)

            # REAPPLY TIME PERIOD SPLITS TO GET TIME PERIOD PA

            # Define purpose list
            for purpose in unq_purpose:
                print('Compiling purpose ' +
                      str(purpose) +
                      ' at time period PA')

                # Can name the full export now
                export_name = ('_mode' +
                               str(mode_subset) +
                               '_purpose' +
                               str(purpose))
                if no_segments == False:
                    export_name = row.index[0] + str(row[0]) + export_name

                # External to Internal PA
                subset_external_pa = external_pa[
                        external_pa['mode']==mode_subset]
                subset_external_pa = subset_external_pa[
                        subset_external_pa ['purpose'] == purpose]
                # Flexible segment detection and filter for externals
                if row is not None:
                    for seg_heading in list(unq_seg):
                        subset_external_pa = subset_external_pa[
                                subset_external_pa[seg_heading] == row[seg_heading]]
                        print(seg_heading)

                subset_internal_pa = internal_pa[
                        internal_pa['mode']==mode_subset]
                subset_internal_pa = subset_internal_pa[
                        subset_internal_pa['purpose'] == purpose]

                subset_pa = pd.concat([subset_internal_pa,
                                       subset_external_pa], sort=True)

                _24hr_pa_path = (os.getcwd() +
                                 pa_export +
                                 '/hb_pa_' +
                                 export_name +
                                 '.csv')

                if write_24hr_pa:
                    subset_pa.to_csv(_24hr_pa_path,
                                     index=False)

                # Get subset total for comparison
                s_pa_t = subset_pa['dt'].sum()
                print(s_pa_t)

                # Add model name to resplit 24hr
                # TODO: This is where the segment split loop should start - avoid directly handling unsplit
                subset_tp_pa = dm.resplit_24hr_pa(paths[1],
                                               (model_name.lower() + '_zone_id'),
                                               subset_pa,
                                               # Default - does this need to be GB or Int only?
                                               splits = production_splits,
                                               mode_subset = mode_subset,
                                               purpose_subset = [purpose],
                                               aggregation_cols = None)

                # Comparison total
                s_pa_tp_t = subset_tp_pa['dt'].sum()

                print('Total before: ' + str(s_pa_t))
                print('Total after: ' + str(s_pa_tp_t))

                # TODO: Can delete subset_pa now?
                del(subset_pa)

                # Reindex tp_pa for mode purpose and time only
                # Can drop mode now, if there's only ever one

                # Define export path, export tp pa by mode
                subset_tp_pa_path = (os.getcwd() +
                                     pa_export +
                                     '/hb_tp_pa_' +
                                     export_name +
                                     '.csv')

                # TODO: don't do this if hb
                if write_tp_pa and beta_source is not 'nhb':
                    subset_tp_pa.to_csv(subset_tp_pa_path,
                                        index=False)

                # Split to OD
                # You can't do them in one go.
                
                # If you're doing an HB dist you want to go to OD too.
                if beta_source == 'hb':
                    # TODO: Call this from multiprocessing instead of in this script
                    subset_tp_od = dm.pa_to_od(subset_tp_pa, time_period_splits)

                    # Split from
                    subset_tp_od_from = subset_tp_od[0].reindex(
                            ['o_zone', 'd_zone',
                             'time', 'dt'],
                             axis=1).groupby(
                                     ['o_zone', 'd_zone',
                                      'time']).sum().reset_index()

                    for tp in subset_tp_od_from['time'].drop_duplicates(
                            ).reset_index(drop=True):
                        print(tp)
                        mat = subset_tp_od_from[
                                subset_tp_od_from[
                                        'time']==tp].reset_index(drop=True)
        
                        del(mat['time'])
                        mat = mp.matrix_long_to_wide(mat,
                                                     all_zone_movements,
                                                     merge_cols = ['o_zone',
                                                                   'd_zone'])

                        # Export matrix format OD - leave index on!
                        od_from_path = (os.getcwd() +
                                        od_export +
                                        '/hb_od_tp' +
                                        str(tp) +
                                        '_' +
                                        export_name +
                                        '_from.csv')

                        mat.to_csv(od_from_path)

                    # Split to
                    subset_tp_od_to = subset_tp_od[1].reindex(
                            ['o_zone', 'd_zone',
                             'time', 'dt'], axis=1).groupby(
                             ['o_zone', 'd_zone', 'time']).sum().reset_index()

                    for tp in subset_tp_od_to['time'].drop_duplicates(
                            ).reset_index(drop=True):
                        print(tp)
                        mat = subset_tp_od_to[subset_tp_od_to['time']==tp].reset_index(drop=True)

                        del(mat['time'])
                        mat = mp.matrix_long_to_wide(mat,
                                                     all_zone_movements,
                                                     merge_cols = ['o_zone',
                                                                   'd_zone'])

                        od_to_path = (os.getcwd() +
                                      od_export +
                                      '/hb_od_tp' +
                                      str(tp) +
                                      '_' +
                                      export_name +
                                      '_to.csv')
        
                        mat.to_csv(od_to_path)
                
                elif beta_source == 'nhb':
                    # TODO tp nhb od export loop.
                    subset_tp_od = subset_tp_pa.copy()
                    # TODO: Col renames
                    subset_tp_od = subset_tp_od.reindex(
                            ['o_zone', 'd_zone',
                             'time', 'dt'], axis=1).groupby(
                             ['o_zone', 'd_zone', 'time']).sum().reset_index()

                    for tp in subset_tp_od_to['time'].drop_duplicates(
                            ).reset_index(drop=True):
                        print(tp)
                        mat = subset_tp_od_to[subset_tp_od_to['time']==tp].reset_index(drop=True)

                        del(mat['time'])
                        mat = mp.matrix_long_to_wide(mat,
                                                     all_zone_movements,
                                                     merge_cols = ['o_zone',
                                                                   'd_zone'])
                        
                        od_path = (os.getcwd() +
                                   od_export +
                                   '/nhb_od_tp' +
                                   str(tp) +
                                   '_' +
                                   export_name +
                                    '_to.csv')
        
                        mat.to_csv(od_to_path)

    return(True)