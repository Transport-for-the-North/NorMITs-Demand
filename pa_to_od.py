# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 11:40:10 2020

@author: cruella
"""
import os

import pandas as pd

import matrix_processing as mp
import utils as nup # Folder management, reindexing, optimisation

_default_time_period_splits_path = 'No' # TODO: No. Should be mode specific.

def init(movements, # TBC
         distribution_segments,
         init_params,
         o_paths):

    # BUILD INTERNAL 24HR PA BY MODE
    # get all zone movements for OD conversion
    # TODO: movements should be passed
    all_zone_movements = movements.copy()
    # Edit zone names
    all_zone_movements = all_zone_movements.rename(
                    columns={list(all_zone_movements)[0]:'o_zone',
                             list(all_zone_movements)[1]:'d_zone'})

    # Get time period splits for OD conversion
    # TODO: Add mode differentiation
    time_period_splits = get_time_period_splits(aggregate=True)

    # Fairly tricky bit of code below to deal with segments
    # Copy the dist segments and remove mode and purpose
    # TODO: pass distribution segments
    unq_seg = distribution_segments.copy()
    unq_seg.remove('purpose')
    unq_seg.remove('mode')

    return(None)

def build_tp_matrices(distribution_segments,
                      model_name,
                      internal_24hr_productions,
                      external_pa,
                      initial_betas,
                      paths, # This is complicated
                      time_period_splits,
                      unq_purpose, # This is where you specify HB or NHB
                      production_splits,
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

    unq_seg = distribution_segments.copy()
    unq_seg.remove('purpose')
    unq_seg.remove('mode')

    # Define temporary export list, don't overwrite
    t_el = paths[4]

    # Fusion handling
    if fusion_modes is not None:
        if fusion_only:
            # TODO: This is a bit dangerous in terms of overwriting stuff
            initial_betas = initial_betas[initial_betas['mode'].isin(fusion_modes)].reset_index(drop=True)
            initial_betas['source'] = 'fusion'
            # Set export folders
            pa_export = t_el[12]
            od_export = t_el[13]
        else:
            ValueError('Fusion modes provided but Fusion only not specified')
    else:
        # Set export folders
        pa_export = t_el[6]
        od_export = t_el[8]
        if fusion_only:
            ValueError('Please specify fusion modes, or toggle fusion only to False')

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
            seg_subset = internal_24hr_productions.copy()
            # Need to null out row for compile function
            row = None
        else:
            print('Subsetting segment ' + str(row.index[0]))
            # Filter internal 24 to segs.
            row.index[0]
            seg_subset = internal_24hr_productions[
                    internal_24hr_productions[
                            row.index[0]]==row[0]].reset_index(drop=True)

        # Define unq mode for mode subset GB tp pa
        unq_mode = [1,2,3,5,6]

        for mode_subset in unq_mode:

            print(mode_subset)
            # Reimport full set of betas
            mode_betas = seg_betas[
                    seg_betas[
                            'mode']==mode_subset].reset_index(drop=True)

            # So far takes 'cjtw', 'synthetic', 'moira_fusion'
            # TODO: This is pretty handy now but need to optimise output
            reimport_dict = {}
            reimport_dict.update({'synthetic':t_el[2]})
            reimport_dict.update({'cjtw':t_el[3]})
            reimport_dict.update({'moira_fusion':t_el[1]})

            # Filter internal_24hr_productions to mode only
            dist_subset = seg_subset[
                    seg_subset[
                            'mode']==mode_subset].reset_index(drop=True)

            # Should never balance if there are fusion modes
            if fusion_modes is not None:
                balance = False
            else:
                balance = True

            # Create full matrix from re-imports
            internal_pa = compile_internal_pa(reimport_dict,
                                              mode_betas, # initial_betas
                                              (model_name.lower() + '_zone_id'),
                                              dist_subset, # internal_24hr_productions
                                              distribution_segments, # distribution_segments
                                              row, # current_segment
                                              balance)

            # Resum segments in case any duplication
            ipa_cols = ['p_zone', 'a_zone']
            for ds in distribution_segments:
                ipa_cols.append(ds)

            internal_pa = internal_pa.groupby(
                    ipa_cols).sum().reset_index()

            # TODO: production report or trip length audit call

            # Export 24hr PA matrix
            print('exporting 24hr HB PA matrix')

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
                if no_segments:
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
                subset_tp_pa = resplit_24hr_pa(paths[1],
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
                # TODO: Call this from mp instead of in this script
                subset_tp_od = tp_pa_to_od(subset_tp_pa,
                                        time_period_splits = time_period_splits)

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

                # Export arrivals
                arrivals = subset_tp_od[2]

                arrivals_path = (os.getcwd() +
                                 export_list[7] +
                                 '/hb_' +
                                 export_name +
                                 '_arrivals.csv')

                if write_arrivals:
                    arrivals.to_csv(arrivals_path, index=False)
                # End of HB OD

    return(True)

def compile_internal_pa(reimport_dict,
                        initial_betas,
                        ia_name,
                        internal_24hr_productions,
                        distribution_segments,
                        current_segment = None,
                        balance = True):
    """
    This function takes a dictionary containing a pair of name to folder name,
    iterates over the items in the dictionary and imports the relevant
    distributed matrices based on the betas passed to the function.
    Then, it compiles them into a single distributed matrix for a full model
    area based on import parameters contained in the initial betas parameter
    table.

    Parameters
    ----------
    reimport_dict:
        Target folder. Will work for any folder containing csvs.

    initial_betas:
        Betas.

    ia_name:
        Name of internal area of model.

    internal_24hr_productions:
        Trips for internal area aggregated to 24hr.

    distribution_segments:
        List of column headings describing current segmentation. For grouping.

    current_segments = None:
        Current segment parameters for reimport. Needs to be a row or df.

    balance = True:
        Adjust reimported productions back up to target productions or not.

    Returns
    ----------
    matrix_list:
        Compiled csvs.
    """

    internal_24hr_productions = internal_24hr_productions.rename(
            columns={ia_name:'p_zone'})

    # Build columns to group and join by below
    group_cols = ['p_zone', 'a_zone']
    for ds in distribution_segments:
        group_cols.append(ds)

    index_cols = group_cols.copy()
    index_cols.append('dt')

    # Need to to this to get productions only
    group_cols.remove('a_zone')

    print(group_cols)
    print(index_cols)

    # Reimport
    hb_pa = []
    for name,path in reimport_dict.items():
        print(name)
        print(path)
        # Define mode and purposes required from this reimport
        mode_purpose = initial_betas[initial_betas['source']==name]

        #
        segment_lists = []

        if current_segment.index != 'null_seg':

            try:
                current_segment.index
            except:
                print('No segments')
            else:
                for seg in current_segment.index:
                    print(seg)
                    ph_segment = mode_purpose[seg].drop_duplicates(
                            ).reset_index(drop=True)
                    segment_lists.append(ph_segment)

        mode_list = mode_purpose['mode'].drop_duplicates(
                ).reset_index(drop=True)

        purpose_list = mode_purpose['purpose'].drop_duplicates(
                ).reset_index(drop=True)

        if name == 'cjtw':
            cjtw_sub = mode_list.copy()
            mode_list = None

        matrix = import_loop(import_folder = path,
                             mode_list = mode_list,
                             purpose_list = purpose_list,
                             segment_lists = segment_lists)

        if matrix is not None:

            # If it's cjtw then filter the matrix to the mode list
            # for single mode builds
            # This used to be outside the matrix loop call
            if name == 'cjtw':
                matrix = matrix[matrix['mode'].isin(
                        cjtw_sub)].reset_index(drop=True)

            print(list(matrix))
            hb_pa.append(matrix)

    hb_pa = pd.concat(hb_pa, sort=True)

    if balance:
        hb_pa_totals = hb_pa.drop('a_zone',axis=1).groupby(
                group_cols).sum().reset_index()
        hb_pa_totals = hb_pa_totals.merge(internal_24hr_productions,
                                          how='left',
                                          on = group_cols)
        hb_pa_totals['growth'] = hb_pa_totals['trips'] / hb_pa_totals['dt']
        hb_pa_totals = hb_pa_totals.drop(['dt', 'trips'],axis=1)
        hb_pa = hb_pa.merge(hb_pa_totals,
                            how='left',
                            on=group_cols)
        hb_pa['dt'] = hb_pa['dt'] * hb_pa['growth']
        hb_pa = hb_pa.drop(['growth'],axis=1)

    hb_pa = hb_pa.reindex(index_cols,axis=1)
    return(hb_pa)

def import_loop(import_folder,
                mode_list = None,
                purpose_list = None,
                segment_lists = None):
    """
    This function imports every csv from an import folder and appends them to a
    single DataFrame. Filters out distributions based on mode list and purpose
    list passed to function. The code is madness, frankly but it works.

    Parameters
    ----------
    import_folder:
        Target folder. Will work for any folder containing csvs.

    Returns
    ----------
    matrix_list:
        Compiled csvs.
    """
    target_dir = os.listdir(os.getcwd() + import_folder)
    if 'cjtw_purpose_1.csv' in target_dir:
        cjtw = True
    else:
        cjtw = False

    # Turn other segments into something we can check outputs for
    if segment_lists is not None:
        lookup_segment_list = []
        for segment in segment_lists:
            seg_ph = []
            for seg in segment:
                seg_ph.append(segment.name + '_' + str(seg))
                seg_ph.append(segment.name + str(seg))
            lookup_segment_list.append(seg_ph)
            del(seg_ph)

    # Turn mode list into something we can check outputs for
    if mode_list is not None:
        lookup_mode_list = []
        for mode in mode_list:
            lookup_mode_list.append('mode_' + str(mode))
            lookup_mode_list.append('mode' + str(mode))

    # Turn purpose list into something we can check outputs for
    if purpose_list is not None:
        lookup_purpose_list = []
        for purpose in purpose_list:
            lookup_purpose_list.append('purpose_' + str(purpose))
            lookup_purpose_list.append('purpose' + str(purpose) + '.')
    # Filter out omitted segments:
    if segment_lists is not None:
        segments_for_import = []
        for s_list in lookup_segment_list:
            ph_segment = []
            for segment in s_list:
                ph_segment.append([x for x in target_dir if segment in x])
            segments_for_import.append(ph_segment)
            del(ph_segment)
        for segment in segments_for_import:
            print(segment)
            segment = [inner for outer in segment for inner in outer]
    else:
        segments_for_import = target_dir

    # Filter out omitted modes
    if mode_list is not None:
        m_for_import = []
        for mode in lookup_mode_list:
            m_for_import.append([x for x in target_dir if mode in x])
        m_for_import = [inner for outer in m_for_import for inner in outer]
    else:
        m_for_import = target_dir

    # Filter out omitted purposes
    if purpose_list is not None:
        p_for_import = []
        for purpose in lookup_purpose_list:
            p_for_import.append([x for x in target_dir if purpose in x])
        p_for_import = [inner for outer in p_for_import for inner in outer]
    else:
        p_for_import = target_dir

    # Go back to target dir and filter out all omitted modes and purposes
    import_list = []
    for import_path in target_dir:
        if import_path in m_for_import and import_path in p_for_import:
            # If there are segments for import, append based on them
            if len(segments_for_import) > 0:
                for segment in segments_for_import:
                    if any(import_path in s for s in segment):
                        import_list.append(import_path)
            # Otherwise append above segment
            else:
                import_list.append(import_path)

    # Hard exception for census journey to work
    if cjtw:
        import_list.append(target_dir[0])
        print(import_list)
        print('Cjtw run found')

    matrix_list = []
    for matrix in import_list:
        # Import matrices and append to list
        print('Re-importing ' + matrix)
        ph = pd.read_csv(os.getcwd() +
                         import_folder +
                         '/' +
                         matrix)

        matrix_list.append(ph)
        del(ph)

    # Concatenate the imports, if they're there
    try:
        matrix_list = pd.concat(matrix_list)
        return(matrix_list)
    except ValueError:
        print('Nothing in', import_folder)
        return(None)

def tp_pa_to_od(mainland_gb_pa,
                time_period_splits = None):

    # TODO: Check if this will run with A non-mainland matrix
    """
    This function takes a mainland gb pa matrix and splits the trips out into
    O-D format. It also counts the arrivals

    Parameters
    ----------
    mainland_gb_pa:
        Matrix of mainland GB productions split by time and mode.

    time_period_splits:
        A dataframe of factors for splitting out 'from' trips to 'to' trips.

    Returns
    ----------
    [0] od_from:
        Origin half of PA matrix, from home legs.

    [1] od_to:
        Destination half of PA matrix, to home legs.

    [2] arrivals:
        Origin grouped od_to, retains split time period for NHB.
    """
    # Factor down to reduce segments

    # Get total 'before' productions
    total_pa_productions = mainland_gb_pa['dt'].sum()
    print(total_pa_productions)

    od_from = mainland_gb_pa.copy()
    del(mainland_gb_pa)
    od_from['dt'] = od_from['dt'].values/2
    od_from = od_from.rename(columns={'p_zone':'o_zone',
                                      'a_zone':'d_zone'})

    # Get total 'from' productions, should be half total above
    total_od_from_productions = od_from['dt'].sum()
    print('total od from productions ' + str(total_od_from_productions))

    # Rename columns in time_period_splits for
    time_period_splits = time_period_splits.rename(
            columns={'purpose_from_home':'purpose',
                     'time_from_home':'time'})

    # Covert to OD
    print('merging time splits')
    od_to = od_from.copy()

    # Possibly the culprit
    # Flip O-D
    od_to = od_to.rename(columns={'o_zone':'new_d_zone',
                                  'd_zone':'new_o_zone'})
    od_to = od_to.rename(columns={'new_d_zone':'d_zone',
                                  'new_o_zone':'o_zone'})

    # Get uniq purpose
    unq_purpose = od_to['purpose'].drop_duplicates().reset_index(drop=True)

    od_bin = []
    for pp in unq_purpose:
        print('Appending to purpose ' + str(pp))

        to_sub = od_to[od_to['purpose']==pp]
        to_sub = to_sub.merge(time_period_splits,
                              how = 'inner',
                              on = ['purpose','time'])

        del(to_sub['purpose'], to_sub['time'])
        to_sub = to_sub.rename(
                columns={'purpose_to_home':'purpose',
                         'time_to_home':'time'})

        to_sub['dt'] = (to_sub['dt'].values *
             to_sub['direction_factor'].values)
        del(to_sub['direction_factor'])

        # Regroup
        to_sub = to_sub.groupby(
                ['o_zone', 'd_zone', 'purpose',
                 'mode','time']).sum().reset_index()

        od_bin.append(to_sub)
        # End of loop

    od_to = pd.concat(od_bin, sort=True)
    del(od_bin)
    od_to = od_to.reindex(list(od_from),axis=1).reset_index(drop=True)

    total_od_to_productions = od_from['dt'].sum()
    print('total od to productions' + str(total_od_to_productions))

    od_from = nup.optimise_data_types(od_from)
    od_to = nup.optimise_data_types(od_to)

    arrival_cols = ['o_zone', 'purpose', 'mode', 'time', 'dt']
    arrivals = od_to.reindex(arrival_cols, axis=1).groupby(
            ['o_zone', 'purpose', 'mode', 'time']).sum().reset_index()

    return(od_from, od_to, arrivals)


def resplit_24hr_pa(model_lookup_path,
                    ia_name,
                    internal_pa,
                    splits = None,
                    mode_subset = None,
                    purpose_subset = None,
                    aggregation_cols = None):
    """
    This function takes a distributed 24hr PA matrix, imports the production
    splits from the relevant model folder, rejoins them and multiplies the
    distributed trips back out by the splitting factors.
    The subset depends on the splits given. If it goes to default it will do
    it for internal only.
    This is the crucial function in the model as it defines output splits
    and presents the largest memory bottleneck.

    Parameters
    ----------
    model_lookup_path:
        PA matrix for a given distribution.

    ia_name:
         Dataframe row containing target parameters for mode and purpose.

    internal_pa:
        I don't actually know! Can I add a mainland GB by mode here?

    aggregation_cols:
        Columns to aggregate by for aggregation exports. Includes core of
        ['p_zone', 'a_zone', 'mode', 'purpose'] by default. Time is non-core
        and will need to be specified.

    Returns
    ----------
    internal_pa:
        PA Matrix with time period splits added back in.
    """
    # TODO: Fix temporary fix here - whole thing should be import only.
    if splits is None:
        # Import internal splits splits
        splits = pd.read_csv(model_lookup_path + '/production_splits.csv')

    # Rename to generic column for join
    splits = splits.rename(columns={ia_name:'p_zone'})

    # Get pre re-split total
    pre_split_total = sum(internal_pa['dt'])

    # Filter on mode - takes int not list, assumes we only ever want 1 mode
    # Otherwise this will need to be as a list
    if mode_subset is not None:
        internal_pa = internal_pa[internal_pa['mode']==mode_subset]

    if purpose_subset is not None:
        internal_pa = internal_pa[internal_pa['purpose'].isin(purpose_subset)]

    # Chunk join for memory efficiency
    unq_purpose_mode = internal_pa.reindex(
            ['purpose', 'mode'],axis=1).drop_duplicates(
                    ).reset_index(drop=True)

    merge_placeholder = []

    for index,row in unq_purpose_mode.iterrows():
        print(row)
        ph_dat = internal_pa[internal_pa['purpose']==row['purpose']]
        ph_dat = ph_dat[ph_dat['mode']==row['mode']]
        print('subset before re-split: ' + str(ph_dat['dt'].sum()))

        ph_dat = ph_dat.merge(splits,
                              how='inner',
                              on=['p_zone',
                                  'mode',
                                  'purpose'])

        # Re-split trip rates
        ph_dat['dt'] = (ph_dat['dt'].values *
              ph_dat['time_split'].values)

        fpa_cols = ['p_zone',
                    'a_zone',
                    'mode',
                    'time',
                    'purpose',
                    'car_availability',
                    'employment_type',
                    'age',
                    'dt']
        ph_dat = ph_dat.reindex(fpa_cols,axis=1)

        print('subset after re-split: ' + str(ph_dat['dt'].sum()))

        # Loop to subset by given model categories for PA level writes
        # Needed for Norms output audits (Norms does its own PA-OD conversion)

        # Define core aggregations
        format_list = ['p_zone',
                       'a_zone',
                       'purpose',
                       'mode',
                       'time']

        write_list = ['purpose',
                      'mode',
                      'time']

        # append any aggregation cols
        if aggregation_cols is not None:

            for agg in aggregation_cols:
                format_list.append(agg)
                write_list.append(agg)

        format_list_dt = format_list.copy()
        format_list_dt.append('dt')

        print(format_list_dt)
        ph_dat = ph_dat.reindex(format_list_dt,
                                 axis=1).groupby(
                                         format_list).sum().reset_index()

        print('subset after aggregation: ' + str(ph_dat['dt'].sum()))

        print(list(ph_dat))
        # Optimise data types
        ph_dat = nup.optimise_data_types(ph_dat)

        merge_placeholder.append(ph_dat)
        del(ph_dat)

    # Concatenate df
    internal_pa = pd.concat(
            merge_placeholder,
            sort=False)
    del(merge_placeholder)

    # Reset index
    internal_pa = internal_pa.reset_index(
            drop=True)

    # Print audit
    print('total before re-split: ' + str(pre_split_total))
    print('after re-split: ' + str(sum(internal_pa['dt'])))

    return(internal_pa)


def get_time_period_splits(path = _default_time_period_splits_path,
                           aggregate = True):
    """
    This function imports time period split factors from a given path.

    Parameters
    ----------
    path:
        Path to .csv file containing required target trip lengths.

    Returns:
    ----------
    period_time_splits:
        DataFrame containing time split factors for pa to od.
    """
    # TODO: I'm sure there's a more efficient way to do what I've done below.

    period_time_splits = pd.read_csv(path)
    # Audit new totals

    if aggregate:
        # Remove none weekday time periods and refactor.
        from_cols = ['purpose_from_home', 'direction_factor']
        original_from_totals = period_time_splits.reindex(from_cols,axis=1)
        from_cols.remove('direction_factor')
        original_from_totals = original_from_totals.groupby(
                from_cols).sum().reset_index()
        original_from_totals = original_from_totals.rename(
                columns={'direction_factor':'old_total_purpose_factor'})

        period_time_splits = period_time_splits.merge(original_from_totals,
                                                      how='left',
                                                      on='purpose_from_home')
        period_time_splits.loc[:,'org_purpose_share'] = (period_time_splits[
                'direction_factor']/period_time_splits[
                        'old_total_purpose_factor'])
        period_time_splits = period_time_splits.rename(
                columns={'direction_factor':'original_direction_factor'})

        # Define new time periods
        time_periods = [1,2,3,4]
        weekday_time_splits = period_time_splits[
                period_time_splits['time_from_home'].isin(time_periods)]
        weekday_time_splits = weekday_time_splits[
                weekday_time_splits['time_to_home'].isin(time_periods)]

        from_cols = ['purpose_from_home', 'original_direction_factor']
        new_from_totals = weekday_time_splits.reindex(from_cols,axis=1)
        from_cols.remove('original_direction_factor')
        new_from_totals = new_from_totals.groupby(
                from_cols).sum().reset_index()
        new_from_totals = new_from_totals.rename(
                columns={'original_direction_factor':
                    'new_total_purpose_factor'})

        weekday_time_splits = weekday_time_splits.merge(
                new_from_totals,
                how='left',
                on='purpose_from_home')
        weekday_time_splits.loc[:,'direction_factor'] = (weekday_time_splits[
                'org_purpose_share'] * weekday_time_splits[
                        'new_total_purpose_factor'])

        del(weekday_time_splits['original_direction_factor'],
            weekday_time_splits['old_total_purpose_factor'],
            weekday_time_splits['org_purpose_share'],
            weekday_time_splits['new_total_purpose_factor'])

        # Factor back up so share == 1
        from_cols = ['purpose_from_home', 'time_from_home', 'direction_factor']
        wday_from_totals = weekday_time_splits.reindex(from_cols,axis=1)
        from_cols.remove('direction_factor')
        wday_from_totals = wday_from_totals.groupby(
                from_cols).sum().reset_index()
        wday_from_totals.loc[:,'gf'] = 1/wday_from_totals['direction_factor']
        del(wday_from_totals['direction_factor'])

        weekday_time_splits = weekday_time_splits.merge(
                wday_from_totals,
                how='left',
                on=['purpose_from_home', 'time_from_home'])
        weekday_time_splits.loc[:,'direction_factor'] = (weekday_time_splits[
                'direction_factor'] * weekday_time_splits['gf'])
        del(weekday_time_splits['gf'])

        # Audit new totals
        from_cols = ['purpose_from_home', 'time_from_home', 'direction_factor']
        wday_from_totals = weekday_time_splits.reindex(from_cols,axis=1)
        from_cols.remove('direction_factor')
        wday_from_totals = wday_from_totals.groupby(
                from_cols).sum().reset_index()

        # TODO: Proper error handle
        print('From-To split factors - should return 1s or conversion will' +
              ' drop trips')
        print(wday_from_totals['direction_factor'].drop_duplicates())

        # Assign back to main dataframe
        period_time_splits = weekday_time_splits.copy()

    period_time_splits = nup.optimise_data_types(period_time_splits)

    return(period_time_splits)