# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 13:47:56 2020

@author: genie
"""

import os

import pandas as pd
import numpy as np

from normits_demand.concurrency import multiprocessing as mp

from normits_demand.utils import utils as nup
from normits_demand.reports import reports_audits as ra

_TINY_INFILL = 1*10**-8

def read_pa_seg(pa_df,
                exc = ['trips','attractions','productions'],
                in_exc = ['zone']):
    """
    """
    cols = list(pa_df)
    cols = [x for x in cols if x not in exc]
    
    for ie in in_exc:
        drops = []
        for col in cols:
            if ie in col:
                drops.append(col)
    for d in drops:
        cols.remove(d)
    
    seg = pa_df.reindex(cols, axis=1).drop_duplicates().reset_index(drop=True)

    return(seg)
    
def _build_enhanced_pa(tld_mat,
                       df,
                       df_value,
                       col_limit,
                       ia_name = None,
                       unq_zone_list=None):
    """
    Private
    """
    if ia_name == None:
        ia_name = list(df)[0]
    if unq_zone_list == None:
        unq_zone_list = nup.get_zone_range(tld_mat[ia_name])

    out_list = []
    for itl, itd in tld_mat.iterrows():
        te = df.copy()
        calib_params = {}
        for col in col_limit:
            if itd[col] != 'none':
                if col in ['soc', 'ns']:
                    value = itd[col]
                else:
                    value = int(itd[col])
                calib_params.update({col:value})
        te, total = nup.filter_pa_vector(te,
                                         ia_name,
                                         calib_params,
                                         round_val = 3,
                                         value_var = df_value,
                                         echo=False)
        te = nup.df_to_np(te,
                          v_heading = ia_name,
                          values = df_value,
                          unq_internal_zones=unq_zone_list)

        if total == 0:
            raise Warning('Filter returned 0 trips, failing')
        calib_params.update({'trips':te})
        out_list.append(calib_params)

    return(out_list)

def _control_a_to_seed_matrix(sd,
                              attr_list):
    """
    Function to get target attraction vectors by splitting seed matrix
    cellwise on attractions
    
    Parameters
    ----------
    sd:
        Seed matrix, numpy format - as imported from TMS & compiled
    attr_list:
        List of attractions for new segments.
    Returns:
    ----------
    new_attr:
        Updated vector
    """
    cell_sum =  []
    for a in attr_list:
        cell_sum.append(a['trips'])
    cell_sum = np.where(sum(cell_sum)==0,0.000001,sum(cell_sum))
    
    cell_splits = []
    for a in attr_list:
        cell_splits.append(a['trips']/cell_sum)

    mat_attr = sd.sum(axis=0)

    new_attr = attr_list.copy()

    for ait,dat in enumerate(new_attr,0):
        new_attr[ait]['trips'] = mat_attr * cell_splits[ait]

    audit_sum =  []
    for a in new_attr:
        audit_sum.append(a['trips'])
    audit_sum = sum(audit_sum)

    audit = audit_sum.round() == mat_attr.round()

    return new_attr, audit

def _control_a_to_enhanced_p(prod_list,
                             attr_list):
    """
    Function to control a best estimate list of attraction vectors to a similar
    enhanced list of production vectors, as production vectors are more reliable.
    Looks up a fitting attraction vector for the productions, adds segments
    from productions to attractions, balances attractions to productions.
    
    Takes
    ------
    prod_list:
        List like of a dictionary containing production vectors (calib params
        plus zonal prods)
    attr_list:
        As above, but attractions. Should have less detailed segmentation.
    Returns
    -----
    new_attr:
        Further segmented, balanced attractions
    audit:
        Boolean - did it work
    """
    # Define empty list for output
    new_attr = []
    for prod in prod_list:
        # Build empty dict for calib params only
        p_types = {}
        # Get sum of productions
        p_tot = prod['trips'].sum()
        # Iterate over prod items and build calib params dict
        for p_item, p_dat in prod.items():
            if p_item != 'trips':
                p_types.update({p_item:p_dat})
        # Iterate over attr items
        for attr in attr_list:
            work_dict = attr.copy()
            # Find one that matches
            for item, dat in work_dict.items():
                record_match = True
                # If they don't match kick the loop
                if item in p_types.keys():
                    if dat != p_types[item]:
                        record_match = False
                        break
            # If the record matches, build out calib params and balance
            if record_match:
                # Update dict with anything not there
                for p_item, p_dat in p_types.items():
                    if p_item not in work_dict.keys():
                        work_dict.update({p_item:p_dat})
                # Balance - make copy of original
                new_frame = work_dict['trips'].copy()
                # Get demand as factor
                new_frame = new_frame/sum(new_frame)
                # Multiply factor by production demand
                new_frame = new_frame*p_tot
                # Put new df in dict
                work_dict.update({'trips':new_frame})
                new_attr.append(work_dict)
                # Don't do it again (will square)
                break

    audit = []
    # Audit balance
    for ri in list(range(len(prod_list))):

        a = prod_list[ri]['trips'].sum().round(0)
        b = new_attr[ri]['trips'].sum().round(0)

        if a == b:
            audit.append(True)
        else:
            audit.append(False)

    # Lists should have popped in order and so x_prod, should equal x_attr

    return(new_attr, audit)

"""
import_folder = 'Y:/NorMITs Synthesiser/Noham/iter8c/Distribution Outputs/PA Matrices'
target_tld_folder = 'Y:/NorMITs Synthesiser/import/trip_length_bands/north/enhanced_segments'
# Using single tld for whole country - run North and GB and compare
base_hb_productions = 'Y:/NorMITs Synthesiser/Noham/iter8c/Production Outputs/hb_productions_noham.csv'
base_nhb_productions = 'Y:/NorMITs Synthesiser/Noham/iter8c/Production Outputs/nhb_productions_noham.csv'
base_hb_attractions = 'Y:/NorMITs Synthesiser/Noham/iter8c/Attraction Outputs/noham_hb_attractions.csv'
base_nhb_attractions = 'Y:/NorMITs Synthesiser/Noham/iter8c/Attraction Outputs/noham_nhb_attractions.csv'
export_folder = 'Y:/NorMITs Synthesiser/Noham/iter8c/Distribution Outputs/Segmented Distributions'
lookup_folder = 'Y:/NorMITs Synthesiser/Noham/Model Zone Lookups'

base_productions_path = base_hb_productions
base_attractions_path = base_hb_attractions

"""

def disaggregate_segments(import_folder,
                          target_tld_folder,
                          base_productions_path,
                          base_attractions_path,
                          export_folder,
                          lookup_folder,
                          aggregate_surplus_segments = True,
                          rounding = 5,
                          trip_origin = 'hb',
                          tp = '24hr',
                          iz_infill = 0.5,
                          furness_loops = 1999,
                          min_pa_diff = .1,
                          bs_con_crit = .975,
                          mp_threads = -1,
                          export_original = True,
                          export_furness = False):

    """
    Parameters
    ----------
    trip_origin:
        from 'hb', 'nhb', 'both'
    
    aggregate_surplus_segments = True:
        If there are segments on the left hand side that aren't in the
        enhanced segmentation, aggregated them. Will 
    """

    # Look at segmentation in base matrices
    base_mat_seg = nup.parse_mat_output(os.listdir(import_folder),
                                        sep = '_',
                                        mat_type = 'pa',
                                        file_name = 'base_seg').drop(
                                                'mat_type', axis=1)

    # Look at segmentation in tld
    tld_seg = nup.parse_mat_output(os.listdir(target_tld_folder),
                                   sep = '_',
                                   mat_type = 'tlb',
                                   file_name = 'enh_tld').drop(
                                                'mat_type', axis=1)

    # Look at segmentation in base p
    base_p = pd.read_csv(base_productions_path)
    # Get unq zones
    ia_name = list(base_p)[0]
    min_zone = min(base_p[ia_name])
    max_zone = max(base_p[ia_name])
    unq_zones = pd.DataFrame(
            [i for i in range(int(min_zone),int(max_zone)+1)])
    unq_zones = unq_zones.rename(
            columns={list(unq_zones)[0]:ia_name})

    # Get zone range
    unq_zone_list = nup.get_zone_range(base_p[ia_name])

    # Look at segmentation in base a
    base_a = pd.read_csv(base_attractions_path)

    # Get p_seg
    p_seg = read_pa_seg(base_p,
                        exc = ['trips','attractions','productions'],
                        in_exc = ['zone'])

    # get a
    a_seg = read_pa_seg(base_a,
                        exc = ['trips','attractions','productions'],
                        in_exc = ['zone'])

    ## Build iteration parameters

    # Get cols in both the base and the tld
    merge_cols = [x for x in list(tld_seg) if x in list(base_mat_seg)]

    # Get cols in base that will have to be aggregated
    agg_cols = [x for x in list(
            base_mat_seg) if x not in list(tld_seg)]
    agg_cols.remove('base_seg')

    # Get cols in tld that form the added segments
    add_cols = [x for x in list(tld_seg) if x not in list(base_mat_seg)]
    add_cols.remove('enh_tld')

    # Define costs to lookup for later
    # If time period is involved in the merge, it'll be tp

    # Build a set of every tld split and every aggregation
    it_set = base_mat_seg.merge(tld_seg,
                                how = 'inner',
                                on = merge_cols)
    
    # Drop surplus segments on import descriptors - if you want
    if aggregate_surplus_segments:
        it_set = it_set.drop(agg_cols,axis=1)
        base_mat_seg = base_mat_seg.drop(agg_cols,axis=1)

    # Get cols in productions that are also in it set
    p_cols = [x for x in list(it_set) if x in list(p_seg)]
    # get cols in attractions that are also in it set
    a_cols = [x for x in list(it_set) if x in list(a_seg)]

    # Apply any hard constraints here eg, origin, mode
    if trip_origin != 'both':
        it_set = it_set[
                it_set['trip_origin']==trip_origin].reset_index(drop=True)

    # Build a set of every unique combo of base / target split - main iterator
    unq_splits = it_set.copy(
            ).reindex(
                    merge_cols, axis=1).drop_duplicates(
                            ).reset_index(drop=True)

    # Main disaggregation loop

    unchanging_kwargs = {'unq_splits':unq_splits, # unchanging
                         'it_set':it_set, # unchanging
                         'base_mat_seg':base_mat_seg, # unchanging
                         'import_folder':import_folder, # unchanging
                         'add_cols':add_cols, # unchanging
                         'target_tld_folder':target_tld_folder, # unchanging
                         'base_p':base_p, # unchanging
                         'p_cols':p_cols, # unchanging
                         'base_a':base_a, # unchanging
                         'a_cols':a_cols, # unchanging
                         'unq_zone_list':unq_zone_list, # unchanging
                         'lookup_folder':lookup_folder, # unchanging
                         'tp':tp, # unchanging
                         'iz_infill':iz_infill, # unchanging
                         'furness_loops':furness_loops, # unchanging
                         'min_pa_diff':min_pa_diff, # unchanging
                         'bs_con_crit':bs_con_crit,
                         'export_original':export_original,
                         'export_furness':export_furness,
                         'ia_name':ia_name,
                         'export_folder':export_folder,
                         'trip_origin':trip_origin} # unchanging

    out_dat = []
    for i in unq_splits.index:
        # Build a list of kwargs for each function call
        kwargs_list = list()
        for i in unq_splits.index:
            kwargs = unchanging_kwargs.copy()
            kwargs['agg_split_index'] = i
            kwargs_list.append(kwargs)

        # Call using multiple threads
        mp.multiprocess(
            _segment_build_worker,
            kwargs=kwargs_list,
            process_count=mp_threads)

    return out_dat

def _segment_build_worker(agg_split_index,
                          unq_splits,
                          it_set,
                          base_mat_seg,
                          import_folder,
                          add_cols,
                          target_tld_folder,
                          base_p,
                          p_cols,
                          base_a,
                          a_cols,
                          unq_zone_list,
                          lookup_folder,
                          tp,
                          iz_infill,
                          furness_loops,
                          min_pa_diff,
                          bs_con_crit,
                          export_original,
                          export_furness,
                          ia_name,
                          export_folder,
                          trip_origin
                          ):
    
    """
    Worker for running segment disaggregator in parallel.
    """
        
    ie_params = it_set.copy()
    import_params = base_mat_seg.copy()

    # Get required distributions for import
    agg_split = unq_splits.loc[agg_split_index]

    for name, desc in agg_split.iteritems():
        ie_params = ie_params[ie_params[name]==desc]
        import_params = import_params[import_params[name]==desc]
    ie_params = ie_params.reset_index(drop=True)
    import_params = import_params.reset_index(drop=True)

    # Ah calib params
    # Define from the agg split
    calib_params = agg_split.to_dict()
    # Handle data type
    for item, dat in calib_params.items():
        if dat.isnumeric():
            calib_params.update({item:int(dat)})

    # Mat list
    mat_list = []

    # Import and aggregate
    for ipi, ipr in import_params.iterrows():

        # Name params calib_params style
        base_params = ipr.to_dict()
        # Handle data type
        for item, dat in base_params.items():
            if dat.isnumeric():
                base_params.update({item:int(dat)})

        print('Importing ' + ipr['base_seg'])
        sd = pd.read_csv(os.path.join(import_folder, ipr['base_seg']))

        if 'zone' in list(sd)[0] or 'Unnamed' in list(sd)[0]:
            sd = sd.drop(list(sd)[0], axis=1)
        sd = sd.values
        mat_list.append(sd)

        # Compile matrices in segment into single matrix
        sd = sum(mat_list) # I can't believe this works

        # Get required tld for splitting
        tld_mat = ie_params.copy().drop(
                'base_seg',
                axis=1).drop_duplicates(
                        ).reset_index(drop=True)

    tld_list = []
    for itl, itd in tld_mat.iterrows():
        tld_dict = {}
        for col in add_cols:
            if itd[col] != 'none':
                tld_dict.update({col:itd[col]})
        tld_dict.update({'enh_tld':pd.read_csv(
        os.path.join(target_tld_folder,
                     itd['enh_tld']))})
        tld_list.append(tld_dict)

    # Get best possible production subset - control to matrix total
    prod_list = _build_enhanced_pa(tld_mat,
                                   base_p,
                                   'trips',
                                   p_cols,
                                   unq_zone_list=unq_zone_list)

    # Get best possible attraction subset
    attr_list = _build_enhanced_pa(tld_mat,
                                   base_a,
                                   'attractions',
                                   a_cols,
                                   unq_zone_list=unq_zone_list)

    # control to sum of target share of attraction vector, cell-wise.
    attr_list, control_aud = _control_a_to_seed_matrix(sd,
                                                       attr_list)
    # Control a to p, exactly this time
    attr_list, bal_aud = _control_a_to_enhanced_p(prod_list,
                                                      attr_list)
    
    # Check audit vectors
    if sum(control_aud) != len(control_aud):
        # TODO: Error format
        raise Warning('PA Vectors not balanced')
    if sum(bal_aud) != len(bal_aud):
        # TODO: Error format
        raise Warning('PA Vectors not balanced')

    # Get distance/cost
    # Costs should be same for each segment, so get here
    costs, c_name = nup.get_costs(lookup_folder,
                                  calib_params,
                                  tp=tp,
                                  iz_infill = iz_infill)

    print('Cost lookup returned ' + c_name)
    costs = nup.df_to_np(costs,
                         v_heading = 'p_zone',
                         h_heading = 'a_zone',
                         values = 'cost',
                         unq_internal_zones=unq_zone_list)

    # Pass to dissagg function
    out_mats, out_reps, fd = _dissag_seg(prod_list,
                                         attr_list,
                                         tld_list,
                                         sd,
                                         costs,
                                         furness_loops = furness_loops,
                                         min_pa_diff = min_pa_diff,
                                         bs_con_crit = bs_con_crit)

    # Unpack list of lists
    # Read, convert, build path and write out
    for oml in out_mats:
        item = oml.copy() 

        furness_mat = item.pop('furness_mat')
        mat = item.pop('mat')

        if export_original:
            mat = pd.DataFrame(mat,
                               index=unq_zone_list,
                               columns=unq_zone_list).reset_index()

            mat = mat.rename(columns={'index':ia_name})

            # Path export
            es_path = os.path.join(export_folder,
                                   trip_origin + '_enhpa')

            if tp in add_cols:
                es_path = nup.build_path(es_path,
                                         item,
                                         tp=tp)
            else:
                es_path = nup.build_path(es_path,
                                         item)

            mat.to_csv(es_path, index=False)

        if export_furness:
            furness_mat = pd.DataFrame(furness_mat,
                                       index=unq_zone_list,
                                       columns=unq_zone_list).reset_index()

            furness_mat = furness_mat.rename(columns={'index':ia_name})

            # Path export
            es_path = os.path.join(export_folder,
                                   trip_origin + '_enhpafn')

            if tp in add_cols:
                es_path = nup.build_path(es_path,
                                         item,
                                         tp=tp)
            else:
                es_path = nup.build_path(es_path,
                                         item)

            furness_mat.to_csv(es_path, index=False)

    # Unpack list of lists
    for orl in out_reps:
        item = orl.copy()

        tlb_con = item.pop('tlb_con')[1]
        ftlb_con = item.pop('final_tlb_con')[1]

        bs_con = item.pop('bs_con')
        fbs_con = item.pop('final_bs_con')
        
        pa_diff = item.pop('pa_diff')

        ea = item.pop('estimated_a')
        ep = item.pop('estimated_p')

        tar_a = item.pop('target_a')
        tar_p = item.pop('target_p')

        # TODO: package these up
        del(tlb_con, bs_con, fbs_con,
            pa_diff, ea, ep, tar_a, tar_p)

        # Build report
        report_path = os.path.join(export_folder,
                                   trip_origin + '_disagg_report')

        if tp in add_cols:
            report_path = nup.build_path(report_path,
                                         item,
                                         tp=tp)
        else:
            report_path = nup.build_path(report_path,
                                         item)
        
        # Write final tlb_con
        ftlb_con.to_csv(report_path,index=False)

    return out_mats, out_reps, fd 

def _dissag_seg(prod_list,
                attr_list,
                tld_list,
                sd,
                costs,
                furness_loops = 1500,
                min_pa_diff = .1,
                bs_con_crit = .975):
    """
    prod_list:
        List of production vector dictionaries
    attr_list:
        List of attraction vector dictionaries
    tld_list:
        List of tld dictionaries
    sd:
        Base matrix
    """

    # build prod cube
    # build attr cube
    # build tld cube

    out_mats = []

    seg_cube = np.ndarray((len(sd),len(sd),len(prod_list)))
    factor_cube = np.ndarray((len(sd),len(sd),len(prod_list)))
    out_cube = np.ndarray((len(sd),len(sd),len(prod_list)))

    # seg_x = 0,prod_list[0]
    seg_audit = []

    for seg_x in enumerate(prod_list,0):

        # Build audit dict
        audit_dict = {}
        for name,dat in prod_list[seg_x[0]].items():
            if name != 'trips':
                audit_dict.update({name:dat})

        new_mat = sd/sd.sum()
        new_mat = new_mat*seg_x[1]['trips'].sum()

        # Unpack target p/a vectors
        target_p =  prod_list[seg_x[0]]['trips']
        target_a = attr_list[seg_x[0]]['trips']
        target_tld = tld_list[seg_x[0]]['enh_tld']

        # Add to audit_dict
        audit_dict.update({'target_p':target_p.sum(),
                           'target_a':target_a.sum()})

        tlb_con = ra.get_trip_length_by_band(target_tld,
                                             costs,
                                             new_mat)

        bs_con = max(1-np.sum(
                (tlb_con[1]['bs']-tlb_con[1]['tbs'])**2)/np.sum(
                        (tlb_con[1]['tbs']-np.sum(
                                tlb_con[1]['tbs'])/len(
                                        tlb_con[1]['tbs']))**2),0)

        # Unpack tld
        min_dist, max_dist, obs_trip, obs_dist = nup.unpack_tlb(target_tld)
        # get num band
        num_band = len(min_dist)

        # TLB balance/furness
        tlb_loop = 1
        while bs_con < bs_con_crit:

            # Seed con crit
            est_trip, est_dist, cij_freq = [0]*num_band, [0]*num_band, [0]*num_band

            for row in range(num_band):
                est_trip[row] = np.sum(np.where(
                        (costs>=min_dist[row]) & (
                                costs<max_dist[row]),new_mat,0))
                est_dist[row] = np.sum(np.where(
                        (costs>=min_dist[row]) & (
                                costs<max_dist[row]), costs*new_mat,0))
                est_dist[row] = np.where(
                        est_trip[row]>0,est_dist[row]/est_trip[row],(
                                min_dist[row]+max_dist[row])/2)
                obs_dist[row] = np.where(
                        obs_dist[row]>0,obs_dist[row],est_dist[row])
                est_trip[row] = est_trip[row]/np.sum(new_mat)*100
                cij_freq[row] = np.sum(np.where(
                        (costs>=min_dist[row]) & (
                                costs<max_dist[row]),len(costs),0))
                cij_freq[row] = cij_freq[row]/np.sum(len(costs))*100

            # Get k factors
            k_factors = k_factors = costs**0
            # k_factors = k_factors**0
            kfc_dist = [0]*num_band

            for row in range(num_band):
                kfc_dist[row] = np.where(
                        est_trip[row]>0,min(
                                max(obs_trip[row]/est_trip[row],.001),10),1)
                k_factors = np.where(
                        (costs>=min_dist[row]) & (
                                costs<max_dist[row]),kfc_dist[row],k_factors)

            # Run furness
            new_mat = k_factors*new_mat

            # Furness
            print('Furnessing')
            for fur_loop in range(furness_loops):

                fur_loop += 1

                mat_d = np.sum(new_mat,axis=0)
                mat_d[target_a==0]=1
                new_mat = new_mat*target_a/mat_d
                mat_o = np.sum(new_mat,axis=1)
                mat_o[mat_o==0]=1
                new_mat = (new_mat.T*target_p/mat_o).T

                # Get pa diff
                mat_o = np.sum(new_mat,axis=1)
                mat_d = np.sum(new_mat,axis=0)
                pa_diff = nup.get_pa_diff(mat_o,
                                          target_p,
                                          mat_d,
                                          target_a) #.max()

                if pa_diff < min_pa_diff or np.isnan(np.sum(new_mat)):
                    print(str(fur_loop) + ' loops')
                    break

            prior_bs_con = bs_con

            tlb_con = ra.get_trip_length_by_band(target_tld,
                                                 costs,
                                                 new_mat)
        
            bs_con = max(1-np.sum(
                    (tlb_con[1]['bs']-tlb_con[1]['tbs'])**2)/np.sum(
                            (tlb_con[1]['tbs']-np.sum(
                                    tlb_con[1]['tbs'])/len(
                                            tlb_con[1]['tbs']))**2),0)

            print('Loop ' + str(tlb_loop))
            print('Band share convergence: ' + str(bs_con))

            # If tiny improvement, exit loop
            if bs_con - prior_bs_con < .001:
                break

            tlb_loop += 1

            # Add to audit dict
            audit_dict.update({'estimated_p':mat_o.sum(),
                               'estimated_a':mat_d.sum(),
                               'pa_diff':pa_diff,
                               'bs_con':bs_con,
                               'tlb_con':tlb_con})

        # Append dict
        seg_audit.append(audit_dict)

        # Push back to the cube
        seg_cube[:,:,seg_x[0]] = new_mat
    
    # ORDER OF PREFERENCE
    # Balance to p/a sd, tld, cells @ sd if possible, P/A slice
        
    """
    Snippet for unpacking a test
    test = []
    for seg_x in enumerate(out_mats,0):
        test.append(seg_x[1]['mat'])
        seg_cube[:,:,seg_x[0]] = seg_x[1]['mat']
        test = sum(test)
   """

    # Get total through calc matrix
    cube_sd = seg_cube.sum(axis=2)

    for seg_x in enumerate(prod_list,0):
        # Get share of cell values from original matrix
        factor_cube[:,:,seg_x[0]] = seg_cube[:,:,seg_x[0]]/np.where(
            cube_sd==0,_TINY_INFILL,cube_sd)
        # Multiply original matrix by share to get cell balanced out matrix
        out_cube[:,:,seg_x[0]] = sd * factor_cube[:,:,seg_x[0]]
        print(out_cube[:,:,seg_x[0]].sum())

        # Get trip length by band
        tlb_con = ra.get_trip_length_by_band(tld_list[seg_x[0]]['enh_tld'],
                                             costs,
                                             out_cube[:,:,seg_x[0]])
        seg_audit[seg_x[0]].update({'final_tlb_con':tlb_con})

        bs_con = max(1-np.sum(
                (tlb_con[1]['bs']-tlb_con[1]['tbs'])**2)/np.sum(
                        (tlb_con[1]['tbs']-np.sum(
                                tlb_con[1]['tbs'])/len(
                                        tlb_con[1]['tbs']))**2),0)
        # Get final bs con r2
        seg_audit[seg_x[0]].update({'final_bs_con':bs_con})

        out_dict = {'furness_mat':seg_cube[:,:,seg_x[0]],
                    'mat':out_cube[:,:,seg_x[0]]}

        # Add lavels back on to export list
        for item,dat in prod_list[seg_x[0]].items():
            if item != 'trips':
                out_dict.update({item:dat})
        out_mats.append(out_dict)

    out_sd = out_cube.sum(axis=2)

    final_pa_diff = nup.get_pa_diff(out_sd.sum(axis=1),
                                    sd.sum(axis=1),
                                    out_sd.sum(axis=0),
                                    sd.sum(axis=0))

    return(out_mats, seg_audit, final_pa_diff)