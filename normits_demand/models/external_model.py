# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 13:45:03 2020

@author: cruella
"""
import os
import warnings

import pandas as pd
import numpy as np

from normits_demand.reports import reports_audits as ra
from normits_demand.utils import utils as nup

# TODO: Object layer

 # Folder management, reindexing, optimisation

_default_pld_path = ('Y:/NorMITs Synthesiser/import/' +
                     'pld_od_matrices/external_pa_pld.csv')

class ExternalModel:
    
    def __init__(self):
        print('inniting')
    
    def path_config(file_drive,
                    model_name ,
                    iteration,
                    trip_origin = 'hb'):
        """
        """
    
        # Set base dir
        home_path = (file_drive +
                    'NorMITs Synthesiser/')
    
        # Set synth import folder
        import_path = (home_path +
                       'import/')
    
        # Set top level model folder, leave the slash on
        model_path = (home_path +
                      model_name +
                      '/' +
                      iteration + '/')
    
        # Set model lookups location
        model_lookup_path = (home_path +
                             model_name +
                             '/Model Zone Lookups')
    
        # Set production path
        production_path = (model_path +
                           'Production Outputs/')
    
        # Set attraction path
        attraction_path = (model_path +
                         'Attraction Outputs/')
    
        # Production import path
        if trip_origin =='hb':
            p_import_path = (production_path +
                             'hb_productions_' +
                             model_name.lower() +
                             '.csv')
            a_import_path = (attraction_path +
                             model_name +
                             '_hb_attractions.csv')
        elif trip_origin == 'nhb':
            p_import_path = (production_path +
                             'nhb_productions_' +
                             model_name.lower() +
                             '.csv')
            a_import_path = (attraction_path +
                             model_name +
                             '_nhb_attractions.csv')
    
        # Raise user warning if no productions by this name
        if not os.path.exists(p_import_path):
            warnings.warn('No productions in folder.' +
                          'Check path or run production model')
    
    
        # Raise user warning if no productions by this name
        if not os.path.exists(a_import_path):
            warnings.warn('No attractions in folder.' +
                          'Check path or run attraction model')
    
        # Create project folders
        distribution_path = (model_path + 'Distribution Outputs')
        nup.create_folder(distribution_path, chDir=False)
    
        # Set distribution outputs (synthetic)
        reports = (distribution_path + '/Logs & Reports')
        nup.create_folder(reports)
    
        external_export = (distribution_path + '/External Distributions')
        nup.create_folder(external_export)
        
        full_export = (distribution_path + '/24hr Non Dist Matrices')
        nup.create_folder(full_export)
    
        # Compile into import and export
        imports = {'imports':import_path,
                   'lookups':model_lookup_path,
                   'production_import':p_import_path,
                   'attraction_import':a_import_path}
    
        exports = {'production_export':production_path,
                   'attraction_export':attraction_path,
                   'external':external_export,
                   'reports':reports,
                   'full_export':full_export}
        return(imports, exports)
    
    def run(file_drive,
                           model_name,
                           iteration,
                           tlb_area = 'gb',
                           segmentation = 'tfn',
                           distribution_segments = ['p', 'm'],
                           trip_origin = 'hb',
                           cost_type = '24hr',
                           seed_infill = .1,
                           export_ext_modes = [1,2,3,4,5,6],
                           export_non_dist_modes = []):
    
        """
        """
        # TODO: PLD infill
    
        # Config paths
        paths = path_config(file_drive,
                            model_name,
                            iteration,
                            trip_origin)
        i_paths = paths[0]
        o_paths = paths[1]
        del(paths)
    
        ia_name = (model_name.lower() + '_zone_id')
    
        # Do imports
    
        # Import PA
        pa = nup.import_pa(i_paths['production_import'], # p import path
                           i_paths['attraction_import'])
        productions = pa[0]
        attractions = pa[1]
        del(pa)
    
        # Get unique zones from productions
        unq_zones = nup.get_zone_range(productions[ia_name])
    
        # Get internal area
        int_area = nup.get_internal_area(i_paths['lookups'])
        ext_area = nup.get_external_area(i_paths['lookups'])
    
        init_params = nup.get_init_params(i_paths['lookups'],
                                          distribution_type = trip_origin,
                                          model_name = model_name,
                                          mode_subset = None,
                                          purpose_subset = None)
    
        # Path tlb folder
        tlb_folder = (i_paths['imports'] +
                      'trip_length_bands/' +
                      tlb_area + '/standard_segments')
    
        # Drop any sic or soc segments from init params - not needed for externals
        init_params = init_params.reindex(distribution_segments,
                                          axis=1).drop_duplicates(
                                                  ).reset_index(drop=True)
        
        # Define mode subset
        unq_mode = export_ext_modes.copy()
        [unq_mode.append(x) for x in export_non_dist_modes if x not in unq_mode] 
        init_params = init_params[
                init_params['m'].isin(
                        export_non_dist_modes)].reset_index(drop=True)
    
        # External index
        ei = init_params.index
    
        # Import cjtw
        print('Importing cjtw')
        cjtw = nup.get_cjtw(i_paths['lookups'],
                            model_name.lower(),
                            subset=None,
                            reduce_to_pa_factors = False)
        # Aggregate mode
        p_col = list(cjtw)[0]
        a_col = list(cjtw)[1]
    
        cjtw = cjtw.reindex([p_col,
                             a_col,
                             'trips'],
        axis=1).groupby([p_col, a_col]).sum().reset_index()
    
        # Sum over mode - not seperating so we can aggregate by mode
    
        # Handle cols - transpose
        cjtw = nup.df_to_np(cjtw,
                            v_heading = p_col,
                            h_heading = a_col,
                            values = 'trips',
                            unq_internal_zones=unq_zones)
    
        # Small infill
        cjtw = np.where(cjtw == 0,seed_infill,cjtw)
    
        # external distribution in external index
        internal_bin = []
        full_bin = []
    
        for ed in ei:
            calib_params = {}
            for ds in distribution_segments:
                calib_params.update({ds:init_params[ds][ed]})
    
            # Get target tlb
            tlb = nup.get_trip_length_bands(tlb_folder,
                                            calib_params,
                                            segmentation = 'ntem', # normal segments - due a rewrite
                                            trip_origin = trip_origin)
            calib_params.update({'tlb':tlb})
    
            # Filter productions to target distribution type
            sub_p = nup.filter_pa_vector(productions,
                                         ia_name,
                                         calib_params,
                                         round_val = 3,
                                         value_var = 'trips',
                                         echo=False)
    
            # Get the productions from the tuple
            sub_p = sub_p[0]
            sub_p = sub_p.rename(columns={'trips':'productions'})
    
            # Work out which attractions to use from purpose
            #
            sub_a = nup.filter_pa_vector(attractions,
                                         ia_name,
                                         calib_params,
                                         round_val = 3,
                                         value_var = 'attractions',
                                         echo=False)
            # Get the Attractions from the tuple
            sub_a = sub_a[0]
    
            a_t = nup.get_attraction_type(calib_params)
            print(a_t)
    
            # Balance a to p
            print(sub_p['productions'].sum())
            sub_a = nup.balance_a_to_p(ia_name,
                                       sub_p,
                                       sub_a,
                                       round_val = 3,
                                       echo=False)
    
            # TODO: Pass car ownership params for costs
    
            # Import costs based on distribution parameters & car availability
            print('Importing costs')
            internal_costs = nup.get_costs(i_paths['lookups'],
                                           calib_params,
                                           tp = cost_type,
                                           iz_infill = 0.5)
    
            print('Cost lookup returned ' + internal_costs[1])
            internal_costs = internal_costs[0].copy()
    
            # TODO: Replace with the newer way of doing this
            unq_internal_zones = internal_costs[
                'p_zone'].drop_duplicates().reset_index(drop=True).copy()
            
            # Join ps and a's onto full unq internal vector (infill placeholders)
            uiz_vector = pd.DataFrame(unq_internal_zones)
            uiz_vector = uiz_vector.rename(columns={
                    'p_zone':(model_name.lower() + '_zone_id')})
            
            sub_p = uiz_vector.merge(sub_p,
                                     how='left',
                                     on = (model_name.lower() + '_zone_id'))
            sub_p['productions'] = sub_p['productions'].replace(np.nan,0)
    
            sub_a = uiz_vector.merge(sub_a,
                                     how='left',
                                     on = (model_name.lower() + '_zone_id'))
            sub_a['attractions'] = sub_a['attractions'].replace(np.nan,0)
    
            # Translate costs to array
            costs = nup.df_to_np(internal_costs,
                                 v_heading = 'p_zone',
                                 h_heading = 'a_zone',
                                 values = 'cost',
                                 unq_internal_zones=unq_zones)
            
            # Get area of zone, if zone was a circle
            # zone_blob_area = np.pi*np.diag(costs)**2
     
            # Sort iz to be more friendly
            costs = iz_costs_to_mean(costs)
    
            # Hive off external and i_i trip ends
            internal_index = int_area[ia_name].values - 1
            external_index = ext_area[ia_name].values -1 
            
            # TODO: Cost direction audit
            ie_cost = nup.n_matrix_split(costs,
                                         indices = [internal_index,
                                                    external_index],
                                                    index_names = ['i', 'e'],
                                                    summarise = False)
            ie_cost[0]['dat'].mean()
            ie_cost[1]['dat'].mean()
            ie_cost[2]['dat'].mean()
            ie_cost[3]['dat'].mean()
    
            # TODO Currently does not balance at row level
            external_out = external_model(sub_p,
                                          sub_a,
                                          cjtw,
                                          costs,
                                          calib_params,
                                          i_paths['lookups'],
                                          model_name.lower(),
                                          trip_origin)
    
            external_pa, external_pa_p, external_pa_a, tl_con, bs_con, max_diff = external_out
    
            report = tl_con[1]
            report['bs_con'] = bs_con
            report['max_diff'] = max_diff
    
            # perc_iz = np.diag(external_pa)/external_pa.sum(axis=1)
    
            print(tl_con)
            del(external_out)
    
            # Check row totals
    
            # External pa backup
            full_epa = external_pa.copy()
    
            # Sidepot full pa
            full_pa_p = external_pa_p.copy()
            full_pa_a = external_pa_a.copy()
    
            # Do int-ext report
            ie_report = nup.n_matrix_split(external_pa_p,
                                           indices = [internal_index,
                                                      external_index],
                                                      index_names = ['i', 'e'],
                                                      summarise = True)
    
            ie_report = pd.DataFrame(ie_report)
    
            # Sub off internal only
            # Now for pure internal with better balance and p balanced for external
            internal_pa = np.take(external_pa, internal_index, axis=1)
            internal_pa = internal_pa.take(internal_index, axis=0)
    
            internal_pa_p = np.take(external_pa_p, internal_index, axis=1)
            internal_pa_p = internal_pa_p.take(internal_index, axis=0)
    
            i_ph = np.zeros([len(external_pa_p), len(external_pa_p)])
            i_ph[0:len(internal_pa_p),0:len(internal_pa_p)] = internal_pa_p
    
            external_pa_p = external_pa_p - i_ph
            del(i_ph)
    
            internal_bin.append({'cp':calib_params,
                                'int_pa':internal_pa_p})
            full_bin.append({'cp':calib_params,
                             'full_pa_p':full_pa_p,
                             'full_pa_a':full_pa_a})
    
            audit_path = os.path.join(o_paths['reports'],
                                      trip_origin + '_external_report')
            audit_path = nup.build_path(audit_path,
                                        calib_params)
    
            report.to_csv(audit_path, index=False)
    
            ie_path = os.path.join(o_paths['reports'],
                                   trip_origin + '_ie_report')
            ie_path = nup.build_path(ie_path,
                                     calib_params)
            
            ie_report.to_csv(ie_path, index=False)
    
            # Export 
            if (calib_params['m'] in export_ext_modes or
                str(calib_params['m']) in export_ext_modes):
                # Append zone names
                external_pa_p = pd.DataFrame(external_pa_p,
                                           index=unq_zones,
                                           columns=unq_zones).reset_index()
                external_pa_p = external_pa_p.rename(columns={'index':ia_name})
    
                # Path export
                ext_path = os.path.join(o_paths['external'],
                                        trip_origin + '_external')
    
                ext_path = nup.build_path(ext_path,
                                          calib_params)
            
                external_pa_p.to_csv(ext_path, index=False)
            
            # Export full external matrices, for best guess non dist demand mats
            if (calib_params['m'] in export_non_dist_modes or
                str(calib_params['m']) in export_non_dist_modes):
    
                full_epa = pd.DataFrame(full_epa,
                                        index=unq_zones,
                                        columns=unq_zones).reset_index()
                full_epa = full_epa.rename(columns={'index':ia_name})
    
                # Path export
                ext_path = os.path.join(o_paths['full_export'],
                                        trip_origin + '_nongrav')
    
                ext_path = nup.build_path(ext_path,
                                          calib_params)
    
                full_epa.to_csv(ext_path, index=False)
    
                del(full_epa)
    
        # Unpack and compile internal productions and attractions
    
        # Build uip
        p_ph = int_area.copy().rename(columns={ia_name:'p_zone'})
        a_ph = int_area.copy().rename(columns={ia_name:'a_zone'})
    
        for int_r in internal_bin:
            
            cp = int_r['cp'].copy()
    
            # Get name in string
            int_path = ''
            int_path = nup.build_path(int_path,
                                      int_r['cp'],
                                      no_csv = True)
            print(int_path)
    
            if (cp['m'] in export_ext_modes or
                str(cp['m'] in export_ext_modes)):
                # Format and prepare for export
                int_pa = pd.DataFrame(int_r['int_pa'],
                                        index=int_area[ia_name],
                                        columns=int_area[ia_name]).reset_index()
                int_pa = pd.melt(int_pa,
                                 id_vars=[ia_name],
                                 var_name='a_zone',
                                 value_name='dt',
                                 col_level=0)
                int_pa = int_pa.rename(columns={ia_name:'p_zone'})
    
                int_p = int_pa.reindex(['p_zone','dt'],
                                       axis=1).groupby(
                                               ['p_zone']).sum().reset_index()
                p_ph = p_ph.merge(int_p,
                                  how='left',
                                  on = 'p_zone')
                p_ph = p_ph.rename(columns={'dt':int_path.replace('_','')})
    
                # Same for attractions            
                print(list(int_pa))
    
                int_a = int_pa.reindex(['a_zone','dt'],
                                       axis=1).groupby(
                                               ['a_zone']).sum().reset_index()
                a_ph = a_ph.merge(int_a,
                                  how='left',
                                  on = 'a_zone')
                a_ph = a_ph.rename(columns={'dt':int_path.replace('_','')})
    
        # Export
        p_ph_path = os.path.join(o_paths['production_export'],
                                 model_name.lower() +
                                 '_' +
                                 trip_origin +
                                 '_internal_productions.csv')
        a_ph_path = os.path.join(o_paths['attraction_export'],
                                 model_name.lower() +
                                 '_' +
                                 trip_origin +
                                 '_internal_attractions.csv')
    
        p_ph.to_csv(p_ph_path, index=False)
        a_ph.to_csv(a_ph_path, index=False)
    
        audit = True
    
        # Unpack and compile full productions and attractions
    
        p_ph = pd.DataFrame(unq_zones)
        a_ph = pd.DataFrame(unq_zones)
        p_ph = p_ph.rename(columns={0:'p_zone'})
        a_ph = a_ph.rename(columns={0:'a_zone'})
    
        for full_r in full_bin:
            
            cp = full_r['cp'].copy()
    
            # Get name in string
            full_path = ''
            full_path = nup.build_path(full_path,
                                       full_r['cp'],
                                       no_csv = True)
            print(full_path)
    
            # Format and prepare for export
            full_pa_p = pd.DataFrame(full_r['full_pa_p'],
                                     index=unq_zones,
                                     columns=unq_zones).reset_index()
            full_pa_p = full_pa_p.rename(columns={'index':ia_name})
            full_pa_p = pd.melt(full_pa_p,
                                id_vars=[ia_name],
                                var_name='a_zone',
                                value_name='dt',
                                col_level=0)
            full_pa_p = full_pa_p.rename(columns={ia_name:'p_zone'})
    
            full_p = full_pa_p.reindex(['p_zone','dt'],
                                       axis=1).groupby(
                                               ['p_zone']).sum().reset_index()
    
            full_p = full_p.rename(columns={'dt':full_path.replace('_','')})
            
            # Same for attractions
            full_pa_a = pd.DataFrame(full_r['full_pa_a'],
                                     index=unq_zones,
                                     columns=unq_zones).reset_index()
            full_pa_a = full_pa_a.rename(columns={'index':ia_name})
            full_pa_a = pd.melt(full_pa_a,
                                id_vars=[ia_name],
                                var_name='a_zone',
                                value_name='dt',
                                col_level=0)
            full_pa_a = full_pa_a.rename(columns={ia_name:'p_zone'})
    
            full_a = full_pa_a.reindex(['a_zone','dt'],
                                       axis=1).groupby(
                                               ['a_zone']).sum().reset_index()
            full_a = full_a.rename(columns={'dt':full_path.replace('_','')})
    
            p_ph = p_ph.merge(full_p,
                              how='left',
                              on=['p_zone'])
            a_ph = a_ph.merge(full_a,
                              how='left',
                              on=['a_zone'])
    
        # Export
        full_p_ph_path = os.path.join(o_paths['production_export'],
                                      model_name.lower() +
                                      '_' +
                                      trip_origin +
                                      '_full_productions.csv')
        full_a_ph_path = os.path.join(o_paths['attraction_export'],
                                      model_name.lower() +
                                      '_' +
                                      trip_origin +
                                      '_full_attractions.csv')
    
        p_ph.to_csv(full_p_ph_path, index=False)
        a_ph.to_csv(full_a_ph_path, index=False)
    
        audit = True
    
        return(audit)
    
    def external_model(p,
                       a,
                       base_matrix,
                       costs,
                       calib_params,
                       model_lookup_path,
                       model_name,
                       trip_origin):
        
        """
        p:
            Production vector
        a:
            Attraction vector
        base_matrix:
            Seed matrix for transformation. Will usually be cjtw for shape.
            Could be 1s.
        costs:
            Cost/distance matrix for transformation
        calib_params:
            Distribution calib params. For write out desc only.
        model_lookup_path:
            Where are the lookups, for finding cjtw.
        model_name:
            Model name. Needs to be lower case I think.
        trip_origin:
            Is this HB or NHB? Takes 'hb' or 'nhb', shockingly.
        """
    
        # Transform original p/a into vectors
        target_p = p['productions'].values
        target_a = a['attractions'].values
        # Infill zeroes
        target_p = np.where(target_p==0, 0.0001, target_p)
        target_a = np.where(target_a==0, 0.0001, target_a)
    
        # Seed base
        external_pa = base_matrix.copy()
    
        # Seed con crit
        bs_con = 0
        print('Initial band share convergence: ' + str(bs_con))
    
        # Calibrate
        bls = 1
        while bs_con <.9:
            print('Band share loop: ' + str(bls))
    
            # Get mat by band
            band_totals = ra.get_matrix_by_band(calib_params['tlb'],
                                                costs,
                                                external_pa)
            # Correct a band share
            external_pa = nup.correct_band_share(external_pa,
                                                 calib_params['tlb'],
                                                 band_totals,
                                                 axis=0)
            # Get mat by band
            band_totals = ra.get_matrix_by_band(calib_params['tlb'],
                                                costs,
                                                external_pa)
            # Correct p band share
            external_pa = nup.correct_band_share(external_pa,
                                                 calib_params['tlb'],
                                                 band_totals,
                                                 axis=1)
    
            new_p = external_pa.sum(axis=1)
            new_a = external_pa.sum(axis=0)
    
            # Check validation
            pa_diff = nup.get_pa_diff(new_p,
                                      target_p,
                                      new_a,
                                      target_a)
        
            print('Max p/a diff: ' + str(pa_diff.max()))
    
            # Furness to tl con
            fl = 1
            while pa_diff.max() > .1:
                # Balance a
                external_pa = nup.balance_columns(external_pa,
                                                  target_a,
                                                  infill = 0.001)
    
                # Balance p
                external_pa = nup.balance_rows(external_pa,
                                               target_p,
                                               infill = 0.001)
    
                # Round
                external_pa = external_pa.round(5)
    
                new_p = external_pa.sum(axis=1)
                new_a = external_pa.sum(axis=0)
    
                # Check validation
                prev_pa_diff = pa_diff.copy()
                pa_diff = nup.get_pa_diff(new_p,
                                          target_p,
                                          new_a,
                                          target_a)
    
                f_change = prev_pa_diff.mean() - pa_diff.mean()
    
                print('Loop: ' + str(fl) + ' Max diff: ' + str(pa_diff.max()))
                if f_change < .0001:
                    # If not converging hard balance
                    external_pa = nup.single_balance(external_pa,
                                                     target_a,
                                                     target_p)
                    new_p = external_pa.sum(axis=1)
                    new_a = external_pa.sum(axis=0)
    
                    # Check validation
                    prev_pa_diff = pa_diff.copy()
                    pa_diff = nup.get_pa_diff(new_p,
                                          target_p,
                                          new_a,
                                          target_a)
                    break
    
                # iterate
                fl +=1
    
     
            print('Max p/a diff: ' + str(pa_diff.max()))
    
            # Get convergence
            tlb_con = ra.get_trip_length_by_band(calib_params['tlb'],
                                                 costs,
                                                 external_pa)
            
            prior_bs_con = bs_con
            bs_con = max(1-np.sum(
                (tlb_con[1]['bs']-tlb_con[1]['tbs'])**2)/np.sum(
                        (tlb_con[1]['tbs']-np.sum(
                                tlb_con[1]['tbs'])/len(
                                        tlb_con[1]['tbs']))**2),0)
    
            diff = prior_bs_con - bs_con
    
            print(tlb_con[1])
    
            print('Band share convergence: ' + str(bs_con))
            print('Improvement: ' + str(diff))
    
            if diff < .0001:
                break
    
            bls += 1
    
        # Gives great convergence on PA here - but we want the bands more
        # Do another hard balance, control to P - then stop
    
        # Rebalance tlb
        # Get mat by band
        band_totals = ra.get_matrix_by_band(calib_params['tlb'],
                                            costs,
                                            external_pa)
        # Correct a band share
        external_pa_balanced = nup.correct_band_share(external_pa,
                                                      calib_params['tlb'],
                                                      band_totals,
                                                      axis=0)
        # Get mat by band
        band_totals = ra.get_matrix_by_band(calib_params['tlb'],
                                            costs,
                                            external_pa_balanced)
        # Correct p band share
        external_pa_balanced = nup.correct_band_share(external_pa_balanced,
                                                      calib_params['tlb'],
                                                      band_totals,
                                                      axis=1)
    
        new_p = external_pa_balanced.sum(axis=1)
        new_a = external_pa_balanced.sum(axis=0)
    
        # Single balance (productions)
        p_balanced = nup.balance_rows(external_pa_balanced,
                                      target_p,
                                      infill = 0.001)
        
        a_balanced = nup.balance_columns(external_pa_balanced,
                                         target_a,
                                         infill = 0.001)
    
        # get new tlb con
        tlb_con = ra.get_trip_length_by_band(calib_params['tlb'],
                                             costs,
                                             p_balanced)
    
        # Check validation
        pa_diff = nup.get_pa_diff(new_p,
                                  target_p,
                                   new_a,
                                   target_a)
        
        bs_con = max(1-np.sum(
                (tlb_con[1]['bs']-tlb_con[1]['tbs'])**2)/np.sum(
                        (tlb_con[1]['tbs']-np.sum(
                                tlb_con[1]['tbs'])/len(
                                        tlb_con[1]['tbs']))**2),0)
    
        return(external_pa, p_balanced, a_balanced, tlb_con, bs_con, pa_diff.max())
    
    def adjust_trip_length_by_band(band_atl,
                                   adj_fac,
                                   distance,
                                   base_matrix):
        """
        Go over atl, adjust trips as required
        """
    
        # TODO: Drop averages of nothing in trip length band
        # reset index, needed or not
        band_atl = band_atl.reset_index(drop=True)
    
        # Get global atl
        global_atl = ra.get_trip_length(distance,
                                        base_matrix)
    
        # Get min max for each
        ph = band_atl['tlb_desc'].str.split('-', n=1, expand=True)
        band_atl['min'] = ph[0].str.replace('(','')
        band_atl['max'] = ph[1].str.replace('[','')
        band_atl['min'] = band_atl['min'].str.replace('(','').values
        band_atl['max'] = band_atl['max'].str.replace(']','').values
        del(ph)
    
        mat_ph = []
        out_mat = np.empty(shape=[len(base_matrix),len(base_matrix)])
    
        # Loop over rows in band_atl
        for index, row in band_atl.iterrows():
    
            # Get band mask
            band_mat = np.where(
                    (distance>=float(row['min'])) & (
                            distance<float(row['max'])),True,False)
    
            adj_mat = band_mat * base_matrix * adj_fac['adj_fac'][index]
    
            mat_ph.append(adj_mat)
        
        
        for o_mat in mat_ph:
            out_mat = out_mat + o_mat
    
        return(out_mat)
    
    def pld_infill(external_pa,
                   pld_path = _default_pld_path,
                   infill_mode = 6):
        """
        external_pa:
            Full set of external pa before infill
    
        pld_path:
            Path to pld_data. Auto fills from script. Just keep an eye on it.
    
        infill_mode:
            mode to apply the infill to.
    
        """
        pld_pa = pd.read_csv(pld_path)
        
        # Translate the ca/nca to 2,1
        ca_mat = pd.DataFrame({'ca':['ca','nca'],
                               'ca_code':[2,1]})
        pld_pa = pld_pa.merge(ca_mat,
                              how='left',
                              on='ca')
        del(pld_pa['ca'])
        pld_pa = pld_pa.rename(columns={'ca_code':'ca',
                                        'purpose':'ap'})
        pld_pa = pld_pa.reindex(['p_zone',
                                 'a_zone',
                                 'ap',
                                 'ca',
                                 'dt'], axis=1)
    
        # Get original total
        pld_total = pld_pa['dt'].sum()
    
        # Get external pa
        w_external_pa = external_pa.copy()
        
        # Isolate mode to be infilled
        infill_subset = w_external_pa[
                w_external_pa['m']==infill_mode].copy()
        other_mode = w_external_pa[
                w_external_pa['m']!=infill_mode].copy()
    
        # Get original pa total    
        opa_total = infill_subset['dt'].sum()
    
        # Get shares by zone for infill
        infill_subset = infill_subset.reindex(['p_zone',
                                               'p',
                                               'ca',
                                               'dt'],
        axis=1).groupby(
        ['p_zone',
         'p',
         'ca']).sum().reset_index()
    
        # Build aggregate purpose
        agg_purp = pd.DataFrame({'p':[1,2,3,4,5,6,7,8],
                                 'ap':['commute',
                                       'business',
                                       'other',
                                       'other',
                                       'other',
                                       'other',
                                       'other',
                                       'other']})
    
        infill_subset = infill_subset.merge(agg_purp,
                                            how='left',
                                            on='p')
    
        # Get zone totals by aggregate purpose and ca
        zone_totals = infill_subset.reindex(
                ['p_zone', 'ap', 'ca', 'dt'],
                axis=1).groupby(
                        ['p_zone', 'ap', 'ca']).sum().reset_index()
        zone_totals = zone_totals.rename(columns={'dt':'total'})
        
        infill_subset = infill_subset.merge(zone_totals,
                                            how='left',
                                            on = ['p_zone', 'ap', 'ca'])
        infill_subset['factor'] = infill_subset['dt']/infill_subset['total']
        infill_subset = infill_subset.drop(['dt', 'total'], axis=1)
    
        # Factor test
        factor_test = infill_subset.reindex(
                ['p_zone', 'ap', 'ca', 'factor'],
                axis=1).groupby(['p_zone', 'ap', 'ca']).sum().reset_index()
        
        print('Min factor: ' + str(min(factor_test['factor'])))
        print('Max factor: ' + str(max(factor_test['factor'])))
    
        # Gonna be:
        # Infill left = [p_zone, a_zone, ap, ca, dt]
        # Infill right = [p_zone, ap, ca, p, factor]]
        # On = [p_zone, ap, ca]
        # dt * factor
        infill_output = pld_pa.merge(infill_subset,
                                     how='left',
                                     on = ['p_zone', 'ap', 'ca'])
        infill_output['dt'] = infill_output['dt'] * infill_output['factor']
        infill_output['m'] = infill_mode
        list(infill_output)
    
        # Get original cols
        org_cols = list(other_mode)
        og_cols = org_cols.copy()
        og_cols.remove('dt')
        
        # Reindex for reentry
        infill_output = infill_output.reindex(org_cols,
                                              axis=1).groupby(
                                                      og_cols).sum(
                                                              ).reset_index(
                                                              )
    
        # Correct
        corr_fac = pld_total/infill_output['dt'].sum()
        infill_output['dt'] = (infill_output['dt']*corr_fac).round(3)
        new_total = infill_output['dt'].sum()
    
        # Benchmark
        print('Orignal pld total: ' + str(pld_total))
        print('Original external total: ' + str(opa_total))
        print('New total: ' + str(new_total))
    
        # Reappend
        w_external_pa = pd.concat([infill_output, other_mode], sort=True)
    
        return(w_external_pa)
    
    def iz_costs_to_mean(costs):
        """
        Sort bands that are too big outside of the north
        - nudge towards intrazonal
        """
        # Get mean
        diag_mean = np.mean(np.diag(costs))
        diag = costs.diagonal()
        diag = np.where(diag > diag_mean,diag_mean,diag)
        
        np.fill_diagonal(costs, diag)
        
        return(costs)