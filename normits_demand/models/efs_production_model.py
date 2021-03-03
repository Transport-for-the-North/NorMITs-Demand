# -*- coding: utf-8 -*-
"""
Created on: Mon Dec  9 12:13:07 2019
Updated on:

Original author: Ben Taylor
Last update made by:
Other updates made by:

File purpose:
EFS Production Model
"""
# Builtins
import os
import warnings
import operator
from functools import reduce

from typing import List
from typing import Dict
from typing import Tuple
from typing import Callable

# Third Party
import numpy as np
import pandas as pd
from tqdm import tqdm

# Local Imports
import normits_demand as nd
from normits_demand import efs_constants as consts

from normits_demand.utils import general as du
from normits_demand.utils import timing


from normits_demand.d_log import processor as dlog_p
from normits_demand.concurrency import multiprocessing
from normits_demand.constraints import ntem_control as ntem


# BACKLOG: Integrate EFS Production Model into TMS trip end models
#  labels: EFS, demand merge


class EFSProductionGenerator:
    
    def __init__(self,
                 model_name: str,
                 seg_level: str = 'tfn',
                 zoning_system: str = 'msoa',
                 tag_certainty_bounds=consts.TAG_CERTAINTY_BOUNDS
                 ):
        """
        #TODO
        """
        # Validate inputs
        seg_level = du.validate_seg_level(seg_level)
        model_name = du.validate_model_name(model_name)
        zoning_system = du.validate_zoning_system(zoning_system)

        # Assign
        self.tag_certainty_bounds = tag_certainty_bounds

        self.model_name = model_name

        self.zoning_system = zoning_system
        self.zone_col = '%s_zone_id' % zoning_system

        self.pop_fname = consts.POP_FNAME % zoning_system

        # Define the segmentation we're using
        if seg_level == 'tfn':
            self.segments = ['area_type', 'p', 'soc', 'ns', 'ca']
            self.return_segments = [self.zone_col] + self.segments
            self.return_segments.remove('area_type')
        else:
            raise ValueError(
                "'%s' is a valid segmentation level, but I don't have a way "
                "of determining which segments to use for it. You should add "
                "one!" % seg_level
            )

        # Remove ca segmentation for some models
        if model_name == 'noham':
            self.return_segments.remove('ca')

    def run(self,
            out_path: str,
            base_year: str,
            future_years: List[str],

            # Population data
            population_import_path: nd.PathLike,
            population_constraint: pd.DataFrame,

            # Build I/O paths
            import_home: str,
            export_home: str,

            # Alternate population/production creation files
            trip_rates_path: str = None,
            time_splits_path: str = None,
            mean_time_splits_path: str = None,
            mode_share_path: str = None,
            msoa_lookup_path: str = None,

            # Alternate output paths
            audit_write_dir: str = None,

            # Production control file
            ntem_control_dir: str = None,
            lad_lookup_dir: str = None,
            control_productions: bool = True,
            control_fy_productions: bool = False,

            # D-Log
            dlog: str = None,

            # Population constraints
            pre_dlog_constraint: bool = False,
            post_dlog_constraint: bool = None,
            designated_area: pd.DataFrame = None,

            # Segmentation Controls
            m_needed: List[int] = consts.MODES_NEEDED,
            segmentation_cols: List[str] = None,
            external_zone_col: str = 'model_zone_id',

            # Handle outputs
            audits: bool = True,
            recreate_productions: bool = True,
            ) -> pd.DataFrame:
        """
        Production model for the external forecast system. This has been
        written to align with TMS production generation, with the addition of
        future year population growth and production generation.

        Performs the following functions:
            - Reads in the base year land use data to create the base year
              population numbers
            - Grows the base year population by pop_growth factors,
              resulting in future year population numbers.
            - Combines base and future year population numbers with trip_rates
              (currently the same across all years) to produce the base and
              future year production values (for all modes).
            - Finally, splits the produced productions to only return the
              desired mode. This dataframe is then returned.

        Parameters
        ----------
        out_path:
            Path to the directory to output the population and productions
            dataframes.

        base_year:
            The base year of the forecast.

        future_years:
            The future years to forecast.

        population_import_path:
            Path to the directory containing the NorMITs Land Use outputs
            for future year population estimates. The filenames will
            be automatically generated based on consts.LU_POP_FNAME

        population_constraint:
            TODO: Need to clarify if population constrain is still needed,
             where the values come from, and how exactly the constrainer works.

        import_home:
            The home directory to find all the production imports. Usually
            Y:/NorMITs Demand/import

        export_home:
            Path to the export home of this instance of outputs. This is
            usually related to a specific run of the ExternalForecastSystem,
            and should be gotten from there using generate_output_paths().
            e.g. 'E:/NorMITs Demand/norms_2015/v2_3-EFS_Output/iter1'

        trip_rates_path:
            The path to alternate trip rates data. If left as None, the
            production model will use the default trip rates data.

        time_splits_path:
            The path to alternate time splits data. If left as None, the
            production model will use the default time splits data.

        mean_time_splits_path:
            The path to alternate mean time splits data. If left as None, the
            production model will use the default mean time splits data.

        mode_share_path:
            The path to alternate mode share data. If left as None, the
            production model will use the default mode share data.

        msoa_lookup_path:
            The path to alternate msoa lookup import data. If left as None,
            the production model will use the default msoa lookup path.

        audit_write_dir:
            Alternate path to write the audits. If left as None, the default
            location is used.

        ntem_control_dir:
            The path to alternate ntem control directory. If left as None, the
            production model will use the default path.

        lad_lookup_dir:
            The path to alternate lad to msoa import data. If left as None, the
            production model will use the default path.

        control_productions:
            Whether to control the generated production to the constraints
            given in ntem_control_dir or not.

        control_fy_productions:
            Whether to control the generated future year productions to the
            constraints given in ntem_control_dir or not. When running for
            scenarios other than the base NTEM, this should be False.

        dlog:
            TODO: Clarify what format D_log data comes in as

        pre_dlog_constraint:
            Whether to constrain the population before applying the dlog or
            not.

        post_dlog_constraint
            Whether to constrain the population after applying the dlog or
            not.

        designated_area:
            TODO: Clarify what the designated area actually is

        m_needed:
            Which mode to return productions for.

        segmentation_cols:
            The levels of segmentation that exist in the land use data. If
            not defined, will default to: ['area_type', 'traveller_type',
            'soc', 'ns', 'ca'].

        external_zone_col:
            The name of the zone column, as used externally to this production
            model. This is used to make sure this model can translate to the
            zoning name used internally in land_use and trip_rates data.

        audits:
            Whether to output print_audits to the terminal during running. This can
            be used to monitor the population and production numbers being
            generated and constrained.

        recreate_productions:
            Whether to recreate the productions or not. If False, it will
            look in out_path for previously produced productions and return
            them. If none can be found, they will be generated.

        Returns
        -------
        Segmented_productions:
            Productions for mode m_needed, segmented by all segments possible
            in the input data.
        """
        # Return previously created productions if we can
        fname = consts.PRODS_FNAME % (self.zoning_system, 'hb')
        final_output_path = os.path.join(out_path, fname)

        if not recreate_productions and os.path.isfile(final_output_path):
            print("Found some already produced productions. Using them!")
            return pd.read_csv(final_output_path)

        # Init
        all_years = [str(x) for x in [base_year] + future_years]

        # If not set, perform the post_dlog_constrain if dlog is on
        if post_dlog_constraint is None:
            post_dlog_constraint = dlog is not None

        # TODO: Make this more adaptive
        # Set the level of segmentation being used
        if segmentation_cols is None:
            segmentation_cols = [
                'area_type',
                'traveller_type',
                'soc',
                'ns',
                'ca'
            ]

        # Fix column naming if different
        if external_zone_col != self.zone_col:
            designated_area = designated_area.copy().rename(
                columns={external_zone_col: self.zone_col}
            )
            population_constraint = population_constraint.rename(
                columns={external_zone_col: self.zone_col}
            )

        # Build paths to the needed files
        imports = build_production_imports(
            import_home=import_home,
            trip_rates_path=trip_rates_path,
            time_splits_path=time_splits_path,
            mean_time_splits_path=mean_time_splits_path,
            mode_share_path=mode_share_path,
            msoa_lookup_path=msoa_lookup_path,
            ntem_control_dir=ntem_control_dir,
            lad_lookup_dir=lad_lookup_dir,
            set_controls=control_productions
        )

        exports = build_production_exports(
            export_home=export_home,
            audit_write_dir=audit_write_dir
        )

        # # ## READ IN POPULATION DATA ## #
        population = get_pop_data_from_land_use(
            import_path=population_import_path,
            years=all_years,
            segmentation_cols=segmentation_cols,
        )

        # ## PRE D-LOG CONSTRAINT ## #
        if pre_dlog_constraint:
            print("Performing the first constraint on population...")
            print(". Pre Constraint:\n%s" % population[future_years].sum())
            constraint_segments = du.intersection(segmentation_cols,
                                                  population_constraint)

            population = dlog_p.constrain_forecast(
                population,
                population_constraint,
                designated_area,
                base_year,
                future_years,
                self.zone_col,
                segment_cols=constraint_segments
            )
            print(". Post Constraint:\n%s" % population[future_years].sum())

        # ## INTEGRATE D-LOG ## #
        if dlog is not None:
            print("Integrating the development log...")
            # Remove the columns not used to split the dlog_processor data
            seg_groups = [x for x in segmentation_cols 
                          if x not in ['area_type', "traveller_type"]]

            population, hg_zones = dlog_p.apply_d_log(
                pre_dlog_df=population,
                base_year=base_year,
                future_years=future_years,
                dlog_path=dlog,
                msoa_conversion_path=imports['msoa_lookup'],
                constraints_zone_equivalence=designated_area,
                segment_cols=segmentation_cols,
                segment_groups=seg_groups,
                dlog_conversion_factor=1.0,
                dlog_data_column_key="population",
                perform_constraint=False,
                audit_location=out_path
            )
            # Save High Growth (Exceptional) zones to file
            hg_zones.to_csv(os.path.join(out_path, consts.EG_FNAME),
                            index=False)

        # ## POST D-LOG CONSTRAINT ## #
        if post_dlog_constraint:
            print("Performing the post-development log constraint on population...")
            print(". Pre Constraint:\n%s" % population[future_years].sum())
            print(". Constraint:\n%s" % population_constraint[future_years].sum())
            constraint_segments = du.intersection(segmentation_cols,
                                                  population_constraint)

            population = dlog_p.constrain_forecast(
                population,
                population_constraint,
                designated_area,
                base_year,
                future_years,
                self.zone_col,
                segment_cols=constraint_segments
            )
            print(". Post Constraint:\n%s" % population[future_years].sum())

        # Reindex and sum
        group_cols = [self.zone_col] + segmentation_cols
        index_cols = group_cols.copy() + all_years
        population = population.reindex(index_cols, axis='columns')
        population = population.groupby(group_cols).sum().reset_index()

        # Population Audit
        if audits:
            print('\n', '-'*15, 'Population Audit', '-'*15)
            for year in all_years:
                print('. Total population for year %s is: %.4f'
                      % (year, population[year].sum()))
            print('\n')

        # Write the produced population to file
        print("Writing population to file...")
        population_output = os.path.join(out_path, self.pop_fname)
        population.to_csv(population_output, index=False)

        # ## CREATE PRODUCTIONS ## #
        print("Population generated. Converting to productions...")
        productions = generate_productions(
            population=population,
            group_cols=group_cols,
            base_year=base_year,
            future_years=future_years,
            trip_origin='hb',
            trip_rates_path=imports['trip_rates'],
            time_splits_path=imports['time_splits'],
            mean_time_splits_path=imports['mean_time_splits'],
            mode_share_path=imports['mode_share'],
            audit_dir=exports['audits'],
        )

        # Optionally control to ntem
        lad_lookup_path = os.path.join(imports['lad_lookup'],
                                       consts.DEFAULT_LAD_LOOKUP)

        productions = control_productions_to_ntem(
            productions=productions,
            trip_origin='hb',
            ntem_dir=imports['ntem_control'],
            lad_lookup_path=lad_lookup_path,
            base_year=base_year,
            future_years=future_years,
            control_base_year=control_productions,
            control_future_years=control_fy_productions,
            audit_dir=exports['audits']
        )

        # Write productions to file
        print("Writing productions to file...")
        fname = consts.PRODS_FNAME % (self.zoning_system, 'raw_hb')
        path = os.path.join(out_path, fname)
        productions.to_csv(path, index=False)

        # ## CONVERT TO OLD EFS FORMAT ## #
        # Make sure columns are the correct data type
        productions['area_type'] = productions['area_type'].astype(int)
        productions['p'] = productions['p'].astype(int)
        productions['m'] = productions['m'].astype(int)
        productions['soc'] = productions['soc'].astype(str)
        productions['ns'] = productions['ns'].astype(str)
        productions['ca'] = productions['ca'].astype(int)
        productions.columns = productions.columns.astype(str)

        # Extract just the needed mode
        mask = productions['m'].isin(m_needed)
        productions = productions[mask]
        productions = productions.drop('m', axis='columns')

        # Reindex to just the wanted return cols
        group_cols = self.return_segments
        index_cols = group_cols.copy() + all_years

        productions = productions.reindex(index_cols, axis='columns')
        productions = productions.groupby(group_cols).sum().reset_index()

        print("Writing HB productions to disk...")
        fname = consts.PRODS_FNAME % (self.zoning_system, 'hb')
        path = os.path.join(out_path, fname)
        productions.to_csv(path, index=False)

        return productions


class NhbProductionModel:

    def __init__(self,
                 import_home: str,
                 export_home: str,
                 msoa_conversion_path: str,
                 model_name: str,
                 seg_level: str = 'tfn',
                 return_segments: List[str] = None,

                 base_year: str = consts.BASE_YEAR_STR,
                 future_years: List[str] = consts.FUTURE_YEARS_STR,
                 m_needed: List[int] = consts.MODES_NEEDED,

                 # Alternate input paths
                 hb_prods_path: str = None,
                 hb_attrs_path: str = None,
                 trip_rates_path: str = None,
                 mode_splits_path: str = None,
                 time_splits_path: str = None,

                 # Production control file
                 ntem_control_dir: str = None,
                 lad_lookup_dir: str = None,
                 control_productions: bool = True,
                 control_fy_productions: bool = True,

                 # Alternate output paths
                 audit_write_dir: str = None,

                 # Converting back to old EFS format
                 external_zone_col: str = 'model_zone_id',

                 # Alternate col names from inputs
                 m_col: str = 'm',
                 m_share_col: str = 'mode_share',
                 tp_col: str = 'tp',
                 tp_share_col: str = 'time_share',

                 # Col names for nhb trip rates
                 soc_col: str = 'soc',
                 ns_col: str = 'ns',
                 nhb_p_col: str = 'nhb_p',
                 trip_rate_col: str = 'trip_rate',

                 zoning_system: str = 'msoa',
                 audits: bool = True,
                 process_count: int = consts.PROCESS_COUNT
                 ):
        """
        Parameters
        ----------
        import_home:
            Path to the import home of NorMITs Demand. This can be gotten from
            an instance of the ExternalForecastSystem. Usually
            'Y:/NorMITs Demand/import'

        export_home:
            Path to the export home of this instance of outputs. This is
            usually related to a specific run of the ExternalForecastSystem,
            and should be gotten from there using generate_output_paths().
            e.g. 'E:/NorMITs Demand/norms_2015/v2_3-EFS_Output/iter1'

        msoa_conversion_path:
            Path to the file containing the conversion from msoa integer
            identifiers to the msoa string code identifiers. Hoping to remove
            this in a future update and align all of EFS to use msoa string
            code identifiers.

        model_name:
            The name of the model being run. This is usually something like:
            norms, norms_2015, or noham.

        seg_level:
            The level of segmentation to run at. This is used to determine
            how to produce the NHB Productions. Should be one of the values
            from consts.SEG_LEVELS

        return_segments:
            Which segmentation to use when returning the NHB productions.
            If left as None, it is automatically determined based on seg_level.

        base_year:
            The base year of the hb productions and attractions

        future_years:
            The future years of nhb productions to create - these years must
            also exist in the hb productions and attractions.

        m_needed:
            The modes to return when calling run.

        hb_prods_path:
            An alternate path to hb productions. If left as None, the NHB
            production model will look in the default output location of
            ProductionModel.

        hb_attrs_path:
            An alternate path to hb attractions. If left as None, the NHB
            production model will look in the default output location of
            AttractionModel.
        
        trip_rates_path:
            An alternate path to nhb trip rates. Any alternate inputs must be
            in the same format as the default. If left as None, the NHB
            production model will look in the default import location.
        
        mode_splits_path:
            An alternate path to nhb mode splits. Any alternate inputs must be
            in the same format as the default. If left as None, the NHB
            production model will look in the default import location.

        time_splits_path:
            An alternate path to nhb time splits. Any alternate inputs must be
            in the same format as the default. If left as None, the NHB
            production model will look in the default import location.
        
        ntem_control_dir:
            The path to alternate ntem control directory. If left as None, the
            production model will use the default import location.

        lad_lookup_dir:
            The path to alternate lad to msoa import data. If left as None, the
            production model will use the default import location.

        control_productions:
            Whether to control the generated productions to the constraints
            given in ntem_control_dir or not.

        control_fy_productions:
            Whether to control the generated future year productions to the
            constraints given in ntem_control_dir or not. When running for
            scenarios other than the base NTEM, this should be False.

        audit_write_dir:
            Alternate path to write the audits. If left as None, the default
            location is used.

        m_col:
            The name of the column in the mode_splits and time_splits that
            relate to mode.

        m_share_col:
            The name of the column in mode_splits that contains the mode
            share amount.

        tp_col:
            The name of the column in time_splits that contains the time_period
            id
        
        tp_share_col:
            The name of the column in time_splits that contains the time
            share amount.
        
        soc_col:
            The name of the column in trip_rates that contains the soc
            data.
        
        ns_col:
            The name of the column in trip_rates that contains the ns
            data.
        
        nhb_p_col:
            The name of the column in trip_rates that contains the nhb purpose
            data.
        
        trip_rate_col:
            The name of the column in trip_rates that contains the trip_rate
            data.
        
        zoning_system:
            The zoning system being used by the import files

        audits:
            Whether to print out audits or not.

        process_count:
            The number of processes to use in the NHB production model when
            multiprocessing is available.
        """
        # Validate inputs
        zoning_system = du.validate_zoning_system(zoning_system)
        model_name = du.validate_model_name(model_name)
        seg_level = du.validate_seg_level(seg_level)

        # Assign
        self.model_name = model_name
        self.seg_level = seg_level
        self.return_segments = return_segments
        self.msoa_conversion_path = msoa_conversion_path

        self.base_year = base_year
        self.future_years = future_years
        self.all_years = [str(x) for x in [base_year] + future_years]
        self.m_needed = m_needed

        self.zoning_system = zoning_system
        self.zone_col = '%s_zone_id' % zoning_system

        self.control_productions = control_productions
        self.control_fy_productions = control_fy_productions
        if not control_productions:
            self.control_fy_productions = False

        self.m_col = m_col
        self.m_share_col = m_share_col
        self.tp_col = tp_col
        self.tp_share_col = tp_share_col

        self.soc_col = soc_col
        self.ns_col = ns_col
        self.nhb_p_col = nhb_p_col
        self.trip_rate_col = trip_rate_col

        self.print_audits = audits
        self.process_count = process_count
        self.internal_zone_col = 'msoa_zone_id'
        self.external_zone_col = external_zone_col

        self.imports, self.exports = self._build_paths(
            import_home=import_home,
            export_home=export_home,
            hb_prods_path=hb_prods_path,
            hb_attrs_path=hb_attrs_path,
            trip_rates_path=trip_rates_path,
            mode_splits_path=mode_splits_path,
            time_splits_path=time_splits_path,
            ntem_control_dir=ntem_control_dir,
            lad_lookup_dir=lad_lookup_dir,
            audit_write_dir=audit_write_dir,
        )
        
        if seg_level == 'tfn':
            self.segments = ['area_type', 'p', 'soc', 'ns', 'ca']
            self.return_segments = [self.zone_col] + self.segments
            self.return_segments.remove('area_type')
        else:
            raise ValueError(
                "'%s' is a valid segmentation level, but I don't have a way "
                "of determining which segments to use for it. You should add "
                "one!" % seg_level
            )

        # Remove ca segmentation for some models
        if model_name == 'noham':
            self.return_segments.remove('ca')

    def _build_paths(self,
                     import_home: str,
                     export_home: str,
                     hb_prods_path: str,
                     hb_attrs_path: str,
                     trip_rates_path: str,
                     mode_splits_path: str,
                     time_splits_path: str,
                     ntem_control_dir: str,
                     lad_lookup_dir: str,
                     audit_write_dir: str,
                     ) -> Tuple[Dict[str, str], Dict[str, str]]:
        """
        Builds a dictionary of import and export paths, forming a standard
        calling procedure for I/O. Arguments allow default paths to be
        replaced.
        """
        # Set all unset import paths to default values
        if hb_prods_path is None:
            fname = consts.PRODS_FNAME % (self.zoning_system, 'raw_hb')
            hb_prods_path = os.path.join(export_home,
                                         consts.PRODUCTIONS_DIRNAME,
                                         fname)

        if hb_attrs_path is None:
            fname = consts.ATTRS_FNAME % (self.zoning_system, 'raw_hb')
            hb_attrs_path = os.path.join(export_home,
                                         consts.ATTRACTIONS_DIRNAME,
                                         fname)

        if trip_rates_path is None:
            trip_rates_path = os.path.join(import_home,
                                           consts.NHB_PARAMS_DIRNAME,
                                           'nhb_ave_wday_enh_trip_rates.csv')

        if mode_splits_path is None:
            mode_splits_path = os.path.join(import_home,
                                            consts.NHB_PARAMS_DIRNAME,
                                            'nhb_ave_wday_mode_split.csv')

        if time_splits_path is None:
            time_splits_path = os.path.join(import_home,
                                            consts.NHB_PARAMS_DIRNAME,
                                            'nhb_ave_wday_time_split.csv')

        if ntem_control_dir is None:
            path = 'ntem_constraints'
            ntem_control_dir = os.path.join(import_home, path)

        if lad_lookup_dir is None:
            path = os.path.join('zone_translation', 'no_overlap')
            lad_lookup_dir = os.path.join(import_home, path)

        if audit_write_dir is None:
            audit_write_dir = os.path.join(export_home,
                                           consts.AUDITS_DIRNAME,
                                           'Productions')
        du.create_folder(audit_write_dir, chDir=False)

        # Build the imports dictionary
        imports = {
            'productions': hb_prods_path,
            'attractions': hb_attrs_path,
            'trip_rates': trip_rates_path,
            'mode_splits': mode_splits_path,
            'time_splits': time_splits_path,
            'ntem_control': ntem_control_dir,
            'lad_lookup': lad_lookup_dir
        }

        # Make sure all import paths exit
        for key, path in imports.items():
            if not os.path.exists(path):
                raise IOError(
                    "NHB Production Model Imports: The path for %s does not "
                    "exist.\nFull path: %s" % (key, path)
                )

        # Build the efs_exports dictionary
        exports = {
            'productions': os.path.join(export_home, consts.PRODUCTIONS_DIRNAME),
            'attractions': os.path.join(export_home, consts.ATTRACTIONS_DIRNAME),
            'audits': audit_write_dir
        }

        # Make sure all export paths exit
        for key, path in exports.items():
            if not os.path.exists(path):
                raise IOError(
                    "NHB Production Model Exports: The path for %s does not "
                    "exist.\nFull path: %s" % (key, path)
                )

        return imports, exports

    def _apply_mode_splits(self,
                           nhb_prods: pd.DataFrame,
                           verbose: bool = True
                           ) -> pd.DataFrame:
        """
        Applies Mode splits on the given NHB productions

        Parameters
        ----------
        nhb_prods:
            Dataframe containing the NHB productions to split.
            Needs the following column names to merge with the mode splits:
            ['area_type', 'p', 'ca', 'nhb_p']

        verbose:
            Whether to print a progress bar while applying the splits or not

        Returns
        -------
        mode_split_nhb_prods:
            The given nhb_prods additionally split by mode
        """
        # Init
        m_col = self.m_col
        m_share_col = self.m_share_col
        col_names = list(nhb_prods)

        mode_splits = pd.read_csv(self.imports['mode_splits'])
        unq_m = mode_splits[m_col].unique()

        merge_cols = du.intersection(list(nhb_prods), list(mode_splits))
        expected_merge_cols = ['area_type', 'p', 'ca', 'nhb_p']

        # Validate the merge columns
        if not du.equal_ignore_order(merge_cols, expected_merge_cols):
            raise du.NormitsDemandError(
                "Expecting to merge on %s, but only found %s columns. Has "
                "something gone wrong?"
                % (str(expected_merge_cols), str(merge_cols))
            )

        # Apply the mode splits
        eff_m_split = list()
        desc = 'Splitting by mode'
        for mode in tqdm(unq_m, desc=desc, disable=not verbose):
            # Get just this mode share
            needed_cols = merge_cols.copy() + [m_col, m_share_col]
            m_subset = mode_splits.reindex(needed_cols, axis='columns').copy()

            mask = (mode_splits[m_col] == mode)
            m_subset = m_subset[mask]

            # Merge and infill missing modes with 0
            m_split = pd.merge(
                nhb_prods.copy(),
                m_subset,
                how='left',
                on=merge_cols
            ).fillna(0)

            # Multiply by mode share
            for year in self.all_years:
                m_split[year] *= m_split[m_share_col]

            # Drop the unneeded cols
            m_split = m_split.drop([m_col, m_share_col], axis='columns')

            eff_m_split.append({
                'm': mode,
                'df': m_split
            })

        # Compile back into a fill mat and return
        col_names += [m_col]
        return du.compile_efficient_df(eff_m_split, col_names=col_names)

    def _apply_time_splits(self,
                           nhb_prods: pd.DataFrame,
                           verbose: bool = True
                           ) -> pd.DataFrame:
        """
        Applies time periods splits to NHB Productions

        Parameters
        ----------
        nhb_prods:
            Dataframe containing the NHB productions to split.
            Needs the following column names to merge with the mode splits:
            ['area_type', 'ca', 'nhb_p', 'm']

        verbose:
            Whether to print a progress bar while applying the splits or not

        Returns
        -------
        time_split_nhb_prods:
            The given nhb_prods additionally split by time periods
        """
        # Init
        tp_col = self.tp_col
        tp_share_col = self.tp_share_col
        col_names = list(nhb_prods)

        time_splits = pd.read_csv(self.imports['time_splits'])
        unq_tp = time_splits[tp_col].unique()

        merge_cols = du.intersection(list(nhb_prods), list(time_splits))
        expected_merge_cols = ['area_type', 'ca', 'nhb_p', 'm']

        # Validate the merge columns
        if not du.equal_ignore_order(merge_cols, expected_merge_cols):
            raise du.NormitsDemandError(
                "Expecting to merge on %s, but only found %s columns. Has "
                "something gone wrong?"
                % (str(expected_merge_cols), str(merge_cols))
            )

        # Apply the mode splits
        eff_tp_split = list()
        desc = 'Splitting by time period'
        for tp in tqdm(unq_tp, desc=desc, disable=not verbose):
            # Get just this mode share
            needed_cols = merge_cols.copy() + [tp_col, tp_share_col]
            tp_subset = time_splits.reindex(needed_cols, axis='columns').copy()

            mask = (time_splits[tp_col] == tp)
            tp_subset = tp_subset[mask]

            # Merge and infill missing modes with 0
            tp_split = pd.merge(
                nhb_prods.copy(),
                tp_subset,
                how='left',
                on=merge_cols
            ).fillna(0)

            # Multiply by mode share
            for year in self.all_years:
                tp_split[year] *= tp_split[tp_share_col]

            # Drop the unneeded cols
            tp_split = tp_split.drop([tp_col, tp_share_col], axis='columns')

            eff_tp_split.append({
                'tp': tp,
                'df': tp_split
            })

        # Compile back into a fill mat and return
        col_names += [tp_col]
        return du.compile_efficient_df(eff_tp_split, col_names=col_names)

    def _gen_base_productions(self,
                              verbose=True
                              ) -> pd.DataFrame:
        """
        Generates the base NHB Productions from HB Productions and Attractions

        Performs a kind of pseudo distribution in order to retain the HB
        production segmentation in the attractions. The segmentation needs
        to be retained in order to apply the NHB trip rates (Gathered from
        NTS data).

        Parameters
        ----------
        verbose:
            Whether to print progress bars during processing or not.

        Returns
        -------
        NHB_productions:
            A base set of NHB productions based on the HB productions and
            attractions. Return will be segmented by level passed into class
            constructor.
        """

        # Read in files
        dtypes = {self.soc_col: str, self.ns_col: str}
        prods = pd.read_csv(self.imports['productions'], dtype=dtypes)
        attrs = pd.read_csv(self.imports['attractions'], dtype=dtypes)

        # Determine all unique segments - ignore mode
        seg_params = du.build_seg_params(self.seg_level, prods)

        # We ignore mode here because it saves us many many loop iterations.
        # We believe the same results come out the other side whether we include
        # mode in the loop or not - If we run into some weird problems, this
        # should be looked at again
        if 'm_needed' in seg_params:
            seg_params['m_needed'] = [None]

        # Area Type is a special one here - make it the outer loop
        unique_at = prods['area_type'].unique().tolist()

        # ## MULTIPROCESS ## #
        # Build the unchanging arguments
        unchanging_kwargs = {
            'area_type': unique_at[0],
            'seg_level': self.seg_level,
            'seg_params': seg_params,
            'segments': self.segments,
            'trip_rates_path': self.imports['trip_rates'],
            'all_years': self.all_years,
            'prods': prods,
            'attrs': attrs,
            'zone_col': self.zone_col,
            'nhb_p_col': self.nhb_p_col,
            'trip_rate_col': self.trip_rate_col,
            'verbose': verbose
        }

        # Add in the changing kwargs
        kwargs_list = list()
        for area_type in unique_at:
            kwargs = unchanging_kwargs.copy()
            kwargs['area_type'] = area_type
            kwargs_list.append(kwargs)

        returns = multiprocessing.multiprocess(
            _gen_base_productions_internal,
            kwargs=kwargs_list,
            process_count=self.process_count
        )
        eff_nhb_prods = reduce(operator.concat, returns)

        # Compile segmented
        print("Compiling full NHB Productions...")

        segments = list(eff_nhb_prods[0].keys())
        segments.remove('df')
        col_names = [self.zone_col] + segments + self.all_years
        return du.compile_efficient_df(eff_nhb_prods, col_names=col_names)

    def run(self,
            output_raw: bool = True,
            recreate_productions: bool = True,
            verbose: bool = True,
            ) -> pd.DataFrame:
        """
        Runs the whole NHB production model

        Performs the following actions:
          - Pseudo distributes the HB productions and attractions, retaining
            the productions segmentation in the attractions
          - Applies the NHB Trip Rates to produce segmented NHB Productions
          - Applies mode spits the the NHB productions
          - Applies time period splits to the NHB productions
          - Optionally Controls to NTEM
          - Extracts the requested mode, and returns NHB Productions at the
            requested segmentation level (as defined in the class constructor)

        Parameters
        ----------
        output_raw:
            Whether to output the raw nhb productions before aggregating to
            the required segmentation and mode.

        recreate_productions:
            Whether to recreate the nhb productions or not. If False, it will
            look in out_path for previously produced productions and return
            them. If none can be found, they will be generated.

        verbose:
            Whether to print progress bars during processing or not.

        Returns
        -------
        NHB_Productions:
            NHB productions for the mode and segmentation requested in the
            class constructor
        """
        # Return previously created productions if we can
        fname = consts.PRODS_FNAME % (self.zoning_system, 'nhb')
        final_output_path = os.path.join(self.exports['productions'], fname)

        if not recreate_productions and os.path.isfile(final_output_path):
            print("Found some already produced nhb productions. Using them!")
            return pd.read_csv(final_output_path)

        # Initialise timing
        start_time = timing.current_milli_time()
        du.print_w_toggle(
            "Starting NHB Production Model at: %s" % timing.get_datetime(),
            echo=verbose
        )

        nhb_prods = self._gen_base_productions(verbose=verbose)

        # Reindex and tidy
        group_cols = [self.zone_col] + self.segments + ['nhb_p']
        index_cols = group_cols.copy() + self.all_years

        nhb_prods = nhb_prods.reindex(index_cols, axis='columns')
        nhb_prods = nhb_prods.groupby(group_cols).sum().reset_index()

        # ## SPLIT PRODUCTIONS BY MODE AND TIME ## #
        print("Splitting NHB productions by mode and time...")
        nhb_prods = self._apply_mode_splits(nhb_prods, verbose=verbose)

        # No longer need HB purpose
        extra_segments = [self.m_col, 'nhb_p']
        group_cols = [self.zone_col] + self.segments + extra_segments
        group_cols.remove('p')
        index_cols = group_cols.copy() + self.all_years

        nhb_prods = nhb_prods.reindex(index_cols, axis='columns')
        nhb_prods = nhb_prods.groupby(group_cols).sum().reset_index()

        nhb_prods = self._apply_time_splits(nhb_prods, verbose=verbose)

        # Reindex and tidy
        group_cols += [self.tp_col]
        index_cols = group_cols.copy() + self.all_years

        nhb_prods = nhb_prods.reindex(index_cols, axis='columns')
        nhb_prods = nhb_prods.groupby(group_cols).sum().reset_index()

        nhb_prods = nhb_prods.rename(columns={'nhb_p': 'p'})

        # Population Audit
        if self.print_audits:
            print('\n', '-' * 15, 'Uncorrected NHB Production Audit', '-' * 15)
            for year in self.all_years:
                print('. Total population for year %s is: %.4f'
                      % (year, nhb_prods[year].sum()))
            print('\n')

        # ## OPTIONALLY CONSTRAIN TO NTEM ## #
        lad_lookup_path = os.path.join(self.imports['lad_lookup'],
                                       consts.DEFAULT_LAD_LOOKUP)

        nhb_prods = control_productions_to_ntem(
            productions=nhb_prods,
            trip_origin='nhb',
            ntem_dir=self.imports['ntem_control'],
            lad_lookup_path=lad_lookup_path,
            base_year=self.base_year,
            future_years=self.future_years,
            control_base_year=self.control_productions,
            control_future_years=self.control_fy_productions,
            ntem_control_cols=['p', 'm', 'tp'],
            ntem_control_dtypes=[int, int, int],
            audit_dir=self.exports['audits']
        )

        # Output productions before any aggregation
        if output_raw:
            print("Writing raw NHB Productions to disk...")
            fname = consts.PRODS_FNAME % (self.zoning_system, 'raw_nhb')
            path = os.path.join(self.exports['productions'], fname)
            nhb_prods.to_csv(path, index=False)

        # ## TIDY UP AND AGGREGATE ## #
        print("Aggregating to required output format...")
        group_cols = list(nhb_prods)
        for year in self.all_years:
            group_cols.remove(year)
        nhb_prods = nhb_prods.groupby(group_cols).sum().reset_index()

        # Extract just the needed mode
        mask = nhb_prods['m'].isin(self.m_needed)
        nhb_prods = nhb_prods[mask]
        nhb_prods = nhb_prods.drop('m', axis='columns')

        # Reindex to just the wanted return cols
        group_cols = self.return_segments
        index_cols = group_cols.copy() + self.all_years

        nhb_prods = nhb_prods.reindex(index_cols, axis='columns')
        nhb_prods = nhb_prods.groupby(group_cols).sum().reset_index()

        # Output the aggregated productions
        print("Writing NHB Productions to disk...")
        fname = consts.PRODS_FNAME % (self.zoning_system, 'nhb')
        path = os.path.join(self.exports['productions'], fname)
        nhb_prods.to_csv(path, index=False)

        # End timing
        end_time = timing.current_milli_time()
        du.print_w_toggle(
            "Finished NHB Production Model at: %s" % timing.get_datetime(),
            echo=verbose
        )
        du.print_w_toggle(
            "NHB Production Model took: %s"
            % timing.time_taken(start_time, end_time),
            echo=verbose
        )

        return nhb_prods


def _gen_base_productions_internal(area_type,
                                   seg_level,
                                   seg_params,
                                   segments,
                                   trip_rates_path,
                                   all_years,
                                   prods,
                                   attrs,
                                   zone_col,
                                   nhb_p_col,
                                   trip_rate_col,
                                   verbose
                                   ):
    # init
    nhb_trip_rates = pd.read_csv(trip_rates_path)

    total = du.seg_level_loop_length(seg_level, seg_params)
    desc = "Calculating NHB Productions at %s" % str(area_type)
    p_bar = tqdm(total=total, desc=desc, disable=not verbose)

    eff_nhb_prods = list()
    for seg_vals in du.seg_level_loop_generator(seg_level, seg_params):
        # Add in area type - check our segments are correct
        seg_vals['area_type'] = area_type
        if not all([k in segments for k in seg_vals.keys()]):
            raise KeyError(
                "Our seg_vals and segments disagree on which segments "
                "should be used. Has one been changed without the "
                "other?"
            )

        # ## PSEUDO DISTRIBUTE EACH SEGMENT ## #
        # We do this to retain segments from productions

        # Filter the productions and attractions
        p_subset = du.filter_by_segmentation(prods, seg_vals, fit=True)

        # Soc0 is always special - do this to avoid dropping demand
        if seg_vals.get('soc', -1) == '0':
            temp_seg_vals = seg_vals.copy()
            temp_seg_vals.pop('soc')
            a_subset = du.filter_by_segmentation(attrs, temp_seg_vals, fit=True)
        else:
            a_subset = du.filter_by_segmentation(attrs, seg_vals, fit=True)

        # Remove all segmentation from the attractions
        group_cols = [zone_col]
        index_cols = group_cols.copy() + all_years
        a_subset = a_subset.reindex(index_cols, axis='columns')
        a_subset = a_subset.groupby(group_cols).sum().reset_index()

        # Balance P/A to pseudo distribute
        a_subset = du.balance_a_to_p(
            productions=p_subset,
            attractions=a_subset,
            unique_cols=all_years,
        )

        # ## APPLY NHB TRIP RATES ## #
        # Subset the trip_rates
        tr_index = [nhb_p_col, trip_rate_col]
        tr_subset = du.filter_by_segmentation(nhb_trip_rates, seg_vals,
                                              fit=True)
        tr_subset = tr_subset.reindex(tr_index, axis='columns')

        # Validate
        if len(tr_subset) > len(consts.ALL_NHB_P):
            raise du.NormitsDemandError(
                "We have more than %d rows after filtering the nhb trip "
                "rates. There are probably duplicates in the filter "
                "somehow" % len(consts.ALL_NHB_P)
            )

        # Convert to a dictionary for speed
        tr_dict = dict(zip(tr_subset[nhb_p_col].values,
                           tr_subset[trip_rate_col].values))
        del tr_subset

        # Build the trip rates data for this segment
        for p, trip_rate in tr_dict.items():
            # Get the actual productions
            nhb_prods_df = a_subset.copy()
            for year in all_years:
                nhb_prods_df[year] *= trip_rate

            # Store for compile later
            seg_nhb_prods = seg_vals.copy()
            seg_nhb_prods.update({
                'nhb_p': p,
                'df': nhb_prods_df,
            })

            # Add soc/ns in as needed
            if 'soc' in seg_nhb_prods and 'ns' not in seg_nhb_prods:
                seg_nhb_prods['ns'] = 'none'
            if 'ns' in seg_nhb_prods and 'soc' not in seg_nhb_prods:
                seg_nhb_prods['soc'] = 'none'

            eff_nhb_prods.append(seg_nhb_prods)

        p_bar.update(1)

    p_bar.close()
    return eff_nhb_prods


def build_production_imports(import_home: str,
                             trip_rates_path: str = None,
                             time_splits_path: str = None,
                             mean_time_splits_path: str = None,
                             mode_share_path: str = None,
                             msoa_lookup_path: str = None,
                             ntem_control_dir: str = None,
                             lad_lookup_dir: str = None,
                             set_controls: bool = True,
                             ) -> Dict[str, str]:
    """
    Builds a dictionary of production import paths, forming a standard calling
    procedure for production imports. Arguments allow default paths to be
    replaced.

    Parameters
    ----------
    import_home:
        The base path to base all of the other import paths from. This
        should usually be "Y:/NorMITs Demand/import" for default inputs.

    lu_import_path:
        An alternate land use import path to use. File will need to follow the
        same format as default file.

    trip_rates_path:
        An alternate trip rates import path to use. File will need to follow the
        same format as default file.

    time_splits_path:
        An alternate time splits import path to use. File will need to follow
        the same format as default file.

    mean_time_splits_path:
        An alternate mean time splits import path to use. File will need to
        follow the same format as default file.

    mode_share_path:
        An alternate mode share import path to use. File will need to follow
        the same format as default file.

    msoa_lookup_path:
        An alternate msoa lookup import path to use. File will need to follow
        the same format as default file.

    ntem_control_dir:
        An alternate ntem control directory to use. File will need to follow
        the same format as default files.

    lad_lookup_dir:
        An alternate lad lookup directory to use. File will need to follow
        the same format as default file.

    set_controls:
        If False 'ntem_control' and 'lad_lookup' outputs will be set to None,
        regardless of any other inputs.

    Returns
    -------
    import_dict:
        A dictionary of paths with the following keys:
        'land_use',
        'trip_rates',
        'time_splits',
        'mean_time_splits',
        'mode_share_path',
        'ntem_control',
        'lad_lookup',
    """
    # Set all unset import paths to default values
    if trip_rates_path is None:
        path = 'tfn_segment_production_params\hb_trip_rates.csv'
        trip_rates_path = os.path.join(import_home, path)

    if time_splits_path is None:
        path = 'tfn_segment_production_params\hb_time_split.csv'
        time_splits_path = os.path.join(import_home, path)

    if mean_time_splits_path is None:
        path = 'tfn_segment_production_params\hb_ave_time_split.csv'
        mean_time_splits_path = os.path.join(import_home, path)

    if mode_share_path is None:
        path = 'tfn_segment_production_params\hb_mode_split.csv'
        mode_share_path = os.path.join(import_home, path)

    if msoa_lookup_path is None:
        path = "zone_translation\msoa_zones.csv"
        msoa_lookup_path = os.path.join(import_home, path)

    if set_controls and ntem_control_dir is None:
        path = 'ntem_constraints'
        ntem_control_dir = os.path.join(import_home, path)

    if set_controls and lad_lookup_dir is None:
        path = os.path.join('zone_translation', 'no_overlap')
        lad_lookup_dir = os.path.join(import_home, path)

    # Assign to dict
    imports = {
        'trip_rates': trip_rates_path,
        'time_splits': time_splits_path,
        'mean_time_splits': mean_time_splits_path,
        'mode_share': mode_share_path,
        'msoa_lookup': msoa_lookup_path,
        'ntem_control': ntem_control_dir,
        'lad_lookup': lad_lookup_dir,
    }

    # Make sure all import paths exit
    for key, path in imports.items():
        if not os.path.exists(path):
            raise IOError(
                "HB Production Model Imports: The path for %s does not "
                "exist.\nFull path: %s" % (key, path)
            )

    return imports


def build_production_exports(export_home: str,
                             audit_write_dir: str = None
                             ) -> Dict[str, str]:
    """
    Builds a dictionary of production export paths, forming a standard calling
    procedure for production efs_exports. Arguments allow default paths to be
    replaced.

    Parameters
    ----------
    export_home:
        Usually the export home for this run of the EFS. Can be automatically
        generated using du.build_io_paths()

    audit_write_dir:
        An alternate export path for the audits. By default this will be:
        audits/productions/

    Returns
    -------
    export_dict:
        A dictionary of paths with the following keys:
        'audits'

    """
    # Set all unset export paths to default values
    if audit_write_dir is None:
        audit_write_dir = os.path.join(export_home,
                                       consts.AUDITS_DIRNAME,
                                       'Productions')
    du.create_folder(audit_write_dir, chDir=False)

    # Build the efs_exports dictionary
    exports = {
        'audits': audit_write_dir
    }

    # Make sure all export paths exit
    for key, path in exports.items():
        if not os.path.exists(path):
            raise IOError(
                "HB Production Model Exports: The path for %s does not "
                "exist.\nFull path: %s" % (key, path)
            )

    return exports


def get_pop_data_from_land_use(import_path: nd.PathLike,
                               years: List[str],
                               segmentation_cols: List[str],
                               lu_zone_col: str = 'msoa_zone_id',
                               ) -> pd.DataFrame:
    """
    Reads in land use outputs and aggregates up to segmentation_cols.

    Combines all the dataframe from each into a single dataframe.

    Parameters
    ----------
    import_path:
        Path to the land use directory containing population data for years

    years:
        The years of future year population data to read in.

    segmentation_cols:
        The columns to keep in the land use data. If None, defaults to:
         [
            'area_type',
            'traveller_type',
            'soc',
            'ns',
        ]

    lu_zone_col:
        The name of the column in the land use data that refers to the zones.

    Returns
    -------
    population:
        A dataframe of population data for all years segmented by
        segmentation_cols. Will also include lu_zone_col and year cols
        from land use.
    """
    # Init
    if segmentation_cols is None:
        # Assume full segmentation if not told otherwise
        segmentation_cols = [
            'area_type',
            'traveller_type',
            'soc',
            'ns',
            'ca'
        ]
    group_cols = [lu_zone_col] + segmentation_cols

    all_pop_ph = list()
    for year in tqdm(years):
        # BACKLOG: REMOVE SKIP OVER 2018!!!
        if year == '2018':
            continue

        # Build the path to this years data
        fname = consts.LU_POP_FNAME % str(year)
        lu_path = os.path.join(import_path, fname)
        year_pop = pd.read_csv(lu_path)

        # ## FILTER TO JUST THE DATA WE NEED ## #
        # Set up the columns to keep
        index_cols = group_cols.copy() + [year]

        # Check all columns exist
        year_pop_cols = list(year_pop)
        for col in index_cols:
            if col not in year_pop_cols:
                raise nd.NormitsDemandError(
                    "Tried to read in population data from NorMITs Land Use "
                    "for year %s. Cannot find all the needed columns in the "
                    "data. Specifically, column %s does not exist."
                    % (year, col)
                )

        # Filter down
        year_pop = year_pop.reindex(columns=index_cols)
        year_pop = year_pop.groupby(group_cols).sum().reset_index()

        all_pop_ph.append(year_pop)

    return du.merge_df_list(all_pop_ph, on=group_cols)


def merge_pop_trip_rates(population: pd.DataFrame,
                         group_cols: List[str],
                         trip_rates_path: str,
                         time_splits_path: str,
                         mean_time_splits_path: str,
                         mode_share_path: str,
                         audit_out: str,
                         tp_needed: List[int] = consts.TP_NEEDED,
                         traveller_type_col: str = 'traveller_type',
                         ) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Converts a single year of population into productions
    # TODO: Update merge_pop_trip_rates() docs

    Carries out the following actions:
        - Calculates the weekly trip rates
        - Convert to average weekday trip rate, and split by time period
        - Further splits the productions by mode

    Parameters
    ----------
    population:
        Dataframe containing the segmented population values for this year.

    group_cols:
        A list of all non-unique columns in population. This will be used to
        retain any already given segmentation.

    trip_rates_path:
        Path to the file of trip rates data. Will try to merge with the
        population on all possible columns.

    time_splits_path:
        Path to the file of time splits by ['area_type', 'traveller_type', 'p'].

    mean_time_splits_path:
        Path to the file of mean time splits by 'p'

    mode_share_path:
        Path to the file of mode shares by ['area_type', 'ca', 'p']

    audit_out:
        The directory to write out any audit files produced.

    control_path:
        Path to the file containing the data to control the produced
        productions to. If left as None, no control will be carried out.


    lad_lookup_dir:
        Path to the file containing the conversion from msoa zoning to LAD
        zoning, to be used for controlling the productions. If left as None, no
        control will be carried out.

    lad_lookup_name:
        The name of the file in lad_lookup_dir that contains the msoa zoning
        to LAD zoning conversion.

    tp_needed:
        A list of the time periods to split the productions by.

    traveller_type_col:
        The name of the column in population that contains the traveller type
        information.

    Returns
    -------
    Productions:
        The population converted to productions for this year. Will try to keep
        all segmentation in the given population if possible, and add more.
    """
    # Init
    group_cols = group_cols.copy()
    group_cols.insert(2, 'p')

    index_cols = group_cols.copy()
    index_cols.append('trips')

    # ## GET WEEKLY TRIP RATE FROM POPULATION ## #
    # Init
    trip_rates = pd.read_csv(trip_rates_path)

    # TODO: Make the production model more adaptable
    # Merge on all possible columns
    tr_cols = list(trip_rates)
    pop_cols = list(population)
    tr_merge_cols = [x for x in tr_cols if x in pop_cols]

    purpose_ph = dict()
    all_purposes = trip_rates['p'].drop_duplicates().reset_index(drop=True)
    desc = "Building trip rates by purpose"
    for p in tqdm(all_purposes, desc=desc):
        trip_rate_subset = trip_rates[trip_rates['p'] == p].copy()
        ph = population.copy()

        if p in consts.SOC_P:
            # Update ns with none
            ph['ns'] = 'none'
            ph['soc'] = ph['soc'].astype(float).astype(int)
            # Insurance policy
            trip_rate_subset['ns'] = 'none'
            trip_rate_subset['soc'] = trip_rate_subset['soc'].astype(int)

        elif p in consts.NS_P:
            # Update soc with none
            ph['soc'] = 'none'
            ph['ns'] = ph['ns'].astype(float).astype(int)
            # Insurance policy
            trip_rate_subset['soc'] = 'none'
            trip_rate_subset['ns'] = trip_rate_subset['ns'].astype(int)

        # Merge and calculate productions
        ph = ph[ph['people'] > 0].copy()
        ph = pd.merge(ph, trip_rate_subset, on=tr_merge_cols)

        ph['trips'] = ph['trip_rate'] * ph['people']
        ph = ph.drop(['trip_rate'], axis=1)

        # Group and sum
        ph = ph.reindex(index_cols, axis='columns')
        ph = ph.groupby(group_cols).sum().reset_index()

        # Update dictionary
        purpose_ph[p] = ph
    del trip_rates
    # Results in weekly trip rates by purpose and segmentation

    # ## SPLIT WEEKLY TRIP RATES BY TIME PERIOD ## #
    # Also converts to average weekday trips!
    # Init
    time_splits = pd.read_csv(time_splits_path)
    mean_time_splits = pd.read_csv(mean_time_splits_path)
    tp_merge_cols = ['area_type', 'traveller_type', 'p']

    # Convert tp nums to strings
    tp_needed = ['tp' + str(x) for x in tp_needed]

    tp_ph = dict()
    desc = 'Splitting trip rates by time period'
    for tp in tqdm(tp_needed, desc=desc):
        needed_cols = tp_merge_cols.copy() + [tp]
        tp_subset = time_splits.reindex(needed_cols, axis='columns').copy()
        tp_mean_subset = mean_time_splits.reindex(['p', tp], axis='columns').copy()

        for p, p_df in purpose_ph.items():
            # Get mean for infill
            tp_mean = tp_mean_subset[tp_mean_subset['p'] == p][tp]

            # Merge and infill
            tp_mat = p_df.copy()
            tp_mat = pd.merge(
                tp_mat,
                tp_subset,
                how='left',
                on=tp_merge_cols
            )
            tp_mat[tp] = tp_mat[tp].fillna(tp_mean)

            # Apply tp split and divide by 5 to get average weekday by tp
            tp_mat['trips'] = (tp_mat['trips'] * tp_mat[tp]) / 5

            # Group and sum
            tp_mat = tp_mat.reindex(index_cols, axis='columns')
            tp_mat = tp_mat.groupby(group_cols).sum().reset_index()

            # Add to compilation dict
            tp_ph[(p, tp)] = tp_mat
    del time_splits
    del mean_time_splits
    del purpose_ph
    # Results in average weekday trips by purpose, tp, and segmentation

    # Quick Audit
    approx_tp_totals = []
    for key, dat in tp_ph.items():
        total = dat['trips'].sum()
        approx_tp_totals.append(total)
    ave_wday = sum(approx_tp_totals)
    print('. Average weekday productions: %.2f' % ave_wday)

    # ## SPLIT AVERAGE WEEKDAY TRIP RATES BY MODE ## #
    # TODO: Apply at MSOA level rather than area type
    # Init
    mode_share = pd.read_csv(mode_share_path)
    m_merge_cols = ['area_type', 'ca', 'p']
    target_modes = ['m1', 'm2', 'm3', 'm5', 'm6']

    # Can get rid of traveller type now - too much detail
    # If we keep it we WILL have memory problems
    group_cols.remove(traveller_type_col)
    index_cols.remove(traveller_type_col)

    m_ph = dict()
    desc = 'Applying mode share splits'
    for m in tqdm(target_modes, desc=desc):
        needed_cols = m_merge_cols.copy() + [m]
        m_subset = mode_share.reindex(needed_cols, axis='columns').copy()

        for (p, tp), dat in tp_ph.items():
            m_mat = dat.copy()

            # Would merge all purposes, but left join should pick out target mode
            m_mat = pd.merge(
                m_mat,
                m_subset,
                how='left',
                on=m_merge_cols
            )

            # Apply m split
            m_mat['trips'] = (m_mat['trips'] * m_mat[m])

            # Reindex cols for efficiency
            m_mat = m_mat.reindex(index_cols, axis='columns')
            m_mat = m_mat.groupby(group_cols).sum().reset_index()

            m_mat = m_mat[m_mat['trips'] > 0]

            m_ph[(p, tp, m)] = m_mat
    del mode_share
    del tp_ph
    # Results in average weekday trips by purpose, tp, mode, and segmentation

    print("Writing topline audit...")
    approx_mode_totals = []
    for key, dat in m_ph.items():
        total = dat['trips'].sum()
        approx_mode_totals.append([key, total])

    # Build topline report
    topline = pd.DataFrame(approx_mode_totals, columns=['desc', 'total'])
    # Split key into components
    topline['p'], topline['tp'], topline['m'] = list(zip(*topline['desc']))
    topline = topline.reindex(['p', 'tp', 'm', 'total'], axis=1)
    topline = topline.groupby(['p', 'tp', 'm']).sum().reset_index()
    topline.to_csv(os.path.join(audit_out), index=False)

    # ## COMPILE ALL MATRICES INTO ONE ## #
    output_ph = list()
    desc = 'Compiling productions'
    for (p, tp, m), dat in tqdm(m_ph.items(), desc=desc):
        dat['p'] = p
        dat['tp'] = tp
        dat['m'] = m
        output_ph.append(dat)
    msoa_output = pd.concat(output_ph)

    # We now need to deal with tp and mode in one big matrix
    group_cols = group_cols + ['tp', 'm']
    index_cols = group_cols.copy()
    index_cols.append('trips')

    # Ensure matrix is in the correct format
    msoa_output = msoa_output.reindex(index_cols, axis='columns')
    msoa_output = msoa_output.groupby(group_cols).sum().reset_index()
    msoa_output['m'] = [int(m[1:]) for m in msoa_output['m']]
    msoa_output['tp'] = [int(tp[2:]) for tp in msoa_output['tp']]
    msoa_output['p'] = msoa_output['p'].astype(int)
    msoa_output['m'] = msoa_output['m'].astype(int)

    return msoa_output


def control_productions_to_ntem(productions: pd.DataFrame,
                                trip_origin: str,
                                ntem_dir: str,
                                lad_lookup_path: str,
                                base_year: str,
                                future_years: List[str] = None,
                                control_base_year: bool = True,
                                control_future_years: bool = False,
                                ntem_control_cols: List[str] = None,
                                ntem_control_dtypes: List[Callable] = None,
                                audit_dir: str = None
                                ) -> pd.DataFrame:
    # TODO: Write control_productions_to_ntem() docs
    # Set up default args
    if ntem_control_cols is None:
        ntem_control_cols = ['p', 'm']

    if ntem_control_dtypes is None:
        ntem_control_dtypes = [int, int]

    # Init
    future_years = list() if future_years is None else future_years
    all_years = [base_year] + future_years
    init_index_cols = list(productions)
    init_group_cols = du.list_safe_remove(list(productions), all_years)

    # Use sorting to avoid merge. Productions is a BIG DF
    all_years = [base_year] + future_years
    sort_cols = du.list_safe_remove(list(productions), all_years)
    productions = productions.sort_values(sort_cols)

    # Do we need to grow on top of a controlled base year? (multiplicative)
    grow_over_base = control_base_year and not control_future_years

    # Get growth values over base
    if grow_over_base:
        growth_factors = productions.copy()
        for year in future_years:
            growth_factors[year] /= growth_factors[base_year]
        growth_factors.drop(columns=[base_year], inplace=True)

        # Output an audit of the growth factors calculated
        if audit_dir is not None:
            fname = consts.PRODS_MG_FNAME % ('msoa', trip_origin)
            path = os.path.join(audit_dir, fname)
            pd.DataFrame(growth_factors).to_csv(path, index=False)

    # ## NTEM CONTROL YEARS ## #
    # Figure out which years to control
    control_years = list()
    if control_base_year:
        control_years.append(base_year)
    if control_future_years:
        control_years += future_years

    audits = list()
    for year in control_years:
        # Init audit
        year_audit = {'year': year}

        # Setup paths
        ntem_fname = consts.NTEM_CONTROL_FNAME % ('pa', year)
        ntem_path = os.path.join(ntem_dir, ntem_fname)

        # Read in control files
        ntem_totals = pd.read_csv(ntem_path)
        ntem_lad_lookup = pd.read_csv(lad_lookup_path)

        print("\nPerforming NTEM constraint for %s..." % year)
        productions, audit, *_, = ntem.control_to_ntem(
            control_df=productions,
            ntem_totals=ntem_totals,
            zone_to_lad=ntem_lad_lookup,
            constraint_cols=ntem_control_cols,
            base_value_name=year,
            ntem_value_name='productions',
            trip_origin=trip_origin
        )

        # Update Audits for output
        year_audit.update(audit)
        audits.append(year_audit)

    # Controlling to NTEM seems to change some of the column dtypes
    dtypes = {c: d for c, d in zip(ntem_control_cols, ntem_control_dtypes)}
    productions = productions.astype(dtypes)

    # Write the audit to disk
    if len(audits) > 0 and audit_dir is not None:
        fname = consts.PRODS_FNAME % ('msoa', trip_origin)
        path = os.path.join(audit_dir, fname)
        pd.DataFrame(audits).to_csv(path, index=False)

    if not grow_over_base:
        return productions

    # ## ADD PRE CONTROL GROWTH BACK ON ## #
    # Merge on all possible columns
    merge_cols = du.list_safe_remove(list(growth_factors), all_years)
    productions = pd.merge(
        productions,
        growth_factors,
        how='left',
        on=merge_cols,
        suffixes=['_orig', '_gf'],
    ).fillna(1)

    # Add growth back on
    for year in future_years:
        productions[year] = productions[base_year] * productions["%s_gf" % year].values

    # make sure we only have the columns we started with
    productions = productions.reindex(columns=init_index_cols)
    productions = productions.groupby(init_group_cols).sum().reset_index()

    return productions


def generate_productions(population: pd.DataFrame,
                         group_cols: List[str],
                         base_year: str,
                         future_years: List[str],
                         trip_origin: str,
                         trip_rates_path: str,
                         time_splits_path: str,
                         mean_time_splits_path: str,
                         mode_share_path: str,
                         audit_dir: str,
                         process_count: int = consts.PROCESS_COUNT
                         ) -> pd.DataFrame:
    # TODO: write generate_productions() docs
    # Init
    all_years = [base_year] + future_years
    audit_base_fname = 'yr%s_%s_production_topline.csv'

    # ## MULTIPROCESS ## #
    # Build the unchanging arguments
    unchanging_kwargs = {
        'group_cols': group_cols,
        'trip_rates_path': trip_rates_path,
        'time_splits_path': time_splits_path,
        'mean_time_splits_path': mean_time_splits_path,
        'mode_share_path': mode_share_path,
    }

    # Add in the changing kwargs
    kwargs_list = list()
    for year in all_years:
        # Build the topline output path
        audit_fname = audit_base_fname % (year, trip_origin)
        audit_out = os.path.join(audit_dir, audit_fname)

        # Get just the pop for this year
        yr_pop = population.copy().reindex(group_cols + [year], axis='columns')
        yr_pop = yr_pop.rename(columns={year: 'people'})

        # Build the kwargs for this call
        kwargs = unchanging_kwargs.copy()
        kwargs['population'] = yr_pop
        kwargs['audit_out'] = audit_out
        kwargs_list.append(kwargs)

    # Make the function calls
    yearly_productions = multiprocessing.multiprocess(
        merge_pop_trip_rates,
        kwargs=kwargs_list,
        process_count=process_count,
        in_order=True
    )

    # Stick into a dict, ready to recombine
    yr_ph = {y: p for y, p in zip(all_years, yearly_productions)}

    # Join all productions into one big matrix
    # TODO: Convert code to use du.compile_efficient_df()
    productions = du.combine_yearly_dfs(
        yr_ph,
        unique_col='trips'
    )

    return productions


# BACKLOG: Point code to here get_production_time_split()
#  this is more updated and flexible
#  labels: demand merge, EFS
def get_production_time_split(productions,
                              non_split_cols,
                              tp_col: str = 'tp',
                              data_cols: List[str] = None,
                              ) -> pd.DataFrame:
    """
    # TODO: Write get_production_time_split() docs

    Parameters
    ----------
    productions
    non_split_cols
    tp_col
    data_cols

    Returns
    -------

    """
    # Init
    data_cols = ['trips'] if data_cols is None else data_cols

    # Figure out which columns to keep, excluding tp
    if tp_col in non_split_cols:
        seg_cols = du.list_safe_remove(non_split_cols, [tp_col])
    else:
        seg_cols = non_split_cols.sopy()

    # Get the totals per zone
    group_cols = seg_cols
    index_cols = group_cols.copy() + data_cols

    p_totals = productions.reindex(columns=index_cols)
    p_totals = p_totals.groupby(group_cols).sum().reset_index()

    # Get tp splits per zone
    group_cols = seg_cols + [tp_col]
    index_cols = group_cols.copy() + data_cols

    tp_totals = productions.reindex(columns=index_cols)
    tp_totals = tp_totals.groupby(group_cols).sum().reset_index()

    # Avoid name clashes on merge
    rename_cols = {col: '%s_total' % col for col in data_cols}
    p_totals = p_totals.rename(columns=rename_cols)

    time_splits = pd.merge(
        tp_totals,
        p_totals,
        how='left',
        on=seg_cols,
    )

    # Calculate time splits per zone for each data col
    for col in data_cols:
        time_splits[col] /= time_splits['%s_total' % col]
        time_splits = time_splits.drop(columns=['%s_total' % col])

    return time_splits
