
import os

from typing import List

import pandas as pd

from tqdm import tqdm

import normits_demand as nd
from normits_demand import constants as consts
from normits_demand.utils import general as du
from normits_demand.matrices import matrix_processing as mat_p


def get_internal_vectors(model_names: List[str],
                         scenario_names: List[str],
                         iter_num,
                         years,
                         import_home,
                         export_home,
                         cache_drive,
                         agg_cols=None
                         ):

    return_dict = dict()

    vector_order = [
        'hb_productions',
        'nhb_productions',
        'hb_attractions',
        'nhb_attractions',
    ]

    for model_name in model_names:
        model_zone_col = "%s_zone_id" % model_name

        model_dict = dict()
        for scenario_name in scenario_names:
            efs = nd.EfsReporter(
                iter_num=iter_num,
                model_name=model_name,
                scenario_name=scenario_name,
                years_needed=years,
                import_home=import_home,
                export_home=export_home,
            )

            cache_path = os.path.join(cache_drive, 'cache', model_name, scenario_name)
            du.create_folder(cache_path)

            # Read in internal P/A Vectors
            vectors = mat_p.maybe_convert_matrices_to_vector(
                mat_import_dir=efs.imports['matrices']['pa_24'],
                years_needed=efs.years_needed,
                cache_path=cache_path,
                matrix_format='pa',
                model_zone_col=model_zone_col,
                internal_zones=efs.model_internal_zones,
            )

            # Aggregate to needed data
            if agg_cols is not None:
                group_cols = [model_zone_col] + agg_cols
                index_cols = group_cols.copy() + years

                vector_list = list()
                for vector in vectors:
                    vector = vector.reindex(columns=index_cols)
                    vector = vector.groupby(group_cols).sum().reset_index()
                    vector_list.append(vector)

                vectors = vector_list

            vectors = {name: vec for name, vec in zip(vector_order, vectors)}

            model_dict[scenario_name] = vectors

        return_dict[model_name] = model_dict

    return return_dict


def translate_noham_to_norms(vector_dict,
                             pop_translation,
                             emp_translation,
                             split_cols,
                             ):
    # init
    vector_dict = vector_dict.copy()

    # Set up the zone translator
    zt = nd.ZoneTranslator()

    # Translate all the Noham vectors into norms zones
    noham_vectors = vector_dict['noham'].copy()

    for scenario in noham_vectors.keys():
        new_vectors = dict()

        for name, vector in noham_vectors[scenario].items():

            # Determine the translation to use
            if 'productions' in name:
                tdf = pop_translation
            elif 'attractions' in name:
                tdf = emp_translation
            else:
                raise ValueError("TRANSLATION PROBLEM")

            non_split_cols = du.list_safe_remove(list(vector), split_cols)
            new_vectors[name] = zt.run(
                dataframe=vector,
                translation_df=tdf,
                from_zoning='noham',
                to_zoning='norms',
                non_split_cols=non_split_cols,
            )

        vector_dict['noham'][scenario] = new_vectors

    return vector_dict


def translate_area_type_to_norms(df):
    # Grab the translation
    msoa2norms = du.get_zone_translation(
        import_dir=r'I:\NorMITs Demand\import\zone_translation\one_to_one',
        from_zone='msoa',
        to_zone='norms'
    )

    # Get counts of area types in norms
    df['msoa_zone_id'] = df['msoa_zone_id'].map(msoa2norms)
    df['count'] = 1
    df['count'] = df.groupby(['msoa_zone_id', 'area_type']).transform('sum')
    df = df.drop_duplicates().dropna()

    # Translate to norms area types - taking most popular
    new = list()
    for zone_num in df['msoa_zone_id'].unique():
        zone_vals = df[df['msoa_zone_id']==zone_num].copy()

        area_type = zone_vals.loc[zone_vals['count'].idxmax(), 'area_type']
        new.append({
            'norms_zone_id': int(zone_num),
            'area_type': int(area_type),
        })

    return pd.DataFrame(new)


def get_norms_area_types_per_year(years,
                                  scenario_names,
                                  lu_drive,
                                  land_use_iter,
                                  ):
    # Init
    needed_cols = ['msoa_zone_id', 'area_type']
    base_lu_path = os.path.join(*[
        lu_drive,
        'NorMITs Land Use',
        land_use_iter,
        'outputs',
    ])

    # First get the 2018 Land Use - this never changes per scenario
    fname = 'land_use_output_safe_msoa.csv'
    path = os.path.join(base_lu_path, fname)
    base_at = pd.read_csv(path, usecols=needed_cols)
    base_at = base_at.drop_duplicates().reset_index(drop=True)
    base_at = translate_area_type_to_norms(base_at)

    # progress bar
    p_bar = tqdm(
        desc="Getting area types",
        total=len(scenario_names) * (len(years) - 1),
    )

    # Get all the years and scenarios
    area_types = dict()
    for scenario in scenario_names:
        scenario_dict = {years[0]: base_at}

        for year in years[1:]:
            fname = 'land_use_%s_pop.csv' % year
            path = os.path.join(base_lu_path, 'scenarios', scenario, fname)

            year_at = pd.read_csv(path, usecols=needed_cols)
            year_at = year_at.drop_duplicates().reset_index(drop=True)
            year_at = translate_area_type_to_norms(year_at)

            scenario_dict[year] = year_at
            p_bar.update(1)

        area_types[scenario] = scenario_dict

    p_bar.close()
    return area_types


def join_norms_noham(vectors_dict, scenario_names, years):
    # init
    joined_vectors = dict()

    # Aggregate purposes dict
    p_dict = {
        'commute': [1],
        'business': [2, 12],
        'other': [3, 4, 5, 6, 7, 8, 13, 14, 15, 16, 18]
    }

    def fn(x):
        setted = False
        for k, v in p_dict.items():
            if x['p'] in v:
                x['p'] = k
                setted = True
                break

        if not setted:
            raise ValueError("I no work!")

        return x

    # progress bar
    p_bar = tqdm(
        desc="Combining norms and noham",
        total=len(scenario_names) * 4,
    )

    for scenario in scenario_names:
        scenario_dict = dict()
        vector_types = list(vectors_dict['noham'][scenario].keys())

        for v_type in vector_types:
            ph = list()
            # Aggregate away purposes
            for model_name in ['noham', 'norms']:
                vec = vectors_dict[model_name][scenario][v_type].copy()
                vec = vec.apply(fn, axis='columns')

                group_cols = du.list_safe_remove(list(vec), years)
                vec = vec.groupby(group_cols).sum().reset_index()
                ph.append(vec)

            scenario_dict[v_type] = pd.concat(ph)
            p_bar.update(1)

        joined_vectors[scenario] = scenario_dict

    p_bar.close()
    return joined_vectors


def translate_to_tfn_sectors(vectors_dict,
                             area_types,
                             scenario_names,
                             years,
                             pop_translation,
                             emp_translation,
                             ):
    # Init
    return_dict = dict()
    external_sectors = [17, 18, 19, 20, 21, 22, 23, 24]

    # Set up the zone translator
    zt = nd.ZoneTranslator()

    # progress bar
    p_bar = tqdm(
        desc="Converting to tfn sectors",
        total=len(scenario_names) * 4,
    )

    for scenario in scenario_names:
        scenario_dict = dict()
        vector_types = list(vectors_dict[scenario].keys())

        # Attach area types to Vectors
        for v_type in vector_types:
            vec = vectors_dict[scenario][v_type].copy()
            group_cols = du.list_safe_remove(list(vec), years)

            # Determine the translation to use
            if 'productions' in v_type:
                tdf = pop_translation
            elif 'attractions' in v_type:
                tdf = emp_translation
            else:
                raise ValueError("TRANSLATION PROBLEM")

            year_ph = list()
            for year in years:
                at = area_types[scenario][year]

                # split out year
                year_vec = vec.reindex(columns=group_cols + [year])
                year_vec = pd.melt(
                    year_vec,
                    id_vars=group_cols,
                    var_name='year',
                    value_name='trips',
                )

                # attach area types
                year_vec = pd.merge(
                    year_vec,
                    at,
                    how='left',
                    on='norms_zone_id'
                )

                # Convert to TfN sectors
                non_split_cols = du.list_safe_remove(list(year_vec), ['trips'])
                year_vec = zt.run(
                    dataframe=year_vec,
                    translation_df=tdf,
                    from_zoning='norms',
                    to_zoning='ca_sector_2020',
                    non_split_cols=non_split_cols,
                )

                # Drop external sectors
                mask = year_vec['ca_sector_2020_zone_id'].astype(int).isin(external_sectors)
                year_vec = year_vec[~mask]
                year_ph.append(year_vec)

            scenario_dict[v_type] = pd.concat(year_ph)
            p_bar.update(1)

        return_dict[scenario] = scenario_dict

    p_bar.close()
    return return_dict


def join_scenarios(vector_dict, scenario_names):
    # Init
    vector_types = list(vector_dict[scenario_names[0]].keys())

    v_type_ph = list()
    for v_type in vector_types:
        for scenario in scenario_names:
            vec = vector_dict[scenario][v_type]
            vec['scenario'] = scenario
            vec['type'] = v_type
            v_type_ph.append(vec)

    return pd.concat(v_type_ph)


def main():

    # ## SETUP ## #
    trans_home = r'I:\NorMITs Demand\import\zone_translation\weighted'
    pop_translation = pd.read_csv(os.path.join(trans_home, 'norms_noham_pop_weighted_lookup.csv'))
    emp_translation = pd.read_csv(os.path.join(trans_home, 'norms_noham_emp_weighted_lookup.csv'))

    sec_pop_trans = pd.read_csv(os.path.join(trans_home, 'norms_ca_sector_2020_pop_weighted.csv'))
    sec_emp_trans = pd.read_csv(os.path.join(trans_home, 'norms_ca_sector_2020_emp_weighted.csv'))

    lu_iter = 'iter3b'
    lu_drive = 'Y:/'

    # Controls I/O
    scenario_names = consts.TFN_SCENARIOS
    iter_num = '3g'
    import_home = "I:/"
    export_home = "I:/"
    cache_drive = "E:/"
    model_names = ['noham', 'norms']

    years = ['2018', '2033', '2035', '2050']

    # Read in internal P/A Vectors
    print("Reading in vectors...")
    vectors_dict = get_internal_vectors(
        model_names,
        scenario_names,
        iter_num,
        years,
        import_home,
        export_home,
        cache_drive,
        agg_cols=['p', 'm'],
    )

    # Translate noham to norms system
    print("Translating to norms zoning...")
    vectors_dict = translate_noham_to_norms(
        vectors_dict,
        pop_translation,
        emp_translation,
        split_cols=years,
    )

    vectors_dict = join_norms_noham(
        vectors_dict,
        scenario_names,
        years=years,
    )

    # Get norms AT per zone per year
    print("Getting area_types for norms...")
    area_types = get_norms_area_types_per_year(
        years=years,
        scenario_names=scenario_names,
        lu_drive=lu_drive,
        land_use_iter=lu_iter,
    )

    # Translate P/A Vectors to TfN Sectors
    print("Converting to TfN Sectors...")
    sector_vectors_dict = translate_to_tfn_sectors(
        vectors_dict,
        area_types,
        scenario_names,
        years,
        sec_pop_trans,
        sec_emp_trans,
    )

    # Join vector types
    print("Joining scenarios together...")
    type_dict = join_scenarios(
        sector_vectors_dict,
        scenario_names,
    )


if __name__ == '__main__':
    main()
