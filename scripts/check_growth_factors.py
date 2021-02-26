import os
import sys
import functools

import numpy as np
import pandas as pd
from tqdm import tqdm

# HACKY METHOD to import normits demand if it's not on the path
sys.path.append(r'C:\Users\Sneezy\Desktop\GitHub\NorMITs-Demand')
import normits_demand as nd
from normits_demand.utils import general as du
from normits_demand import efs_constants as consts
from normits_demand.models import efs_zone_translator as zt

IN_GF_PATH = r"I:\NorMITs Demand\import\default\population\ss\based on pa\future_population_growth.csv"
OUT_GF_PATH = r"E:\NorMITs Demand\noham\v0.3-EFS_Output\NTEM\iter2\Audits\Productions\msoa_hb_productions_multiplicative_growth.csv"

MSOA_TO_LAD_PATH = r"I:\NorMITs Demand\import\zone_translation\no_overlap\lad_to_msoa.csv"

IMPORT = "I:/"
EXPORT = "E:/"
MODEL_NAME = 'noham'
SCENARIO = consts.SC00_NTEM
ITER = 2
ITER_NAME = du.create_iter_name(ITER)


IMPORTS, EXPORTS, _ = du.build_efs_io_paths(
    import_location=IMPORT,
    export_location=EXPORT,
    model_name=MODEL_NAME,
    iter_name=ITER_NAME,
    scenario_name=SCENARIO,
    demand_version=nd.ExternalForecastSystem.__version__,
)

YEARS = ['2018', '2033', '2035', '2050']


def check_in_out():
    print("doing setup")
    pa_in = pd.read_csv(IN_GF_PATH)
    pa_out = pd.read_csv(OUT_GF_PATH)

    years = ['2033', '2035', '2050']
    zones = pa_in['msoa_zone_id'].unique()

    out_ph = list()

    for zone in tqdm(zones):
        mask = pa_in['msoa_zone_id'] == zone
        in_p = pa_in[mask].copy()

        mask = pa_out['msoa_zone_id'] == zone
        out_p = pa_out[mask].copy()

        out_p = out_p.groupby(['msoa_zone_id']).mean().reset_index()

        for year in years:
            print(
                "%s\t%s\t%s"
                % (in_p[year].mean(), out_p[year].mean(), zone)
            )
            if round(in_p[year].mean(), 3) != round(out_p[year].mean(), 3):
                out_ph.append({
                    'Zone': zone,
                    'Year': year,
                    'gf_in': in_p[year].mean(),
                    'gf_out': out_p[year].mean(),
                })
        print('\n')

    # Write Out
    df = pd.DataFrame(out_ph)
    path = r"E:\test.csv"
    df.to_csv(path, index=False)


def get_ntem_data():
    base_year, future_years = du.split_base_future_years([int(x) for x in YEARS])
    base_year_str = str(base_year)
    future_years_str = [str(x) for x in future_years]

    # What growth factors is NTEM getting at lad?
    ntem_ph = list()
    ntem_mzc = 'lad_zone_id'
    for year in YEARS:
        ntem_fname = consts.NTEM_CONTROL_FNAME % ('pa', year)
        ntem_path = os.path.join(IMPORTS['ntem_control'], ntem_fname)
        ntem = pd.read_csv(ntem_path)

        group_cols = [ntem_mzc]
        needed_cols = group_cols.copy() + ['productions', 'attractions']

        ntem = ntem.reindex(columns=needed_cols)
        ntem = ntem.groupby(group_cols).sum().reset_index()

        col_rename = {
            'productions': 'p_%s' % year,
            'attractions': 'a_%s' % year,
        }
        ntem = ntem.rename(columns=col_rename)
        ntem_ph.append(ntem)

    ntem = functools.reduce(lambda x, y: pd.merge(x, y, on=ntem_mzc), ntem_ph)
    ntem_gf = ntem.copy()

    for col_name in ['p_%s', 'a_%s']:
        for year in reversed(YEARS):
            ntem_gf[col_name % year] /= ntem_gf[col_name % base_year_str]

    # split ntem into p and a
    cols = du.list_safe_remove(list(ntem), [ntem_mzc])
    p_cols = [ntem_mzc] + [x for x in cols if 'p' in x]
    a_cols = [ntem_mzc] + [x for x in cols if 'a' in x]

    p_rename = {'p_%s' % x: x for x in YEARS}
    a_rename = {'a_%s' % x: x for x in YEARS}

    p_ntem = ntem.reindex(columns=p_cols).rename(columns=p_rename)
    p_gf_ntem = ntem_gf.reindex(columns=p_cols).rename(columns=p_rename)
    a_ntem = ntem.reindex(columns=a_cols).rename(columns=a_rename)
    a_gf_ntem = ntem_gf.reindex(columns=a_cols).rename(columns=a_rename)

    return p_ntem, p_gf_ntem, a_ntem, a_gf_ntem


def get_efs_p_data():
    base_year, future_years = du.split_base_future_years([int(x) for x in YEARS])
    base_year_str = str(base_year)
    future_years_str = [str(x) for x in future_years]

    # EFS
    # What growth factors is EFS getting at LAD?
    efs_gf = pd.read_csv(OUT_GF_PATH)

    efs_mzc = 'msoa_zone_id'
    group_cols = [efs_mzc]
    needed_cols = group_cols.copy() + future_years_str

    efs_gf = efs_gf.reindex(columns=needed_cols)
    efs_gf = efs_gf.groupby(group_cols).mean().reset_index()

    # Translate to lad
    t_df = pd.read_csv(MSOA_TO_LAD_PATH)
    trans = zt.ZoneTranslator()
    efs_gf = trans.run(
        efs_gf,
        from_zoning='msoa',
        to_zoning='lad',
        non_split_cols=group_cols,
        translation_df=t_df,
        aggregate_method='mean'
    )

    return efs_gf


def get_efs_p_in_data():
    # EFS
    # What growth factors is EFS getting at LAD?
    efs_gf = pd.read_csv(IN_GF_PATH)

    efs_mzc = 'msoa_zone_id'

    # Translate to lad
    t_df = pd.read_csv(MSOA_TO_LAD_PATH)
    trans = zt.ZoneTranslator()
    efs_gf = trans.run(
        efs_gf,
        from_zoning='msoa',
        to_zoning='lad',
        non_split_cols=[efs_mzc],
        translation_df=t_df,
        aggregate_method='mean'
    )

    return efs_gf


def check_application():
    base_year, future_years = du.split_base_future_years([int(x) for x in YEARS])
    base_year_str = str(base_year)
    future_years_str = [str(x) for x in future_years]

    # NTEM
    p_ntem, p_gf_ntem, a_ntem, a_gf_ntem = get_ntem_data()

    # EFS
    p_gf_efs = get_efs_p_data()

    p_gf_in_efs = get_efs_p_in_data()

    mzc = 'lad_zone_id'
    for df in [p_gf_ntem, p_gf_efs, p_gf_in_efs]:
        df.set_index([mzc], inplace=True)

    ntem_gf_in_diff = np.absolute((p_gf_ntem - p_gf_in_efs).values)
    ntem_gf_out_diff = np.absolute((p_gf_ntem - p_gf_efs).values)
    efs_gf_in_diff = np.absolute((p_gf_efs - p_gf_in_efs).values)

    print("NTEM/GF_IN total diff:\t%s" % str(ntem_gf_in_diff.sum(axis=0)))
    print("NTEM/GF_IN avg diff:\t%s" % str(ntem_gf_in_diff.mean(axis=0)))
    print()
    print("NTEM/GF_OUT total diff:\t%s" % str(ntem_gf_out_diff.sum(axis=0)))
    print("NTEM/GF_OUT avg diff:\t%s" % str(ntem_gf_out_diff.mean(axis=0)))
    print()
    print("GF_IN/GF_OUT total diff:\t%s" % str(efs_gf_in_diff.sum(axis=0)))
    print("GF_IN/GF_OUT total diff:\t%s" % str(efs_gf_in_diff.mean(axis=0)))



def main():
    # check_in_out()
    check_application()


if __name__ == '__main__':
    main()
