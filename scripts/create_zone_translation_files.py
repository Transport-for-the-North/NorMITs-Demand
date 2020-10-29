# -*- coding: utf-8 -*-
"""
Created on: Fri October 16 15:29:12 2020
Updated on:

Original author: Ben Taylor
Last update made by:
Other updates made by:

File purpose:
Converts outputs from the zone_translation files into a simpler format for
converting from norms/noham zones to LA or TfN Sector
"""
import os

import pandas as pd

from collections import defaultdict


# Global variables
OVERLAP_TOL = 0.500000000001

NOHAM2LAD_FNAME = 'noham_lad_pop_weighted_lookup.csv'
NOHAM2MSOA_FNAME = 'msoa_noham_pop_weighted_lookup.csv'
NOHAM2TFN_FNAME = 'tfn_sector_noham_pop_weighted_lookup.csv'

NORMS2LAD_FNAME = 'norms_2015_lad_pop_weighted_lookup.csv'
NORMS2MSOA_FNAME = 'norms_2015_msoa_pop_weighted_lookup.csv '
NORMS2TFN_FNAME = 'tfn_sector_norms_2015_pop_weighted_lookup.csv '

MSOA2TFN_FNAME = 'tfn_sector_msoa_pop_weighted_lookup.csv'

# No longer used - got newer translation files
def noham_to_tfn_via_msoa(in_dir, out_path):
    noham2msoa = pd.read_csv(os.path.join(in_dir, NOHAM2MSOA_FNAME))
    msoa2tfn = pd.read_csv(os.path.join(in_dir, MSOA2TFN_FNAME))

    # Translate noham to msoa
    noham_col = 'noham_zone_id'
    msoa_col = 'msoa_zone_id'
    overlap_col = 'overlap_msoa_split_factor'
    needed_cols = [noham_col, msoa_col, overlap_col]
    noham2msoa = noham2msoa.reindex(needed_cols, axis='columns')

    noham2msoa, min_overlap = create_translation_dataframe(
        file=noham2msoa,
        model_col=noham_col,
        aggregate_col=msoa_col,
        overlap_col=overlap_col,
        keep_overlap=True
    )
    noham2msoa_split = 'noham2msoa_split'
    noham2msoa = noham2msoa.rename(columns={overlap_col: noham2msoa_split})
    print("Noham to MSOA min overlap: %.4f" % min_overlap)

    # Translate msoa to Tfn
    tfn_col = 'tfn_sectors_zone_id'
    overlap_col = 'overlap_msoa_split_factor'
    needed_cols = [tfn_col, msoa_col, overlap_col]
    msoa2tfn = msoa2tfn.reindex(needed_cols, axis='columns')

    msoa2tfn, min_overlap = create_translation_dataframe(
        file=msoa2tfn,
        model_col=msoa_col,
        aggregate_col=tfn_col,
        overlap_col=overlap_col,
        keep_overlap=True
    )
    msoa2tfn_split = 'msoa2tfn_split'
    msoa2tfn = msoa2tfn.rename(columns={overlap_col: msoa2tfn_split})
    print("MSOA to TfN min overlap: %.4f" % min_overlap)

    noham2tfn = pd.merge(
        noham2msoa,
        msoa2tfn,
        on=msoa_col,
        how='outer'
    )
    print('NaN counts:')
    print('\t noham %d' % noham2tfn[noham_col].isnull().sum())
    print('\t msoa %d' % noham2tfn[msoa_col].isnull().sum())
    print('\t tfn %d' % noham2tfn[tfn_col].isnull().sum())

    # Get rid of unneeded stuff
    noham2tfn = noham2tfn.dropna()
    noham2tfn = noham2tfn.drop(msoa_col, axis='columns')

    # ## Deal with duplicate rows ## #
    # Keep a count of overlaps
    group_cols = [noham_col, tfn_col]
    noham2tfn = noham2tfn.groupby(group_cols).sum().reset_index()

    # If duplicates remain - deal with them
    if len(noham2tfn) != len(noham2tfn[noham_col].unique()):
        # Loop init
        unq_zones = noham2tfn[noham_col].unique()
        new_df = defaultdict(list)

        # Only keep duplicates with highest overlap
        for zone in unq_zones:
            # Get relevant zones
            zone_subset = noham2tfn[noham2tfn[noham_col] == zone].copy()

            # Need to pick the row with the biggest overlap
            if len(zone_subset) > 1:
                zone_subset = zone_subset.loc[zone_subset[noham2msoa_split].idxmax()]
            else:
                # Turn single row df into series for consistent access
                zone_subset = zone_subset.iloc[0]

            new_df[noham_col].append(zone)
            new_df[tfn_col].append(zone_subset[tfn_col])

        noham2tfn = pd.DataFrame(new_df)

    # Reindex and write to disk
    noham2tfn = noham2tfn.reindex(group_cols, axis='columns')
    noham2tfn.to_csv(out_path, index=False)


def norms_to_tfn(in_dir, out_path):
    # Should Norms should nest nicely into TfN? Take biggest always
    # Min overlap = 0.6455
    file = pd.read_csv(os.path.join(in_dir, NORMS2TFN_FNAME))

    # get just the needed columns
    norms_col = 'old_norms_zone_id'
    tfn_col = 'tfn_sectors_zone_id'
    overlap_col = 'overlap_old_norms_split_factor'
    needed_cols = [norms_col, tfn_col, overlap_col]
    file = file.reindex(needed_cols, axis='columns')

    # Output columns name
    norms_out = 'norms_zone_id'

    # Output to file
    create_translation_dataframe(
        file=file,
        model_col=norms_col,
        aggregate_col=tfn_col,
        overlap_col=overlap_col,
        out_path=out_path,
        model_col_out=norms_out
    )


def noham_to_tfn(in_dir, out_path):
    # Should Norms should nest nicely into TfN? Take biggest always
    # Min overlap = 0.6545
    file = pd.read_csv(os.path.join(in_dir, NOHAM2TFN_FNAME))

    # get just the needed columns
    noham_col = 'noham_zone_id'
    tfn_col = 'tfn_sectors_zone_id'
    overlap_col = 'overlap_noham_split_factor'
    needed_cols = [noham_col, tfn_col, overlap_col]
    file = file.reindex(needed_cols, axis='columns')

    # Output to file
    create_translation_dataframe(
        file=file,
        model_col=noham_col,
        aggregate_col=tfn_col,
        overlap_col=overlap_col,
        out_path=out_path
    )


def norms_to_lad(in_dir, out_path):
    # Should Norms should nest nicely into LAD? Take biggest always
    # Min overlap = 0.5082
    file = pd.read_csv(os.path.join(in_dir, NORMS2LAD_FNAME))

    # get just the needed columns
    norms_col = 'norms_2015_zone_id'
    lad_col = 'lad_zone_id'
    overlap_col = 'overlap_lad_split_factor'
    needed_cols = [norms_col, lad_col, overlap_col]
    file = file.reindex(needed_cols, axis='columns')
    
    # Output columns name
    norms_out = 'norms_zone_id'

    # Output to file
    create_translation_dataframe(
        file=file,
        model_col=norms_col,
        aggregate_col=lad_col,
        overlap_col=overlap_col,
        out_path=out_path,
        model_col_out=norms_out
    )


def noham_to_lad(in_dir, out_path):
    # Should Noham should nest nicely into LAD? Take biggest always
    # Min overlap = 0.5108
    file = pd.read_csv(os.path.join(in_dir, NOHAM2LAD_FNAME))

    # get just the needed columns
    noham_col = 'noham_zone_id'
    lad_col = 'lad_zone_id'
    overlap_col = 'overlap_noham_split_factor'
    needed_cols = [noham_col, lad_col, overlap_col]
    file = file.reindex(needed_cols, axis='columns')

    # Output to file
    create_translation_dataframe(
        file=file,
        model_col=noham_col,
        aggregate_col=lad_col,
        overlap_col=overlap_col,
        out_path=out_path
    )


def create_translation_dataframe(file,
                                 model_col,
                                 aggregate_col,
                                 overlap_col,
                                 keep_overlap=False,
                                 out_path=None,
                                 model_col_out=None,
                                 aggregate_col_out=None
                                 ):
    # Create the output lines
    model2agg = defaultdict(list)
    min_overlap = 1
    for _, row in file.iterrows():
        # Unless overlap is less than tol, ignore
        if OVERLAP_TOL > row[overlap_col] > 1 - OVERLAP_TOL:
            raise ValueError("Got a row outside of the overlap tolerance. "
                             "Row overlap: %.3f" % row[overlap_col])

        # Only keep the rows that above the tolerance
        if row[overlap_col] < OVERLAP_TOL:
            continue

        # Update minimum overlap found
        if row[overlap_col] > 0.5:
            min_overlap = min(min_overlap, row[overlap_col])

        # Save the row
        model2agg[model_col].append(row[model_col])
        model2agg[aggregate_col].append(row[aggregate_col])
        if keep_overlap:
            model2agg[overlap_col].append(row[overlap_col])

    # Output
    df = pd.DataFrame(model2agg)
    if model_col_out is not None:
        df = df.rename(columns={model_col: model_col_out})

    if aggregate_col_out is not None:
        df = df.rename(columns={aggregate_col: aggregate_col_out})

    if out_path is not None:
        df.to_csv(out_path, index=False)
        print("%s min overlap: %.4f" % (out_path, min_overlap))

    return df, min_overlap


def main():
    out_dir = r'Y:\NorMITs Demand\import\zone_translation'
    in_dir = r'Y:\NorMITs Demand\import\zone_translation\src'

    base_fname = '%s_to_%s.csv'

    # Create all translation files
    noham_to_lad(
        in_dir=in_dir,
        out_path=os.path.join(out_dir, base_fname % ('noham', 'lad')),
    )
    noham_to_tfn(
        in_dir=in_dir,
        out_path=os.path.join(out_dir, base_fname % ('noham', 'tfn_sectors')),
    )

    norms_to_lad(
        in_dir=in_dir,
        out_path=os.path.join(out_dir, base_fname % ('norms', 'lad')),
    )
    norms_to_tfn(
        in_dir=in_dir,
        out_path=os.path.join(out_dir, base_fname % ('norms', 'tfn_sectors')),
    )

    pass


if __name__ == '__main__':
    main()