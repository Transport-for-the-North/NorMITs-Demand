# -*- coding: utf-8 -*-
"""
Created on: 13/04/2022
Updated on:

Original author: Ben Taylor
Last update made by:
Other updates made by:

File purpose:
Functions to generate off-the-shelf matrix reports
"""
# Built-Ins
import pathlib

from typing import Any
from typing import Dict
from typing import Tuple

# Third Party
import numpy as np
import pandas as pd
import tqdm
import operator

# Local Imports
# pylint: disable=import-error
from normits_demand import constants as nd_constants
from normits_demand import reports
from normits_demand import core as nd_core

from normits_demand.utils import file_ops
from normits_demand.utils import translation
from normits_demand.utils import pandas_utils as pd_utils
from normits_demand.cost import utils as cost_utils
from normits_demand.reports import templates
# pylint: enable=import-error


def matrix_to_trip_ends(
    matrix: pd.DataFrame,
    segment_params: Dict[str, Any],
    zoning_name: str = "zone",
    val_col_name: str = "val",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Convert a matrix into trip ends

    Parameters
    ----------
    matrix:
        The matrix to convert. Columns and Index should be the zoning system

    segment_params:
            A dictionary of {segment_name: segment_value}.

    zoning_name:
        The name to give to the zoning system column in the return dataframes.

    val_col_name:
        The name to give to the value column in the return dataframes.


    Returns
    -------

    """
    # Convert segment dictionary into column names and vals
    col_names, col_vals = zip(*segment_params.items())

    def summarise(axis: int):
        vector = matrix.sum(axis=axis).to_frame(name=val_col_name)
        vector = pd_utils.prepend_cols(vector, col_names=col_names, col_vals=col_vals)
        vector.index.name = zoning_name
        return vector.reset_index()

    return summarise(axis=1), summarise(axis=0)


def write_trip_end_sector_reports(
    trip_end: pd.DataFrame,
    segmentation: nd_core.SegmentationLevel,
    zoning_system: nd_core.ZoningSystem,
    output_dir: pathlib.Path,
    fname_prefix: str = None,
) -> None:
    """Write out standard vector reports based on `trip_end`

    Parameters
    ----------
    trip_end:
        The trip end to generate the vector reports for

    segmentation:
        The segmentation of `trip_end`. Will be passed into DVector
        constructor alongside `trip_end` and `zoning_system`.

    zoning_system:
        The zoning system of `trip_end`. Will be passed into DVector
        constructor alongside `trip_end` and `segmentation`.

    output_dir:
        The directory to write out the reports to.

    fname_prefix:
        The name to prefix onto each of the report names to make them unique.
        Useful when writing multiple trip end reports out to the same
        directory.

    Returns
    -------
    None
    """
    # Build output fnames
    segment_totals_fname = '_segment_totals.csv'
    ca_sector_totals_fname = 'ca_sector_totals.csv'
    ie_sector_totals_fname = 'ie_sector_totals.csv'

    if fname_prefix is not None:
        segment_totals_fname = f'{fname_prefix}_{segment_totals_fname}'
        ca_sector_totals_fname = f'{fname_prefix}_{ca_sector_totals_fname}'
        ie_sector_totals_fname = f'{fname_prefix}_{ie_sector_totals_fname}'

    # Translate into a DVector and generate reports
    dvec = nd_core.DVector(
        import_data=trip_end,
        segmentation=segmentation,
        zoning_system=zoning_system,
        time_format='avg_day'
    )

    dvec.write_sector_reports(
        segment_totals_path=output_dir / segment_totals_fname,
        ca_sector_path=output_dir / ca_sector_totals_fname,
        ie_sector_path=output_dir / ie_sector_totals_fname,
    )


def matrix_sector_summary(
    matrix: pd.DataFrame,
    segment_params: Dict[str, Any],
    matrix_zoning_system: nd_core.ZoningSystem,
    index_col_name: str = "sector_row",
    columns_col_name: str = "sector_column",
    val_col_name: str = "val",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate sector summaries of given matrix

    Parameters
    ----------
    matrix:
        The matrix to generate the sector summaries for


    segment_params:
        A dictionary of {segment_name: segment_value}.

    matrix_zoning_system:
        The zoning system of the given `matrix`.

    index_col_name:
        The name to give to the column in `long_sector_matrix` that was
        the index of the `sector_matrix`.

    columns_col_name:
        The name to give to the column in `long_sector_matrix` that was
        the column names of the `sector_matrix`.

    val_col_name:
        The name to give to the value column in the return `long_sector_matrix`.

    Returns
    -------
    sector_matrix:
        The given `matrix` translated into sector zoning

    long_sector_matrix:
        `sector_matrix`, converted to wide format, with `segment_params`
        attached as columns.

    sector_df_intra:
        The intrazonal trips of the given matrix summed into sector zoning
        (output as tripends)
    """
    # Translate into sector zoning
    sector_zoning = nd_core.get_zoning_system("ca_sector_2020")
    sector_matrix = translation.translate_matrix_zoning(
        matrix=matrix,
        from_zoning_system=matrix_zoning_system,
        to_zoning_system=sector_zoning,
    )

    # ## CREATE A LONG VERSION, WITH SEGMENTS ATTACHED ## #
    long_sector_matrix = pd_utils.wide_to_long_infill(
        df=sector_matrix,
        index_col_1_name=index_col_name,
        index_col_2_name=columns_col_name,
        value_col_name=val_col_name,
    )

    # ## INTRAZONALS at Sector Level ##
    df_intra = np.diagonal(matrix)
    df_intra = pd.DataFrame(
        df_intra,
        index=matrix_zoning_system.unique_zones,
        columns=['val']
    )
    print(df_intra)
    print(df_intra.sum())

    # Build sector intrazonal totals
    sector_df_intra = translation.translate_vector_zoning(
        vector=df_intra,
        from_zoning_system=matrix_zoning_system,
        to_zoning_system=sector_zoning
    )

    # Add in segment cols
    for key, val in segment_params.items():
        long_sector_matrix[key] = val
        sector_df_intra[key] = val

    sector_df_intra = sector_df_intra.reset_index().rename(
        columns={'index': index_col_name})
    print(sector_df_intra.head())

    # Re-order cols
    col_order = (
        [index_col_name, columns_col_name]
        + list(segment_params.keys())
        + [val_col_name]
    )
    long_sector_matrix = long_sector_matrix.reindex(columns=col_order)

    col_order = (
        [index_col_name]
        + list(segment_params.keys())
        + [val_col_name]
    )
    sector_df_intra = sector_df_intra.reindex(columns=col_order)
    print(sector_df_intra.head())

    return sector_matrix, long_sector_matrix, sector_df_intra


def generate_excel_sector_report(
    sector_report_data: templates.DistributionModelMatrixReportSectorData,
    output_path: pathlib.Path,
    sector_intras,
    tld_dist
) -> None:
    """Generate a standard sector report

    Parameters
    ----------
    sector_report_data:
        A DistributionModelMatrixReportSectorData object which defines
        the sector data and how each columns links to the needed data.

    output_path:
        The full path, including filename, to output the completed report
        to.

    Returns
    -------
    None
    """
    # Init
    report_template = reports.ReportTemplates.DISTRIBUTION_MODEL_MATRIX
    report_template.copy_report_template(output_path)

    # Add sector data to the report
    pd_utils.append_df_to_excel(
        df=sector_report_data.sector_data,
        path=output_path,
        sheet_name='sector_data',
        index=False,
        header=True,
        keep_data_validation=True,
    )

    # Add Sector Intras to excel report
    pd_utils.append_df_to_excel(
        df=sector_intras.sector_data,
        path=output_path,
        sheet_name='sector_intra_data',
        index=False,
        header=True,
        keep_data_validation=True,
    )

    # Add TLDs to excel report
    pd_utils.append_df_to_excel(
        df=tld_dist.tld_data,
        path=output_path,
        sheet_name='tld_data',
        index=False,
        header=True,
        keep_data_validation=True,
    )


def generate_matrix_reports(
    matrix_dir: pathlib.Path,
    report_dir: pathlib.Path,
    cost_matrices: Dict[str, np.ndarray],
    matrix_segmentation: nd_core.SegmentationLevel,
    matrix_zoning_system: nd_core.ZoningSystem,
    matrix_fname_template: str,
    row_name: str,
    col_name: str,
):
    # PA RUN REPORTS
    # Matrix Trip ENd totals
    #   Inter / Intra Report by segment?
    #   Aggregate segments and report again too? (CBO)
    # nd.constants.USER_CLASS_PURPOSES
    # TLD curve
    #   single mile bands - p/m (ca ) segments full matrix

    # TODO(PW, BT): Add constants such as this to the report on the fly in
    #  one of the reference sheets
    nd_constants.USER_CLASS_PURPOSES

    # Init
    val_col_name = "val"

    # Build needed paths
    sector_matrix_dir = report_dir / 'sector_matrices'
    trip_end_report_dir = report_dir / 'trip_ends'
    sector_report_path = report_dir / 'Sector_Report_Summary.xlsx'

    # Ensure paths exist
    file_ops.create_folder(sector_matrix_dir)
    file_ops.create_folder(trip_end_report_dir)

    # ## READ IN EACH MATRIX AND SUMMARISE ## #
    trip_end_rows = list()
    trip_end_cols = list()
    long_sector_matrices = list()
    sector_intras_full = list()
    tld_list = list()

    desc = "Generating PA Reports"
    for segment_params in tqdm.tqdm(matrix_segmentation, desc=desc):
        # ## TRIP END SUMMARY ## #
        segment_fname = matrix_segmentation.generate_file_name_from_template(
            template=matrix_fname_template,
            segment_params=segment_params,
        )
        path = matrix_dir / segment_fname
        matrix = file_ops.read_df(path, find_similar=True, index_col=0)

        # Summarise into trip ends
        # TODO Add ie to excel template
        row_summary, col_summary = matrix_to_trip_ends(
            matrix=matrix,
            segment_params=segment_params,
        )
        trip_end_rows.append(row_summary)
        trip_end_cols.append(col_summary)

        # ## SECTOR SUMMARY ## #
        sector_matrix, long_sector_matrix, sector_df_intra = matrix_sector_summary(
            matrix=matrix,
            segment_params=segment_params,
            matrix_zoning_system=matrix_zoning_system,
            index_col_name=row_name,
            columns_col_name=col_name,
            val_col_name=val_col_name,
        )

        long_sector_matrices.append(long_sector_matrix)
        sector_intras_full.append(sector_df_intra)

        # Export sector matrix
        sector_fname = segment_fname.replace('synthetic_pa', 'ca_sector')
        sector_matrix.to_csv(sector_matrix_dir / sector_fname)

        # ## GENERATE COST DISTRIBUTION CURVES ## #
        # TODO(BT, PW): TLD code works - to be tidied
        #  ee masking is included by default - may wish to toggle this
        external_mask = pd_utils.get_wide_mask(
            df=matrix,
            zones=matrix_zoning_system.external_zones,
            join_fn=operator.and_
        )
        matrix_no_ee = matrix * ~external_mask

        segment_name = matrix_segmentation.get_segment_name(segment_params)
        bin_edges = np.array([0, 1, 2, 3, 5, 10, 15, 25, 35, 50, 100, 200, np.inf]) * 1.609344
        distribution, achieved_distribution = cost_utils.calculate_reporting_cost_distribution(
            matrix=matrix_no_ee.values,
            cost_matrix=cost_matrices[segment_name],
            bin_edges=bin_edges
        )

        tld = pd.DataFrame(
            {'lower': bin_edges[:-1],
             'upper': bin_edges[1:],
             'distribution': distribution,
             'distribution_pct': achieved_distribution
             }
        )
        print(tld)

        # Add in segment cols before class fills defaults
        for key, val in segment_params.items():
            tld[key] = val

        # Re-order cols
        col_order = (
                list(segment_params.keys())
                + ['lower', 'upper', 'distribution', 'distribution_pct']
        )
        tld = tld.reindex(columns=col_order)

        tld_list.append(tld)

    # ## GENERATE TRIP END REPORTS ## #
    kwargs = {
        "segmentation": matrix_segmentation,
        "zoning_system": matrix_zoning_system,
        "output_dir": trip_end_report_dir,
    }

    trip_end = pd.concat(trip_end_rows, ignore_index=True)
    write_trip_end_sector_reports(trip_end, fname_prefix=row_name, **kwargs)  # type: ignore

    trip_end = pd.concat(trip_end_cols, ignore_index=True)
    write_trip_end_sector_reports(trip_end, fname_prefix=col_name, **kwargs)  # type: ignore

    # ## GENERATE EXCEL SECTOR REPORT ## #
    sector_report_data = templates.DistributionModelMatrixReportSectorData(
        sector_data=pd.concat(long_sector_matrices, ignore_index=True),
        from_zone_col=row_name,
        to_zone_col=col_name,
        val_col=val_col_name,
    )
    print("sector intras full table")
    print(sector_intras_full)
    sector_intras_full = templates.DistributionModelTripEndReportData(
        sector_data=pd.concat(sector_intras_full, ignore_index=True),
        from_zone_col=row_name,
        val_col=val_col_name,
    )

    tld_full = templates.DistributionModelReportTLDData(
        tld_data=pd.concat(tld_list, ignore_index=True),
        lower_col='lower',
        upper_col='upper',
        distribution_col='distribution',
        distribution_pct_col='distribution_pct'
    )

    generate_excel_sector_report(
        sector_report_data=sector_report_data,
        output_path=sector_report_path,
        sector_intras=sector_intras_full,
        tld_dist=tld_full
    )


    # TODO: Add TLD calculation and export
