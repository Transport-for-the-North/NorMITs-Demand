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
import pandas as pd
import tqdm

# Local Imports
# pylint: disable=import-error
from normits_demand import reports
from normits_demand import core as nd_core

from normits_demand.utils import file_ops
from normits_demand.utils import translation
from normits_demand.utils import pandas_utils as pd_utils
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
        zoning_system=zoning_system
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
) -> Tuple[pd.DataFrame, pd.DataFrame]:
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

    # Add in segment cols
    for key, val in segment_params.items():
        long_sector_matrix[key] = val

    # Re-order cols
    col_order = (
        [index_col_name, columns_col_name]
        + list(segment_params.keys())
        + [val_col_name]
    )
    long_sector_matrix = long_sector_matrix.reindex(columns=col_order)

    return sector_matrix, long_sector_matrix


def generate_excel_sector_report(
    sector_report_data: templates.DistributionModelMatrixReportSectorData,
    output_path: pathlib.Path,
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

    print(sector_report_data.sector_data)

    # Add sector data to the report
    pd_utils.append_df_to_excel(
        df=sector_report_data.sector_data,
        path=output_path,
        sheet_name='sector_data',
        index=False,
        header=True,
        keep_data_validation=True,
    )


def generate_matrix_reports(
    matrix_dir: pathlib.Path,
    report_dir: pathlib.Path,
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

    desc = "Generating PA Reports"
    for segment_params in tqdm.tqdm(matrix_segmentation, desc=desc):
        # Read in the matrix
        segment_fname = matrix_segmentation.generate_file_name_from_template(
            template=matrix_fname_template,
            segment_params=segment_params,
        )
        path = matrix_dir / segment_fname
        matrix = file_ops.read_df(path, find_similar=True, index_col=0)

        # Summarise into trip ends
        # row_summary, col_summary = matrix_to_trip_ends(
        #     matrix=matrix,
        #     segment_params=segment_params,
        # )
        # trip_end_rows.append(row_summary)
        # trip_end_cols.append(col_summary)

        # Summarise as sectors
        sector_matrix, long_sector_matrix = matrix_sector_summary(
            matrix=matrix,
            segment_params=segment_params,
            matrix_zoning_system=matrix_zoning_system,
            index_col_name=row_name,
            columns_col_name=col_name,
            val_col_name=val_col_name,
        )
        long_sector_matrices.append(long_sector_matrix)

        # Export sector matrix
        sector_fname = segment_fname.replace('synthetic_pa', 'ca_sector')
        sector_matrix.to_csv(sector_matrix_dir / sector_fname)

    # ## GENERATE TRIP END REPORTS ## #
    # kwargs = {
    #     "segmentation": matrix_segmentation,
    #     "zoning_system": matrix_zoning_system,
    #     "output_dir": trip_end_report_dir,
    # }
    #
    # trip_end = pd.concat(trip_end_rows, ignore_index=True)
    # write_trip_end_sector_reports(trip_end, fname_prefix=row_name, **kwargs)  # type: ignore
    #
    # trip_end = pd.concat(trip_end_cols, ignore_index=True)
    # write_trip_end_sector_reports(trip_end, fname_prefix=col_name, **kwargs)  # type: ignore

    # ## GENERATE EXCEL SECTOR REPORT ## #
    sector_report_data = templates.DistributionModelMatrixReportSectorData(
        sector_data=pd.concat(long_sector_matrices, ignore_index=True),
        from_zone_col=row_name,
        to_zone_col=col_name,
        val_col=val_col_name,
    )

    generate_excel_sector_report(
        sector_report_data=sector_report_data,
        output_path=sector_report_path,
    )
