import os
import re
from glob import glob
from typing import List, Union
from itertools import product

import pandas as pd
from tqdm import tqdm

from demand_utilities.sector_reporter_v2 import SectorReporter
from matrix_processing import aggregate_matrices
import efs_constants as consts


# Define the possible values for trip origin type - 
# homebased or non-homebased
VALID_TRIP_ORIGIN = ["hb", "nhb"]


def matrix_reporting(matrix_directory: str,
                     output_dir: str,
                     trip_origin: str,
                     matrix_format: str,
                     segments_needed: dict = {},
                     zone_file: str = None,
                     sectors_file: str = None,
                     sectors_name: str = "Sectors",
                     aggregation_method: str = "sum",
                     overwrite_dir: bool = True):
    """
    TODO: write documentataion
    Options to aggregate any matrix segment.
    Either select the segment names to keep or supply "Agg"
    to keep disaggregated or "Keep" to aggregate this field

    Option to aggregate sectors

    segments_needed can have the keys:
        years, p, m, soc, ns, ca, tp
        where each is a list of the required segments

    """

    success = True

    # Check Inputs are valid
    if trip_origin not in VALID_TRIP_ORIGIN:
        raise ValueError(f"{trip_origin} is not a valid option")
    if matrix_format not in consts.VALID_MATRIX_FORMATS:
        raise ValueError(f"{matrix_format} is not a valid option")
    if sectors_file is not None and not os.path.isfile(sectors_file):
        raise ValueError(f"{sectors_file} does not exist")

    if overwrite_dir:
        for file_name in glob(os.path.join(output_dir, "*.csv")):
            os.remove(file_name)

    # Get the lists of all segments
    all_segments = {
        "years": consts.FUTURE_YEARS,
        "p": consts.ALL_HB_P if trip_origin == "hb" else consts.ALL_NHB_P,
        "m": consts.ALL_MODES,
        "soc": consts.SOC_NEEDED,
        "ns": consts.NS_NEEDED,
        "ca": consts.CA_NEEDED,
        "tp": consts.TIME_PERIODS
    }

    # Parse the input segments
    # Checks are done on the supplied segments in the aggregation stage
    parsed_segments = {}
    for segment in all_segments:
        parsed_segments[segment] = parse_segments(
            segments_needed.get(segment),
            all_segments[segment]
        )

    # Aggregate the matrices
    try:
        # TODO add aggregation_method to this function
        output_files = aggregate_matrices(
            import_dir=matrix_directory,
            export_dir=output_dir,
            trip_origin=trip_origin,
            matrix_format=matrix_format,
            years_needed=parsed_segments["years"],
            p_needed=parsed_segments["p"],
            m_needed=parsed_segments["m"],
            soc_needed=parsed_segments["soc"],
            ns_needed=parsed_segments["ns"],
            ca_needed=parsed_segments["ca"],
            tp_needed=parsed_segments["tp"],
            return_paths=True
        )
    except AttributeError:
        # If there are no matrices available for these segments
        # This should probably be handled in matrix_processing
        print("ERROR::MISSING_SEGMENTS")
        success = False

    # Aggregate sectors if requried
    if sectors_file is not None:
        sr = SectorReporter(
            default_zone_file=zone_file,
            default_sector_grouping_file=sectors_file
        )
        valid_files = output_files.copy()
        output_files = []
        for matrix_file in valid_files:
            # Write the sectored matrices in place
            sr.aggregate_matrix_sectors(
                matrix_file,
                zone_system_name=sectors_name,
                aggregation_method=aggregation_method
            )
            new_file = matrix_file.replace(".csv", "_sector.csv")
            output_files.append(new_file)
            os.replace(matrix_file, new_file)

    # Create a GIS format report
    generate_gis_report(
        output_files,
        parsed_segments["years"],
        parsed_segments["p"]
    )

    return success


def generate_gis_report(all_files: List[int],
                        years_needed: List[int],
                        purposes_needed: List[int],
                        aggregate_purposes: bool = True,
                        aggregate_years: bool = True):
    """
    TODO
    """

    # Get the base file names so that purpose and year can be combined to
    # one file
    replaces = (
        ("_p", "{purpose}"),
        ("_yr", "{year}")
    )
    re_string = r"(?<={old})(\d+)"
    base_files = all_files.copy()
    for old, new in replaces:
        base_files = set(
            [
                re.sub(re_string.format(old=old), new, x)
                for x in base_files
            ]
        )

    # Loop over all required segments and aggregate to a stacked matrix for GIS
    # format
    for file_base in tqdm(base_files):

        trip_ends = pd.DataFrame()
        matrix = pd.DataFrame()

        for year, purpose in tqdm(product(years_needed, purposes_needed)):

            file_name = file_base.format(
                year=year,
                purpose=purpose
            )
            df = pd.read_csv(file_name, index_col=0).stack()
            df.columns = ["v"]

            if matrix.empty:
                matrix = pd.DataFrame(index=df.index)

            column_name = f"{year}_p{purpose}"
            matrix[column_name] = df

        if aggregate_purposes:
            years = [str(x) for x in years_needed]
            for year in years:
                matrix[year] = matrix[
                    [col for col in matrix.columns if year in col]
                ].sum(axis=1)

        if aggregate_years:
            purposes = [str(x) for x in purposes_needed]
            for purpose in purposes:
                matrix[purpose] = matrix[
                    [col for col in matrix.columns if f"_p{purpose}" in col]
                ].sum(axis=1)

        trip_ends = matrix.groupby(level=0).sum().merge(
            matrix.groupby(level=1).sum(),
            left_index=True,
            right_index=True,
            suffixes=("_o", "_d"),
        )

        out_file = file_base.format(year="_all", purpose="_all")
        trip_ends.to_csv(out_file.replace(".csv", "_te_gis_report.csv"))
        matrix.to_csv(out_file.replace(".csv", "_gis_report.csv"))


def parse_segments(required_segments: Union[List[int], str],
                   all_segments: List[int]):
    """
    TODO
    """

    # If the segment is to be aggregated then pass None
    if required_segments == "Agg":
        return None
    elif required_segments == "Keep":
        return all_segments
    else:
        return required_segments


def test():
    """
    TODO: write docs
    Provides test paths
    """

    test_directories = {
        "pa": r"Y:\NorMITs Demand\norms\v2_2-EFS_Output\iter1\PA Matrices"
    }

    test_output_dir = (
        r"C:\Users\Monopoly\Documents\EFS\data\summaries")
    test_trip_origin = "hb"
    test_segments_needed = {
        "years": [2018, 2033, 2035, 2050],
        "p": "Keep",
        "m": [6],
        "soc": "Agg",
        "ns": "Agg",
        "ca": "Agg",
        "tp": "Agg"
    }

    zoning = r"C:\Users\Monopoly\Documents\EFS\data\zoning"
    test_zones_file = os.path.join(
        zoning, "msoa_zones.csv"
    )
    test_sectors_file = os.path.join(
        zoning, "tfn_level_one_sectors_norms_grouping.csv"
    )

    errors = []
    overwrite = True

    for test_matrix_format in test_directories.keys():

        test_directory = test_directories[test_matrix_format]

        successful = matrix_reporting(
            test_directory,
            test_output_dir,
            test_trip_origin,
            test_matrix_format,
            segments_needed=test_segments_needed,
            zone_file=test_zones_file,
            sectors_name="tfn_sectors",
            sectors_file=test_sectors_file,
            aggregation_method="sum",
            overwrite_dir=overwrite
        )

        if not successful:
            errors.append([test_matrix_format, test_segments_needed])

        overwrite = False

    print("Errors:")
    print(*errors, sep="\n")


if __name__ == "__main__":
    test()
