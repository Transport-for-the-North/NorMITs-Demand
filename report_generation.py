import os
from glob import glob
from typing import List, Union

from demand_utilities.sector_reporter_v2 import SectorReporter
from matrix_processing import aggregate_matrices
import efs_constants as consts


VALID_TRIP_ORIGIN = ["hb", "nhb"]


def matrix_reporting(matrix_directory: str,
                     output_dir: str,
                     trip_origin: str,
                     matrix_format: str,
                     segments_needed: dict = {},
                     zone_file: str = None,
                     sectors_file: str = None,
                     sectors_name: str = "Sectors",
                     aggregation_method: str = "sum"):
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
        aggregate_matrices(
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
            tp_needed=parsed_segments["tp"]
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
        # TODO change this to just get the new matrices
        valid_files = (
            set(glob(os.path.join(output_dir, "*.csv")))
            - set(glob(os.path.join(output_dir, "*_agg.csv")))
        )
        for matrix_file in valid_files:
            # Write the sectored matrices in place
            sr.aggregate_matrix_sectors(
                matrix_file,
                zone_system_name=sectors_name,
                aggregation_method=aggregation_method
            )
            os.replace(matrix_file, matrix_file.replace(".csv", "_agg.csv"))

    return success


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
        "od": r"C:\Users\japeach\Documents\49808 - EFS\EFS Data\OD Matrices",
        "pa": r"C:\Users\japeach\Documents\49808 - EFS\EFS Data\PA Matrices"
    }

    test_output_dir = (
        r"C:\Users\japeach\Documents\49808 - EFS\EFS Data\Summaries")
    test_trip_origin = "hb"
    test_segments_needed = {
        "years": [2018],
        "p": [1, 2],
        "m": [6],
        "soc": "Agg",
        "ns": "Agg",
        "ca": "Agg",
        "tp": [1]
    }

    zoning = r"C:\Users\japeach\Documents\49808 - EFS\EFS Data\zoning"
    test_zones_file = os.path.join(
        zoning, "msoa_zones.csv"
    )
    test_sectors_file = os.path.join(
        zoning, "tfn_level_one_sectors_norms_grouping.csv"
    )

    errors = []

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
            aggregation_method="sum"
        )

        if not successful:
            errors.append([test_matrix_format, test_segments_needed])

    print("Errors:")
    print(*errors, sep="\n")


if __name__ == "__main__":
    test()
