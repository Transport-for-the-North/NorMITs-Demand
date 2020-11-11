import os
import re
from glob import glob
from typing import List, Union
from itertools import product

import pandas as pd
import numpy as np
from tqdm import tqdm

from demand_utilities import utils as du
from demand_utilities import reports as dr
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
                     tld_path: str = None,
                     cost_path: str = None,
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

    # Extract the trip length distributions
    if cost_path is not None and tld_path is not None:
        tld_reporting(
            output_dir,
            matrix_format,
            tld_path=tld_path,
            cost_lookup_path=cost_path
        )

    output_files = glob(os.path.join(output_dir, "*.csv"))

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


def tld_reporting(matrix_dir: str,
                  matrix_type: str,
                  tld_path: str,
                  cost_lookup_path: str):
    """TODO"""

    # Create the tld directory if it doesn't exist
    output_dir = os.path.join(matrix_dir, "tld")
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    else:
        for file_name in glob(os.path.join(output_dir, "*.csv")):
            os.remove(file_name)

    # Loop through all matrices, pulling the trip length dists as required
    matrices = os.listdir(matrix_dir)
    mat_df = du.parse_mat_output(
        matrices,
        sep="_",
        mat_type=matrix_type,
        file_format=".csv",
        file_name="matrix"
    )
    for _, mat_desc in mat_df.iterrows():
        mat_dict = mat_desc.to_dict()

        # Extract trip matrix info for each file
        matrix = mat_dict.pop("matrix")
        matrix = pd.read_csv(
            os.path.join(matrix_dir, matrix)
        )

        # Extract segments if they exist - remove from calib params if needed
        # purpose, mode, segment(optional) required in tlb function
        trip_origin = mat_dict.pop("trip_origin")
        year = mat_dict.pop("yr")
        purpose = mat_dict.get("p")
        mode = mat_dict.get("m")
        soc = mat_dict.get("soc")
        ns = mat_dict.get("ns")
        ca = mat_dict.pop("ca", None)

        # Choose the correct segmentation for tlb and dist names
        # Non-home based in "standard_segments" can use tp to select tlb
        # Time period is not an option otherwise
        if soc is None and ns is None:
            seg_tld_path = os.path.join(tld_path, "standard_segments")
            segment = None
            if trip_origin == "hb":
                tp = mat_dict.pop("tp", None)
            else:
                tp = mat_dict.get("tp")
        else:
            seg_tld_path = os.path.join(tld_path, "enhanced_segments")
            segment = soc if du.is_none_like(ns) else ns
            tp = mat_dict.pop("tp", None)

        for item, dat in mat_dict.items():
            if dat.isnumeric():
                mat_dict.update({item: int(dat)})

        # Get the relevant trip length bands
        # Some legacy code here - "ntem"
        # TODO Use the year here to get the forecast/base tlb when they exist
        print(mat_dict)
        year_tld_path = seg_tld_path
        tlb = du.get_trip_length_bands(
            year_tld_path,
            mat_dict,
            "ntem",
            trip_origin,
            replace_nan=False,
            echo=True
        )

        
        # Set the string sent to the costs function
        # TODO fix for NORMS tp costs - do not exist at the moment
        tp_str = "24hr" if tp is None else "tp"
        tp_str = "24hr"
        _ = str(mat_dict.pop("tp", None))

        # Get the cost data for the purpose/mode
        costs, cost_name = du.get_costs(
            cost_lookup_path,
            mat_dict,
            tp=tp_str,
            iz_infill=0.5
        )

        # Convert to a square numpy matrix
        unq_zones = list(range(1, (costs[list(costs)[0]].max())+1))
        costs = du.df_to_np(
            costs,
            v_heading='p_zone',
            h_heading='a_zone',
            values='cost',
            unq_internal_zones=unq_zones
        )

        matrix = matrix.drop(list(matrix)[0], axis=1).values

        # This bit matches the shape to the NORMS cost zones
        # TODO see why this is needed - zoning mismatch
        pad_matrix = np.zeros((1300, 1300))
        pad_matrix[:matrix.shape[0], :matrix.shape[1]] = matrix

        # Get trip length by band
        
        (trip_lengths_by_band_km,
         band_shares_by_band,
         average_trip_length) = dr.get_trip_length_by_band(tlb,
                                                           costs,
                                                           pad_matrix)

        # Merge into single dataframe on the band index
        tld_results = trip_lengths_by_band_km.merge(
            band_shares_by_band,
            on="tlb_index"
        ).fillna(0.0)

        # Save individual bands and band shares to separate csv files
        out_file = du.get_dist_name(
            trip_origin,
            matrix_type,
            year,
            purpose,
            mode,
            segment=segment,
            car_availability=ca,
            tp=tp,
            csv=True,
            suffix="_tld"
        )
        out_file = os.path.join(output_dir, out_file)
        tld_results.to_csv(out_file, index=False)

    # Concatenate all files into a single stacked csv
    concat_vector_folder(
        output_dir,
        matrix_type=matrix_type,
        output_name="tld_dists.csv"
    )


def concat_vector_folder(data_dir: str,
                         matrix_type: str,
                         output_name: str = None):
    """TODO"""

    # Override default file name
    output_name = output_name or "concatenated_data.csv"

    # Fetch a list of all .csv files in the directory
    files = os.listdir(data_dir)

    file_df = du.parse_mat_output(
        files,
        sep="_",
        mat_type=matrix_type,
        file_format=".csv",
        file_name="file"
    )

    vector_df = pd.DataFrame()

    for _, row in file_df.iterrows():

        single_vector = pd.read_csv(
            os.path.join(data_dir, row.pop("file"))
        )

        # Add additional columns for each segment e.g. purpose, mode, soc/ns
        for key, value in row.items():
            single_vector[key] = value

        if vector_df.empty:
            vector_df = single_vector
        else:
            vector_df = pd.concat(
                (vector_df, single_vector),
                axis=0
            )
            
    # Remove columns that just contain "none" - e.g. suffixes on the file name
    vector_df = vector_df[
        [col for col in vector_df 
         if next(iter(set(vector_df[col]))) != "none"]
    ]

    vector_df.to_csv(
        os.path.join(data_dir, output_name),
        index=False
    )
    
    # TODO add option to remove individual files if needed


def test(param_file):
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
