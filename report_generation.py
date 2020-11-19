import os
import re
import json
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
import external_forecast_system as efs


# Define the possible values for trip origin type -
# homebased or non-homebased
VALID_TRIP_ORIGIN = ["hb", "nhb"]


def matrix_reporting(matrix_directory: str,
                     output_dir: str,
                     trip_origin: str,
                     matrix_format: str,
                     segments_needed: dict = {},
                     zone_file: str = None,
                     sectors_files: List[str] = None,
                     zones_name: str = "model",
                     sectors_names: List[str] = ["sector"],
                     aggregation_method: str = "sum",
                     tld_path: str = None,
                     cost_path: str = None,
                     overwrite_dir: bool = True,
                     collate_years: bool = False):
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
    if len(sectors_files) != len(sectors_names):
        raise AttributeError("Number of sector files must match the number "
                             "of sector names")
    for sectors_file in sectors_files:
        if sectors_file is not None and not os.path.isfile(sectors_file):
            raise ValueError(f"{sectors_file} does not exist")

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

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
    # TODO Do this in a loop so that matrices can be sectored as produced
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
    if sectors_files is not None:
        sr = SectorReporter()
        valid_files = output_files.copy()
        output_files = []
        for sectors_file, sectors_name in zip(sectors_files, sectors_names):
            for matrix_file in valid_files:
                # Write the sectored matrices in place
                sr.aggregate_matrix_sectors(
                    matrix_file,
                    zone_system_name=zones_name,
                    zone_system_file=zone_file,
                    sector_grouping_file=sectors_file,
                    sector_system_name=sectors_name,
                    aggregation_method=aggregation_method
                )
                suffix = f"_{sectors_name}.csv"
                new_file = matrix_file.replace(".csv", suffix)
                output_files.append(new_file)
                os.replace(matrix_file, new_file)

    if collate_years:
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
    """Collates aggregated matrices together to create a single report file
    that can be easily read by GIS programs

    Parameters
    ----------
    all_files : List[int]
        List of all aggregated matrix files to include in the report.
        Should all be the same format
    years_needed : List[int]
        List of years to include.
    purposes_needed : List[int]
        List of purposes to include.
    aggregate_purposes : bool, optional
        Wh, by default True
    aggregate_years : bool, optional
        [description], by default True
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
            try:
                df = pd.read_csv(file_name, index_col=0).stack()
            except FileNotFoundError:
                # Warn that the file was not found, but this is likely
                # because of a soc/ns segment mismatch
                print(f"Warning: File {file_name} does not exist")
                continue
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
    """Converts required segment strings into those expected by 
    aggregate_matrices

    Parameters
    ----------
    required_segments : Union[List[int], str]
        Unparsed segment arguments can be a list of segments or 
        "Keep" or "Agg"
    all_segments : List[int]
        List of all possible segments

    Returns
    -------
    List[int] or None
        Parsed segments for the aggregate_matrices function
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
    """Generates the trip length distributions of a directory of matrices

    Parameters
    ----------
    matrix_dir : str
        Path to the directory containing EFS format matrices
    matrix_type : str
        "pa" or "od"
    tld_path : str
        Path to the trip length band folder
    cost_lookup_path : str
        Path to the base cost folder
    """

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
        fill_shape = 1246
        pad_matrix = np.zeros((fill_shape, fill_shape))
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
    """Concatenates a folder of "long" format .csv files to a single file

    Parameters
    ----------
    data_dir : str
        Path to the directory containing the CSV files
    matrix_type : str
        The matrix type - "pa" or "od"
    output_name : str, optional
        Name of the concatenated output file, by default None
    """

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


def load_report_params(param_file: str) -> None:
    """Load report generation parameters from file.
    Allows a number of options to be set in a json file.

    Parameters
    ----------
    param_file : str
        Path to the options file - json format.
        Should contain the required keys:
         - "matrix_directories"
         - "output_dir"
         - "matrix_format"
         - "trip_origin"
         - "segments_needed": {
             "years"
             "p"
             "m"
             "soc"
             "ns"
             "ca"
             "tp"
             }
         - "zones_file"
         - "sectors_file"
         - "cost_path"
         - "tld_path"

    Raises
    ------
    FileNotFoundError
        If the json file does not exist
    """

    if os.path.isfile(param_file):
        with open(param_file) as f:
            params = json.load(f)
    else:
        raise FileNotFoundError("Parameter File Does Not Exist")

    return params


def main(param_file, imports, exports):
    """Reads in a parameter file (JSON) and creates the defined matrix
    summaries

    Parameters
    ----------
    param_file : str
        Path to the parameter JSON file. See load_report_params
        for requirements
    """

    params = load_report_params(param_file)

    errors = []
    overwrite = True
    
    output_dir = os.path.join(exports["reports"], params["output_dir"])
    zones_file = os.path.join(imports["zoning"], params["zones_file"])
    sectors_files = [os.path.join(imports["zone_translation"], x) 
                    for x in params["sectors_files"]]
    tld_path = os.path.join(imports["home"], params["tld_path"])
    cost_path = os.path.join(imports["home"], params["cost_path"])
    
    overwrite = params["overwrite_outputs"]
    collate_years = params["collate_years"]
    
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    for matrix_format in params["matrix_directories"].keys():

        matrix_dir = exports[params["matrix_directories"][matrix_format]]

        successful = matrix_reporting(
            matrix_directory=matrix_dir,
            output_dir=output_dir,
            trip_origin=params["trip_origin"],
            matrix_format=params["matrix_format"],
            segments_needed=params["segments_needed"],
            zone_file=zones_file,
            zones_name=params["zones_name"],
            sectors_names=params["sectors_names"],
            sectors_files=sectors_files,
            aggregation_method="sum",
            overwrite_dir=overwrite,
            tld_path=tld_path,
            cost_path=cost_path,
            collate_years=collate_years
        )

        if not successful:
            errors.append([matrix_format, params["segments_needed"]])

        overwrite = False

    print("Errors:")
    print(*errors, sep="\n")


if __name__ == "__main__":
    # Run the configuration files to produce the report formats required by
    # Power BI
    
    import_location = "Y:/"
    output_location = "C:/"
    model_name = "norms_2015"
    iter_num = 1
    iter_name = f"iter{iter_num}"
    
    efs_main = efs.ExternalForecastSystem(
        model_name=model_name,
        import_location=import_location,
        output_location=output_location
    )
    
    imports, exports, _ = efs_main.generate_output_paths(
        output_location,
        model_name,
        iter_name
    )
    
    pa_params = r"C:\Users\Monopoly\Documents\EFS\data\params\pav2.json"
    main(pa_params, imports, exports)
    # tp_pa_params = r"C:\Users\Monopoly\Documents\EFS\data\params\tp_pa.json"
    # main(tp_pa_params)
    # nhb_params = r"C:\Users\Monopoly\Documents\EFS\data\params\nhb_pa.json"
    # main(nhb_params)
    # test_params = r"C:\Users\Monopoly\Documents\EFS\data\params\test.json"
    # main(test_params)
