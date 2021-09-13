# -*- coding: utf-8 -*-
"""
Created on: 13/09/2021
Updated on:

Original author: Ben Taylor
Last update made by:
Other updates made by:

File purpose:
Script to decompress TfN NorMITs Tools output matrices. Decompressed matrices
are output into .csv files.

In order to run the following python files are needed:
pandas
tqdm

By default this script will find all compressed files in the current directory,
it will then decompress them and write them out to csv files with the same
name (only the filename suffix being different).

Input and output locations can be customised using the -i, and -o arguments
to this script, respectively. For example the following call will read in the
matrix at "C:/test_matrix.pbz2", decompress it and write it out to "C:/output.csv".

`python decompress_matrices.py -i "C:/test_matrix.pbz2" -o "C:/output.csv"`

The following would also work:
`python decompress_matrices.py --input "C:/test_matrix.pbz2" --output "C:/output.csv"`

run `python decompress_matrices.py -h` or `python decompress_matrices.py --help`
for further information on what arguments can be passed to the script to
customise how it runs.
"""
# Built-Ins
import os
import bz2
import pickle
import pathlib
import argparse

from typing import Tuple

# Third Party
import tqdm

# Local Imports

VALID_SUFFIXES = ['.pbz2']


def is_path_to_file(path: pathlib.Path) -> bool:
    if path.suffix == '':
        return False
    return True


def get_cmd_args() -> Tuple[pathlib.Path, pathlib.Path, int]:
    # ## SET UP THE ARGUMENT PARSER ## #
    parser = argparse.ArgumentParser()

    # Input
    parser.add_argument(
        '-i',
        '--input',
        type=str,
        default='.',
        help=(
            "The file, or directory of files, to decompress. If not set, "
            "defaults to the directory this script is being run in."
        )
    )

    # Output
    parser.add_argument(
        '-o',
        '--output',
        type=str,
        default='.',
        help=(
            "Where to output the decompressed matrices. If not set, "
            "defaults to the directory this script is being run in."
        )
    )

    # Output
    parser.add_argument(
        '-r',
        '--round',
        type=int,
        help=(
            "The number of decimal places to round the values of the matrices "
            "to before writing out. Can reduce output file sizes if less "
            "precise matrices are needed (usually not the case!)."
        )
    )

    # ## PARSE THE ARGUMENTS ## #
    args = parser.parse_args()

    # Assign
    input_path = pathlib.Path(args.input)
    output_path = pathlib.Path(args.output)

    # Check inputs are valid pairs
    if not is_path_to_file(input_path) and is_path_to_file(output_path):
        raise ValueError(
            "If the input path is a directory, then the output cannot be "
            "a filename.\n"
            "\tInput path: %s\n"
            "\tOutput path: %s"
            % (input_path, output_path)
        )

    # Set output to input filename.csv
    if is_path_to_file(input_path) and not is_path_to_file(output_path):
        output_path = output_path.parent / pathlib.Path(input_path.stem + '.csv')

    return input_path, output_path, args.round


def read_compressed_df(path: pathlib.Path):
    """
    Reads the data at path, decompresses, and returns the object.

    Parameters
    ----------
    path:
        The full path to the object to read

    Returns
    -------
    object:
        The object that was read in from disk.
    """
    if path.suffix not in VALID_SUFFIXES:
        raise ValueError(
            'Can only decompress files with %s suffixes. Got the '
            'following invalid filename: %s'
            % (VALID_SUFFIXES, path)
        )
    return pickle.load(bz2.BZ2File(path, 'rb'))


def main():
    input_path, output_path, round_dp = get_cmd_args()

    # If single path, input and out easy
    if is_path_to_file(input_path):
        df = read_compressed_df(input_path)
        if round_dp is not None:
            df = df.round(round_dp)
        df.to_csv(output_path)
        return

    # Otherwise loop over all the .pbz2 files in this directory
    matrices = [pathlib.Path(x) for x in os.listdir(input_path)]
    matrices = [x for x in matrices if x.suffix in VALID_SUFFIXES]

    if len(matrices) == 0:
        raise ValueError(
            "No valid files found to decompress. Searching for files ending "
            "in: %s\n"
            "\tHere: %s"
            % (VALID_SUFFIXES, input_path)
        )

    for file in tqdm.tqdm(matrices, desc='Decompressing files', unit='file'):
        df = read_compressed_df(input_path / file)
        if round_dp is not None:
            df = df.round(round_dp)
        df.to_csv(output_path / pathlib.Path(file.stem + '.csv'))


if __name__ == '__main__':
    main()
