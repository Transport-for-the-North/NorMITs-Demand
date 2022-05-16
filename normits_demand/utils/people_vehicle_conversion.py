# -*- coding: utf-8 -*-
"""
Created on: 25/04/2022
Updated on:

Original author: Ben Taylor
Last update made by:
Other updates made by:

File purpose:
Convert matrices between people and vehicle formats. Also handles
hourly total and hourly average conversions.
"""
from __future__ import annotations

# Built-Ins
import dataclasses
import pathlib

from typing import Dict
from typing import List
from typing import Optional

# Third Party
import pandas as pd

# Local Imports
from normits_demand import constants
from normits_demand.utils import file_ops
from normits_demand.concurrency import multiprocessing


@dataclasses.dataclass
class PurposeVehicleOccupancies:
    """
    Store vehicle occupancy factors for a specific purpose / file type
    """

    # Occupancy factors
    tp1: float
    tp2: float
    tp3: float
    tp4: float

    purpose_str: str
    unique_str: Optional[str] = None

    def __iter__(self):
        tp_dict = {
            1: self.tp1,
            2: self.tp2,
            3: self.tp3,
            4: self.tp4,
        }
        for tp_name, tp_val in tp_dict.items():
            yield tp_name, tp_val

    @staticmethod
    def from_df(
        df: pd.DataFrame,
        purpose_str: str,
        unique_str: str = None,
        tp_col: str = "tp",
        occupancy_col: str = "occupancy",
    ) -> PurposeVehicleOccupancies:
        """Create a PurposeVehicleOccupancies object from a pandas DataFrame

        Parameters
        ----------
        df:
            The dataframe to convert

        purpose_str:
            A string describing the purpose of these vehicle occupancies

        unique_str:
            A string describing any other unique feature of these vehicle
            occupancies. Used to filter down filenames when looking for the
            files to apply these factors to.

        tp_col:
            The columns of df containing the time period data.
            Expected values: [1, 2, 3, 4].

        occupancy_col:
            The columns of df containing occupancy factors.

        Returns
        -------
        vehicle_occupancies:
            A PurposeVehicleOccupancies object from the given data.
        """
        # Init
        expected_tp_vals = {1, 2, 3, 4}
        got_tp_vals = set(df[tp_col].unique())

        # Check the expected values exist
        missing_tp_vals = expected_tp_vals - got_tp_vals
        if len(missing_tp_vals) > 0:
            raise ValueError(
                "Missing time period values in given df. "
                f"Expected {expected_tp_vals}, but missing {missing_tp_vals}."
            )

        # Extract the values for each tp
        def extract_tp_val(tp: int):
            local_df = df[df[tp_col] == tp].copy()
            if len(local_df) > 1:
                raise ValueError(
                    f"More than one value found for tp{tp}, purpose {purpose_str}. "
                    f"Found {len(local_df)} values:\n{local_df}"
                )

            return local_df[occupancy_col].squeeze()

        return PurposeVehicleOccupancies(
            tp1=extract_tp_val(1),
            tp2=extract_tp_val(2),
            tp3=extract_tp_val(3),
            tp4=extract_tp_val(4),
            purpose_str=purpose_str,
            unique_str=unique_str,
        )

    def filter_filenames(
        self,
        paths: List[pathlib.Path],
        filter_unique: bool = True,
    ) -> List[pathlib.Path]:
        """Filter a list of paths to be relevant to these occupancies.

        Filters the given list of `paths` by only returning those which
        contain the string in `self.purpose_str` and
        (optionally) `self.unique_str`.

        Parameters
        ----------
        paths:
            The list of paths to consider and filter.

        filter_unique:
            Whether to also filter the `paths` by the string in
            `self.unique_str`. If `self.unique_str` is None, then this
            argument is ignored.

        Returns
        -------
        filtered_paths:
            A list of paths that are relevant to these vehicle occupancies
        """
        filtered = [x for x in paths if self.purpose_str in str(x)]

        if self.unique_str is not None and filter_unique:
            filtered = [x for x in filtered if self.unique_str in str(x)]

        return filtered


@dataclasses.dataclass
class VehicleOccupancies:
    """
    Store vehicle occupancy factors for 3 user classes
    """

    business_occupancies: PurposeVehicleOccupancies
    commute_occupancies: PurposeVehicleOccupancies
    other_occupancies: PurposeVehicleOccupancies

    def __iter__(self):
        items = [self.business_occupancies, self.commute_occupancies, self.other_occupancies]
        for each in items:
            yield each

    @staticmethod
    def from_df(
        df: pd.DataFrame,
        purpose_col: str = "p",
        tp_col: str = "tp",
        occupancy_col: str = "occupancy",
        unique_str: str = None,
    ) -> VehicleOccupancies:
        """Create a VehicleOccupancies object from a pandas DataFrame

        Parameters
        ----------
        df:
            The dataframe to convert

        purpose_col:
            The columns of df containing the purpose data.
            Expected purpose values: ["business", "commute", "other"].

        tp_col:
            The columns of df containing the time period data.
            Expected values: [1, 2, 3, 4].

        occupancy_col:
            The columns of df containing occupancy factors.

        unique_str:
            A string describing any other unique feature of these vehicle
            occupancies. Used to filter down filenames when looking for the
            files to apply these factors to.

        Returns
        -------
        vehicle_occupancies:
            A VehicleOccupancies object from the given data.
        """
        # Init
        expected_p_vals = {"business", "commute", "other"}
        expected_tp_vals = {1, 2, 3, 4}

        # Check the expected values exist
        got_p_vals = set(df[purpose_col].unique())
        got_tp_vals = set(df[tp_col].unique())

        missing_p_vals = expected_p_vals - got_p_vals
        if len(missing_p_vals) > 0:
            raise ValueError(
                "Missing purpose values in given df. "
                f"Expected {expected_p_vals}, but missing {missing_p_vals}."
            )

        missing_tp_vals = expected_tp_vals - got_tp_vals
        if len(missing_tp_vals) > 0:
            raise ValueError(
                "Missing time period values in given df. "
                f"Expected {expected_tp_vals}, but missing {missing_tp_vals}."
            )

        # Get each df section
        def extract_purpose_data(purpose_str: str):
            return PurposeVehicleOccupancies.from_df(
                df=df[df[purpose_col] == purpose_str].copy(),
                purpose_str=purpose_str,
                unique_str=unique_str,
                tp_col=tp_col,
                occupancy_col=occupancy_col,
            )

        return VehicleOccupancies(
            business_occupancies=extract_purpose_data("business"),
            commute_occupancies=extract_purpose_data("commute"),
            other_occupancies=extract_purpose_data("other"),
        )


def _conversion_core(
    import_path: pathlib.Path,
    export_path: pathlib.Path,
    multiplier: float,
    round_dp: int,
) -> None:
    """Read in matrix, multiply, and write out"""
    df = file_ops.read_df(import_path, find_similar=True, index_col=0)
    df *= multiplier
    df = df.round(decimals=round_dp)
    file_ops.write_df(df, export_path)


def people_to_vehicle_units(*args, **kwargs) -> None:
    """Convert matrices from people units to vehicle units

    Uses the given `vehicle_occupancies` to convert the matrices in
    `import_dir` into vehicle units and writes them out to `export_dir`.
    In short this is done as:
    out_matrix = in_matrix * occupancy factor

    When hourly factors are passed in as well, this becomes
    out_matrix = in_matrix * occupancy_factor * hourly_factor

    Parameters
    ----------
    import_dir:
        The directory containing the matrices to convert.

    export_dir:
        The directory to output any converted matrices.

    vehicle_occupancies:
        A VehicleOccupancies describing the vehicle occupancy factors for
        different time periods and purposes.

    hourly_factors:
        The factors to apply to each hour. Can be used to convert between
        average hour and total hour matrices.

    round_dp:
        The number of decimal places to round the output up to.

    process_count:
        The number of processes to use when converting the matrices.

    Returns
    -------
    None
    """
    _vehicle_people_conversion(*args, **dict(kwargs, invert_occupancies=True))


def vehicle_to_people_units(*args, **kwargs) -> None:
    """Convert matrices from vehicle units to people units

    Uses the given `vehicle_occupancies` to convert the matrices in
    `import_dir` into people units and writes them out to `export_dir`.
    In short this is done as:
    out_matrix = in_matrix * occupancy factor

    When hourly factors are passed in as well, this becomes
    out_matrix = in_matrix * occupancy_factor * hourly_factor

    Parameters
    ----------
    import_dir:
        The directory containing the matrices to convert.

    export_dir:
        The directory to output any converted matrices.

    vehicle_occupancies:
        A VehicleOccupancies describing the vehicle occupancy factors for
        different time periods and purposes.

    hourly_factors:
        The factors to apply to each hour. Can be used to convert between
        average hour and total hour matrices.

    round_dp:
        The number of decimal places to round the output up to.

    process_count:
        The number of processes to use when converting the matrices.

    Returns
    -------
    None
    """
    _vehicle_people_conversion(*args, **dict(kwargs, invert_occupancies=False))


def _vehicle_people_conversion(
    import_dir: pathlib.Path,
    export_dir: pathlib.Path,
    vehicle_occupancies: VehicleOccupancies,
    invert_occupancies: bool,
    hourly_factors: Dict[int, float] = None,
    round_dp: int = constants.DEFAULT_ROUNDING,
    process_count: int = constants.PROCESS_COUNT,
) -> None:
    """Convert matrices from vehicle units to people units

    Uses the given `vehicle_occupancies` to convert the matrices in
    `import_dir` into people units and writes them out to `export_dir`.
    In short this is done as:
    out_matrix = in_matrix * occupancy factor

    When hourly factors are passed in as well, this becomes
    out_matrix = in_matrix * occupancy_factor * hourly_factor

    Parameters
    ----------
    import_dir:
        The directory containing the matrices to convert.

    export_dir:
        The directory to output any converted matrices.

    vehicle_occupancies:
        A VehicleOccupancies describing the vehicle occupancy factors for
        different time periods and purposes.

    invert_occupancies:
        Whether to invert the passed in vehicle occupancy factors or not.
        Should only be used internally - useful to set to True when converting
        from people units into car units. Multiplication becomes:
        out_matrix = in_matrix * (1 / occupancy factor)

    hourly_factors:
        The factors to apply to each hour. Can be used to convert between
        average hour and total hour matrices.

    round_dp:
        The number of decimal places to round the output up to.

    process_count:
        The number of processes to use when converting the matrices.

    Returns
    -------
    None
    """
    # Init
    all_filenames = file_ops.list_files(import_dir)
    pbar_kwargs = {
        "desc": "converting matrices to people units",
        "unit": "matrix",
    }

    # Generate the kwargs to multiprocess
    kwarg_list = list()
    for purpose_occupancies in vehicle_occupancies:
        # Filter down to this purpose
        purpose_filenames = purpose_occupancies.filter_filenames(all_filenames)

        # Filter down for each tp
        for tp_name, tp_factor in purpose_occupancies:
            tp_filenames = [x for x in purpose_filenames if f"tp{tp_name}" in x]

            # Calculate the multiplier to use
            multiplier = tp_factor
            if invert_occupancies:
                multiplier = 1 / multiplier

            if hourly_factors is not None:
                multiplier *= hourly_factors[tp_name]

            # Add to list of jobs
            for fname in tp_filenames:
                kwarg_list.append({
                    "import_path": import_dir / fname,
                    "export_path": export_dir / fname,
                    "multiplier": multiplier,
                    "round_dp": round_dp,
                })

    # Multiprocess the conversion
    multiprocessing.multiprocess(
        fn=_conversion_core,
        kwargs=kwarg_list,
        pbar_kwargs=pbar_kwargs,
        process_count=process_count,
    )
