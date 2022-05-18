# -*- coding: utf-8 -*-
"""
Created on: 10/05/2022
Updated on:

Original author: Ben Taylor
Last update made by:
Other updates made by:

File purpose:
Module to take NoRMS output matrices and generalise them into a standard
format where they can be picked up by other models
"""
# Built-Ins
import os
import logging
import pathlib
import itertools
import dataclasses

from typing import Any
from typing import List
from typing import Dict

# Third Party
import tqdm
import numpy as np
import pandas as pd

# Local Imports

from normits_demand import core as nd_core
from normits_demand import logging as nd_log
from normits_demand.utils import file_ops
from normits_demand.utils import general as du
from normits_demand.utils import pandas_utils as pd_utils
from normits_demand.matrices import pa_to_od as pa2od
from normits_demand.matrices import utils as matrix_utils
from normits_demand.matrices import matrix_processing
from normits_demand.tools.norms import tp_proportion_extractor
from normits_demand.tools.norms import tp_proportion_converter


LOG = logging.getLogger(__name__)


@dataclasses.dataclass
class NormsTpProportionFiles:
    """Store the filenames of all CUBE tour proportion file names"""

    home_dir: pathlib.Path
    extracted_pickle_fname: str = "norms_time_period_proportions.pkl"

    # Home-based tour proportions
    hb_internal_business_fname: str = "SplitFactors_D1.mat"
    hb_internal_commute_fname: str = "SplitFactors_D2.mat"
    hb_internal_other_fname: str = "SplitFactors_D3.mat"

    # Non-home-based time period splitting factors
    nhb_internal_split_factors_fnames = {
        1: "OD_Prop_AM_PT.mat",
        2: "OD_Prop_IP_PT.mat",
        3: "OD_Prop_PM_PT.mat",
        4: "OD_Prop_OP_PT.mat",
    }

    external_split_factors_fnames = {
        1: "Time_of_Day_Factors_Zonal_AM.mat",
        2: "Time_of_Day_Factors_Zonal_IP.mat",
        3: "Time_of_Day_Factors_Zonal_PM.mat",
        4: "Time_of_Day_Factors_Zonal_OP.mat",
    }

    def get_extracted_pickle_path(self):
        """Build and return the path for the extracted pickle data"""
        return self.home_dir / self.extracted_pickle_fname

    def all_files_exist(self, throw_error: bool = False) -> bool:
        """Check if all files listed in this object exist

        Parameters
        ----------
        throw_error:
            Whether to throw and error when a file doesn't exist or not.t

        Returns
        -------
        all_files_exist:
            A boolean stating whether all the files listed in this object
            exist or not.
        """
        check_files = [
            self.hb_internal_business_fname,
            self.hb_internal_commute_fname,
            self.hb_internal_other_fname,
        ]
        check_files += list(self.nhb_internal_split_factors_fnames.values())
        check_files += list(self.external_split_factors_fnames.values())

        for fname in check_files:
            if not (self.home_dir / fname).exists():
                if throw_error:
                    raise ValueError(
                        "Not all the expected files exist. At least one file "
                        f"is missing. Cannot find '{self.home_dir / fname}'."
                    )
                return False

        return True


class NoRMSPostMeTpProportions:
    """Extraction and conversion of NoRMS tour proportions from CUBE"""

    _fh_th_valid_ca = [1, 2]
    _int_tp_split_valid_ca = [1, 2]
    _ext_tp_split_valid_ca = ["ca_fh", "ca_th", "nca"]

    _commute_key = "commute"
    _business_key = "business"
    _other_key = "other"
    _valid_purpose = [_commute_key, _business_key, _other_key]

    def __init__(self, tour_prop_import: os.PathLike):
        tour_prop_pickle = pd.read_pickle(tour_prop_import)
        self.zoning_system = nd_core.get_zoning_system("norms")

        # ## CONVERT DICTIONARY INTO COMPONENTS ## #
        fh_th_factors = tp_proportion_converter.convert_tour_proportions(
            tour_prop_pickle,
            self.zoning_system,
        )
        self.internal_fh_factors = fh_th_factors[0]
        self.internal_th_factors = fh_th_factors[1]

        self.internal_tp_split_factors = (
            tp_proportion_converter.convert_internal_tp_split_factors(
                tour_prop_pickle,
                self.zoning_system,
            )
        )

        self.external_tp_split_factors = (
            tp_proportion_converter.convert_external_tp_split_factors(
                tour_prop_pickle,
                self.zoning_system,
            )
        )

        # Should check that all have matching time periods
        ca_dict = list(self.internal_th_factors.values())[0]
        tp_dict = list(ca_dict.values())[0]
        self.time_periods = list(tp_dict.keys())

    def _validate_ca(self, ca: Any, valid_vals: List[Any]) -> None:
        if ca not in valid_vals:
            raise ValueError(
                f"Invalid ca value received. Expected one of "
                f"'{self._fh_th_valid_ca}'. Got '{ca}'"
            )

    def _validate_internal_fh_th_ca(self, ca: int) -> None:
        self._validate_ca(ca, self._fh_th_valid_ca)

    def _validate_internal_tp_split_ca(self, ca: int) -> None:
        self._validate_ca(ca, self._int_tp_split_valid_ca)

    def _validate_external_tp_split_ca(self, ca: str) -> None:
        self._validate_ca(ca, self._ext_tp_split_valid_ca)

    def _validate_purpose(self, purpose: str) -> None:
        if purpose not in self._valid_purpose:
            raise ValueError(
                f"Invalid purpose value received. Expected one of "
                f"'{self._valid_purpose}'. Got '{purpose}'"
            )

    def get_internal_business_fh_factors(self, ca: int) -> Dict[int, np.ndarray]:
        """Return a dictionary of from home factors"""
        self._validate_internal_fh_th_ca(ca)
        return self.internal_fh_factors[self._business_key][ca]

    def get_internal_business_th_factors(self, ca: int) -> Dict[int, np.ndarray]:
        """Return a dictionary of to home factors"""
        self._validate_internal_fh_th_ca(ca)
        return self.internal_th_factors[self._business_key][ca]

    def get_internal_commute_fh_factors(self, ca: int) -> Dict[int, np.ndarray]:
        """Return a dictionary of from home factors"""
        self._validate_internal_fh_th_ca(ca)
        return self.internal_fh_factors[self._commute_key][ca]

    def get_internal_commute_th_factors(self, ca: int) -> Dict[int, np.ndarray]:
        """Return a dictionary of to home factors"""
        self._validate_internal_fh_th_ca(ca)
        return self.internal_th_factors[self._commute_key][ca]

    def get_internal_other_fh_factors(self, ca: int,) -> Dict[int, np.ndarray]:
        """Return a dictionary of from home factors"""
        self._validate_internal_fh_th_ca(ca)
        return self.internal_fh_factors[self._other_key][ca]

    def get_internal_other_th_factors(self, ca: int,) -> Dict[int, np.ndarray]:
        """Return a dictionary of to home factors"""
        self._validate_internal_fh_th_ca(ca)
        return self.internal_th_factors[self._other_key][ca]

    def get_internal_fh_factors(self, purpose: str, ca: int,) -> Dict[int, np.ndarray]:
        """Return a dictionary of from home factors"""
        self._validate_purpose(purpose)
        self._validate_internal_fh_th_ca(ca)
        return self.internal_fh_factors[purpose][ca]

    def get_internal_th_factors(self, purpose: str, ca: int,) -> Dict[int, np.ndarray]:
        """Return a dictionary of to home factors"""
        self._validate_purpose(purpose)
        self._validate_internal_fh_th_ca(ca)
        return self.internal_th_factors[purpose][ca]

    def get_internal_nhb_tp_split_factors(
        self, purpose: str, ca: int,
    ) -> Dict[int, np.ndarray]:
        """Return a dictionary of tp split factors"""
        self._validate_purpose(purpose)
        self._validate_internal_tp_split_ca(ca)
        return self.internal_tp_split_factors[purpose][ca]

    def get_external_tp_split_factors(self, purpose: str, ca: str,) -> Dict[int, np.ndarray]:
        """Return a dictionary of tp split factors"""
        self._validate_purpose(purpose)
        self._validate_external_tp_split_ca(ca)
        return self.external_tp_split_factors[purpose][ca]


class NormsOutputToOD:
    """Converter of NoRMS matrices to standard OD format

    Takes matrices int the format of NoRMS output and converts them into
    the standard OD format. That is, three user classes and four time periods.
    All intermediate steps to get there can be output by this class.
    """

    # Class constants
    _hb_purpose_vals = ["business", "commute", "other"]
    _nhb_purpose_vals = ["business", "other"]
    _purpose_vals = _hb_purpose_vals
    _ca_vals = [1, 2]
    _mode = nd_core.Mode.TRAIN
    _int_ca = [1, 2]

    _internal_suffix = "_int"
    _external_suffix = "_ext"

    # Output folder names
    _renamed_dirname = "renamed"
    _od_dirname = "OD Matrices"
    _compiled_od_dirname = "Compiled OD Matrices"
    _internal_compiled_od_dirname = "Internal Compiled OD Matrices"

    def __init__(
        self,
        matrix_dir: pathlib.Path,
        matrix_year: int,
        time_period_proportions: NoRMSPostMeTpProportions,
        output_dir: pathlib.Path,
        matrix_renaming: os.PathLike = None,
    ):
        # Assign attributes
        self.matrix_dir = matrix_dir
        self.matrix_year = matrix_year
        self.time_period_proportions = time_period_proportions
        self.output_dir = output_dir

        # Create the output paths
        file_ops.create_folder(self.renamed_path)
        file_ops.create_folder(self.od_path)
        file_ops.create_folder(self.compiled_od_path)
        file_ops.create_folder(self.internal_compiled_od_path)

        # Rename the given matrices if needed
        self._rename_matrices(matrix_renaming)

    @property
    def renamed_path(self):
        """Path to the folder containing renamed matrices"""
        return self.output_dir / self._renamed_dirname

    @property
    def od_path(self):
        """Path to the folder containing initial OD matrices"""
        return self.output_dir / self._od_dirname

    @property
    def compiled_od_path(self):
        """Path to the folder containing compiled OD matrices"""
        return self.output_dir / self._compiled_od_dirname

    @property
    def internal_compiled_od_path(self):
        """Path to the folder containing compiled internal OD matrices"""
        return self.output_dir / self._internal_compiled_od_dirname

    def _rename_matrices(self, matrix_renaming: os.PathLike = None):
        """Rename the given matrices"""
        # Init
        in_col = 'input'
        out_col = 'output'
        expected_cols = [in_col, out_col]

        # Get a list of all the expected filenames
        expected_filenames = self._get_hb_internal_filenames()
        expected_filenames += self._get_nhb_internal_filenames()
        expected_filenames += self._get_external_filenames()

        # Build the list of files to copy
        if matrix_renaming is not None:
            # ## RENAME FILES WHILE WE COPY ## #
            # Validate the given file
            if not os.path.exists(matrix_renaming):
                raise ValueError(
                    "Cannot find the matrix renaming file given. Path does " "not exist."
                )

            # Read in and check that all cols exist
            rename_df = pd.read_csv(matrix_renaming)
            rename_df = pd_utils.reindex_cols(rename_df, expected_cols)

            # Check that all filenames exist in the out_col
            got_filenames = set(rename_df[out_col].to_list())
            missing_fnames = set(expected_filenames) - got_filenames
            if len(missing_fnames) > 0:
                raise ValueError(
                    "The given rename file does not contain all the expected "
                    "output filenames. The following filenames are missing:\n"
                    f"{missing_fnames}"
                )

            # Build the list of paths to copy
            in_paths = [self.matrix_dir / x for x in rename_df[in_col].to_list()]
            out_paths = [self.renamed_path / x for x in rename_df[out_col].to_list()]
            files_to_copy = list(zip(in_paths, out_paths))

        else:
            # ## JUST COPY FILES AS IS ## #
            # Make sure all files exist ???
            in_paths = [self.matrix_dir / x for x in expected_filenames]
            out_paths = [self.renamed_path / x for x in expected_filenames]

            # Build the list of files to copy
            files_to_copy = list(zip(in_paths, out_paths))

        # Copy and rename the files if needed
        file_ops.copy_and_rename_files(files=files_to_copy)

    def _get_filename(
        self,
        trip_origin: str = None,
        matrix_format: str = None,
        purpose: str = None,
        ca: str = None,
        tp: str = None,
        suffix: str = None,
    ) -> str:
        """Build a filename using the default format"""
        if matrix_format is None:
            matrix_format = "{purpose}"
        else:
            matrix_format += "_{purpose}"

        template = du.get_dist_name(
            trip_origin=trip_origin,
            matrix_format=matrix_format,
            year=str(self.matrix_year),
            mode=self._mode.get_mode_num(),
            car_availability=ca,
            tp=tp,
            suffix=suffix,
            csv=True,
        )
        return template.format(purpose=purpose)

    def _get_internal_filename(self, *args, **kwargs) -> str:
        """Build an internal filename using the default format"""
        return self._get_filename(*args, **dict(kwargs, suffix=self._internal_suffix))

    def _get_external_filename(self, *args, **kwargs) -> str:
        """Build an external filename using the default format"""
        return self._get_filename(*args, **dict(kwargs, suffix=self._external_suffix))

    def _get_hb_internal_filenames(self) -> List[str]:
        """Build a list of filenames needed for the int home-based pa files"""
        return list(self._get_hb_internal_filenames_with_segments().keys())

    def _get_nhb_internal_filenames(self) -> list[str]:
        """Build a list of filenames needed for the 24hr int non-home-based files"""
        return list(self._get_nhb_internal_filenames_with_segments().keys())

    def _get_external_filenames(self) -> list[str]:
        """Build a list of filenames needed for the 24hr external files"""
        return list(self._get_external_filenames_with_segments().keys())

    def _get_hb_internal_filenames_with_segments(self) -> Dict[str, Dict[str, Any]]:
        """Build a dictionary of filenames with segment_params as keys"""
        filenames_to_segments = dict()
        for segment_params in self._get_hb_internal_segments():
            fname = self._get_internal_filename(
                trip_origin='hb',
                matrix_format='pa',
                purpose=segment_params['p'],
                ca=segment_params['ca'],
            )
            filenames_to_segments[fname] = segment_params

        return filenames_to_segments

    def _get_nhb_internal_filenames_with_segments(self) -> Dict[str, Dict[str, Any]]:
        """Build a dictionary of filenames with segment_params as keys"""
        filenames_to_segments = dict()
        for segment_params in self._get_nhb_internal_segments():
            fname = self._get_internal_filename(
                trip_origin='nhb',
                matrix_format='od',
                purpose=segment_params['p'],
                ca=segment_params['ca'],
            )
            filenames_to_segments[fname] = segment_params

        return filenames_to_segments

    def _get_external_filenames_with_segments(self) -> Dict[str, Dict[str, Any]]:
        """Build a dictionary of filenames with segment_params as keys"""
        filenames_to_segments = dict()
        for segment_params in self._get_external_segments():
            fname = self._get_external_filename(
                matrix_format=segment_params['matrix_format'],
                purpose=segment_params['p'],
                ca=segment_params['ca'],
            )
            filenames_to_segments[fname] = segment_params

        return filenames_to_segments

    def _get_hb_internal_segments(self) -> List[Dict[str, Any]]:
        return [
            {'p': p, 'ca': ca}
            for p, ca in itertools.product(self._hb_purpose_vals, self._ca_vals)
        ]

    def _get_nhb_internal_segments(self) -> List[Dict[str, Any]]:
        return [
            {'p': p, 'ca': ca}
            for p, ca in itertools.product(self._nhb_purpose_vals, self._ca_vals)
        ]

    def _get_external_segments(self) -> List[Dict[str, Any]]:
        segment_params = [
            {'p': p, 'ca': ca}
            for p, ca in itertools.product(self._purpose_vals, self._ca_vals)
        ]

        # Add in matrix formats as it changes depending on CA
        final_segment_params = list()
        for params in segment_params:
            # Determine the matrix format
            if params['ca'] == 1:
                new_params = params.copy()
                new_params['matrix_format'] = 'od'
                final_segment_params.append(new_params)
            else:
                for matrix_format in ['od_from', 'od_to']:
                    new_params = params.copy()
                    new_params['matrix_format'] = matrix_format
                    final_segment_params.append(new_params)

        return final_segment_params

    def convert_hb_internal(self):
        """Convert the home-based internal matrices to tp split OD"""

        # Convert into OD-from and OD-to matrices and write out
        desc = "Converting segments to OD matrices"
        filename_dict = self._get_hb_internal_filenames_with_segments()
        for fname, segment_params in tqdm.tqdm(filename_dict.items(), desc=desc):
            # Grab the factors
            kwargs = {"purpose": segment_params['p'], "ca": segment_params['ca']}
            fh_factors = self.time_period_proportions.get_internal_fh_factors(**kwargs)
            th_factors = self.time_period_proportions.get_internal_th_factors(**kwargs)

            # Convert into od-from and od-to
            od_from, od_to = pa2od.to_od_via_fh_th_factors(
                pa_24=file_ops.read_df(self.renamed_path / fname, index_col=0),
                fh_factor_dict=fh_factors,
                th_factor_dict=th_factors,
                tp_needed=self.time_period_proportions.time_periods,
            )

            # Write out to disk
            for mat_format, mat_dict in [('od_from', od_from), ('od_to', od_to)]:
                for tp, mat in mat_dict.items():
                    mat = mat.round(8)
                    fname = self._get_internal_filename(
                        trip_origin='hb',
                        matrix_format=mat_format,
                        tp=tp,
                        **kwargs,  # purpose and ca
                    )
                    file_ops.write_df(mat, self.od_path / fname)

    def convert_nhb_internal(self):
        """Split the NHB internal OD matrices by time period"""

        # Convert into tp split and write out
        desc = "Splitting nhb internal by time periods"
        filename_dict = self._get_nhb_internal_filenames_with_segments()
        for fname, segment_params in tqdm.tqdm(filename_dict.items(), desc=desc):
            # Grab the factors
            kwargs = {"purpose": segment_params['p'], "ca": segment_params['ca']}
            split_factors = self.time_period_proportions.get_internal_nhb_tp_split_factors(**kwargs)

            # Convert into tp split matrices
            tp_mats = matrix_utils.split_matrix_by_time_periods(
                mat_24=file_ops.read_df(self.renamed_path / fname, index_col=0),
                tp_factor_dict=split_factors,
            )

            # Write out to disk
            for tp, mat in tp_mats.items():
                mat = mat.round(8)
                fname = self._get_internal_filename(
                    trip_origin='nhb',
                    matrix_format='od',
                    tp=tp,
                    **kwargs,  # purpose and ca
                )
                file_ops.write_df(mat, self.od_path / fname)

    def convert_external(self):
        """Split the external OD matrices by time period"""

        # Convert into tp split matrices and write out
        desc = "Splitting external by time periods"
        filename_dict = self._get_external_filenames_with_segments()
        for fname, segment_params in tqdm.tqdm(filename_dict.items(), desc=desc):
            # Grab the factors
            matrix_format = segment_params['matrix_format']
            if matrix_format == 'od':
                ca_key = 'nca'
                split_into_from_to = True
            elif matrix_format == 'od_from':
                ca_key = 'ca_fh'
                split_into_from_to = False
            elif matrix_format == 'od_to':
                ca_key = 'ca_th'
                split_into_from_to = False
            else:
                raise ValueError(f"Can't deal with matrix format '{matrix_format}'")

            split_factors = self.time_period_proportions.get_external_tp_split_factors(
                purpose=segment_params['p'],
                ca=ca_key,
            )

            # Convert into tp split matrices
            tp_mats = matrix_utils.split_matrix_by_time_periods(
                mat_24=file_ops.read_df(self.renamed_path / fname, index_col=0),
                tp_factor_dict=split_factors,
            )

            # Write matrix out to disk
            for tp, mat in tp_mats.items():
                mat = mat.round(8)

                if split_into_from_to:
                    # Split the matrix in half to get the od-from and od-to
                    # Assuming External is HB only here otherwise this doesn't
                    # hold
                    for matrix_format in ["od_from", "od_to"]:
                        temp_mat = mat * 0.5
                        fname = self._get_external_filename(
                            matrix_format=matrix_format,
                            purpose=segment_params['p'],
                            ca=segment_params['ca'],
                            tp=tp,
                        )
                        file_ops.write_df(temp_mat, self.od_path / fname)
                else:
                    # Just write out
                    fname = self._get_external_filename(
                        matrix_format=segment_params['matrix_format'],
                        purpose=segment_params['p'],
                        ca=segment_params['ca'],
                        tp=tp,
                    )
                    file_ops.write_df(mat, self.od_path / fname)


    def compile_matrices(self, compile_params_path: os.PathLike) -> None:
        """Compile the matrices into a more aggregate format"""
        matrix_processing.compile_matrices(
            mat_import=self.od_path,
            mat_export=self.compiled_od_path,
            compile_params_path=compile_params_path,
        )

    def compile_internal_matrices(self, compile_params_path: os.PathLike) -> None:
        """Compile the matrices into a more aggregate format"""
        matrix_processing.compile_matrices(
            mat_import=self.od_path,
            mat_export=self.internal_compiled_od_path,
            compile_params_path=compile_params_path,
        )


def get_norms_post_me_tp_proportions(
    norms_files: NormsTpProportionFiles,
    overwrite_extracted_pickle: bool = False,
) -> NoRMSPostMeTpProportions:
    """Create a NoRMSInternalPostMETourProportions object

    Checks what data is available and what steps have already been completed
    in `import_path` before creating a NoRMSInternalPostMETourProportions object.
    If the tour proportions have already been extracted from CUBE, then
    it will not be repeated, unless explicitly told to. Otherwise the data
    will be converted

    Parameters
    ----------
    norms_files:
        A NormsTpProportionFiles object stating where all the files
        needed to generate the time period proportions should be.

    overwrite_extracted_pickle:
        Whether to overwrite the data extracted from CUBE if it already exists.

    Returns
    -------

    """
    # Decide if we need to extract data from CUBE or not
    extracted_pickle_path = norms_files.get_extracted_pickle_path()
    if overwrite_extracted_pickle or not extracted_pickle_path.exists():
        # Create the pickle again
        LOG.info("Extracting time period proportions from CUBE")
        tp_proportion_extractor.main(
            norms_files.home_dir,
            extracted_pickle_path,
        )

    return NoRMSPostMeTpProportions(tour_prop_import=extracted_pickle_path)
