# -*- coding: utf-8 -*-
"""
    Module for converting matrices to/from CUBE's .mat format.
"""

##### IMPORTS #####
# Standard imports
from pathlib import Path
import re
import subprocess

# Third party imports

# Local imports
from normits_demand import logging as nd_log
from normits_demand.utils import general

##### CONSTANTS #####
LOG = nd_log.get_logger(__name__)

##### CLASSES #####
class CUBEMatConverterError(general.NormitsDemandError):
    """Errors when converting to/from CUBE's .mat format."""


class CUBEMatConverter:
    """Class for converting to/from CUBE's .mat format.

    Parameters
    ----------
    cube_path : Path
        Path to the CUBE Voyager executable file.

    Raises
    ------
    FileNotFoundError
        If `cube_voyager_path` doesn't exist, or isn't a file.
    """

    def __init__(self, voyager_path: Path) -> None:
        self.voyager_path = voyager_path
        if not self.voyager_path.is_file():
            raise FileNotFoundError(f"cannot find CUBE Voyager: {self.voyager_path}")

    def csv_to_mat(
        self,
        num_zones: int,
        csv_paths: dict[str, Path],
        mat_path: Path,
        mat_factor: float = 1,
    ) -> Path:
        """Convert CSVs to CUBE's .mat format.

        CSVs should have 3 columns: origin, destination and trips,
        with no header row.

        Parameters
        ----------
        num_zones : int
            Number of zones in the matrix.
        csv_paths : dict[str, Path]
            Paths to the CSVs to be used for creating the matrix,
            the CSVs should have 3 columns: origin, destination and trips
            with no header row. The dictionary keys provide the name for
            the matrix level in the output file.
        mat_path : Path
            Output CUBE .mat file to create.
        mat_factor : float, default
            Factor to divide matrix by upon creation.

        Returns
        -------
        Path
            CUBE matrix created.

        Raises
        ------
        FileNotFoundError
            If any of the input CSVs don't exist.
        CUBEMatConverterError
            If the process fails creating the CUBE matrix.
        """
        LOG.info("Converting CSVs to CUBE .mat format")
        for path in csv_paths.values():
            if not path.is_file():
                raise FileNotFoundError(f"cannot find CSV: {path}")

        if mat_path.suffix != ".mat":
            mat_path.with_suffix(".mat")

        # Create CUBE Voyager script
        script_text = [
            "RUN PGM=MATRIX",
            f'FILEO MATO[1]="{mat_path.resolve()}",',
            f"      mo=1-{len(csv_paths)},dec={len(csv_paths)}*d,name="
            + ",".join(str(nm) for nm in csv_paths),
        ]
        for n, path in enumerate(csv_paths.values(), 1):
            script_text.append(f'FILEI MATI[{n}]="{path.resolve()}",')
            script_text.append("      fields=#1,2,3, pattern=ij:v")
        script_text += [
            "",
            f"zones={num_zones}",
            "fillmw",
        ]
        script_text += [f"mw[{n}]=mi.{n}.1/{mat_factor}" for n in range(1, len(csv_paths) + 1)]
        script_text += ["", "ENDRUN"]

        script_path = mat_path.with_name(mat_path.stem + "-CONVERSION.s")
        with open(script_path, "wt") as file:
            file.write("\n".join(script_text))
        LOG.debug("Written: %s", script_path)

        # Run CUBE
        args = [
            self.voyager_path.resolve(),
            script_path.resolve(),
            "-Pvdmi",
            "/Start",
            "/Hide",
            "/HideScript",
        ]
        comp_proc = subprocess.run([str(a) for a in args], capture_output=True, check=False)
        LOG.debug(
            "CSV to CUBE .mat Voyager output:%s%s",
            _stdout_decode(comp_proc.stdout),
            _stdout_decode(comp_proc.stderr),
        )

        if not mat_path.is_file():
            raise CUBEMatConverterError("error converting CSV to CUBE .mat")

        # Cleanup files
        script_path.unlink()
        script_path.with_name("TPPL.PRJ").unlink()
        del_pat = re.compile(r"(vdmi.*)\.(prn|var)", re.I)
        for path in script_path.parent.iterdir():
            match = del_pat.match(path.name)
            if match:
                path.unlink()

        return mat_path


##### FUNCTIONS #####
def _stdout_decode(stdout: bytes) -> str:
    """Convert bytes to string starting with a newline, or return empty string."""
    stdout = stdout.decode().strip()
    if stdout != "":
        stdout = "\n" + stdout
    return stdout
