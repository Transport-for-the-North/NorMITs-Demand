"""Module storing/retrieving package version information."""
import subprocess
from . import _version
__version__ = _version.get_versions()['version']


def git_repository_description() -> str:
    """Get repository commit/tag description from git.

    Uses command `git describe --tags --always --dirty` to extract
    tag and commit names, see https://git-scm.com/docs/git-describe
    for details on output format.

    Returns
    -------
    str
        String describing the current repository commit/tag.

    Raises
    ------
    RuntimeError
        If the `git describe` command fails.
    """
    try:
        comp_proc = subprocess.run(
            ["git", "describe", "--tags", "--always", "--dirty"],
            check=True,
            capture_output=True,
        )
        return comp_proc.stdout.strip().decode()
    except (FileNotFoundError, subprocess.CalledProcessError) as err:
        raise RuntimeError("could not determine git repository commit") from err


def version_info() -> str:
    """Text containing the package version and repository version if available."""
    nd_version = f"NorMITs-Demand v{__version__}"
    try:
        git_version = git_repository_description()
        return f"{nd_version}, repository version: {git_version}"
    except RuntimeError:
        return nd_version
