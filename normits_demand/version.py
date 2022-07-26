"""Module storing/retrieving package version information."""

import re
import subprocess

__version__ = "0.5.1"

# fmt: off
# Regex taken from semver.org
SEMVER_REGEX = (
    r"^(?P<major>0|[1-9]\d*)"
    r"\.(?P<minor>0|[1-9]\d*)"
    r"\.(?P<patch>0|[1-9]\d*)"
    r"(?:-(?P<prerelease>(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)"
        r"(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))"
    r"?(?:\+(?P<buildmetadata>[0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$"
)
# fmt: on

version_match = re.search(SEMVER_REGEX, __version__)
MAJOR = version_match.group("major")
MINOR = version_match.group("minor")
PATCH = version_match.group("patch")
PRE_RELEASE = version_match.group("prerelease")
BUILD_META_DATA = version_match.group("buildmetadata")


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
