import re

__version__ = "0.4.8"

# fmt: off
# Regex taken from semver.org
semver_regex = (
    "^(?P<major>0|[1-9]\d*)"
    "\.(?P<minor>0|[1-9]\d*)"
    "\.(?P<patch>0|[1-9]\d*)"
    "(?:-(?P<prerelease>(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)"
        "(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))"
    "?(?:\+(?P<buildmetadata>[0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$")
# fmt: on

version_match = re.search(semver_regex, __version__)
MAJOR = version_match.group('major')
MINOR = version_match.group('minor')
PATCH = version_match.group('patch')
PRE_RELEASE = version_match.group('prerelease')
BUILD_META_DATA = version_match.group('buildmetadata')
