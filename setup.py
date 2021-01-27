from pathlib import Path
from setuptools import setup

# BACKLOG: Properly integrate setup.py
#  labels: demand merge


def get_version(pkg_name):
    version_filename = Path(__file__).parent / pkg_name / "version.py"
    with open(version_filename) as f:
        for line in f.readlines():
            if line.startswith("__version__"):
                sep = '"' if '"' in line else "'"
                return line.split(sep)[1]
    raise RuntimeError(f"Version not found in {version_filename}")


PACKAGE_NAME = "NorMITs_Demand"

setup(
    name=PACKAGE_NAME,
    version=get_version(PACKAGE_NAME),
    description="Add description of NorMITs Demand",
    packages=[PACKAGE_NAME],
)
