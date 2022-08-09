from setuptools import setup
import versioneer

# BACKLOG: Properly integrate setup.py
#  labels: demand merge

PACKAGE_NAME = "normits_demand"

setup(
    name=PACKAGE_NAME,
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="Add description of NorMITs Demand",
    packages=[PACKAGE_NAME],
)
