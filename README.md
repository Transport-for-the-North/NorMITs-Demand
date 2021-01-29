# NorMITs-Demand

## Python Environment

The Python version and package requirements are given in "Pipfile" and "Pipfile.lock",
these files are created and maintained automatically by [pipenv](https://pypi.org/project/pipenv/).
To install the packages required to run the External Forecast System first install Python 3.8 and 
run the following command to install pipenv.

`pip install pipenv`

Once pipenv has been installed successfully then run the `pipenv install` (from within
the NorMITs-Demand folder), which will read "Pipfile.lock" and install all the packages required.

Once the above command has ran you can activate the environment by running `pipenv shell` then
you can run any Python files with the usual `python` command. To add any new Python
packages to the environment use `pipenv install package_name` instead of pip, this will download
and install the package and update "Pipfile" and "Pipfile.lock" with the version information.

## Testing
Unit tests have been created using the pytest package and can be ran automatically using
the command `pytest` in the main repository folder. Pytest will search for any files named
`test_*.py` or `*_test.py` and run all tests within them, see
[pytest documentation](https://docs.pytest.org/en/stable/) for more information.

## Versioning
The NorMITs Demand codebase follows [Semantic Versioning](https://semver.org/); the convention
for most software products. In Summary this means the version numbers should be read in the 
following way.

Given a version number MAJOR.MINOR.PATCH (e.g. 1.0.0), increment the:

- MAJOR version when you make incompatible API changes,
- MINOR version when you add functionality in a backwards compatible manner, and
- PATCH version when you make backwards compatible bug fixes.

Below, a brief summary of previous version can be found. Note that this should be moved in future
to reflect future releases to avoid filling up this readme with version notes.

### v0.1.0
Initial version number. Before the TMS codebase has been merged into this codebase (which is really
only EFS). No Elasticity model is included, and neither is the exceptional/bespoke zones. The first 
version of EFS outputs can be found from this version onwards.

### v0.2.0
The Elasticity Model and Bespoke Zones code now exist in the codebase - although they are not fully
integrated. The D-Log has been added and can be optionally integrated. Observed data can now be 
properly integrated into the base year.

### v0.2.1
Updated the furness auditing so summary files per year are now written to disk.
Versioning has also been updated in the EFS class - meaning the version for the class is now
pulled straight from the module version.

Code has been refactored in this version - ready for merge with TMS
