# NorMITs Demand

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

Note that the master branch of this repository contains a work in progress, and  may **not**
contain a stable version of the codebase. We aim to keep the master branch stable, but for the
most stable versions, please see the
[releases](https://github.com/Transport-for-the-North/NorMITs-Demand/releases)
page on GitHub.

Below, a brief summary of patches made since the previous version can be found.


### V0.4.6
- Minor Updates to NoTEM
  - Logging properly integrated into NoTEM
  - New HB Production Trip rates being read in
  - New NHB Production Trip rates being read in
    - NoTEM now outputs NHB Purpose 11
  - Soc 4 Added into HB and NHB models. This describes work-based trips
    for people without a formal job.
  - Segmentations now have typing - meaning `DVector.to_df()` should always 
    return in the correct datatypes.
  - Spatial balance added to Attraction Models. By default spatial balance 
    covers Government Office Regions (GORs) for each spatial balance.
- Added a Tram Model
  - Tram Model added which is able to integrate tram production and attraction
    vectors into NoTEM outputs, assuming tram trip were originally nested into
    NoTEM train trips
- When reading in pickled files, the `__version__` attribute will now be 
  checked (if it exists). It will be compared to the version of the code being
  run. If different, a warning will be raised.
- When reading in pickled files the process count attribute will be reset
  for the current system.
- Added a script to generate pre-me tour props based on NTS phis and NoTEM
  tp splits.

     
