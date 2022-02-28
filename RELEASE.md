# Release Notes

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
page on GitHub. A log of all patches made between versions can also be found
there.

Below, a brief summary of patches made since the previous version can be found.

### V0.4.8
- Distribution Model
  - Cache was incorrectly not converting trip ends to average weekday.
    This has been corrected in this version of the model.
- General
  - Low memory matrix translations now return all multiprocessed results in
    order, instead of potentially out of order.
- Tools
  - A new folder has been created `run_tools`. This is for running tools that 
    generate data for the NorMITs Framework, but aren't a model themselves.
    The first tool to be added is the `TripLengthDistributionBuilder`.
  - Added version and zoning systems into the pre-me tour proportions
    path building.
- NTEM Forecast
  - Added an NTEM Forecast model. Similar to the EFS, but for NTEM trip ends
    specifically.
- Core
  - Updates to the DVector to allow almost any operation to be applied to a 
    DVector.
  - Added a division function to the Dvector.
  - Started moving towards using `.csv.bz2` compression by default.
    This removes dependencies on pandas versions and we can then depend on
    pandas `.to_csv()` and `.from_csv` to handle csv data I/O. 
