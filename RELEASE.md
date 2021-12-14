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

### V0.4.7
- TramModel
  - takes further segmentation from NoTEM Bus Trip Ends
  - Fixed an issue so the attractions are balanced to the productions 
- TMS Updates
  - Optional read in of TramModel trip ends into TMS
  - Gravity Model Updates
    - Updated to read in initial cost params based on cost function names
      by default
    - Updated the furness to calculate Root Mean Squared Error weighted by the
      number of zones