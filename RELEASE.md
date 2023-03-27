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

### V0.5.3
- Tools
  - Traveller Segment Tool
    - Added parameters for custom input matrix segmentation
    - Added flexible aggregation of input matrices to new segmentation
    - Output naming aligned with rest of framework
    - Front end run script finalised
    - Added summary reports
  - Renamed the incorrectly named "TemproExtractor" to "NTEM Extractor"
    - Updated tool to extract different versions of NTEM data
    - Updated tool to extract different scenarios of NTEM data
  - TLD Builder
    - Fixed a bug where demand on bounds was being dropped
    - Added functionality to generate dynamic bands to create a log curve for demand
- Core
  - Added Midlands Connect zoning systems as built-ins
  - Added functionality to convert matrices into SATURN and CUBE formats
  - Added functionality to extract matrices from SATURN and CUBE formats
- Models
  - Adapted NoTEM to optionally perform steps needed for Midlands Models
    -  Including optional localised trip end adjustments
  - Added a temporary front end script for `Midlands Distribution Model`
  - Fixed a bug in the default distribution model run, running at the incorrect time format
  - Created a generic front end for all forecasting functionality
    - Can run NTEM based forecasts, or synthetic trip end based forecasts from 
      the same front end
    - Updated the NTEM Forecasting model to select between different versions
      of NTEM data and different scenarios.
- Bug Fixes
  - Various errors to adapt code to work more flexibly with other zoning systems
