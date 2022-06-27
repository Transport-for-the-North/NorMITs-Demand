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

### V0.5.1
- Core
  - 
- NoTEM
  - 
- Distribution Model
  - 
- NTEM Forecasting
  - 
- Tools
  - TLD Tool
    - Built a class to handle the `CostDistributions`. Will be used internally to integrate 
      TLDs more efficiently.
    - Outputs graphs and a log of the run alongside the generated TLD csvs
    - Built a front end to generate all the TLDs needed for the Distribution Model
    - Integrated `SegmentationLevels` into TLD tool
  - Core
    - Updated how `SegmentationLevel` handles name generation. Segment types are now used
      to cast the outputs and NaN values are ignored to add support for 
      non-complete segmentations.
- Bug Fixes
  - Fixed a bug where segment parts were being added twice while generating segment names
  - 
