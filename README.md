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

### v0.3.0
Code has been refactored in this version - ready for merge with TMS

### v0.3.1
- TMS added to codebase.
- A bunch of minor fixes were added to get EFS and TMS working correctly.
- EFS PA to OD process was fixed for NoHAM outputs.

### v0.3.2
- Added a new EFS reporter class, capable of comparing EFS outputs to NTEM.
- Changed NTEM scenario future year growth for productions to match other scenarios.
Only constrained in base year, and multiplicative growth on top.

### v0.3.3
- Removed pop/emp growth from EFS. Future year population and employment
is now read in straight from NorMITs Land Use.
- Integration of NoRMS IO to and from PA NoRMS VDM compilation
- Fixes to bring NoHAM outputs closer to NTEM
    - Splitting out internal and external distribution
        * Internal distribution follows the furness
        *  External distribution is a grown post-ME PA matrix. Grown by synth FY / synth BY productions
    - Furness uses post-me matrices as is for seed matrices
    - post-me vector is now used for the base year P/A vectors (Internal demand only)
    - multiplicative growth added into the attractions
    - Attractions now take segmentation from the post-me matrices, Grows them using synth fy/by
    
### v0.3.4
- Updated the EFS output paths to no longer include the version number in the name, but align
better with TMS output paths.
    - Paths look more like `NorMITs Demand/nohmam/EFS/iter1/NTEM/...`
    - The Land Use iteration being used is now included in the inputs.txt
- Updated the PopEmpComparator to use the new land use imports to compare to.
- Fixed issue in Production/Attraction models dropping population and employment
- Updated NoRMS I/O to decompile post-ME and compile outputs correctly for NoRMS
    - Fixed an issue with demand being dropped - issue added to backlog for a full fix 
- CS - Write updates to TMS since last merge
- Added Mobile Network Data LOT 2 segmentor and privacy masker
- Future years changed - 2035 has been swapped for 2040

### v0.3.5
- Updates to the elasticity model:
    -  Renamed some variables for readability
    - Add in inter/intra sector cost adjustments
    - Added a new front end to allow easy calling per scenario
    - Optimised the model
        - Multiprocessed the elasticity
        - Able to read in / write out compressed matrices 
        - Multiprocessed matrix compilation
    - Added script to analyse the elasticity outputs
    - Added progress bar to multiprocessing
    - Started off a cost builder class for handling all costs
    - Elasticity now operates on internal demand only
    - Reads in new land use data, split by future years
    - Added in WFH adjustments after matrix distribution
- Updated run_tms.py to export a text file containing inputs
- Added versioning to run_tms.py
- Added versioning to reports_audits.py and updated code to include versioning 
as part of their file name in distribution reports

### v0.3.6
- Updated EFS Production and Attraction models to accept new Land Use outputs
    - Note that this is no longer backwards compatible. If old (pre iter3d)
      land use inputs are needed, you need to revert to an older version of 
      NorMITs Demand.
      
### v0.3.7
- Updated the Furness so productions are done last in each iteration. This 
  means productions will always closely match, rather than the attractions.


### v0.4.0 (NoTEM Development)
- Added the SegmentationLevel Class
- Added the ZoningSystem Class
- Added the DVector class (built on two previous classes)
- Added the new NoTEM Production model - built on top of DVec
    - Uses new, more disaggregate segmentation.
    - Optimised for run times using numpy.
- Added the new NoTEM Attraction model - built on top of DVec
    - Uses new, more disaggregate segmentation.
    - Optimised for run times using numpy.
- Added the new NoTEM NHB Production model - built on top of DVec
    - Uses new, more disaggregate segmentation.
    - Optimised for run times using numpy.
- Added the new NoTEM NHB Attraction model - built on top of DVec
    - Uses new, more disaggregate segmentation.
    - Optimised for run times using numpy.    
- Added a new NoTEM wrapper around the new production and attraction models 
- Added new pathing module, which can be used to access NoTEM output paths


### v0.4.1 (NoTEM Enhancements)
- SegmentationLevel and ZoningSystem have been fully documented
- SegmentationLevel and ZoningSystem have had read only properties added
- Segmentation names have changed. NoTEM specific segmentations now have 
  'notem_' prefixed onto their name
- Logging module has been added to codebase. Partially integrated into NoTEM 


### v0.4.2
- A number of minor fixes have been added to get TMS up and running again

### v0.4.3
- Updates to the elasticity model
    - Fix Access/Egress costs changes
    - Update parameters as we are now benchmarking OD matrices against
      NELUM
    - Optimised PA2OD Process
    - Tour Proportions generated for rail PA2OD
    
### v0.4.4
- Integrating NoTEM outputs into TMS
- Number of minor fixes to TMS
    - TMS codebase has been marginally tidied up
    - External model has been updated to use 2 TLD instead of 1
    - External model has been optimised in some places
    - Gravity Model has been update to work with new External Model
    
### v0.4.5 (TMS Overhaul)
- Added internal and external zone definitions into ZoningSystem Class
- Object Orientation of TMS
    - Can now run full multi-modal
    - Added constants class in to define running Mode
- External Model Updates to target a different TLD for internal and external
- Gravity Model code updates to bring in line with OO structure
  (still awaiting full re-write)
- Updated filenames and output paths for TMS
    - A path class is now used for imports and exports, making it easier
      for other models to interact with TMS
    - filenames have been updated to bring in line with EFS naming, forming
      the standard NorMITs Demand naming
- PA to OD process updated to use tour proportions and standard code shared
  with EFS
- Assignment model code updated to use standard code shared with EFS.
- Various tidy ups
    - Constants moved around to make more sense
    - Some utils have been given dedicated utils files to make them easier
      to find.
    - A lot of legacy code has been removed

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
- When reading in pickled files, the `__version__` attribute will now be 
  checked (if it exists). It will be compared to the version of the code being
  run. If different, a warning will be raised.
- Awaiting Tram integratiobefore updating to next version...

     
