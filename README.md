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

## Class Versioning

Versioning follows the following version control, as defined in the TfN Coding Standards
(correct as of 20/01/21):

- Apply version control with the following format Epic.Beta.Alpha 
    - ‘Epic’ is a number corresponding to the Epic currently undertaken 
    - ‘Beta’ is the release version currently available to TfN as a whole 
    - ‘Alpha’ is the release version currently available to the TAME team 

A brief log of the changes and additions made to the NorMITs Demand codebase between
versions of classes. can be found below.



### ExternalForecastSystem

##### V2.4
 - Start of this version documentation.
 - Scenario front end added to allow easy running of different TfN Future Travel Scenarios.
 
##### V2.5
 - Updated Version number in ExternalForecastSystem - replaced underscores with dots. 
 - Additional reporting and audit checks added.
 - Internal PopEmpComparator added to check and report on the expected and produced population/emplyment data of the Production and Attraction models.
 - Reporting tool added to display a dashboard of outputs from the External Forecast System.
 
