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

## Ways Of Working

### GitHub

All new work should be worked on in a separate branch. For significant pieces of new functionality
a draft pull request should be opened during development of the code. This creates a central point
to communicate about the code being written and to keep a record of any potential issues. Once 
development is complete and the code ready for review, the draft pull request can be promoted to
a normal pull request.  

For smaller pieces of work, such as a bug fix, a normal pull request can be made straight away.


### Coding Style

NorMITs Demand follows the
[Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) with some exceptions:

- **Multiline `if`-statements** - To avoid a visual conflict with multiline `if`-statements this
codebase chooses to forgo the space between `if` and the opening bracket. A comment should
also be added, if reasonable. Therefore, the following:
```
if (this_is_one_thing and
    this_is_another_thing):
    do_something()
```
Would become:
```
if(this_is_one_thing and
   this_is_another_thing):
   # As both conditions are true, we should do the thing
    do_something()
```

- **Multiline `for`-statements** - Try to avoid these where possible to prevent (less obstructive)
visual conflicts. Instead, build the generator object before the `for`-statement, then call the
generator. Therefore, the following: 
```
for variable_one, variable_two, variable_three in zip(long_iterable_name_one,
                                                      long_iterable_name_two,
                                                      long_iterable_name_three):
    # Loop over one, two, and three
    do_something()
```
Would become:
```
one_two_three_iterator = zip(long_iterable_name_one,
                             long_iterable_name_two,
                             long_iterable_name_three)

for variable_one, variable_two, variable_three in one_two_three_iterator:
    # Loop over one, two, and three
    do_something()
```

- **Function definitions** - To clearly differentiate function definitions from function calls
at a glance, all function arguments should be on a separate line, in-line with the opening bracket.
The trailing bracket should also be in line with the arguments, on a new line (along with
return type hinting). 
Note: If all function arguments can fit on a single line with the function name, then that is the
preferred option. As an example:
```
def a_long_function_name(variable_name_one: int,
                         variable_name_two: int,
                         variable_name_three: int,
                         variable_name_four: int,
                         ) -> float:
    # Do something
    ...
```

### Special Code Comments
A few unique code comment prefixes have been chosen to help keep track of issues in the codebase.
They are noted below, with a brief description on when to use them:

- `# TODO:` Use this for small bits of future work, such as making a note of potential future
errors that should be checked for, or code that could be better written if you had more time.
- `# OPTIMISE:` Point out code that can be better optimised, but you don't have time/resources
right now, i.e. re-writing code in numpy in place of Pandas.
- `# BACKLOG:` Use to point out bigger pieces of work, such as where new (usually more complex)
functionality can be added in future. Can also be used to point out where assumptions have been
made in the codebase, and the backlog item can be used to track the issue.
 


## Class Versions

Brief log of the changes and additions made to the NorMITs Demand codebase between versions of
classes.

### ExternalForecastSystem

##### V2.4
 - Start of this version documentation.
 - Scenario front end added to allow easy running of different TfN Future Travel Scenarios.
 
##### V2.5
 - Updated Version number in ExternalForecastSystem - replaced underscores with dots. 
 - Additional reporting and audit checks added.
 - Internal PopEmpComparator added to check and report on the expected and produced population/emplyment data of the Production and Attraction models.
 - Reporting tool added to display a dashboard of outputs from the External Forecast System.
 
