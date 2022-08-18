## Python Environment

The Python version and package requirements are listed in [requirements.txt](requirements.txt),
these can be installed into a conda environment using the [conda_install](conda_install.bat) batch
file.

### Conda Environments
Conda environments are folders which contain a specific collection of Python packages and specific
version of Python. They allow you to have multiple versions of Python with different packages
installed on a single PC and switch between them whenever required.

There are two methods for installing and using conda environments:
- [Anaconda](https://www.anaconda.com/products/distribution): GUI software with comes with lots of
  data science packages pre-installed and allows creating / managing conda environments within the
  GUI.
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html): smaller installation which just
  includes the conda command-line tool, can install all the same packages as Anaconda but the
  environments are created / managed using the command-line program.

## Testing
Unit tests have been created using the pytest package and can be ran automatically using
the command `pytest` in the main repository folder. Pytest will search for any files named
`test_*.py` or `*_test.py` and run all tests within them, see
[pytest documentation](https://docs.pytest.org/en/stable/) for more information.