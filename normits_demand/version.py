from packaging import version

__version__ = "V0.3.5"

v = version.parse(__version__)
MAJOR = v.major
MINOR = v.minor
PATCH = v.micro
