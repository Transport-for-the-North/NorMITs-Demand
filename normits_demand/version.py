from packaging import version

__version__ = "0.4.9"

v = version.parse(__version__)
MAJOR = v.major
MINOR = v.minor
PATCH = v.micro
