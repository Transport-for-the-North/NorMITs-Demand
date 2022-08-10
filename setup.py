from setuptools import setup
import versioneer

PACKAGE_NAME = "normits_demand"

setup(
    name=PACKAGE_NAME,
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="Transport for the North's synthetic demand tools",
    packages=[PACKAGE_NAME],
)
