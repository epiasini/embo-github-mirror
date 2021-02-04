from setuptools import setup
import os
import codecs

NAME = "embo"
VERSION_FILE = "VERSION"
INSTALL_REQUIRES = ["numpy"]

here = os.path.abspath(os.path.dirname(__file__))

# get current version
with open(os.path.join(here, NAME, VERSION_FILE)) as version_file:
    VERSION = version_file.read().strip()

# get long description from README
with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    LONG_DESCRIPTION = f.read()

setup (name=NAME,
       version=VERSION,
       url="https://gitlab.com/epiasini/{}".format(NAME),
       description="Empirical Information Bottleneck",
       long_description=LONG_DESCRIPTION,
       long_description_content_type="text/markdown",
       install_requires=INSTALL_REQUIRES,
       python_requires=">=3",
       author="Eugenio Piasini",
       author_email="epiasini@sas.upenn.edu",
       license="GPLv3+",
       classifiers=[
           "Development Status :: 4 - Beta",
           "Intended Audience :: Science/Research",
           "Topic :: Scientific/Engineering :: Information Analysis",
           "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
           "Programming Language :: Python :: 3",
           "Operating System :: OS Independent",
           "Environment :: Console"
       ],
       packages=[NAME,
                 "{}.test".format(NAME)],
       test_suite="{}.test".format(NAME),
       include_package_data=True)
