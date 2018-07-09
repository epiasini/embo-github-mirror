try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
import os
import codecs

NAME = 'embo'
VERSION_FILE = 'VERSION'
INSTALL_REQUIRES = ['numpy', 'scipy']

here = os.path.abspath(os.path.dirname(__file__))

# get current version
with open(os.path.join(here, NAME, VERSION_FILE)) as version_file:
    VERSION = version_file.read().strip()

# get long description from README
with codecs.open(os.path.join(here, "README.rst"), encoding="utf-8") as f:
    LONG_DESCRIPTION = f.read()

setup (name=NAME,
       version=VERSION,
       url="https://gitlab.com/epiasini/embo",
       description="Empirical Information Bottleneck",
       long_description=LONG_DESCRIPTION,
       install_requires=INSTALL_REQUIRES,
       author="Eugenio Piasini",
       author_email="eugenio.piasini@gmail.com",
       license="GPLv3+",
       classifiers=[
           "Development Status :: 3 - Alpha",
           "Intended Audience :: Science/Research",
           "Topic :: Scientific/Engineering :: Information Analysis",
           "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)"
           "Programming Language :: Python :: 2.7",
           "Programming Language :: Python :: 3"
       ],
       packages=["embo",
                 "embo.test"],
       test_suite="embo.test",
       data_files=[(NAME, [os.path.join(NAME,VERSION_FILE)])])

