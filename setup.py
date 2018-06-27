try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
import os
import codecs
import json

VERSION_FILE = 'embo/version.json'
INSTALL_REQUIRES = ['numpy', 'scipy', 'numba']


# get current version
with open(VERSION_FILE, 'r') as _vf:
    version_data = json.load(_vf)
try:
    VERSION = version_data['version']
except KeyError:
    raise KeyError("check version file: no version number")


# get long description from README
here = os.path.abspath(os.path.dirname(__file__))
with codecs.open(os.path.join(here, "README.rst"), encoding="utf-8") as f:
    LONG_DESCRIPTION = f.read()

setup (name="embo",
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
           "Programming Language :: Python :: 3"
       ],
       packages=["embo",
                 "embo.test"],
       test_suite="embo.test")

