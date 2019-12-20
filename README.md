# EMBO - Empirical Bottleneck
[![License](https://img.shields.io/pypi/l/embo)](https://www.gnu.org/licenses/gpl-3.0.txt)
[![PyPI version](https://img.shields.io/pypi/v/embo.svg)](https://pypi.python.org/pypi/embo/)
[![Build status](https://img.shields.io/gitlab/pipeline/epiasini/embo)](https://gitlab.com/epiasini/embo/pipelines)

A Python implementation of the Information Bottleneck analysis
framework (Tishby, Pereira, Bialek 2000), especially geared towards
the analysis of concrete, finite-size data sets.

## Requirements

`embo` requires Python 3, `numpy` and `scipy`.

## Installation
To install the latest release, run:
``` bash
pip install embo
```
(depending on your system, you may need to use `pip3` instead of `pip`
in the command above).

### Testing
(requires `setuptools`). If `embo` is already installed on your
system, look for the copy of the `test_embo.py` script installed
alongside the rest of the `embo` files and execute it. For example:

``` bash
python /usr/lib/python3.X/site-packages/embo/test_embo.py
```

**Alternatively**, if you have downloaded the source, from within the
root folder of the source distribution run:

``` bash
python setup.py test
```

This should run through all tests specified in `embo/test`.

## Usage

You probably want to do something like this:
``` python
import numpy as np
from embo import empirical_bottleneck

# data sequences
x = np.array([0,0,0,1,0,1,0,1,0,1]*300)
y = np.array([1,0,1,0,1,0,1,0,1,0]*300)

# IB bound for different values of beta
i_p,i_f,beta,mi,H_x,H_y = empirical_bottleneck(x,y)
```

## More examples
A simple example of usage with synthetic data can be found in the
source distribution, located at `embo/examples/embo_example.ipynb`.

## Authors
`embo` is maintained by Eugenio Piasini, Alexandre Filipowicz and
Jonathan Levine.
