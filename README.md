# EMBO - Empirical Bottleneck
[![License](https://img.shields.io/pypi/l/embo)](https://www.gnu.org/licenses/gpl-3.0.txt)
[![PyPI version](https://img.shields.io/pypi/v/embo.svg)](https://pypi.python.org/pypi/embo/)
[![Build status](https://img.shields.io/gitlab/pipeline/epiasini/embo)](https://gitlab.com/epiasini/embo/pipelines)

A Python implementation of the Information Bottleneck analysis
framework [[Tishby, Pereira, Bialek
2001]](https://arxiv.org/abs/physics/0004057), especially geared
towards the analysis of concrete, finite-size data sets.

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

### The Information Bottleneck
We refer to [[Tishby, Pereira, Bialek
2001]](https://arxiv.org/abs/physics/0004057) for a general
introduction to the Information Bottleneck. Briefly, if X and Y
are two random variables, we are interested in finding another random
variable M (called the "bottleneck" variable) that solves the
following optimisation problem:

min_{p(m|x)}I(M:X) - β I(M:Y)

for any β>0, and where M is constrained to be independent on
Y conditional on X:

p(x,m,y) = p(x)p(m|x)p(y|x)


Intuitively, we want to find the stochastic mapping p(M|X) that
extracts from X as much information about Y as possible while
forgetting all irrelevant information. β is a free parameter
that sets the relative importance of forgetting irrelevant information
versus remembering useful information. Usually, one is interested in
the curve described by I(M:X) and I(M:Y) at the solution of the
bottleneck problem for a range of values of β. This curve
gives the optimal tradeoff of compression and prediction, telling us
what is the minimum amount of information one needs to know about X
to be able to predict Y to a certain accuracy, or vice versa, what
is the maximum accuracy one can have in predicting Y given a certain
amount of information about X.

### Using `embo`
In embo, we assume that the true joint distribution of X and Y is
not available, and that we only have a set of joint empirical
observations. We also assume that X and Y both take on a finite
number of discrete values. In its most basic usage, the
`empirical_bottleneck` function takes as arguments an array of
observations for X and an (equally long) array of observations for
Y, and it returns a set of β values and the
optimal values of I(M:X) and I(M:Y) corresponding to those
β. The optimal tradeoff can then be visualised by plotting
I(M:Y) vs I(M:Y).

For instance:

``` python
import numpy as np
from matplotlib import pyplot as plt
from embo import empirical_bottleneck

# data sequences
x = np.array([0,0,0,1,0,1,0,1,0,1])
y = np.array([0,1,0,1,0,1,0,1,0,1])

# compute the IB bound from the data
I_x,I_y,_,_,_,_ = empirical_bottleneck(x,y)

# plot the optimal compression-prediction bound
plt.plot(I_x,I_y)
```

### More examples
A simple example of usage with synthetic data is located at
[embo/examples/Basic-Example.ipynb](embo/examples/Basic-Example.ipynb).
A more meaningful example is located at
[embo/examples/Markov-Chains.ipynb](embo/examples/Markov-Chains.ipynb),
where we compute the Information Bottleneck between the past and the
future of time series generated from different Markov chains.

## Authors
`embo` is maintained by Eugenio Piasini, Alexandre Filipowicz and
Jonathan Levine.
