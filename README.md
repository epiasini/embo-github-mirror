# EMBO - Empirical Bottleneck
[![License](https://img.shields.io/pypi/l/embo)](https://www.gnu.org/licenses/gpl-3.0.txt)
[![PyPI version](https://img.shields.io/pypi/v/embo.svg)](https://pypi.python.org/pypi/embo/)
[![Build status](https://img.shields.io/gitlab/pipeline/epiasini/embo)](https://gitlab.com/epiasini/embo/pipelines)

A Python package for working with the Information Bottleneck [[Tishby,
Pereira, Bialek 2001]](https://arxiv.org/abs/physics/0004057) and the
Deterministic (and Generalized) Information Bottleneck [[Strouse and
Schwab 2016]](https://arxiv.org/abs/1604.00268). Embo is especially
geared towards the analysis of concrete, finite-size data sets.

## Requirements

`embo` requires Python 3, `numpy>=1.7` and `scipy`.

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
python /usr/lib/python3.X/site-packages/embo/test/test_embo.py
```

## Usage

### The Information Bottleneck
We refer to [[Tishby, Pereira, Bialek
2001]](https://arxiv.org/abs/physics/0004057) and [[Strouse and Schwab
2016]](https://arxiv.org/abs/1604.00268) for a general introduction to
the Information Bottleneck. Briefly, if X and Y are two random
variables, we are interested in finding another random variable M
(called the "bottleneck" variable) that solves the following
optimisation problem:

min_{p(m|x)} = H(M) - α H(M|X) - β I(M:Y)

for any β>0 and 0≤α≤1, and where M is constrained to be independent on
Y conditional on X:

p(x,m,y) = p(x)p(m|x)p(y|x)

Intuitively, we want to find the stochastic mapping p(M|X) that
extracts from X as much information about Y as possible while
forgetting all irrelevant information. β is a free parameter that sets
the relative importance of forgetting irrelevant information versus
remembering useful information. α determines what notion of
"forgetting" we use: α=1 ("vanilla" bottleneck or IB) implies that we
want to minimise the mutual information I(M:X), α=0 (deterministic
bottleneck or DIB) that we want to make M a good *compression* of X by
minimising its entropy H(M), and intermediate values interpolate
between these two conditions.

Typically, one is interested in the curve described by I(M:Y) as a
function of I(M:X) or H(M) at the solution of the bottleneck problem
for a range of values of β. This curve gives the optimal tradeoff of
compression and prediction, telling us what is the minimum amount of
information one needs to know about X (or minimum amount of entropy
one needs to retain) to be able to predict Y to a certain accuracy, or
vice versa, what is the maximum accuracy one can have in predicting Y
given a certain amount of information about X.

### Using `embo`
Embo can solve the information bottleneck problem for discrete random
variables starting from a set of joint empirical observations. The
main point of entry to the package is the `InformationBottleneck`
class. In its constructor, `InformationBottleneck` takes as arguments an
array of observations for X and an (equally long) array of
observations for Y, together with other optional parameters (see the
docstring for details). In the most basic use case, users can call the
`get_bottleneck` method of an `InformationBottleneck` object, which will
assume α=1 and return a set of β values and the optimal values of
I(M:X), I(M:Y) and H(M) corresponding to those β. The optimal tradeoff
can then be visualised by plotting I(M:Y) vs I(M:Y).

For instance:

``` python
import numpy as np
from matplotlib import pyplot as plt
from embo import InformationBottleneck

# data sequences
x = np.array([0,0,0,1,0,1,0,1,0,1])
y = np.array([0,1,0,1,0,1,0,1,0,1])

# compute the IB bound from the data (vanilla IB; Tishby et al 2001)
I_x,I_y,H_m,β = InformationBottleneck(x,y).get_bottleneck()

# plot the IB bound
plt.plot(I_x,I_y)
```

Embo can also operate starting from a joint (X,Y) probability
distribution, encoded as a 2D array containing the probability
of each combination of states for X and Y.

``` python
# define joint probability mass function for a 2x2 joint pmf
pxy = np.array([[0.1, 0.4],[0.35, 0.15]]),

# compute IB
I_x,I_y,H_m,β = InformationBottleneck(pxy=pxy).get_bottleneck()

# plot I(M:Y) vs I(M:X)
plt.plot(I_x,I_y)
```

The deterministic and generalised bottleneck can be computed by
setting appropriately the parameter `alpha`:

``` python
# compute Deterministic Information Bottleneck (Strouse 2016)
I_x,I_y,H_m,β = InformationBottleneck(pxy=pxy, alpha=0).get_bottleneck()

# plot I(M:Y) vs H(M)
plt.plot(H_m,I_y)
```

### More examples
The `embo/examples` directory contains some Jupyter notebook that
should exemplify most of the package's functionality.

- [Basic-Example.ipynb](embo/examples/Basic-Example.ipynb): basics;
  how to compute and plot an IB bound.
- [Markov-Chains.ipynb](embo/examples/Markov-Chains.ipynb): using embo
  for *past-future bottleneck* type analyses on data from Markov
  chains.
- [Deterministic-Bottleneck.ipynb](embo/examples/Deterministic-Bottleneck.ipynb):
  Deterministic and Generalized Information Bottleneck. Here we
  reproduce a key figure from the Deterministic Bottleneck paper, and
  we explore the algorithm's behaviour as α changes from 0 to 1.
- [Compare-embo-dit.ipynb](embo/examples/Compare-embo-dit.ipynb): here
  we compare embo with [dit](https://pypi.org/project/dit) [[James et
  al 2018]](https://doi.org/10.21105/joss.00738). We compare the
  solutions found by the two packages on a set of simple IB problems
  (including a problem taken from dit's documentation), and we show
  that embo is orders of magnitude faster than dit.

### Further details
For more details, please consult the docstrings in
`InformationBottleneck`.

## Changelog
See the [CHANGELOG.md](CHANGELOG.md) file for a list of changes from
older versions.

## Authors
`embo` is maintained by Eugenio Piasini, Alexandre Filipowicz and
Jonathan Levine.
