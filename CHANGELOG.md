# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a
Changelog](https://keepachangelog.com/en/1.0.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2021-02-22
### Added
- Added support for Generalized Information Bottleneck (GIB;
  introduced by [Strouse and Schwab
  2016](https://arxiv.org/abs/1604.00268)). This includes the
  "vanilla" IB and the Deterministic IB as special cases.
- Improved performance by vectorizing some key sections of the code.
- Added support for computing IB/DIB/GIB directly from a table giving
  a joint probability distribution p(x,y) rather than empirical data.
- Added detailed example notebook for the Deterministic Information
  Bottleneck (`examples/Deterministic-Bottleneck.ipynb`).  This
  reproduces a key figure from the DIB paper and showcases how easy it
  now is to turn a knob and go from IB to DIB via various flavours of
  GIB.
- Added example notebook (`examples/Compare-embo-dit.ipynb`) with a
  comparison vs [dit](https://doi.org/10.21105/joss.00738), another
  information theory package that also implements the IB.  Here we
  show that the two packages give matching results on a few sample
  problems (except when dit crashes), and we compare embo and dit
  speed-wise. Embo is much faster.
- Added tests for GIB (actually just took most previous tests and made
  them to run over a range of IB types, from basic IB to DIB passing
  by some intermediate ones)
- Added tests to check that our internal implementation of KL
  divergence and entropy give exactly the same results as the
  corresponding functions from SciPy (they should, as we use the same
  low-level functions from `scipy.special`).
- Added tests on randomly-generated empirical data, sampled from a
  fixed probability distribution. This is better than what we had
  before because it makes it easier to catch corner cases.
- Added tests on randomly-generated probability distributions of
  various shapes and sizes. This is motivated along the same lines as
  the previous point; only it's the whole distribution that is random
  here, not (just) the data.
- Always run all tests with multiple versions of NumPy
  (1.18,1.19,1.20).
- Add license notices to files containing the core code.
- Added changelog file.
### Changed
- Renamed `EmpiricalBottleneck` class to `InformationBottleneck`, and
  `get_empirical_bottleneck` method to `get_bottleneck`.
- Made a small fix to the convergence criterion of the algorithm. This
  is nicer from a motivation standpoint (we monitor the loss function
  instead of the candidate pmf for changes), and also matches what is
  done in other papers such as Strouse.
- Changed initialization rule for the bottleneck "encoder" (p(m|x)),
  using a recipe inspired by what was done in the DIB paper. This was
  done mostly to aid in the comparison with the figure from that paper
  in the example notebooks, but it seems like a nice thing to have in
  general so I've switched to that by default. The previous
  initialization method is still used when the dimensionality of M is
  lower than that of X.
- Improved handling of corrupted input (NaNs, Infs, empty arrays)
- Switched to [tox](https://tox.readthedocs.io/en/latest/index.html)
  for managing unit tests, as using pure setuptools via `python
  setup.py test` is now discouraged.
- Removed usage of dtypes such as `np.int`, which were generating
  numpy deprecation warnings in NumPy 1.20.
- Removed some somewhat misleading/unnecessary references to
  past/future bottleneck in the docstrings and comments, where more
  general notions of IB were involved.
