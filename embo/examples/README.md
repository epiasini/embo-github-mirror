# Example notebooks
The Jupyter notebooks in this folder give usage examples for embo.

- [Basic-Example.ipynb](Basic-Example.ipynb): basics; how to compute
  and plot an IB bound.
- [Markov-Chains.ipynb](Markov-Chains.ipynb): using embo for
  *past-future bottleneck* type analyses on data from Markov chains.
- [Deterministic-Bottleneck.ipynb](Deterministic-Bottleneck.ipynb):
  Deterministic and Generalized Information Bottleneck. Here we
  reproduce a key figure from the Deterministic Bottleneck paper, and
  we explore the algorithm's behaviour as α changes from 0 to 1.
- [Compare-embo-dit.ipynb](Compare-embo-dit.ipynb): here we compare
  embo with [dit](https://pypi.org/project/dit) [[James et al
  2018]](https://doi.org/10.21105/joss.00738). We compare the
  solutions found by the two packages on a set of simple IB problems
  (including a problem taken from dit's documentation), and we show
  that embo is orders of magnitude faster than dit.

