EMBO - Empirical Bottleneck
===========================

A Python implementation of the Information Bottleneck analysis
framework (Tishby, Pereira, Bialek 2000), especially geared towards
the analysis of concrete, finite-size data sets.

Installation
------------

For the moment, just add this folder to your `PYTHONPATH` by doing
something like
::
   import sys
   sys.path.append("../src/embo") 
   from embo import embo

You should even be able to install via `python3 setup.py install`, but
I'd discourage it for the moment while this library is still expected
to change often. Later on, it will be very easy to upload this to the
PyPI and make it accessible via `pip`.

Testing
-------
From within the root folder of the package (i.e. this folder), run
.. codeblock:: bash
	       python3 setup.py test

This should run through all tests specified in `embo/test`. These
should generate a fair number of `numba` warnings, but they should run
successfully (look for the summary at the end of the output).

Usage
-----

You probably want to do something like this:
::
   from embo.embo import empirical_bottleneck

   # data sequences
   x = np.array([0,0,0,1,0,1,0,1,0,1]*300)
   y = np.array([1,0,1,0,1,0,1,0,1,0]*300)

   # IB bound for different values of beta
   i_p,i_f,beta,mi = empirical_bottleneck(x,y,2,2)

   
   


