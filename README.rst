EMBO - Empirical Bottleneck
===========================

A Python implementation of the Information Bottleneck analysis
framework (Tishby, Pereira, Bialek 2000), especially geared towards
the analysis of concrete, finite-size data sets.

Installation
------------

For the moment, just clone this repo by doing something like

.. codeblock:: bash
   cd /home/username/src
   git clone git@gitlab.com:epiasini/embo.git
   
and add its location to your `PYTHONPATH`:

::
   import sys
   sys.path.append("/home/username/src/embo") 

Alternatively, you can simply install via `python setup.py install
--user`, but I'd discourage it for the moment if you feel like you
will be making changes to the library as you use it. From a technical
standpoint, this package is essentially ready for upload to the Python
Package Index and distribution via `pip`, so this will be easy to do
when we feel like sharing it more publicly.

Testing
-------
From within the root folder of the package (i.e. this folder), run
.. codeblock:: bash
   python3 setup.py test

This should run through all tests specified in `embo/test`. These may
generate some `numba` warnings, but they should run successfully (look
for the summary at the end of the output).

Usage
-----

You probably want to do something like this:
::
   from embo.embo import empirical_bottleneck

   # data sequences
   x = np.array([0,0,0,1,0,1,0,1,0,1]*300)
   y = np.array([1,0,1,0,1,0,1,0,1,0]*300)

   # IB bound for different values of beta
   i_p,i_f,beta,mi,H_x,H_y = empirical_bottleneck(x,y)

   


