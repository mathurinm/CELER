celer
=====

|image0| |image1|

Fast algorithm to solve Lasso-like problems with dual extrapolation. Currently, the package handles the following problems:

- Lasso
- weighted Lasso
- adaptive Lasso (:math:`\ell_{0.5}` penalized least-squares, solved with iterative reweighting/majorization-minimization)
- Sparse Logistic regression
- Group Lasso
- Multitask Lasso.

The estimators follow the scikit-learn API, come with automated parallel cross-validation, and support both sparse and dense data, with optionally feature centering, normalization, and unpenalized intercept fitting.
The solvers used allow for solving large scale problems with millions of features, up to 100 times faster than scikit-learn.

Documentation
=============

Please visit https://mathurinm.github.io/celer/ for the latest version
of the documentation.

Install the released version
============================

Assuming you have a working Python environment, e.g., with Anaconda you
can `install celer with pip <https://pypi.python.org/pypi/celer/>`__.

From a console or terminal install celer with pip:

::

    pip install -U celer

Install and work with the development version
=============================================

From a console or terminal clone the repository and install Celer:

::

    git clone https://github.com/mathurinm/celer.git
    cd celer/
    pip install -e .

To build the documentation you will need to run:


::

    pip install -U sphinx_gallery sphinx_bootstrap_theme
    cd doc/
    make html


Demos & Examples
================

In the `example section <https://mathurinm.github.io/celer/auto_examples/index.html>`__ of the documentation,
you will find numerous examples on real life datasets,
timing comparison with other estimators, easy and fast ways to perform cross validation, etc.


Dependencies
============

All dependencies are in the ``./requirements.txt`` file.
They are installed automatically when ``pip install -e .`` is run.

Cite
====

If you use this code, please cite:

::

    @InProceedings{pmlr-v80-massias18a,
      title = 	 {Celer: a Fast Solver for the Lasso with Dual Extrapolation},
      author = 	 {Massias, Mathurin and Gramfort, Alexandre and Salmon, Joseph},
      booktitle = 	 {Proceedings of the 35th International Conference on Machine Learning},
      pages = 	 {3321--3330},
      year = 	 {2018},
      volume = 	 {80},
    }


    @article{massias2020dual,
      author  = {Mathurin Massias and Samuel Vaiter and Alexandre Gramfort and Joseph Salmon},
      title   = {Dual Extrapolation for Sparse GLMs},
      journal = {Journal of Machine Learning Research},
      year    = {2020},
      volume  = {21},
      number  = {234},
      pages   = {1-33},
      url     = {http://jmlr.org/papers/v21/19-587.html}
    }


ArXiv links:

- https://arxiv.org/abs/1802.07481
- https://arxiv.org/abs/1907.05830

.. |image0| image:: https://github.com/mathurinm/celer/workflows/build/badge.svg
   :target: https://github.com/mathurinm/celer/actions?query=workflow%3Abuild
.. |image1| image:: https://codecov.io/gh/mathurinm/celer/branch/master/graphs/badge.svg?branch=master
   :target: https://codecov.io/gh/mathurinm/celer
