.. adjustText documentation master file, created by
   sphinx-quickstart on Sun Apr 29 08:25:05 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the documentation for adjustText!
======================================
adjustText is a small library to help you adjust text positions on matplotlib plots to remove or minimize overlaps with each other and data points. The approach is based on overlaps of bounding boxes and iteratively moving them to reduce overlaps. The idea is from the ggrepel package for R/ggplot2 (https://github.com/slowkow/ggrepel).

The repository with the issue tracker can be found here: https://github.com/Phlya/adjustText/

.. toctree::
   :maxdepth: 1

   Examples.ipynb
   Examples-for-multiple-subplots.ipynb


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Module documentation
====================

.. automodule:: adjustText
   :members: adjust_text
