# PyDDSBB
PyDDSBB is the Python implementation of the data-driven spatial branch-and-bound algorithm for simulation-based optimization. 

## About
PyDDSBB is an open-source Python library for simulation-based optimization with continuous variables. The algorithm is capable of handling equation-based constraints and simulation-based constraints. 

The package is built on NumPy, Pyomo, scikit-learn, ipopt, and glpk. 

The PyDDSBB package consists of two main parts: (a) DDSBBModel and (b) DDSBB. DDSBBModel is an object that allows users to define the simulation-based optimization problem. DDSBB object is the solver that solves the DDSBBModel defined by the user.

If you have any questions or concerns, please send an email to zhaijianyuan@gmail.com or fani.boukouvala@chbe.gatech.edu

## Installation

If using Anaconda, first run: 
conda install git pip

The code can be directly installed from github using the following command: 
pip install git+git://github.com/DDPSE/PyDDSBB/
