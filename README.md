# PyDDSBB
PyDDSBB is the Python implementation of the data-driven spatial branch-and-bound algorithm for simulation-based optimization. 

## About
PyDDSBB is an open-source Python library for simulation-based optimization with continuous variables. The algorithm is capable of handling equation-based constraints and simulation-based constraints. 

The package is built on NumPy, Pyomo, scikit-learn, ipopt, and glpk. Please install ipopt and glpk before installing the PyDDSBB package.  

The PyDDSBB package consists of two main parts: (a) DDSBBModel and (b) DDSBB. DDSBBModel is an object that allows users to define the simulation-based optimization problem. DDSBB object is the solver that solves the DDSBBModel defined by the user.

If you have any questions or concerns, please send an email to zhaijianyuan@gmail.com or fani.boukouvala@chbe.gatech.edu

## Installation

If using Anaconda, first run: 
conda install git pip

The code can be directly installed from github using the following command: 
```bash
pip install git+git://github.com/DDPSE/PyDDSBB/
```
## Usage
The following code is available in test.py
```Python
import PyDDSBB

def objective(x):
    return x[0]
def constraints(x):
    if (x[0]-2.0)**2 + (x[1]-4.0)**2 <= 4.0 and (x[0]-3.0)**2 + (x[1]-3.0)**2 <= 4.0:
        return 1.
    else:
        return 0.
    
### Define the model   
model = PyDDSBB.DDSBBModel.Problem() ## Initializa a model
model.add_objective(objective, sense = 'minimize') ## Add objective function 
model.add_unknown_constraint(constraints) ## Add unknown constraints
model.add_known_constraint('(x0-2.0)**2 + (x1-4.0)**2 <= 4.0') ## Add known constraint
model.add_variable(1., 5.5) ## Add the first variable
model.add_variable(1., 5.5) ## Add the second variable

### Initialize the solver ##

solver = PyDDSBB.DDSBB(23,split_method = 'equal_bisection', variable_selection = 'longest_side', multifidelity = False, stop_option = {'absolute_tolerance': 0.05, 'relative_tolerance': 0.01, 'minimum_bound': 0.05, 'sampling_limit': 500, 'time_limit': 5000}) 

### Solve the model 
solver.optimize(model)     
solver.print_result()
### Extract solution from the solver 
yopt = solver.yopt_global  ### Get optimal solution 
xopt = solver.xopt_global ### Get optimizer 
lowerbound = solver.lowerbound_global ### Get lower bound 
```
## License
[MIT](https://choosealicense.com/licenses/mit/)
