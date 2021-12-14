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
The package can be updated from github using the following command: 
```bash
pip install update git+git://github.com/DDPSE/PyDDSBB/
```
## Usage
To import the package after installation, 
```Python
import PyDDSBB
```
The package contains two main objects. PyDDSBB.DDSBBModel establishes the optimization problems including the objective functions and constraints. PyDDSBB.DDSBB is the solver that solves the problem wrapped in PyDDSBB.DDSBBModel object. 
For the example problem is shown below. The objective function should return float type. The constraint returns 1. for feasible point and 0. for infeasible point. Both functions take array of input variables. The array is ordered. 
```Python
def objective(x):  ### Objective Function 
    return x[0]    
def constraints(x):
    if (x[0]-2.0)**2 + (x[1]-4.0)**2 <= 4.0 and (x[0]-3.0)**2 + (x[1]-3.0)**2 <= 4.0:   ### Constraints 
        return 1.  ## 1 as feasible 
    else:
        return 0.  ## 0 as infeasible
```
To create a problem, one can initialize the an abstract model by:
```Python
model = PyDDSBB.DDSBBModel.Problem() ## Initializa a model
```
The object function can be added via .add_objective() method:
```Python
model.add_objective(objective, sense = 'minimize') ## Add objective function 
```
Known constraints and unknown constraints can be added using .add_unknown_constraint() and .add_unknown_constraint():
```Python
model.add_unknown_constraint(constraints) ## Add unknown constraints
model.add_known_constraint('(x0-2.0)**2 + (x1-4.0)**2 <= 4.0') ## Add known constraint
```
The methods above can be called without specific order. However, to add variables, the order must align exactly with the order of inputs to the black-box simulation. 
```Python
model.add_variable(1., 5.5) ## Add the first variable
model.add_variable(1., 5.5) ## Add the second variable
```
To initialize the solver, only one required input is needed, which is number of initial samples. To initialize the solver with all default settings:
```Python
solver = PyDDSBB.DDSBB(23)
```

Here are the options for the DDSBB solver: 
```Python
        multifidelity: bool  
                       True to turn on multifidelity approach 
                       False to turn off multifidelity approach (default)
        split_method: str
                      Methods to determine split point on one dimension
                      select from:
                      For all types of problems:
                            'equal_bisection' (default), 'golden_section' 
                      For constrained problems:
                            'purity', 'gini'
        variable_selection: str
                      Methods to determine which dimension to be splitted on.
                      select from: longest_side (default, for all problems), svr_var_select (for all problem)
                                   purity, gini (constrained problems)
        underestimator_option: str
                      Underestimator type 
                      Default: Quadratic
        stop_option: dict
                    Stopping criteria 
                    absolute_tolerance: float  (tolerance for gap between the lower and the upper bound)
                    relative_tolerance: float  (tolerace for relative gap between the lower and the upper bound: absolute_gap/|lower bound| if it is a minimization problem)
                    minimum_bound: float (minimum bound distance on the input space to avoid cutting the search space too small)
                    sampling_limit: int (maximum number of samples)
                    time_limit: float (maximum run time (s))
        sense: str
               select from: minimize, maximize (inform the solver the direction of optimization)
        adaptive_sampling: function of level, dimension (method for adaptive sampling, can be a function of level, dimenion) 
```
```Python
solver = PyDDSBB.DDSBB(23,split_method = 'equal_bisection', variable_selection = 'longest_side', multifidelity = False, stop_option = {'absolute_tolerance': 0.05, 'relative_tolerance': 0.01, 'minimum_bound': 0.05, 'sampling_limit': 500, 'time_limit': 5000}) 
```
To solve the problem, .solve() method is called. 
```Python
solver.optimize(model)   
```
To print the results and get optimum and the optimizers:
```Python
yopt = solver.get_optimum()  ### Get optimal solution 
xopt = solver.get_optimizer() ### Get optimizer 
```

The search can be resumed after reaching the stopping limits. User can change the stopping criteria by calling .resume(). For example, increase the sampling limit to 1000, and lower the absolute tolerance to 0.01:
```Python
solver.resume({'sampling_limit': 1000, 'absolute_tolerance': 0.001})
```
The following code is available in test.py. 
```Python
import PyDDSBB
### Test black-box problem with constraints ###
def objective(x):  ### Objective Function 
    return x[0]    
def constraints(x):
    if (x[0]-2.0)**2 + (x[1]-4.0)**2 <= 4.0 and (x[0]-3.0)**2 + (x[1]-3.0)**2 <= 4.0:   ### Constraints 
        return 1.  ## 1 as feasible 
    else:
        return 0.  ## 0 as infeasible
    
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
yopt = solver.get_optimum()  ### Get optimal solution 
xopt = solver.get_optimizer() ### Get optimizer
### Resume search 
solver.resume({'sampling_limit': 1000, 'absolute_tolerance': 0.001})
solver.print_result()
```
## License
MIT License

Copyright (c) 2021 GT-DDPSE

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
