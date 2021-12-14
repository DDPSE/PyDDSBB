#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyDDSBB test example with unknown constraints and known constraints

@author: JianyuanZhai
"""

import PyDDSBB

def objective(x):
    return x[0]
def constraints(x):
    if (x[0]-2.0)**2 + (x[1]-4.0)**2 <= 4.0 and (x[0]-3.0)**2 + (x[1]-3.0)**2 <= 4.0:
        return 1.
    else:
        return 0.
    
### Define the model   
Model = PyDDSBB.DDSBBModel.Problem() ## Initializa a model
Model.add_objective(objective, sense = 'minimize') ## Add objective function 
Model.add_unknown_constraint(constraints) ## Add unknown constraints
Model.add_known_constraint('(x0-2.0)**2 + (x1-4.0)**2 <= 4.0') ## Add known constraint
Model.add_variable(1., 5.5) ## Add the first variable
Model.add_variable(1., 5.5) ## Add the second variable

### Initialize the solver ##

solver = PyDDSBB.DDSBB(23,split_method = 'equal_bisection', variable_selection = 'longest_side', multifidelity = False, stop_option = {'absolute_tolerance': 0.05, 'relative_tolerance': 0.01, 'minimum_bound': 0.05, 'sampling_limit': 500, 'time_limit': 5000}) 

### Solve the model 
solver.optimize(Model)     
solver.print_result()
###
yopt = solver.get_optimum()  ### Get optimal solution 
xopt = solver.get_optimizer() ### Get optimizer 


